"""
Core pipeline for microbiome differential co-abundance network analysis.

Contains:
  (A) Faithful reproduction of the published pipeline (build_networks.py logic):
      group-specific prevalence/mean/var filtering -> log(x+eps) -> Spearman -> |rho|>=tau.
  (B) Robust extensions used for the revision:
      - CLR transform (compositionality-aware)
      - covariate residualization
      - shared-feature (intersection) variant to separate rewiring from filtering
      - differential-network construction with artifact decomposition
      - node-level rewiring scores

This module is imported by the analysis driver scripts. Data are read from
/mnt/shared-workspace/micro/data as sparse .mtx + row/col + metadata.tsv.
"""
import os
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.stats import spearmanr, rankdata

DATA = os.environ.get("DIFFNET_DATA", "data")   # override with DIFFNET_DATA env var

# --- published defaults (from build_networks.py) ---
MIN_PREVALENCE = 0.1
MAX_PREVALENCE = 0.9
MIN_MEAN = 1e-6
MIN_VAR = 1e-8
PSEUDOCOUNT = 1e-6
TAU = 0.30
DELTA = 0.30


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def _read_lines(path):
    """Read a row/col id file as raw lines. IDs (e.g. MetaCyc pathway names) can contain
    commas, so we must NOT parse as CSV."""
    with open(path) as fh:
        return [ln.rstrip("\n") for ln in fh if ln.strip() != ""]


def load_cohort(cohort, kind="gene_families"):
    """Return (X csr [features x samples], feature_names, sample_ids)."""
    X = mmread(os.path.join(DATA, f"{cohort}_{kind}.mtx")).tocsr()
    rows = _read_lines(os.path.join(DATA, f"{cohort}_{kind}_rows.txt"))
    cols = _read_lines(os.path.join(DATA, f"{cohort}_{kind}_cols.txt"))
    return X, rows, cols


def load_meta(cohort):
    m = pd.read_csv(os.path.join(DATA, f"{cohort}_metadata.tsv"), sep="\t", dtype=str)
    m = m.set_index("sample_id")
    for c in ("age", "BMI", "number_reads", "number_bases", "median_read_length"):
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")
    m["group"] = m["study_condition"].astype(str) + "_" + m["gender"].astype(str)
    return m


# ----------------------------------------------------------------------
# Feature filtering (published logic)
# ----------------------------------------------------------------------
def filter_features(X_group):
    prevalence = np.asarray((X_group > 0).sum(axis=1)).ravel() / X_group.shape[1]
    mean = np.asarray(X_group.mean(axis=1)).ravel()
    mean2 = np.asarray(X_group.power(2).mean(axis=1)).ravel()
    var = mean2 - mean**2
    var[var < 0] = 0
    mask = ((prevalence >= MIN_PREVALENCE) & (prevalence <= MAX_PREVALENCE) &
            (mean >= MIN_MEAN) & (var >= MIN_VAR))
    return np.where(mask)[0]


# ----------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------
def log_transform(X_dense):
    return np.log(X_dense + PSEUDOCOUNT)


def clr_transform(X_dense, pseudocount=None):
    """Centered log-ratio across features (rows=features, cols=samples).
    CLR is computed per sample (column) over the feature composition."""
    if pseudocount is None:
        # half of smallest non-zero, per common practice
        nz = X_dense[X_dense > 0]
        pseudocount = (nz.min() / 2.0) if nz.size else 1e-6
    Xp = X_dense + pseudocount
    logX = np.log(Xp)
    gm = logX.mean(axis=0, keepdims=True)   # geometric mean per sample (column)
    return logX - gm


def residualize(mat_fs, covar_df):
    """Regress each feature (row) on covariates; return residuals.
    mat_fs: [features x samples] (already transformed).
    covar_df: DataFrame indexed by sample_id, columns = covariates (samples in same order).
    Numeric covariates z-scored; categoricals one-hot. Rows with all-NA covariate dropped -> mean-imputed.
    """
    S = mat_fs.shape[1]
    # Build design matrix
    dm_parts = [np.ones((S, 1))]
    for c in covar_df.columns:
        col = covar_df[c]
        if pd.api.types.is_numeric_dtype(col):
            v = col.to_numpy(dtype=float)
            if np.all(np.isnan(v)):
                continue
            mu = np.nanmean(v); sd = np.nanstd(v)
            v = np.where(np.isnan(v), mu, v)
            if sd > 0:
                v = (v - mu) / sd
            dm_parts.append(v.reshape(-1, 1))
        else:
            d = pd.get_dummies(col.astype(str).fillna("NA"), drop_first=True)
            if d.shape[1] > 0:
                dm_parts.append(d.to_numpy(dtype=float))
    Dm = np.hstack(dm_parts)
    # Least squares hat: resid = Y - D (D^+ Y)
    Y = mat_fs.T  # samples x features
    beta, *_ = np.linalg.lstsq(Dm, Y, rcond=None)
    resid = Y - Dm @ beta
    return resid.T  # features x samples


# ----------------------------------------------------------------------
# Correlation / network construction
# ----------------------------------------------------------------------
def spearman_matrix(mat_fs):
    """Spearman correlation among features (rows). Returns dense corr [F x F]."""
    n = mat_fs.shape[0]
    if n < 2:
        return np.zeros((n, n))
    corr, _ = spearmanr(mat_fs, axis=1)
    corr = np.atleast_2d(corr)
    if corr.shape[0] != n:  # spearmanr returns scalar for n==2 sometimes
        corr = np.corrcoef(rankdata(mat_fs, axis=1))
    corr = np.nan_to_num(corr)
    np.fill_diagonal(corr, 0.0)
    return corr


def edges_from_corr(corr, feature_names, tau=TAU):
    """Return dict {(a,b) sorted: rho} for |rho|>=tau."""
    n = corr.shape[0]
    d = {}
    iu = np.triu_indices(n, k=1)
    vals = corr[iu]
    keep = np.abs(vals) >= tau
    for i, j, v in zip(iu[0][keep], iu[1][keep], vals[keep]):
        a, b = feature_names[i], feature_names[j]
        key = (a, b) if a <= b else (b, a)
        d[key] = float(v)
    return d


def build_group_network(X, row_names, sample_ids, group_samples,
                        transform="log", covar_df=None, tau=TAU,
                        feature_subset=None):
    """Build one group's co-abundance edge dict.
    transform: 'log' (published), 'clr', or 'clr_resid'/'log_resid' with covar_df.
    feature_subset: optional list of feature names to force (shared-feature variant);
                    if None, uses group-specific filtering.
    Returns (edge_dict, feature_names_used, corr_matrix).
    """
    sample_map = {s: i for i, s in enumerate(sample_ids)}
    idx = [sample_map[s] for s in group_samples if s in sample_map]
    Xg = X[:, idx]

    if feature_subset is None:
        feat_idx = filter_features(Xg)
        feat_names = [row_names[i] for i in feat_idx]
    else:
        name_to_row = {n: i for i, n in enumerate(row_names)}
        feat_idx = np.array([name_to_row[n] for n in feature_subset if n in name_to_row])
        feat_names = [row_names[i] for i in feat_idx]

    Xg = Xg[feat_idx, :].toarray().astype(float)

    if transform.startswith("clr"):
        M = clr_transform(Xg)
    else:
        M = log_transform(Xg)

    if transform.endswith("resid") and covar_df is not None:
        cov = covar_df.loc[[s for s in group_samples if s in sample_map]]
        M = residualize(M, cov)

    corr = spearman_matrix(M)
    ed = edges_from_corr(corr, feat_names, tau=tau)
    return ed, feat_names, corr


# ----------------------------------------------------------------------
# Differential network + artifact decomposition
# ----------------------------------------------------------------------
def classify_edge(r1, r2, tau=TAU, delta=DELTA):
    p1 = abs(r1) >= tau
    p2 = abs(r2) >= tau
    if p1 and p2 and (r1 * r2 < 0):
        return "sign_changed"
    if p1 and not p2:
        return "exclusive_G1"
    if p2 and not p1:
        return "exclusive_G2"
    if max(abs(r1), abs(r2)) >= tau and abs(r1 - r2) >= delta:
        return "differentially_weighted"
    return None


def differential_network(d1, d2, feats1, feats2, tau=TAU, delta=DELTA):
    """Build DN and decompose each rewired edge by artifact source.
    Returns DataFrame with columns:
      Node1,Node2,r1,r2,Delta,Abs_Delta,Type,artifact
    artifact in {true_change, node_missing_G2, node_missing_G1, both_missing}
      - node_missing_*: an endpoint absent from that group's *feature set* (filtering),
        so the edge could never exist there regardless of biology.
    """
    f1 = set(feats1); f2 = set(feats2)
    keys = set(d1) | set(d2)
    rows = []
    for e in keys:
        r1 = d1.get(e, 0.0)
        r2 = d2.get(e, 0.0)
        t = classify_edge(r1, r2, tau, delta)
        if t is None:
            continue
        a, b = e
        a_in1, b_in1 = a in f1, b in f1
        a_in2, b_in2 = a in f2, b in f2
        present_feat_1 = a_in1 and b_in1
        present_feat_2 = a_in2 and b_in2
        if present_feat_1 and present_feat_2:
            artifact = "true_change"          # both endpoints testable in both groups
        elif present_feat_1 and not present_feat_2:
            artifact = "node_missing_G2"       # edge absent in G2 only due to filtering
        elif present_feat_2 and not present_feat_1:
            artifact = "node_missing_G1"
        else:
            artifact = "both_missing"
        rows.append(dict(Node1=a, Node2=b, r1=r1, r2=r2,
                         Delta=r1 - r2, Abs_Delta=abs(r1 - r2),
                         Type=t, artifact=artifact))
    return pd.DataFrame(rows)


def rewiring_scores(dn_df):
    s = {}
    for _, r in dn_df.iterrows():
        s[r.Node1] = s.get(r.Node1, 0.0) + r.Abs_Delta
        s[r.Node2] = s.get(r.Node2, 0.0) + r.Abs_Delta
    return (pd.DataFrame([{"Node": k, "Score": v} for k, v in s.items()])
            .sort_values("Score", ascending=False).reset_index(drop=True))
