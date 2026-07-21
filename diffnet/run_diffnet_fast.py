"""
Crash-proof SINGLE-PROCESS differential-network permutation runner (fast matvec filter).

Why single-process: forking N workers each copied the ~4 GB (B,V,V2) superset matrices
-> ~18 GB/worker resident -> OOM-crashed the spec'd sandboxes. One process holds ONE
copy (~4 GB) and the vectorized matvec filter makes each permutation ~0.5 s, so 1000
perms x 2 conditions completes in ~15-20 min with no memory blowup.

Faithfulness: filter (matvec form verified identical to reduction form), CLR/log over
each half's filtered feature set, tau/delta, classify_edge, artifact decomposition are
all identical to pipeline_core / the naive serial null (verified edge-for-edge).

Checkpointing: writes null rows incrementally to <cond>__null.tsv every CHUNK perms and
records progress in <cond>__progress.json so a rerun resumes.

Usage:
  python run_diffnet_fast.py <cohort> --variant clr|log|clr_resid --nperm 1000 \
        [--ibd-mode samples|subject] [--chunk 100]
Outputs -> /mnt/shared-workspace/micro/diffnet/<cohort>/<variant>[_subject]/
"""
import os, sys, json, argparse, time, math
import numpy as np
import pandas as pd
from scipy import sparse
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # import pipeline_core from this folder
from pipeline_core import (load_cohort, load_meta, clr_transform, log_transform,
                           residualize, spearman_matrix, edges_from_corr,
                           differential_network, rewiring_scores,
                           TAU, DELTA, MIN_PREVALENCE, MAX_PREVALENCE,
                           MIN_MEAN, MIN_VAR)

DIFFDIR = _os.environ.get("DIFFNET_DIFFDIR", "diffnet_out")
CONDS = {
    "acvd": [("control", "control_female", "control_male"),
             ("ACVD", "ACVD_female", "ACVD_male")],
    "ibd":  [("control", "control_female", "control_male"),
             ("IBD", "IBD_female", "IBD_male")],
    "t2d":  [("control", "control_female", "control_male"),
             ("T2D", "T2D_female", "T2D_male")],
}


def dense_filter_keep(Xd):
    prev = (Xd > 0).sum(axis=1) / Xd.shape[1]
    mean = Xd.mean(axis=1)
    var = (Xd**2).mean(axis=1) - mean**2
    var[var < 0] = 0
    m = ((prev >= MIN_PREVALENCE) & (prev <= MAX_PREVALENCE) &
         (mean >= MIN_MEAN) & (var >= MIN_VAR))
    return np.where(m)[0]


def matvec_filter_keep(B, V, V2, local_cols, ncols):
    h = len(local_cols)
    ind = np.zeros((ncols, 1))
    ind[local_cols] = 1.0
    prev = np.asarray(B @ ind).ravel() / h
    mean = np.asarray(V @ ind).ravel() / h
    mean2 = np.asarray(V2 @ ind).ravel() / h
    var = mean2 - mean**2
    var[var < 0] = 0
    m = ((prev >= MIN_PREVALENCE) & (prev <= MAX_PREVALENCE) &
         (mean >= MIN_MEAN) & (var >= MIN_VAR))
    return np.where(m)[0]


def half_net(Xss, B, V, V2, ss_names, local_cols, ncols, transform, tau):
    keep = matvec_filter_keep(B, V, V2, local_cols, ncols)
    if len(keep) < 2:
        return {}, set()
    sub = np.asarray(Xss[keep][:, local_cols].todense()).astype(float)
    M = clr_transform(sub) if transform.startswith("clr") else log_transform(sub)
    names = [ss_names[i] for i in keep]
    return edges_from_corr(spearman_matrix(M), names, tau=tau), set(names)


def gstat(dn):
    if len(dn) == 0:
        return dict(n_rewired=0, n_true=0, n_exclusive=0, n_sign=0,
                    frac_true=float("nan"), max_node_score=0.0)
    n = len(dn)
    n_true = int((dn.artifact == "true_change").sum())
    n_excl = int(dn.Type.isin(["exclusive_G1", "exclusive_G2"]).sum())
    n_sign = int((dn.Type == "sign_changed").sum())
    rs = rewiring_scores(dn)
    return dict(n_rewired=n, n_true=n_true, n_exclusive=n_excl, n_sign=n_sign,
                frac_true=n_true / n if n else float("nan"),
                max_node_score=float(rs.Score.max()) if len(rs) else 0.0)


def run_condition(cond, gf, gm, X, rows, cols, meta, transform, covar_df,
                  tau, nperm, chunk, outdir):
    smap = {s: i for i, s in enumerate(cols)}
    sf = [s for s in meta.index[meta["group"] == gf] if s in smap]
    sm = [s for s in meta.index[meta["group"] == gm] if s in smap]
    idx_f = [smap[s] for s in sf]; idx_m = [smap[s] for s in sm]
    n_m, n_f = len(idx_m), len(idx_f)
    if n_m < 3 or n_f < 3:
        print(f"[{cond}] SKIP n_m={n_m} n_f={n_f}", flush=True)
        return dict(observed=None, perm=None, n_m=n_m, n_f=n_f, skipped=True)
    print(f"[{cond}] n_male={n_m} n_female={n_f}", flush=True)

    # ---------- OBSERVED ----------
    Xg_m = X[:, idx_m].toarray().astype(float)
    Xg_f = X[:, idx_f].toarray().astype(float)
    mk = dense_filter_keep(Xg_m); fk = dense_filter_keep(Xg_f)
    feat_m = [rows[i] for i in mk]; feat_f = [rows[i] for i in fk]

    def _obs_net(Xg, keep, samples):
        sub = Xg[keep, :]
        M = clr_transform(sub) if transform.startswith("clr") else log_transform(sub)
        if transform.endswith("resid") and covar_df is not None:
            M = residualize(M, covar_df.loc[samples])
        return edges_from_corr(spearman_matrix(M), [rows[i] for i in keep], tau=tau)

    ed_m = _obs_net(Xg_m, mk, sm); ed_f = _obs_net(Xg_f, fk, sf)
    dn_obs = differential_network(ed_m, ed_f, feat_m, feat_f, tau=tau)
    dn_obs.to_csv(os.path.join(outdir, f"{cond}__diffnet.tsv"), sep="\t", index=False)
    rewiring_scores(dn_obs).to_csv(os.path.join(outdir, f"{cond}__rewiring_scores.tsv"),
                                   sep="\t", index=False)
    if len(dn_obs):
        (dn_obs.groupby(["Type", "artifact"]).size().reset_index(name="count")
         .to_csv(os.path.join(outdir, f"{cond}__type_artifact_breakdown.tsv"),
                 sep="\t", index=False))
    obs = gstat(dn_obs)
    print(f"[{cond}] OBSERVED {obs}", flush=True)
    del Xg_m, Xg_f

    if transform.endswith("resid"):
        return dict(observed=obs, perm=None, n_m=n_m, n_f=n_f,
                    note="resid: observed only (label-shuffle covariate design undefined)")

    # ---------- NULL setup: superset + matvec matrices (single copy) ----------
    pooled = idx_m + idx_f
    thr = max(1, math.ceil(MIN_PREVALENCE * min(n_m, n_f)))
    nz = np.asarray((X[:, pooled] > 0).sum(axis=1)).ravel()
    ss_rows = np.where(nz >= thr)[0]
    Xss = X[ss_rows][:, pooled].tocsr()
    ss_names = [rows[i] for i in ss_rows]
    B = (Xss > 0).astype(np.float64); Vv = Xss.astype(np.float64); V2 = Vv.multiply(Vv)
    ncols = len(pooled)
    print(f"[{cond}] superset={len(ss_names)} (thr>={thr}) pooled={ncols}", flush=True)

    # resume support
    null_path = os.path.join(outdir, f"{cond}__null.tsv")
    keys = ["n_rewired", "n_true", "n_exclusive", "n_sign", "max_node_score"]
    done = 0
    rows_acc = []
    prog_path = os.path.join(outdir, f"{cond}__progress.json")
    if os.path.exists(null_path) and os.path.exists(prog_path):
        try:
            prev = pd.read_csv(null_path, sep="\t")
            if list(prev.columns) == keys and len(prev) <= nperm:
                rows_acc = prev.values.tolist(); done = len(prev)
                print(f"[{cond}] resuming from {done} perms", flush=True)
        except Exception:
            done = 0; rows_acc = []

    t0 = time.time()
    for b in range(done, nperm):
        rng = np.random.default_rng(b)
        perm = rng.permutation(ncols)
        cm, cf = perm[:n_m], perm[n_m:]
        e_m, s_m = half_net(Xss, B, Vv, V2, ss_names, cm, ncols, transform, tau)
        e_f, s_f = half_net(Xss, B, Vv, V2, ss_names, cf, ncols, transform, tau)
        g = gstat(differential_network(e_m, e_f, list(s_m), list(s_f), tau=tau))
        rows_acc.append([g[k] for k in keys])
        if (b + 1) % chunk == 0 or (b + 1) == nperm:
            pd.DataFrame(rows_acc, columns=keys).to_csv(null_path, sep="\t", index=False)
            with open(prog_path, "w") as fh:
                json.dump(dict(done=b + 1, nperm=nperm), fh)
            rate = (time.time() - t0) / (b + 1 - done)
            print(f"[{cond}] {b+1}/{nperm} ({rate:.2f}s/perm, "
                  f"eta {rate*(nperm-b-1)/60:.1f} min)", flush=True)

    null = pd.DataFrame(rows_acc, columns=keys)
    pvals = {}
    for k in keys:
        arr = null[k].to_numpy(dtype=float)
        pvals[k] = dict(observed=obs[k], null_mean=float(np.nanmean(arr)),
                        null_sd=float(np.nanstd(arr)),
                        p_ge=float((np.sum(arr >= obs[k]) + 1) / (len(arr) + 1)),
                        p_le=float((np.sum(arr <= obs[k]) + 1) / (len(arr) + 1)))
    print(f"[{cond}] p(n_rewired)={pvals['n_rewired']['p_ge']:.4f} "
          f"p(n_true)={pvals['n_true']['p_ge']:.4f}", flush=True)
    return dict(observed=obs, perm=pvals, n_m=n_m, n_f=n_f, n_perm=nperm,
                superset=len(ss_names))


def subset_first_visit(meta):
    m = meta.reset_index()
    m["_d"] = pd.to_numeric(m.get("days_from_first_collection", 0), errors="coerce").fillna(0)
    keep = m.loc[m.groupby("subject_id")["_d"].idxmin(), "sample_id"]
    return meta.loc[meta.index.isin(set(keep))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cohort")
    ap.add_argument("--variant", default="clr", choices=["log", "clr", "clr_resid"])
    ap.add_argument("--nperm", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--ibd-mode", default="samples", choices=["samples", "subject"])
    ap.add_argument("--only", default=None, help="run only this condition (control/DISEASE)")
    args = ap.parse_args()
    transform = {"log": "log", "clr": "clr", "clr_resid": "clr_resid"}[args.variant]

    X, rows, cols = load_cohort(args.cohort, "gene_families")
    meta = load_meta(args.cohort)
    suffix = args.variant
    if args.cohort == "ibd" and args.ibd_mode == "subject":
        meta = subset_first_visit(meta)
        suffix = args.variant + "_subject"
        print(f"[ibd] subject-level: {meta.shape[0]} samples", flush=True)

    covars = ["age", "age_category", "BMI", "number_reads", "median_read_length",
              "antibiotics_current_use", "country", "non_westernized"]
    cp = [c for c in covars if c in meta.columns and meta[c].notna().mean() > 0.5
          and meta[c].nunique() > 1]
    covar_df = meta[cp] if transform.endswith("resid") else None

    outdir = os.path.join(DIFFDIR, args.cohort, suffix)
    os.makedirs(outdir, exist_ok=True)
    sfile = os.path.join(outdir, "summary.json")
    summary = json.load(open(sfile)) if os.path.exists(sfile) else {}
    for cond, gf, gm in CONDS[args.cohort]:
        if args.only and cond != args.only:
            continue
        summary[cond] = run_condition(cond, gf, gm, X, rows, cols, meta, transform,
                                      covar_df, args.tau, args.nperm, args.chunk, outdir)
        with open(sfile, "w") as fh:
            json.dump(summary, fh, indent=2, default=float)
    print("DIFFNET_FAST_DONE", args.cohort, suffix, flush=True)


if __name__ == "__main__":
    main()
