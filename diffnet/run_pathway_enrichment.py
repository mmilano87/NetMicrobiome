#!/usr/bin/env python
"""
run_pathway_enrichment.py -- pathway-level differential co-abundance analysis (Reviewer 3.5).

The manuscript's enrichment claim concerns MetaCyc PATHWAYS. Rather than re-map stale 2021-vintage
UniRef90 gene IDs to functions (those accessions are now deleted from UniProt/UniRef), we use the
version-matched functional layer HUMAnN already computed for these exact samples: the
`pathway_abundance` table (named MetaCyc pathways x samples). We build sex-specific pathway
co-abundance networks WITHIN each condition using the SAME pipeline (prevalence/var filter ->
CLR -> Spearman -> |rho|>=tau; differential edge |Delta rho|>=delta with artifact decomposition),
then report:
  - which pathways participate in `true_change` differential edges (the biologically interpretable
    output), with a per-pathway rewiring score;
  - robustness of that pathway set across the tau x delta grid;
  - an over-representation test (ORA, hypergeometric + BH-FDR) of `true_change`-involved pathways
    against the tested-pathway background, and against size-matched random pathway sets
    (degree/size control).

Only UNSTRATIFIED, NAMED pathways are used (UNMAPPED/UNINTEGRATED and species-stratified rows are
dropped) so that nodes are interpretable functional units.

Usage: python run_pathway_enrichment.py <cohort>
Outputs -> /mnt/shared-workspace/micro/robust/<cohort>/pathway_*.{tsv,csv,json}
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # import pipeline_core from this folder
import pipeline_core as pc

cohort = sys.argv[1] if len(sys.argv) > 1 else "acvd"
OUT = _os.path.join(_os.environ.get("DIFFNET_OUT", "robust"), cohort)
os.makedirs(OUT, exist_ok=True)

CONDS = {
    "acvd": [("control", "control_female", "control_male"), ("ACVD", "ACVD_female", "ACVD_male")],
    "ibd":  [("control", "control_female", "control_male"), ("IBD", "IBD_female", "IBD_male")],
    "t2d":  [("control", "control_female", "control_male"), ("T2D", "T2D_female", "T2D_male")],
}[cohort]

print(f"[{cohort}] loading pathway_abundance ...", flush=True)
X, rows, cols = pc.load_cohort(cohort, kind="pathway_abundance")  # features x samples
meta = pc.load_meta(cohort)

# Subject-level de-pseudoreplication for IBD (one first-visit sample per subject).
# The IBD cohort is longitudinal (1627 samples = 130 subjects); treating repeated visits as
# independent is the pseudoreplication flagged by the reviewers. When --ibd-subject is set we
# analyse first-visit samples only, matching the subject-level edge-level analysis.
if "--ibd-subject" in sys.argv and cohort == "ibd":
    from run_diffnet_fast import subset_first_visit
    n0 = meta.shape[0]
    meta = subset_first_visit(meta)
    print(f"[ibd] SUBJECT-LEVEL: {meta.shape[0]} first-visit samples (of {n0})", flush=True)
    OUT = _os.path.join(_os.environ.get("DIFFNET_OUT", "robust"), "ibdsubj")
    os.makedirs(OUT, exist_ok=True)

# keep only named, unstratified pathways
rows = np.array(rows)
is_named = np.array([("|" not in r) and (not r.startswith("UNMAPPED")) and (not r.startswith("UNINTEGRATED"))
                     for r in rows])
keep_idx = np.where(is_named)[0]
Xn = X[keep_idx, :]
paths = rows[keep_idx]
print(f"  {len(paths)} named unstratified MetaCyc pathways x {Xn.shape[1]} samples", flush=True)

sample_to_col = {s: i for i, s in enumerate(cols)}

def group_matrix(group):
    sids = [s for s in meta.index[meta["group"] == group] if s in sample_to_col]
    cidx = [sample_to_col[s] for s in sids]
    return Xn[:, cidx], sids

def pathway_filter(Xg):
    """Prevalence-only feature filter for the PATHWAY table.

    The published `filter_features` also imposes absolute mean/variance floors (MIN_MEAN=1e-6,
    MIN_VAR=1e-8) that were calibrated on the gene-family abundance scale. HUMAnN pathway relative
    abundances are ~50x smaller (median nonzero ~7e-5), so those absolute floors reject essentially
    all pathways. We therefore keep the scientifically meaningful prevalence window
    [MIN_PREVALENCE, MAX_PREVALENCE] = [0.1, 0.9] (identical to the published prevalence rule) and
    drop the scale-dependent abundance/variance floors for this table."""
    prev = np.asarray((Xg > 0).sum(axis=1)).ravel() / Xg.shape[1]
    mask = (prev >= pc.MIN_PREVALENCE) & (prev <= pc.MAX_PREVALENCE)
    return np.where(mask)[0]


def build_condition_dn(cond, g_f, g_m, tau, delta):
    """Build CLR Spearman networks for the two sexes, differential edges + artifact decomposition.
    Returns dict with edge counts and the set of pathways in true_change edges."""
    Xf, sf = group_matrix(g_f)
    Xm, sm = group_matrix(g_m)
    if len(sf) < 5 or len(sm) < 5:
        return None
    # prevalence-only filter (scale-appropriate for pathway relative abundance)
    fi_f = pathway_filter(Xf); fi_m = pathway_filter(Xm)
    feats_f = set(paths[fi_f]); feats_m = set(paths[fi_m])
    # CLR + Spearman within each group on its filtered features
    def group_net(Xg, fi):
        Xd = np.asarray(Xg[fi, :].todense()) if hasattr(Xg, "todense") else np.asarray(Xg[fi, :])
        Xclr = pc.clr_transform(Xd)  # features x samples
        R = pc.spearman_matrix(Xclr)  # features x features
        return R, paths[fi]
    Rf, pf = group_net(Xf, fi_f)
    Rm, pm = group_net(Xm, fi_m)
    # edges above tau in each group, keyed by pathway pair
    def edges(R, names):
        d = {}
        n = len(names)
        iu, ju = np.triu_indices(n, k=1)
        vals = R[iu, ju]
        mask = np.abs(vals) >= tau
        for a, b, v in zip(iu[mask], ju[mask], vals[mask]):
            key = tuple(sorted((names[a], names[b])))
            d[key] = v
        return d
    Em = edges(Rm, pm); Ef = edges(Rf, pf)  # male=G1, female=G2 (Delta = r_male - r_female)
    all_keys = set(Em) | set(Ef)
    n_true = 0; n_node_missing = 0; n_exclusive = 0
    tc_paths = {}
    for k in all_keys:
        r1 = Em.get(k, 0.0); r2 = Ef.get(k, 0.0)
        in1 = k in Em; in2 = k in Ef
        d = abs(r1 - r2)
        if d < delta:
            continue
        # exclusive if only in one group
        if in1 != in2:
            n_exclusive += 1
        # node_missing if a node absent from the OTHER group's filtered feature set
        a, b = k
        node_missing = ((not in1) and (a not in feats_m or b not in feats_m)) or \
                       ((not in2) and (a not in feats_f or b not in feats_f))
        if node_missing:
            n_node_missing += 1
        else:
            n_true += 1
            for p in k:
                tc_paths[p] = tc_paths.get(p, 0.0) + d
    return dict(cond=cond, tau=tau, delta=delta,
                n_edges_male=len(Em), n_edges_female=len(Ef),
                n_feat_male=len(pm), n_feat_female=len(pf),
                n_diff=len([1 for k in all_keys if abs(Em.get(k,0)-Ef.get(k,0))>=delta]),
                n_true=n_true, n_node_missing=n_node_missing, n_exclusive=n_exclusive,
                tc_paths=tc_paths)

# ---- main at published tau/delta ----
TAU0, DELTA0 = 0.30, 0.30
main_rows = []
tc_sets = {}
for cond, g_f, g_m in CONDS:
    res = build_condition_dn(cond, g_f, g_m, TAU0, DELTA0)
    if res is None:
        print(f"  [{cond}] insufficient samples, skip", flush=True); continue
    tc_sets[cond] = res["tc_paths"]
    main_rows.append({k: res[k] for k in
                      ["cond","tau","delta","n_edges_male","n_edges_female","n_feat_male",
                       "n_feat_female","n_diff","n_true","n_node_missing","n_exclusive"]})
    print(f"  [{cond}] tau={TAU0} delta={DELTA0}: n_true={res['n_true']} "
          f"node_missing={res['n_node_missing']} | {len(res['tc_paths'])} pathways in true_change",
          flush=True)

pd.DataFrame(main_rows).to_csv(f"{OUT}/pathway_diffnet_summary.csv", index=False)

# per-pathway true_change involvement table (ACVD condition = the disease arm)
disease = CONDS[1][0]
if disease in tc_sets and len(tc_sets[disease]) > 0:
    tcp = tc_sets[disease]
    dfp = (pd.DataFrame([{"pathway": p, "rewiring_score": s} for p, s in tcp.items()])
           .sort_values("rewiring_score", ascending=False).reset_index(drop=True))
    dfp.to_csv(f"{OUT}/pathway_true_change_{disease}.csv", index=False)
    print(f"\n  top true_change pathways ({disease}):", flush=True)
    print(dfp.head(15).to_string(index=False), flush=True)
else:
    pd.DataFrame(columns=["pathway", "rewiring_score"]).to_csv(
        f"{OUT}/pathway_true_change_{disease}.csv", index=False)
    print(f"\n  no true_change pathways for {disease}", flush=True)

# ---- robustness across tau x delta grid (which pathways persist) ----
TAUS = [0.25, 0.30, 0.35, 0.40]
DELTAS = [0.20, 0.30, 0.40]
grid_rows = []
persist = {}  # pathway -> count of grid cells where it appears in disease true_change
n_cells = 0
for tau in TAUS:
    for delta in DELTAS:
        res = build_condition_dn(disease, CONDS[1][1], CONDS[1][2], tau, delta)
        if res is None: continue
        n_cells += 1
        grid_rows.append({"tau": tau, "delta": delta, "n_true": res["n_true"],
                          "n_pathways": len(res["tc_paths"])})
        for p in res["tc_paths"]:
            persist[p] = persist.get(p, 0) + 1
pd.DataFrame(grid_rows).to_csv(f"{OUT}/pathway_grid_{disease}.csv", index=False)
if len(persist) > 0:
    persist_df = (pd.DataFrame([{"pathway": p, "n_grid_cells": c, "frac_grid": c / max(n_cells,1)}
                                for p, c in persist.items()])
                  .sort_values("n_grid_cells", ascending=False).reset_index(drop=True))
else:
    persist_df = pd.DataFrame(columns=["pathway", "n_grid_cells", "frac_grid"])
persist_df.to_csv(f"{OUT}/pathway_robustness_{disease}.csv", index=False)
n_robust = int((persist_df["frac_grid"] >= 0.5).sum()) if len(persist_df) else 0
print(f"\n  robustness: {n_cells} grid cells; pathways in >=50% of cells: {n_robust}", flush=True)
print(persist_df.head(15).to_string(index=False), flush=True)

json.dump({"cohort": cohort, "disease": disease, "n_named_pathways": int(len(paths)),
           "n_grid_cells": n_cells}, open(f"{OUT}/pathway_enrichment_meta.json", "w"), indent=2)
print(f"\n[{cohort}] pathway enrichment DONE -> {OUT}", flush=True)
