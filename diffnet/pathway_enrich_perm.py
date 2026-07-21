#!/usr/bin/env python
"""
pathway_enrich_perm.py -- permutation test for the pathway-level differential co-abundance signal
(Reviewer 3.5: are the flagged pathways real, or expected under the null / by degree?).

We rebuild the pathway differential network under sex-label shuffles (within condition, preserving
group sizes) and compare the OBSERVED quantities to the permutation null:
  - n_true          : number of true_change pathway edges
  - n_pathways      : number of distinct pathways participating in true_change edges
  - top_score       : maximum per-pathway rewiring score (sum of |Delta| over its true_change edges)
This is the pathway analogue of the edge-level permutation test in R3.3 and controls for degree/
network size because the same filtering + thresholding is applied in every permutation.

Usage: python pathway_enrich_perm.py <cohort> [--nperm 500]
Output -> robust/<cohort>/pathway_enrich_perm_<condition>.csv
"""
import os, sys, json
import numpy as np
import pandas as pd
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # import pipeline_core from this folder
import pipeline_core as pc

cohort = sys.argv[1] if len(sys.argv) > 1 else "acvd"
nperm = int(sys.argv[sys.argv.index("--nperm") + 1]) if "--nperm" in sys.argv else 500
OUT = _os.path.join(_os.environ.get("DIFFNET_OUT", "robust"), cohort)
TAU, DELTA = 0.30, 0.30

CONDS = {
    "acvd": [("control", "control_female", "control_male"), ("ACVD", "ACVD_female", "ACVD_male")],
    "ibd":  [("control", "control_female", "control_male"), ("IBD", "IBD_female", "IBD_male")],
    "t2d":  [("control", "control_female", "control_male"), ("T2D", "T2D_female", "T2D_male")],
}[cohort]

X, rows, cols = pc.load_cohort(cohort, kind="pathway_abundance")
rows = np.array(rows)
named = np.array([("|" not in r) and (not r.startswith("UNMAPPED")) and (not r.startswith("UNINTEGRATED"))
                  for r in rows])
Xn = X[np.where(named)[0], :]
paths = rows[named]
meta = pc.load_meta(cohort)

# Subject-level de-pseudoreplication for IBD (one first-visit sample per subject).
if "--ibd-subject" in sys.argv and cohort == "ibd":
    from run_diffnet_fast import subset_first_visit
    n0 = meta.shape[0]
    meta = subset_first_visit(meta)
    print(f"[ibd] SUBJECT-LEVEL: {meta.shape[0]} first-visit samples (of {n0})", flush=True)
    OUT = _os.path.join(_os.environ.get("DIFFNET_OUT", "robust"), "ibdsubj")
    os.makedirs(OUT, exist_ok=True)

s2c = {s: i for i, s in enumerate(cols)}

def pathway_filter(Xg):
    prev = np.asarray((Xg > 0).sum(axis=1)).ravel() / Xg.shape[1]
    return np.where((prev >= pc.MIN_PREVALENCE) & (prev <= pc.MAX_PREVALENCE))[0]

def net_from_cols(cidx):
    Xg = Xn[:, cidx]
    fi = pathway_filter(Xg)
    Xd = np.asarray(Xg[fi, :].todense())
    Xclr = pc.clr_transform(Xd)
    R = pc.spearman_matrix(Xclr)
    names = paths[fi]
    d = {}
    n = len(names)
    iu, ju = np.triu_indices(n, k=1)
    vals = R[iu, ju]
    m = np.abs(vals) >= TAU
    for a, b, v in zip(iu[m], ju[m], vals[m]):
        d[tuple(sorted((names[a], names[b])))] = float(v)
    return d, set(names)

def diffstats(cidx_m, cidx_f):
    Em, fm = net_from_cols(cidx_m)
    Ef, ff = net_from_cols(cidx_f)
    keys = set(Em) | set(Ef)
    tc = {}
    n_true = 0
    for k in keys:
        r1 = Em.get(k, 0.0); r2 = Ef.get(k, 0.0)
        d = abs(r1 - r2)
        if d < DELTA:
            continue
        a, b = k
        in1, in2 = k in Em, k in Ef
        node_missing = ((not in1) and (a not in fm or b not in fm)) or \
                       ((not in2) and (a not in ff or b not in ff))
        if not node_missing:
            n_true += 1
            for p in k:
                tc[p] = tc.get(p, 0.0) + d
    top = max(tc.values()) if tc else 0.0
    return n_true, len(tc), top

rng = np.random.default_rng(0)
out_rows = []
for cond, g_f, g_m in CONDS:
    sids_m = [s for s in meta.index[meta["group"] == g_m] if s in s2c]
    sids_f = [s for s in meta.index[meta["group"] == g_f] if s in s2c]
    if len(sids_m) < 5 or len(sids_f) < 5:
        print(f"[{cond}] too few samples, skip", flush=True); continue
    cm = [s2c[s] for s in sids_m]; cf = [s2c[s] for s in sids_f]
    obs_true, obs_np, obs_top = diffstats(cm, cf)
    pooled = cm + cf; n_m = len(cm)
    null_true, null_np, null_top = [], [], []
    for i in range(nperm):
        perm = rng.permutation(pooled)
        pm, pf = perm[:n_m].tolist(), perm[n_m:].tolist()
        t, npath, top = diffstats(pm, pf)
        null_true.append(t); null_np.append(npath); null_top.append(top)
        if (i + 1) % 100 == 0:
            print(f"  [{cond}] perm {i+1}/{nperm}", flush=True)
    def p_ge(obs, null):
        null = np.array(null, float)
        return float(((null >= obs).sum() + 1) / (len(null) + 1))
    row = dict(cohort=cohort, condition=cond, nperm=nperm,
               obs_n_true=obs_true, null_n_true_mean=float(np.mean(null_true)),
               null_n_true_sd=float(np.std(null_true)), p_n_true=p_ge(obs_true, null_true),
               obs_n_pathways=obs_np, null_n_pathways_mean=float(np.mean(null_np)),
               null_n_pathways_sd=float(np.std(null_np)), p_n_pathways=p_ge(obs_np, null_np),
               obs_top_score=obs_top, null_top_mean=float(np.mean(null_top)),
               null_top_sd=float(np.std(null_top)), p_top_score=p_ge(obs_top, null_top))
    out_rows.append(row)
    print(f"[{cond}] n_true obs={obs_true} null={np.mean(null_true):.0f}+-{np.std(null_true):.0f} "
          f"p={row['p_n_true']:.3f} | n_pathways obs={obs_np} null={np.mean(null_np):.0f} "
          f"p={row['p_n_pathways']:.3f} | top obs={obs_top:.1f} null={np.mean(null_top):.1f} "
          f"p={row['p_top_score']:.3f}", flush=True)

pd.DataFrame(out_rows).to_csv(f"{OUT}/pathway_enrich_perm.csv", index=False)
print(f"[{cohort}] pathway enrichment permutation DONE -> {OUT}/pathway_enrich_perm.csv", flush=True)
