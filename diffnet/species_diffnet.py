#!/usr/bin/env python
"""
species_diffnet.py -- build OUR differential co-abundance network on SPECIES counts, so it can be
compared head-to-head with NetCoMi (Reviewer 3.1) and with the SPIEC-EASI species networks.

Same pipeline as the manuscript (CLR -> Spearman -> |rho|>=tau; differential edge |Delta|>=delta;
artifact decomposition), but on the MetaPhlAn species integer-count table (the feature space where
count-based tools are valid). Male=G1, Female=G2 (Delta = r_male - r_female), WITHIN each condition.

Usage: python species_diffnet.py <cohort>
Outputs -> /mnt/shared-workspace/micro/robust/<cohort>/species_diffnet_<condition>.tsv + summary.
"""
import os, sys, json
import numpy as np
import pandas as pd
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # import pipeline_core from this folder
import pipeline_core as pc

cohort = sys.argv[1] if len(sys.argv) > 1 else "acvd"
OUT = _os.path.join(_os.environ.get("DIFFNET_OUT", "robust"), cohort)
os.makedirs(OUT, exist_ok=True)
TAU, DELTA, PREV = 0.30, 0.30, 0.20

CONDS = {
    "acvd": [("control", "control_female", "control_male"), ("ACVD", "ACVD_female", "ACVD_male")],
    "ibd":  [("control", "control_female", "control_male"), ("IBD", "IBD_female", "IBD_male")],
    "ibdsubj": [("control", "control_female", "control_male"), ("IBD", "IBD_female", "IBD_male")],
    "t2d":  [("control", "control_female", "control_male"), ("T2D", "T2D_female", "T2D_male")],
}[cohort]

X, rows, cols = pc.load_cohort(cohort, kind="species_counts")
rows = np.array(rows)
meta = pc.load_meta(cohort)
s2c = {s: i for i, s in enumerate(cols)}

def group_net(group):
    sids = [s for s in meta.index[meta["group"] == group] if s in s2c]
    cidx = [s2c[s] for s in sids]
    Xg = X[:, cidx]
    prev = np.asarray((Xg > 0).sum(axis=1)).ravel() / max(len(cidx), 1)
    keep = np.where(prev >= PREV)[0]
    Xd = np.asarray(Xg[keep, :].todense())
    Xclr = pc.clr_transform(Xd)
    R = pc.spearman_matrix(Xclr)
    names = rows[keep]
    d = {}
    n = len(names)
    iu, ju = np.triu_indices(n, k=1)
    vals = R[iu, ju]
    m = np.abs(vals) >= TAU
    for a, b, v in zip(iu[m], ju[m], vals[m]):
        d[tuple(sorted((names[a], names[b])))] = float(v)
    return d, set(names), len(sids)

summary = []
for cond, g_f, g_m in CONDS:
    Em, fm, nm = group_net(g_m)   # male G1
    Ef, ff, nf = group_net(g_f)   # female G2
    keys = set(Em) | set(Ef)
    recs = []
    n_true = n_nm = 0
    for k in keys:
        r1 = Em.get(k, 0.0); r2 = Ef.get(k, 0.0)
        d = abs(r1 - r2)
        if d < DELTA:
            continue
        a, b = k
        in1, in2 = k in Em, k in Ef
        node_missing = ((not in1) and (a not in fm or b not in fm)) or \
                       ((not in2) and (a not in ff or b not in ff))
        art = "node_missing" if node_missing else "true_change"
        if art == "true_change": n_true += 1
        else: n_nm += 1
        recs.append(dict(Node1=a, Node2=b, r_male=r1, r_female=r2, Delta=r1 - r2,
                         Abs_Delta=d, artifact=art))
    df = pd.DataFrame(recs)
    df.to_csv(f"{OUT}/species_diffnet_{cond}.tsv", sep="\t", index=False)
    summary.append(dict(cohort=cohort, condition=cond, n_male=nm, n_female=nf,
                        n_feat_male=len(fm), n_feat_female=len(ff),
                        n_edges_male=len(Em), n_edges_female=len(Ef),
                        n_diff=len(df), n_true_change=n_true, n_node_missing=n_nm))
    print(f"[{cohort}][{cond}] male={nm}(f{len(fm)}) female={nf}(f{len(ff)}) "
          f"edges m/f={len(Em)}/{len(Ef)} diff={len(df)} true={n_true} node_missing={n_nm}",
          flush=True)

pd.DataFrame(summary).to_csv(f"{OUT}/species_diffnet_summary.csv", index=False)
print(f"[{cohort}] species diffnet DONE -> {OUT}", flush=True)
