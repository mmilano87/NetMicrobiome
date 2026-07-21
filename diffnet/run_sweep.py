#!/usr/bin/env python3
"""run_sweep.py -- threshold sensitivity sweep (Reviewer 1.3 / 3.2).

Computes the group correlation matrices ONCE per (cohort, condition, transform),
then sweeps tau x delta, recording for each combination:
  n_rewired, n_true, frac_true, n_exclusive, n_sign, n_diffweighted,
  and per-group node/edge/density.

This is cheap: the O(p^2) Spearman is computed only once per group (~350-390
features); every (tau,delta) is a fast re-threshold + re-classify.

Usage:
  python3 run_sweep.py acvd --variant clr
  python3 run_sweep.py acvd --variant log
Outputs -> /mnt/shared-workspace/micro/robust/<cohort>/sweep_<variant>.csv
"""
import argparse, os, json, itertools
import numpy as np
import pandas as pd

import sys
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # import pipeline_core from this folder
from pipeline_core import (load_cohort, load_meta, build_group_network, edges_from_corr,
                           differential_network, rewiring_scores)


def subset_first_visit(meta):
    """IBD subject-level: keep the earliest-visit sample per subject (matches run_diffnet_fast)."""
    m = meta.reset_index()
    m["_d"] = pd.to_numeric(m.get("days_from_first_collection", 0), errors="coerce").fillna(0)
    keep = m.loc[m.groupby("subject_id")["_d"].idxmin(), "sample_id"]
    return meta.loc[meta.index.isin(set(keep))]

CONDS = {
    "acvd": [("control", "control_female", "control_male"),
             ("ACVD", "ACVD_female", "ACVD_male")],
    "ibd":  [("control", "control_female", "control_male"),
             ("IBD", "IBD_female", "IBD_male")],
    "t2d":  [("control", "control_female", "control_male"),
             ("T2D", "T2D_female", "T2D_male")],
}

TAUS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
DELTAS = [0.1, 0.2, 0.3, 0.4, 0.5]


def gstats(dn):
    if len(dn) == 0:
        return dict(n_rewired=0, n_true=0, frac_true=np.nan, n_exclusive=0,
                    n_sign=0, n_diffweighted=0, max_node_score=0.0)
    n = len(dn)
    n_true = int((dn.artifact == "true_change").sum())
    return dict(
        n_rewired=n, n_true=n_true, frac_true=n_true / n,
        n_exclusive=int(dn.Type.isin(["exclusive_G1", "exclusive_G2"]).sum()),
        n_sign=int((dn.Type == "sign_changed").sum()),
        n_diffweighted=int((dn.Type == "differentially_weighted").sum()),
        max_node_score=float(rewiring_scores(dn).Score.max()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cohort")
    ap.add_argument("--variant", default="clr", choices=["log", "clr"])
    ap.add_argument("--ibd-mode", default="samples", choices=["samples", "subject"])
    args = ap.parse_args()

    X, rows, sample_ids = load_cohort(args.cohort, "gene_families")
    meta = load_meta(args.cohort)
    if args.cohort == "ibd" and args.ibd_mode == "subject":
        meta = subset_first_visit(meta)
        print(f"[ibd] subject-level: {len(meta)} samples", flush=True)

    smap = set(sample_ids)
    outdir = _os.path.join(_os.environ.get("DIFFNET_OUT", "robust"), args.cohort)
    os.makedirs(outdir, exist_ok=True)

    records = []
    for cond, g_f, g_m in CONDS[args.cohort]:
        sm = [s for s in meta.index[meta["group"] == g_m] if s in smap]
        sf = [s for s in meta.index[meta["group"] == g_f] if s in smap]
        print(f"[{cond}] male={len(sm)} female={len(sf)}", flush=True)

        # correlations ONCE (tau here only affects the returned edge dict, which we
        # rebuild per-tau below; feature sets are tau-independent)
        _, feat_m, corr_m = build_group_network(X, rows, sample_ids, sm,
                                                transform=args.variant, tau=0.0)
        _, feat_f, corr_f = build_group_network(X, rows, sample_ids, sf,
                                                transform=args.variant, tau=0.0)
        print(f"  features: male={len(feat_m)} female={len(feat_f)}", flush=True)

        for tau in TAUS:
            d_m = edges_from_corr(corr_m, feat_m, tau=tau)
            d_f = edges_from_corr(corr_f, feat_f, tau=tau)
            # NOTE DN convention: male=G1, female=G2 (Delta = rho_male - rho_female)
            for delta in DELTAS:
                dn = differential_network(d_m, d_f, feat_m, feat_f, tau=tau, delta=delta)
                st = gstats(dn)
                st.update(dict(cohort=args.cohort, variant=args.variant, condition=cond,
                               tau=tau, delta=delta,
                               n_edges_male=len(d_m), n_edges_female=len(d_f),
                               n_feat_male=len(feat_m), n_feat_female=len(feat_f)))
                records.append(st)
            print(f"  tau={tau}: male_edges={len(d_m)} female_edges={len(d_f)}", flush=True)

    df = pd.DataFrame(records)
    out = os.path.join(outdir, f"sweep_{args.variant}.csv")
    df.to_csv(out, index=False)
    print(f"SWEEP_DONE {args.cohort} {args.variant} -> {out} ({len(df)} rows)", flush=True)


if __name__ == "__main__":
    main()
