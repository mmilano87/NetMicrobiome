#!/usr/bin/env python3
"""run_downsample.py -- equal-n downsampling experiment (Reviewer 3.3).

Reviewer 3 asks whether rewiring is sex-associated or merely sample-size-associated,
given the imbalance (e.g. 53 ACVD-F vs 157 ACVD-M). Here we downsample the LARGER sex
to n = min(n_male, n_female) so both networks are built on EQUAL sample sizes, recompute
the differential network, and repeat over R random subsamples to get a stable estimate.

We report, per condition: distribution of n_rewired, n_true, frac_true under balanced
sampling, versus the (imbalanced) observed values. If balanced rewiring is similar to
observed, the imbalance was not the driver; if true_change collapses toward the null, the
apparent signal was size-driven.

Reuses the exact pipeline (same filtering, same transform, same tau/delta).

Usage:
  python3 run_downsample.py acvd --variant clr --reps 50
Outputs -> /mnt/shared-workspace/micro/robust/<cohort>/downsample_<variant>.csv
"""
import argparse, os
import numpy as np
import pandas as pd
import sys
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # import pipeline_core from this folder
from pipeline_core import (load_cohort, load_meta, build_group_network,
                           differential_network, rewiring_scores, TAU, DELTA)

CONDS = {
    "acvd": [("control", "control_female", "control_male"),
             ("ACVD", "ACVD_female", "ACVD_male")],
    "ibd":  [("control", "control_female", "control_male"),
             ("IBD", "IBD_female", "IBD_male")],
    "t2d":  [("control", "control_female", "control_male"),
             ("T2D", "T2D_female", "T2D_male")],
}


def subset_first_visit(meta):
    m = meta.reset_index()
    m["_d"] = pd.to_numeric(m.get("days_from_first_collection", 0), errors="coerce").fillna(0)
    keep = m.loc[m.groupby("subject_id")["_d"].idxmin(), "sample_id"]
    return meta.loc[meta.index.isin(set(keep))]


def diffnet_stats(X, rows, sample_ids, sm, sf, transform, tau, delta):
    ed_m, feat_m, _ = build_group_network(X, rows, sample_ids, sm, transform=transform, tau=tau)
    ed_f, feat_f, _ = build_group_network(X, rows, sample_ids, sf, transform=transform, tau=tau)
    dn = differential_network(ed_m, ed_f, feat_m, feat_f, tau=tau, delta=delta)
    if len(dn) == 0:
        return dict(n_rewired=0, n_true=0, frac_true=np.nan)
    n = len(dn); nt = int((dn.artifact == "true_change").sum())
    return dict(n_rewired=n, n_true=nt, frac_true=nt / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cohort")
    ap.add_argument("--variant", default="clr", choices=["log", "clr"])
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--ibd-mode", default="samples", choices=["samples", "subject"])
    args = ap.parse_args()

    X, rows, sample_ids = load_cohort(args.cohort, "gene_families")
    meta = load_meta(args.cohort)
    if args.cohort == "ibd" and args.ibd_mode == "subject":
        meta = subset_first_visit(meta)
    smap = set(sample_ids)

    outdir = _os.path.join(_os.environ.get("DIFFNET_OUT", "robust"), args.cohort)
    os.makedirs(outdir, exist_ok=True)

    records = []
    for cond, g_f, g_m in CONDS[args.cohort]:
        sm = [s for s in meta.index[meta["group"] == g_m] if s in smap]
        sf = [s for s in meta.index[meta["group"] == g_f] if s in smap]
        n_bal = min(len(sm), len(sf))
        print(f"[{cond}] male={len(sm)} female={len(sf)} -> balanced n={n_bal}", flush=True)
        if n_bal < 10:
            print(f"  n_bal<10, skip"); continue

        # observed (imbalanced) for reference
        obs = diffnet_stats(X, rows, sample_ids, sm, sf, args.variant, TAU, DELTA)
        records.append(dict(cohort=args.cohort, variant=args.variant, condition=cond,
                            rep=-1, kind="observed_imbalanced", n_bal=n_bal,
                            n_m=len(sm), n_f=len(sf), **obs))

        for r in range(args.reps):
            rng = np.random.default_rng(r)
            sm_d = list(rng.choice(sm, n_bal, replace=False)) if len(sm) > n_bal else sm
            sf_d = list(rng.choice(sf, n_bal, replace=False)) if len(sf) > n_bal else sf
            st = diffnet_stats(X, rows, sample_ids, sm_d, sf_d, args.variant, TAU, DELTA)
            records.append(dict(cohort=args.cohort, variant=args.variant, condition=cond,
                                rep=r, kind="balanced", n_bal=n_bal,
                                n_m=n_bal, n_f=n_bal, **st))
        # quick summary
        bal = pd.DataFrame([x for x in records if x["condition"] == cond and x["kind"] == "balanced"])
        print(f"  balanced n_rewired mean={bal.n_rewired.mean():.0f}+/-{bal.n_rewired.std():.0f} "
              f"n_true mean={bal.n_true.mean():.0f}+/-{bal.n_true.std():.0f} "
              f"(obs imbalanced n_rewired={obs['n_rewired']} n_true={obs['n_true']})", flush=True)

    df = pd.DataFrame(records)
    out = os.path.join(outdir, f"downsample_{args.variant}.csv")
    df.to_csv(out, index=False)
    print(f"DOWNSAMPLE_DONE {args.cohort} {args.variant} -> {out} ({len(df)} rows)", flush=True)


if __name__ == "__main__":
    main()
