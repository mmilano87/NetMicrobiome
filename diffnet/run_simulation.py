#!/usr/bin/env python3
"""run_simulation.py -- ground-truth simulation for the differential-network method (Reviewer 3.6).

We generate two groups whose association structure differs in a KNOWN set of edges, sample data
at realistic per-group sample sizes, run the exact pipeline rewiring detection, and measure how
well it recovers the truly-rewired edges (precision/recall) and how finite-n noise inflates the
apparent rewiring. We also verify the artifact decomposition: when we additionally drop features
in one group (mimicking prevalence filtering), those edges should be classified node_missing, not
true_change.

Design
------
- P features, multivariate Gaussian latent -> rank/Spearman networks (matches the pipeline).
- Group 1 correlation matrix R1: block structure with `n_edges_base` strong pairs (|rho|>=~0.5).
- Group 2 = R1 but with a KNOWN rewired set:
    * `n_rewire_on`  base non-edges turned into strong edges (appear in G2 only)
    * `n_rewire_off` base edges removed (disappear in G2)
    * `n_rewire_sign` base edges sign-flipped
  All other pairs are identical in expectation -> NOT truly rewired.
- Sample n1, n2 observations; Spearman; threshold tau; differential_network; compare Type against
  the ground-truth rewired set. Repeat R reps.

Usage:
  python run_simulation.py --p 300 --n1 157 --n2 53 --reps 30 --tau 0.30 --delta 0.30 \
      [--drop-frac 0.15]   # optionally drop features in G2 to test node_missing decomposition
Output -> robust/simulation/sim_results.csv  and  sim_summary.json
"""
import argparse, os, json, math
import numpy as np
import pandas as pd
from numpy.linalg import eigh

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_core import (edges_from_corr, differential_network, spearman_matrix,
                           classify_edge, TAU, DELTA)

OUT = _os.path.join(_os.environ.get("DIFFNET_OUT", "robust"), "simulation")
os.makedirs(OUT, exist_ok=True)


def nearest_pd(A, eps=1e-6):
    """Project a symmetric matrix to the nearest positive-definite correlation matrix."""
    A = (A + A.T) / 2
    w, V = eigh(A)
    w = np.clip(w, eps, None)
    A2 = (V * w) @ V.T
    # renormalize to unit diagonal (correlation)
    d = np.sqrt(np.diag(A2))
    A2 = A2 / np.outer(d, d)
    np.fill_diagonal(A2, 1.0)
    return A2


def build_truth(P, n_edges_base, n_rewire_on, n_rewire_off, n_rewire_sign, rng, rho=0.6):
    """Return (R1, R2, rewired_set) where rewired_set is the set of truly-changed pairs."""
    # all upper-triangle pairs
    iu = np.triu_indices(P, k=1)
    all_pairs = list(zip(iu[0].tolist(), iu[1].tolist()))
    rng.shuffle(all_pairs)

    R1 = np.eye(P)
    # base edges (shared by both groups)
    base = all_pairs[:n_edges_base]
    ptr = n_edges_base
    for (i, j) in base:
        R1[i, j] = R1[j, i] = rho

    R2 = R1.copy()
    rewired = set()

    # rewire OFF: remove some base edges in G2
    off = base[:n_rewire_off]
    for (i, j) in off:
        R2[i, j] = R2[j, i] = 0.0
        rewired.add((i, j))
    # rewire SIGN: flip some other base edges in G2
    sign = base[n_rewire_off:n_rewire_off + n_rewire_sign]
    for (i, j) in sign:
        R2[i, j] = R2[j, i] = -rho
        rewired.add((i, j))
    # rewire ON: add new edges in G2 from the non-edge pool
    on = all_pairs[ptr:ptr + n_rewire_on]
    for (i, j) in on:
        R2[i, j] = R2[j, i] = rho
        rewired.add((i, j))

    R1 = nearest_pd(R1)
    R2 = nearest_pd(R2)
    return R1, R2, rewired


def sample_group(R, n, rng):
    """Sample n obs from N(0, R); return as (features x samples) to match pipeline convention."""
    P = R.shape[0]
    L = np.linalg.cholesky(R)
    Z = rng.standard_normal((n, P))
    X = Z @ L.T                      # n x P
    return X.T                       # P x n (features x samples)


def run_once(P, n1, n2, tau, delta, truth_params, drop_frac, rng):
    R1, R2, rewired = build_truth(P, **truth_params, rng=rng)
    feat_names = [f"f{i}" for i in range(P)]

    X1 = sample_group(R1, n1, rng)   # P x n1
    X2 = sample_group(R2, n2, rng)   # P x n2

    feats1 = list(feat_names)
    feats2 = list(feat_names)

    # optional: drop features in G2 to mimic prevalence filtering -> should become node_missing
    dropped = set()
    if drop_frac and drop_frac > 0:
        k = int(round(drop_frac * P))
        drop_idx = rng.choice(P, size=k, replace=False)
        dropped = set(drop_idx.tolist())
        keep2 = [i for i in range(P) if i not in dropped]
        X2 = X2[keep2, :]
        feats2 = [feat_names[i] for i in keep2]

    # pipeline: Spearman on each group's (present) features, threshold, differential network
    c1 = spearman_matrix(X1)
    c2 = spearman_matrix(X2)
    d1 = edges_from_corr(c1, feats1, tau=tau)
    d2 = edges_from_corr(c2, feats2, tau=tau)
    dn = differential_network(d1, d2, feats1, feats2, tau=tau, delta=delta)

    # map rewired ground-truth (index pairs) to name pairs (sorted, matching edges_from_corr keys)
    def key(i, j):
        a, b = feat_names[i], feat_names[j]
        return tuple(sorted((a, b)))
    truth_keys = {key(i, j) for (i, j) in rewired}

    # detected rewired edges = all rows in dn (any Type)
    detected = set(zip(dn["Node1"], dn["Node2"])) if len(dn) else set()
    detected = {tuple(sorted(e)) for e in detected}

    # Restrict evaluation to pairs that are TESTABLE in both groups (both endpoints present),
    # because dropped-feature edges are node_missing by construction (evaluated separately).
    present_both = set(feats1) & set(feats2)
    def testable(e): return e[0] in present_both and e[1] in present_both

    truth_testable = {e for e in truth_keys if testable(e)}
    detected_testable = {e for e in detected if testable(e)}

    tp = len(truth_testable & detected_testable)
    fp = len(detected_testable - truth_testable)
    fn = len(truth_testable - detected_testable)
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = (2 * precision * recall / (precision + recall)
          if (precision and recall and precision + recall > 0) else np.nan)

    # artifact-decomposition check: of detected edges, how many are true_change vs node_missing;
    # and of node_missing edges, how many involve a dropped feature (should be ~all).
    n_true_change = int((dn["artifact"] == "true_change").sum()) if len(dn) else 0
    n_node_missing = int(dn["artifact"].isin(["node_missing_G1", "node_missing_G2"]).sum()) if len(dn) else 0
    # among true_change detected edges, precision against ground truth (should be high)
    tc_edges = {tuple(sorted((r.Node1, r.Node2)))
                for _, r in dn[dn["artifact"] == "true_change"].iterrows()} if len(dn) else set()
    tc_tp = len(tc_edges & truth_keys)
    tc_precision = tc_tp / len(tc_edges) if tc_edges else np.nan

    return dict(
        n_detected=len(detected), n_detected_testable=len(detected_testable),
        n_truth=len(truth_keys), n_truth_testable=len(truth_testable),
        tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1,
        n_true_change=n_true_change, n_node_missing=n_node_missing,
        true_change_precision=tc_precision, n_dropped=len(dropped),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=300)
    ap.add_argument("--n1", type=int, default=157)   # e.g. ACVD male
    ap.add_argument("--n2", type=int, default=53)    # e.g. ACVD female
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--n-edges-base", type=int, default=400)
    ap.add_argument("--n-rewire-on", type=int, default=60)
    ap.add_argument("--n-rewire-off", type=int, default=60)
    ap.add_argument("--n-rewire-sign", type=int, default=30)
    ap.add_argument("--drop-frac", type=float, default=0.0)
    ap.add_argument("--label", default="balancedN")
    args = ap.parse_args()

    truth_params = dict(n_edges_base=args.n_edges_base, n_rewire_on=args.n_rewire_on,
                        n_rewire_off=args.n_rewire_off, n_rewire_sign=args.n_rewire_sign)

    rows = []
    for r in range(args.reps):
        rng = np.random.default_rng(r)
        res = run_once(args.p, args.n1, args.n2, args.tau, args.delta,
                       truth_params, args.drop_frac, rng)
        res.update(rep=r, p=args.p, n1=args.n1, n2=args.n2, tau=args.tau, delta=args.delta,
                   drop_frac=args.drop_frac, label=args.label,
                   n_true_edges=args.n_rewire_on + args.n_rewire_off + args.n_rewire_sign)
        rows.append(res)
        if (r + 1) % 10 == 0:
            print(f"  rep {r+1}/{args.reps}: precision={res['precision']:.3f} "
                  f"recall={res['recall']:.3f} fp={res['fp']} tp={res['tp']}", flush=True)

    df = pd.DataFrame(rows)
    tag = f"{args.label}_p{args.p}_n{args.n1}v{args.n2}_drop{args.drop_frac}"
    csv = os.path.join(OUT, f"sim_{tag}.csv")
    df.to_csv(csv, index=False)

    summ = {k: float(df[k].mean()) for k in
            ["precision", "recall", "f1", "tp", "fp", "fn",
             "n_detected_testable", "n_true_change", "n_node_missing", "true_change_precision"]}
    summ.update(reps=args.reps, p=args.p, n1=args.n1, n2=args.n2, tau=args.tau, delta=args.delta,
                drop_frac=args.drop_frac, n_true_edges=int(df["n_true_edges"].iloc[0]))
    with open(os.path.join(OUT, f"sim_summary_{tag}.json"), "w") as fh:
        json.dump(summ, fh, indent=2)

    print(f"\nSIM_DONE {tag}")
    print(f"  precision={summ['precision']:.3f} recall={summ['recall']:.3f} f1={summ['f1']:.3f}")
    print(f"  mean tp={summ['tp']:.1f} fp={summ['fp']:.1f} fn={summ['fn']:.1f} "
          f"(true edges={summ['n_true_edges']})")
    print(f"  detected(testable)={summ['n_detected_testable']:.1f}  "
          f"true_change={summ['n_true_change']:.1f}  node_missing={summ['n_node_missing']:.1f}")
    print(f"  true_change_precision={summ['true_change_precision']:.3f}")
    print(f"  -> {csv}")


if __name__ == "__main__":
    main()
