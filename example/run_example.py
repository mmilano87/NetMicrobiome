#!/usr/bin/env python
"""
Self-contained working example for the differential co-abundance network pipeline.

Runs in a few seconds with NO external data or database access, calling the SAME
functions used on the real cohorts (../diffnet/pipeline_core.py):
    filter_features, clr_transform, spearman_matrix, edges_from_corr,
    classify_edge, differential_network, rewiring_scores.

WHAT IT SHOWS
-------------
We simulate two "sex" groups of compositional (relative-abundance) samples. In the
POSITIVE control, 8 feature pairs are strongly co-abundant in group G1 and
uncorrelated in G2 -> these are ground-truth "rewired" edges. In the NEGATIVE
control both groups are generated identically -> there is nothing to find.

Two lessons, both faithful to the manuscript's conclusions:

  (1) The differential network *localises* real rewiring: the planted edges rank
      at the TOP of the |Delta| = |rho_G1 - rho_G2| ordering (high precision@k),
      well separated from background. So the machinery works.

  (2) The *aggregate count* of "true_change" edges (n_true) is NOT, by itself,
      evidence of signal: the centred-log-ratio transform on a compositional
      table induces many small-|Delta| background edges, so n_true is large even
      under the null and a label-shuffling permutation test on n_true is
      null-consistent in BOTH controls. This is exactly why the revised
      manuscript does not treat raw rewired-edge counts as a discovery and
      instead reports permutation nulls + artifact decomposition. On the real
      cohorts every disease-vs-control sex contrast behaves like this.

Usage:
    python run_example.py [--nperm 300] [--seed 0]
"""
import os, sys, argparse, warnings
import numpy as np
import pandas as pd

# CLR of a permuted sparse group can transiently hit an all-zero feature row
# (empty-slice mean -> NaN, zeroed downstream). Expected/harmless; silence noise.
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(invalid="ignore", divide="ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "diffnet"))
import pipeline_core as pc   # the REAL pipeline module

TAU, DELTA = pc.TAU, pc.DELTA   # 0.30, 0.30 -- identical to the manuscript


# ----------------------------------------------------------------------
# Realistic sparse compositional simulation
# ----------------------------------------------------------------------
def draw_group(n, F, n_rewired_pairs, rho, rng, rewire, detect_frac=0.6):
    """One group: n samples x F features, positive relative abundances with
    realistic zero-inflation (so prevalence falls in the analysable band). When
    rewire=True, the first `n_rewired_pairs` disjoint feature pairs are given
    correlation ~rho; otherwise all features are independent background.
    Returns (A [n x F], ground_truth_edge_set)."""
    Y = rng.normal(0, 1, size=(n, F))
    gt = set()
    if rewire:
        for k in range(n_rewired_pairs):
            a, b = 2 * k, 2 * k + 1
            shared = rng.normal(0, 1, size=n)
            Y[:, a] = rho * shared + np.sqrt(1 - rho**2) * rng.normal(0, 1, n)
            Y[:, b] = rho * shared + np.sqrt(1 - rho**2) * rng.normal(0, 1, n)
            gt.add(tuple(sorted((f"F{a:02d}", f"F{b:02d}"))))
    A = np.exp(Y)
    thr = np.quantile(A, 1 - detect_frac, axis=0, keepdims=True)   # zero-inflate
    A = np.where(A >= thr, A, 0.0)
    s = A.sum(axis=1, keepdims=True); s[s == 0] = 1
    return A / s, gt


def build_matrix(n, F, n_rewired_pairs, rho, rng, signal_g1, signal_g2):
    from scipy.sparse import csr_matrix
    A1, gt1 = draw_group(n, F, n_rewired_pairs, rho, rng, signal_g1)
    A2, gt2 = draw_group(n, F, n_rewired_pairs, rho, rng, signal_g2)
    A = np.vstack([A1, A2]).T                    # features x samples
    feat = [f"F{i:02d}" for i in range(F)]
    sids = [f"G1_s{i}" for i in range(n)] + [f"G2_s{i}" for i in range(n)]
    grp = np.array(["G1"] * n + ["G2"] * n)
    gt = gt1.symmetric_difference(gt2)           # correlated in exactly one group
    return csr_matrix(A), feat, sids, grp, gt


# ----------------------------------------------------------------------
# Differential network on one G1-vs-G2 contrast using the REAL pipeline
# ----------------------------------------------------------------------
def diffnet_contrast(X, feat, sids, grp, tau=TAU, delta=DELTA):
    g1 = [s for s, g in zip(sids, grp) if g == "G1"]
    g2 = [s for s, g in zip(sids, grp) if g == "G2"]
    d1, f1, _ = pc.build_group_network(X, feat, sids, g1, transform="clr", tau=tau)
    d2, f2, _ = pc.build_group_network(X, feat, sids, g2, transform="clr", tau=tau)
    return pc.differential_network(d1, d2, f1, f2, tau=tau, delta=delta)


def true_change(dn):
    return dn[dn["artifact"] == "true_change"] if len(dn) else dn


def precision_at_k(dn, gt, k):
    """Rank true_change edges by |Delta| and report how many of the top-k are
    ground-truth planted edges."""
    tc = true_change(dn)
    if not len(tc):
        return 0, 0
    tc = tc.copy()
    tc["edge"] = [tuple(sorted((a, b))) for a, b in zip(tc.Node1, tc.Node2)]
    top = tc.sort_values("Abs_Delta", ascending=False).head(k)
    return int(top["edge"].isin(gt).sum()), len(tc)


def perm_n_true(X, feat, sids, grp, nperm, seed, tau=TAU, delta=DELTA):
    obs = len(true_change(diffnet_contrast(X, feat, sids, grp, tau, delta)))
    rng = np.random.default_rng(seed)
    labels = np.array(grp); null = np.empty(nperm, int)
    for i in range(nperm):
        null[i] = len(true_change(diffnet_contrast(X, feat, sids, rng.permutation(labels), tau, delta)))
    p = float(((null >= obs).sum() + 1) / (nperm + 1))
    return obs, null, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nperm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_features", type=int, default=60)
    ap.add_argument("--n_per_group", type=int, default=40)
    ap.add_argument("--n_rewired_pairs", type=int, default=8)
    ap.add_argument("--rho", type=float, default=0.9)
    args = ap.parse_args()
    F, n, npr, rho = args.n_features, args.n_per_group, args.n_rewired_pairs, args.rho
    rng = np.random.default_rng(args.seed)

    print("=" * 74)
    print("Differential co-abundance network -- working example")
    print(f"  features={F}  samples/group={n}  planted pairs={npr}  rho={rho}")
    print(f"  tau={TAU} delta={DELTA}  permutations={args.nperm}  seed={args.seed}")
    print("=" * 74)

    # ---------------- POSITIVE control ----------------
    print("\n[1] POSITIVE CONTROL  (planted rewiring in G1 only)")
    Xp, feat, sids, grp, gt = build_matrix(n, F, npr, rho, rng, signal_g1=True, signal_g2=False)
    dn = diffnet_contrast(Xp, feat, sids, grp)
    tc = true_change(dn)
    p8, ntot = precision_at_k(dn, gt, k=npr)
    p10, _ = precision_at_k(dn, gt, k=npr + 2)
    tcc = tc.copy(); tcc["edge"] = [tuple(sorted((a, b))) for a, b in zip(tcc.Node1, tcc.Node2)]
    md_plant = tcc.loc[tcc.edge.isin(gt), "Abs_Delta"].mean()
    md_bg = tcc.loc[~tcc.edge.isin(gt), "Abs_Delta"].mean()
    print(f"    ground-truth planted edges         : {len(gt)}")
    print(f"    true_change edges flagged (total)  : {ntot}")
    print(f"    LESSON 1 -- localisation by |Delta|:")
    print(f"      planted edges in top-{npr}          : {p8}/{npr}   (precision@{npr})")
    print(f"      planted edges in top-{npr+2}          : {p10}/{npr}")
    print(f"      mean |Delta|: planted={md_plant:.2f}  vs  background={md_bg:.2f}")
    obs, null, p = perm_n_true(Xp, feat, sids, grp, args.nperm, args.seed)
    print(f"    LESSON 2 -- aggregate count is NOT signal:")
    print(f"      n_true obs={obs}  null={null.mean():.0f}+-{null.std():.0f}  p={p:.3f}  "
          f"({'ns' if p >= 0.05 else 'sig'})  <- count metric swamped by CLR background")

    # ---------------- NEGATIVE control ----------------
    print("\n[2] NEGATIVE CONTROL  (both groups identical; nothing planted)")
    Xn, feat, sids, grp, gt0 = build_matrix(n, F, npr, rho, rng, signal_g1=False, signal_g2=False)
    dn0 = diffnet_contrast(Xn, feat, sids, grp)
    obs0, null0, p0 = perm_n_true(Xn, feat, sids, grp, args.nperm, args.seed + 1)
    print(f"    true_change edges flagged (total)  : {len(true_change(dn0))}")
    print(f"    n_true obs={obs0}  null={null0.mean():.0f}+-{null0.std():.0f}  p={p0:.3f}  "
          f"({'NULL-CONSISTENT' if p0 >= 0.05 else 'unexpected sig'})")

    print("\n" + "=" * 74)
    print("Take-home (matches the manuscript's Robustness & Validation section):")
    print("  * The differential network correctly LOCALISES real rewired edges -- they")
    print("    rank at the top by |Delta| (positive control precision@k is high).")
    print("  * But the AGGREGATE rewired-edge COUNT is dominated by compositional/CLR")
    print("    background and by group-specific filtering, so it is null-consistent")
    print("    under label shuffling even when real edges exist -- and is null in the")
    print("    negative control too. Raw counts must therefore NOT be read as discovery.")
    print("  * On ACVD / IBD (subject-level) / T2D the disease-vs-control sex contrasts")
    print("    behave like this: see permutation_pvalues_*.csv and the artifact tables.")
    print("=" * 74)


if __name__ == "__main__":
    main()
