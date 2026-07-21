# Differential Analysis of Microbial Association Networks — Reproducible Code

This package accompanies the revised manuscript *"Differential Analysis of Microbial
Association Networks"* (Milano & Guzzi). It contains the core analysis pipeline used
on all three cohorts (ACVD, IBD, T2D) and a **self-contained working example** that
runs in seconds without any external data or database access.

The code is intentionally small and dependency-light. All statistical choices
(prevalence filtering, CLR transform, Spearman correlation, τ/δ thresholds,
artifact decomposition, permutation nulls) are implemented in one readable module,
`diffnet/pipeline_core.py`, and are exercised end-to-end by `example/run_example.py`.

---

## Contents

```
reproducible_code/
├── README.md                 # this file
├── requirements.txt          # Python dependencies (numpy, scipy, pandas)
├── diffnet/
│   └── pipeline_core.py       # the analysis pipeline (published + robust variants)
└── example/
    └── run_example.py         # runnable working example (positive + negative control)
```

The full set of driver scripts used for the paper's cohort-level runs
(`run_diffnet_fast.py`, `pathway_enrich_perm.py`, `run_pathway_enrichment.py`,
`benchmark_netcomi_species.R`, `da_baseline_species.R`, the cohort-fetch scripts,
`run_simulation.py`, `run_sweep.py`, `run_downsample.py`, etc.) are archived alongside
the results tables and are all thin wrappers around `pipeline_core.py`. They read the
cohort matrices as sparse `.mtx` files plus `_rows.txt` / `_cols.txt` / `_metadata.tsv`
from a data directory (set with the `DIFFNET_DATA` environment variable). The public
cohorts are obtained from `curatedMetagenomicData`:

| Cohort | `curatedMetagenomicData` study | Layer used |
|---|---|---|
| ACVD | `JieZ_2017` | gene families, species relative abundance |
| IBD  | `HMP_2019_ibdmdb` (longitudinal iHMP) | gene families; analysed **subject-level** (one first visit / subject) |
| T2D  | `MetaCardis_2020_a` | gene families, MetaCyc `pathway_abundance` |

---

## Quick start (working example)

```bash
pip install -r requirements.txt        # numpy, scipy, pandas
python example/run_example.py          # ~15-40 s
```

You should see two blocks — a **positive control** (a known rewiring signal planted
in one group) and a **negative control** (nothing planted). Options:

```bash
python example/run_example.py --nperm 500 --seed 1 --n_features 80
```

### What the example demonstrates (and why it matches the paper)

The example simulates two "sex" groups of compositional (relative-abundance) samples
and builds a sex differential network with the **same code** used on the real data.

1. **The pipeline localises real rewiring.** In the positive control, 8 feature pairs
   are strongly co-abundant in group G1 and uncorrelated in G2. These ground-truth
   edges rank at the **top** of the |Δ| = |ρ<sub>G1</sub> − ρ<sub>G2</sub>| ordering
   (typically 7/8 in the top 10), with roughly double the mean |Δ| of background edges.
   The differential-network machinery works.

2. **The aggregate rewired-edge *count* is not, by itself, evidence.** The centred
   log-ratio transform on a compositional table induces many small-|Δ| background
   edges, so the total number of "true_change" edges (`n_true`) is large even under
   the null. A label-shuffling permutation test on `n_true` is therefore
   **null-consistent in both controls** — including when real edges exist. This is
   precisely why the revised manuscript does **not** treat raw rewired-edge counts as
   a discovery, and instead reports permutation nulls, an artifact decomposition, and
   edge-level effect sizes.

On the real cohorts, every disease-vs-control sex contrast behaves like the negative
control at the aggregate level: large raw counts, but permutation-null-consistent
`true_change` (see `permutation_pvalues_*.csv` and `pathway_permutation_crosscohort.csv`
in the results tables).

---

## The pipeline (`diffnet/pipeline_core.py`)

### Parameters (published defaults)

| Constant | Value | Meaning |
|---|---|---|
| `MIN_PREVALENCE` | 0.10 | drop features present in <10% of a group's samples |
| `MAX_PREVALENCE` | 0.90 | drop features present in >90% of a group's samples |
| `MIN_MEAN` | 1e-6 | minimum mean abundance (gene-family scale) |
| `MIN_VAR` | 1e-8 | minimum variance |
| `PSEUDOCOUNT` | 1e-6 | log/CLR pseudocount |
| `TAU` | 0.30 | correlation threshold for an edge (|ρ| ≥ τ) |
| `DELTA` | 0.30 | minimum |Δρ| for a differential edge |

For **pathway-abundance** networks the mean/variance floors are dropped (relative
abundances are ~50× smaller than gene-family values); a prevalence-only filter is used.
Correlations are Spearman on the CLR- (or log-) transformed matrix. Groups are compared
male (G1) vs female (G2); Δ<sub>ij</sub> = ρ<sup>G1</sup><sub>ij</sub> − ρ<sup>G2</sup><sub>ij</sub>;
the node rewiring score is R<sub>i</sub> = Σ<sub>k</sub> |Δ<sub>ik</sub>|.

### Key functions

| Function | Role |
|---|---|
| `load_cohort(cohort, kind)` | read sparse `.mtx` + row/col ids from `$DIFFNET_DATA` |
| `load_meta(cohort)` | read sample metadata; build `study_condition_gender` group labels |
| `filter_features(X_group)` | published prevalence/mean/variance filter |
| `log_transform` / `clr_transform` | log(x+ε) or per-sample centred log-ratio |
| `residualize(mat, covars)` | regress out age/BMI/depth/antibiotics/country |
| `spearman_matrix` / `edges_from_corr` | correlation network above τ |
| `build_group_network(...)` | one group's edge set (with optional shared-feature mode) |
| `classify_edge` / `differential_network` | build the differential network **and** tag each edge as `true_change`, `node_missing_G1/G2`, or `both_missing` |
| `rewiring_scores` | per-node aggregated |Δ| |

### The artifact decomposition (central to the revision)

`differential_network()` labels every flagged edge by *why* it differs:

- **`true_change`** — both endpoints pass filtering in **both** groups, so the change is
  a genuine correlation shift that could in principle be tested.
- **`node_missing_G1` / `node_missing_G2`** — an endpoint is absent from one group's
  feature set (removed by prevalence/abundance filtering), so the edge trivially "changes"
  regardless of biology.
- **`both_missing`** — both endpoints missing in one group.

On the real cohorts, 80–91 % of "exclusive" edges are `node_missing` artifacts, not
`true_change`; separating these is what turns an apparently huge rewiring signal into a
null-consistent one.

---

## Reproducing the cohort-level results

The cohort runs require the three `curatedMetagenomicData` studies above (multi-GB
gene-family matrices), so they are not bundled here. Once the matrices are staged in
`$DIFFNET_DATA` as `<cohort>_<kind>.mtx` (+ `_rows.txt`, `_cols.txt`, `_metadata.tsv`):

```bash
export DIFFNET_DATA=/path/to/data
# global differential network + artifact decomposition + permutation null:
python run_diffnet_fast.py acvd --transform clr --nperm 1000
# MetaCyc pathway rewiring + degree-controlled permutation:
python run_pathway_enrichment.py t2d
python pathway_enrich_perm.py t2d --nperm 1000
# IBD must be run subject-level to avoid pseudoreplication:
python run_diffnet_fast.py ibd --ibd-subject --transform clr --nperm 1000
```

Outputs are written as CSV tables (`permutation_pvalues_*.csv`,
`pathway_permutation_crosscohort.csv`, `artifact_decomposition_*.csv`, …) matching those
in the manuscript's *Robustness and Validation* section.

---

## Notes on determinism

The working example seeds NumPy's `default_rng`; results are reproducible for a fixed
`--seed`. Small permutation p-values (`--nperm 200`) will fluctuate at the ±0.02 level;
increase `--nperm` for tighter estimates. The RuntimeWarnings that CLR can emit on an
all-zero permuted feature row are expected and silenced in the example.
