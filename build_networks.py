import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.io import mmread
from scipy.stats import spearmanr

BASE_DIR = "data/ibd"
OUTPUT_DIR = "results/networks"

MATRIX_FILE = os.path.join(BASE_DIR, "gene_families_matrix.mtx")
ROWS_FILE = os.path.join(BASE_DIR, "gene_families_rows.txt")
COLS_FILE = os.path.join(BASE_DIR, "gene_families_cols.txt")
META_FILE = os.path.join(BASE_DIR, "metadata.tsv")

MIN_PREVALENCE = 0.1
MAX_PREVALENCE = 0.9
MIN_MEAN = 1e-6
MIN_VAR = 1e-8
CORR_THRESHOLD = 0.3
PSEUDOCOUNT = 1e-6

GROUPS = [
    "control_female",
    "control_male",
    "IBD_female",
    "IBD_male"
]

def load_data():
    X = mmread(MATRIX_FILE).tocsr()
    rows = pd.read_csv(ROWS_FILE, sep="\t", header=None)[0].astype(str).tolist()
    cols = pd.read_csv(COLS_FILE, sep="\t", header=None)[0].astype(str).tolist()
    meta = pd.read_csv(META_FILE, sep="\t", index_col=0)
    return X, rows, cols, meta

def filter_features(X_group):
    prevalence = np.asarray(X_group.getnnz(axis=1)).ravel() / X_group.shape[1]
    mean = np.asarray(X_group.mean(axis=1)).ravel()
    mean2 = np.asarray(X_group.power(2).mean(axis=1)).ravel()
    var = mean2 - (mean**2)
    var[var < 0] = 0

    mask = (
        (prevalence >= MIN_PREVALENCE) &
        (prevalence <= MAX_PREVALENCE) &
        (mean >= MIN_MEAN) &
        (var >= MIN_VAR)
    )
    return np.where(mask)[0]

def build_network(X_group, feature_names):
    X_dense = X_group.toarray().astype(float)
    X_log = np.log(X_dense + PSEUDOCOUNT)

    corr, _ = spearmanr(X_log, axis=1)
    n = X_log.shape[0]
    corr = corr[:n, :n]
    corr = np.nan_to_num(corr)
    np.fill_diagonal(corr, 0)

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr[i, j]) >= CORR_THRESHOLD:
                edges.append((feature_names[i], feature_names[j], corr[i, j]))

    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    return G

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X, row_names, col_names, meta = load_data()
    meta["group"] = meta["study_condition"] + "_" + meta["gender"]

    sample_map = {s: i for i, s in enumerate(col_names)}

    for group in GROUPS:
        samples = meta.index[meta["group"] == group].tolist()
        idx = [sample_map[s] for s in samples if s in sample_map]

        X_group = X[:, idx]
        feat_idx = filter_features(X_group)

        X_group = X_group[feat_idx, :]
        feature_names = [row_names[i] for i in feat_idx]

        G = build_network(X_group, feature_names)

        group_dir = os.path.join(OUTPUT_DIR, group)
        os.makedirs(group_dir, exist_ok=True)

        nx.write_edgelist(G, os.path.join(group_dir, "edges.txt"), data=["weight"])
        pd.Series(list(G.nodes())).to_csv(os.path.join(group_dir, "nodes.txt"), index=False)

if __name__ == "__main__":
    main()