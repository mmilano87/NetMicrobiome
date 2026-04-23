import os
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.stats import spearmanr

BASE_DIR = "data/ibd"
RESULTS_DIR = "results/pathway_enrichment"

TOP_N = 200

def load_matrix(mtx, rows, cols):
    X = mmread(mtx).toarray()
    r = pd.read_csv(rows, sep="\t", header=None)[0].tolist()
    c = pd.read_csv(cols, sep="\t", header=None)[0].tolist()
    return pd.DataFrame(X, index=r[:X.shape[0]], columns=c[:X.shape[1]])

def spearman_safe(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    return spearmanr(x, y)

def bh(p):
    p = np.array(p)
    n = len(p)
    o = np.argsort(p)
    r = p[o]
    adj = np.empty(n)
    prev = 1
    for i in range(n - 1, -1, -1):
        val = r[i] * n / (i + 1)
        prev = min(prev, val)
        adj[i] = prev
    out = np.empty(n)
    out[o] = adj
    return out

def load_edges(path):
    df = pd.read_csv(path, sep=" ", header=None)
    df.columns = ["Node1", "Node2", "Weight"]
    return df

def load_rew(path):
    return pd.read_csv(path, sep="\t").sort_values("Score", ascending=False)

def run(label, male_edge, female_edge, dn_edge, rew_file, G, P):

    out = os.path.join(RESULTS_DIR, label)
    os.makedirs(out, exist_ok=True)

    male = load_edges(male_edge)
    female = load_edges(female_edge)
    dn = load_edges(dn_edge)
    rew = load_rew(rew_file)

    male_nodes = pd.concat([male["Node1"], male["Node2"]]).value_counts().head(TOP_N).index
    female_nodes = pd.concat([female["Node1"], female["Node2"]]).value_counts().head(TOP_N).index
    dn_nodes = rew["Node"].head(TOP_N)

    def score(nodes):
        nodes = list(set(nodes) & set(G.index))
        return G.loc[nodes].mean(axis=0)

    modules = {
        "male": score(male_nodes),
        "female": score(female_nodes),
        "dn": score(dn_nodes)
    }

    results = {}

    for k, s in modules.items():
        rows = []
        for p in P.index:
            rho, pv = spearman_safe(s, P.loc[p])
            if not np.isnan(rho):
                rows.append([p, rho, pv])

        df = pd.DataFrame(rows, columns=["Pathway", "rho", "pval"])
        df["FDR"] = bh(df["pval"])
        results[k] = df[(df["FDR"] < 0.05) & (df["rho"].abs() > 0.3)]

    results["dn"].to_csv(os.path.join(out, "dn_significant.txt"), sep="\t", index=False)

def main():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    G = load_matrix(
        os.path.join(BASE_DIR, "gene_families_matrix.mtx"),
        os.path.join(BASE_DIR, "gene_families_rows.txt"),
        os.path.join(BASE_DIR, "gene_families_cols.txt")
    )

    P = load_matrix(
        os.path.join(BASE_DIR, "pathway_abundance_matrix.mtx"),
        os.path.join(BASE_DIR, "pathway_abundance_rows.txt"),
        os.path.join(BASE_DIR, "pathway_abundance_cols.txt")
    )

    run(
        "IBD",
        "results/networks/IBD_male/edges.txt",
        "results/networks/IBD_female/edges.txt",
        "results/differential_networks/IBD_male_vs_female/edges.txt",
        "results/differential_networks/IBD_male_vs_female/rewiring_scores.txt",
        G, P
    )

if __name__ == "__main__":
    main()