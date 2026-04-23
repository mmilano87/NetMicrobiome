import os
import numpy as np
import pandas as pd
import networkx as nx

BASE_DIR = "results/networks"
OUTPUT_DIR = "results/differential_networks"

TAU = 0.30
DELTA = 0.30

NETWORK_FILES = {
    "control_female": os.path.join(BASE_DIR, "control_female", "edges.txt"),
    "control_male": os.path.join(BASE_DIR, "control_male", "edges.txt"),
    "IBD_female": os.path.join(BASE_DIR, "IBD_female", "edges.txt"),
    "IBD_male": os.path.join(BASE_DIR, "IBD_male", "edges.txt"),
}

def load_edges(path):
    df = pd.read_csv(path, sep=" ")
    df.columns = ["Node1", "Node2", "Weight"]
    return df

def edge_dict(df):
    d = {}
    for _, r in df.iterrows():
        key = tuple(sorted([r["Node1"], r["Node2"]]))
        d[key] = r["Weight"]
    return d

def classify(r1, r2):
    p1 = abs(r1) > TAU
    p2 = abs(r2) > TAU
    diff = r1 - r2

    if p1 and p2 and r1 * r2 < 0:
        return "sign_changed"
    if p1 and not p2:
        return "exclusive_G1"
    if p2 and not p1:
        return "exclusive_G2"
    if max(abs(r1), abs(r2)) > TAU:
        if diff > DELTA:
            return "G1_specific"
        elif -diff > DELTA:
            return "G2_specific"
    return None

def build_dn(d1, d2):
    edges = set(d1) | set(d2)
    rows = []

    for e in edges:
        r1 = d1.get(e, 0)
        r2 = d2.get(e, 0)

        t = classify(r1, r2)
        if t:
            rows.append({
                "Node1": e[0],
                "Node2": e[1],
                "Delta": r1 - r2,
                "Abs_Delta": abs(r1 - r2),
                "Type": t
            })

    df = pd.DataFrame(rows)

    G = nx.Graph()
    for _, r in df.iterrows():
        G.add_edge(r["Node1"], r["Node2"], weight=r["Abs_Delta"])

    return G, df

def rewiring(df):
    scores = {}
    for _, r in df.iterrows():
        scores[r["Node1"]] = scores.get(r["Node1"], 0) + r["Abs_Delta"]
        scores[r["Node2"]] = scores.get(r["Node2"], 0) + r["Abs_Delta"]

    return pd.DataFrame([
        {"Node": k, "Score": v} for k, v in scores.items()
    ]).sort_values("Score", ascending=False)

def run(name, f1, f2):
    d1 = edge_dict(load_edges(f1))
    d2 = edge_dict(load_edges(f2))

    G, df = build_dn(d1, d2)
    rew = rewiring(df)

    out = os.path.join(OUTPUT_DIR, name)
    os.makedirs(out, exist_ok=True)

    nx.write_edgelist(G, os.path.join(out, "edges.txt"), data=["weight"])
    rew.to_csv(os.path.join(out, "rewiring_scores.txt"), sep="\t", index=False)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run("CONTROL_male_vs_female",
        NETWORK_FILES["control_male"],
        NETWORK_FILES["control_female"])

    run("IBD_male_vs_female",
        NETWORK_FILES["IBD_male"],
        NETWORK_FILES["IBD_female"])

if __name__ == "__main__":
    main()