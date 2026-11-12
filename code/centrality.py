"""Compute the eight centrality measures reported in the manuscript.

Inputs: data/nodes.csv, data/edges.csv
Output: centrality_metrics.csv
"""
from pathlib import Path
import pandas as pd
import networkx as nx

BASE = Path(__file__).resolve().parent
nodes = pd.read_csv(BASE / "data" / "nodes.csv")
edges = pd.read_csv(BASE / "data" / "edges.csv")

variables = nodes["variable"].tolist()
domains = dict(zip(nodes["variable"], nodes["domain"]))
metric_names = [
    "strength", "expected_influence",
    "wellbeing_strength", "wellbeing_expected_influence",
    "bridge_strength", "bridge_expected_influence",
    "bridge_betweenness", "bridge_closeness",
]
metrics = {name: dict.fromkeys(variables, 0.0) for name in metric_names}

# Direct (signed) network metrics; every undirected edge contributes to both endpoints.
G = nx.Graph()
G.add_nodes_from(variables)
for edge in edges.itertuples(index=False):
    u, v, w = edge.source_variable, edge.target_variable, float(edge.weight)
    for node, other in ((u, v), (v, u)):
        metrics["strength"][node] += abs(w)
        metrics["expected_influence"][node] += w
        if domains[other] == "wellbeing":
            metrics["wellbeing_strength"][node] += abs(w)
            metrics["wellbeing_expected_influence"][node] += w
        if domains[node] != domains[other]:
            metrics["bridge_strength"][node] += abs(w)
            metrics["bridge_expected_influence"][node] += w
    if w > 0:
        G.add_edge(u, v, distance=1.0 / w)

# Path-based bridge metrics use positive edges only, with distance = 1/weight.
distances = dict(nx.all_pairs_dijkstra_path_length(G, weight="distance"))
for u in variables:
    total = sum(d for v, d in distances[u].items() if domains[u] != domains[v])
    metrics["bridge_closeness"][u] = 1.0 / total if total > 0 else 0.0

for i, u in enumerate(variables):
    for v in variables[i + 1:]:
        if domains[u] == domains[v] or not nx.has_path(G, u, v):
            continue
        paths = list(nx.all_shortest_paths(G, u, v, weight="distance"))
        for path in paths:
            for node in path[1:-1]:
                metrics["bridge_betweenness"][node] += 1.0 / len(paths)

results = nodes[["variable", "domain"]].copy()
for name in metric_names:
    results[name] = results["variable"].map(metrics[name])
    sd = results[name].std(ddof=0)
    results[name + "_z"] = (results[name] - results[name].mean()) / sd if sd else 0.0

results.to_csv(BASE / "centrality_metrics.csv", index=False)
print("Saved centrality_metrics.csv (48 attributes; 8 metrics and Z-scores)")
