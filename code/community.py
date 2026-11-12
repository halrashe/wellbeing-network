"""Signed community detection: Louvain, Leiden, and Spinglass.

Reads data/edges.csv and produces community assignments and run summaries.
"""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import igraph as ig
import louvain
import leidenalg
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

BASE = Path(__file__).resolve().parent
EDGE_FILE = BASE / "data" / "edges.csv"
OUTPUT = BASE / "community_results"
N_RUNS = 1000
SPINS = 25

variable_groups = {'wellbeing': ['HAPPY',
               'LIFE_SAT',
               'MENTAL_HEALTH',
               'PHYSICAL_HLTH',
               'WORTHWHILE',
               'LIFE_PURPOSE',
               'PROMOTE_GOOD',
               'GIVE_UP',
               'CONTENT',
               'SAT_RELATNSHP',
               'WORRY_SAFETY',
               'EXPENSES'],
 'childhood': ['MOTHER_RELATN', 'FATHER_RELATN', 'SVCS_12YRS', 'HEALTH_GROWUP', 'INCOME_12YRS'],
 'health': ['CIGARETTES',
            'DRINKS',
            'DAYS_EXERCISE',
            'BODILY_PAIN',
            'DEPRESSED',
            'FEEL_ANXIOUS',
            'CONTROL_WORRY',
            'INTEREST',
            'THREAT_LIFE'],
 'religion': ['ATTEND_SVCS', 'PRAY_MEDITATE', 'CONNECTED_REL', 'SACRED_TEXTS', 'SVCS_FATHER', 'SVCS_MOTHER'],
 'relationships': ['LONELY', 'BELONGING', 'PEOPLE_HELP', 'GROUP_NOT_REL', 'SHOW_LOVE'],
 'character': ['HOPE_FUTURE', 'GRATEFUL', 'FORGIVE'],
 'happiness': ['FREEDOM',
               'CAPABLE',
               'WB_FIVEYRS',
               'WB_TODAY',
               'EXPECT_GOOD',
               'SUFFERING',
               'LIFE_BALANCE',
               'PEACE']}
variables = [v for group in variable_groups.values() for v in group]
domains = {v: d for d, group in variable_groups.items() for v in group}
index = {v: i for i, v in enumerate(variables)}
domain_ids = {d: i for i, d in enumerate(variable_groups)}
true_labels = [domain_ids[domains[v]] for v in variables]

edges = pd.read_csv(EDGE_FILE)
edges = edges[edges.source_variable.isin(index) & edges.target_variable.isin(index) & (edges.weight != 0)]


def graph(rows, absolute=False):
    g = ig.Graph(n=len(variables), directed=False)
    g.vs["name"] = variables
    g.add_edges([(index[a], index[b]) for a, b in zip(rows.source_variable, rows.target_variable)])
    g.es["weight"] = [abs(float(w)) if absolute else float(w) for w in rows.weight]
    return g


positive = graph(edges[edges.weight > 0])
negative = graph(edges[edges.weight < 0], absolute=True)
signed = graph(edges)


def run_algorithm(method, seed):
    if method in ("louvain", "leiden"):
        package = louvain if method == "louvain" else leidenalg
        p = package.ModularityVertexPartition(positive, weights="weight")
        n = package.ModularityVertexPartition(negative, weights="weight")
        optimizer = package.Optimiser()
        optimizer.set_rng_seed(seed)
        optimizer.optimise_partition_multiplex([p, n], layer_weights=[1, -1])
        return np.asarray(p.membership, dtype=int)
    random.seed(seed)
    ig.set_random_number_generator(random)
    return np.asarray(signed.community_spinglass(
        weights="weight", spins=SPINS, implementation="neg",
        gamma=1.0, lambda_=1.0).membership, dtype=int)


def analyze(method):
    memberships = np.asarray([run_algorithm(method, seed) for seed in range(N_RUNS)])
    ari = np.asarray([adjusted_rand_score(true_labels, row) for row in memberships])
    nmi = np.asarray([normalized_mutual_info_score(true_labels, row) for row in memberships])
    counts = np.asarray([len(np.unique(row)) for row in memberships])

    # Representative partition: maximum mean ARI against all other runs.
    sums = np.zeros(N_RUNS)
    pairwise_scores = []
    for i in range(N_RUNS):
        for j in range(i + 1, N_RUNS):
            score = adjusted_rand_score(memberships[i], memberships[j])
            sums[i] += score
            sums[j] += score
            pairwise_scores.append(score)
    representative = int(np.argmax(sums / (N_RUNS - 1)))
    chosen = memberships[representative]
    relabel = {old: new + 1 for new, old in enumerate(sorted(set(chosen)))}
    chosen = np.asarray([relabel[c] for c in chosen])

    # Frequency with which each node co-clusters with its representative community.
    coassignment = np.zeros((len(variables), len(variables)))
    for membership in memberships:
        coassignment += membership[:, None] == membership[None, :]
    coassignment /= N_RUNS
    stability = []
    for i in range(len(variables)):
        peers = np.where((chosen == chosen[i]) & (np.arange(len(variables)) != i))[0]
        stability.append(float(coassignment[i, peers].mean()) if len(peers) else np.nan)

    assignments = pd.DataFrame({
        "algorithm": method, "variable": variables,
        "predefined_domain": [domains[v] for v in variables],
        "detected_community": chosen, "community_stability": stability
    })
    runs = pd.DataFrame({"algorithm": method, "seed": range(N_RUNS),
                         "n_communities": counts, "ARI_domains": ari, "NMI_domains": nmi})
    summary = {"algorithm": method, "representative_seed": representative,
               "representative_communities": len(set(chosen)),
               "representative_ARI_domains": ari[representative],
               "representative_NMI_domains": nmi[representative],
               "mean_pairwise_ARI": float(np.mean(pairwise_scores)),
               "sd_pairwise_ARI": float(np.std(pairwise_scores, ddof=1)),
               "mean_ARI_domains": float(np.mean(ari)),
               "mean_NMI_domains": float(np.mean(nmi))}
    print(method, "representative communities:", len(set(chosen)),
          "mean pairwise ARI:", round(summary["mean_pairwise_ARI"], 4))
    return assignments, runs, summary


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = [analyze(method) for method in ("louvain", "leiden", "spinglass")]
    pd.concat([r[0] for r in results], ignore_index=True).to_csv(
        OUTPUT / "community_assignments.csv", index=False)
    pd.concat([r[1] for r in results], ignore_index=True).to_csv(
        OUTPUT / "community_runs.csv", index=False)
    pd.DataFrame([r[2] for r in results]).to_csv(
        OUTPUT / "community_summary.csv", index=False)
    print("Saved results in", OUTPUT)
