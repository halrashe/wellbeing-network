"""Build the published 48-node well-being network from GFS Wave 1 data.

Requires: Python 3, numpy, pandas, scikit-learn.
Run: python3 network_construction_minimal.py --input wellbeing-dataset.csv --output data
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import graphical_lasso

# EBIC-selected alpha reported in the accompanying manuscript.
# Fixed here to reproduce the selected network, not the EBIC search.
ALPHA = 0.002573

variable_groups = {
    "wellbeing": [
        "HAPPY", "LIFE_SAT", "MENTAL_HEALTH", "PHYSICAL_HLTH",
        "WORTHWHILE", "LIFE_PURPOSE", "PROMOTE_GOOD", "GIVE_UP",
        "CONTENT", "SAT_RELATNSHP", "WORRY_SAFETY", "EXPENSES",
    ],
    "childhood": [
        "MOTHER_RELATN", "FATHER_RELATN", "SVCS_12YRS",
        "HEALTH_GROWUP", "INCOME_12YRS",
    ],
    "health": [
        "CIGARETTES", "DRINKS", "DAYS_EXERCISE", "BODILY_PAIN",
        "DEPRESSED", "FEEL_ANXIOUS", "CONTROL_WORRY",
        "INTEREST", "THREAT_LIFE",
    ],
    "religion": [
        "ATTEND_SVCS", "PRAY_MEDITATE", "CONNECTED_REL",
        "SACRED_TEXTS", "SVCS_FATHER", "SVCS_MOTHER",
    ],
    "relationships": [
        "LONELY", "BELONGING", "PEOPLE_HELP",
        "GROUP_NOT_REL", "SHOW_LOVE",
    ],
    "character": ["HOPE_FUTURE", "GRATEFUL", "FORGIVE"],
    "happiness": [
        "FREEDOM", "CAPABLE", "WB_FIVEYRS", "WB_TODAY",
        "EXPECT_GOOD", "SUFFERING", "LIFE_BALANCE", "PEACE",
    ],
}

variable_labels = {
    "CONTENT": "WB01",
    "EXPENSES": "WB02",
    "GIVE_UP": "WB03",
    "HAPPY": "WB04",
    "LIFE_PURPOSE": "WB05",
    "LIFE_SAT": "WB06",
    "MENTAL_HEALTH": "WB07",
    "PHYSICAL_HLTH": "WB08",
    "PROMOTE_GOOD": "WB09",
    "SAT_RELATNSHP": "WB10",
    "WORRY_SAFETY": "WB11",
    "WORTHWHILE": "WB12",
    "BODILY_PAIN": "HL01",
    "CIGARETTES": "HL02",
    "CONTROL_WORRY": "HL03",
    "DAYS_EXERCISE": "HL04",
    "DEPRESSED": "HL05",
    "DRINKS": "HL06",
    "FEEL_ANXIOUS": "HL07",
    "INTEREST": "HL08",
    "THREAT_LIFE": "HL09",
    "ATTEND_SVCS": "RG01",
    "CONNECTED_REL": "RG02",
    "PRAY_MEDITATE": "RG03",
    "SACRED_TEXTS": "RG04",
    "SVCS_FATHER": "RG05",
    "SVCS_MOTHER": "RG06",
    "CAPABLE": "HP01",
    "EXPECT_GOOD": "HP02",
    "FREEDOM": "HP03",
    "LIFE_BALANCE": "HP04",
    "PEACE": "HP05",
    "SUFFERING": "HP06",
    "WB_FIVEYRS": "HP07",
    "WB_TODAY": "HP08",
    "FORGIVE": "CV01",
    "GRATEFUL": "CV02",
    "HOPE_FUTURE": "CV03",
    "BELONGING": "RS01",
    "GROUP_NOT_REL": "RS02",
    "LONELY": "RS03",
    "PEOPLE_HELP": "RS04",
    "SHOW_LOVE": "RS05",
    "FATHER_RELATN": "CH01",
    "HEALTH_GROWUP": "CH02",
    "INCOME_12YRS": "CH03",
    "MOTHER_RELATN": "CH04",
    "SVCS_12YRS": "CH05",
}

VARIABLE_SPECIAL_MISSING = {
    "MOTHER_RELATN": [97],
    "FATHER_RELATN": [97],
    "SVCS_FATHER": [97],
    "SVCS_MOTHER": [97],
}


def main():
    parser = argparse.ArgumentParser(description="Build the published well-being network")
    parser.add_argument("--input", required=True, type=Path, help="GFS Wave 1 survey CSV")
    parser.add_argument("--output", default=Path("data"), type=Path, help="Output folder")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    variables = [v for group in variable_groups.values() for v in group]
    raw = pd.read_csv(args.input, low_memory=False)
    data = raw[variables].apply(pd.to_numeric, errors="coerce")

    # Non-substantive responses are missing; special 97 codes apply to four items.
    data = data.replace([-98, 98, 99], np.nan)
    for variable, codes in VARIABLE_SPECIAL_MISSING.items():
        data[variable] = data[variable].replace(codes, np.nan)

    # Pairwise-complete Spearman correlations across the 48 attributes.
    correlation = data.corr(method="spearman")
    S = correlation.to_numpy()
    pairwise_n = data.notna().astype(np.int64).T @ data.notna().astype(np.int64)

    # Regularized precision matrix and associated signed partial correlations.
    _, precision = graphical_lasso(
        emp_cov=S, alpha=ALPHA, mode="cd",
        tol=1e-5, enet_tol=1e-5, max_iter=5000,
    )
    diagonal = np.sqrt(np.diag(precision))
    partial = -precision / np.outer(diagonal, diagonal)
    np.fill_diagonal(partial, 0.0)

    domains = {v: group for group, items in variable_groups.items() for v in items}
    nodes = pd.DataFrame([{
        "label": variable_labels[v], "variable": v, "domain": domains[v],
        "N_valid": int(data[v].notna().sum()),
        "missing_percent": round(float(data[v].isna().mean() * 100), 3),
    } for v in variables])

    edges = []
    for i, a in enumerate(variables):
        for j in range(i + 1, len(variables)):
            b = variables[j]
            weight = float(partial[i, j])
            if abs(weight) <= 1e-10:
                continue
            edges.append({
                "source": variable_labels[a], "target": variable_labels[b],
                "source_variable": a, "target_variable": b,
                "source_domain": domains[a], "target_domain": domains[b],
                "weight": weight, "abs_weight": abs(weight),
                "direction": "positive" if weight > 0 else "negative",
                "pairwise_N": int(pairwise_n.loc[a, b]),
            })

    edges = pd.DataFrame(edges).sort_values("abs_weight", ascending=False).reset_index(drop=True)
    edges.insert(0, "rank", np.arange(1, len(edges) + 1))
    nodes.to_csv(args.output / "nodes.csv", index=False)
    edges.to_csv(args.output / "edges.csv", index=False)
    print(f"Saved {len(nodes)} nodes and {len(edges)} edges to {args.output}")
    print(f"Fixed alpha: {ALPHA}")


if __name__ == "__main__":
    main()
