"""Construct the pooled within-country well-being network (Python 3.8+).


Requires: numpy, pandas, scikit-learn.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import graphical_lasso
from sklearn.exceptions import ConvergenceWarning

# The same 48 attributes and seven domains used in the manuscript.
GROUPS = {
    "wellbeing": ["HAPPY", "LIFE_SAT", "MENTAL_HEALTH", "PHYSICAL_HLTH",
                  "WORTHWHILE", "LIFE_PURPOSE", "PROMOTE_GOOD", "GIVE_UP",
                  "CONTENT", "SAT_RELATNSHP", "WORRY_SAFETY", "EXPENSES"],
    "childhood": ["MOTHER_RELATN", "FATHER_RELATN", "SVCS_12YRS",
                  "HEALTH_GROWUP", "INCOME_12YRS"],
    "health": ["CIGARETTES", "DRINKS", "DAYS_EXERCISE", "BODILY_PAIN",
               "DEPRESSED", "FEEL_ANXIOUS", "CONTROL_WORRY", "INTEREST", "THREAT_LIFE"],
    "religion": ["ATTEND_SVCS", "PRAY_MEDITATE", "CONNECTED_REL", "SACRED_TEXTS",
                 "SVCS_FATHER", "SVCS_MOTHER"],
    "relationships": ["LONELY", "BELONGING", "PEOPLE_HELP", "GROUP_NOT_REL", "SHOW_LOVE"],
    "character": ["HOPE_FUTURE", "GRATEFUL", "FORGIVE"],
    "happiness": ["FREEDOM", "CAPABLE", "WB_FIVEYRS", "WB_TODAY",
                  "EXPECT_GOOD", "SUFFERING", "LIFE_BALANCE", "PEACE"],
}

# Manuscript Table 1 codes, in domain order.
CODES = {
    "wellbeing": ["WB04", "WB06", "WB07", "WB08", "WB12", "WB05", "WB09", "WB03",
                  "WB01", "WB10", "WB11", "WB02"],
    "childhood": ["CH04", "CH01", "CH05", "CH02", "CH03"],
    "health": ["HL02", "HL06", "HL04", "HL01", "HL05", "HL07", "HL03", "HL08", "HL09"],
    "religion": ["RG01", "RG03", "RG02", "RG04", "RG05", "RG06"],
    "relationships": ["RS03", "RS01", "RS04", "RS02", "RS05"],
    "character": ["CV03", "CV02", "CV01"],
    "happiness": ["HP03", "HP01", "HP07", "HP08", "HP02", "HP06", "HP04", "HP05"],
}
VARIABLES = [name for group in GROUPS.values() for name in group]
DOMAINS = {name: group for group, names in GROUPS.items() for name in names}
LABELS = {name: code for group in GROUPS for name, code in zip(GROUPS[group], CODES[group])}
GAMMA = 0.5
EDGE_TOL = 1e-10


def ebic(corr, precision, n):
    p = corr.shape[0]
    sign, logdet = np.linalg.slogdet(precision)
    if sign <= 0:
        return np.inf
    edges = np.count_nonzero(np.triu(np.abs(precision), k=1) > EDGE_TOL)
    return (-n * (logdet - np.trace(corr @ precision))
            + edges * np.log(n) + 4 * GAMMA * edges * np.log(p))


def select_glasso(corr, n):
    """EBIC selection on the original log-spaced alpha grid, extending as needed."""
    upper = np.max(np.abs(corr - np.diag(np.diag(corr))))
    lower = upper * 0.01
    best_score, best_alpha, best_precision = np.inf, None, None
    for _ in range(4):
        viable = []
        stage_best_score, stage_best_alpha = np.inf, None
        for alpha in np.logspace(np.log10(upper), np.log10(lower), 100):
            # A non-converged candidate cannot be selected.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                try:
                    _, precision = graphical_lasso(
                        corr, float(alpha), mode="cd", tol=1e-5,
                        enet_tol=1e-5, max_iter=5000)
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    continue
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                continue
            if not np.isfinite(precision).all():
                continue
            score = ebic(corr, precision, n)
            if np.isfinite(score):
                viable.append(float(alpha))
                if score < stage_best_score:
                    stage_best_score, stage_best_alpha = score, float(alpha)
                if score < best_score:
                    best_score, best_alpha, best_precision = score, float(alpha), precision.copy()
        if viable and stage_best_alpha > min(viable):
            break
        upper, lower = lower, max(lower / 10, 1e-6)
        if upper <= lower:
            break
    if best_precision is None:
        raise RuntimeError("No converged graphical LASSO candidate found")
    return best_alpha, best_precision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="wellbeing-dataset.csv", help="Original GFS CSV")
    parser.add_argument("--output", default="data/pooled", help="Output directory")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, low_memory=False)
    data = df[VARIABLES].apply(pd.to_numeric, errors="coerce")
    data = data.replace([-98, 98, 99], np.nan)
    for name in ["MOTHER_RELATN", "FATHER_RELATN", "SVCS_FATHER", "SVCS_MOTHER"]:
        data[name] = data[name].replace(97, np.nan)
    country = df["COUNTRY"].replace([-98, 98, 99, ""], np.nan).astype("string").str.strip()
    mask = country.notna() & country.ne("").fillna(False)
    data, country = data.loc[mask], country.loc[mask]

    p = len(VARIABLES)
    weighted_z = np.zeros((p, p))
    total_weight = np.zeros((p, p))
    pooled_n = np.zeros((p, p), dtype=np.int64)
    for _, indices in country.groupby(country).groups.items():
        sample = data.loc[indices]
        pair_n = (sample.notna().astype(np.int64).T @
                  sample.notna().astype(np.int64)).to_numpy()
        corr = sample.corr(method="spearman", min_periods=4).to_numpy()
        eligible = np.isfinite(corr) & (pair_n >= 4)
        weight = np.where(eligible, np.maximum(pair_n - 3, 0), 0)
        z = np.zeros((p, p))
        z[eligible] = np.arctanh(np.clip(corr[eligible], -1 + 1e-12, 1 - 1e-12))
        weighted_z += weight * z
        total_weight += weight
        pooled_n += pair_n

    pooled = np.tanh(np.divide(weighted_z, total_weight,
                               out=np.zeros_like(weighted_z), where=total_weight > 0))
    pooled = (pooled + pooled.T) / 2
    np.fill_diagonal(pooled, 1)
    # Pairwise-deletion matrices can be slightly non-positive-definite.
    if np.linalg.eigvalsh(pooled).min() <= 1e-8:
        values, vectors = np.linalg.eigh(pooled)
        pooled = (vectors * np.maximum(values, 1e-6)) @ vectors.T
        scale = np.sqrt(np.diag(pooled))
        pooled /= np.outer(scale, scale)
        pooled = (pooled + pooled.T) / 2
        np.fill_diagonal(pooled, 1)

    n_ebic = int(pooled_n[np.triu_indices(p, 1)].min())
    alpha, precision = select_glasso(pooled, n_ebic)
    diagonal = np.diag(precision)
    partial = -precision / np.sqrt(np.outer(diagonal, diagonal))
    np.fill_diagonal(partial, 0)

    nodes = pd.DataFrame({
        "label": [LABELS[v] for v in VARIABLES],
        "variable": VARIABLES,
        "domain": [DOMAINS[v] for v in VARIABLES],
        "N_valid": data.notna().sum().reindex(VARIABLES).to_numpy(),
        "missing_percent": data.isna().mean().reindex(VARIABLES).to_numpy() * 100,
    })
    records = []
    for i in range(p):
        for j in range(i + 1, p):
            w = float(partial[i, j])
            if abs(w) <= EDGE_TOL:
                continue
            a, b = VARIABLES[i], VARIABLES[j]
            records.append({
                "source": LABELS[a], "target": LABELS[b],
                "source_variable": a, "target_variable": b,
                "source_domain": DOMAINS[a], "target_domain": DOMAINS[b],
                "weight": w, "abs_weight": abs(w),
                "direction": "positive" if w > 0 else "negative",
                "pairwise_N": int(pooled_n[i, j]),
            })
    edges = pd.DataFrame(records).sort_values("abs_weight", ascending=False).reset_index(drop=True)
    edges.insert(0, "rank", np.arange(1, len(edges) + 1))
    nodes.to_csv(output / "nodes.csv", index=False)
    edges.to_csv(output / "edges.csv", index=False)
    print(f"Pooled network: {len(nodes)} nodes, {len(edges)} edges; selected alpha={alpha:.8f}")
    print(f"Saved: {output / 'nodes.csv'} and {output / 'edges.csv'}")


if __name__ == "__main__":
    main()
