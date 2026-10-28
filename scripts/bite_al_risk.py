#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Risk-aware retrospective AL.

Fixed protocol: two systems, 2 initial support curves, budget of 2 additional
measurements.  The risk-aware strategy uses the CRCT residual gate, chemical
distance and temperature coverage to discount high-risk candidates.  Ridge has
no equivalent risk score and is ranked by predicted peak.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dcsa.baseline import PARAMS, _features, _load_points
from src.dcsa.crct import FeatureScaler, predict_episode
from src.dcsa.data_source import PROJECT_ROOT

SYSTEMS = {"Bi-Sb-Te": {"Bi", "Sb", "Te"}, "Bi-Se-Te": {"Bi", "Se", "Te"}}
N_INITIAL = 2
BUDGET = 2
N_SEEDS = 40
SEED = 20261023
PREFIX = "bite_al_risk"


def elements(value: str) -> frozenset[str]:
    return frozenset(re.findall(r"[A-Z][a-z]?", str(value)))


def zscore(values: np.ndarray) -> np.ndarray:
    scale = float(np.std(values, ddof=0))
    return (values - float(np.mean(values))) / (scale if scale > 0 else 1.0)


def candidate_table(curves: pd.DataFrame, support: pd.DataFrame, query: pd.DataFrame,
                    prediction, feature_columns: list[str], model, ridge, ridge_scaler,
                    remaining: list[str]) -> pd.DataFrame:
    frame = query[["composition_normalized", "temperature"]].copy()
    frame["crct"] = prediction.prediction
    frame["gate"] = prediction.gate
    frame["distance"] = prediction.nearest_distance
    support_min = float(support.temperature.min())
    support_max = float(support.temperature.max())
    frame["outside"] = ((frame.temperature < support_min) | (frame.temperature > support_max)).astype(float)
    grouped = frame.groupby("composition_normalized").agg(
        crct=("crct", "max"), gate=("gate", "mean"), distance=("distance", "mean"),
        outside=("outside", "mean"))
    ridge_x = query[feature_columns + ["temperature"]].to_numpy(float)
    ridge_pred = ridge.predict(ridge_scaler.transform(ridge_x))
    ridge_series = (pd.DataFrame({"c": query.composition_normalized, "s": ridge_pred})
                    .groupby("c").s.max())
    grouped["ridge"] = ridge_series.reindex(grouped.index)
    grouped["risk"] = (zscore(grouped.distance.to_numpy(float))
                       + zscore(grouped.outside.to_numpy(float))
                       + zscore(1.0 - grouped.gate.to_numpy(float)))
    grouped["risk_rel"] = grouped.risk - grouped.risk.min()
    grouped["crct_risk"] = grouped.crct / (1.0 + grouped.risk_rel)
    return grouped.reindex(remaining)


def run(curves: pd.DataFrame, true_peak: pd.Series, compositions: list[str], initial: list[str],
        model, scaler, feature_columns: list[str], rng: np.random.Generator) -> list[dict]:
    rows = []
    remaining = [c for c in compositions if c not in initial]
    for strategy in ("crct", "crct_risk", "ridge", "random"):
        support_comps = list(initial)
        remaining_s = list(remaining)
        support = curves[curves.composition_normalized.isin(support_comps)].copy()
        best = float(true_peak.loc[support_comps].max())
        selected: list[str] = []
        for _ in range(min(BUDGET, len(remaining_s))):
            query = curves[curves.composition_normalized.isin(remaining_s)].copy()
            support_base = model.predict(_features(support, feature_columns))
            query_base = model.predict(_features(query, feature_columns))
            prediction = predict_episode(
                support, query, feature_columns, support_base=support_base,
                query_base=query_base, scaler=scaler, tau=0.25, blend=1.0,
                neighbor_k=2, gate_tau=1.0)
            ridge_x = support[feature_columns + ["temperature"]].to_numpy(float)
            ridge_scaler = StandardScaler().fit(ridge_x)
            ridge = Ridge(alpha=1.0).fit(
                ridge_scaler.transform(ridge_x), support.target.to_numpy(float))
            table = candidate_table(curves, support, query, prediction, feature_columns,
                                    model, ridge, ridge_scaler, remaining_s)
            if strategy == "random":
                pick = str(rng.choice(remaining_s))
            elif strategy == "ridge":
                pick = str(table.ridge.idxmax())
            elif strategy == "crct":
                pick = str(table.crct.idxmax())
            else:
                pick = str(table.crct_risk.idxmax())
            pred_col = {"crct": "crct", "crct_risk": "crct_risk",
                        "ridge": "ridge", "random": "crct"}[strategy]
            pred = float(table.loc[pick, pred_col])
            true = float(true_peak.loc[pick])
            selected.append(pick)
            remaining_s.remove(pick)
            best = max(best, true)
            support_comps.append(pick)
            support = curves[curves.composition_normalized.isin(support_comps)].copy()
            rows.append({
                "strategy": strategy, "step": len(selected), "selected": pick,
                "predicted_peak": pred, "true_peak": true,
                "abs_error": abs(pred - true),
                "true_percentile": float(true_peak.rank(pct=True).loc[pick]),
                "best_so_far": best,
            })
    return rows


def main() -> None:
    points, _ = _load_points(PROJECT_ROOT)
    points = points[(points.temperature >= 200) & (points.temperature <= 800)].copy()
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    for system_name, elements_set in SYSTEMS.items():
        subset = points[points.composition_normalized.map(elements).eq(elements_set)].copy()
        counts = subset.groupby("doi").composition_normalized.nunique()
        eligible = sorted(counts[counts >= N_INITIAL + 2].index)
        feature_columns = sorted(c for c in subset.columns if c.startswith("f_"))
        print(system_name, "papers", len(eligible))
        for doi_index, doi in enumerate(eligible):
            curves = subset[subset.doi.eq(doi)].copy()
            compositions = sorted(curves.composition_normalized.unique())
            true_peak = curves.groupby("composition_normalized").target.max()
            train = subset[~subset.doi.eq(doi)].copy()
            model = lgb.LGBMRegressor(**PARAMS, random_state=SEED + doi_index)
            model.fit(_features(train, feature_columns), train.target.to_numpy(float))
            scaler = FeatureScaler.fit(train, feature_columns)
            for seed in range(N_SEEDS):
                initial = None
                for _ in range(200):
                    candidate = list(rng.choice(compositions, size=N_INITIAL, replace=False))
                    if float(true_peak.loc[candidate].max()) < float(true_peak.max()) - 1e-12:
                        initial = candidate
                        break
                if initial is None:
                    continue
                for row in run(curves, true_peak, compositions, initial, model, scaler,
                               feature_columns, rng):
                    row.update({"system": system_name, "doi": doi, "seed": seed})
                    rows.append(row)

    detail = pd.DataFrame(rows)
    out = PROJECT_ROOT / "results"
    detail.to_csv(out / f"{PREFIX}_episode.csv", index=False)

    metrics = ["abs_error", "true_percentile", "true_peak", "best_so_far"]
    per_doi = detail.groupby(["doi", "strategy"])[metrics].mean().reset_index()
    boot_rng = np.random.default_rng(SEED)
    summary = []
    for strategy, group in per_doi.groupby("strategy", sort=True):
        for metric in metrics:
            values = group[metric].dropna().to_numpy(float)
            draws = values[boot_rng.integers(0, len(values), size=(5000, len(values)))]
            boot = draws.mean(axis=1)
            summary.append({"strategy": strategy, "metric": metric, "mean": float(values.mean()),
                            "n_dois": int(len(values)),
                            "ci95_low": float(np.quantile(boot, .025)),
                            "ci95_high": float(np.quantile(boot, .975))})
    summary = pd.DataFrame(summary)
    summary.to_csv(out / f"{PREFIX}_summary.csv", index=False)

    paired = []
    for left, right in (("crct_risk", "crct"), ("crct_risk", "ridge"), ("crct", "ridge")):
        for metric in metrics:
            a = per_doi[per_doi.strategy.eq(left)].set_index("doi")[metric]
            b = per_doi[per_doi.strategy.eq(right)].set_index("doi")[metric]
            diff = (a - b).dropna().to_numpy(float)
            draws = diff[boot_rng.integers(0, len(diff), size=(5000, len(diff)))].mean(axis=1)
            paired.append({"left": left, "right": right, "metric": metric,
                           "mean_diff": float(diff.mean()),
                           "ci95_low": float(np.quantile(draws, .025)),
                           "ci95_high": float(np.quantile(draws, .975)),
                           "wins": int((diff < 0).sum()) if metric in ("abs_error",) else int((diff > 0).sum()),
                           "losses": int((diff > 0).sum()) if metric in ("abs_error",) else int((diff < 0).sum()),
                           "n_dois": int(len(diff))})
    paired = pd.DataFrame(paired)
    paired.to_csv(out / f"{PREFIX}_paired.csv", index=False)

    (out / f"{PREFIX}_manifest.json").write_text(json.dumps({
        "systems": list(SYSTEMS), "n_initial": N_INITIAL, "budget": BUDGET,
        "n_seeds": N_SEEDS, "strategies": ["crct", "crct_risk", "ridge", "random"],
        "risk": "z(chemical distance)+z(temperature outside support)+z(1-gate)",
        "acquisition": "predicted peak / (1 + relative risk)",
        "interpretation": "retrospective AL; risk-aware acquisition is CRCT-specific",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("episodes", len(detail) // 4)
    print(summary.pivot(index="metric", columns="strategy", values="mean").round(3).to_string())
    print("\npaired comparisons")
    print(paired.pivot(index="metric", columns=["left", "right"], values="mean_diff").round(3).to_string())


if __name__ == "__main__":
    main()
