#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retrospective active learning for Bi-Sb-Te.

Each episode starts from three measured compositions of one paper.  At every
round a strategy selects one remaining composition, its measured peak ZT is
revealed, and its full curve is added to the support set.  The simulation is
retrospective: the hidden labels come from the same measured database.
"""
from __future__ import annotations

import json
import os
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

N_INITIAL = 3
BUDGET = 2
N_SEEDS = 60
SYSTEM = {"Bi", "Sb", "Te"}
PREFIX = os.environ.get("PREFIX", "bisbte_al")


def elements(value: str) -> frozenset[str]:
    return frozenset(re.findall(r"[A-Z][a-z]?", str(value)))


def load_system() -> tuple[pd.DataFrame, list[str], list[str]]:
    points, _ = _load_points(PROJECT_ROOT)
    points = points[points.composition_normalized.map(elements).eq(SYSTEM)].copy()
    points = points[(points.temperature >= 200) & (points.temperature <= 800)]
    counts = points.groupby("doi").composition_normalized.nunique()
    eligible = sorted(counts[counts >= N_INITIAL + 2].index)
    feature_columns = sorted(c for c in points.columns if c.startswith("f_"))
    return points, eligible, feature_columns


def run_episode(curves: pd.DataFrame, true_peak: pd.Series, initial: list[str],
                strategies: tuple[str, ...], model, scaler, feature_columns: list[str],
                rng: np.random.Generator) -> list[dict]:
    compositions = sorted(curves.composition_normalized.unique())
    remaining = [c for c in compositions if c not in initial]
    support_comps = list(initial)
    support = curves[curves.composition_normalized.isin(support_comps)].copy()
    query_all = curves[curves.composition_normalized.isin(remaining)].copy()
    query_base_all = model.predict(_features(query_all, feature_columns))
    best_curve = [float(true_peak.loc[support_comps].max())]
    selected_order: list[str] = []
    rows: list[dict] = []

    for strategy in strategies:
        # replay from the same initial state for every strategy
        support_comps_s = list(initial)
        support_s = curves[curves.composition_normalized.isin(support_comps_s)].copy()
        remaining_s = list(remaining)
        best = float(true_peak.loc[support_comps_s].max())
        curve = [best]
        order: list[str] = []
        for _ in range(min(BUDGET, len(remaining))):
            query = curves[curves.composition_normalized.isin(remaining_s)].copy()
            if strategy == "random":
                scores = pd.Series(rng.random(len(remaining_s)), index=remaining_s)
            elif strategy == "prior":
                scores = (pd.DataFrame({"c": query.composition_normalized,
                                        "s": model.predict(_features(query, feature_columns))})
                          .groupby("c").s.max())
            elif strategy == "nearest":
                sf = support_s.drop_duplicates("composition_normalized")[
                    ["composition_normalized", *feature_columns]].set_index("composition_normalized")
                qf = query.drop_duplicates("composition_normalized")[
                    ["composition_normalized", *feature_columns]].set_index("composition_normalized")
                dist = ((qf.to_numpy(float)[:, None, :] - sf.to_numpy(float)[None, :, :]) ** 2).sum(-1)
                scores = pd.Series(-dist.min(axis=1), index=qf.index)
            elif strategy in ("crct", "crct_gate"):
                support_base = model.predict(_features(support_s, feature_columns))
                query_base = model.predict(_features(query, feature_columns))
                prediction = predict_episode(
                    support_s, query, feature_columns, support_base=support_base,
                    query_base=query_base, scaler=scaler, tau=0.25, blend=1.0,
                    neighbor_k=2, gate_tau=1.0)
                frame = pd.DataFrame({"c": query.composition_normalized,
                                      "s": prediction.prediction, "g": prediction.gate})
                grouped = frame.groupby("c").agg(s=("s", "max"), g=("g", "mean"))
                scores = grouped.s if strategy == "crct" else grouped.s * grouped.g
            elif strategy == "ridge":
                x = support_s[feature_columns + ["temperature"]].to_numpy(float)
                scaler_r = StandardScaler().fit(x)
                ridge = Ridge(alpha=1.0).fit(
                    scaler_r.transform(x), support_s.target.to_numpy(float))
                qx = query[feature_columns + ["temperature"]].to_numpy(float)
                scores = (pd.DataFrame({"c": query.composition_normalized,
                                        "s": ridge.predict(scaler_r.transform(qx))})
                          .groupby("c").s.max())
            else:
                raise ValueError(strategy)

            scores = scores.reindex(remaining_s)
            pick = scores.idxmax()
            remaining_s.remove(pick)
            order.append(pick)
            best = max(best, float(true_peak.loc[pick]))
            curve.append(best)
            support_s = curves[curves.composition_normalized.isin(support_comps_s + [pick])].copy()
            support_comps_s.append(pick)

        final_best = float(curve[-1])
        true_max = float(true_peak.max())
        target = 0.9 * true_max
        hit = [i for i, value in enumerate(curve) if value >= target]
        measurements_to_90 = float(hit[0] + N_INITIAL) if hit else np.nan
        x = np.linspace(0, 1, len(curve))
        rows.append({
            "strategy": strategy,
            "n_initial": N_INITIAL,
            "n_pool": len(remaining),
            "final_best": final_best,
            "final_regret": true_max - final_best,
            "normalized_auc": float(np.trapz(np.asarray(curve) / true_max, x)),
            "hit_at_budget": float(final_best >= true_max - 1e-12),
            "measurements_to_90": measurements_to_90,
        })
    return rows


def main() -> None:
    points, eligible, feature_columns = load_system()
    rng = np.random.default_rng(20261023)
    strategies = ("crct", "crct_gate", "ridge", "prior", "nearest", "random")
    rows: list[dict] = []
    for doi_index, doi in enumerate(eligible):
        curves = points[points.doi.eq(doi)].copy()
        compositions = sorted(curves.composition_normalized.unique())
        true_peak = curves.groupby("composition_normalized").target.max()
        train = points[~points.doi.eq(doi)].copy()
        model = lgb.LGBMRegressor(**PARAMS, random_state=20261023 + doi_index)
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
            episode = run_episode(curves, true_peak, initial, strategies, model, scaler,
                                  feature_columns, rng)
            for row in episode:
                row.update({"doi": doi, "seed": seed})
                rows.append(row)

    detail = pd.DataFrame(rows)
    out = PROJECT_ROOT / "results"
    detail.to_csv(out / f"{PREFIX}_episode.csv", index=False)

    metrics = ["final_best", "final_regret", "normalized_auc", "hit_at_budget", "measurements_to_90"]
    per_doi = detail.groupby(["doi", "strategy"])[metrics].mean().reset_index()
    boot_rng = np.random.default_rng(20261023)
    summary = []
    for strategy, group in per_doi.groupby("strategy", sort=True):
        for metric in metrics:
            values = group[metric].dropna().to_numpy(float)
            if not len(values):
                continue
            draws = values[boot_rng.integers(0, len(values), size=(5000, len(values)))]
            boot = draws.mean(axis=1)
            summary.append({"strategy": strategy, "metric": metric,
                            "mean": float(values.mean()), "n_dois": int(len(values)),
                            "ci95_low": float(np.quantile(boot, .025)),
                            "ci95_high": float(np.quantile(boot, .975))})
    summary = pd.DataFrame(summary)
    summary.to_csv(out / f"{PREFIX}_summary.csv", index=False)

    paired = []
    for baseline in ("ridge", "prior", "nearest", "random"):
        for metric in ("final_best", "final_regret", "normalized_auc", "hit_at_budget", "measurements_to_90"):
            a = per_doi[per_doi.strategy.eq("crct")].set_index("doi")[metric]
            b = per_doi[per_doi.strategy.eq(baseline)].set_index("doi")[metric]
            diff = (a - b).dropna().to_numpy(float)
            if not len(diff):
                continue
            draws = diff[boot_rng.integers(0, len(diff), size=(5000, len(diff)))].mean(axis=1)
            paired.append({"baseline": baseline, "metric": metric,
                           "mean_diff": float(diff.mean()),
                           "ci95_low": float(np.quantile(draws, .025)),
                           "ci95_high": float(np.quantile(draws, .975)),
                           "n_dois": int(len(diff))})
    paired = pd.DataFrame(paired)
    paired.to_csv(out / f"{PREFIX}_paired.csv", index=False)

    (out / f"{PREFIX}_manifest.json").write_text(json.dumps({
        "system": "Bi-Sb-Te", "n_initial": N_INITIAL, "n_seeds": N_SEEDS,
        "eligible_dois": eligible, "n_eligible_dois": len(eligible),
        "strategies": list(strategies),
        "protocol": "retrospective AL; initial 3 measured compositions, budget 2 additional measurements, reveal measured peak and add full curve",
        "interpretation": "retrospective simulation, not prospective experimental validation",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("eligible", len(eligible), "episodes", len(detail) // len(strategies))
    print(summary.pivot(index="metric", columns="strategy", values="mean").round(3).to_string())
    print("\npaired CRCT minus baseline")
    print(paired.pivot(index="metric", columns="baseline", values="mean_diff").round(3).to_string())


if __name__ == "__main__":
    main()
