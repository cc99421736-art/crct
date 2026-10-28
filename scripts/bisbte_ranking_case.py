#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bi-Sb-Te within-system case study: measure 3, rank the rest.

For every Bi-Sb-Te paper with at least five measured compositions, we hide all
but three compositions, predict the remaining curves, and rank the hidden
compositions by predicted peak ZT.  We compare CRCT against four baselines:
global prior, prior + paper offset, Ridge fitted on the support curves, nearest
support composition, and random ordering.
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
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dcsa.baseline import PARAMS, _features, _load_points
from src.dcsa.crct import FeatureScaler, predict_episode
from src.dcsa.data_source import PROJECT_ROOT

N_SUPPORT = int(os.environ.get("N_SUPPORT", "3"))
PREFIX = os.environ.get("PREFIX", "bisbte_ranking")
N_REPEATS = 20
N_BOOT = 5000
SEED = 20261023
SYSTEM = {"Bi", "Sb", "Te"}


def elements(value: str) -> frozenset[str]:
    return frozenset(re.findall(r"[A-Z][a-z]?", str(value)))


def ndcg(true: np.ndarray, order: np.ndarray, k: int) -> float:
    k = min(k, len(true))
    gain = np.maximum(true[order[:k]], 0.0)
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gain * discount))
    ideal = np.sort(np.maximum(true, 0.0))[::-1][:k]
    denom = float(np.sum(ideal * discount))
    return dcg / denom if denom else 1.0


def score_order(true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    order = np.argsort(-score, kind="mergesort")
    best = float(np.nanmax(true))
    top1 = float(true[order[0]])
    out = {
        "top1_hit": float(top1 == best),
        "top1_regret": float(best - top1),
        "ndcg1": ndcg(true, order, 1),
        "spearman": float(spearmanr(score, true).statistic)
        if len(true) > 1 else np.nan,
    }
    for k in (3, 5):
        if k <= len(true):
            chosen = order[:k]
            out[f"top{k}_hit"] = float(np.nanmax(true[chosen]) == best)
            out[f"top{k}_regret"] = float(best - np.nanmax(true[chosen]))
            out[f"ndcg{k}"] = ndcg(true, order, k)
        else:
            for name in (f"top{k}_hit", f"top{k}_regret", f"ndcg{k}"):
                out[name] = np.nan
    return out


def build_episodes() -> tuple[pd.DataFrame, list[str]]:
    points, _ = _load_points(PROJECT_ROOT)
    points = points[points.composition_normalized.map(elements).eq(SYSTEM)].copy()
    points = points[(points.temperature >= 200) & (points.temperature <= 800)]
    counts = points.groupby("doi").composition_normalized.nunique()
    eligible = sorted(counts[counts >= N_SUPPORT + 2].index)
    feature_columns = sorted(c for c in points.columns if c.startswith("f_"))
    rows: list[dict] = []
    rng = np.random.default_rng(SEED)

    for doi_index, doi in enumerate(eligible):
        curves = points[points.doi.eq(doi)].copy()
        compositions = sorted(curves.composition_normalized.unique())
        true_peak = curves.groupby("composition_normalized").target.max()
        train = points[~points.doi.eq(doi)].copy()
        model = lgb.LGBMRegressor(**PARAMS, random_state=SEED + doi_index)
        model.fit(_features(train, feature_columns), train.target.to_numpy(float))
        scaler = FeatureScaler.fit(train, feature_columns)

        for repeat in range(N_REPEATS):
            support_comps = list(rng.choice(compositions, size=N_SUPPORT, replace=False))
            query_comps = [c for c in compositions if c not in support_comps]
            if len(query_comps) < 2:
                continue
            support = curves[curves.composition_normalized.isin(support_comps)].copy()
            query = curves[curves.composition_normalized.isin(query_comps)].copy()
            support_base = model.predict(_features(support, feature_columns))
            query_base = model.predict(_features(query, feature_columns))
            prediction = predict_episode(
                support, query, feature_columns, support_base=support_base,
                query_base=query_base, scaler=scaler, tau=0.25, blend=1.0,
                neighbor_k=2, gate_tau=1.0)

            query_frame = query[["composition_normalized"]].copy()
            query_frame["crct"] = prediction.prediction
            query_frame["prior"] = query_base
            offset = float(np.mean(support.target.to_numpy(float) - support_base))
            query_frame["offset"] = query_base + offset

            ridge_x = support[feature_columns + ["temperature"]].to_numpy(float)
            ridge_scaler = StandardScaler().fit(ridge_x)
            ridge = Ridge(alpha=1.0).fit(
                ridge_scaler.transform(ridge_x), support.target.to_numpy(float))
            query_x = query[feature_columns + ["temperature"]].to_numpy(float)
            query_frame["ridge"] = ridge.predict(ridge_scaler.transform(query_x))

            support_features = support.drop_duplicates("composition_normalized")[
                ["composition_normalized", *feature_columns]].set_index("composition_normalized")
            query_features = query.drop_duplicates("composition_normalized")[
                ["composition_normalized", *feature_columns]].set_index("composition_normalized")
            distances = ((query_features.to_numpy(float)[:, None, :]
                          - support_features.to_numpy(float)[None, :, :]) ** 2).sum(-1)
            nearest = support_features.index.to_numpy()[distances.argmin(axis=1)]
            nearest_peak = {
                comp: float(true_peak.loc[nearest[i]])
                for i, comp in enumerate(query_features.index)}
            query_frame["nearest"] = query_frame.composition_normalized.map(nearest_peak)
            query_frame["random"] = rng.random(len(query_frame))

            per_comp = query_frame.groupby("composition_normalized").mean(numeric_only=True)
            truth = true_peak.loc[per_comp.index].to_numpy(float)
            for method in ("crct", "prior", "offset", "ridge", "nearest", "random"):
                row = score_order(truth, per_comp[method].to_numpy(float))
                row.update({"doi": doi, "repeat": repeat, "n_query": len(per_comp),
                            "method": method})
                rows.append(row)

    return pd.DataFrame(rows), eligible


def summarize(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["top1_hit", "top1_regret", "ndcg1", "top3_hit", "top3_regret",
               "ndcg3", "spearman"]
    per_doi = detail.groupby(["doi", "method"])[metrics].mean().reset_index()
    rng = np.random.default_rng(SEED)
    rows = []
    for method, group in per_doi.groupby("method", sort=True):
        for metric in metrics:
            values = group[metric].dropna().to_numpy(float)
            if not len(values):
                continue
            draws = values[rng.integers(0, len(values), size=(N_BOOT, len(values)))]
            boot = draws.mean(axis=1)
            rows.append({"method": method, "metric": metric, "mean": float(values.mean()),
                         "n_dois": int(len(values)),
                         "ci95_low": float(np.quantile(boot, .025)),
                         "ci95_high": float(np.quantile(boot, .975))})
    summary = pd.DataFrame(rows)

    paired = []
    for baseline in ("prior", "offset", "ridge", "nearest", "random"):
        for metric in ("top1_hit", "top3_hit", "ndcg3", "top1_regret"):
            a = per_doi[per_doi.method.eq("crct")].set_index("doi")[metric]
            b = per_doi[per_doi.method.eq(baseline)].set_index("doi")[metric]
            diff = (a - b).dropna().to_numpy(float)
            if not len(diff):
                continue
            draws = diff[rng.integers(0, len(diff), size=(N_BOOT, len(diff)))].mean(axis=1)
            paired.append({"baseline": baseline, "metric": metric, "mean_diff": float(diff.mean()),
                           "ci95_low": float(np.quantile(draws, .025)),
                           "ci95_high": float(np.quantile(draws, .975)),
                           "n_dois": int(len(diff))})
    return summary, pd.DataFrame(paired)


def main() -> None:
    detail, eligible = build_episodes()
    out = PROJECT_ROOT / "results"
    detail.to_csv(out / f"{PREFIX}_episode.csv", index=False)
    summary, paired = summarize(detail)
    summary.to_csv(out / f"{PREFIX}_summary.csv", index=False)
    paired.to_csv(out / f"{PREFIX}_paired.csv", index=False)
    (out / f"{PREFIX}_manifest.json").write_text(json.dumps({
        "system": "Bi-Sb-Te", "support_curves": N_SUPPORT, "repeats": N_REPEATS,
        "eligible_dois": eligible, "n_eligible_dois": len(eligible),
        "n_boot": N_BOOT, "seed": SEED,
        "protocol": "leave-one-DOI-out prior; 3 support compositions; rank remaining measured compositions by predicted peak ZT",
        "interpretation": "retrospective masked ranking; not prospective experimental validation",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("eligible DOIs", len(eligible), "episodes", len(detail) // 6)
    print(summary.pivot(index="metric", columns="method", values="mean").round(3).to_string())
    print("\npaired CRCT minus baseline")
    print(paired.pivot(index="metric", columns="baseline", values="mean_diff").round(3).to_string())


if __name__ == "__main__":
    main()
