#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Candidate-level four-feature risk audit.

For each candidate composition in Bi-Sb-Te / Bi-Se-Te papers, compute the
ensemble peak disagreement R_e (leave-one-support-out, hyperparameter
perturbations, Ridge anchor), plus R_d, R_T, R_g, and test whether the frozen
four-feature risk score ranks candidate-level prediction error.
"""
from __future__ import annotations

import json
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

SYSTEMS = {"Bi-Sb-Te": {"Bi", "Sb", "Te"}, "Bi-Se-Te": {"Bi", "Se", "Te"}}
N_SUPPORT = 3
SEED = 20261023
PREFIX = "candidate_confidence_4f"


def elements(value: str) -> frozenset[str]:
    return frozenset(re.findall(r"[A-Z][a-z]?", str(value)))


def peaks(prediction, query: pd.DataFrame) -> pd.Series:
    return (pd.DataFrame({"c": query.composition_normalized, "s": prediction.prediction})
            .groupby("c").s.max())


def main() -> None:
    points, _ = _load_points(PROJECT_ROOT)
    points = points[(points.temperature >= 200) & (points.temperature <= 800)].copy()
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []

    for system_name, elements_set in SYSTEMS.items():
        subset = points[points.composition_normalized.map(elements).eq(elements_set)].copy()
        counts = subset.groupby("doi").composition_normalized.nunique()
        eligible = sorted(counts[counts >= N_SUPPORT + 2].index)
        feature_columns = sorted(c for c in subset.columns if c.startswith("f_"))
        print(system_name, "papers", len(eligible))

        for doi_index, doi in enumerate(eligible):
            curves = subset[subset.doi.eq(doi)].copy()
            compositions = sorted(curves.composition_normalized.unique())
            support_comps = list(rng.choice(compositions, size=N_SUPPORT, replace=False))
            query_comps = [c for c in compositions if c not in support_comps]
            support = curves[curves.composition_normalized.isin(support_comps)].copy()
            query = curves[curves.composition_normalized.isin(query_comps)].copy()
            true_peak = curves.groupby("composition_normalized").target.max()

            train = subset[~subset.doi.eq(doi)].copy()
            model = lgb.LGBMRegressor(**PARAMS, random_state=SEED + doi_index)
            model.fit(_features(train, feature_columns), train.target.to_numpy(float))
            scaler = FeatureScaler.fit(train, feature_columns)
            support_base = model.predict(_features(support, feature_columns))
            query_base = model.predict(_features(query, feature_columns))

            member_peaks: list[pd.Series] = []
            base = predict_episode(
                support, query, feature_columns, support_base=support_base,
                query_base=query_base, scaler=scaler, tau=0.25, blend=1.0,
                neighbor_k=2, gate_tau=1.0)
            member_peaks.append(peaks(base, query))
            main = base

            # leave-one-support-out
            for left_out in support_comps:
                support_sub = support[~support.composition_normalized.eq(left_out)].copy()
                if support_sub.composition_normalized.nunique() < 2:
                    continue
                support_sub_base = model.predict(_features(support_sub, feature_columns))
                prediction = predict_episode(
                    support_sub, query, feature_columns, support_base=support_sub_base,
                    query_base=query_base, scaler=scaler, tau=0.25, blend=1.0,
                    neighbor_k=2, gate_tau=1.0)
                member_peaks.append(peaks(prediction, query))

            # hyperparameter perturbations
            for tau, k, gate in ((0.5, 2, 1.0), (1.0, 2, 1.0), (0.25, 1, 1.0),
                                 (0.25, 3, 1.0), (0.25, 2, 0.5), (0.25, 2, None)):
                prediction = predict_episode(
                    support, query, feature_columns, support_base=support_base,
                    query_base=query_base, scaler=scaler, tau=tau, blend=1.0,
                    neighbor_k=k, gate_tau=gate)
                member_peaks.append(peaks(prediction, query))

            # Ridge anchor
            ridge_x = support[feature_columns + ["temperature"]].to_numpy(float)
            ridge_scaler = StandardScaler().fit(ridge_x)
            ridge = Ridge(alpha=1.0).fit(
                ridge_scaler.transform(ridge_x), support.target.to_numpy(float))
            query_x = query[feature_columns + ["temperature"]].to_numpy(float)
            ridge_peaks = (pd.DataFrame({"c": query.composition_normalized,
                                         "s": ridge.predict(ridge_scaler.transform(query_x))})
                           .groupby("c").s.max())
            member_peaks.append(ridge_peaks)

            ensemble = pd.concat(member_peaks, axis=1)
            r_e = ensemble.std(axis=1, ddof=0)
            main_peak = peaks(main, query)
            support_min = float(support.temperature.min())
            support_max = float(support.temperature.max())
            outside = (query.assign(outside=((query.temperature < support_min)
                                             | (query.temperature > support_max)).astype(float))
                       .groupby("composition_normalized").outside.mean())
            frame = pd.DataFrame({
                "system": system_name, "doi": doi,
                "composition": r_e.index,
                "predicted_peak": main_peak.reindex(r_e.index),
                "true_peak": true_peak.reindex(r_e.index),
                "R_e": r_e,
                "R_T": outside.reindex(r_e.index),
                "R_d": (pd.DataFrame({"c": query.composition_normalized,
                                      "d": main.nearest_distance}).groupby("c").d.mean()
                        .reindex(r_e.index)),
                "R_g": 1.0 - (pd.DataFrame({"c": query.composition_normalized,
                                            "g": main.gate}).groupby("c").g.mean()
                              .reindex(r_e.index)),
            })
            frame["abs_error"] = (frame.predicted_peak - frame.true_peak).abs()
            frame["risk_score"] = 0.0279 * frame.R_e + 0.0077 * frame.R_T
            rows.append(frame)

    detail = pd.concat(rows, ignore_index=True)
    out = PROJECT_ROOT / "results"
    detail.to_csv(out / f"{PREFIX}_candidates.csv", index=False)

    rho = spearmanr(detail.risk_score, detail.abs_error).statistic
    detail["risk_q"] = pd.qcut(detail.risk_score, 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    quartiles = detail.groupby("risk_q", observed=True).agg(
        n=("abs_error", "size"), mean_abs_error=("abs_error", "mean"),
        median_abs_error=("abs_error", "median"), mean_R_e=("R_e", "mean")).reset_index()
    quartiles.to_csv(out / f"{PREFIX}_quartiles.csv", index=False)

    summary = {
        "n_candidates": int(len(detail)),
        "n_dois": int(detail.doi.nunique()),
        "spearman_risk_abs_error": float(rho),
        "pearson_R_e_abs_error": float(np.corrcoef(detail.R_e, detail.abs_error)[0, 1]),
    }
    (out / f"{PREFIX}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(quartiles.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
