#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Expanded candidate-level confidence audit.

All eight Bi-Te/Ag-Te/Pb-Te/Sb-Sn/Mg-Si systems; two support curves per paper;
every remaining composition is a candidate.  For each candidate we compute the
four-feature risk (R_e from leave-one-support-out, hyperparameter perturbations
and a Ridge anchor) and test whether risk ranks candidate curve MAE.
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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dcsa.baseline import PARAMS, _features, _load_points
from src.dcsa.crct import FeatureScaler, predict_episode
from src.dcsa.data_source import PROJECT_ROOT

SYSTEMS = {
    "Ag-S-Te": {"Ag", "S", "Te"},
    "Ag-S-Se": {"Ag", "S", "Se"},
    "Cu-S-Se": {"Cu", "S", "Se"},
    "Bi-Sb-Te": {"Bi", "Sb", "Te"},
    "Bi-Se-Te": {"Bi", "Se", "Te"},
    "Pb-Se-Te": {"Pb", "Se", "Te"},
    "Sb-Sn-Te": {"Sb", "Sn", "Te"},
    "Mg-Si-Sn": {"Mg", "Si", "Sn"},
}
N_SUPPORT = 2
SEED = 20261023
PREFIX = "candidate_confidence_expanded"


def elements(value: str) -> frozenset[str]:
    return frozenset(re.findall(r"[A-Z][a-z]?", str(value)))


def peaks(prediction, query: pd.DataFrame) -> pd.Series:
    return (pd.DataFrame({"c": query.composition_normalized, "s": prediction.prediction})
            .groupby("c").s.max())


def main() -> None:
    points, _ = _load_points(PROJECT_ROOT)
    points = points[(points.temperature >= 200) & (points.temperature <= 800)].copy()
    rng = np.random.default_rng(SEED)
    frames: list[pd.DataFrame] = []

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
            if query.empty or support.empty:
                continue
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

            for left_out in support_comps:
                support_sub = support[~support.composition_normalized.eq(left_out)].copy()
                if support_sub.composition_normalized.nunique() < 1:
                    continue
                support_sub_base = model.predict(_features(support_sub, feature_columns))
                prediction = predict_episode(
                    support_sub, query, feature_columns, support_base=support_sub_base,
                    query_base=query_base, scaler=scaler, tau=0.25, blend=1.0,
                    neighbor_k=1, gate_tau=1.0)
                member_peaks.append(peaks(prediction, query))

            for tau, k, gate in ((0.5, 2, 1.0), (1.0, 2, 1.0), (0.25, 1, 1.0),
                                 (0.25, 3, 1.0), (0.25, 2, 0.5), (0.25, 2, None)):
                prediction = predict_episode(
                    support, query, feature_columns, support_base=support_base,
                    query_base=query_base, scaler=scaler, tau=tau, blend=1.0,
                    neighbor_k=k, gate_tau=gate)
                member_peaks.append(peaks(prediction, query))

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
            main_peak = peaks(base, query)
            curve_error = (pd.DataFrame({
                "c": query.composition_normalized,
                "e": np.abs(base.prediction - query.target.to_numpy(float))})
                .groupby("c").e.mean())
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
                "curve_mae": curve_error.reindex(r_e.index),
                "R_e": r_e,
                "R_T": outside.reindex(r_e.index),
                "R_d": (pd.DataFrame({"c": query.composition_normalized,
                                      "d": base.nearest_distance}).groupby("c").d.mean()
                        .reindex(r_e.index)),
                "R_g": 1.0 - (pd.DataFrame({"c": query.composition_normalized,
                                            "g": base.gate}).groupby("c").g.mean()
                              .reindex(r_e.index)),
                "prior_peak": (pd.DataFrame({"c": query.composition_normalized,
                                             "s": query_base}).groupby("c").s.max()
                               .reindex(r_e.index)),
                "ridge_peak": ridge_peaks.reindex(r_e.index),
                "support": ",".join(support_comps),
            })
            frame["abs_peak_error"] = (frame.predicted_peak - frame.true_peak).abs()
            frame["risk_score"] = 0.0279 * frame.R_e + 0.0077 * frame.R_T
            frames.append(frame)

    detail = pd.concat(frames, ignore_index=True)
    out = PROJECT_ROOT / "results"
    detail.to_csv(out / f"{PREFIX}_candidates.csv", index=False)

    detail["risk_q"] = pd.qcut(detail.risk_score, 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    quartiles = detail.groupby("risk_q", observed=True).agg(
        n=("curve_mae", "size"),
        curve_mae=("curve_mae", "mean"),
        median_curve_mae=("curve_mae", "median"),
        peak_error=("abs_peak_error", "mean"),
        mean_R_e=("R_e", "mean"),
    ).reset_index()
    quartiles.to_csv(out / f"{PREFIX}_quartiles.csv", index=False)

    rng_boot = np.random.default_rng(SEED)
    dois = detail.doi.unique()
    rhos, gaps = [], []
    for _ in range(5000):
        picked = rng_boot.choice(dois, len(dois), replace=True)
        sample = pd.concat([detail[detail.doi.eq(d)] for d in picked])
        rhos.append(spearmanr(sample.risk_score, sample.curve_mae).statistic)
        q = sample.groupby(pd.qcut(sample.risk_score, 4, labels=False, duplicates="drop")).curve_mae.mean()
        if 0 in q and 3 in q:
            gaps.append(float(q[3] - q[0]))

    high_error = (detail.curve_mae > detail.curve_mae.median()).astype(int)
    summary = {
        "n_candidates": int(len(detail)),
        "n_dois": int(detail.doi.nunique()),
        "n_systems": int(detail.system.nunique()),
        "spearman_risk_curve_mae": float(spearmanr(detail.risk_score, detail.curve_mae).statistic),
        "spearman_ci": [float(np.quantile(rhos, .025)), float(np.quantile(rhos, .975))],
        "q4_minus_q1_curve_mae": float(np.mean(gaps)),
        "q4_minus_q1_ci": [float(np.quantile(gaps, .025)), float(np.quantile(gaps, .975))],
        "auroc_high_curve_error": float(roc_auc_score(high_error, detail.risk_score)),
    }
    (out / f"{PREFIX}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(quartiles.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
