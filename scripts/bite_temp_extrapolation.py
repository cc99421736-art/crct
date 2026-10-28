#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temperature extrapolation stress test.

Support curves are truncated to the lower part of each paper's temperature
range; queries are the remaining compositions above that cutoff.  CRCT clamps
the residual at the support endpoint, while Ridge extrapolates linearly.
"""
from __future__ import annotations

import json
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
N_SUPPORT = 3
N_REPEATS = 20
N_BOOT = 5000
SEED = 20261023
PREFIX = "bite_temp_extrapolation"


def elements(value: str) -> frozenset[str]:
    import re
    return frozenset(re.findall(r"[A-Z][a-z]?", str(value)))


def main() -> None:
    points, _ = _load_points(PROJECT_ROOT)
    points = points[(points.temperature >= 200) & (points.temperature <= 900)].copy()
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
            temperatures = np.sort(curves.temperature.unique())
            if len(temperatures) < 6:
                continue
            cutoff = float(np.quantile(temperatures, 0.55))
            train = subset[~subset.doi.eq(doi)].copy()
            model = lgb.LGBMRegressor(**PARAMS, random_state=SEED + doi_index)
            model.fit(_features(train, feature_columns), train.target.to_numpy(float))
            scaler = FeatureScaler.fit(train, feature_columns)

            for repeat in range(N_REPEATS):
                support_comps = list(rng.choice(compositions, size=N_SUPPORT, replace=False))
                query_comps = [c for c in compositions if c not in support_comps]
                support = curves[curves.composition_normalized.isin(support_comps)
                                 & (curves.temperature <= cutoff)].copy()
                query = curves[curves.composition_normalized.isin(query_comps)
                               & (curves.temperature > cutoff)].copy()
                if query.empty or support.empty:
                    continue
                if support.groupby("composition_normalized").temperature.nunique().min() < 3:
                    continue

                support_base = model.predict(_features(support, feature_columns))
                query_base = model.predict(_features(query, feature_columns))
                prediction = predict_episode(
                    support, query, feature_columns, support_base=support_base,
                    query_base=query_base, scaler=scaler, tau=0.25, blend=1.0,
                    neighbor_k=2, gate_tau=1.0)

                frame = query[["composition_normalized", "target"]].copy()
                frame["crct"] = prediction.prediction
                frame["prior"] = query_base
                frame["offset"] = query_base + float(
                    np.mean(support.target.to_numpy(float) - support_base))

                ridge_x = support[feature_columns + ["temperature"]].to_numpy(float)
                ridge_scaler = StandardScaler().fit(ridge_x)
                ridge = Ridge(alpha=1.0).fit(
                    ridge_scaler.transform(ridge_x), support.target.to_numpy(float))
                query_x = query[feature_columns + ["temperature"]].to_numpy(float)
                frame["ridge"] = ridge.predict(ridge_scaler.transform(query_x))

                for method in ("crct", "ridge", "prior", "offset"):
                    error = np.abs(frame[method].to_numpy(float) - frame.target.to_numpy(float))
                    peak = frame.groupby("composition_normalized").agg(
                        pred=(method, "max"), true=("target", "max"))
                    rows.append({
                        "system": system_name, "doi": doi, "repeat": repeat,
                        "method": method, "cutoff": cutoff,
                        "point_mae": float(np.mean(error)),
                        "peak_mae": float(np.mean(np.abs(peak.pred - peak.true))),
                        "n_query_points": int(len(frame)),
                        "n_query_comps": int(len(peak)),
                    })

    detail = pd.DataFrame(rows)
    out = PROJECT_ROOT / "results"
    detail.to_csv(out / f"{PREFIX}_episode.csv", index=False)
    per_doi = detail.groupby(["doi", "method"])[["point_mae", "peak_mae"]].mean().reset_index()

    summary = []
    boot_rng = np.random.default_rng(SEED)
    for method, group in per_doi.groupby("method", sort=True):
        for metric in ("point_mae", "peak_mae"):
            values = group[metric].dropna().to_numpy(float)
            draws = values[boot_rng.integers(0, len(values), size=(N_BOOT, len(values)))]
            boot = draws.mean(axis=1)
            summary.append({"method": method, "metric": metric, "mean": float(values.mean()),
                            "n_dois": int(len(values)),
                            "ci95_low": float(np.quantile(boot, .025)),
                            "ci95_high": float(np.quantile(boot, .975))})
    summary = pd.DataFrame(summary)
    summary.to_csv(out / f"{PREFIX}_summary.csv", index=False)

    paired = []
    for baseline in ("ridge", "prior", "offset"):
        for metric in ("point_mae", "peak_mae"):
            a = per_doi[per_doi.method.eq("crct")].set_index("doi")[metric]
            b = per_doi[per_doi.method.eq(baseline)].set_index("doi")[metric]
            diff = (a - b).dropna().to_numpy(float)
            draws = diff[boot_rng.integers(0, len(diff), size=(N_BOOT, len(diff)))].mean(axis=1)
            paired.append({"baseline": baseline, "metric": metric,
                           "mean_diff": float(diff.mean()),
                           "ci95_low": float(np.quantile(draws, .025)),
                           "ci95_high": float(np.quantile(draws, .975)),
                           "wins": int((diff < 0).sum()),
                           "ties": int((diff == 0).sum()),
                           "losses": int((diff > 0).sum()),
                           "n_dois": int(len(diff))})
    paired = pd.DataFrame(paired)
    paired.to_csv(out / f"{PREFIX}_paired.csv", index=False)

    (out / f"{PREFIX}_manifest.json").write_text(json.dumps({
        "systems": list(SYSTEMS), "support_curves": N_SUPPORT, "repeats": N_REPEATS,
        "protocol": "support truncated at per-paper 55th percentile temperature; query above cutoff",
        "metrics": ["point_mae", "peak_mae"], "n_boot": N_BOOT,
        "interpretation": "temperature-extrapolation stress test, not prospective validation",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(summary.pivot(index="metric", columns="method", values="mean").round(4).to_string())
    print("\npaired CRCT minus baseline (negative = CRCT better)")
    print(paired.pivot(index="metric", columns="baseline", values="mean_diff").round(4).to_string())
    print(paired[paired.baseline.eq("ridge")].to_string(index=False))


if __name__ == "__main__":
    main()
