"""Experiment: amplitude calibration of the transported residual.

The convex transport cannot produce a residual larger than the support
residuals, and the diagnostics show that 14% of query points exceed the support
amplitude envelope while carrying 24% of the absolute error.  This script
evaluates a single global gain and a distance-dependent gain:

    pred = y0 + s * g * r_tr
    s    = gain                                  (global)
    s    = 1 + slope * (d_min - median(d_min))   (distance-dependent)

The grid is evaluated on the outer test episodes to bound the achievable gain;
a deployable version must fit the gain on training DOIs.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.dcsa.baseline import PARAMS, _features, _load_points, _split_dois
from src.dcsa.crct import FeatureScaler, predict_episode
from src.dcsa.data_source import PROJECT_ROOT, input_paths, sha256_file
from src.dcsa.masking_validation import select_support

SHOT = 5
TAU = 0.25
NEIGHBOR_K = 2
GATE_TAU = 1.0
GAINS = (0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3)
SLOPES = (0.0, 0.1, 0.2, 0.3, 0.5)


def run(root: Path = PROJECT_ROOT, *, seeds=(11, 22, 33)) -> pd.DataFrame:
    points, valid = _load_points(root)
    feature_columns = sorted(column for column in points.columns if column.startswith("f_"))
    feature_table = points[["material", *feature_columns]].drop_duplicates("material")
    valid = valid.merge(feature_table, on="material", how="inner")
    all_dois = points.doi.unique()
    rows: list[dict[str, object]] = []

    for seed in seeds:
        train_dois, test_dois = _split_dois(all_dois, seed)
        train = points[points.doi.isin(train_dois)]
        test = points[points.doi.isin(test_dois)]
        prior = lgb.LGBMRegressor(**PARAMS, random_state=seed)
        prior.fit(_features(train, feature_columns), train.target.to_numpy(float))
        scaler = FeatureScaler.fit(train, feature_columns)
        eligible = valid[valid.doi.isin(test_dois)].groupby("doi").filter(
            lambda group: group.material.nunique() > SHOT)
        for doi, curves in eligible.groupby("doi", sort=True):
            support_meta, query_meta = select_support(curves, SHOT, "diverse_support", seed=seed)
            support = test[test.curve_row_id.isin(support_meta.curve_row_id)].copy()
            query = test[test.curve_row_id.isin(query_meta.curve_row_id)].copy()
            support_base = prior.predict(_features(support, feature_columns))
            query_base = prior.predict(_features(query, feature_columns))
            prediction = predict_episode(
                support, query, feature_columns, support_base=support_base,
                query_base=query_base, scaler=scaler, tau=TAU, blend=1.0,
                neighbor_k=NEIGHBOR_K, gate_tau=GATE_TAU)
            target = query.target.to_numpy(float)
            base_residual = prediction.chemistry_residual
            distance = prediction.nearest_distance
            row = {"seed": seed, "doi": doi, "median_distance": float(np.median(distance))}
            for gain in GAINS:
                row[f"mae_gain_{gain}"] = float(np.mean(np.abs(
                    target - (query_base + gain * base_residual))))
            for slope in SLOPES:
                scale = 1.0 + slope * (distance - np.median(distance))
                scale = np.clip(scale, 0.5, 2.0)
                row[f"mae_slope_{slope}"] = float(np.mean(np.abs(
                    target - (query_base + scale * base_residual))))
            rows.append(row)
        print(f"seed={seed}: episodes={len(rows)}", flush=True)

    detail = pd.DataFrame(rows)
    output = root / "results"
    detail.to_csv(output / "amplitude_calibration.csv", index=False)
    metrics = [c for c in detail.columns if c.startswith("mae_")]
    summary = detail[metrics].mean().sort_values().to_frame("mae_mean")
    baseline = summary.loc["mae_gain_1.0", "mae_mean"]
    summary["relative_to_gain_1_pct"] = 100.0 * (1.0 - summary.mae_mean / baseline)
    summary.to_csv(output / "amplitude_calibration_summary.csv")
    (output / "amplitude_calibration_manifest.json").write_text(json.dumps({
        "seeds": list(seeds), "shot": SHOT, "tau": TAU, "neighbor_k": NEIGHBOR_K,
        "gate_tau": GATE_TAU, "gains": list(GAINS), "slopes": list(SLOPES),
        "note": "grid on outer test; a deployable gain must be fitted on training DOIs",
        "input_sha256": {name: sha256_file(path) for name, path in input_paths().items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.round(6).to_string())
    return summary


if __name__ == "__main__":
    run()
