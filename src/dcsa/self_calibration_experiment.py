"""Experiment: leave-one-support-out self-calibration.

The transport is a biased estimator within a paper (the diagnostics show a mean
signed bias of about -0.015 and a mean absolute per-curve bias of 0.066).  The
five measured support curves allow an internal estimate of that bias: predict
each support curve from the other four and average the signed error.  If the
bias is shared inside the paper, subtracting it from the query prediction should
help.

No query labels are used.
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
DAMPING = (0.0, 0.25, 0.5, 0.75, 1.0)


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
            base_pred = prediction.variants["chemistry_only"]
            target = query.target.to_numpy(float)
            # internal bias estimate from leave-one-support-out predictions
            held_out_errors = []
            for curve_id in support_meta.curve_row_id:
                inner_meta = support_meta[support_meta.curve_row_id != curve_id]
                held = support[support.curve_row_id.eq(curve_id)]
                if held.empty or len(inner_meta) < 2:
                    continue
                inner_support = support[support.curve_row_id.isin(inner_meta.curve_row_id)]
                inner_base = prior.predict(_features(inner_support, feature_columns))
                held_base = prior.predict(_features(held, feature_columns))
                loso = predict_episode(
                    inner_support, held, feature_columns, support_base=inner_base,
                    query_base=held_base, scaler=scaler, tau=TAU, blend=1.0,
                    neighbor_k=min(NEIGHBOR_K, len(inner_meta)), gate_tau=GATE_TAU)
                held_out_errors.append(float(np.mean(
                    loso.variants["chemistry_only"] - held.target.to_numpy(float))))
            bias_estimate = float(np.mean(held_out_errors)) if held_out_errors else 0.0
            query_true_bias = float(np.mean(base_pred - target))
            row = {"seed": seed, "doi": doi, "bias_estimate": bias_estimate,
                   "query_true_bias": query_true_bias,
                   "mae_base": float(np.mean(np.abs(base_pred - target)))}
            for damping in DAMPING:
                corrected = base_pred - damping * bias_estimate
                row[f"mae_damp_{damping}"] = float(np.mean(np.abs(corrected - target)))
            rows.append(row)
        print(f"seed={seed}: episodes={len(rows)}", flush=True)

    detail = pd.DataFrame(rows)
    output = root / "results"
    detail.to_csv(output / "self_calibration.csv", index=False)
    metrics = [c for c in detail.columns if c.startswith("mae_")]
    summary = detail[metrics].mean().sort_values().to_frame("mae_mean")
    summary["relative_to_base_pct"] = 100.0 * (
        1.0 - summary.mae_mean / summary.loc["mae_base", "mae_mean"])
    correlation = float(detail.bias_estimate.corr(detail.query_true_bias))
    summary.loc["bias_query_correlation", "mae_mean"] = correlation
    summary.to_csv(output / "self_calibration_summary.csv")
    (output / "self_calibration_manifest.json").write_text(json.dumps({
        "seeds": list(seeds), "shot": SHOT, "tau": TAU, "neighbor_k": NEIGHBOR_K,
        "gate_tau": GATE_TAU, "damping": list(DAMPING),
        "bias_estimate": "mean signed LOSO error over the five support curves",
        "bias_query_correlation": correlation,
        "input_sha256": {name: sha256_file(path) for name, path in input_paths().items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.round(6).to_string())
    return summary


if __name__ == "__main__":
    run()
