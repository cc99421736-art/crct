"""Error budget for CRCT: where does the absolute error actually live?

For every query point the script records:

  outside    the temperature lies outside every support curve's range
             (the current code clamps to the endpoint residual there)
  exceed     |true residual| is larger than the largest support |residual| at
             that temperature, i.e. the transport cannot reach the amplitude
  peak       the point is within 10% of the query curve's temperature span of
             the true peak temperature

It then reports each category's share of points and of the total absolute
error, plus the MAE that would result if that category behaved like the rest.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.dcsa.baseline import PARAMS, _features, _load_points, _split_dois
from src.dcsa.crct import FeatureScaler, _curve_residuals, predict_episode
from src.dcsa.data_source import PROJECT_ROOT, input_paths, sha256_file
from src.dcsa.masking_validation import select_support

SHOT = 5
TAU = 0.25
NEIGHBOR_K = 2
GATE_TAU = 1.0


def run(root: Path = PROJECT_ROOT, *, seeds=(11, 22, 33)) -> pd.DataFrame:
    points, valid = _load_points(root)
    feature_columns = sorted(column for column in points.columns if column.startswith("f_"))
    feature_table = points[["material", *feature_columns]].drop_duplicates("material")
    valid = valid.merge(feature_table, on="material", how="inner")
    all_dois = points.doi.unique()
    records = []

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
            pred = prediction.variants["chemistry_only"]
            target = query.target.to_numpy(float)
            error = np.abs(pred - target)
            temperature = query.temperature.to_numpy(float)
            true_residual = target - query_base
            _, s_temps, s_res = _curve_residuals(support, support_base, feature_columns)
            ranges = support.groupby("curve_row_id").temperature.agg(["min", "max"]).to_numpy(float)
            covered = np.zeros(len(query), dtype=bool)
            for lower, upper in ranges:
                covered |= (temperature >= lower) & (temperature <= upper)
            envelope = np.column_stack([
                np.interp(temperature, t, np.abs(r)) for t, r in zip(s_temps, s_res)]).max(axis=1)
            exceed = np.abs(true_residual) > envelope
            # distance to the true peak temperature of each query curve
            curve_ids = query.curve_row_id.to_numpy()
            peak_distance = np.zeros(len(query))
            for curve_id in pd.unique(curve_ids):
                mask = curve_ids == curve_id
                t_values = temperature[mask]
                r_values = true_residual[mask]
                peak_t = float(t_values[int(np.argmax(r_values))])
                span = max(float(t_values.max() - t_values.min()), 1.0)
                peak_distance[mask] = np.abs(t_values - peak_t) / span
            near_peak = peak_distance <= 0.10
            records.append(pd.DataFrame({
                "seed": seed, "doi": doi, "error": error, "outside": ~covered,
                "exceed": exceed, "near_peak": near_peak,
            }))
        print(f"seed={seed}: episodes={len(records)}", flush=True)

    frame = pd.concat(records, ignore_index=True)
    total = float(frame.error.sum())
    rows = []
    for name, mask in (("outside", frame.outside),
                       ("inside", ~frame.outside),
                       ("exceed", frame.exceed),
                       ("not_exceed", ~frame.exceed),
                       ("near_peak", frame.near_peak),
                       ("tail", ~frame.near_peak)):
        subset = frame[mask]
        rows.append({"category": name, "n_points": int(mask.sum()),
                     "share_points": float(mask.mean()),
                     "mae": float(subset.error.mean()) if len(subset) else float("nan"),
                     "share_abs_error": float(subset.error.sum() / total) if total else float("nan")})
    table = pd.DataFrame(rows)
    inside_mae = float(frame.loc[~frame.outside, "error"].mean())
    counterfactual = (frame.error.sum() - frame.loc[frame.outside, "error"].sum()
                      + inside_mae * int(frame.outside.sum())) / len(frame)
    summary = {
        "n_points": int(len(frame)),
        "mae": float(frame.error.mean()),
        "counterfactual_outside_as_inside": float(counterfactual),
        "mean_abs_residual": float(frame.error.mean()),
        "input_sha256": {name: sha256_file(path) for name, path in input_paths().items()},
    }
    output = root / "results"
    table.to_csv(output / "error_budget.csv", index=False)
    (output / "error_budget_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(table.round(4).to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return table


if __name__ == "__main__":
    run()
