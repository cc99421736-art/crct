"""Weighted additive sensitivity: y0 + lambda_r * beta_r + lambda_c * g * r_tr.

This is the two-parameter additive family that contains ridge-only
(lambda_r=1, lambda_c=0), chemistry-only (0, 1), the raw additive model (1, 1)
and the convex blend at equal weights (0.5, 0.5).  It is evaluated at the fixed
5-shot diverse-support protocol so that only the residual weights change.
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
LAMBDA_R = (0.0, 0.25, 0.5, 1.0)
LAMBDA_C = (0.5, 1.0)


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
            target = query.target.to_numpy(dtype=float)
            anchor = prediction.ridge_residual
            transport = prediction.chemistry_residual
            for lambda_r in LAMBDA_R:
                for lambda_c in LAMBDA_C:
                    values = query_base + lambda_r * anchor + lambda_c * transport
                    rows.append({"seed": seed, "doi": doi,
                                 "lambda_r": lambda_r, "lambda_c": lambda_c,
                                 "mae": float(np.mean(np.abs(target - values)))})
        print(f"seed={seed}: episodes={len(rows)}", flush=True)

    detail = pd.DataFrame(rows)
    summary = (detail.groupby(["lambda_r", "lambda_c"], as_index=False)
               .agg(mae_mean=("mae", "mean"), mae_median=("mae", "median"),
                    n_episodes=("mae", "size")))
    output = root / "results"
    detail.to_csv(output / "crct_lambda_sensitivity.csv", index=False)
    summary.to_csv(output / "crct_lambda_sensitivity_summary.csv", index=False)
    (output / "crct_lambda_sensitivity_manifest.json").write_text(json.dumps({
        "seeds": list(seeds), "shot": SHOT, "support_selection": "diverse_support",
        "tau": TAU, "neighbor_k": NEIGHBOR_K, "gate_tau": GATE_TAU,
        "formula": "y0 + lambda_r * beta_r + lambda_c * g * r_tr",
        "lambda_r": list(LAMBDA_R), "lambda_c": list(LAMBDA_C),
        "input_sha256": {name: sha256_file(path) for name, path in input_paths().items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run()
