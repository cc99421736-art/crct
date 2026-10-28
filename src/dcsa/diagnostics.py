"""Diagnostics for CRCT: within-paper ceiling and error decomposition.

Three questions:

1. How well can the method predict a *measured* support curve of the same paper
   from the other four support curves?  This leave-one-support-out (LOSO) error
   is an optimistic ceiling for within-paper transfer.
2. Where does the query error come from: additive bias, residual amplitude, or
   residual shape?
3. How much could a per-curve amplitude/offset calibration recover?  The oracle
   versions fit the scale and offset on the *true* query curve, so they are
   upper bounds for any calibration module.
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


def _curve_rows(seed: str, doi: str, query: pd.DataFrame, target: np.ndarray,
                base: np.ndarray, prediction: np.ndarray, residual: np.ndarray,
                distance: np.ndarray, outside: np.ndarray) -> list[dict[str, object]]:
    rows = []
    frame = query.assign(_target=target, _base=base, _pred=prediction,
                         _res=residual, _dist=distance, _out=outside)
    for curve_id, group in frame.groupby("curve_row_id", sort=True):
        y = group._target.to_numpy(float)
        pred = group._pred.to_numpy(float)
        res_true = y - group._base.to_numpy(float)
        res_pred = group._res.to_numpy(float)
        error = pred - y
        mae = float(np.mean(np.abs(error)))
        # Oracle corrections fitted on the true query curve.
        denom = float(np.dot(res_pred, res_pred))
        scale = float(np.dot(res_true, res_pred) / denom) if denom > 1e-12 else 1.0
        offset = float(np.mean(res_true - res_pred))
        affine = np.polyfit(res_pred, res_true, 1) if len(res_pred) > 1 else np.array([1.0, 0.0])
        mae_scale = float(np.mean(np.abs(group._base.to_numpy(float) + scale * res_pred - y)))
        mae_offset = float(np.mean(np.abs(pred + offset - y)))
        mae_affine = float(np.mean(np.abs(group._base.to_numpy(float)
                                        + np.polyval(affine, res_pred) - y)))
        corr = float(np.corrcoef(res_true, res_pred)[0, 1]) if len(res_true) > 1 else np.nan
        rows.append({
            "seed": seed, "doi": doi, "curve_row_id": curve_id,
            "n_points": int(len(group)), "mae": mae,
            "bias": float(np.mean(error)),
            "true_res_mean": float(np.mean(res_true)), "true_res_std": float(np.std(res_true)),
            "true_res_range": float(np.max(res_true) - np.min(res_true)),
            "pred_res_mean": float(np.mean(res_pred)), "pred_res_std": float(np.std(res_pred)),
            "pred_res_range": float(np.max(res_pred) - np.min(res_pred)),
            "amplitude_slope": scale,
            "shape_corr": corr,
            "mae_oracle_scale": mae_scale,
            "mae_oracle_offset": mae_offset,
            "mae_oracle_affine": mae_affine,
            "mean_distance": float(np.mean(group._dist)),
            "fraction_outside": float(np.mean(group._out)),
        })
    return rows


def run(root: Path = PROJECT_ROOT, *, seeds=(11, 22, 33)) -> pd.DataFrame:
    points, valid = _load_points(root)
    feature_columns = sorted(column for column in points.columns if column.startswith("f_"))
    feature_table = points[["material", *feature_columns]].drop_duplicates("material")
    valid = valid.merge(feature_table, on="material", how="inner")
    all_dois = points.doi.unique()
    episode_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []

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
            base_all = prior.predict(_features(pd.concat([support, query]), feature_columns))
            support_base = base_all[:len(support)]
            query_base = base_all[len(support):]
            query_temperature = query.temperature.to_numpy(float)
            ranges = support.groupby("curve_row_id").temperature.agg(["min", "max"]).to_numpy(float)
            covered = np.zeros(len(query), dtype=bool)
            for lower, upper in ranges:
                covered |= (query_temperature >= lower) & (query_temperature <= upper)
            outside_point = 1.0 - covered.astype(float)
            prediction = predict_episode(
                support, query, feature_columns, support_base=support_base,
                query_base=query_base, scaler=scaler, tau=TAU, blend=1.0,
                neighbor_k=NEIGHBOR_K, gate_tau=GATE_TAU)
            target = query.target.to_numpy(float)
            residual = prediction.variants["chemistry_only"] - query_base
            curve_rows.extend(_curve_rows(str(seed), doi, query, target, query_base,
                                          prediction.variants["chemistry_only"], residual,
                                          prediction.nearest_distance,
                                          outside_point))
            # Leave-one-support-out: predict each measured support curve from the
            # other four support curves of the same paper.
            loso = []
            support_curve_ids = list(support_meta.curve_row_id)
            for held_out in support_curve_ids:
                inner_support_meta = support_meta[support_meta.curve_row_id != held_out]
                held_rows = support[support.curve_row_id.eq(held_out)].copy()
                if held_rows.empty or len(inner_support_meta) < 2:
                    continue
                inner_support = support[support.curve_row_id.isin(inner_support_meta.curve_row_id)].copy()
                inner_support_base = prior.predict(_features(inner_support, feature_columns))
                held_base = prior.predict(_features(held_rows, feature_columns))
                pred = predict_episode(
                    inner_support, held_rows, feature_columns,
                    support_base=inner_support_base, query_base=held_base,
                    scaler=scaler, tau=TAU, blend=1.0,
                    neighbor_k=min(NEIGHBOR_K, len(inner_support_meta)),
                    gate_tau=GATE_TAU)
                loso.append(float(np.mean(np.abs(
                    pred.variants["chemistry_only"] - held_rows.target.to_numpy(float)))))
            episode_rows.append({
                "seed": seed, "doi": doi, "n_support": int(support.curve_row_id.nunique()),
                "n_query_curves": int(query.curve_row_id.nunique()),
                "n_query_points": int(len(query)),
                "mae_chem": float(np.mean(np.abs(target - prediction.variants["chemistry_only"]))),
                "loso_mae": float(np.mean(loso)) if loso else np.nan,
                "mean_distance": float(np.mean(prediction.nearest_distance)),
                "fraction_outside": float(np.mean(outside_point)),
            })
        print(f"seed={seed}: episodes={len(episode_rows)}", flush=True)

    episodes = pd.DataFrame(episode_rows)
    curves = pd.DataFrame(curve_rows)
    output = root / "results"
    episodes.to_csv(output / "diagnostics_episodes.csv", index=False)
    curves.to_csv(output / "diagnostics_curves.csv", index=False)
    summary = {
        "n_episodes": int(len(episodes)),
        "query_mae": float(episodes.mae_chem.mean()),
        "loso_mae": float(episodes.loso_mae.mean()),
        "loso_query_gap": float(episodes.loso_mae.mean() - episodes.mae_chem.mean()),
        "curve_mae": float(curves.mae.mean()),
        "oracle_scale_mae": float(curves.mae_oracle_scale.mean()),
        "oracle_offset_mae": float(curves.mae_oracle_offset.mean()),
        "oracle_affine_mae": float(curves.mae_oracle_affine.mean()),
        "mean_amplitude_slope": float(curves.amplitude_slope.mean()),
        "median_amplitude_slope": float(curves.amplitude_slope.median()),
        "mean_shape_corr": float(curves.shape_corr.mean()),
        "mean_bias": float(curves.bias.mean()),
        "mean_abs_bias": float(curves.bias.abs().mean()),
        "input_sha256": {name: sha256_file(path) for name, path in input_paths().items()},
    }
    (output / "diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return episodes


if __name__ == "__main__":
    run()
