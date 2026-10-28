"""Experiment: local-linear residual transport versus local-constant transport.

The current CRCT transport is a zeroth-order estimator: the query residual is a
kernel-weighted *average* of support residuals, so it can only interpolate
between support residual levels and cannot extrapolate an offset or an
amplitude.  A first-order (local-linear) fit in a 1D composition coordinate can
extrapolate both.

Coordinates tested:
  ``axis``  signed projection on the query direction, query at +||z_q-centroid||
  ``pca``   first principal coordinate of the support+query compositions
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
RIDGE = 1e-8
ALPHAS = (0.25, 0.5, 0.75, 1.0)


def _weighted_linear(query_z: np.ndarray, support_z: np.ndarray,
                     weights: np.ndarray, residual: np.ndarray,
                     mode: str) -> np.ndarray:
    """Local-linear prediction of the residual for every query point."""
    centroid = support_z.mean(axis=0)
    n_query = len(query_z)
    out = np.zeros(n_query)
    for i in range(n_query):
        if mode == "axis":
            direction = query_z[i] - centroid
            norm = float(np.linalg.norm(direction))
            if norm < 1e-12:
                out[i] = np.dot(weights[i], residual[i])
                continue
            unit = direction / norm
            coord_support = (support_z - centroid) @ unit
            coord_query = norm
        elif mode == "pca":
            stack = np.vstack([support_z, query_z[i]])
            centered = stack - stack.mean(axis=0)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            unit = vt[0]
            if np.dot(unit, query_z[i] - centroid) < 0:
                unit = -unit
            coord_support = (support_z - centroid) @ unit
            coord_query = (query_z[i] - centroid) @ unit
        else:
            raise ValueError(mode)
        w = weights[i]
        if w.sum() <= 0:
            out[i] = np.dot(weights[i], residual[i])
            continue
        center = float(np.dot(w, coord_support) / w.sum())
        x = np.column_stack([np.ones(len(coord_support)), coord_support - center])
        xtw = x.T * w
        a = xtw @ x + RIDGE * np.eye(2)
        b = xtw @ residual[i]
        try:
            beta = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            beta = np.array([float(np.dot(w, residual[i]) / w.sum()), 0.0])
        out[i] = beta[0] + beta[1] * (coord_query - center)
    return out


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
            target = query.target.to_numpy(float)

            baseline = predict_episode(
                support, query, feature_columns, support_base=support_base,
                query_base=query_base, scaler=scaler, tau=TAU, blend=1.0,
                neighbor_k=NEIGHBOR_K, gate_tau=GATE_TAU)
            full = predict_episode(
                support, query, feature_columns, support_base=support_base,
                query_base=query_base, scaler=scaler, tau=TAU, blend=1.0,
                neighbor_k=None, gate_tau=GATE_TAU)
            compositions, s_temps, s_res = _curve_residuals(support, support_base, feature_columns)
            q_temp = query.temperature.to_numpy(float)
            residual_matrix = np.column_stack([
                np.interp(q_temp, t, r) for t, r in zip(s_temps, s_res)])
            support_z = scaler.transform(np.stack(compositions))
            query_z = scaler.transform(query[feature_columns].to_numpy(dtype=float))
            weights = full.weights  # all-support kernel weights
            gate = full.gate
            chem_gated = full.chemistry_residual  # already multiplied by the gate

            row = {"seed": seed, "doi": doi, "n_query_points": int(len(query)),
                   "mae_chem": float(np.mean(np.abs(target - baseline.variants["chemistry_only"])))}
            for mode in ("axis", "pca"):
                ll_residual = _weighted_linear(query_z, support_z, weights,
                                               residual_matrix, mode)
                ll_gated = gate * ll_residual
                for alpha in ALPHAS:
                    blended = (1.0 - alpha) * chem_gated + alpha * ll_gated
                    pred = query_base + blended
                    row[f"mae_ll_{mode}_a{alpha}"] = float(np.mean(np.abs(target - pred)))
                # offset-only: shift the transported curve to the local-linear mean
                shift = float(np.mean(ll_gated - chem_gated))
                row[f"mae_offset_{mode}"] = float(np.mean(np.abs(
                    target - (query_base + chem_gated + shift))))
                # amplitude-only: scale the transported curve by the local-linear amplitude
                ll_amp = float(np.std(ll_gated))
                cc_amp = float(np.std(chem_gated))
                scale = ll_amp / cc_amp if cc_amp > 1e-12 else 1.0
                scale = float(np.clip(scale, 0.5, 2.0))
                row[f"mae_amplitude_{mode}"] = float(np.mean(np.abs(
                    target - (query_base + scale * chem_gated))))
            rows.append(row)
        print(f"seed={seed}: episodes={len(rows)}", flush=True)

    detail = pd.DataFrame(rows)
    output = root / "results"
    detail.to_csv(output / "local_linear_experiment.csv", index=False)
    metrics = [c for c in detail.columns if c.startswith("mae_")]
    summary = detail[metrics].mean().sort_values().to_frame("mae_mean")
    summary["relative_to_chem_pct"] = 100.0 * (1.0 - summary.mae_mean / summary.loc["mae_chem", "mae_mean"])
    summary.to_csv(output / "local_linear_summary.csv")
    (output / "local_linear_manifest.json").write_text(json.dumps({
        "seeds": list(seeds), "shot": SHOT, "tau": TAU, "gate_tau": GATE_TAU,
        "baseline_neighbor_k": NEIGHBOR_K, "fit_neighbor_k": "all supports",
        "coordinates": ["axis", "pca"], "alphas": list(ALPHAS), "ridge": RIDGE,
        "input_sha256": {name: sha256_file(path) for name, path in input_paths().items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.round(6).to_string())
    return summary


if __name__ == "__main__":
    run()
