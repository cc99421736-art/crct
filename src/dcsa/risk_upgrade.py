"""Risk-layer upgrade: multi-source risk features with inner calibration.

Features
    R_d  mean nearest composition distance
    R_T  fraction of query temperatures outside the support range
    R_g  1 - mean distance gate (gate insufficiency)
    R_e  prediction disagreement: SD of predicted peaks over an ensemble of
         support subsets, transport hyper-parameters and the Ridge anchor

Calibration (inner only)
    standardise features on inner episodes
    non-negative least squares of MAE on the four features (weights >= 0)
    optional isotonic mapping from the combined score to MAE

Evaluation (outer only)
    Spearman / Kendall with true MAE, AURC, retained 25/50/75 % MAE,
    risk-quartile MAE and monotonicity.

The outer test DOIs never enter the prior or the weight calibration.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import kendalltau, spearmanr
from sklearn.isotonic import IsotonicRegression

from src.dcsa.baseline import PARAMS, _features, _load_points, _split_dois
from src.dcsa.crct import FeatureScaler, predict_episode
from src.dcsa.data_source import PROJECT_ROOT, input_paths, sha256_file
from src.dcsa.masking_validation import select_support

SHOT = 5
FEATURES = ("R_d", "R_T", "R_g", "R_e")
BASE = {"tau": 0.25, "neighbor_k": 2, "gate_tau": 1.0}
PERTURB = (
    {"tau": 0.20, "neighbor_k": 2, "gate_tau": 1.0},
    {"tau": 0.30, "neighbor_k": 2, "gate_tau": 1.0},
    {"tau": 0.25, "neighbor_k": 1, "gate_tau": 1.0},
    {"tau": 0.25, "neighbor_k": 3, "gate_tau": 1.0},
    {"tau": 0.25, "neighbor_k": 2, "gate_tau": 0.8},
    {"tau": 0.25, "neighbor_k": 2, "gate_tau": 1.2},
)


def _episode_features(support: pd.DataFrame, query: pd.DataFrame, support_meta: pd.DataFrame,
                      feature_columns: list[str], prior, scaler: FeatureScaler) -> dict[str, float]:
    support_base = prior.predict(_features(support, feature_columns))
    query_base = prior.predict(_features(query, feature_columns))
    base_prediction = predict_episode(
        support, query, feature_columns, support_base=support_base,
        query_base=query_base, scaler=scaler, **BASE)
    peaks = []
    # support subsets: leave one support out
    curve_ids = list(support_meta.curve_row_id)
    for held in curve_ids:
        inner = support_meta[support_meta.curve_row_id != held]
        if len(inner) < 2:
            continue
        inner_support = support[support.curve_row_id.isin(inner.curve_row_id)]
        inner_base = prior.predict(_features(inner_support, feature_columns))
        subset = predict_episode(
            inner_support, query, feature_columns, support_base=inner_base,
            query_base=query_base, scaler=scaler,
            tau=BASE["tau"], neighbor_k=min(BASE["neighbor_k"], len(inner)),
            gate_tau=BASE["gate_tau"])
        peaks.append(float(np.max(subset.variants["chemistry_only"])))
    # transport hyper-parameter perturbations
    for config in PERTURB:
        perturbed = predict_episode(
            support, query, feature_columns, support_base=support_base,
            query_base=query_base, scaler=scaler, tau=config["tau"],
            neighbor_k=config["neighbor_k"], gate_tau=config["gate_tau"])
        peaks.append(float(np.max(perturbed.variants["chemistry_only"])))
    # Ridge anchor as an additional ensemble member
    peaks.append(float(np.max(base_prediction.variants["ridge_only"])))

    query_temperature = query.temperature.to_numpy(float)
    ranges = support.groupby("curve_row_id").temperature.agg(["min", "max"]).to_numpy(float)
    covered = np.zeros(len(query), dtype=bool)
    for lower, upper in ranges:
        covered |= (query_temperature >= lower) & (query_temperature <= upper)
    return {
        "R_d": float(np.mean(base_prediction.nearest_distance)),
        "R_T": float(1.0 - covered.mean()),
        "R_g": float(1.0 - np.mean(base_prediction.gate)),
        "R_e": float(np.std(peaks)),
    }


def _episodes(points, valid, train_dois, eval_dois, feature_columns, seed):
    train = points[points.doi.isin(train_dois)]
    test = points[points.doi.isin(eval_dois)]
    prior = lgb.LGBMRegressor(**PARAMS, random_state=seed)
    prior.fit(_features(train, feature_columns), train.target.to_numpy(float))
    scaler = FeatureScaler.fit(train, feature_columns)
    eligible = valid[valid.doi.isin(eval_dois)].groupby("doi").filter(
        lambda group: group.material.nunique() > SHOT)
    rows = []
    for doi, curves in eligible.groupby("doi", sort=True):
        support_meta, query_meta = select_support(curves, SHOT, "diverse_support", seed=seed)
        support = test[test.curve_row_id.isin(support_meta.curve_row_id)]
        query = test[test.curve_row_id.isin(query_meta.curve_row_id)]
        prediction = predict_episode(
            support, query, feature_columns,
            support_base=prior.predict(_features(support, feature_columns)),
            query_base=prior.predict(_features(query, feature_columns)),
            scaler=scaler, **BASE)
        target = query.target.to_numpy(float)
        rows.append({"seed": seed, "doi": doi,
                     "mae": float(np.mean(np.abs(target - prediction.variants["chemistry_only"]))),
                     **_episode_features(support, query, support_meta, feature_columns, prior, scaler)})
    return pd.DataFrame(rows)


def _aurc(risk: np.ndarray, error: np.ndarray) -> float:
    order = np.argsort(risk, kind="mergesort")
    cumulative = np.cumsum(error[order]) / np.arange(1, len(error) + 1)
    return float(np.mean(cumulative))


def _evaluate(frame: pd.DataFrame, risk_column: str) -> dict[str, float]:
    risk = frame[risk_column].to_numpy(float)
    error = frame.mae.to_numpy(float)
    rho = spearmanr(risk, error).statistic
    tau = kendalltau(risk, error).statistic
    result = {"method": risk_column, "n_episodes": int(len(frame)),
              "spearman": float(rho), "kendall": float(tau),
              "aurc": _aurc(risk, error), "mae_all": float(error.mean())}
    quartile = pd.qcut(risk, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    grouped = frame.assign(_q=quartile).groupby("_q", observed=False).mae.mean()
    for name, value in grouped.items():
        result[f"mae_{name}"] = float(value)
    for fraction in (0.25, 0.50, 0.75):
        threshold = np.quantile(risk, fraction)
        result[f"mae_lowest_{int(fraction * 100)}"] = float(error[risk <= threshold].mean())
    return result


def run(root: Path = PROJECT_ROOT, *, seeds=(11, 22, 33)) -> pd.DataFrame:
    points, valid = _load_points(root)
    feature_columns = sorted(column for column in points.columns if column.startswith("f_"))
    feature_table = points[["material", *feature_columns]].drop_duplicates("material")
    valid = valid.merge(feature_table, on="material", how="inner")
    all_dois = points.doi.unique()
    inner_frames, outer_frames = [], []
    weights = []
    for seed in seeds:
        outer_train, outer_test = _split_dois(all_dois, seed)
        inner_train, inner_val = _split_dois(np.array(sorted(outer_train)), seed + 10000)
        inner = _episodes(points, valid, inner_train, inner_val, feature_columns, seed + 10000)
        outer = _episodes(points, valid, outer_train, outer_test, feature_columns, seed)
        mean = inner[list(FEATURES)].mean()
        scale = inner[list(FEATURES)].std(ddof=0).replace(0, 1.0)
        x_inner = ((inner[list(FEATURES)] - mean) / scale).to_numpy(float)
        x_outer = ((outer[list(FEATURES)] - mean) / scale).to_numpy(float)
        y_inner = inner.mae.to_numpy(float)
        # non-negative weights plus a free intercept
        design = np.column_stack([np.ones(len(x_inner)), x_inner])
        weights_raw, _ = nnls(design, y_inner)
        intercept, w = float(weights_raw[0]), weights_raw[1:]
        score_inner = intercept + x_inner @ w
        score_outer = intercept + x_outer @ w
        isotonic = IsotonicRegression(out_of_bounds="clip").fit(score_inner, y_inner)
        # old risk layer: equal-weight z(d) + z(T), standardised on inner
        old_inner = x_inner[:, 0] + x_inner[:, 1]
        old_outer = x_outer[:, 0] + x_outer[:, 1]
        outer = outer.assign(new_score=score_outer,
                             new_risk=isotonic.predict(score_outer),
                             old_risk=old_outer)
        inner = inner.assign(new_score=score_inner, old_risk=old_inner)
        inner_frames.append(inner)
        outer_frames.append(outer)
        weights.append({"seed": seed, "intercept": intercept,
                        **{f"w_{name}": float(value) for name, value in zip(FEATURES, w)}})
        print(f"seed={seed}: weights={np.round(w, 4)}", flush=True)

    inner_all = pd.concat(inner_frames, ignore_index=True)
    outer_all = pd.concat(outer_frames, ignore_index=True)
    inner_all.to_csv(root / "results" / "risk_upgrade_inner.csv", index=False)
    rows = []
    for split, frame in (("inner", inner_all), ("outer", outer_all)):
        for method in ("old_risk", "new_score"):
            row = _evaluate(frame, method)
            row["split"] = split
            rows.append(row)
    row = _evaluate(outer_all, "new_risk")
    row["split"] = "outer"
    rows.append(row)
    table = pd.DataFrame(rows)
    output = root / "results"
    table.to_csv(output / "risk_upgrade_evaluation.csv", index=False)
    pd.DataFrame(weights).to_csv(output / "risk_upgrade_weights.csv", index=False)
    outer_all.to_csv(output / "risk_upgrade_outer.csv", index=False)
    (output / "risk_upgrade_manifest.json").write_text(json.dumps({
        "seeds": list(seeds), "shot": SHOT, "features": list(FEATURES),
        "calibration": "nnls weights + isotonic mapping, fitted on inner episodes only",
        "evaluation": "outer test episodes only",
        "old_risk": "equal-weight z(d_min) + z(f_outside), inner standardisation",
        "aurc": "mean cumulative MAE when episodes are sorted by increasing risk (lower is better)",
        "input_sha256": {name: sha256_file(path) for name, path in input_paths().items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(table.round(4).to_string(index=False))
    return table


if __name__ == "__main__":
    run()
