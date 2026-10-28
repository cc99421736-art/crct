"""Historical 5-shot masking validation for the DCSA application protocol."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from src.dcsa.baseline import PARAMS, _features, _load_points, _split_dois, paper_mean_offset
from src.dcsa.baseline_stats import clustered_bootstrap
from src.dcsa.data_source import PROJECT_ROOT

SHOT = 5
STRATEGIES = ("ordered", "random", "central_support", "diverse_support")


def _standardized_composition_features(curves: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Return finite, per-episode standardized composition features."""
    feature_columns = sorted(c for c in curves.columns if c.startswith("f_"))
    if not feature_columns:
        raise ValueError("diverse_support requires composition features")
    values = curves[feature_columns].to_numpy(dtype=float)
    mean = np.nanmean(values, axis=0)
    scale = np.nanstd(values, axis=0)
    mean[~np.isfinite(mean)] = 0.0
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    standardized = (values - mean) / scale
    return feature_columns, np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)


def _diverse_support(distinct: pd.DataFrame, shot: int) -> pd.DataFrame:
    """Select a deterministic farthest-point approximation to k-center.

    The first support is the composition closest to the candidate centroid;
    each subsequent support maximizes its distance to the selected set. Only
    ``f_*`` columns are used, so measured target values cannot affect selection.
    """
    _, features = _standardized_composition_features(distinct)
    centroid = features.mean(axis=0)
    selected = [int(np.argmin(np.linalg.norm(features - centroid, axis=1)))]
    min_distance = np.linalg.norm(features - features[selected[0]], axis=1)
    while len(selected) < shot:
        min_distance[selected] = -np.inf
        next_index = int(np.argmax(min_distance))
        selected.append(next_index)
        distance = np.linalg.norm(features - features[next_index], axis=1)
        min_distance = np.minimum(min_distance, distance)
    return distinct.iloc[selected]


def select_support(curves: pd.DataFrame, shot: int, strategy: str, *, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select whole composition curves; return support and disjoint query metadata."""
    distinct = curves.sort_values(["material", "curve_row_id"], kind="mergesort").drop_duplicates("material").copy()
    if len(distinct) <= shot:
        raise ValueError("at least one query curve is required")
    if strategy == "ordered":
        support = distinct.head(shot)
    elif strategy == "random":
        rng = np.random.default_rng(seed)
        support = distinct.iloc[np.sort(rng.choice(len(distinct), size=shot, replace=False))]
    elif strategy == "central_support":
        feature_columns = sorted(c for c in distinct.columns if c.startswith("f_"))
        if not feature_columns:
            raise ValueError("central_support requires composition features")
        _, z = _standardized_composition_features(distinct)
        distance = np.linalg.norm(z - z.mean(axis=0), axis=1)
        support = distinct.iloc[np.argsort(distance, kind="stable")[:shot]]
    elif strategy == "diverse_support":
        support = _diverse_support(distinct, shot)
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    support_materials = set(support.material)
    query = distinct.loc[~distinct.material.isin(support_materials)]
    return support.reset_index(drop=True), query.reset_index(drop=True)


def run_masking_validation(root: Path = PROJECT_ROOT, *, seeds=(11, 22, 33)) -> pd.DataFrame:
    points, valid = _load_points(root)
    feature_columns = sorted(c for c in points.columns if c.startswith("f_"))
    feature_table = (points[["material", *feature_columns]]
                     .drop_duplicates("material"))
    valid = valid.merge(feature_table, on="material", how="inner")
    rows: list[dict[str, object]] = []
    for seed in seeds:
        train_dois, test_dois = _split_dois(points.doi.unique(), seed)
        train = points[points.doi.isin(train_dois)]
        model = lgb.LGBMRegressor(**PARAMS, random_state=seed)
        model.fit(_features(train, feature_columns), train.target.to_numpy(float))
        eligible = valid[valid.doi.isin(test_dois)].groupby("doi").filter(lambda g: g.material.nunique() > SHOT)
        for doi, curves in eligible.groupby("doi", sort=True):
            for strategy in STRATEGIES:
                support_meta, query_meta = select_support(curves, SHOT, strategy, seed=seed)
                support = points[points.curve_row_id.isin(support_meta.curve_row_id)].copy()
                query = points[points.curve_row_id.isin(query_meta.curve_row_id)].copy()
                support["base"] = model.predict(_features(support, feature_columns))
                query_base = model.predict(_features(query, feature_columns))
                support["residual"] = support.target - support.base
                s_t = support.temperature.to_numpy(float) / 1000.0
                q_t = query.temperature.to_numpy(float) / 1000.0
                design_s = np.column_stack([np.ones(len(support)), s_t, s_t ** 2])
                design_q = np.column_stack([np.ones(len(query)), q_t, q_t ** 2])
                ridge = Ridge(alpha=1.0).fit(design_s, support.residual)
                predictions = {
                    "zero_shot": query_base,
                    "paper_offset": query_base + paper_mean_offset(support[["curve_row_id", "residual"]]),
                    "temperature_ridge": query_base + ridge.predict(design_q),
                }
                for model_name, prediction in predictions.items():
                    rows.append({"seed": seed, "doi": doi, "strategy": strategy,
                                 "shot": SHOT, "model": model_name,
                                 "n_support_curves": len(support_meta),
                                 "n_query_curves": len(query_meta),
                                 "n_query_points": len(query),
                                 "mae": float(mean_absolute_error(query.target, prediction))})
        print(f"seed={seed}: train_dois={len(train_dois)}, test_dois={len(test_dois)}", flush=True)
    result = pd.DataFrame(rows)
    output = root / "results"
    result.to_csv(output / "masking_validation.csv", index=False)
    result["episode_id"] = result.seed.astype(str) + ":" + result.doi.astype(str)
    summary = (result.groupby(["shot", "strategy", "model"], as_index=False)
               .agg(mae_mean=("mae", "mean"), mae_median=("mae", "median"),
                    n_episodes=("episode_id", "nunique"), n_rows=("mae", "size")))
    summary.to_csv(output / "masking_validation_summary.csv", index=False)
    bootstrap = []
    for strategy, group in result.groupby("strategy", sort=True):
        boot = clustered_bootstrap(group.drop(columns="episode_id"), n_boot=2000,
                                   seed=20261001 + list(STRATEGIES).index(strategy))
        boot.insert(1, "strategy", strategy)
        bootstrap.append(boot)
    pd.concat(bootstrap, ignore_index=True).to_csv(
        output / "masking_validation_bootstrap.csv", index=False)
    (output / "masking_validation_manifest.json").write_text(json.dumps({
        "shot": SHOT, "strategies": list(STRATEGIES), "seeds": list(seeds),
        "split": "DOI-held-out", "target": "ZT",
    }, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(run_masking_validation().to_string(index=False))
