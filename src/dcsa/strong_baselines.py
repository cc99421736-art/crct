"""Strong composition-temperature baselines for CMS submission."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.dcsa.baseline import _features, _load_points, _split_dois, make_episode
from src.dcsa.data_source import PROJECT_ROOT, input_paths, sha256_file


def _models(seed: int) -> dict[str, object]:
    return {
        "extra_trees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2,
            max_features=0.8, random_state=seed, n_jobs=4),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
            max_features=0.8, random_state=seed, n_jobs=4),
        "xgboost": XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85, objective="reg:squarederror",
            n_jobs=4, random_state=seed),
        "catboost": CatBoostRegressor(iterations=300, depth=7, learning_rate=0.04,
            loss_function="RMSE", verbose=False, random_seed=seed, thread_count=4),
    }


def run_strong_baselines(root: Path = PROJECT_ROOT, *, seeds=(11, 22, 33), shots=(1, 2, 3, 5)) -> pd.DataFrame:
    points, valid = _load_points(root)
    feature_columns = sorted(c for c in points.columns if c.startswith("f_"))
    rows: list[dict[str, object]] = []
    for seed in seeds:
        train_dois, test_dois = _split_dois(points.doi.unique(), seed)
        train = points[points.doi.isin(train_dois)]
        test = points[points.doi.isin(test_dois)]
        fitted = {}
        x_train = _features(train, feature_columns)
        y_train = train.target.to_numpy(float)
        for name, model in _models(seed).items():
            model.fit(x_train, y_train)
            fitted[name] = model
        for shot in shots:
            eligible = valid[valid.doi.isin(test_dois)].groupby("doi").filter(
                lambda g: g.material.nunique() >= shot + 1)
            for doi, curves in eligible.groupby("doi", sort=True):
                episode = make_episode(curves, shot)
                query = test[test.curve_row_id.isin(set(episode.query.curve_row_id))]
                x_query = _features(query, feature_columns)
                for name, model in fitted.items():
                    pred = model.predict(x_query)
                    rows.append({"seed": seed, "shot": shot, "doi": doi,
                                 "model": name, "n_query_points": len(query),
                                 "mae": float(mean_absolute_error(query.target, pred))})
        print(f"seed={seed}: train_dois={len(train_dois)} test_dois={len(test_dois)}", flush=True)
    result = pd.DataFrame(rows)
    result["episode_id"] = result.seed.astype(str) + ":" + result.doi.astype(str)
    output = root / "results"
    result.to_csv(output / "strong_baselines.csv", index=False)
    summary = (result.groupby(["shot", "model"], as_index=False)
               .agg(mae_mean=("mae", "mean"), mae_median=("mae", "median"),
                    n_episodes=("episode_id", "nunique")))
    summary.to_csv(output / "strong_baselines_summary.csv", index=False)
    manifest = {"seeds": list(seeds), "shots": list(shots), "split": "DOI-held-out",
                "models": list(_models(0)), "feature_columns": feature_columns,
                "input_sha256": {name: sha256_file(path) for name, path in input_paths().items()}}
    (output / "strong_baselines_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(run_strong_baselines().to_string(index=False))
