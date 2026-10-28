"""Fair support-corrected comparisons for strong composition priors."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb

from src.dcsa.baseline import PARAMS, _features, _load_points, _split_dois
from src.dcsa.masking_validation import select_support
from src.dcsa.crct import FeatureScaler, predict_episode
from src.dcsa.data_source import PROJECT_ROOT, input_paths, sha256_file


def _models(seed: int) -> dict[str, object]:
    return {
        "extra_trees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.8, random_state=seed, n_jobs=4),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.8, random_state=seed, n_jobs=4),
        "xgboost": XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, objective="reg:squarederror", n_jobs=4, random_state=seed),
        "catboost": CatBoostRegressor(iterations=300, depth=7, learning_rate=0.04, loss_function="RMSE", verbose=False, random_seed=seed, thread_count=4),
    }


def run_fair_baselines(root: Path = PROJECT_ROOT, *, seeds=(11, 22, 33), shots=(5,)) -> pd.DataFrame:
    points, valid = _load_points(root)
    feature_columns = sorted(c for c in points.columns if c.startswith("f_"))
    valid = valid.merge(points[["material", *feature_columns]].drop_duplicates("material"), on="material", how="inner")
    selected = pd.read_csv(root / "results" / "crct_selected_parameters.csv").set_index("seed")
    splits = []
    rows: list[dict[str, object]] = []
    for seed in seeds:
        train_dois, test_dois = _split_dois(points.doi.unique(), seed)
        train = points[points.doi.isin(train_dois)]
        test = points[points.doi.isin(test_dois)]
        fitted = {}
        x_train = _features(train, feature_columns)
        y_train = train.target.to_numpy(float)
        models = {"lightgbm": lgb.LGBMRegressor(**PARAMS, random_state=seed), **_models(seed)}
        scaler = FeatureScaler.fit(train, feature_columns)
        parameters = selected.loc[seed]
        gate_tau = None if pd.isna(parameters.gate_tau) else float(parameters.gate_tau)
        for name, model in models.items():
            model.fit(x_train, y_train)
            fitted[name] = model
        for shot in shots:
            eligible = valid[valid.doi.isin(test_dois)].groupby("doi").filter(lambda g: g.material.nunique() >= shot + 1)
            for doi, curves in eligible.groupby("doi", sort=True):
                sm, qm = select_support(curves, shot, "diverse_support", seed=seed)
                support = test[test.curve_row_id.isin(set(sm.curve_row_id))]
                query = test[test.curve_row_id.isin(set(qm.curve_row_id))]
                assert set(support.material).isdisjoint(set(query.material))
                splits.append({"seed": seed, "doi": doi, "support": sm.curve_row_id.tolist(), "query": qm.curve_row_id.tolist()})
                xs, xq = _features(support, feature_columns), _features(query, feature_columns)
                for name, model in fitted.items():
                    support_base = model.predict(xs)
                    query_base = model.predict(xq)
                    st = support.temperature.to_numpy(float) / 1000.0
                    qt = query.temperature.to_numpy(float) / 1000.0
                    design_s = np.column_stack([np.ones(len(st)), st, st**2])
                    design_q = np.column_stack([np.ones(len(qt)), qt, qt**2])
                    residual = support.target.to_numpy(float) - support_base
                    ridge = Ridge(alpha=1.0).fit(design_s, residual)
                    pred = query_base + ridge.predict(design_q)
                    predictions = {name + "+zero_shot": query_base, name + "+support_ridge": pred}
                    if name == "lightgbm":
                        for label, k, gate in [("all_support", None, None), ("two_neighbors", 2, None), ("full_crct", 2, gate_tau)]:
                            predictions[label] = predict_episode(support, query, feature_columns, support_base=support_base, query_base=query_base, scaler=scaler, tau=float(parameters.tau), neighbor_k=k, gate_tau=gate).prediction
                    for label, prediction in predictions.items():
                        rows.append({"seed": seed, "shot": shot, "doi": doi, "model": label, "mae": float(mean_absolute_error(query.target, prediction)), "n_query_points": len(query)})
        print(f"seed={seed}: train_dois={len(train_dois)} test_dois={len(test_dois)}", flush=True)
    result = pd.DataFrame(rows)
    result["episode_id"] = result.seed.astype(str) + ":" + result.doi.astype(str)
    output = root / "results"
    result.to_csv(output / "matched_baselines_ablation.csv", index=False)
    summary = result.groupby(["shot", "model"], as_index=False).agg(mae_mean=("mae", "mean"), mae_median=("mae", "median"), n_episodes=("episode_id", "nunique"))
    summary.to_csv(output / "matched_baselines_ablation_summary.csv", index=False)
    manifest = {"seeds": list(seeds), "shots": list(shots), "split": "DOI-held-out", "support_correction": "quadratic temperature Ridge fitted only on support residuals", "input_sha256": {name: sha256_file(path) for name, path in input_paths().items()}}
    manifest.update(support_selection="diverse_support", baseline_tuning="fixed parameters; not nested tuned", ablation="remove components while keeping full-model inner-selected tau fixed", episodes=splits)
    (output / "matched_baselines_ablation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(run_fair_baselines().to_string(index=False))
