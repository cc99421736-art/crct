"""Strict DOI-held-out baselines for the independent DCSA project."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from src.dcsa.data_source import PROJECT_ROOT, input_paths, sha256_file
from src.dcsa.support_audit import canonical_composition
from src.features.element_property_features import element_features

PARAMS = dict(n_estimators=120, learning_rate=0.05, num_leaves=31,
              n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True)
SHOTS = (1, 2, 3, 5)


@dataclass(frozen=True)
class Episode:
    doi: str
    support: pd.DataFrame
    query: pd.DataFrame


def make_episode(curves: pd.DataFrame, shot: int) -> Episode:
    if shot < 1 or curves.doi.nunique() != 1:
        raise ValueError("invalid episode")
    distinct = curves.sort_values(["material", "curve_row_id"], kind="mergesort").drop_duplicates("material")
    support = distinct.head(shot).copy()
    query = distinct.loc[~distinct.material.isin(set(support.material))].copy()
    return Episode(str(distinct.iloc[0].doi), support, query)


def paper_mean_offset(residuals: pd.DataFrame) -> float:
    if residuals.empty:
        return 0.0
    return float(residuals.groupby("curve_row_id").residual.mean().mean())


def _load_points(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = pd.read_csv(root / "results" / "support_curve_valid_curves.csv")
    long = pd.read_parquet(input_paths()["curves"], columns=[
        "curve_row_id", "composition_normalized", "property_class",
        "temperature_numeric", "y_numeric",
    ])
    points = long[long.property_class.eq("ZT")].copy()
    points = points.merge(valid[["doi", "material", "composition_normalized", "curve_row_id"]],
                          on=["composition_normalized", "curve_row_id"], how="inner")
    points["temperature"] = pd.to_numeric(points.temperature_numeric, errors="coerce")
    points["target"] = pd.to_numeric(points.y_numeric, errors="coerce")
    points = points.dropna(subset=["temperature", "target"])
    feature_rows = []
    for formula in points.composition_normalized.drop_duplicates():
        f = canonical_composition(formula)
        parsed = element_features(formula)
        if f and parsed:
            feature_rows.append({"composition_normalized": formula, **{
                f"f_{k}": float(v) for k, v in parsed.items()
                if isinstance(v, (int, float)) and np.isfinite(v)
            }})
    features = pd.DataFrame(feature_rows)
    points = points.merge(features, on="composition_normalized", validate="many_to_one")
    return points.reset_index(drop=True), valid


def _split_dois(dois: np.ndarray, seed: int) -> tuple[set[str], set[str]]:
    rng = np.random.default_rng(seed)
    ordered = np.array(sorted(dois))
    n_test = max(1, int(np.ceil(.2 * len(ordered))))
    test = set(rng.permutation(ordered)[:n_test])
    return set(ordered) - test, test


def _features(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return frame[columns + ["temperature"]].to_numpy(dtype=np.float32)


def run_baseline(root: Path = PROJECT_ROOT, *, seeds=(11, 22, 33), shots=SHOTS) -> pd.DataFrame:
    output = root / "results"
    points, valid = _load_points(root)
    feature_columns = sorted(c for c in points.columns if c.startswith("f_"))
    rows = []
    for seed in seeds:
        train_dois, test_dois = _split_dois(points.doi.unique(), seed)
        train = points[points.doi.isin(train_dois)].copy()
        test_curves = valid[valid.doi.isin(test_dois)]
        model = lgb.LGBMRegressor(**PARAMS, random_state=seed)
        model.fit(_features(train, feature_columns), train.target.to_numpy(float))
        for shot in shots:
            eligible = test_curves.groupby("doi").filter(lambda g: g.material.nunique() >= shot + 1)
            for doi, curves in eligible.groupby("doi", sort=True):
                episode = make_episode(curves, shot)
                support_ids = set(episode.support.curve_row_id)
                query_ids = set(episode.query.curve_row_id)
                support = points[points.curve_row_id.isin(support_ids)].copy()
                query = points[points.curve_row_id.isin(query_ids)].copy()
                support["base"] = model.predict(_features(support, feature_columns))
                query_base = model.predict(_features(query, feature_columns))
                support["residual"] = support.target - support.base
                design_s = np.column_stack([np.ones(len(support)), support.temperature / 1000., (support.temperature / 1000.) ** 2])
                design_q = np.column_stack([np.ones(len(query)), query.temperature / 1000., (query.temperature / 1000.) ** 2])
                ridge = Ridge(alpha=1.0).fit(design_s, support.residual)
                predictions = {
                    "zero_shot": query_base,
                    "paper_offset": query_base + paper_mean_offset(support[["curve_row_id", "residual"]]),
                    "temperature_ridge": query_base + ridge.predict(design_q),
                }
                for model_name, prediction in predictions.items():
                    rows.append({"seed": seed, "shot": shot, "doi": doi,
                                 "model": model_name, "n_query_points": len(query),
                                 "mae": float(mean_absolute_error(query.target, prediction))})
        print(f"seed={seed}: train_dois={len(train_dois)}, test_dois={len(test_dois)}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(output / "support_curve_baseline.csv", index=False)
    result["episode_id"] = result["seed"].astype(str) + ":" + result["doi"]
    summary = (result.groupby(["shot", "model"], as_index=False)
               .agg(mae_mean=("mae", "mean"), mae_median=("mae", "median"),
                    n_episodes=("episode_id", "nunique"), n_rows=("seed", "size")))
    summary.to_csv(output / "support_curve_baseline_summary.csv", index=False)
    manifest = {"seeds": list(seeds), "shots": list(shots), "target": "ZT",
                "split": "DOI-held-out", "feature_columns": feature_columns,
                "input_sha256": {name: sha256_file(path) for name, path in input_paths().items()}}
    (output / "support_curve_baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# DCSA baseline", "", "| shot | model | mean MAE | DOI episodes |",
             "|---:|---|---:|---:|"]
    lines.extend(f"| {r.shot} | {r.model} | {r.mae_mean:.6f} | {r.n_episodes} |"
                 for r in summary.itertuples())
    (output / "support_curve_baseline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(run_baseline().to_string(index=False))
