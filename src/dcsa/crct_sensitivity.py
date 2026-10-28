"""Parameter sensitivity experiments for CRCT masked validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.dcsa.baseline import PARAMS, _features, _load_points, _split_dois
from src.dcsa.crct import FeatureScaler, evaluate_episode
from src.dcsa.data_source import PROJECT_ROOT, input_paths, sha256_file
from src.dcsa.masking_validation import SHOT, select_support


DEFAULT_TAUS = (0.25, 0.5, 1.0)
DEFAULT_BLENDS = (0.0, 0.5, 1.0)
DEFAULT_SUPPORT_STRATEGIES = ("random", "diverse_support")


def validate_parameter_grid(
    taus: Iterable[float],
    blends: Iterable[float],
    support_strategies: Iterable[str],
) -> None:
    """Validate the sensitivity grid before touching the dataset."""
    taus = tuple(float(value) for value in taus)
    blends = tuple(float(value) for value in blends)
    support_strategies = tuple(support_strategies)
    if not taus or any(not np.isfinite(value) or value <= 0.0 for value in taus):
        raise ValueError("tau values must be finite and positive")
    if not blends or any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in blends):
        raise ValueError("blend values must be finite and between zero and one")
    if not support_strategies or any(not isinstance(value, str) or not value for value in support_strategies):
        raise ValueError("support strategies must be non-empty strings")


def parameter_grid(
    taus: Iterable[float] = DEFAULT_TAUS,
    blends: Iterable[float] = DEFAULT_BLENDS,
    support_strategies: Iterable[str] = DEFAULT_SUPPORT_STRATEGIES,
) -> list[dict[str, float | str]]:
    """Return the complete support-selector and CRCT hyperparameter grid."""
    taus = tuple(float(value) for value in taus)
    blends = tuple(float(value) for value in blends)
    support_strategies = tuple(support_strategies)
    validate_parameter_grid(taus, blends, support_strategies)
    return [
        {"support_selection": strategy, "tau": tau, "blend": blend}
        for strategy in support_strategies
        for tau in taus
        for blend in blends
    ]


def summarize_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate episode results for every protocol and distance subgroup."""
    required = {"support_selection", "tau", "blend", "distance_group", "mae"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"sensitivity results are missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("sensitivity results must not be empty")
    group_columns = [
        column for column in
        ("shot", "method", "support_selection", "tau", "blend", "distance_group")
        if column in frame.columns
    ]
    aggregations: dict[str, tuple[str, str]] = {
        "mae_mean": ("mae", "mean"),
        "mae_median": ("mae", "median"),
        "n_episodes": ("mae", "size"),
    }
    for source, output in (
        ("chemistry_mae", "chemistry_mae_mean"),
        ("ridge_mae", "ridge_mae_mean"),
        ("peak_zt_mae", "peak_zt_mae_mean"),
        ("optimum_temperature_mae", "optimum_temperature_mae_mean"),
    ):
        if source in frame.columns:
            aggregations[output] = (source, "mean")
    if "doi" in frame.columns:
        aggregations["n_dois"] = ("doi", "nunique")
    return (
        frame.groupby(group_columns, as_index=False)
        .agg(**aggregations)
        .sort_values(group_columns, kind="mergesort")
        .reset_index(drop=True)
    )


def run_crct_sensitivity(
    root: Path = PROJECT_ROOT,
    *,
    seeds: Iterable[int] = (11, 22, 33),
    shot: int = SHOT,
    taus: Iterable[float] = DEFAULT_TAUS,
    blends: Iterable[float] = DEFAULT_BLENDS,
    support_strategies: Iterable[str] = DEFAULT_SUPPORT_STRATEGIES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one strict DOI-held-out experiment over the CRCT parameter grid.

    Every parameter combination uses the same train/test DOI split and the
    same selected support/query curves for a given seed, DOI, and strategy.
    Results are written to dedicated sensitivity files and existing CRCT
    result files are left untouched.
    """
    seeds = tuple(int(seed) for seed in seeds)
    taus = tuple(float(value) for value in taus)
    blends = tuple(float(value) for value in blends)
    support_strategies = tuple(support_strategies)
    validate_parameter_grid(taus, blends, support_strategies)
    if not seeds:
        raise ValueError("seeds must not be empty")
    if shot < 1:
        raise ValueError("shot must be positive")

    points, valid = _load_points(root)
    feature_columns = sorted(column for column in points.columns if column.startswith("f_"))
    feature_table = points[["material", *feature_columns]].drop_duplicates("material")
    valid = valid.merge(feature_table, on="material", how="inner")
    rows: list[dict[str, object]] = []

    for seed in seeds:
        train_dois, test_dois = _split_dois(points.doi.unique(), seed)
        train = points[points.doi.isin(train_dois)].copy()
        test = points[points.doi.isin(test_dois)]
        prior = lgb.LGBMRegressor(**PARAMS, random_state=seed)
        prior.fit(_features(train, feature_columns), train.target.to_numpy(dtype=float))
        scaler = FeatureScaler.fit(train, feature_columns)
        eligible = valid[valid.doi.isin(test_dois)].groupby("doi").filter(
            lambda group: group.material.nunique() > shot
        )

        for doi, curves in eligible.groupby("doi", sort=True):
            for strategy in support_strategies:
                support_meta, query_meta = select_support(curves, shot, strategy, seed=seed)
                support = test[test.curve_row_id.isin(support_meta.curve_row_id)].copy()
                query = test[test.curve_row_id.isin(query_meta.curve_row_id)].copy()
                support_base = prior.predict(_features(support, feature_columns))
                query_base = prior.predict(_features(query, feature_columns))
                for tau in taus:
                    for blend in blends:
                        metrics = evaluate_episode(
                            support,
                            query,
                            feature_columns,
                            support_base=support_base,
                            query_base=query_base,
                            scaler=scaler,
                            tau=tau,
                            blend=blend,
                        )
                        rows.append({
                            "seed": seed,
                            "doi": doi,
                            "shot": shot,
                            "method": "crct",
                            "support_selection": strategy,
                            "tau": tau,
                            "blend": blend,
                            **metrics,
                        })
        print(f"seed={seed}: train_dois={len(train_dois)}, test_dois={len(test_dois)}", flush=True)

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("sensitivity run produced no eligible episodes")
    result["episode_id"] = result.seed.astype(str) + ":" + result.doi.astype(str)
    result["distance_group"] = ""
    thresholds: dict[str, float] = {}
    for strategy, group in result.groupby("support_selection", sort=True):
        threshold = float(group.mean_nearest_distance.quantile(0.75))
        thresholds[strategy] = threshold
        result.loc[group.index, "distance_group"] = np.where(
            group.mean_nearest_distance.to_numpy(float) >= threshold,
            "far_q25",
            "near_q75",
        )

    output = root / "results"
    result.to_csv(output / "crct_parameter_sensitivity.csv", index=False)
    summary = summarize_sensitivity(result)
    summary.to_csv(output / "crct_parameter_sensitivity_summary.csv", index=False)
    manifest = {
        "seeds": list(seeds),
        "shot": shot,
        "taus": list(taus),
        "blends": list(blends),
        "support_strategies": list(support_strategies),
        "grid_size": len(parameter_grid(taus, blends, support_strategies)),
        "split": "DOI-held-out",
        "support_query_protocol": "same seed/DOI/strategy support selection across all tau and blend values",
        "distance": "Euclidean on training-DOI standardized composition features",
        "distance_thresholds_mean_nearest_distance_q75": thresholds,
        "temperature_interpolation": "linear with endpoint clamping",
        "target": "ZT",
        "input_sha256": {name: sha256_file(path) for name, path in input_paths().items()},
    }
    (output / "crct_parameter_sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result, summary


if __name__ == "__main__":
    _, output_summary = run_crct_sensitivity()
    print(output_summary.to_string(index=False))
