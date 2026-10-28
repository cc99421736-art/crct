"""Shot sensitivity: 1/2/3/5 support curves for every model family.

Every shot level and support strategy uses the same DOI fold, the same seed,
the same support selector and the same transport hyper-parameters
(tau=0.25, k=2, gate=1.0).  That isolates the effect of the number of measured
support curves from hyper-parameter tuning.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.dcsa.baseline import PARAMS, _features, _load_points, _split_dois
from src.dcsa.crct import FeatureScaler, predict_episode, variant_metrics
from src.dcsa.data_source import PROJECT_ROOT, input_paths, sha256_file
from src.dcsa.masking_validation import select_support

SHOTS = (1, 2, 3, 5)
STRATEGIES = ("diverse_support", "random")
TAU = 0.25
NEIGHBOR_K = 2
GATE_TAU = 1.0
MODELS = ("zero_shot", "paper_offset", "ridge_only", "chemistry_only",
          "additive_raw", "additive_orthogonal", "convex_blend")


def _paper_offset(support: pd.DataFrame, support_base: np.ndarray) -> float:
    residual = support.target.to_numpy(dtype=float) - np.asarray(support_base, dtype=float)
    frame = pd.DataFrame({"curve_row_id": support.curve_row_id.to_numpy(), "residual": residual})
    return float(frame.groupby("curve_row_id").residual.mean().mean())


def _bootstrap_improvement(frame: pd.DataFrame, baseline: str, method: str,
                           n_boot: int = 5000, seed: int = 20261020) -> dict[str, float]:
    improvement = (frame[baseline] - frame[method]).astype(float)
    by_doi = improvement.groupby(frame.doi).mean()
    values = by_doi.to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = values[draws].mean(axis=1)
    return {"n_dois": int(len(values)),
            "mean_improvement": float(improvement.mean()),
            "ci95_low": float(np.quantile(boot, .025)),
            "ci95_high": float(np.quantile(boot, .975)),
            "win_rate": float(np.mean(improvement > 0))}


def run(root: Path = PROJECT_ROOT, *, seeds=(11, 22, 33), shots=SHOTS,
        suffix: str = "") -> pd.DataFrame:
    points, valid = _load_points(root)
    feature_columns = sorted(column for column in points.columns if column.startswith("f_"))
    feature_table = points[["material", *feature_columns]].drop_duplicates("material")
    valid = valid.merge(feature_table, on="material", how="inner")
    all_dois = points.doi.unique()
    rows: list[dict[str, object]] = []
    indices: list[dict[str, object]] = []

    for seed in seeds:
        train_dois, test_dois = _split_dois(all_dois, seed)
        train = points[points.doi.isin(train_dois)]
        test = points[points.doi.isin(test_dois)]
        prior = lgb.LGBMRegressor(**PARAMS, random_state=seed)
        prior.fit(_features(train, feature_columns), train.target.to_numpy(float))
        scaler = FeatureScaler.fit(train, feature_columns)
        for shot in shots:
            eligible = valid[valid.doi.isin(test_dois)].groupby("doi").filter(
                lambda group: group.material.nunique() > shot)
            for strategy in STRATEGIES:
                for doi, curves in eligible.groupby("doi", sort=True):
                    support_meta, query_meta = select_support(
                        curves, shot, strategy, seed=seed)
                    indices.append({"seed": seed, "doi": doi, "shot": shot,
                                    "support_selection": strategy,
                                    "support_curve_ids": "|".join(
                                        str(int(v)) for v in support_meta.curve_row_id),
                                    "query_curve_ids": "|".join(
                                        str(int(v)) for v in query_meta.curve_row_id)})
                    support = test[test.curve_row_id.isin(support_meta.curve_row_id)].copy()
                    query = test[test.curve_row_id.isin(query_meta.curve_row_id)].copy()
                    if query.empty:
                        continue
                    support_base = prior.predict(_features(support, feature_columns))
                    query_base = prior.predict(_features(query, feature_columns))
                    prediction = predict_episode(
                        support, query, feature_columns, support_base=support_base,
                        query_base=query_base, scaler=scaler, tau=TAU,
                        blend=1.0, neighbor_k=NEIGHBOR_K, gate_tau=GATE_TAU)
                    metrics = variant_metrics(query, prediction)
                    target = query.target.to_numpy(dtype=float)
                    offset = _paper_offset(support, support_base)
                    rows.append({
                        "seed": seed, "doi": doi, "shot": shot,
                        "support_selection": strategy,
                        "n_support_curves": int(support.curve_row_id.nunique()),
                        "n_query_curves": int(query.curve_row_id.nunique()),
                        "n_query_points": int(len(query)),
                        "mae_zero_shot": float(np.mean(np.abs(target - query_base))),
                        "mae_paper_offset": float(np.mean(np.abs(target - (query_base + offset)))),
                        **{f"mae_{model}": metrics[f"mae_{model}"] for model in
                           ("ridge_only", "chemistry_only", "additive_raw",
                            "additive_orthogonal", "convex_blend")},
                    })
        print(f"seed={seed}: train={len(train_dois)} test={len(test_dois)}", flush=True)

    detail = pd.DataFrame(rows)
    detail.to_csv(root / "results" / f"crct_shot_sensitivity{suffix}.csv", index=False)
    pd.DataFrame(indices).to_csv(
        root / "results" / f"crct_shot_sensitivity{suffix}_episode_indices.csv", index=False)

    long_rows = []
    for (shot, strategy), group in detail.groupby(["shot", "support_selection"], sort=True):
        for model in MODELS:
            column = f"mae_{model}"
            long_rows.append({"shot": shot, "support_selection": strategy, "model": model,
                              "n_episodes": int(len(group)), "n_dois": int(group.doi.nunique()),
                              "mae_mean": float(group[column].mean()),
                              "mae_median": float(group[column].median())})
    long = pd.DataFrame(long_rows)
    long.to_csv(root / "results" / f"crct_shot_sensitivity{suffix}_summary.csv", index=False)

    wide_rows = []
    for (shot, strategy), group in detail.groupby(["shot", "support_selection"], sort=True):
        stats = _bootstrap_improvement(group, "mae_ridge_only", "mae_chemistry_only")
        row = {"shot": shot, "support_selection": strategy,
               "n_episodes": int(len(group)), "n_dois": int(group.doi.nunique())}
        for model in MODELS:
            row[model] = float(group[f"mae_{model}"].mean())
        row["chemistry_improvement_vs_ridge"] = float(
            (group.mae_ridge_only - group.mae_chemistry_only).mean())
        row["chemistry_relative_reduction_pct"] = 100.0 * (
            1.0 - row["chemistry_only"] / row["ridge_only"])
        row["ci95_low"] = stats["ci95_low"]
        row["ci95_high"] = stats["ci95_high"]
        row["win_rate"] = stats["win_rate"]
        wide_rows.append(row)
    wide = pd.DataFrame(wide_rows)
    wide.to_csv(root / "results" / f"crct_shot_sensitivity{suffix}_table.csv", index=False)

    (root / "results" / f"crct_shot_sensitivity{suffix}_manifest.json").write_text(json.dumps({
        "seeds": list(seeds), "shots": list(shots), "support_strategies": list(STRATEGIES),
        "transport": {"tau": TAU, "neighbor_k": NEIGHBOR_K, "gate_tau": GATE_TAU,
                      "blend": 1.0},
        "models": list(MODELS),
        "split": "DOI-held-out; same folds and seeds for every shot level",
        "bootstrap": "DOI-clustered, 5000 draws, seed 20261020",
        "input_sha256": {name: sha256_file(path) for name, path in input_paths().items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(wide.to_string(index=False))
    return wide


if __name__ == "__main__":
    run()

