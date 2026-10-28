"""Nested DOI-held-out calibration for CRCT hyperparameters."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.dcsa.baseline import PARAMS, _features, _load_points, _split_dois
from src.dcsa.crct import FeatureScaler, evaluate_episode, predict_episode
from src.dcsa.active_design import curve_candidates, rank_candidates, evaluate_ranking
from src.dcsa.data_source import PROJECT_ROOT, input_paths, sha256_file
from src.dcsa.masking_validation import SHOT, select_support

CANDIDATES = (
    {"tau": 0.25, "blend": 1.0, "neighbor_k": 2, "gate_tau": None},
    {"tau": 0.25, "blend": 1.0, "neighbor_k": 2, "gate_tau": 0.25},
    {"tau": 0.25, "blend": 1.0, "neighbor_k": 2, "gate_tau": 0.5},
    {"tau": 0.25, "blend": 1.0, "neighbor_k": 2, "gate_tau": 1.0},
    {"tau": 0.5, "blend": 1.0, "neighbor_k": 2, "gate_tau": None},
    {"tau": 0.5, "blend": 1.0, "neighbor_k": 2, "gate_tau": 0.5},
)


def choose_best(frame: pd.DataFrame) -> dict[str, float]:
    required = {"tau", "blend", "mae"}
    if missing := required - set(frame.columns):
        raise ValueError(f"calibration frame missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("calibration frame must not be empty")
    has_neighbor = "neighbor_k" in frame.columns
    has_gate = "gate_tau" in frame.columns
    if not has_neighbor:
        frame = frame.assign(neighbor_k=None)
    if not has_gate:
        frame = frame.assign(gate_tau=None)
    summary = frame.groupby(["tau", "blend", "neighbor_k", "gate_tau"], as_index=False, dropna=False).mae.mean()
    # Prefer simpler full chemistry transfer on exact ties.
    row = summary.sort_values(["mae", "blend", "tau"], ascending=[True, False, True], kind="mergesort").iloc[0]
    result = {"tau": float(row.tau), "blend": float(row.blend)}
    if has_neighbor:
        result["neighbor_k"] = None if pd.isna(row.neighbor_k) else int(row.neighbor_k)
    if has_gate:
        result["gate_tau"] = None if pd.isna(row.gate_tau) else float(row.gate_tau)
    return result


def _evaluate_split(points: pd.DataFrame, valid: pd.DataFrame,
                    train_dois: set[str], eval_dois: set[str],
                    feature_columns: list[str], candidates: tuple[dict[str, float], ...],
                    seed: int, shot: int = SHOT,
                    candidate_rows: list[dict[str, object]] | None = None) -> pd.DataFrame:
    train = points[points.doi.isin(train_dois)]
    test = points[points.doi.isin(eval_dois)]
    prior = lgb.LGBMRegressor(**PARAMS, random_state=seed)
    prior.fit(_features(train, feature_columns), train.target.to_numpy(float))
    scaler = FeatureScaler.fit(train, feature_columns)
    eligible = valid[valid.doi.isin(eval_dois)].groupby("doi").filter(
        lambda group: group.material.nunique() > shot)
    rows: list[dict[str, object]] = []
    for doi, curves in eligible.groupby("doi", sort=True):
        support_meta, query_meta = select_support(curves, shot, "diverse_support", seed=seed)
        support = test[test.curve_row_id.isin(support_meta.curve_row_id)].copy()
        query = test[test.curve_row_id.isin(query_meta.curve_row_id)].copy()
        support_base = prior.predict(_features(support, feature_columns))
        query_base = prior.predict(_features(query, feature_columns))
        for candidate in candidates:
            metrics = evaluate_episode(
                support, query, feature_columns, support_base=support_base,
                query_base=query_base, scaler=scaler,
                tau=candidate["tau"], blend=candidate["blend"],
                neighbor_k=candidate["neighbor_k"], gate_tau=candidate["gate_tau"])
            if candidate_rows is not None:
                prediction = predict_episode(
                    support, query, feature_columns, support_base=support_base,
                    query_base=query_base, scaler=scaler, tau=candidate["tau"],
                    blend=candidate["blend"], neighbor_k=candidate["neighbor_k"],
                    gate_tau=candidate["gate_tau"])
                candidates_frame = curve_candidates(
                    query, prediction.prediction, prediction.gate,
                    prediction.nearest_distance)
                ranked = rank_candidates(candidates_frame)
                for row in ranked.to_dict("records"):
                    row.update({"seed": seed, "doi": doi, "outer_eval": True,
                                "tau": candidate["tau"], "neighbor_k": candidate["neighbor_k"],
                                "gate_tau": candidate["gate_tau"]})
                    candidate_rows.append(row)
            rows.append({"seed": seed, "doi": doi, "shot": shot,
                         "support_selection": "diverse_support",
                         "tau": candidate["tau"], "blend": candidate["blend"],
                         "neighbor_k": candidate["neighbor_k"],
                         "gate_tau": candidate["gate_tau"],
                         "mae": metrics["mae"], "ridge_mae": metrics["ridge_mae"],
                         "mean_nearest_distance": metrics["mean_nearest_distance"],
                         "fraction_outside_support_temperature": metrics[
                             "fraction_outside_support_temperature"]})
    return pd.DataFrame(rows)


def run_nested_selection(root: Path = PROJECT_ROOT, *, seeds=(11, 22, 33),
                         candidates=CANDIDATES) -> tuple[pd.DataFrame, pd.DataFrame]:
    points, valid = _load_points(root)
    feature_columns = sorted(column for column in points.columns if column.startswith("f_"))
    feature_table = points[["material", *feature_columns]].drop_duplicates("material")
    valid = valid.merge(feature_table, on="material", how="inner")
    inner_rows: list[pd.DataFrame] = []
    outer_rows: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    all_dois = points.doi.unique()
    for seed in seeds:
        outer_train, outer_test = _split_dois(all_dois, seed)
        inner_train, inner_val = _split_dois(np.array(sorted(outer_train)), seed + 10000)
        calibration = _evaluate_split(points, valid, inner_train, inner_val,
                                      feature_columns, tuple(candidates), seed + 10000)
        calibration["outer_seed"] = seed
        inner_rows.append(calibration)
        best = choose_best(calibration)
        selected.append({"seed": seed, "inner_train_dois": len(inner_train),
                         "inner_validation_dois": len(inner_val), **best})
        final = _evaluate_split(points, valid, outer_train, outer_test,
                                feature_columns, (best,), seed,
                                candidate_rows=candidate_rows)
        final["outer_seed"] = seed
        outer_rows.extend(final.to_dict("records"))
        print(f"seed={seed}: selected tau={best['tau']}, blend={best['blend']}", flush=True)
    inner = pd.concat(inner_rows, ignore_index=True)
    outer = pd.DataFrame(outer_rows)
    output = root / "results"
    inner.to_csv(output / "crct_inner_calibration.csv", index=False)
    outer.to_csv(output / "crct_nested_outer_validation.csv", index=False)
    candidate_frame = pd.DataFrame(candidate_rows)
    if not candidate_frame.empty:
        candidate_frame.to_csv(output / "crct_nested_candidate_predictions.csv", index=False)
    selected_frame = pd.DataFrame(selected)
    selected_frame.to_csv(output / "crct_selected_parameters.csv", index=False)
    summary = (outer.groupby(["tau", "blend", "neighbor_k", "gate_tau"], as_index=False, dropna=False)
               .agg(mae_mean=("mae", "mean"), mae_median=("mae", "median"),
                    ridge_mae_mean=("ridge_mae", "mean"), n_episodes=("mae", "size")))
    summary.to_csv(output / "crct_nested_outer_summary.csv", index=False)
    (output / "crct_nested_selection_manifest.json").write_text(json.dumps({
        "seeds": list(seeds), "shot": SHOT, "support_selection": "diverse_support",
        "candidate_grid": list(candidates), "inner_split": "DOI-held-out within outer train",
        "outer_split": "DOI-held-out", "input_sha256": {
            name: sha256_file(path) for name, path in input_paths().items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return selected_frame, summary


if __name__ == "__main__":
    selected, summary = run_nested_selection()
    print(selected.to_string(index=False))
    print(summary.to_string(index=False))
