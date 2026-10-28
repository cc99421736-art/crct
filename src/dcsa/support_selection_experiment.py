"""Experiment: support-set design for within-paper few-shot transfer.

``diverse_support`` maximises chemical coverage, which tends to pick the two
ends of a doping series and to skip the composition where the performance peaks.
Because the transport is a convex combination, it cannot reproduce a query peak
that is higher than every support peak.

This script compares support-selection strategies that use only composition
features and the *predicted* peak (no measured query labels):

  diverse     current k-center strategy
  peak1       predicted-peak maximum + 4 diverse
  peak2       two predicted-peak maxima + 3 diverse
  spread      quantiles of the predicted peak (max, 75, 50, 25, min)
  random      reference
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
from src.dcsa.masking_validation import _diverse_support, select_support

SHOT = 5
TAU = 0.25
NEIGHBOR_K = 2
GATE_TAU = 1.0


def _curve_table(curves: pd.DataFrame, prior, feature_columns: list[str]) -> pd.DataFrame:
    distinct = (curves.sort_values(["material", "curve_row_id"], kind="mergesort")
                .drop_duplicates("material").reset_index(drop=True))
    base_all = prior.predict(_features(curves, feature_columns))
    peaks = (curves.assign(_base=base_all)
             .groupby("material")._base.max())
    distinct = distinct.copy()
    distinct["predicted_peak"] = distinct.material.map(peaks).to_numpy(float)
    return distinct


def _select(curves: pd.DataFrame, distinct: pd.DataFrame, strategy: str,
            seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if strategy == "diverse":
        return select_support(curves, SHOT, "diverse_support", seed=seed)
    if strategy == "random":
        return select_support(curves, SHOT, "random", seed=seed)
    if strategy in {"peak1", "peak2"}:
        n_peak = 1 if strategy == "peak1" else 2
        ordered = distinct.sort_values(["predicted_peak", "material"],
                                       ascending=[False, True], kind="mergesort")
        chosen = list(ordered.head(n_peak).index)
        rest = distinct.drop(index=chosen)
        if len(rest) >= SHOT - n_peak:
            extra = _diverse_support(rest, SHOT - n_peak)
            chosen += list(extra.index)
        else:
            chosen += list(rest.index)
        support = distinct.loc[chosen].reset_index(drop=True)
    elif strategy == "spread":
        order = distinct.sort_values(["predicted_peak", "material"], kind="mergesort")
        positions = np.linspace(0, len(order) - 1, SHOT).round().astype(int)
        support = order.iloc[sorted(set(positions))].reset_index(drop=True)
    else:
        raise ValueError(strategy)
    query = distinct.loc[~distinct.material.isin(set(support.material))].reset_index(drop=True)
    return support, query


def run(root: Path = PROJECT_ROOT, *, seeds=(11, 22, 33)) -> pd.DataFrame:
    points, valid = _load_points(root)
    feature_columns = sorted(column for column in points.columns if column.startswith("f_"))
    feature_table = points[["material", *feature_columns]].drop_duplicates("material")
    valid = valid.merge(feature_table, on="material", how="inner")
    all_dois = points.doi.unique()
    strategies = ("diverse", "peak1", "peak2", "spread", "random")
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
            paper_points = test[test.curve_row_id.isin(curves.curve_row_id)]
            distinct = _curve_table(paper_points, prior, feature_columns)
            row = {"seed": seed, "doi": doi}
            for strategy in strategies:
                support_meta, query_meta = _select(paper_points, distinct, strategy, seed)
                support = test[test.curve_row_id.isin(support_meta.curve_row_id)].copy()
                query = test[test.curve_row_id.isin(query_meta.curve_row_id)].copy()
                if support.empty or query.empty:
                    continue
                support_base = prior.predict(_features(support, feature_columns))
                query_base = prior.predict(_features(query, feature_columns))
                prediction = predict_episode(
                    support, query, feature_columns, support_base=support_base,
                    query_base=query_base, scaler=scaler, tau=TAU, blend=1.0,
                    neighbor_k=NEIGHBOR_K, gate_tau=GATE_TAU)
                target = query.target.to_numpy(float)
                row[f"mae_{strategy}"] = float(np.mean(np.abs(
                    target - prediction.variants["chemistry_only"])))
            rows.append(row)
        print(f"seed={seed}: episodes={len(rows)}", flush=True)

    detail = pd.DataFrame(rows)
    output = root / "results"
    detail.to_csv(output / "support_selection_experiment.csv", index=False)
    metrics = [c for c in detail.columns if c.startswith("mae_")]
    summary = detail[metrics].mean().to_frame("mae_mean")
    summary["relative_to_diverse_pct"] = 100.0 * (
        1.0 - summary.mae_mean / summary.loc["mae_diverse", "mae_mean"])
    summary.to_csv(output / "support_selection_experiment_summary.csv")
    (output / "support_selection_experiment_manifest.json").write_text(json.dumps({
        "seeds": list(seeds), "shot": SHOT, "tau": TAU, "neighbor_k": NEIGHBOR_K,
        "gate_tau": GATE_TAU, "strategies": list(strategies),
        "note": "support selection uses composition features and predicted peak only; no measured query labels",
        "input_sha256": {name: sha256_file(path) for name, path in input_paths().items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.round(6).to_string())
    return summary


if __name__ == "__main__":
    run()
