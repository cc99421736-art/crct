"""Fixed-subset shot curve: same papers, varying number of support curves.

The 1/2/3/5 comparison uses a different eligible DOI set at every shot level,
and the 8/10-shot papers are intrinsically harder (their zero-shot error is
higher).  This script fixes the evaluation set to the DOIs with at least
``MIN_COMPOSITIONS`` valid compositions and then varies the number of support
curves inside that same set, so the only changing variable is the measurement
budget.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.dcsa.baseline import PARAMS, _features, _load_points, _split_dois
from src.dcsa.crct import FeatureScaler, evaluate_episode
from src.dcsa.data_source import PROJECT_ROOT, input_paths, sha256_file
from src.dcsa.masking_validation import select_support

MIN_COMPOSITIONS = 9
SHOTS = (1, 2, 3, 5, 8)
TAU = 0.25
NEIGHBOR_K = 2
GATE_TAU = 1.0


def _bootstrap(frame: pd.DataFrame, baseline: str, method: str,
               n_boot: int = 5000, seed: int = 20261021) -> dict[str, float]:
    improvement = (frame[baseline] - frame[method]).astype(float)
    by_doi = improvement.groupby(frame.doi).mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(by_doi), size=(n_boot, len(by_doi)))
    boot = by_doi[draws].mean(axis=1)
    return {"n_dois": int(len(by_doi)), "mean_improvement": float(improvement.mean()),
            "ci95_low": float(np.quantile(boot, .025)),
            "ci95_high": float(np.quantile(boot, .975)),
            "win_rate": float(np.mean(improvement > 0))}


def run(root: Path = PROJECT_ROOT, *, seeds=(11, 22, 33)) -> pd.DataFrame:
    points, valid = _load_points(root)
    feature_columns = sorted(column for column in points.columns if column.startswith("f_"))
    feature_table = points[["material", *feature_columns]].drop_duplicates("material")
    valid = valid.merge(feature_table, on="material", how="inner")
    doi_summary = pd.read_csv(root / "results" / "support_curve_doi_summary.csv")
    fixed_dois = set(doi_summary.loc[
        doi_summary.n_valid_compositions >= MIN_COMPOSITIONS, "doi"].astype(str))
    all_dois = points.doi.unique()
    rows: list[dict[str, object]] = []

    for seed in seeds:
        train_dois, test_dois = _split_dois(all_dois, seed)
        train = points[points.doi.isin(train_dois)]
        test = points[points.doi.isin(test_dois)]
        prior = lgb.LGBMRegressor(**PARAMS, random_state=seed)
        prior.fit(_features(train, feature_columns), train.target.to_numpy(float))
        scaler = FeatureScaler.fit(train, feature_columns)
        eligible = valid[valid.doi.isin(test_dois & fixed_dois)].groupby("doi").filter(
            lambda group: group.material.nunique() >= MIN_COMPOSITIONS)
        for shot in SHOTS:
            for doi, curves in eligible.groupby("doi", sort=True):
                if curves.material.nunique() <= shot:
                    continue
                support_meta, query_meta = select_support(curves, shot, "diverse_support", seed=seed)
                support = test[test.curve_row_id.isin(support_meta.curve_row_id)].copy()
                query = test[test.curve_row_id.isin(query_meta.curve_row_id)].copy()
                support_base = prior.predict(_features(support, feature_columns))
                query_base = prior.predict(_features(query, feature_columns))
                metrics = evaluate_episode(
                    support, query, feature_columns, support_base=support_base,
                    query_base=query_base, scaler=scaler, tau=TAU, blend=1.0,
                    neighbor_k=NEIGHBOR_K, gate_tau=GATE_TAU)
                rows.append({"seed": seed, "doi": doi, "shot": shot,
                             "mae_zero_shot": float(np.mean(
                                 np.abs(query.target.to_numpy(float) - query_base))),
                             "mae_ridge": metrics["ridge_mae"],
                             "mae_chem": metrics["mae"],
                             "n_query_curves": int(query.curve_row_id.nunique())})
        print(f"seed={seed}: rows={len(rows)}", flush=True)

    detail = pd.DataFrame(rows)
    output = root / "results"
    detail.to_csv(output / "shot_fixed_subset.csv", index=False)
    summary = []
    for shot, group in detail.groupby("shot", sort=True):
        stats = _bootstrap(group, "mae_ridge", "mae_chem")
        summary.append({"shot": shot, "n_episodes": int(len(group)),
                        "n_dois": int(group.doi.nunique()),
                        "zero_shot": float(group.mae_zero_shot.mean()),
                        "ridge": float(group.mae_ridge.mean()),
                        "chem": float(group.mae_chem.mean()),
                        "reduction_pct": 100.0 * (1 - group.mae_chem.mean() / group.mae_ridge.mean()),
                        **{k: v for k, v in stats.items() if k != "mean_improvement"}})
    table = pd.DataFrame(summary)
    table.to_csv(output / "shot_fixed_subset_summary.csv", index=False)
    (output / "shot_fixed_subset_manifest.json").write_text(json.dumps({
        "seeds": list(seeds), "shots": list(SHOTS), "min_compositions": MIN_COMPOSITIONS,
        "fixed_dois": len(fixed_dois), "tau": TAU, "neighbor_k": NEIGHBOR_K,
        "gate_tau": GATE_TAU, "support_selection": "diverse_support",
        "note": "same DOI set at every shot level; only the number of support curves changes",
        "input_sha256": {name: sha256_file(path) for name, path in input_paths().items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(table.round(4).to_string(index=False))
    return table


if __name__ == "__main__":
    run()
