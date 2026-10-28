"""Chronological DOI holdout for CRCT."""
from __future__ import annotations
import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from src.dcsa.baseline import PARAMS, _features, _load_points
from src.dcsa.crct import FeatureScaler, evaluate_episode
from src.dcsa.data_source import PROJECT_ROOT
from src.dcsa.masking_validation import SHOT, select_support


def run_temporal_validation(root: Path = PROJECT_ROOT, *, shot: int = SHOT,
                            train_fraction: float = 0.8) -> pd.DataFrame:
    points, valid = _load_points(root)
    points["doi"] = points["doi"].astype(str)
    valid["doi"] = valid["doi"].astype(str)
    samples = pd.read_csv(root / "data/processed/starrydata2_samples_clean.csv",
                          usecols=["DOI", "created_at"])
    dates = (samples.dropna(subset=["DOI", "created_at"])
             .assign(doi=lambda x: x.DOI.astype(str),
                     date=lambda x: pd.to_numeric(x.created_at.astype(str).str.extract(r"(20\d{2})")[0], errors="coerce"))
             .dropna(subset=["date"]).groupby("doi").date.min())
    dates.index = dates.index.astype(str)
    date_map = dates.to_dict()
    doi_dates = pd.DataFrame({"doi": points.doi.unique()})
    doi_dates["date"] = doi_dates["doi"].astype(str).map(date_map)
    doi_dates = doi_dates.dropna()
    doi_dates = doi_dates.sort_values(["date", "doi"], kind="mergesort")
    cut = max(1, int(np.floor(train_fraction * len(doi_dates))))
    train_dois = set(doi_dates.iloc[:cut].doi)
    test_dois = set(doi_dates.iloc[cut:].doi)
    feature_columns = sorted(c for c in points.columns if c.startswith("f_"))
    train = points[points.doi.isin(train_dois)]
    test = points[points.doi.isin(test_dois)]
    valid_test = valid[valid.doi.isin(test_dois)]
    feature_table = points[["material", *feature_columns]].drop_duplicates("material")
    valid_test = valid_test.merge(feature_table, on="material", how="inner")
    prior = lgb.LGBMRegressor(**PARAMS, random_state=20261003)
    prior.fit(_features(train, feature_columns), train.target.to_numpy(float))
    scaler = FeatureScaler.fit(train, feature_columns)
    rows = []
    for doi, curves in valid_test.groupby("doi", sort=True):
        if curves.material.nunique() <= shot:
            continue
        support_meta, query_meta = select_support(curves, shot, "diverse_support", seed=20261003)
        support = test[test.curve_row_id.isin(support_meta.curve_row_id)].copy()
        query = test[test.curve_row_id.isin(query_meta.curve_row_id)].copy()
        sb = prior.predict(_features(support, feature_columns)); qb = prior.predict(_features(query, feature_columns))
        metrics = evaluate_episode(support, query, feature_columns, support_base=sb, query_base=qb,
                                   scaler=scaler, tau=0.25, blend=1.0, neighbor_k=2, gate_tau=1.0)
        rows.append({"doi": doi, "year": int(dates.loc[str(doi)]), **metrics})
    result = pd.DataFrame(rows)
    out = root / "results"
    result.to_csv(out / "temporal_validation.csv", index=False)
    summary = pd.DataFrame([{"train_dois": len(train_dois), "test_dois": len(test_dois),
                             "n_episodes": len(result), "crct_mae": result.mae.mean(),
                             "ridge_mae": result.ridge_mae.mean()}])
    summary.to_csv(out / "temporal_validation_summary.csv", index=False)
    (out / "temporal_validation_manifest.json").write_text(json.dumps({
        "split": "chronological DOI holdout", "train_fraction": train_fraction,
        "shot": shot, "tau": 0.25, "neighbor_k": 2, "gate_tau": 1.0,
        "date_field": "created_at", "n_dated_dois": len(doi_dates),
    }, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(run_temporal_validation().to_string(index=False))
