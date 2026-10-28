"""Risk-aware candidate ranking: dual scores and retrospective audit.

Two ranking scores are compared on the same masked episodes:

    S_peak = predicted peak ZT
    S_risk = predicted peak ZT / (1 + R),   R = z(d_min) + z(f_outside)

``R`` is the unified pre-measurement risk score used by the selective
prediction layer.  The old ``peak * gate / (1 + distance)`` score is kept as an
extra reference column.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.dcsa.active_design import add_risk_aware_score, curve_candidates
from src.dcsa.baseline import PARAMS, _features, _load_points, _split_dois
from src.dcsa.crct import FeatureScaler, predict_episode
from src.dcsa.data_source import PROJECT_ROOT, input_paths, sha256_file
from src.dcsa.masking_validation import select_support
from src.dcsa import ranking_audit

SHOT = 5
TAU = 0.25
NEIGHBOR_K = 2
GATE_TAU = 1.0
METHODS = ("risk_aware_peak", "predicted_peak", "nearest_distance", "random",
           "gate_distance_score")


def build_candidates(root: Path = PROJECT_ROOT, *, seeds=(11, 22, 33),
                     family: str = "chemistry_only") -> pd.DataFrame:
    points, valid = _load_points(root)
    feature_columns = sorted(column for column in points.columns if column.startswith("f_"))
    feature_table = points[["material", *feature_columns]].drop_duplicates("material")
    valid = valid.merge(feature_table, on="material", how="inner")
    all_dois = points.doi.unique()
    frames = []
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
            support_meta, query_meta = select_support(curves, SHOT, "diverse_support", seed=seed)
            support = test[test.curve_row_id.isin(support_meta.curve_row_id)].copy()
            query = test[test.curve_row_id.isin(query_meta.curve_row_id)].copy()
            support_base = prior.predict(_features(support, feature_columns))
            query_base = prior.predict(_features(query, feature_columns))
            prediction = predict_episode(
                support, query, feature_columns, support_base=support_base,
                query_base=query_base, scaler=scaler, tau=TAU, blend=1.0,
                neighbor_k=NEIGHBOR_K, gate_tau=GATE_TAU)
            values = prediction.variants[family] if family in prediction.variants else prediction.prediction
            candidates = curve_candidates(query, values, prediction.gate,
                                          prediction.nearest_distance, support=support)
            candidates = add_risk_aware_score(candidates)
            candidates["screening_score"] = (
                candidates.predicted_peak_zt * candidates.gate
                / (1.0 + candidates.nearest_distance))
            candidates["seed"] = seed
            candidates["doi"] = doi
            candidates["family"] = family
            frames.append(candidates)
        print(f"seed={seed}: episodes={sum(len(f) for f in frames)}", flush=True)
    return pd.concat(frames, ignore_index=True)


def run(root: Path = PROJECT_ROOT, *, seeds=(11, 22, 33), family: str = "chemistry_only") -> pd.DataFrame:
    candidates = build_candidates(root, seeds=seeds, family=family)
    output = root / "results"
    candidates.to_csv(output / "crct_risk_candidate_predictions.csv", index=False)
    summary = ranking_audit.run(
        root, input_name="crct_risk_candidate_predictions.csv",
        output_prefix="ranking_risk_audit", methods=METHODS)
    (output / "ranking_risk_audit_manifest.json").write_text(json.dumps({
        "source": "crct_risk_candidate_predictions.csv",
        "family": family, "methods": list(METHODS), "cluster": "doi",
        "risk_score": "z(nearest_distance) + z(fraction_outside_support_temperature)",
        "risk_index": "risk_score shifted by its minimum in the candidate pool",
        "risk_aware_score": "predicted_peak_zt / (1 + risk_index)",
        "input_sha256": {name: sha256_file(path) for name, path in input_paths().items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run()
