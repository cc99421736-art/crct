"""Experiment: does the composition representation limit the transport distance?

The current feature vector has 11 element-property statistics (weighted means,
one range, one std, total valence electrons) but *no element fractions*.  Two
compositions of the same element set with different stoichiometry can therefore
be almost indistinguishable.

This script compares the transport error under four distance definitions:

  feature    current 11-dim standardized feature space (baseline)
  fraction   standardized element-fraction vector
  combined   feature + fraction blocks, each standardized
  cosine     cosine distance on the fraction vector
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.dcsa.baseline import PARAMS, _features, _load_points, _split_dois
from src.dcsa.crct import FeatureScaler, _curve_residuals
from src.dcsa.data_source import PROJECT_ROOT, input_paths, sha256_file
from src.features.composition_features import parse_composition
from src.dcsa.masking_validation import select_support

SHOT = 5
TAU = 0.25
NEIGHBOR_K = 2
GATE_TAU = 1.0
MIN_ELEMENT_COUNT = 20
MAX_ELEMENTS = 45


def _fraction_vocabulary(compositions: pd.Series) -> list[str]:
    counts: dict[str, int] = {}
    for formula in compositions.drop_duplicates():
        for element in parse_composition(formula):
            counts[element] = counts.get(element, 0) + 1
    kept = [element for element, count in counts.items() if count >= MIN_ELEMENT_COUNT]
    kept.sort(key=lambda element: (-counts[element], element))
    return kept[:MAX_ELEMENTS]


def _fraction_vector(formula: str, vocabulary: list[str]) -> np.ndarray:
    amounts = parse_composition(formula)
    total = sum(amounts.values())
    vector = np.zeros(len(vocabulary))
    if total <= 0:
        return vector
    index = {element: i for i, element in enumerate(vocabulary)}
    for element, amount in amounts.items():
        if element in index:
            vector[index[element]] = amount / total
    return vector


def _weights(distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, k = distances.shape
    kk = min(NEIGHBOR_K, k)
    nearest = np.argpartition(distances, kth=kk - 1, axis=1)[:, :kk]
    mask = np.ones_like(distances, dtype=bool)
    mask[np.arange(n)[:, None], nearest] = False
    logits = -0.5 * (distances / TAU) ** 2
    logits[mask] = -np.inf
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights, distances.min(axis=1)


def run(root: Path = PROJECT_ROOT, *, seeds=(11, 22, 33)) -> pd.DataFrame:
    points, valid = _load_points(root)
    feature_columns = sorted(column for column in points.columns if column.startswith("f_"))
    feature_table = points[["material", *feature_columns]].drop_duplicates("material")
    valid = valid.merge(feature_table, on="material", how="inner")
    vocabulary = _fraction_vocabulary(points.composition_normalized)
    fractions = pd.DataFrame({
        "composition_normalized": points.composition_normalized.drop_duplicates(),
    }).reset_index(drop=True)
    matrix = np.stack([_fraction_vector(f, vocabulary) for f in fractions.composition_normalized])
    fraction_table = pd.DataFrame(matrix, columns=[f"e_{e}" for e in vocabulary])
    fraction_table["composition_normalized"] = fractions.composition_normalized.to_numpy()
    points = points.merge(fraction_table, on="composition_normalized", how="left")
    fraction_columns = [f"e_{e}" for e in vocabulary]
    print(f"fraction vocabulary: {len(vocabulary)} elements", flush=True)

    all_dois = points.doi.unique()
    rows: list[dict[str, object]] = []
    for seed in seeds:
        train_dois, test_dois = _split_dois(all_dois, seed)
        train = points[points.doi.isin(train_dois)]
        test = points[points.doi.isin(test_dois)]
        prior = lgb.LGBMRegressor(**PARAMS, random_state=seed)
        prior.fit(_features(train, feature_columns), train.target.to_numpy(float))
        feature_scaler = FeatureScaler.fit(train, feature_columns)
        fraction_scaler = FeatureScaler.fit(train, fraction_columns)
        eligible = valid[valid.doi.isin(test_dois)].groupby("doi").filter(
            lambda group: group.material.nunique() > SHOT)
        for doi, curves in eligible.groupby("doi", sort=True):
            support_meta, query_meta = select_support(curves, SHOT, "diverse_support", seed=seed)
            support = test[test.curve_row_id.isin(support_meta.curve_row_id)].copy()
            query = test[test.curve_row_id.isin(query_meta.curve_row_id)].copy()
            support_base = prior.predict(_features(support, feature_columns))
            query_base = prior.predict(_features(query, feature_columns))
            target = query.target.to_numpy(float)
            compositions, s_temps, s_res = _curve_residuals(support, support_base, feature_columns)
            q_temp = query.temperature.to_numpy(float)
            residual_matrix = np.column_stack([
                np.interp(q_temp, t, r) for t, r in zip(s_temps, s_res)])
            feature_support = feature_scaler.transform(np.stack(compositions))
            feature_query = feature_scaler.transform(query[feature_columns].to_numpy(float))
            support_curves = support.drop_duplicates("curve_row_id")[fraction_columns].to_numpy(float)
            if len(support_curves) != len(compositions):
                raise ValueError("support curve order does not match residual order")
            fraction_support = fraction_scaler.transform(support_curves)
            fraction_query = fraction_scaler.transform(query[fraction_columns].to_numpy(float))

            def _distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
                return np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)

            metrics = {
                "feature": _distances(feature_query, feature_support),
                "fraction": _distances(fraction_query, fraction_support),
                "combined": _distances(
                    np.hstack([feature_query, fraction_query]),
                    np.hstack([feature_support, fraction_support])),
            }
            norm_q = np.linalg.norm(fraction_query, axis=1, keepdims=True)
            norm_s = np.linalg.norm(fraction_support, axis=1, keepdims=True)
            cosine = 1.0 - np.clip((fraction_query @ fraction_support.T)
                                   / np.maximum(norm_q * norm_s.T, 1e-12), -1.0, 1.0)
            metrics["cosine"] = cosine

            row = {"seed": seed, "doi": doi}
            for name, distances in metrics.items():
                weights, nearest = _weights(distances)
                gate = np.exp(-0.5 * (nearest / GATE_TAU) ** 2)
                transported = np.sum(weights * residual_matrix, axis=1) * gate
                row[f"mae_{name}"] = float(np.mean(np.abs(
                    target - (query_base + transported))))
            rows.append(row)
        print(f"seed={seed}: episodes={len(rows)}", flush=True)

    detail = pd.DataFrame(rows)
    output = root / "results"
    detail.to_csv(output / "distance_experiment.csv", index=False)
    metrics = [c for c in detail.columns if c.startswith("mae_")]
    summary = detail[metrics].mean().sort_values().to_frame("mae_mean")
    summary["relative_to_feature_pct"] = 100.0 * (
        1.0 - summary.mae_mean / summary.loc["mae_feature", "mae_mean"])
    summary.to_csv(output / "distance_experiment_summary.csv")
    (output / "distance_experiment_manifest.json").write_text(json.dumps({
        "seeds": list(seeds), "shot": SHOT, "tau": TAU, "neighbor_k": NEIGHBOR_K,
        "gate_tau": GATE_TAU, "fraction_vocabulary_size": len(vocabulary),
        "min_element_count": MIN_ELEMENT_COUNT, "max_elements": MAX_ELEMENTS,
        "input_sha256": {name: sha256_file(path) for name, path in input_paths().items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.round(6).to_string())
    return summary


if __name__ == "__main__":
    run()
