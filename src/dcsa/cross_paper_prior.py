"""Cross-paper system residual prior.

For a query composition, other papers may contain the same (or a nearly
identical) composition.  Their out-of-fold residuals relative to the prior are
a cross-paper estimate of the composition-specific deviation, complementary to
the within-paper transport.

Pipeline per outer seed
-----------------------
1. Fit the prior on the inner training DOIs; build 5-fold out-of-fold (OOF)
   residuals inside the inner training set; build a composition -> residual
   curve lookup; select the blending weight ``w`` on inner validation episodes.
2. Fit the prior on the full outer training set; build OOF residuals and the
   lookup there; apply the frozen ``w`` to outer test episodes.

Blend:  r = (1 - w) * g * r_tr + w * r_cross,  y = y0 + r.
Points without a cross-paper match fall back to the within-paper transport.
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
from src.dcsa.masking_validation import select_support
from src.features.composition_features import parse_composition

SHOT = 5
TAU = 0.25
NEIGHBOR_K = 2
GATE_TAU = 1.0
N_FOLDS = 5
TOP_K = 5
MIN_SIM = 0.99
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
MIN_ELEMENT_COUNT = 20
MAX_ELEMENTS = 45


def _vocabulary(compositions: pd.Series) -> list[str]:
    counts: dict[str, int] = {}
    for formula in compositions.drop_duplicates():
        for element in parse_composition(formula):
            counts[element] = counts.get(element, 0) + 1
    kept = [element for element, count in counts.items() if count >= MIN_ELEMENT_COUNT]
    kept.sort(key=lambda element: (-counts[element], element))
    return kept[:MAX_ELEMENTS]


def _fraction(formula: str, vocabulary: list[str]) -> np.ndarray:
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


def _oof_residuals(points: pd.DataFrame, feature_columns: list[str], seed: int) -> pd.DataFrame:
    dois = np.array(sorted(points.doi.unique()))
    folds = np.array_split(dois, N_FOLDS)
    frames = []
    for fold in folds:
        train = points[~points.doi.isin(fold)]
        valid = points[points.doi.isin(fold)]
        if train.empty or valid.empty:
            continue
        prior = lgb.LGBMRegressor(**PARAMS, random_state=seed)
        prior.fit(_features(train, feature_columns), train.target.to_numpy(float))
        base = prior.predict(_features(valid, feature_columns))
        frames.append(valid.assign(oof_base=base, oof_residual=valid.target.to_numpy(float) - base))
    return pd.concat(frames, ignore_index=True)


def _lookup(oof: pd.DataFrame, vocabulary: list[str]) -> dict[str, object]:
    curves: list[dict[str, object]] = []
    for (composition, doi), group in oof.groupby(["composition_normalized", "doi"], sort=True):
        group = group.sort_values("temperature")
        temperature = group.temperature.to_numpy(float)
        residual = group.oof_residual.to_numpy(float)
        unique, index = np.unique(temperature, return_index=True)
        values = residual[index]
        curves.append({"composition": composition, "doi": doi,
                       "temperature": unique, "residual": values,
                       "centered": values - float(np.mean(values))})
    vectors = np.stack([_fraction(curve["composition"], vocabulary) for curve in curves])
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norm, 1e-12)
    return {"curves": curves, "vectors": vectors,
            "dois": np.array([curve["doi"] for curve in curves], dtype=object)}


def _cross_residual(query: pd.DataFrame, lookup: dict[str, object],
                    query_doi: str, vocabulary: list[str],
                    field: str = "residual") -> np.ndarray:
    result = np.full(len(query), np.nan)
    temperature = query.temperature.to_numpy(float)
    compositions = query.composition_normalized.to_numpy()
    for composition in pd.unique(compositions):
        mask = compositions == composition
        vector = _fraction(composition, vocabulary)
        norm = np.linalg.norm(vector)
        if norm <= 0:
            continue
        unit = vector / norm
        similarity = lookup["vectors"] @ unit
        blocked = lookup["dois"] == query_doi
        similarity[blocked] = -1.0
        order = np.argsort(-similarity)[:TOP_K]
        order = [int(i) for i in order if similarity[i] >= MIN_SIM]
        if not order:
            continue
        weights = similarity[order]
        weights = weights / weights.sum()
        values = np.zeros(int(mask.sum()))
        for weight, index in zip(weights, order):
            curve = lookup["curves"][index]
            values += weight * np.interp(temperature[mask], curve["temperature"], curve[field])
        result[mask] = values
    return result


def _episode_maes(support, query, prior, scaler, feature_columns, lookup,
                  vocabulary, query_doi) -> tuple[dict[str, float], float, float]:
    support_base = prior.predict(_features(support, feature_columns))
    query_base = prior.predict(_features(query, feature_columns))
    prediction = predict_episode(
        support, query, feature_columns, support_base=support_base,
        query_base=query_base, scaler=scaler, tau=TAU, blend=1.0,
        neighbor_k=NEIGHBOR_K, gate_tau=GATE_TAU)
    chem = prediction.chemistry_residual
    cross_raw = _cross_residual(query, lookup, query_doi, vocabulary, "residual")
    cross_center = _cross_residual(query, lookup, query_doi, vocabulary, "centered")
    mask = np.isfinite(cross_raw)
    target = query.target.to_numpy(float)
    mean_chem = float(np.mean(chem))
    chem_center = chem - mean_chem
    maes: dict[str, float] = {}
    for weight in WEIGHTS:
        residual = chem.copy()
        residual[mask] = (1.0 - weight) * chem[mask] + weight * cross_raw[mask]
        maes[f"mae_raw_w{weight}"] = float(np.mean(np.abs(target - (query_base + residual))))
        shape = chem.copy()
        shape[mask] = mean_chem + (1.0 - weight) * chem_center[mask] + weight * cross_center[mask]
        maes[f"mae_shape_w{weight}"] = float(np.mean(np.abs(target - (query_base + shape))))
    return maes, float(mask.mean()), float(np.mean(np.abs(target - (query_base + chem))))


def _run_split(points, valid, train_dois, eval_dois, feature_columns, vocabulary,
               seed, lookup_source: pd.DataFrame) -> pd.DataFrame:
    train = points[points.doi.isin(train_dois)]
    test = points[points.doi.isin(eval_dois)]
    prior = lgb.LGBMRegressor(**PARAMS, random_state=seed)
    prior.fit(_features(train, feature_columns), train.target.to_numpy(float))
    scaler = FeatureScaler.fit(train, feature_columns)
    lookup = _lookup(lookup_source, vocabulary)
    eligible = valid[valid.doi.isin(eval_dois)].groupby("doi").filter(
        lambda group: group.material.nunique() > SHOT)
    rows = []
    for doi, curves in eligible.groupby("doi", sort=True):
        support_meta, query_meta = select_support(curves, SHOT, "diverse_support", seed=seed)
        support = test[test.curve_row_id.isin(support_meta.curve_row_id)]
        query = test[test.curve_row_id.isin(query_meta.curve_row_id)]
        maes, coverage, mae_chem = _episode_maes(
            support, query, prior, scaler, feature_columns, lookup, vocabulary, doi)
        row = {"seed": seed, "doi": doi, "coverage": coverage, "mae_chem": mae_chem}
        row.update(maes)
        rows.append(row)
    return pd.DataFrame(rows)


def run(root: Path = PROJECT_ROOT, *, seeds=(11, 22, 33)) -> pd.DataFrame:
    points, valid = _load_points(root)
    feature_columns = sorted(column for column in points.columns if column.startswith("f_"))
    feature_table = points[["material", *feature_columns]].drop_duplicates("material")
    valid = valid.merge(feature_table, on="material", how="inner")
    vocabulary = _vocabulary(points.composition_normalized)
    fraction_table = (points[["composition_normalized"]].drop_duplicates()
                      .assign(**{f"e_{e}": 0.0 for e in vocabulary}))
    fractions = np.stack([_fraction(f, vocabulary) for f in fraction_table.composition_normalized])
    for i, element in enumerate(vocabulary):
        fraction_table[f"e_{element}"] = fractions[:, i]
    points = points.merge(fraction_table, on="composition_normalized", how="left")
    print(f"vocabulary: {len(vocabulary)}", flush=True)

    all_dois = points.doi.unique()
    metric_columns = [f"mae_{mode}_w{weight}"
                      for mode in ("raw", "shape") for weight in WEIGHTS]
    inner_rows, outer_rows, selections = [], [], []
    for seed in seeds:
        outer_train, outer_test = _split_dois(all_dois, seed)
        inner_train, inner_val = _split_dois(np.array(sorted(outer_train)), seed + 10000)
        inner_oof = _oof_residuals(points[points.doi.isin(inner_train)], feature_columns, seed + 10000)
        inner = _run_split(points, valid, inner_train, inner_val, feature_columns,
                           vocabulary, seed + 10000, inner_oof)
        inner["split"] = "inner"
        inner_rows.append(inner)
        means = {column: float(inner[column].mean()) for column in metric_columns}
        best_column = min(means, key=means.get)
        selections.append({"seed": seed, "selected_column": best_column,
                           **{f"inner_{column}": means[column] for column in metric_columns},
                           "inner_coverage": float(inner.coverage.mean())})
        outer_oof = _oof_residuals(points[points.doi.isin(outer_train)], feature_columns, seed)
        outer = _run_split(points, valid, outer_train, outer_test, feature_columns,
                           vocabulary, seed, outer_oof)
        outer["split"] = "outer"
        outer["selected_column"] = best_column
        outer_rows.append(outer)
        print(f"seed={seed}: selected={best_column} inner_mae={means[best_column]:.6f} "
              f"coverage={inner.coverage.mean():.3f}", flush=True)

    inner_all = pd.concat(inner_rows, ignore_index=True)
    outer_all = pd.concat(outer_rows, ignore_index=True)
    output = root / "results"
    inner_all.to_csv(output / "cross_paper_prior_inner.csv", index=False)
    outer_all.to_csv(output / "cross_paper_prior_outer.csv", index=False)
    pd.DataFrame(selections).to_csv(output / "cross_paper_prior_selection.csv", index=False)

    summary_rows = [{"column": column, "outer_mae": float(outer_all[column].mean())}
                    for column in metric_columns]
    summary = pd.DataFrame(summary_rows)
    selected = outer_all.apply(lambda row: row[row.selected_column], axis=1)
    improvement = outer_all.mae_chem - selected
    rng = np.random.default_rng(20261022)
    by_doi = improvement.groupby(outer_all.doi).mean().to_numpy(float)
    draws = rng.integers(0, len(by_doi), size=(5000, len(by_doi)))
    boot = by_doi[draws].mean(axis=1)
    summary["selected_policy_mae"] = float(selected.mean())
    summary["outer_coverage"] = float(outer_all.coverage.mean())
    summary["selected_improvement"] = float(improvement.mean())
    summary["selected_ci95_low"] = float(np.quantile(boot, .025))
    summary["selected_ci95_high"] = float(np.quantile(boot, .975))
    summary.to_csv(output / "cross_paper_prior_summary.csv", index=False)
    (output / "cross_paper_prior_manifest.json").write_text(json.dumps({
        "seeds": list(seeds), "shot": SHOT, "tau": TAU, "neighbor_k": NEIGHBOR_K,
        "gate_tau": GATE_TAU, "n_folds": N_FOLDS, "top_k": TOP_K, "min_similarity": MIN_SIM,
        "weights": list(WEIGHTS), "vocabulary_size": len(vocabulary),
        "modes": ["raw", "shape"],
        "blend_raw": "r = (1-w) * g * r_tr + w * r_cross",
        "blend_shape": "r = mean(g*r_tr) + (1-w) * center(g*r_tr) + w * center(r_cross)",
        "selection": "mode and weight chosen on inner validation episodes only",
        "outer_coverage": float(outer_all.coverage.mean()),
        "input_sha256": {name: sha256_file(path) for name, path in input_paths().items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.round(6).to_string(index=False))
    return summary


if __name__ == "__main__":
    run()

