"""Experiment: does adding element-fraction features to the prior help?

The current prior sees 11 element-property statistics and temperature.  This
script refits the prior with an additional element-fraction block (45 elements)
and compares:

  zero-shot MAE of the base prior and the fraction prior
  CRCT MAE on top of each prior, with the *same* transport distance

Keeping the transport distance fixed isolates the effect of the prior.
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
from src.features.composition_features import parse_composition

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


def _fraction_frame(compositions: pd.Series, vocabulary: list[str]) -> pd.DataFrame:
    index = {element: i for i, element in enumerate(vocabulary)}
    rows = []
    for formula in compositions.drop_duplicates():
        amounts = parse_composition(formula)
        total = sum(amounts.values())
        vector = np.zeros(len(vocabulary))
        if total > 0:
            for element, amount in amounts.items():
                if element in index:
                    vector[index[element]] = amount / total
        row = {f"e_{e}": vector[i] for i, e in enumerate(vocabulary)}
        row["composition_normalized"] = formula
        rows.append(row)
    return pd.DataFrame(rows)


def run(root: Path = PROJECT_ROOT, *, seeds=(11, 22, 33)) -> pd.DataFrame:
    points, valid = _load_points(root)
    base_features = sorted(column for column in points.columns if column.startswith("f_"))
    feature_table = points[["material", *base_features]].drop_duplicates("material")
    valid = valid.merge(feature_table, on="material", how="inner")
    vocabulary = _fraction_vocabulary(points.composition_normalized)
    fractions = _fraction_frame(points.composition_normalized, vocabulary)
    points = points.merge(fractions, on="composition_normalized", how="left")
    fraction_features = [f"e_{e}" for e in vocabulary]
    print(f"fraction vocabulary: {len(vocabulary)} elements", flush=True)

    all_dois = points.doi.unique()
    rows: list[dict[str, object]] = []
    for seed in seeds:
        train_dois, test_dois = _split_dois(all_dois, seed)
        train = points[points.doi.isin(train_dois)]
        test = points[points.doi.isin(test_dois)]
        eligible = valid[valid.doi.isin(test_dois)].groupby("doi").filter(
            lambda group: group.material.nunique() > SHOT)
        priors = {}
        scalers = {}
        for name, columns in (("base", base_features),
                              ("fraction", base_features + fraction_features)):
            prior = lgb.LGBMRegressor(**PARAMS, random_state=seed)
            prior.fit(_features(train, columns), train.target.to_numpy(float))
            priors[name] = (prior, columns)
            scalers[name] = FeatureScaler.fit(train, base_features)
            print(f"seed={seed} prior={name} fitted", flush=True)
        for doi, curves in eligible.groupby("doi", sort=True):
            support_meta, query_meta = select_support(curves, SHOT, "diverse_support", seed=seed)
            support = test[test.curve_row_id.isin(support_meta.curve_row_id)].copy()
            query = test[test.curve_row_id.isin(query_meta.curve_row_id)].copy()
            target = query.target.to_numpy(float)
            row = {"seed": seed, "doi": doi}
            for name, (prior, columns) in priors.items():
                support_base = prior.predict(_features(support, columns))
                query_base = prior.predict(_features(query, columns))
                row[f"zero_shot_{name}"] = float(np.mean(np.abs(target - query_base)))
                metrics = evaluate_episode(
                    support, query, base_features, support_base=support_base,
                    query_base=query_base, scaler=scalers["base"], tau=TAU,
                    blend=1.0, neighbor_k=NEIGHBOR_K, gate_tau=GATE_TAU)
                row[f"crct_{name}"] = metrics["mae"]
                row[f"crct_{name}_ridge"] = metrics["ridge_mae"]
            rows.append(row)
        print(f"seed={seed}: episodes={len(rows)}", flush=True)

    detail = pd.DataFrame(rows)
    output = root / "results"
    detail.to_csv(output / "prior_fraction_experiment.csv", index=False)
    metrics = [c for c in detail.columns if c.startswith(("zero_shot_", "crct_"))]
    summary = detail[metrics].mean().to_frame("mae_mean")
    summary["relative_to_base_pct"] = summary.apply(
        lambda r: 100.0 * (1.0 - r.mae_mean / summary.loc[
            "crct_base" if r.name.startswith("crct") else "zero_shot_base", "mae_mean"]), axis=1)
    summary.to_csv(output / "prior_fraction_experiment_summary.csv")
    (output / "prior_fraction_experiment_manifest.json").write_text(json.dumps({
        "seeds": list(seeds), "shot": SHOT, "tau": TAU, "neighbor_k": NEIGHBOR_K,
        "gate_tau": GATE_TAU, "fraction_vocabulary_size": len(vocabulary),
        "transport_distance": "base 11-dim feature space in both priors",
        "input_sha256": {name: sha256_file(path) for name, path in input_paths().items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.round(6).to_string())
    return summary


if __name__ == "__main__":
    run()
