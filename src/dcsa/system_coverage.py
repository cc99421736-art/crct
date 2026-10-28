"""Coverage analysis for a cross-paper same-system residual prior.

For every test DOI the script asks two questions:

1. Does another (training) paper share the same element set?  Measured by the
   Jaccard overlap of the per-DOI element unions.
2. Are there training compositions that are chemically almost identical?
   Measured by the cosine similarity of element-fraction vectors, excluding the
   test DOI itself.

No model is fitted; this only reads the processed data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.dcsa.baseline import _load_points, _split_dois
from src.dcsa.data_source import PROJECT_ROOT, input_paths, sha256_file
from src.features.composition_features import parse_composition

MIN_ELEMENT_COUNT = 20
MAX_ELEMENTS = 45
JACCARD_LEVELS = (1.0, 0.8, 0.6)


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


def run(root: Path = PROJECT_ROOT, *, seeds=(11, 22, 33)) -> pd.DataFrame:
    points, valid = _load_points(root)
    vocabulary = _vocabulary(points.composition_normalized)
    compositions = points.composition_normalized.drop_duplicates().reset_index(drop=True)
    fraction = np.stack([_fraction(f, vocabulary) for f in compositions])
    norm = np.linalg.norm(fraction, axis=1, keepdims=True)
    fraction_unit = fraction / np.maximum(norm, 1e-12)
    composition_index = {f: i for i, f in enumerate(compositions)}

    doi_elements = (points.groupby("doi").composition_normalized
                    .apply(lambda values: frozenset().union(
                        *[frozenset(parse_composition(v)) for v in values.drop_duplicates()])))
    doi_compositions = points.groupby("doi").composition_normalized.apply(
        lambda values: list(values.drop_duplicates()))
    all_dois = points.doi.unique()
    rows = []
    for seed in seeds:
        train_dois, test_dois = _split_dois(all_dois, seed)
        train_composition_ids = sorted({composition_index[f]
                                        for doi in train_dois
                                        for f in doi_compositions[doi]})
        train_unit = fraction_unit[train_composition_ids]
        train_unions = [(doi, doi_elements[doi]) for doi in train_dois]
        for doi in sorted(test_dois):
            test_union = doi_elements[doi]
            jaccards = [len(test_union & union) / len(test_union | union)
                        for _, union in train_unions]
            max_jaccard = float(np.max(jaccards)) if jaccards else 0.0
            exact_union = any(test_union == union for _, union in train_unions)
            test_ids = [composition_index[f] for f in doi_compositions[doi]]
            similarity = fraction_unit[test_ids] @ train_unit.T
            max_per_composition = similarity.max(axis=1)
            rows.append({
                "seed": seed, "doi": doi, "n_compositions": len(test_ids),
                "max_jaccard": max_jaccard,
                "exact_element_set": bool(exact_union),
                "median_max_cosine": float(np.median(max_per_composition)),
                "frac_cosine_099": float(np.mean(max_per_composition >= 0.99)),
                "frac_cosine_095": float(np.mean(max_per_composition >= 0.95)),
            })
        print(f"seed={seed}: test dois={len(test_dois)}", flush=True)

    frame = pd.DataFrame(rows)
    summary = {
        "n_test_doi_observations": int(len(frame)),
        "share_exact_element_set": float(frame.exact_element_set.mean()),
        "share_max_jaccard_ge_1.0": float((frame.max_jaccard >= 0.999).mean()),
        "share_max_jaccard_ge_0.8": float((frame.max_jaccard >= 0.8).mean()),
        "share_max_jaccard_ge_0.6": float((frame.max_jaccard >= 0.6).mean()),
        "median_max_jaccard": float(frame.max_jaccard.median()),
        "share_doi_with_any_cosine_099": float((frame.frac_cosine_099 > 0).mean()),
        "share_doi_with_half_cosine_099": float((frame.frac_cosine_099 >= 0.5).mean()),
        "median_fraction_of_compositions_with_cosine_099": float(frame.frac_cosine_099.median()),
        "input_sha256": {name: sha256_file(path) for name, path in input_paths().items()},
    }
    output = root / "results"
    frame.to_csv(output / "system_coverage.csv", index=False)
    (output / "system_coverage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return frame


if __name__ == "__main__":
    run()
