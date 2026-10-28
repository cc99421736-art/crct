"""Support-curve coverage audit for the independent DCSA project."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.dcsa.data_source import PROJECT_ROOT, input_paths, sha256_file
from src.features.composition_features import parse_composition

SHOT_LEVELS = (1, 2, 3, 5)
MIN_POINTS = 3


@dataclass(frozen=True)
class AuditResult:
    valid_curves: pd.DataFrame
    doi_summary: pd.DataFrame
    counts: dict[str, int]


def canonical_composition(formula: object) -> str | None:
    amounts = parse_composition(formula)
    if not amounts:
        return None
    total = sum(amounts.values())
    if total <= 0:
        return None
    return "|".join(f"{element}:{amounts[element] / total:.8f}"
                    for element in sorted(amounts))


def normalize_doi(values: pd.Series) -> pd.Series:
    doi = values.astype("string").str.strip().str.lower()
    doi = doi.str.replace(r"^https?://(dx\.)?doi\.org/", "", regex=True)
    doi = doi.str.replace(r"^doi:\s*", "", regex=True)
    return doi.where(doi.str.startswith("10."))


def audit_support_coverage(curves: pd.DataFrame, samples: pd.DataFrame,
                           *, min_points: int = MIN_POINTS,
                           target_property: str = "ZT") -> AuditResult:
    if min_points < 1:
        raise ValueError("min_points must be positive")
    required_curves = {"SID", "sample_id", "curve_row_id", "composition_normalized",
                       "property_class", "temperature_numeric", "y_numeric"}
    required_samples = {"SID", "sample_id", "DOI"}
    if missing := required_curves - set(curves.columns):
        raise ValueError(f"curves missing {sorted(missing)}")
    if missing := required_samples - set(samples.columns):
        raise ValueError(f"samples missing {sorted(missing)}")

    counts = {
        "input_rows": int(len(curves)),
        "target_rows": int(curves.property_class.eq(target_property).sum()),
        "excluded_missing_doi_rows": 0,
        "excluded_unparseable_composition_rows": 0,
        "excluded_nonfinite_rows": 0,
        "valid_curve_candidates": 0,
        "excluded_short_curves": 0,
        "valid_curves": 0,
        "valid_dois": 0,
    }
    selected = curves[curves.property_class.eq(target_property)].copy()
    selected["SID"] = selected.SID.astype("string")
    selected["sample_id"] = selected.sample_id.astype("string")
    metadata = samples.copy()
    metadata["SID"] = metadata.SID.astype("string")
    metadata["sample_id"] = metadata.sample_id.astype("string")
    metadata["doi"] = normalize_doi(metadata.DOI)
    metadata = metadata[["SID", "sample_id", "doi"]].drop_duplicates()
    selected = selected.merge(metadata, on=["SID", "sample_id"], how="left")
    missing_doi = selected.doi.isna()
    counts["excluded_missing_doi_rows"] = int(missing_doi.sum())
    selected = selected.loc[~missing_doi].copy()
    temperatures = pd.to_numeric(selected.temperature_numeric, errors="coerce")
    values = pd.to_numeric(selected.y_numeric, errors="coerce")
    nonfinite = temperatures.isna() | values.isna()
    counts["excluded_nonfinite_rows"] = int(nonfinite.sum())
    selected = selected.loc[~nonfinite].copy()
    selected["temperature_numeric"] = temperatures.loc[selected.index].astype(float)
    selected["y_numeric"] = values.loc[selected.index].astype(float)
    selected["material"] = selected.composition_normalized.map(canonical_composition)
    unparseable = selected.material.isna()
    counts["excluded_unparseable_composition_rows"] = int(unparseable.sum())
    selected = selected.loc[~unparseable].copy()
    if selected.empty:
        return AuditResult(pd.DataFrame(), pd.DataFrame(), counts)

    candidates = (selected.groupby(
        ["doi", "SID", "sample_id", "curve_row_id", "material", "composition_normalized"],
        dropna=False, sort=False
    ).agg(n_points=("temperature_numeric", "nunique"),
          temperature_min=("temperature_numeric", "min"),
          temperature_max=("temperature_numeric", "max"))
    .reset_index())
    counts["valid_curve_candidates"] = int(len(candidates))
    short = candidates.n_points < min_points
    counts["excluded_short_curves"] = int(short.sum())
    candidates = candidates.loc[~short].copy()
    if candidates.empty:
        return AuditResult(pd.DataFrame(), pd.DataFrame(), counts)
    candidates["curve_number"] = pd.to_numeric(candidates.curve_row_id, errors="coerce").fillna(np.inf)
    chosen = (candidates.sort_values(
        ["doi", "material", "n_points", "curve_number", "SID", "sample_id"],
        ascending=[True, True, False, True, True, True], kind="mergesort"
    ).drop_duplicates(["doi", "material"], keep="first").drop(columns="curve_number").reset_index(drop=True))
    counts["valid_curves"] = int(len(chosen))

    rows = []
    for doi, group in chosen.groupby("doi", sort=True):
        n = int(group.material.nunique())
        row = {
            "doi": doi, "n_valid_compositions": n, "n_valid_curves": int(len(group)),
            "max_shot": max(n - 1, 0), "n_points_min": int(group.n_points.min()),
            "n_points_median": float(group.n_points.median()), "n_points_max": int(group.n_points.max()),
            "temperature_min": float(group.temperature_min.min()),
            "temperature_max": float(group.temperature_max.max()),
        }
        row.update({f"D_{shot}": int(n >= shot + 1) for shot in SHOT_LEVELS})
        rows.append(row)
    summary = pd.DataFrame(rows)
    counts["valid_dois"] = int(len(summary))
    counts.update({f"D_{shot}": int(summary[f"D_{shot}"].sum()) for shot in SHOT_LEVELS})
    return AuditResult(chosen, summary, counts)


def run_audit(root: Path = PROJECT_ROOT) -> AuditResult:
    paths = input_paths()
    curves = pd.read_parquet(paths["curves"], columns=[
        "curve_row_id", "SID", "sample_id", "composition_normalized",
        "property_class", "temperature_numeric", "y_numeric",
    ])
    samples = pd.read_csv(paths["samples"], usecols=["SID", "sample_id", "DOI"],
                          dtype={"SID": "string", "sample_id": "string"})
    result = audit_support_coverage(curves, samples)
    output = root / "results"
    output.mkdir(parents=True, exist_ok=True)
    result.valid_curves.to_csv(output / "support_curve_valid_curves.csv", index=False)
    result.doi_summary.to_csv(output / "support_curve_doi_summary.csv", index=False)
    manifest = {
        "target_property": "ZT", "min_points": MIN_POINTS, "shot_levels": list(SHOT_LEVELS),
        "counts": result.counts,
        "input_sha256": {name: sha256_file(path) for name, path in paths.items()},
    }
    (output / "support_curve_audit.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = ["# DCSA support-curve audit", "",
             f"Valid DOI count: {result.counts['valid_dois']}",
             f"Valid DOI-composition curves: {result.counts['valid_curves']}", "",
             "| shot | eligible DOI count |", "|---:|---:|"]
    lines.extend(f"| {shot} | {result.counts[f'D_{shot}']} |" for shot in SHOT_LEVELS)
    (output / "support_curve_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run_audit().counts, indent=2, ensure_ascii=False))
