"""Paired uncertainty statistics for CRCT versus temperature-Ridge."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.dcsa.data_source import PROJECT_ROOT


def run_crct_stats(root: Path = PROJECT_ROOT, *, n_boot: int = 2000,
                   seed: int = 20261002,
                   support_strategy: str = "random") -> pd.DataFrame:
    output = root / "results"
    baseline = pd.read_csv(output / "masking_validation.csv")
    suffix = "" if support_strategy == "random" else f"_{support_strategy}"
    crct = pd.read_csv(output / f"crct{suffix}_masked_validation.csv")
    base = baseline.query(f"shot == 5 and strategy == '{support_strategy}' and model == 'temperature_ridge'")
    method = crct.query("shot == 5 and method == 'crct'")
    base = base[["seed", "doi", "mae"]].rename(columns={"mae": "ridge_mae"})
    method = method[["seed", "doi", "mae", "distance_group"]].rename(columns={"mae": "crct_mae"})
    paired = base.merge(method, on=["seed", "doi"], how="inner", validate="one_to_one")
    if len(paired) != len(base) or len(paired) != len(method):
        raise ValueError("CRCT and baseline episodes are not one-to-one")
    paired["improvement"] = paired.ridge_mae - paired.crct_mae
    paired["crct_win"] = paired.improvement > 0
    paired.to_csv(output / f"crct{suffix}_paired_improvement.csv", index=False)
    rows = []
    rng = np.random.default_rng(seed)
    for population, frame in [("overall", paired), *paired.groupby("distance_group", sort=True)]:
        if isinstance(population, tuple):
            population, frame = population
        doi_means = frame.groupby("doi").improvement.mean().to_numpy(float)
        draws = rng.integers(0, len(doi_means), size=(n_boot, len(doi_means)))
        boot = doi_means[draws].mean(axis=1)
        rows.append({"population": population, "n_dois": int(frame.doi.nunique()),
                     "n_episodes": int(len(frame)),
                     "ridge_mae": float(frame.ridge_mae.mean()),
                     "crct_mae": float(frame.crct_mae.mean()),
                     "mean_improvement": float(frame.improvement.mean()),
                     "win_rate": float(frame.crct_win.mean()),
                     "ci95_low": float(np.quantile(boot, .025)),
                     "ci95_high": float(np.quantile(boot, .975))})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / f"crct{suffix}_paired_improvement_summary.csv", index=False)
    (output / f"crct{suffix}_paired_improvement_manifest.json").write_text(json.dumps({
        "cluster": "doi", "n_boot": n_boot, "seed": seed,
        "baseline": "masking_validation.csv", "method": f"crct{suffix}_masked_validation.csv",
        "support_strategy": support_strategy,
    }, indent=2), encoding="utf-8")
    return summary


def run_nested_crct_stats(root: Path = PROJECT_ROOT, *, n_boot: int = 5000,
                           seed: int = 20261002) -> pd.DataFrame:
    """Bootstrap the outer-test episodes after inner parameter selection.

    The nested file contains one selected CRCT row per seed/DOI and its
    matched temperature-Ridge error.  DOI is the resampling unit so repeated
    seeds from one paper cannot dominate the interval.
    """
    output = root / "results"
    frame = pd.read_csv(output / "crct_nested_outer_validation.csv")
    required = {"seed", "doi", "mae", "ridge_mae"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"nested validation missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["improvement"] = frame["ridge_mae"] - frame["mae"]
    frame["crct_win"] = frame["improvement"] > 0
    if "distance_group" not in frame:
        if "mean_nearest_distance" in frame:
            q1, q2 = frame.mean_nearest_distance.quantile([1 / 3, 2 / 3])
            frame["distance_group"] = pd.cut(
                frame.mean_nearest_distance, [-np.inf, q1, q2, np.inf],
                labels=["near", "mid", "far"])
        else:
            frame["distance_group"] = "overall"
    rows = []
    rng = np.random.default_rng(seed)
    populations = [("overall", frame)]
    if frame["distance_group"].nunique() > 1:
        populations.extend((str(name), group) for name, group in frame.groupby("distance_group", sort=True))
    for population, group in populations:
        doi_means = group.groupby("doi", sort=True).improvement.mean().to_numpy(float)
        draws = rng.integers(0, len(doi_means), size=(n_boot, len(doi_means)))
        boot = doi_means[draws].mean(axis=1)
        rows.append({
            "population": population,
            "n_dois": int(group.doi.nunique()),
            "n_episodes": int(len(group)),
            "ridge_mae": float(group.ridge_mae.mean()),
            "crct_mae": float(group.mae.mean()),
            "mean_improvement": float(group.improvement.mean()),
            "win_rate": float(group.crct_win.mean()),
            "ci95_low": float(np.quantile(boot, .025)),
            "ci95_high": float(np.quantile(boot, .975)),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "crct_nested_paired_improvement_summary.csv", index=False)
    frame.to_csv(output / "crct_nested_paired_improvement.csv", index=False)
    (output / "crct_nested_paired_improvement_manifest.json").write_text(json.dumps({
        "source": "crct_nested_outer_validation.csv", "cluster": "doi",
        "n_boot": n_boot, "seed": seed,
        "parameter_selection": "inner DOI validation; outer test untouched",
    }, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(run_crct_stats().to_string(index=False))
