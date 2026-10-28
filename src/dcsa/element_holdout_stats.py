"""Paired statistics for leave-element/family-out validation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.dcsa.data_source import PROJECT_ROOT


def run_element_holdout_stats(root: Path = PROJECT_ROOT, *, n_boot: int = 2000,
                              seed: int = 20261002) -> pd.DataFrame:
    output = root / "results"
    frame = pd.read_csv(output / "element_family_holdout.csv")
    required = {"seed", "doi", "holdout_mode", "model", "mae"}
    if missing := required - set(frame.columns):
        raise ValueError(f"missing columns: {sorted(missing)}")
    pivot = (frame.pivot_table(index=["seed", "doi", "holdout_mode"],
                               columns="model", values="mae", aggfunc="first")
             .dropna(subset=["crct", "temperature_ridge"]).reset_index())
    pivot["improvement"] = pivot.temperature_ridge - pivot.crct
    pivot["crct_win"] = pivot.improvement > 0
    pivot.to_csv(output / "element_family_holdout_paired.csv", index=False)
    rng = np.random.default_rng(seed)
    rows = []
    for mode, group in pivot.groupby("holdout_mode", sort=True):
        doi_values = group.groupby("doi").improvement.mean().to_numpy(float)
        draws = rng.integers(0, len(doi_values), size=(n_boot, len(doi_values)))
        bootstrap = doi_values[draws].mean(axis=1)
        rows.append({"holdout_mode": mode, "n_dois": int(group.doi.nunique()),
                     "n_episodes": int(len(group)),
                     "ridge_mae": float(group.temperature_ridge.mean()),
                     "crct_mae": float(group.crct.mean()),
                     "mean_improvement": float(group.improvement.mean()),
                     "win_rate": float(group.crct_win.mean()),
                     "ci95_low": float(np.quantile(bootstrap, 0.025)),
                     "ci95_high": float(np.quantile(bootstrap, 0.975))})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "element_family_holdout_paired_summary.csv", index=False)
    (output / "element_family_holdout_paired_manifest.json").write_text(json.dumps({
        "cluster": "doi", "n_boot": n_boot, "seed": seed,
        "input": "element_family_holdout.csv",
    }, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(run_element_holdout_stats().to_string(index=False))
