"""DOI-clustered uncertainty summaries for DCSA baseline results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.dcsa.data_source import PROJECT_ROOT


def clustered_bootstrap(frame: pd.DataFrame, *, n_boot: int = 2000, seed: int = 20261001) -> pd.DataFrame:
    """Bootstrap MAE by seed-DOI episode, preserving within-episode points."""
    required = {"shot", "model", "seed", "doi", "mae"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    rng = np.random.default_rng(seed)
    rows = []
    for (shot, model), group in frame.groupby(["shot", "model"], sort=True):
        clusters = group.assign(cluster=group["seed"].astype(str) + ":" + group["doi"])
        values = clusters.groupby("cluster", as_index=False).mae.mean().mae.to_numpy(float)
        draws = rng.integers(0, len(values), size=(n_boot, len(values)))
        means = values[draws].mean(axis=1)
        rows.append({"shot": int(shot), "model": model, "n_clusters": len(values),
                     "mae_mean": float(values.mean()),
                     "ci95_low": float(np.quantile(means, 0.025)),
                     "ci95_high": float(np.quantile(means, 0.975))})
    return pd.DataFrame(rows)


def run_stats(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    result_path = root / "results" / "support_curve_baseline.csv"
    frame = pd.read_csv(result_path)
    summary = clustered_bootstrap(frame)
    summary.to_csv(root / "results" / "support_curve_baseline_bootstrap.csv", index=False)
    manifest = {"input": str(result_path), "cluster": "seed:doi", "n_boot": 2000,
                "seed": 20261001}
    (root / "results" / "support_curve_baseline_bootstrap.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(run_stats().to_string(index=False))
