"""DOI-clustered uncertainty statistics for chronological validation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.dcsa.data_source import PROJECT_ROOT


def run_temporal_stats(root: Path = PROJECT_ROOT, *, n_boot: int = 5000,
                       seed: int = 20261004) -> pd.DataFrame:
    out = root / "results"
    frame = pd.read_csv(out / "temporal_validation.csv")
    frame["improvement"] = frame.ridge_mae - frame.mae
    frame["crct_win"] = frame.improvement > 0
    rows = []
    rng = np.random.default_rng(seed)
    for population, group in [("overall", frame),
                              *[(f"year_{int(year)}", g) for year, g in frame.groupby("year")]]:
        doi_values = group.groupby("doi").improvement.mean().to_numpy(float)
        draws = rng.integers(0, len(doi_values), size=(n_boot, len(doi_values)))
        boot = doi_values[draws].mean(axis=1)
        rows.append({"population": population, "n_dois": int(group.doi.nunique()),
                     "n_episodes": int(len(group)), "crct_mae": float(group.mae.mean()),
                     "ridge_mae": float(group.ridge_mae.mean()),
                     "mean_improvement": float(group.improvement.mean()),
                     "win_rate": float(group.crct_win.mean()),
                     "ci95_low": float(np.quantile(boot, .025)),
                     "ci95_high": float(np.quantile(boot, .975))})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "temporal_validation_stats.csv", index=False)
    (out / "temporal_validation_stats_manifest.json").write_text(json.dumps({
        "source": "temporal_validation.csv", "cluster": "doi",
        "n_boot": n_boot, "seed": seed,
    }, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(run_temporal_stats().to_string(index=False))
