"""Clustered bootstrap summary for retrospective experiment selection."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.dcsa.active_design import evaluate_episode_selection
from src.dcsa.data_source import PROJECT_ROOT


def run_active_design_evaluation(root: Path = PROJECT_ROOT, *, n_boot: int = 5000,
                                 seed: int = 20261003) -> pd.DataFrame:
    output = root / "results"
    candidates = pd.read_csv(output / "crct_nested_candidate_predictions.csv")
    rows = []
    for (episode_seed, doi), group in candidates.groupby(["seed", "doi"], sort=True):
        rows.append({"seed": episode_seed, "doi": doi,
                     **evaluate_episode_selection(group, seed=int(episode_seed))})
    episodes = pd.DataFrame(rows)
    episodes.to_csv(output / "active_design_episode_metrics.csv", index=False)
    rng = np.random.default_rng(seed)
    summaries = []
    metric_columns = [c for c in episodes.columns if c.endswith(("_hit", "_high_value_rate", "_regret"))]
    for metric in metric_columns:
        doi_values = episodes.groupby("doi", sort=True)[metric].mean().dropna().to_numpy(float)
        if len(doi_values) == 0:
            continue
        draws = rng.integers(0, len(doi_values), size=(n_boot, len(doi_values)))
        boot = doi_values[draws].mean(axis=1)
        summaries.append({"metric": metric, "mean": float(episodes[metric].mean()),
                          "n_dois": int(len(doi_values)),
                          "n_episodes": int(episodes[metric].notna().sum()),
                          "ci95_low": float(np.quantile(boot, .025)),
                          "ci95_high": float(np.quantile(boot, .975))})
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "active_design_evaluation.csv", index=False)
    (output / "active_design_evaluation_manifest.json").write_text(json.dumps({
        "source": "crct_nested_candidate_predictions.csv", "cluster": "doi",
        "n_boot": n_boot, "seed": seed, "high_value_quantile": 0.8,
    }, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(run_active_design_evaluation().to_string(index=False))
