"""Paired DOI bootstrap comparing nested strong baselines with CRCT."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.dcsa.data_source import PROJECT_ROOT

def run(root: Path = PROJECT_ROOT, n_boot: int = 5000, seed: int = 20261010):
    base = pd.read_csv(root / "results" / "nested_strong_baselines_outer_doi.csv")
    crct = pd.read_csv(root / "results" / "crct_nested_outer_validation.csv")[["seed", "doi", "mae"]].rename(columns={"mae": "crct_mae"})
    merged = base.merge(crct, on=["seed", "doi"], validate="many_to_one")
    rows = []
    rng = np.random.default_rng(seed)
    for model, group in merged.groupby("model", sort=True):
        paired = group.groupby("doi", as_index=False).agg(crct_mae=("crct_mae", "mean"), baseline_mae=("mae", "mean"))
        improvement = paired.baseline_mae - paired.crct_mae
        draws = rng.integers(0, len(improvement), size=(n_boot, len(improvement)))
        means = improvement.to_numpy()[draws].mean(axis=1)
        rows.append({"model": model, "n_dois": len(paired), "crct_mae": paired.crct_mae.mean(), "baseline_mae": paired.baseline_mae.mean(), "improvement": improvement.mean(), "ci95_low": np.quantile(means, .025), "ci95_high": np.quantile(means, .975), "win_rate": np.mean(improvement > 0)})
    out = pd.DataFrame(rows)
    out.to_csv(root / "results" / "nested_strong_baselines_paired_stats.csv", index=False)
    return out

if __name__ == "__main__": print(run().to_string(index=False))
