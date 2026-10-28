"""Paired DCSA versus temperature-Ridge statistics.

The paired unit is one ``seed``/``DOI`` episode.  Confidence intervals are
bootstrapped by DOI so repeated seed runs for the same paper are not treated
as independent papers.  A far-composition subgroup can be added later once a
query-to-support distance is recorded in the episode results.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.dcsa.data_source import PROJECT_ROOT


@dataclass(frozen=True)
class PairedSummary:
    population: str
    n_dois: int
    n_episodes: int
    mean_improvement: float
    median_improvement: float
    win_rate: float
    ci95_low: float
    ci95_high: float
    far_quartile_available: bool


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def build_paired_improvements(
    baseline: pd.DataFrame,
    dcsa: pd.DataFrame,
    *,
    shot: int = 5,
    strategy: str = "random",
    baseline_model: str = "temperature_ridge",
    dcsa_model: str = "dcsa",
    dcsa_ablation: str = "full",
) -> pd.DataFrame:
    """Pair the strict 5-shot baseline and full DCSA episodes.

    ``improvement`` is positive when DCSA lowers MAE: ``ridge_mae - dcsa_mae``.
    Both inputs must contain exactly one selected row per ``seed``/``DOI``.
    """
    _require_columns(baseline, {"seed", "doi", "shot", "strategy", "model", "mae"}, "baseline")
    _require_columns(dcsa, {"seed", "doi", "shot", "model", "ablation", "mae"}, "dcsa")

    base = baseline.loc[
        baseline.shot.eq(shot)
        & baseline.strategy.eq(strategy)
        & baseline.model.eq(baseline_model),
        ["seed", "doi", "shot", "mae"],
    ].rename(columns={"mae": "ridge_mae"})
    candidate = dcsa.loc[
        dcsa.shot.eq(shot)
        & dcsa.model.eq(dcsa_model)
        & dcsa.ablation.eq(dcsa_ablation),
        ["seed", "doi", "shot", "mae"],
    ].rename(columns={"mae": "dcsa_mae"})

    keys = ["seed", "doi"]
    if base.duplicated(keys).any() or candidate.duplicated(keys).any():
        raise ValueError("selected results contain duplicate seed/DOI episodes")
    if base.empty or candidate.empty:
        raise ValueError("selected baseline and DCSA results must not be empty")

    merged = base.merge(candidate, on=keys, how="outer", suffixes=("_ridge", "_dcsa"), indicator=True)
    if not merged._merge.eq("both").all():
        unmatched = merged.loc[~merged._merge.eq("both"), keys].to_dict("records")
        raise ValueError(f"unmatched seed/DOI episodes: {unmatched[:5]}")
    if not merged.shot_ridge.eq(shot).all() or not merged.shot_dcsa.eq(shot).all():
        raise ValueError("paired episodes have inconsistent shot values")

    out = merged.loc[:, keys].copy()
    out["shot"] = shot
    out["ridge_mae"] = merged.ridge_mae.astype(float).to_numpy()
    out["dcsa_mae"] = merged.dcsa_mae.astype(float).to_numpy()
    out["improvement"] = out.ridge_mae - out.dcsa_mae
    out["dcsa_win"] = out.improvement > 0.0
    return out.sort_values(keys, kind="mergesort").reset_index(drop=True)


def summarize_paired(
    paired: pd.DataFrame,
    *,
    n_boot: int = 2000,
    seed: int = 20261002,
) -> PairedSummary:
    """Summarize paired improvements with a DOI-cluster bootstrap interval."""
    _require_columns(paired, {"seed", "doi", "improvement", "dcsa_win"}, "paired")
    if paired.empty:
        raise ValueError("paired results must not be empty")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")

    values = paired.improvement.astype(float).to_numpy()
    by_doi = paired.groupby("doi", sort=True).improvement.mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(by_doi), size=(n_boot, len(by_doi)))
    bootstrap_means = by_doi[draws].mean(axis=1)
    return PairedSummary(
        population="overall",
        n_dois=int(paired.doi.nunique()),
        n_episodes=int(len(paired)),
        mean_improvement=float(values.mean()),
        median_improvement=float(np.median(values)),
        win_rate=float(paired.dcsa_win.astype(bool).mean()),
        ci95_low=float(np.quantile(bootstrap_means, 0.025)),
        ci95_high=float(np.quantile(bootstrap_means, 0.975)),
        far_quartile_available=False,
    )


def run_paired_stats(
    root: Path = PROJECT_ROOT,
    *,
    n_boot: int = 2000,
    seed: int = 20261002,
) -> pd.DataFrame:
    """Read current result files and write the overall paired summary."""
    output = root / "results"
    baseline = pd.read_csv(output / "masking_validation.csv")
    dcsa = pd.read_csv(output / "dcsa_masked_validation.csv")
    paired = build_paired_improvements(baseline, dcsa)
    paired.to_csv(output / "paired_improvement.csv", index=False)
    summary = summarize_paired(paired, n_boot=n_boot, seed=seed)
    summary_frame = pd.DataFrame([asdict(summary)])
    summary_frame.to_csv(output / "paired_improvement_summary.csv", index=False)
    (output / "paired_improvement_manifest.json").write_text(
        json.dumps({
            "baseline": "masking_validation.csv",
            "dcsa": "dcsa_masked_validation.csv",
            "baseline_protocol": "shot=5,strategy=random,model=temperature_ridge",
            "dcsa_protocol": "shot=5,model=dcsa,ablation=full",
            "cluster": "doi",
            "n_boot": n_boot,
            "seed": seed,
            "far_quartile": "not computed: query distance is absent from result files",
        }, indent=2), encoding="utf-8")
    return summary_frame


if __name__ == "__main__":
    print(run_paired_stats().to_string(index=False))
