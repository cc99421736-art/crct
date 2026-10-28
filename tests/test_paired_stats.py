import pandas as pd
import pytest

from src.dcsa.paired_stats import build_paired_improvements, summarize_paired


def test_build_paired_improvements_filters_protocol_and_joins_seed_doi():
    baseline = pd.DataFrame([
        {"seed": 1, "doi": "a", "shot": 5, "strategy": "random",
         "model": "temperature_ridge", "mae": 0.10},
        {"seed": 1, "doi": "a", "shot": 5, "strategy": "ordered",
         "model": "temperature_ridge", "mae": 0.20},
        {"seed": 1, "doi": "b", "shot": 5, "strategy": "random",
         "model": "temperature_ridge", "mae": 0.30},
    ])
    dcsa = pd.DataFrame([
        {"seed": 1, "doi": "a", "shot": 5, "model": "dcsa",
         "ablation": "full", "mae": 0.08},
        {"seed": 1, "doi": "a", "shot": 5, "model": "dcsa",
         "ablation": "paper_only", "mae": 0.01},
        {"seed": 1, "doi": "b", "shot": 5, "model": "dcsa",
         "ablation": "full", "mae": 0.35},
    ])

    paired = build_paired_improvements(baseline, dcsa)

    assert paired[["seed", "doi"]].values.tolist() == [[1, "a"], [1, "b"]]
    assert paired.improvement.tolist() == pytest.approx([0.02, -0.05])
    assert paired.dcsa_win.tolist() == [True, False]


def test_build_paired_improvements_rejects_unmatched_episodes():
    baseline = pd.DataFrame([{
        "seed": 1, "doi": "a", "shot": 5, "strategy": "random",
        "model": "temperature_ridge", "mae": 0.1,
    }])
    dcsa = pd.DataFrame([{
        "seed": 2, "doi": "a", "shot": 5, "model": "dcsa",
        "ablation": "full", "mae": 0.1,
    }])

    with pytest.raises(ValueError, match="unmatched"):
        build_paired_improvements(baseline, dcsa)


def test_summary_bootstraps_doi_clusters_and_reports_win_rate():
    paired = pd.DataFrame([
        {"seed": 1, "doi": "a", "improvement": 0.2, "dcsa_win": True},
        {"seed": 2, "doi": "a", "improvement": 0.0, "dcsa_win": False},
        {"seed": 1, "doi": "b", "improvement": -0.1, "dcsa_win": False},
    ])

    summary = summarize_paired(paired, n_boot=300, seed=7)

    assert summary.population == "overall"
    assert summary.n_dois == 2
    assert summary.n_episodes == 3
    assert summary.mean_improvement == pytest.approx(0.1 / 3)
    assert summary.win_rate == pytest.approx(1 / 3)
    assert summary.ci95_low <= summary.mean_improvement <= summary.ci95_high
    assert summary.far_quartile_available is False
