import pandas as pd
import pytest

from src.dcsa.crct_stats import run_crct_stats


def test_crct_stats_writes_paired_summary(tmp_path):
    result = tmp_path / "results"
    result.mkdir()
    pd.DataFrame([
        {"seed": 1, "doi": "a", "shot": 5, "strategy": "random", "model": "temperature_ridge", "mae": .2},
        {"seed": 1, "doi": "b", "shot": 5, "strategy": "random", "model": "temperature_ridge", "mae": .3},
    ]).to_csv(result / "masking_validation.csv", index=False)
    pd.DataFrame([
        {"seed": 1, "doi": "a", "shot": 5, "method": "crct", "mae": .1, "distance_group": "near_q75"},
        {"seed": 1, "doi": "b", "shot": 5, "method": "crct", "mae": .4, "distance_group": "far_q25"},
    ]).to_csv(result / "crct_masked_validation.csv", index=False)
    out = run_crct_stats(tmp_path, n_boot=100)
    assert out.loc[out.population.eq("overall"), "mean_improvement"].iloc[0] == pytest.approx(0)
