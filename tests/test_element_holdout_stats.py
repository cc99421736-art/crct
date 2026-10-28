import pandas as pd
import pytest

from src.dcsa.element_holdout_stats import run_element_holdout_stats


def test_element_holdout_stats_pairs_models(tmp_path):
    output = tmp_path / "results"
    output.mkdir()
    pd.DataFrame([
        {"seed": 1, "doi": "a", "holdout_mode": "element", "model": "crct", "mae": .1},
        {"seed": 1, "doi": "a", "holdout_mode": "element", "model": "temperature_ridge", "mae": .2},
        {"seed": 1, "doi": "b", "holdout_mode": "element", "model": "crct", "mae": .3},
        {"seed": 1, "doi": "b", "holdout_mode": "element", "model": "temperature_ridge", "mae": .2},
    ]).to_csv(output / "element_family_holdout.csv", index=False)
    summary = run_element_holdout_stats(tmp_path, n_boot=100)
    row = summary.iloc[0]
    assert row.mean_improvement == pytest.approx(0)
    assert row.win_rate == pytest.approx(.5)
