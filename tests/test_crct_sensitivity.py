import pandas as pd
import pytest

from src.dcsa.crct_sensitivity import (
    DEFAULT_BLENDS,
    DEFAULT_TAUS,
    parameter_grid,
    summarize_sensitivity,
    validate_parameter_grid,
)


def test_parameter_grid_contains_requested_tau_and_blend_values():
    grid = parameter_grid()

    assert DEFAULT_TAUS == (0.25, 0.5, 1.0)
    assert DEFAULT_BLENDS == (0.0, 0.5, 1.0)
    assert len(grid) == 18
    assert {row["support_selection"] for row in grid} == {"random", "diverse_support"}
    assert {row["tau"] for row in grid} == set(DEFAULT_TAUS)
    assert {row["blend"] for row in grid} == set(DEFAULT_BLENDS)


def test_parameter_grid_rejects_nonpositive_tau_and_out_of_range_blend():
    with pytest.raises(ValueError, match="tau"):
        validate_parameter_grid((0.0,), (1.0,), ("random",))
    with pytest.raises(ValueError, match="blend"):
        validate_parameter_grid((0.5,), (1.1,), ("random",))


def test_summarize_sensitivity_aggregates_each_protocol_and_distance_group():
    frame = pd.DataFrame([
        {"support_selection": "random", "tau": 0.25, "blend": 0.0,
         "distance_group": "near_q75", "mae": 0.2,
         "chemistry_mae": 0.4, "ridge_mae": 0.2, "doi": "a"},
        {"support_selection": "random", "tau": 0.25, "blend": 0.0,
         "distance_group": "near_q75", "mae": 0.4,
         "chemistry_mae": 0.6, "ridge_mae": 0.4, "doi": "b"},
        {"support_selection": "random", "tau": 0.25, "blend": 0.0,
         "distance_group": "far_q25", "mae": 0.8,
         "chemistry_mae": 0.9, "ridge_mae": 0.8, "doi": "c"},
    ])

    summary = summarize_sensitivity(frame)

    near = summary.query("support_selection == 'random' and tau == 0.25 and blend == 0.0 and distance_group == 'near_q75'").iloc[0]
    assert near.mae_mean == pytest.approx(0.3)
    assert near.mae_median == pytest.approx(0.3)
    assert near.n_episodes == 2
