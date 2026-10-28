import pandas as pd

import numpy as np

from src.dcsa.active_design import (curve_candidates, evaluate_episode_selection,
                                    evaluate_ranking, rank_candidates)


def test_rank_candidates_prefers_high_peak_with_confidence():
    candidates = pd.DataFrame({
        "material": ["A", "B", "C"],
        "predicted_peak_zt": [1.0, 1.3, 1.1],
        "gate": [1.0, 0.2, 0.9],
        "nearest_distance": [0.2, 0.1, 0.4],
    })
    ranked = rank_candidates(candidates)
    assert ranked.iloc[0].material == "A"
    assert list(ranked.priority_rank) == [1, 2, 3]


def test_rank_candidates_rejects_invalid_values():
    candidates = pd.DataFrame({"predicted_peak_zt": [1.0], "gate": [1.2],
                               "nearest_distance": [0.1]})
    try:
        rank_candidates(candidates)
    except ValueError as exc:
        assert "gate" in str(exc)
    else:
        raise AssertionError("invalid gate should be rejected")


def test_evaluate_ranking_reports_high_value_hits_and_regret():
    candidates = pd.DataFrame({
        "screening_score": [0.9, 0.4, 0.2, 0.1],
        "true_peak_zt": [1.0, 0.8, 1.4, 0.7],
    })
    result = evaluate_ranking(candidates, top_k=2, high_value_quantile=0.5)
    assert result["top_k_high_value_hit_rate"] == 0.5
    assert abs(result["top_k_regret"] - 0.4) < 1e-12


def test_curve_candidates_aggregates_point_predictions():
    query = pd.DataFrame({"curve_row_id": [1, 1, 2, 2],
                          "temperature": [300, 500, 300, 500],
                          "target": [0.4, 0.9, 0.8, 0.7]})
    result = curve_candidates(query, np.array([0.5, 0.8, 0.6, 0.75]),
                              np.ones(4), np.full(4, 0.2))
    assert len(result) == 2
    assert result.loc[result.curve_row_id.eq(1), "predicted_peak_zt"].item() == 0.8


def test_episode_selection_reports_prediction_and_random_metrics():
    candidates = pd.DataFrame({"predicted_peak_zt": [1.0, 0.8, 1.4, 0.7],
                               "true_peak_zt": [1.0, 0.8, 1.4, 0.7]})
    result = evaluate_episode_selection(candidates, seed=4, top_ks=(1, 3))
    assert result["predicted_top1_hit"] == 1.0
    assert result["predicted_top3_regret"] == 0.0
