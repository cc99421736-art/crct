import numpy as np
import pandas as pd
import pytest

from src.dcsa.metrics import curve_level_metrics, evaluate_curve_metrics


def _query():
    return pd.DataFrame({
        "curve_row_id": [1, 1, 1, 2, 2, 2],
        "temperature": [300.0, 500.0, 700.0, 300.0, 500.0, 700.0],
        "target": [1.0, 3.0, 2.0, 2.0, 4.0, 3.0],
    })


def test_curve_level_metrics_measure_peak_and_optimum_temperature():
    table = curve_level_metrics(_query(), [1.1, 2.8, 2.2, 2.1, 3.7, 2.9])
    first = table.iloc[0]
    assert first.curve_row_id == 1
    assert first.true_peak_zt == pytest.approx(3.0)
    assert first.predicted_peak_zt == pytest.approx(2.8)
    assert first.abs_peak_zt_error == pytest.approx(0.2)
    assert first.true_optimum_temperature == pytest.approx(500.0)
    assert first.predicted_optimum_temperature == pytest.approx(500.0)
    assert first.abs_optimum_temperature_error == pytest.approx(0.0)
    assert first.pearson_r > 0.98
    assert first.spearman_r == pytest.approx(1.0)


def test_evaluate_curve_metrics_reports_curve_errors_and_peak_ranking():
    result = evaluate_curve_metrics(_query(), np.array([1.1, 2.8, 2.2, 2.1, 3.7, 2.9]), top_k=1)
    assert result["n_curves"] == 2
    assert result["peak_zt_mae"] == pytest.approx(0.25)
    assert result["optimum_temperature_mae"] == pytest.approx(0.0)
    assert result["curve_pearson_mean"] > 0.98
    assert result["curve_spearman_mean"] == pytest.approx(1.0)
    assert result["peak_ranking_spearman"] == pytest.approx(1.0)
    assert result["peak_ranking_kendall"] == pytest.approx(1.0)
    assert result["top_1_overlap"] == pytest.approx(1.0)
    assert result["top_1_hit"] == 1


def test_curve_metrics_reject_mismatched_predictions_and_undefined_ranking():
    with pytest.raises(ValueError, match="prediction"):
        curve_level_metrics(_query(), [1.0])
    single = _query().iloc[:3].copy()
    out = evaluate_curve_metrics(single, [1.0, 2.0, 1.0])
    assert np.isnan(out["peak_ranking_spearman"])
    assert np.isnan(out["peak_ranking_kendall"])


def test_curve_metrics_reject_empty_query():
    with pytest.raises(ValueError, match="at least one point"):
        evaluate_curve_metrics(_query().iloc[0:0], [])
