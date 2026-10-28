import numpy as np
import pandas as pd

from src.dcsa.active_design import add_risk_aware_score, curve_candidates
from src.dcsa.crct import FeatureScaler, evaluate_episode, predict_episode


def _episode():
    support = pd.DataFrame({
        "curve_row_id": [10, 10, 20, 20],
        "temperature": [300.0, 500.0, 300.0, 500.0],
        "target": [1.0, 3.0, 2.0, 4.0],
        "f_x": [0.0, 0.0, 0.0, 0.0],
    })
    query = pd.DataFrame({
        "curve_row_id": [30, 30, 30],
        "temperature": [200.0, 400.0, 600.0],
        "target": [1.0, 2.0, 3.0],
        "f_x": [0.0, 0.0, 0.0],
    })
    return support, query


def _prediction():
    support, query = _episode()
    scaler = FeatureScaler(mean=np.array([0.0]), scale=np.array([1.0]))
    return predict_episode(
        support, query, ["f_x"], support_base=np.zeros(len(support)),
        query_base=np.zeros(len(query)), scaler=scaler, tau=1.0)


def test_additive_raw_sums_both_residual_corrections():
    out = _prediction()
    assert np.allclose(out.variants["additive_raw"],
                       out.ridge_residual + out.chemistry_residual)
    assert np.allclose(out.variants["chemistry_only"], out.chemistry_prediction)
    assert np.allclose(out.variants["ridge_only"], out.ridge_prediction)


def test_orthogonal_additive_removes_anchor_component():
    out = _prediction()
    chemistry = out.chemistry_residual
    anchor = out.ridge_residual
    assert np.allclose(chemistry, np.array([1.5, 2.5, 3.5]))
    orthogonal = out.variants["additive_orthogonal"] - out.ridge_prediction
    assert abs(float(np.dot(orthogonal, anchor))) < 1e-8


def test_variant_metrics_cover_every_model():
    support, query = _episode()
    scaler = FeatureScaler(mean=np.array([0.0]), scale=np.array([1.0]))
    metrics = evaluate_episode(
        support, query, ["f_x"], support_base=np.zeros(len(support)),
        query_base=np.zeros(len(query)), scaler=scaler)
    for family in ("ridge_only", "chemistry_only", "additive_raw",
                   "additive_orthogonal", "convex_blend"):
        assert f"mae_{family}" in metrics
        assert f"peak_mae_{family}" in metrics


def test_risk_aware_score_is_monotone_in_risk():
    candidates = pd.DataFrame({
        "predicted_peak_zt": [1.0, 1.05],
        "nearest_distance": [0.05, 0.60],
        "fraction_outside_support_temperature": [0.0, 0.40],
    })
    out = add_risk_aware_score(candidates)
    assert set(out.columns) >= {"risk_score", "risk_index", "risk_aware_score"}
    assert float(out.risk_index.min()) == 0.0
    expected = out.predicted_peak_zt / (1.0 + out.risk_index)
    assert np.allclose(expected, out.risk_aware_score)
    # The near, in-range candidate must keep its peak score; the far one is
    # discounted below its raw peak.
    assert out.risk_aware_score.iloc[0] == out.predicted_peak_zt.iloc[0]
    assert out.risk_aware_score.iloc[1] < out.predicted_peak_zt.iloc[1]


def test_curve_candidates_reports_temperature_coverage():
    support, query = _episode()
    out = _prediction()
    frame = curve_candidates(query, out.variants["chemistry_only"], out.gate,
                             out.nearest_distance, support=support)
    assert "fraction_outside_support_temperature" in frame.columns
    # 200 K and 600 K are outside the 300-500 K support range.
    assert np.isclose(frame.fraction_outside_support_temperature.iloc[0], 2 / 3)
