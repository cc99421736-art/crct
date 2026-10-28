import numpy as np
import pandas as pd
import pytest

from src.dcsa.crct import FeatureScaler, evaluate_episode, predict_episode


def _episode():
    support = pd.DataFrame({
        "curve_row_id": [10, 10, 20, 20],
        "temperature": [300.0, 500.0, 300.0, 500.0],
        "target": [1.0, 3.0, 2.0, 4.0],
        "f_x": [0.0, 0.0, 2.0, 2.0],
    })
    query = pd.DataFrame({
        "curve_row_id": [30, 30, 30],
        "temperature": [200.0, 400.0, 600.0],
        "target": [1.0, 2.0, 3.0],
        "f_x": [0.0, 0.0, 0.0],
    })
    return support, query


def test_gaussian_weights_favor_closest_support_and_sum_to_one():
    support, query = _episode()
    scaler = FeatureScaler(mean=np.array([0.0]), scale=np.array([1.0]))
    out = predict_episode(
        support,
        query,
        ["f_x"],
        support_base=np.zeros(len(support)),
        query_base=np.zeros(len(query)),
        scaler=scaler,
        tau=1.0,
    )
    assert out.weights.shape == (len(query), 2)
    assert np.allclose(out.weights.sum(axis=1), 1.0)
    assert np.all(out.weights[:, 0] > out.weights[:, 1])


def test_residual_transport_uses_temperature_endpoints_outside_support_range():
    support, query = _episode()
    scaler = FeatureScaler(mean=np.array([0.0]), scale=np.array([1.0]))
    out = predict_episode(
        support,
        query,
        ["f_x"],
        support_base=np.zeros(len(support)),
        query_base=np.zeros(len(query)),
        scaler=scaler,
        tau=0.1,
    )
    # The query has the first support composition, so 200 K and 600 K use
    # the first curve's endpoint residuals (1 and 3).
    assert np.allclose(out.chemistry_prediction[[0, 2]], [1.0, 3.0], atol=1e-3)
    assert 1.0 < out.chemistry_prediction[1] < 3.0


def test_blend_zero_returns_temperature_ridge_and_evaluation_reports_mae():
    support, query = _episode()
    scaler = FeatureScaler(mean=np.array([0.0]), scale=np.array([1.0]))
    out = predict_episode(
        support,
        query,
        ["f_x"],
        support_base=np.zeros(len(support)),
        query_base=np.zeros(len(query)),
        scaler=scaler,
        blend=0.0,
    )
    assert np.allclose(out.prediction, out.ridge_prediction)
    metrics = evaluate_episode(
        support,
        query,
        ["f_x"],
        support_base=np.zeros(len(support)),
        query_base=np.zeros(len(query)),
        scaler=scaler,
        blend=1.0,
    )
    assert set(("mae", "chemistry_mae", "ridge_mae")) <= metrics.keys()
    assert metrics["n_query_points"] == len(query)


def test_cosine_distance_is_supported_and_invalid_metric_rejected():
    support, query = _episode()
    scaler = FeatureScaler(mean=np.array([0.0]), scale=np.array([1.0]))
    out = predict_episode(support, query, ["f_x"], support_base=np.zeros(len(support)),
                          query_base=np.zeros(len(query)), scaler=scaler,
                          distance_metric="cosine")
    assert out.weights.shape == (3, 2)
    with pytest.raises(ValueError, match="distance_metric"):
        predict_episode(support, query, ["f_x"], support_base=np.zeros(len(support)),
                        query_base=np.zeros(len(query)), scaler=scaler,
                          distance_metric="bad")


def test_neighbor_k_keeps_only_nearest_supports():
    support, query = _episode()
    scaler = FeatureScaler(mean=np.array([0.0]), scale=np.array([1.0]))
    out = predict_episode(
        support, query, ["f_x"], support_base=np.zeros(len(support)),
        query_base=np.zeros(len(query)), scaler=scaler, tau=1.0, neighbor_k=1)
    assert out.weights.shape == (len(query), 2)
    assert np.allclose(out.weights[:, 1], 0.0)
    with pytest.raises(ValueError, match="neighbor_k"):
        predict_episode(
            support, query, ["f_x"], support_base=np.zeros(len(support)),
            query_base=np.zeros(len(query)), scaler=scaler, neighbor_k=0)


def test_distance_gate_reduces_residual_transfer_for_far_queries():
    support, query = _episode()
    query = query.copy()
    query["f_x"] = [0.0, 1.0, 4.0]
    scaler = FeatureScaler(mean=np.array([0.0]), scale=np.array([1.0]))
    out = predict_episode(
        support, query, ["f_x"], support_base=np.zeros(len(support)),
        query_base=np.zeros(len(query)), scaler=scaler, tau=1.0,
        neighbor_k=1, gate_tau=1.0)
    assert out.gate.shape == (len(query),)
    assert out.gate[0] > out.gate[1]
