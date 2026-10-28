"""Curve-level and candidate-ranking metrics for masked ZT prediction."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


_REQUIRED_COLUMNS = {"curve_row_id", "temperature", "target"}


def _validate_inputs(query: pd.DataFrame, prediction: Sequence[float]) -> np.ndarray:
    missing = _REQUIRED_COLUMNS - set(query.columns)
    if missing:
        raise ValueError(f"query is missing columns: {sorted(missing)}")
    if query.empty:
        raise ValueError("query must contain at least one point")
    values = np.asarray(prediction, dtype=float)
    if values.ndim != 1 or len(values) != len(query):
        raise ValueError("prediction must have one value per query point")
    targets = query.target.to_numpy(dtype=float)
    temperatures = query.temperature.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("prediction must contain only finite values")
    if not np.isfinite(targets).all() or not np.isfinite(temperatures).all():
        raise ValueError("query target and temperature must contain only finite values")
    return values


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = float(np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered)))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(x_centered, y_centered) / denominator)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks without requiring scipy for Spearman."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return _pearson(_average_ranks(x), _average_ranks(y))


def curve_level_metrics(query: pd.DataFrame, prediction: Sequence[float]) -> pd.DataFrame:
    """Compute metrics for each predicted query curve.

    Rows are matched by their existing order, then sorted by temperature within
    each ``curve_row_id``.  Ties at a curve maximum use the lowest temperature
    because the first maximum is selected after sorting.
    """
    values = _validate_inputs(query, prediction)
    frame = query.loc[:, ["curve_row_id", "temperature", "target"]].copy()
    frame["prediction"] = values
    rows: list[dict[str, float | int | object]] = []
    for curve_id, curve in frame.groupby("curve_row_id", sort=False):
        curve = curve.sort_values("temperature", kind="mergesort")
        temperature = curve.temperature.to_numpy(dtype=float)
        target = curve.target.to_numpy(dtype=float)
        predicted = curve.prediction.to_numpy(dtype=float)
        target_peak_index = int(np.argmax(target))
        predicted_peak_index = int(np.argmax(predicted))
        rows.append({
            "curve_row_id": curve_id,
            "n_points": int(len(curve)),
            "true_peak_zt": float(target[target_peak_index]),
            "predicted_peak_zt": float(predicted[predicted_peak_index]),
            "abs_peak_zt_error": float(abs(predicted[predicted_peak_index] - target[target_peak_index])),
            "true_optimum_temperature": float(temperature[target_peak_index]),
            "predicted_optimum_temperature": float(temperature[predicted_peak_index]),
            "abs_optimum_temperature_error": float(
                abs(temperature[predicted_peak_index] - temperature[target_peak_index])),
            "pearson_r": _pearson(target, predicted),
            "spearman_r": _spearman(target, predicted),
        })
    return pd.DataFrame(rows)


def _ranking_metrics(curves: pd.DataFrame, top_k: int) -> dict[str, float | int]:
    n_curves = len(curves)
    peak_true = curves.true_peak_zt.to_numpy(dtype=float)
    peak_pred = curves.predicted_peak_zt.to_numpy(dtype=float)
    result: dict[str, float | int] = {
        "peak_ranking_spearman": _spearman(peak_true, peak_pred),
        "peak_ranking_kendall": float("nan"),
    }
    if n_curves >= 2:
        result["peak_ranking_kendall"] = float(kendalltau(peak_true, peak_pred, nan_policy="omit").statistic)
    k = min(top_k, n_curves)
    true_order = np.argsort(-peak_true, kind="mergesort")[:k]
    predicted_order = np.argsort(-peak_pred, kind="mergesort")[:k]
    overlap = len(set(true_order.tolist()).intersection(predicted_order.tolist()))
    result[f"top_{k}_overlap"] = float(overlap / k)
    result[f"top_{k}_hit"] = int(overlap == k)
    return result


def evaluate_curve_metrics(
    query: pd.DataFrame,
    prediction: Sequence[float],
    *,
    top_k: int = 3,
) -> dict[str, float | int]:
    """Aggregate curve errors and ranking metrics for one masked episode."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    curves = curve_level_metrics(query, prediction)

    def mean_or_nan(column: str) -> float:
        values = curves[column].to_numpy(dtype=float)
        return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")

    result: dict[str, float | int] = {
        "n_curves": int(len(curves)),
        "peak_zt_mae": mean_or_nan("abs_peak_zt_error"),
        "optimum_temperature_mae": mean_or_nan("abs_optimum_temperature_error"),
        "curve_pearson_mean": mean_or_nan("pearson_r"),
        "curve_spearman_mean": mean_or_nan("spearman_r"),
    }
    result.update(_ranking_metrics(curves, top_k))
    return result
