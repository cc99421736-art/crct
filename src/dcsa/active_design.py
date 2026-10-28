"""Convert CRCT predictions into an uncertainty-aware experiment ranking."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_risk_aware_score(candidates: pd.DataFrame, *,
                         distance_column: str = "nearest_distance",
                         outside_column: str = "fraction_outside_support_temperature",
                         peak_column: str = "predicted_peak_zt") -> pd.DataFrame:
    """Attach the unified pre-measurement risk score and its ranking score.

    ``risk_score = z(d_min) + z(f_outside)`` is the same definition used by the
    selective-prediction audit.  Because a z-score sum can be negative, it is
    shifted by its minimum inside the ranking pool before it is used as a
    discount; the shift is affine and therefore does not change the ordering.
    ``risk_aware_score = peak / (1 + risk_index)``.
    """
    required = {distance_column, outside_column, peak_column}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidates missing columns: {sorted(missing)}")
    if candidates.empty:
        raise ValueError("candidates must not be empty")
    frame = candidates.copy()

    def _z(values: pd.Series) -> pd.Series:
        scale = float(values.std(ddof=0))
        return (values - float(values.mean())) / (scale if scale > 0 else 1.0)

    risk = _z(frame[distance_column]) + _z(frame[outside_column])
    index = risk - float(risk.min())
    frame["risk_score"] = risk
    frame["risk_index"] = index
    frame["risk_aware_score"] = frame[peak_column] / (1.0 + index)
    return frame


def rank_candidates(candidates: pd.DataFrame, *, gate_exponent: float = 1.0,
                    distance_exponent: float = 1.0) -> pd.DataFrame:
    """Rank candidate compositions by predicted benefit and confidence.

    The score is deliberately transparent: predicted peak ZT is multiplied by
    the distance gate and discounted by composition distance. This is a
    screening priority, not a calibrated probability of experimental success.
    """
    if gate_exponent < 0 or distance_exponent < 0:
        raise ValueError("ranking exponents must be non-negative")
    required = {"predicted_peak_zt", "gate", "nearest_distance"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidates missing columns: {sorted(missing)}")
    if candidates.empty:
        raise ValueError("candidates must not be empty")
    frame = candidates.copy()
    for column in required:
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must contain finite values")
    if ((frame.gate < 0) | (frame.gate > 1)).any():
        raise ValueError("gate must be between zero and one")
    if (frame.nearest_distance < 0).any():
        raise ValueError("nearest_distance must be non-negative")
    frame["screening_score"] = (
        frame.predicted_peak_zt * frame.gate.pow(gate_exponent)
        / (1.0 + frame.nearest_distance).pow(distance_exponent))
    frame = frame.sort_values(["screening_score", "predicted_peak_zt"],
                              ascending=[False, False], kind="mergesort").reset_index(drop=True)
    frame["priority_rank"] = np.arange(1, len(frame) + 1)
    return frame


def evaluate_ranking(candidates: pd.DataFrame, *, top_k: int = 3,
                     high_value_quantile: float = 0.8) -> dict[str, float | int]:
    """Measure whether a ranking recovers genuinely high-value candidates.

    ``true_peak_zt`` is required only for retrospective masked validation. The
    function reports top-k overlap with the true high-value set and regret
    relative to the best candidate in the episode.
    """
    required = {"screening_score", "true_peak_zt"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidates missing columns: {sorted(missing)}")
    if candidates.empty or top_k < 1:
        raise ValueError("candidates must be non-empty and top_k positive")
    if not 0.0 < high_value_quantile <= 1.0:
        raise ValueError("high_value_quantile must be in (0, 1]")
    peaks = candidates.true_peak_zt.to_numpy(dtype=float)
    if not np.isfinite(peaks).all():
        raise ValueError("true_peak_zt must contain finite values")
    k = min(top_k, len(candidates))
    order = np.argsort(-candidates.screening_score.to_numpy(float), kind="mergesort")
    selected = order[:k]
    cutoff = float(np.quantile(peaks, high_value_quantile))
    high_value = peaks >= cutoff
    hit_rate = float(high_value[selected].mean())
    best = float(peaks.max())
    regret = float(best - peaks[selected].max())
    return {"n_candidates": int(len(candidates)), "top_k": int(k),
            "high_value_cutoff": cutoff, "top_k_high_value_hit_rate": hit_rate,
            "top_k_regret": regret}


def select_ranking_exponents(candidates: pd.DataFrame, *,
                             gate_exponents=(0.0, 0.25, 0.5, 0.75, 1.0),
                             distance_exponents=(0.0, 0.1, 0.25, 0.5, 1.0)) -> dict[str, float]:
    """Select ranking exponents by retrospective top-1 hit rate.

    Selection must be called only on an inner validation split. Ties prefer
    weaker penalties to avoid suppressing novel high-prediction candidates.
    """
    rows = []
    for gate_exp in gate_exponents:
        for distance_exp in distance_exponents:
            ranked = rank_candidates(candidates, gate_exponent=gate_exp,
                                     distance_exponent=distance_exp)
            selected = ranked.iloc[0]
            rows.append((float(gate_exp), float(distance_exp),
                         float(selected.true_peak_zt)))
    best = max(rows, key=lambda row: (row[2], -row[0], -row[1]))
    return {"gate_exponent": best[0], "distance_exponent": best[1]}


def curve_candidates(query: pd.DataFrame, prediction: np.ndarray,
                     gate: np.ndarray, nearest_distance: np.ndarray,
                     support: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregate point predictions into one row per candidate composition curve.

    When ``support`` is given, the fraction of a query curve's temperature
    points that fall outside every support curve's temperature range is stored
    as ``fraction_outside_support_temperature``.  It is part of the unified
    pre-measurement risk score.
    """
    required = {"curve_row_id", "temperature", "target"}
    if missing := required - set(query.columns):
        raise ValueError(f"query missing columns: {sorted(missing)}")
    prediction = np.asarray(prediction, dtype=float)
    gate = np.asarray(gate, dtype=float)
    nearest_distance = np.asarray(nearest_distance, dtype=float)
    if not (len(prediction) == len(gate) == len(nearest_distance) == len(query)):
        raise ValueError("curve prediction arrays must match query length")
    rows = []
    frame = query.copy()
    frame["prediction"] = prediction
    frame["gate"] = gate
    frame["nearest_distance"] = nearest_distance
    if support is not None:
        temperatures = frame.temperature.to_numpy(dtype=float)
        ranges = support.groupby("curve_row_id").temperature.agg(["min", "max"]).to_numpy(float)
        covered = np.zeros(len(frame), dtype=bool)
        for lower, upper in ranges:
            covered |= (temperatures >= lower) & (temperatures <= upper)
        frame["_covered"] = covered
    for curve_id, group in frame.groupby("curve_row_id", sort=False):
        row = {
            "curve_row_id": curve_id,
            "predicted_peak_zt": float(group.prediction.max()),
            "true_peak_zt": float(group.target.max()),
            "gate": float(group.gate.mean()),
            "nearest_distance": float(group.nearest_distance.mean()),
        }
        if "_covered" in group:
            row["fraction_outside_support_temperature"] = float(1.0 - group._covered.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_episode_selection(candidates: pd.DataFrame, *, seed: int = 0,
                               top_ks=(1, 3, 5), high_quantile: float = 0.8) -> dict[str, float | int]:
    """Compare peak ranking with random selection for one masked episode."""
    required = {"predicted_peak_zt", "true_peak_zt"}
    if missing := required - set(candidates.columns):
        raise ValueError(f"candidates missing columns: {sorted(missing)}")
    if candidates.empty:
        raise ValueError("candidates must not be empty")
    rng = np.random.default_rng(seed)
    n = len(candidates)
    predicted = candidates.predicted_peak_zt.to_numpy(float)
    true = candidates.true_peak_zt.to_numpy(float)
    predicted_order = np.argsort(-predicted, kind="mergesort")
    random_order = rng.permutation(n)
    cutoff = float(np.quantile(true, high_quantile))
    result: dict[str, float | int] = {"n_candidates": int(n), "true_top_quantile_cutoff": cutoff}
    for k in top_ks:
        if k < 1:
            raise ValueError("top_ks must be positive")
        kk = int(k)
        if kk > n:
            result[f"predicted_top{kk}_hit"] = float("nan")
            result[f"predicted_top{kk}_high_value_rate"] = float("nan")
            result[f"predicted_top{kk}_regret"] = float("nan")
            result[f"random_top{kk}_hit"] = float("nan")
            result[f"random_top{kk}_high_value_rate"] = float("nan")
            result[f"random_top{kk}_regret"] = float("nan")
            continue
        for name, order in (("predicted", predicted_order), ("random", random_order)):
            chosen = order[:kk]
            result[f"{name}_top{kk}_hit"] = float(true[chosen].max() == true.max())
            result[f"{name}_top{kk}_high_value_rate"] = float(np.mean(true[chosen] >= cutoff))
            result[f"{name}_top{kk}_regret"] = float(true.max() - true[chosen].max())
    return result
