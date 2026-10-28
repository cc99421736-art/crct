"""Chemistry-conditioned residual curve transport for masked ZT curves."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.dcsa.baseline import PARAMS, _features, _load_points, _split_dois
from src.dcsa.data_source import PROJECT_ROOT, input_paths, sha256_file
from src.dcsa.masking_validation import SHOT, select_support
from src.dcsa.metrics import evaluate_curve_metrics


@dataclass(frozen=True)
class FeatureScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame, feature_columns: list[str]) -> "FeatureScaler":
        compositions = frame.drop_duplicates("curve_row_id")
        values = compositions[feature_columns].to_numpy(dtype=float)
        mean = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0)
        mean[~np.isfinite(mean)] = 0.0
        scale[~np.isfinite(scale) | (scale == 0)] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / self.scale


@dataclass(frozen=True)
class EpisodePrediction:
    prediction: np.ndarray
    chemistry_prediction: np.ndarray
    ridge_prediction: np.ndarray
    weights: np.ndarray
    nearest_distance: np.ndarray
    gate: np.ndarray
    ridge_residual: np.ndarray
    chemistry_residual: np.ndarray
    variants: dict[str, np.ndarray] = field(default_factory=dict)


def _curve_residuals(support: pd.DataFrame, support_base: np.ndarray,
                     feature_columns: list[str]) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    base = np.asarray(support_base, dtype=float)
    if len(base) == len(support):
        residual = support.target.to_numpy(dtype=float) - base
    else:
        raise ValueError("support_base must have one value per support point")
    frame = support.copy()
    frame["_residual"] = residual
    compositions = []
    temperatures = []
    residuals = []
    for _, curve in frame.groupby("curve_row_id", sort=False):
        curve = curve.sort_values("temperature", kind="mergesort")
        temperature = curve.temperature.to_numpy(dtype=float)
        values = curve._residual.to_numpy(dtype=float)
        unique_temperature, indices = np.unique(temperature, return_index=True)
        compositions.append(curve.iloc[0][feature_columns].to_numpy(dtype=float))
        temperatures.append(unique_temperature)
        residuals.append(values[indices])
    return compositions, temperatures, residuals


def predict_episode(support: pd.DataFrame, query: pd.DataFrame,
                    feature_columns: list[str], *, support_base: np.ndarray,
                    query_base: np.ndarray, scaler: FeatureScaler,
                    tau: float = 0.5, blend: float = 1.0,
                    ridge_alpha: float = 1.0,
                    distance_metric: str = "euclidean",
                    neighbor_k: int | None = None,
                    gate_tau: float | None = None) -> EpisodePrediction:
    """Predict query curves by Gaussian-weighted residual transfer and Ridge blend.

    ``blend=1`` uses chemistry transfer only; ``blend=0`` uses the paper-level
    temperature Ridge residual model only. Temperature interpolation clamps to
    the observed endpoint residual outside each support curve's range.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be between zero and one")
    if distance_metric not in {"euclidean", "cosine"}:
        raise ValueError("distance_metric must be 'euclidean' or 'cosine'")
    if neighbor_k is not None and neighbor_k < 1:
        raise ValueError("neighbor_k must be positive")
    if gate_tau is not None and gate_tau <= 0:
        raise ValueError("gate_tau must be positive")
    if support.empty or query.empty:
        raise ValueError("support and query must both contain points")
    q_base = np.asarray(query_base, dtype=float)
    if len(q_base) != len(query):
        raise ValueError("query_base must have one value per query point")

    compositions, support_temperatures, support_residuals = _curve_residuals(
        support, support_base, feature_columns)
    if not compositions:
        raise ValueError("support must contain at least one curve")

    support_x = scaler.transform(np.stack(compositions))
    query_x = scaler.transform(query[feature_columns].to_numpy(dtype=float))
    if distance_metric == "euclidean":
        distances = np.linalg.norm(query_x[:, None, :] - support_x[None, :, :], axis=2)
    else:
        query_norm = np.linalg.norm(query_x, axis=1, keepdims=True)
        support_norm = np.linalg.norm(support_x, axis=1, keepdims=True)
        cosine = (query_x @ support_x.T) / np.maximum(query_norm * support_norm.T, 1e-12)
        distances = 1.0 - np.clip(cosine, -1.0, 1.0)
    nearest_distance = distances.min(axis=1)
    logits = -0.5 * (distances / tau) ** 2
    if neighbor_k is not None:
        k = min(int(neighbor_k), distances.shape[1])
        nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        mask = np.ones_like(distances, dtype=bool)
        rows = np.arange(len(query))[:, None]
        mask[rows, nearest] = False
        logits[mask] = -np.inf
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)

    query_temperature = query.temperature.to_numpy(dtype=float)
    transferred = np.column_stack([
        np.interp(query_temperature, temperatures, residuals)
        for temperatures, residuals in zip(support_temperatures, support_residuals)
    ])
    chemistry_residual = np.sum(weights * transferred, axis=1)
    gate = (np.exp(-0.5 * (nearest_distance / gate_tau) ** 2)
            if gate_tau is not None else np.ones(len(query)))
    chemistry_residual *= gate

    support_residual = support.target.to_numpy(dtype=float) - np.asarray(support_base, dtype=float)
    support_t = support.temperature.to_numpy(dtype=float) / 1000.0
    query_t = query_temperature / 1000.0
    design_s = np.column_stack([np.ones(len(support)), support_t, support_t ** 2])
    design_q = np.column_stack([np.ones(len(query)), query_t, query_t ** 2])
    ridge = Ridge(alpha=ridge_alpha).fit(design_s, support_residual)
    ridge_residual = ridge.predict(design_q)

    chemistry_prediction = q_base + chemistry_residual
    ridge_prediction = q_base + ridge_residual
    prediction = blend * chemistry_prediction + (1.0 - blend) * ridge_prediction

    # Additive variants.  ``additive_raw`` adds both residual corrections.
    # ``additive_orthogonal`` first removes from the transported chemistry
    # residual the component collinear with the paper-level temperature anchor,
    # so the shared paper offset is not counted twice.
    anchor_energy = float(np.dot(ridge_residual, ridge_residual))
    if anchor_energy > 1e-12:
        projection = (float(np.dot(chemistry_residual, ridge_residual)) / anchor_energy) * ridge_residual
    else:
        projection = np.zeros_like(chemistry_residual)
    chemistry_orthogonal = chemistry_residual - projection
    variants = {
        "ridge_only": ridge_prediction,
        "chemistry_only": chemistry_prediction,
        "additive_raw": q_base + ridge_residual + chemistry_residual,
        "additive_orthogonal": q_base + ridge_residual + chemistry_orthogonal,
        "convex_blend": prediction,
    }
    return EpisodePrediction(prediction, chemistry_prediction, ridge_prediction,
                             weights, nearest_distance, gate,
                             ridge_residual, chemistry_residual, variants)


def variant_metrics(query: pd.DataFrame, prediction: EpisodePrediction) -> dict[str, float]:
    """Curve MAE and peak-ZT MAE for every additive/blend variant.

    Peak ZT is the maximum predicted value inside each query curve; the error
    is averaged over query curves.  Both metrics are returned per variant so a
    model family can be compared without re-running the transport.
    """
    target = query.target.to_numpy(dtype=float)
    curve_ids = query.curve_row_id.to_numpy()
    metrics: dict[str, float] = {}
    for name, values in prediction.variants.items():
        values = np.asarray(values, dtype=float)
        metrics[f"mae_{name}"] = float(np.mean(np.abs(target - values)))
        peak_errors = []
        for curve_id in pd.unique(curve_ids):
            mask = curve_ids == curve_id
            peak_errors.append(abs(float(target[mask].max()) - float(values[mask].max())))
        metrics[f"peak_mae_{name}"] = float(np.mean(peak_errors))
    return metrics


def evaluate_episode(support: pd.DataFrame, query: pd.DataFrame,
                     feature_columns: list[str], *, support_base: np.ndarray,
                     query_base: np.ndarray, scaler: FeatureScaler,
                     tau: float = 0.5, blend: float = 1.0,
                     ridge_alpha: float = 1.0,
                     distance_metric: str = "euclidean",
                     neighbor_k: int | None = None,
                     gate_tau: float | None = None) -> dict[str, float | int]:
    result = predict_episode(
        support, query, feature_columns, support_base=support_base,
        query_base=query_base, scaler=scaler, tau=tau, blend=blend,
        ridge_alpha=ridge_alpha, distance_metric=distance_metric,
        neighbor_k=neighbor_k, gate_tau=gate_tau)
    target = query.target.to_numpy(dtype=float)
    query_temperature = query.temperature.to_numpy(dtype=float)
    ranges = support.groupby("curve_row_id").temperature.agg(["min", "max"])
    covered = np.zeros(len(query), dtype=bool)
    for lower, upper in ranges.to_numpy(float):
        covered |= (query_temperature >= lower) & (query_temperature <= upper)
    curve_metrics = evaluate_curve_metrics(query, result.prediction)
    return {
        "mae": float(np.mean(np.abs(target - result.prediction))),
        "chemistry_mae": float(np.mean(np.abs(target - result.chemistry_prediction))),
        "ridge_mae": float(np.mean(np.abs(target - result.ridge_prediction))),
        "n_support_curves": int(support.curve_row_id.nunique()),
        "n_query_curves": int(query.curve_row_id.nunique()),
        "n_query_points": int(len(query)),
        "mean_nearest_distance": float(np.mean(result.nearest_distance)),
        "max_nearest_distance": float(np.max(result.nearest_distance)),
        "fraction_outside_support_temperature": float(1.0 - covered.mean()),
        **curve_metrics,
        **variant_metrics(query, result),
    }


def run_crct(root: Path = PROJECT_ROOT, *, seeds=(11, 22, 33), shot: int = SHOT,
             tau: float = 0.5, blend: float = 1.0,
             support_strategy: str = "random",
             distance_metric: str = "euclidean") -> pd.DataFrame:
    """Run DOI-held-out masked validation and save per-episode MAEs."""
    points, valid = _load_points(root)
    feature_columns = sorted(c for c in points.columns if c.startswith("f_"))
    feature_table = points[["material", *feature_columns]].drop_duplicates("material")
    valid = valid.merge(feature_table, on="material", how="inner")
    rows: list[dict[str, object]] = []

    for seed in seeds:
        train_dois, test_dois = _split_dois(points.doi.unique(), seed)
        train = points[points.doi.isin(train_dois)].copy()
        test = points[points.doi.isin(test_dois)]
        prior = lgb.LGBMRegressor(**PARAMS, random_state=seed)
        prior.fit(_features(train, feature_columns), train.target.to_numpy(dtype=float))
        scaler = FeatureScaler.fit(train, feature_columns)
        eligible = valid[valid.doi.isin(test_dois)].groupby("doi").filter(
            lambda group: group.material.nunique() > shot)

        for doi, curves in eligible.groupby("doi", sort=True):
            support_meta, query_meta = select_support(curves, shot, support_strategy, seed=seed)
            support = test[test.curve_row_id.isin(support_meta.curve_row_id)].copy()
            query = test[test.curve_row_id.isin(query_meta.curve_row_id)].copy()
            support_base = prior.predict(_features(support, feature_columns))
            query_base = prior.predict(_features(query, feature_columns))
            metrics = evaluate_episode(
                support, query, feature_columns, support_base=support_base,
                query_base=query_base, scaler=scaler, tau=tau, blend=blend,
                distance_metric=distance_metric)
            rows.append({"seed": seed, "doi": doi, "shot": shot,
                         "method": "crct", "support_selection": support_strategy,
                         "tau": tau, "blend": blend,
                         "distance_metric": distance_metric, **metrics})
        print(f"seed={seed}: train_dois={len(train_dois)}, test_dois={len(test_dois)}", flush=True)

    result = pd.DataFrame(rows)
    output = root / "results"
    suffix = "" if support_strategy == "random" else f"_{support_strategy}"
    threshold = float(result.mean_nearest_distance.quantile(0.75))
    result["distance_group"] = np.where(result.mean_nearest_distance >= threshold, "far_q25", "near_q75")
    result.to_csv(output / f"crct{suffix}_masked_validation.csv", index=False)
    summary = (result.groupby(["shot", "method", "support_selection", "tau", "blend", "distance_group"], as_index=False)
               .agg(mae_mean=("mae", "mean"), mae_median=("mae", "median"),
                    chemistry_mae_mean=("chemistry_mae", "mean"),
                    ridge_mae_mean=("ridge_mae", "mean"),
                    n_episodes=("doi", "size")))
    summary.to_csv(output / f"crct{suffix}_masked_validation_summary.csv", index=False)
    manifest = {
        "seeds": list(seeds), "shot": shot, "tau": tau, "blend": blend,
        "split": "DOI-held-out", "support_selection": support_strategy,
        "distance_metric": distance_metric,
        "distance": "Euclidean on training-DOI standardized composition features",
        "temperature_interpolation": "linear with endpoint clamping",
        "far_threshold_mean_nearest_distance_q75": threshold,
        "target": "ZT", "input_sha256": {name: sha256_file(path)
                                               for name, path in input_paths().items()},
    }
    (output / f"crct{suffix}_masked_validation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def run_crct_selection_comparison(
    root: Path = PROJECT_ROOT, *, seeds=(11, 22, 33), shot: int = SHOT,
    tau: float = 0.5, blend: float = 1.0,
    strategies=("random", "diverse_support"),
) -> pd.DataFrame:
    """Run CRCT for each support selector and write a paired comparison."""
    output = root / "results"
    frames = []
    for strategy in strategies:
        run_crct(root, seeds=seeds, shot=shot, tau=tau, blend=blend,
                 support_strategy=strategy)
        suffix = "" if strategy == "random" else f"_{strategy}"
        frames.append(pd.read_csv(output / f"crct{suffix}_masked_validation.csv"))
    comparison = pd.concat(frames, ignore_index=True)
    comparison.to_csv(output / "crct_support_selection_comparison.csv", index=False)
    summary = (comparison.groupby(
        ["shot", "method", "support_selection", "tau", "blend", "distance_group"],
        as_index=False)
        .agg(mae_mean=("mae", "mean"), mae_median=("mae", "median"),
             chemistry_mae_mean=("chemistry_mae", "mean"),
             ridge_mae_mean=("ridge_mae", "mean"),
             n_episodes=("doi", "size")))
    summary.to_csv(output / "crct_support_selection_comparison_summary.csv", index=False)
    return summary


if __name__ == "__main__":
    print(run_crct().to_string(index=False))
