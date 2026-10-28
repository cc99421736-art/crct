"""Train and evaluate DCSA under the five-support masked protocol."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from sklearn.linear_model import Ridge
from torch import nn

from src.dcsa.baseline import PARAMS, _features, _load_points, _split_dois
from src.dcsa.data_source import PROJECT_ROOT, input_paths, sha256_file
from src.dcsa.masking_validation import SHOT, select_support
from src.dcsa.model import DCSA

ABLATIONS = ("full", "base_only", "paper_only", "chemical_only")
SUPPORT_POINTS = 16


def _resample_curve(curve: pd.DataFrame, feature_columns: list[str], n_points: int):
    curve = curve.sort_values("temperature", kind="mergesort")
    temperature = curve.temperature.to_numpy(np.float32) / 1000.
    target = curve.target.to_numpy(np.float32)
    unique_temperature, indices = np.unique(temperature, return_index=True)
    temperature = unique_temperature
    target = target[indices]
    if len(temperature) == 1:
        grid = np.repeat(temperature[0], n_points).astype(np.float32)
        values = np.repeat(target[0], n_points).astype(np.float32)
    else:
        grid = np.linspace(temperature[0], temperature[-1], n_points, dtype=np.float32)
        values = np.interp(grid, temperature, target).astype(np.float32)
    composition = curve.iloc[0][feature_columns].to_numpy(np.float32)
    features = np.repeat(composition[None, :], n_points, axis=0)
    return features, grid, values


def episode_tensors(query: pd.DataFrame, support: pd.DataFrame,
                    feature_columns: list[str], shot: int = SHOT,
                    query_base: np.ndarray | None = None,
                    support_base: np.ndarray | None = None):
    """Convert one query/support episode to tensors with fixed-length curve traces."""
    if support.material.nunique() != shot:
        raise ValueError("support must contain exactly shot curves")
    qx = torch.tensor(query[feature_columns].to_numpy(np.float32))
    qt = torch.tensor((query.temperature.to_numpy(np.float32) / 1000.)[:, None])
    qy = torch.tensor(query.target.to_numpy(np.float32))
    if query_base is not None:
        query_base_tensor = torch.tensor(np.asarray(query_base, dtype=np.float32))[:, None]
    else:
        query_base_tensor = None
    traces = []
    support_groups = list(support.groupby("curve_row_id", sort=False))
    support_base_map = {}
    if support_base is not None:
        base_values = np.asarray(support_base, dtype=float)
        if len(base_values) == len(support):
            support_base_map = (pd.DataFrame({"curve_row_id": support.curve_row_id.to_numpy(),
                                               "base": base_values})
                                .groupby("curve_row_id").base.mean().to_dict())
        elif len(base_values) == len(support_groups):
            support_base_map = dict(zip([key for key, _ in support_groups], base_values))
        else:
            raise ValueError("support_base must have one value per point or curve")
    for curve_id, curve in support_groups:
        trace = _resample_curve(curve, feature_columns, SUPPORT_POINTS)
        if support_base is not None:
            trace = (trace[0], trace[1], trace[2] - support_base_map[curve_id])
        traces.append(trace)
    sx = torch.tensor(np.stack([r[0] for r in traces])[None, ...]).expand(len(query), -1, -1, -1)
    st = torch.tensor(np.stack([r[1] for r in traces])[None, ...], dtype=torch.float32).expand(len(query), -1, -1)
    sy = torch.tensor(np.stack([r[2] for r in traces])[None, ...], dtype=torch.float32).expand(len(query), -1, -1)
    st = st.unsqueeze(-1)
    sy = sy.unsqueeze(-1)
    return qx, qt, sx, st, sy, qy, query_base_tensor


def _standardize(points: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict]:
    out = points.copy()
    stats = {}
    for col in columns:
        mean = float(out[col].mean())
        std = float(out[col].std()) or 1.0
        out[col] = (out[col] - mean) / std
        stats[col] = [mean, std]
    return out, stats


def _ridge_anchor(support: pd.DataFrame, query: pd.DataFrame,
                  support_base: np.ndarray, query_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the paper-level temperature residual and return anchored predictions."""
    support_residual = support.target.to_numpy(float) - np.asarray(support_base, dtype=float)
    support_t = support.temperature.to_numpy(float) / 1000.0
    query_t = query.temperature.to_numpy(float) / 1000.0
    design_s = np.column_stack([np.ones(len(support)), support_t, support_t ** 2])
    design_q = np.column_stack([np.ones(len(query)), query_t, query_t ** 2])
    ridge = Ridge(alpha=1.0).fit(design_s, support_residual)
    return (np.asarray(query_base, dtype=float) + ridge.predict(design_q),
            ridge.predict(design_q))


def _fit_model(train: pd.DataFrame, feature_columns: list[str], seed: int,
               *, epochs: int = 4, max_train_dois: int = 700,
               prior: lgb.LGBMRegressor | None = None) -> DCSA:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = DCSA(len(feature_columns), hidden=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    dois = np.array(sorted(train.doi.unique()))
    if len(dois) > max_train_dois:
        dois = rng.choice(dois, max_train_dois, replace=False)
    for _ in range(epochs):
        rng.shuffle(dois)
        model.train()
        for doi in dois:
            curves = train[train.doi.eq(doi)]
            if curves.material.nunique() <= SHOT:
                continue
            support_meta, query_meta = select_support(curves, SHOT, "random", seed=int(rng.integers(1_000_000)))
            support = curves[curves.curve_row_id.isin(support_meta.curve_row_id)]
            query = curves[curves.curve_row_id.isin(query_meta.curve_row_id)]
            if len(query) > 128:
                query = query.iloc[rng.choice(len(query), 128, replace=False)]
            base = prior.predict(_features(query, feature_columns)) if prior is not None else None
            support_base = prior.predict(_features(support, feature_columns)) if prior is not None else None
            anchor, _ = _ridge_anchor(support, query, support_base, base)
            batch = episode_tensors(query, support, feature_columns, query_base=anchor,
                                    support_base=support_base)
            prediction = model(*batch[:5], query_base=batch[6])
            loss = loss_fn(prediction, batch[5])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def run_training(root: Path = PROJECT_ROOT, *, seeds=(11, 22, 33), epochs: int = 4) -> pd.DataFrame:
    points, valid = _load_points(root)
    feature_columns = sorted(c for c in points.columns if c.startswith("f_"))
    # Standardize only using the training partition inside each seed.
    rows = []
    for seed in seeds:
        train_dois, test_dois = _split_dois(points.doi.unique(), seed)
        train_raw = points[points.doi.isin(train_dois)].copy()
        train, stats = _standardize(train_raw, feature_columns + ["temperature", "target"])
        target_mean, target_std = stats["target"]
        test = points[points.doi.isin(test_dois)].copy()
        for col, (mean, std) in stats.items():
            test[col] = (test[col] - mean) / std
        prior = lgb.LGBMRegressor(**PARAMS, random_state=seed)
        prior.fit(_features(train, feature_columns), train.target.to_numpy(float))
        model = _fit_model(train, feature_columns, seed, epochs=epochs, prior=prior)
        eligible = valid[valid.doi.isin(test_dois)].groupby("doi").filter(lambda g: g.material.nunique() > SHOT)
        for doi, curves in eligible.groupby("doi", sort=True):
            support_meta, query_meta = select_support(curves, SHOT, "random", seed=seed)
            support = test[test.curve_row_id.isin(support_meta.curve_row_id)]
            query = test[test.curve_row_id.isin(query_meta.curve_row_id)]
            base = prior.predict(_features(query, feature_columns))
            support_base = prior.predict(_features(support, feature_columns))
            anchor, _ = _ridge_anchor(support, query, support_base, base)
            batch = episode_tensors(query, support, feature_columns, query_base=anchor,
                                    support_base=support_base)
            with torch.no_grad():
                for ablation in ABLATIONS:
                    pred = model(*batch[:5], query_base=batch[6], ablation=ablation).numpy()
                    pred_raw = pred * target_std + target_mean
                    true_raw = batch[5].numpy() * target_std + target_mean
                    mae = float(np.mean(np.abs(pred_raw - true_raw)))
                    rows.append({"seed": seed, "doi": doi, "shot": SHOT,
                                 "model": "dcsa", "ablation": ablation,
                                 "n_query_points": len(query), "mae": mae})
        print(f"seed={seed}: train_dois={len(train_dois)}, test_dois={len(test_dois)}", flush=True)
    result = pd.DataFrame(rows)
    output = root / "results"
    result.to_csv(output / "dcsa_masked_validation.csv", index=False)
    result["episode_id"] = result.seed.astype(str) + ":" + result.doi.astype(str)
    summary = (result.groupby(["shot", "model", "ablation"], as_index=False)
               .agg(mae_mean=("mae", "mean"), mae_median=("mae", "median"),
                    n_episodes=("episode_id", "nunique"), n_rows=("mae", "size")))
    summary.to_csv(output / "dcsa_masked_validation_summary.csv", index=False)
    manifest = {"seeds": list(seeds), "shot": SHOT, "epochs": epochs,
                "split": "DOI-held-out", "feature_columns": feature_columns,
                "input_sha256": {name: sha256_file(path) for name, path in input_paths().items()}}
    (output / "dcsa_masked_validation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(run_training().to_string(index=False))
