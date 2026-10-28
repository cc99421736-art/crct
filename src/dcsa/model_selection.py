"""Nested selection between paper-anchor, chemistry-transport and additive models.

The five comparable model families are

    ridge_only           y0 + beta_r
    chemistry_only       y0 + g * r_tr
    additive_raw         y0 + beta_r + g * r_tr
    additive_orthogonal  y0 + beta_r + g * (r_tr - proj(r_tr, beta_r))
    convex_blend         blend * chemistry_only + (1 - blend) * ridge_only

For every outer seed the model family and the transport hyper-parameters are
selected on an inner DOI-held-out split of the outer training DOIs only.  The
outer test DOIs are evaluated afterwards with the frozen selection, once.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.dcsa.baseline import PARAMS, _features, _load_points, _split_dois
from src.dcsa.crct import FeatureScaler, predict_episode, variant_metrics
from src.dcsa.data_source import PROJECT_ROOT, input_paths, sha256_file
from src.dcsa.masking_validation import SHOT, select_support

FAMILIES = ("chemistry_only", "additive_orthogonal", "additive_raw",
            "ridge_only", "convex_blend")
TRANSPORT_GRID = tuple(
    {"tau": tau, "gate_tau": gate, "neighbor_k": 2}
    for tau in (0.25, 0.5)
    for gate in (None, 1.0)
)
BLEND = 1.0
TIE_TOL = 1e-4


def _evaluate_split(points: pd.DataFrame, valid: pd.DataFrame,
                    train_dois: set[str], eval_dois: set[str],
                    feature_columns: list[str], seed: int,
                    shot: int = SHOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every (transport config, model family) on one DOI split."""
    train = points[points.doi.isin(train_dois)]
    test = points[points.doi.isin(eval_dois)]
    prior = lgb.LGBMRegressor(**PARAMS, random_state=seed)
    prior.fit(_features(train, feature_columns), train.target.to_numpy(float))
    scaler = FeatureScaler.fit(train, feature_columns)
    eligible = valid[valid.doi.isin(eval_dois)].groupby("doi").filter(
        lambda group: group.material.nunique() > shot)
    rows: list[dict[str, object]] = []
    indices: list[dict[str, object]] = []
    for doi, curves in eligible.groupby("doi", sort=True):
        support_meta, query_meta = select_support(curves, shot, "diverse_support", seed=seed)
        indices.append({"seed": seed, "doi": doi, "shot": shot,
                        "support_selection": "diverse_support",
                        "support_curve_ids": "|".join(str(int(v)) for v in support_meta.curve_row_id),
                        "query_curve_ids": "|".join(str(int(v)) for v in query_meta.curve_row_id)})
        support = test[test.curve_row_id.isin(support_meta.curve_row_id)].copy()
        query = test[test.curve_row_id.isin(query_meta.curve_row_id)].copy()
        support_base = prior.predict(_features(support, feature_columns))
        query_base = prior.predict(_features(query, feature_columns))
        for config in TRANSPORT_GRID:
            prediction = predict_episode(
                support, query, feature_columns, support_base=support_base,
                query_base=query_base, scaler=scaler, tau=config["tau"],
                blend=BLEND, neighbor_k=config["neighbor_k"],
                gate_tau=config["gate_tau"])
            metrics = variant_metrics(query, prediction)
            for family in FAMILIES:
                rows.append({
                    "seed": seed, "doi": doi, "family": family,
                    "tau": config["tau"], "gate_tau": config["gate_tau"],
                    "neighbor_k": config["neighbor_k"],
                    "mae": metrics[f"mae_{family}"],
                    "peak_mae": metrics[f"peak_mae_{family}"],
                    "n_query_curves": int(query.curve_row_id.nunique()),
                    "n_query_points": int(len(query)),
                    "mean_nearest_distance": float(np.mean(prediction.nearest_distance)),
                })
    return pd.DataFrame(rows), pd.DataFrame(indices)


def _pick(frame: pd.DataFrame, family: str | None = None) -> dict[str, object]:
    """Pick the inner-best transport config, optionally restricted to a family."""
    subset = frame if family is None else frame[frame.family.eq(family)]
    if subset.empty:
        raise ValueError(f"no inner rows for family {family!r}")
    summary = (subset.groupby(["family", "tau", "gate_tau", "neighbor_k"], as_index=False, dropna=False)
               .agg(mae=("mae", "mean"), peak_mae=("peak_mae", "mean"), n=("mae", "size")))
    summary = summary.sort_values(["mae", "peak_mae", "family", "tau"],
                                  ascending=[True, True, True, True], kind="mergesort")
    row = summary.iloc[0]
    return {"family": str(row.family), "tau": float(row.tau),
            "gate_tau": None if pd.isna(row.gate_tau) else float(row.gate_tau),
            "neighbor_k": int(row.neighbor_k), "inner_mae": float(row.mae),
            "inner_peak_mae": float(row.peak_mae)}


def _paired_by_doi(outer: pd.DataFrame, left: str, right: str) -> dict[str, object]:
    """DOI-clustered bootstrap of ``mae_right - mae_left`` for two policies."""
    pivot = outer.pivot_table(index="doi", columns="policy", values="mae", aggfunc="mean")
    improvement = (pivot[right] - pivot[left]).dropna().to_numpy(float)
    rng = np.random.default_rng(20261019)
    draws = rng.integers(0, len(improvement), size=(5000, len(improvement)))
    boot = improvement[draws].mean(axis=1)
    return {"n_dois": int(len(improvement)),
            "mean_improvement": float(improvement.mean()),
            "ci95_low": float(np.quantile(boot, .025)),
            "ci95_high": float(np.quantile(boot, .975)),
            "win_rate": float(np.mean(improvement > 0))}


def run(root: Path = PROJECT_ROOT, *, seeds=(11, 22, 33)) -> pd.DataFrame:
    points, valid = _load_points(root)
    feature_columns = sorted(column for column in points.columns if column.startswith("f_"))
    feature_table = points[["material", *feature_columns]].drop_duplicates("material")
    valid = valid.merge(feature_table, on="material", how="inner")
    all_dois = points.doi.unique()

    inner_frames: list[pd.DataFrame] = []
    outer_frames: list[pd.DataFrame] = []
    index_frames: list[pd.DataFrame] = []
    selections: list[dict[str, object]] = []

    for seed in seeds:
        outer_train, outer_test = _split_dois(all_dois, seed)
        inner_train, inner_val = _split_dois(np.array(sorted(outer_train)), seed + 10000)
        inner, inner_idx = _evaluate_split(points, valid, inner_train, inner_val,
                                           feature_columns, seed + 10000)
        inner["outer_seed"] = seed
        inner_frames.append(inner)
        index_frames.append(inner_idx.assign(split="inner"))

        best_overall = _pick(inner)
        best_chemistry = _pick(inner, "chemistry_only")
        best_additive = _pick(inner, "additive_orthogonal")
        policies = {"selected_overall": best_overall,
                    "chemistry_only": best_chemistry,
                    "additive_orthogonal": best_additive}

        outer, outer_idx = _evaluate_split(points, valid, outer_train, outer_test,
                                           feature_columns, seed)
        index_frames.append(outer_idx.assign(split="outer"))
        for policy, choice in policies.items():
            rows = outer[outer.family.eq(choice["family"])
                         & outer.tau.eq(choice["tau"])
                         & outer.neighbor_k.eq(choice["neighbor_k"])
                         & (outer.gate_tau.isna() if choice["gate_tau"] is None
                            else outer.gate_tau.eq(choice["gate_tau"]))].copy()
            rows["policy"] = policy
            rows["outer_seed"] = seed
            outer_frames.append(rows)

        for policy, choice in policies.items():
            selections.append({"seed": seed, "policy": policy,
                               "inner_train_dois": len(inner_train),
                               "inner_validation_dois": len(inner_val),
                               **choice})
        print(f"seed={seed}: overall={best_overall['family']}/tau={best_overall['tau']}/"
              f"gate={best_overall['gate_tau']}  chemistry={best_chemistry['tau']}/"
              f"{best_chemistry['gate_tau']}  additive={best_additive['tau']}/"
              f"{best_additive['gate_tau']}", flush=True)

    inner_frame = pd.concat(inner_frames, ignore_index=True)
    outer_frame = pd.concat(outer_frames, ignore_index=True)
    selection_frame = pd.DataFrame(selections)

    summary_rows = []
    for policy, group in outer_frame.groupby("policy", sort=True):
        summary_rows.append({"policy": policy, "n_dois": int(group.doi.nunique()),
                             "n_episodes": int(len(group)),
                             "mae": float(group.mae.mean()),
                             "peak_mae": float(group.peak_mae.mean())})
    summary = pd.DataFrame(summary_rows)
    comparisons = pd.DataFrame([
        {"comparison": "additive_orthogonal vs chemistry_only",
         **_paired_by_doi(outer_frame, "chemistry_only", "additive_orthogonal")},
        {"comparison": "selected_overall vs chemistry_only",
         **_paired_by_doi(outer_frame, "chemistry_only", "selected_overall")},
    ])

    output = root / "results"
    inner_frame.to_csv(output / "crct_model_selection_inner.csv", index=False)
    outer_frame.to_csv(output / "crct_model_selection_outer.csv", index=False)
    pd.concat(index_frames, ignore_index=True).to_csv(
        output / "crct_model_selection_episode_indices.csv", index=False)
    selection_frame.to_csv(output / "crct_model_selection_selected.csv", index=False)
    summary.to_csv(output / "crct_model_selection_summary.csv", index=False)
    comparisons.to_csv(output / "crct_model_selection_comparisons.csv", index=False)
    (output / "crct_model_selection_manifest.json").write_text(json.dumps({
        "seeds": list(seeds), "shot": SHOT, "support_selection": "diverse_support",
        "families": list(FAMILIES), "transport_grid": list(TRANSPORT_GRID),
        "blend_for_convex": BLEND,
        "inner_split": "DOI-held-out within outer train",
        "outer_split": "DOI-held-out; evaluated once after selection",
        "selection_rule": "min inner curve MAE; ties by peak-ZT MAE then family name",
        "additive_orthogonal": "y0 + beta_r + g*(r_tr - projection(r_tr, beta_r))",
        "bootstrap": "DOI-clustered, 5000 draws, seed 20261019",
        "input_sha256": {name: sha256_file(path) for name, path in input_paths().items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False))
    print(comparisons.to_string(index=False))
    return summary


if __name__ == "__main__":
    run()
