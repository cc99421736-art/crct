"""Strict retrospective audit for CRCT candidate ranking."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from src.dcsa.data_source import PROJECT_ROOT


def _ndcg(values: np.ndarray, order: np.ndarray, k: int) -> float:
    k = min(k, len(values))
    gain = np.maximum(values[order[:k]], 0.0)
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gain * discount))
    ideal = np.sort(np.maximum(values, 0.0))[::-1][:k]
    return dcg / float(np.sum(ideal * discount)) if np.sum(ideal * discount) else 1.0


def _episode_metrics(frame: pd.DataFrame, method: str, seed: int) -> dict[str, float | int | str]:
    true = frame.true_peak_zt.to_numpy(float)
    if method == "risk_aware_peak":
        score = frame.risk_aware_score.to_numpy(float)
        order = np.argsort(-score, kind="mergesort")
    elif method == "predicted_peak":
        score = frame.predicted_peak_zt.to_numpy(float)
        order = np.argsort(-score, kind="mergesort")
    elif method == "nearest_distance":
        score = -frame.nearest_distance.to_numpy(float)
        order = np.argsort(-score, kind="mergesort")
    elif method == "gate_distance_score":
        score = frame.screening_score.to_numpy(float)
        order = np.argsort(-score, kind="mergesort")
    elif method == "random":
        order = np.random.default_rng(seed).permutation(len(frame))
    else:
        raise ValueError(method)
    pred_rank = np.empty(len(order), dtype=int); pred_rank[order] = np.arange(len(order))
    rank_score = -pred_rank.astype(float)
    rho = spearmanr(rank_score, true).statistic if len(frame) > 1 else np.nan
    tau = kendalltau(rank_score, true).statistic if len(frame) > 1 else np.nan
    result = {"method": method, "n_candidates": int(len(frame)),
              "spearman_predicted_true": float(rho), "kendall_predicted_true": float(tau),
              "top1_hit": float(true[order[0]] == true.max()),
              "top1_regret": float(true.max() - true[order[0]]),
              "ndcg1": _ndcg(true, order, 1)}
    for k in (3, 5):
        if k <= len(frame):
            chosen = order[:k]
            result[f"top{k}_recall_true_top20"] = float(np.mean(true[chosen] >= np.quantile(true, .8)))
            result[f"top{k}_hit"] = float(true[chosen].max() == true.max())
            result[f"top{k}_regret"] = float(true.max() - true[chosen].max())
            result[f"ndcg{k}"] = _ndcg(true, order, k)
        else:
            for name in (f"top{k}_recall_true_top20", f"top{k}_hit", f"top{k}_regret", f"ndcg{k}"):
                result[name] = np.nan
    return result


def run(root: Path = PROJECT_ROOT, n_boot: int = 5000, seed: int = 20261017,
        input_name: str = "crct_nested_candidate_predictions.csv",
        output_prefix: str = "ranking_audit",
        methods: tuple[str, ...] = ("crct_score", "predicted_peak",
                                    "nearest_distance", "random")) -> pd.DataFrame:
    path = root / "results" / input_name
    candidates = pd.read_csv(path)
    required = {"seed", "doi", "predicted_peak_zt", "true_peak_zt", "nearest_distance"}
    if "risk_aware_peak" in methods:
        required.add("risk_aware_score")
    if "crct_score" in methods or "gate_distance_score" in methods:
        required.add("screening_score")
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    rows = []
    for (episode_seed, doi), group in candidates.groupby(["seed", "doi"], sort=True):
        if len(group) < 2:
            continue
        for method in methods:
            row = _episode_metrics(group, method, int(episode_seed) + methods.index(method))
            row.update({"seed": episode_seed, "doi": doi,
                        "candidate_bin": "small_2-5" if len(group) <= 5 else ("medium_6-10" if len(group) <= 10 else "large_>10")})
            rows.append(row)
    detail = pd.DataFrame(rows)
    out = root / "results"
    detail.to_csv(out / f"{output_prefix}_episode.csv", index=False)
    rng = np.random.default_rng(seed)
    summaries = []
    metrics = [c for c in detail.columns if c.startswith(("spearman", "kendall", "top", "ndcg"))]
    for (method, candidate_bin), group in detail.groupby(["method", "candidate_bin"], sort=True):
        doi_values = group.groupby("doi", sort=True)[metrics].mean()
        for metric in metrics:
            vals = doi_values[metric].dropna().to_numpy(float)
            if not len(vals):
                continue
            draws = rng.integers(0, len(vals), size=(n_boot, len(vals)))
            boot = vals[draws].mean(axis=1)
            summaries.append({"method": method, "candidate_bin": candidate_bin, "metric": metric,
                "mean": float(vals.mean()), "n_dois": int(len(vals)),
                "ci95_low": float(np.quantile(boot, .025)), "ci95_high": float(np.quantile(boot, .975))})
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / f"{output_prefix}_summary.csv", index=False)
    (out / f"{output_prefix}_manifest.json").write_text(json.dumps({
        "source": str(path.name), "methods": list(methods), "cluster": "doi",
        "metrics": metrics, "n_boot": n_boot,
        "interpretation": "retrospective masked ranking audit; not prospective experimental validation",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary



if __name__ == "__main__":
    print(run().to_string(index=False))
