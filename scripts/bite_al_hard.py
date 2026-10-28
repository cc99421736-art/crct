#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harder retrospective AL benchmark across two Bi-Te-based systems.

Protocol is fixed before running: 2 initial measured compositions, budget of 2
additional measurements, Bi-Sb-Te and Bi-Se-Te, 60 random initial supports per
paper.  The true maximum is excluded from the initial support so that every
episode has a target to find.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bisbte_active_learning as al
from src.dcsa.baseline import PARAMS, _features
from src.dcsa.crct import FeatureScaler
from src.dcsa.data_source import PROJECT_ROOT

al.N_INITIAL = int(os.environ.get("N_INITIAL", "2"))
al.BUDGET = int(os.environ.get("BUDGET", "2"))
al.N_SEEDS = 60
al.PREFIX = os.environ.get("PREFIX", "bite_al_hard")

SYSTEMS = {
    "Bi-Sb-Te": {"Bi", "Sb", "Te"},
    "Bi-Se-Te": {"Bi", "Se", "Te"},
}
STRATEGIES = ("crct", "crct_gate", "ridge", "prior", "nearest", "random")
METRICS = ["final_best", "final_regret", "normalized_auc", "hit_at_budget", "measurements_to_90"]


def main() -> None:
    rng = np.random.default_rng(20261023)
    rows: list[dict] = []
    for system_name, elements in SYSTEMS.items():
        al.SYSTEM = elements
        points, eligible, feature_columns = al.load_system()
        print(system_name, "papers", len(eligible))
        for doi_index, doi in enumerate(eligible):
            curves = points[points.doi.eq(doi)].copy()
            compositions = sorted(curves.composition_normalized.unique())
            true_peak = curves.groupby("composition_normalized").target.max()
            train = points[~points.doi.eq(doi)].copy()
            model = lgb.LGBMRegressor(**PARAMS, random_state=20261023 + doi_index)
            model.fit(_features(train, feature_columns), train.target.to_numpy(float))
            scaler = FeatureScaler.fit(train, feature_columns)
            for seed in range(al.N_SEEDS):
                initial = None
                for _ in range(300):
                    candidate = list(rng.choice(compositions, size=al.N_INITIAL, replace=False))
                    if float(true_peak.loc[candidate].max()) < float(true_peak.max()) - 1e-12:
                        initial = candidate
                        break
                if initial is None:
                    continue
                episode = al.run_episode(curves, true_peak, initial, STRATEGIES, model, scaler,
                                         feature_columns, rng)
                for row in episode:
                    row.update({"system": system_name, "doi": doi, "seed": seed})
                    rows.append(row)

    detail = pd.DataFrame(rows)
    out = PROJECT_ROOT / "results"
    detail.to_csv(out / f"{al.PREFIX}_episode.csv", index=False)

    per_doi = detail.groupby(["doi", "strategy"])[METRICS].mean().reset_index()
    boot_rng = np.random.default_rng(20261023)
    summary = []
    for strategy, group in per_doi.groupby("strategy", sort=True):
        for metric in METRICS:
            values = group[metric].dropna().to_numpy(float)
            if not len(values):
                continue
            draws = values[boot_rng.integers(0, len(values), size=(5000, len(values)))]
            boot = draws.mean(axis=1)
            summary.append({"strategy": strategy, "metric": metric,
                            "mean": float(values.mean()), "n_dois": int(len(values)),
                            "ci95_low": float(np.quantile(boot, .025)),
                            "ci95_high": float(np.quantile(boot, .975))})
    summary = pd.DataFrame(summary)
    summary.to_csv(out / f"{al.PREFIX}_summary.csv", index=False)

    paired = []
    for baseline in ("ridge", "prior", "nearest", "random"):
        for metric in METRICS:
            a = per_doi[per_doi.strategy.eq("crct")].set_index("doi")[metric]
            b = per_doi[per_doi.strategy.eq(baseline)].set_index("doi")[metric]
            diff = (a - b).dropna().to_numpy(float)
            if not len(diff):
                continue
            draws = diff[boot_rng.integers(0, len(diff), size=(5000, len(diff)))].mean(axis=1)
            paired.append({"baseline": baseline, "metric": metric,
                           "mean_diff": float(diff.mean()),
                           "ci95_low": float(np.quantile(draws, .025)),
                           "ci95_high": float(np.quantile(draws, .975)),
                           "p_better": float((draws > 0).mean()),
                           "wins": int((diff > 0).sum()),
                           "ties": int((diff == 0).sum()),
                           "losses": int((diff < 0).sum()),
                           "n_dois": int(len(diff))})
    paired = pd.DataFrame(paired)
    paired.to_csv(out / f"{al.PREFIX}_paired.csv", index=False)

    (out / f"{al.PREFIX}_manifest.json").write_text(json.dumps({
        "systems": list(SYSTEMS), "n_initial": al.N_INITIAL, "budget": al.BUDGET,
        "n_seeds": al.N_SEEDS, "strategies": list(STRATEGIES),
        "protocol": "fixed before running: initial true maximum excluded; 2 initial + 2 additional measurements",
        "interpretation": "harder retrospective AL; not prospective experimental validation",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("episodes", len(detail) // len(STRATEGIES))
    print(summary.pivot(index="metric", columns="strategy", values="mean").round(3).to_string())
    print("\npaired CRCT minus baseline")
    print(paired.pivot(index="metric", columns="baseline", values="mean_diff").round(3).to_string())
    print("\nCRCT vs Ridge win/tie/loss by metric")
    print(paired[paired.baseline.eq("ridge")][["metric", "wins", "ties", "losses", "p_better"]].to_string(index=False))


if __name__ == "__main__":
    main()
