#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply the frozen four-feature risk layer to the multi-system candidates."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
FEATURES = ["R_d", "R_T", "R_g", "R_e"]

SELECTED = [
    ("Ag-S-Se", "Ag2S0.075Se0.925"), ("Ag-S-Se", "Ag2S0.423Se0.577"),
    ("Ag-S-Te", "Ag2S0.208Te0.792"), ("Ag-S-Te", "Ag2S0.435Te0.565"),
    ("Bi-Sb-Te", "Bi0.293Sb1.707Te3"), ("Bi-Sb-Te", "Bi0.833Sb1.167Te3"),
    ("Bi-Se-Te", "Bi2Se0.219Te2.781"), ("Bi-Se-Te", "Bi2Se0.451Te2.549"),
    ("Cu-S-Se", "Cu1.98S0.172Se0.828"), ("Cu-S-Se", "Cu1.98S0.413Se0.587"),
    ("Mg-Si-Sn", "Mg2Si0.183Sn0.817"), ("Mg-Si-Sn", "Mg2Si0.829Sn0.171"),
    ("Pb-Se-Te", "PbSe0.509Te0.491"), ("Pb-Se-Te", "PbSe0.684Te0.316"),
    ("Sb-Sn-Te", "Sb0.124Sn0.876Te"), ("Sb-Sn-Te", "Sb0.28Sn0.72Te"),
]


def main() -> None:
    pool = pd.read_csv(RESULTS / "multisystem_crct_candidate_screening_extended.csv")
    robustness = pd.read_csv(RESULTS / "multisystem_crct_robustness_summary.csv")
    inner = pd.read_csv(RESULTS / "risk_upgrade_inner.csv")
    weights = pd.read_csv(RESULTS / "risk_upgrade_weights.csv")

    keep = pd.DataFrame(SELECTED, columns=["system", "formula"])
    data = keep.merge(pool[["formula", "mean_nearest_distance",
                            "fraction_outside_support_temperature", "residual_gate",
                            "predicted_peak_zt", "peak_temperature_K"]], on="formula")
    data = data.merge(robustness[["formula", "peak_zt_sd"]], on="formula")
    data["R_d"] = data.mean_nearest_distance
    data["R_T"] = data.fraction_outside_support_temperature
    data["R_g"] = 1.0 - data.residual_gate
    data["R_e"] = data.peak_zt_sd

    scores = []
    for _, row in weights.iterrows():
        seed = int(row.seed) + 10000
        subset = inner[inner.seed.eq(seed)]
        mean = subset[FEATURES].mean()
        scale = subset[FEATURES].std(ddof=0).replace(0, 1.0)
        z = (data[FEATURES] - mean) / scale
        score = float(row.intercept) + z.to_numpy(float) @ np.array(
            [row.w_R_d, row.w_R_T, row.w_R_g, row.w_R_e], dtype=float)
        scores.append(pd.Series(score, index=data.index, name=f"score_seed{seed}"))
    score_frame = pd.concat(scores, axis=1)
    data["risk_score"] = score_frame.mean(axis=1)
    data["risk_score_sd"] = score_frame.std(axis=1, ddof=0)

    inner_scores = []
    for _, row in weights.iterrows():
        seed = int(row.seed) + 10000
        subset = inner[inner.seed.eq(seed)]
        mean = subset[FEATURES].mean()
        scale = subset[FEATURES].std(ddof=0).replace(0, 1.0)
        z = (subset[FEATURES] - mean) / scale
        inner_scores.append(z.to_numpy(float) @ np.array(
            [row.w_R_d, row.w_R_T, row.w_R_g, row.w_R_e], dtype=float)
            + float(row.intercept))
    inner_scores = np.concatenate(inner_scores)
    low_cut, high_cut = np.quantile(inner_scores, [1 / 3, 2 / 3])
    data["risk"] = np.where(data.risk_score <= low_cut, "low",
                            np.where(data.risk_score <= high_cut, "medium", "high"))
    data["risk_threshold_low"] = low_cut
    data["risk_threshold_high"] = high_cut
    data.to_csv(RESULTS / "multisystem_candidate_risk_port.csv", index=False)
    print(data[["system", "formula", "R_d", "R_T", "R_g", "R_e",
                "risk_score", "risk"]].to_string(index=False))
    print("inner tertiles", round(low_cut, 4), round(high_cut, 4))


if __name__ == "__main__":
    main()

