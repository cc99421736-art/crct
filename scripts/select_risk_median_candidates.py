#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Select 3 distinct candidates per system: lowest risk, median peak, top peak."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"

PATTERNS = {
    "Ag-S-Te": r"Ag2S([\d.]+)Te", "Ag-S-Se": r"Ag2S([\d.]+)Se",
    "Cu-S-Se": r"Cu1\.98S([\d.]+)Se", "Bi-Sb-Te": r"Bi([\d.]+)Sb",
    "Bi-Se-Te": r"Bi2Se([\d.]+)Te", "Pb-Se-Te": r"PbSe([\d.]+)Te",
    "Sb-Sn-Te": r"Sb([\d.]+)Sn", "Mg-Si-Sn": r"Mg2Si([\d.]+)Sn",
}


def fraction(system: str, formula: str) -> float:
    return float(re.search(PATTERNS[system], formula).group(1))


def main() -> None:
    pool = pd.read_csv(RESULTS / "multisystem_crct_candidate_screening_extended.csv")
    picks = []
    for system, group in pool.groupby("system"):
        group = group.drop_duplicates("formula").copy()
        group["fraction"] = [fraction(system, value) for value in group.formula]
        chosen: list[float] = []

        def add(row, reason: str) -> None:
            if row.formula not in {item["formula"] for item in picks}:
                chosen.append(float(row.fraction))
                picks.append({**row.to_dict(), "selection_reason": reason,
                              "selection_rank": len(chosen)})

        add(group.sort_values("risk_score").iloc[0], "lowest_risk")
        median = float(group.predicted_peak_zt.median())
        for _, row in group.iloc[(group.predicted_peak_zt - median).abs().argsort()].iterrows():
            if len(chosen) >= 2:
                break
            if abs(row.fraction - chosen[0]) >= 0.10:
                add(row, "median_peak")
        for _, row in group.sort_values("predicted_peak_zt", ascending=False).iterrows():
            if len(chosen) >= 3:
                break
            if all(abs(row.fraction - value) >= 0.10 for value in chosen):
                add(row, "top_peak")
        for _, row in group.sort_values("risk_score").iterrows():
            if len(chosen) >= 3:
                break
            add(row, "fill_risk")
    selection = pd.DataFrame(picks)
    selection.to_csv(RESULTS / "multisystem_crct_candidate_selection24.csv", index=False)
    print(selection[["system", "formula", "fraction", "predicted_peak_zt",
                     "peak_temperature_K", "risk", "risk_score", "selection_reason"]]
          .to_string(index=False))
    print("n", len(selection))


if __name__ == "__main__":
    main()
