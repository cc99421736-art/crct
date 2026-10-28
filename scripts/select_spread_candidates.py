#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Select three spread candidates per system from the extended pool."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"

PATTERNS = {
    "Ag-S-Te": r"Ag2S([\d.]+)Te",
    "Ag-S-Se": r"Ag2S([\d.]+)Se",
    "Cu-S-Se": r"Cu1\.98S([\d.]+)Se",
    "Bi-Sb-Te": r"Bi([\d.]+)Sb",
    "Bi-Se-Te": r"Bi2Se([\d.]+)Te",
    "Pb-Se-Te": r"PbSe([\d.]+)Te",
    "Sb-Sn-Te": r"Sb([\d.]+)Sn",
    "Mg-Si-Sn": r"Mg2Si([\d.]+)Sn",
}


def fraction(system: str, formula: str) -> float:
    match = re.search(PATTERNS[system], formula)
    if not match:
        raise ValueError(f"cannot parse {system} {formula}")
    return float(match.group(1))


def main() -> None:
    pool = pd.read_csv(RESULTS / "multisystem_crct_candidate_screening_extended.csv")
    picks = []
    for system, group in pool.groupby("system"):
        group = group.drop_duplicates("formula").copy()
        group["fraction"] = [fraction(system, value) for value in group.formula]
        low, high = float(group.fraction.min()), float(group.fraction.max())
        targets = [low + 0.20 * (high - low), low + 0.50 * (high - low),
                   low + 0.80 * (high - low)]
        chosen: list[float] = []
        for rank, target in enumerate(targets, start=1):
            window = group[(group.fraction - target).abs() <= 0.08]
            if window.empty:
                window = group
            window = window[~window.fraction.isin(chosen)]
            if window.empty:
                continue
            row = window.sort_values("predicted_peak_zt", ascending=False).iloc[0]
            chosen.append(float(row.fraction))
            picks.append({**row.to_dict(), "selection_rank": rank,
                          "selection_target": target})
    selection = pd.DataFrame(picks)
    selection.to_csv(RESULTS / "multisystem_crct_candidate_selection.csv", index=False)
    columns = ["system", "formula", "fraction", "predicted_peak_zt",
               "peak_temperature_K", "risk", "selection_rank"]
    print(selection[columns].to_string(index=False))
    print("n", len(selection), "systems", selection.system.nunique())


if __name__ == "__main__":
    main()
