#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the multisystem robustness ensemble on the selected 24 candidates."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.dcsa.multisystem_robustness as mr
import src.dcsa.multisystem_screening as ms
from src.dcsa.data_source import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "results"


def main() -> None:
    selection = pd.read_csv(RESULTS / "multisystem_crct_candidate_selection.csv")
    systems = []
    for spec in ms.SYSTEMS:
        formulas = selection[selection.system.eq(spec["system"])].formula.tolist()
        if formulas:
            systems.append({**spec, "candidates": tuple(formulas)})
    ms.SYSTEMS = tuple(systems)
    mr.SYSTEMS = tuple(systems)
    for name in ("multisystem_crct_robustness_summary.csv",
                 "multisystem_crct_robustness_detail.csv"):
        path = RESULTS / name
        if path.exists():
            shutil.copy2(path, RESULTS / name.replace(".csv", "_16.csv"))
    summary = mr.run_robustness()
    print(summary[["system", "formula", "n_runs", "peak_zt_mean", "peak_zt_sd"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
