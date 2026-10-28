#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Expand the multi-system candidate pool with low-discrepancy compositions.

The original 16-candidate screening is preserved by a backup; the extended
pool is written to a separate CSV, and a 3-per-system selection is proposed.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.dcsa.multisystem_screening as ms
from src.dcsa.data_source import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "results"
BACKUP = RESULTS / "multisystem_crct_candidate_screening_16.csv"


def fractions(n: int, low: float, high: float, seed: int) -> list[float]:
    sampler = qmc.Sobol(d=1, scramble=True, seed=seed)
    values = low + (high - low) * sampler.random(n).ravel()
    return [round(float(value), 3) for value in values]


def formula_binary(prefix: str, x: float, left: str, right: str) -> str:
    return f"{prefix}{left}{x:g}{right}{1 - x:g}"


def build_systems() -> tuple:
    systems = []
    for spec in ms.SYSTEMS:
        system = spec["system"]
        if system == "Ag-S-Te":
            xs = fractions(40, 0.03, 0.60, 101)
            candidates = [formula_binary("Ag2S", x, "", "Te") for x in xs]
            candidates = [f"Ag2S{x:g}Te{1 - x:g}" for x in xs]
        elif system == "Ag-S-Se":
            xs = fractions(40, 0.03, 0.60, 102)
            candidates = [f"Ag2S{x:g}Se{1 - x:g}" for x in xs]
        elif system == "Cu-S-Se":
            xs = fractions(40, 0.03, 0.60, 103)
            candidates = [f"Cu1.98S{x:g}Se{1 - x:g}" for x in xs]
        elif system == "Bi-Sb-Te":
            xs = fractions(40, 0.05, 0.95, 104)
            candidates = [f"Bi{x:g}Sb{2 - x:g}Te3" for x in xs]
        elif system == "Bi-Se-Te":
            xs = fractions(40, 0.05, 0.60, 105)
            candidates = [f"Bi2Se{x:g}Te{3 - x:g}" for x in xs]
        elif system == "Pb-Se-Te":
            xs = fractions(40, 0.05, 0.95, 106)
            candidates = [f"PbSe{x:g}Te{1 - x:g}" for x in xs]
        elif system == "Sb-Sn-Te":
            xs = fractions(40, 0.01, 0.40, 107)
            candidates = [f"Sb{x:g}Sn{1 - x:g}Te" for x in xs]
        elif system == "Mg-Si-Sn":
            xs = fractions(40, 0.05, 0.95, 108)
            candidates = [f"Mg2Si{x:g}Sn{1 - x:g}" for x in xs]
        else:
            candidates = list(spec["candidates"])
        systems.append({**spec, "candidates": tuple(candidates)})
    return tuple(systems)


def select_pool(frame: pd.DataFrame, per_system: int = 3) -> pd.DataFrame:
    picks = []
    for system, group in frame.groupby("system"):
        group = group.drop_duplicates("formula").copy()
        chosen: list[str] = []
        # 1) highest predicted peak
        top = group.sort_values("predicted_peak_zt", ascending=False).iloc[0]
        chosen.append(top.formula)
        # 2) best risk-adjusted peak among the lower-risk half
        low_risk = group[group.risk.eq("low")]
        if low_risk.empty:
            low_risk = group.nsmallest(max(3, len(group) // 3), "risk_score")
        second = low_risk.sort_values("s_peak", ascending=False).iloc[0]
        if second.formula not in chosen:
            chosen.append(second.formula)
        # 3) median composition of the remaining pool
        remaining = group[~group.formula.isin(chosen)]
        if not remaining.empty:
            target = group.predicted_peak_zt.median()
            third = remaining.iloc[(remaining.predicted_peak_zt - target).abs().argsort()[:1]].iloc[0]
            chosen.append(third.formula)
        for rank, formula in enumerate(chosen[:per_system], start=1):
            row = group[group.formula.eq(formula)].iloc[0]
            picks.append({**row.to_dict(), "selection_rank": rank})
    return pd.DataFrame(picks)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    current = RESULTS / "multisystem_crct_candidate_screening.csv"
    if current.exists() and not BACKUP.exists():
        shutil.copy2(current, BACKUP)
    extended_systems = build_systems()
    ms.SYSTEMS = extended_systems
    screened = ms.run_multisystem_screening()
    extended_path = RESULTS / "multisystem_crct_candidate_screening_extended.csv"
    screened.to_csv(extended_path, index=False)
    selection = select_pool(screened, per_system=3)
    selection_path = RESULTS / "multisystem_crct_candidate_selection.csv"
    selection.to_csv(selection_path, index=False)
    print("pool", len(screened), "selected", len(selection))
    print(selection[["system", "formula", "predicted_peak_zt", "peak_temperature_K",
                     "risk", "selection_rank"]].to_string(index=False))
    (RESULTS / "multisystem_crct_candidate_selection_manifest.json").write_text(
        json.dumps({"pool_size": int(len(screened)), "per_system": 3,
                    "selection": "top peak; best low-risk peak; median-peak composition",
                    "systems": extended_systems}, indent=2, ensure_ascii=False),
        encoding="utf-8")


if __name__ == "__main__":
    main()
