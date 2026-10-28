"""Uncertainty-aware selective prediction audit for CRCT.

This layer uses only quantities available before query measurements: nearest
chemical distance and the fraction of the query temperature grid outside the
support range. It is a risk score, not a calibrated physical confidence
interval.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.dcsa.data_source import PROJECT_ROOT


def run(root: Path = PROJECT_ROOT, *, n_bins: int = 4) -> pd.DataFrame:
    frame = pd.read_csv(root / "results" / "crct_nested_outer_validation.csv")
    required = {"seed", "doi", "mae", "mean_nearest_distance",
                "fraction_outside_support_temperature"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    distance = frame.mean_nearest_distance.to_numpy(float)
    outside = frame.fraction_outside_support_temperature.to_numpy(float)
    def z(values):
        scale = np.nanstd(values)
        return (values - np.nanmean(values)) / (scale if scale > 0 else 1.0)
    frame["risk_score"] = z(distance) + z(outside)
    frame["risk_group"] = pd.qcut(frame.risk_score, q=n_bins,
                                   labels=[f"Q{i+1}" for i in range(n_bins)],
                                   duplicates="drop")
    summary = (frame.groupby("risk_group", observed=False, as_index=False)
               .agg(mae=("mae", "mean"), median_mae=("mae", "median"),
                    mean_risk=("risk_score", "mean"), n_episodes=("mae", "size")))
    rho, pvalue = spearmanr(frame.risk_score, frame.mae)
    rows = []
    for fraction in (0.25, 0.50, 0.75):
        threshold = frame.risk_score.quantile(fraction)
        retained = frame[frame.risk_score <= threshold]
        rows.append({"selection": f"lowest_{int(fraction*100)}pct_risk",
                     "retained_fraction": len(retained) / len(frame),
                     "retained_mae": retained.mae.mean(),
                     "all_mae": frame.mae.mean(),
                     "n_episodes": len(retained)})
    output = root / "results"
    frame.to_csv(output / "crct_uncertainty_risk_scores.csv", index=False)
    summary.to_csv(output / "crct_uncertainty_risk_summary.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "crct_selective_prediction_summary.csv", index=False)
    (output / "crct_uncertainty_manifest.json").write_text(json.dumps({
        "risk_score": "z(mean_nearest_distance) + z(fraction_outside_support_temperature)",
        "interpretation": "pre-measurement extrapolation risk proxy; not a calibrated confidence interval",
        "spearman_rho": float(rho), "spearman_pvalue": float(pvalue),
        "n_bins": n_bins,
    }, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(run().to_string(index=False))
