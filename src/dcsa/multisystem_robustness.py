"""Parameter and support-subsampling robustness for multisystem candidates."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.dcsa.baseline import PARAMS, _features, _load_points
from src.dcsa.crct import FeatureScaler, predict_episode
from src.dcsa.data_source import PROJECT_ROOT
from src.dcsa.multisystem_screening import SYSTEMS, make_candidate_frame


def run_robustness(root: Path = PROJECT_ROOT, seeds=(7, 19, 31), taus=(0.20, 0.25, 0.35),
                   gate_taus=(0.8, 1.0, 1.2), neighbor_ks=(1, 2, 3)) -> pd.DataFrame:
    points, _ = _load_points(root)
    feature_columns = sorted(c for c in points.columns if c.startswith("f_"))
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for index, spec in enumerate(SYSTEMS):
            support_all = points[points.doi.eq(spec["doi"])].copy()
            if support_all.empty:
                continue
            # Keep the representative DOI but perturb the support composition set by
            # dropping one curve when enough curves are available.
            curves = list(support_all.curve_row_id.drop_duplicates())
            support_sets = [("all", curves)]
            if len(curves) >= 4:
                keep = sorted(rng.choice(curves, size=len(curves) - 1, replace=False).tolist())
                support_sets.append(("leave_one_curve", keep))
            train = points[~points.doi.eq(spec["doi"])].copy()
            model = lgb.LGBMRegressor(**PARAMS, random_state=seed + index)
            model.fit(_features(train, feature_columns), train.target.to_numpy(float))
            scaler = FeatureScaler.fit(train, feature_columns)
            tmin, tmax = spec["temperature"]
            temperatures = np.arange(float(tmin), float(tmax) + 0.1, 10.0)
            for support_kind, selected_curves in support_sets:
                support = support_all[support_all.curve_row_id.isin(selected_curves)].copy()
                support_base = model.predict(_features(support, feature_columns))
                for tau in taus:
                    for gate_tau in gate_taus:
                        for neighbor_k in neighbor_ks:
                            if neighbor_k > support.curve_row_id.nunique():
                                continue
                            for formula in spec["candidates"]:
                                query = make_candidate_frame(formula, temperatures, feature_columns,
                                                             curve_id=f"candidate:{spec['system']}:{formula}")
                                query_base = model.predict(_features(query, feature_columns))
                                pred = predict_episode(support, query, feature_columns,
                                    support_base=support_base, query_base=query_base, scaler=scaler,
                                    tau=tau, blend=1.0, neighbor_k=neighbor_k, gate_tau=gate_tau)
                                i = int(np.argmax(pred.prediction))
                                rows.append({"seed": seed, "support_kind": support_kind,
                                    "system": spec["system"], "formula": formula, "tau": tau,
                                    "gate_tau": gate_tau, "neighbor_k": neighbor_k,
                                    "predicted_peak_zt": float(pred.prediction[i]),
                                    "peak_temperature_K": float(temperatures[i]),
                                    "mean_nearest_distance": float(np.mean(pred.nearest_distance)),
                                    "residual_gate": float(np.mean(pred.gate))})
    detail = pd.DataFrame(rows)
    out = root / "results"
    detail.to_csv(out / "multisystem_crct_robustness_detail.csv", index=False)
    summary = (detail.groupby(["system", "formula"], as_index=False)
        .agg(n_runs=("predicted_peak_zt", "size"), peak_zt_mean=("predicted_peak_zt", "mean"),
             peak_zt_sd=("predicted_peak_zt", "std"), peak_zt_min=("predicted_peak_zt", "min"),
             peak_zt_max=("predicted_peak_zt", "max"), peak_temp_median_K=("peak_temperature_K", "median"),
             low_distance_fraction=("mean_nearest_distance", lambda x: float((x < 0.25).mean())),
             gate_mean=("residual_gate", "mean")))
    summary["ranking_stability_note"] = "higher peak_zt_mean and lower peak_zt_sd indicate more stable screening signal"
    summary.to_csv(out / "multisystem_crct_robustness_summary.csv", index=False)
    (out / "multisystem_crct_robustness_manifest.json").write_text(json.dumps({
        "purpose": "parameter and leave-one-support-curve sensitivity",
        "seeds": list(seeds), "taus": list(taus), "gate_taus": list(gate_taus),
        "neighbor_ks": list(neighbor_ks), "interpretation": "not a calibrated uncertainty interval",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(run_robustness().to_string(index=False))
