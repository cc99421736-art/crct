"""CRCT candidate screening across four thermoelectric application partitions."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.dcsa.baseline import PARAMS, _features, _load_points
from src.dcsa.crct import FeatureScaler, predict_episode
from src.dcsa.data_source import PROJECT_ROOT, input_paths, sha256_file
from src.features.element_property_features import element_features
from src.dcsa.support_audit import canonical_composition


SYSTEMS = (
    {"partition": "flexible", "system": "Ag-S-Te", "doi": "10.1126/sciadv.aaz8423",
     "candidates": ("Ag2S0.05Te0.95", "Ag2S0.15Te0.85"), "temperature": (300, 600)},
    {"partition": "flexible", "system": "Ag-S-Se", "doi": "10.34133/2020/6591981",
     "candidates": ("Ag2S0.05Se0.95", "Ag2S0.25Se0.75"), "temperature": (300, 450)},
    {"partition": "flexible|medium", "system": "Cu-S-Se", "doi": "10.1038/srep40436",
     "candidates": ("Cu1.98S0.12Se0.88", "Cu1.98S0.5Se0.5"), "temperature": (350, 700)},
    {"partition": "low", "system": "Bi-Sb-Te", "doi": "10.1021/acsami.0c07566",
     "candidates": ("Bi0.45Sb1.55Te3", "Bi0.9Sb1.1Te3"), "temperature": (200, 450)},
    {"partition": "low", "system": "Bi-Se-Te", "doi": "10.1016/j.jallcom.2016.06.144",
     "candidates": ("Bi2Se0.25Te2.75", "Bi2Se0.45Te2.55"), "temperature": (200, 450)},
    {"partition": "medium", "system": "Pb-Se-Te", "doi": "10.1007/s11664-013-2645-5",
     "candidates": ("PbSe0.65Te0.35", "PbSe0.35Te0.65"), "temperature": (400, 700)},
    {"partition": "high", "system": "Sb-Sn-Te", "doi": "10.1016/j.cej.2020.126925",
     "candidates": ("Sb0.05Sn0.95Te", "Sb0.10Sn0.90Te"), "temperature": (500, 850)},
    {"partition": "high", "system": "Mg-Si-Sn", "doi": "10.1557/jmr.2011.385",
     "candidates": ("Mg2Si0.65Sn0.35", "Mg2Si0.45Sn0.55"), "temperature": (500, 850)},
)


def make_candidate_frame(formula: str, temperatures: list[float] | np.ndarray,
                         feature_columns: list[str], *, curve_id: str) -> pd.DataFrame:
    features = element_features(formula)
    row = {f"f_{key}": float(value) for key, value in features.items()
           if isinstance(value, (int, float)) and np.isfinite(value)}
    frame = pd.DataFrame({"temperature": np.asarray(temperatures, dtype=float)})
    for column in feature_columns:
        frame[column] = row.get(column, 0.0)
    frame["formula"] = formula
    frame["composition_normalized"] = formula
    frame["curve_row_id"] = curve_id
    frame["target"] = np.nan
    return frame


def _zscore(values: pd.Series) -> pd.Series:
    scale = float(values.std(ddof=0))
    return (values - float(values.mean())) / (scale if scale > 0 else 1.0)


def risk_label(risk_score: float, low: float, high: float) -> str:
    """Assign candidate tiers using the validated selective-risk score."""
    if risk_score <= low:
        return "low"
    if risk_score <= high:
        return "medium"
    return "high"


def _temperature_coverage(support: pd.DataFrame, temperatures: np.ndarray) -> float:
    ranges = support.groupby("curve_row_id").temperature.agg(["min", "max"]).to_numpy(float)
    covered = np.zeros(len(temperatures), dtype=bool)
    for lower, upper in ranges:
        covered |= (temperatures >= lower) & (temperatures <= upper)
    return float(covered.mean())


def run_multisystem_screening(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    points, _ = _load_points(root)
    feature_columns = sorted(column for column in points.columns if column.startswith("f_"))
    results = []
    for index, spec in enumerate(SYSTEMS):
        support = points[points.doi.eq(spec["doi"])].copy()
        if support.empty:
            results.append({"partition": spec["partition"], "system": spec["system"],
                            "status": "missing_support_doi", "support_doi": spec["doi"]})
            continue
        train = points[~points.doi.eq(spec["doi"])].copy()
        model = lgb.LGBMRegressor(**PARAMS, random_state=20261017 + index)
        model.fit(_features(train, feature_columns), train.target.to_numpy(float))
        scaler = FeatureScaler.fit(train, feature_columns)
        support_base = model.predict(_features(support, feature_columns))
        tmin, tmax = spec["temperature"]
        temperatures = np.arange(float(tmin), float(tmax) + 0.1, 10.0)
        coverage = _temperature_coverage(support, temperatures)
        for formula in spec["candidates"]:
            query = make_candidate_frame(formula, temperatures, feature_columns,
                                         curve_id=f"candidate:{spec['system']}:{formula}")
            query_base = model.predict(_features(query, feature_columns))
            prediction = predict_episode(
                support, query, feature_columns, support_base=support_base,
                query_base=query_base, scaler=scaler, tau=0.25, blend=1.0,
                neighbor_k=2, gate_tau=1.0)
            peak_index = int(np.argmax(prediction.prediction))
            gate = float(np.mean(prediction.gate))
            distance = float(np.mean(prediction.nearest_distance))
            results.append({
                "partition": spec["partition"], "system": spec["system"],
                "formula": formula, "support_doi": spec["doi"],
                "support_curves": int(support.curve_row_id.nunique()),
                "temperature_min_K": float(tmin), "temperature_max_K": float(tmax),
                "predicted_peak_zt": float(prediction.prediction[peak_index]),
                "peak_temperature_K": float(temperatures[peak_index]),
                "mean_nearest_distance": distance, "residual_gate": gate,
                "temperature_coverage": coverage,
                "fraction_outside_support_temperature": 1.0 - coverage,
                "in_dataset_exact_composition": any(
                    canonical_composition(value) == canonical_composition(formula)
                    for value in points.composition_normalized.drop_duplicates()),
                "status": "crct_screened",
            })
    result = pd.DataFrame(results)
    screened = result[result.status.eq("crct_screened")].copy()
    # Use exactly the risk definition audited by selective prediction:
    # standardized chemical distance plus standardized temperature extrapolation.
    screened["risk_score"] = (
        _zscore(screened["mean_nearest_distance"])
        + _zscore(screened["fraction_outside_support_temperature"]))
    screened["risk_index"] = screened["risk_score"] - float(screened["risk_score"].min())
    screened["s_peak"] = screened["predicted_peak_zt"]
    screened["s_risk"] = screened["predicted_peak_zt"] / (1.0 + screened["risk_index"])
    low_cut, high_cut = screened.risk_score.quantile([1 / 3, 2 / 3]).to_numpy()
    screened["risk"] = screened.risk_score.map(
        lambda value: risk_label(float(value), float(low_cut), float(high_cut)))
    screened = screened.sort_values(
        ["partition", "system", "risk_score", "predicted_peak_zt"],
        ascending=[True, True, True, False], kind="mergesort")
    screened["priority_rank_within_system"] = (
        screened.groupby("system").predicted_peak_zt.rank(method="first", ascending=False).astype(int))
    screened["priority_rank_by_peak"] = screened["priority_rank_within_system"]
    screened["priority_rank_by_risk"] = (
        screened.groupby("system").s_risk.rank(method="first", ascending=False).astype(int))
    path = root / "results" / "multisystem_crct_candidate_screening.csv"
    screened.to_csv(path, index=False)
    patent_template = pd.DataFrame([{
        "partition": row.partition, "system": row.system, "formula": row.formula,
        "patent_status": "not_searched", "search_query": f'"{row.formula}" thermoelectric OR thermoelectric patent',
        "search_date": "2026-10-17", "notes": "Pending exact-formula and composition-variant patent search",
    } for row in screened.itertuples()])
    patent_template.to_csv(root / "results" / "multisystem_patent_audit.csv", index=False)
    (root / "results" / "multisystem_crct_candidate_screening_manifest.json").write_text(
        json.dumps({"method": "CRCT", "protocol": "representative DOI held out from prior; all its measured curves used as support",
                    "parameters": {"tau": 0.25, "neighbor_k": 2, "gate_tau": 1.0},
                    "screening_score": "predicted_peak_zt; risk reported separately",
                    "risk_score": "z(mean_nearest_distance) + z(fraction_outside_support_temperature)",
                    "risk_tier": "global candidate tertiles: low/medium/high",
                    "systems": list(SYSTEMS), "patent_search": "not yet performed",
                    "input_sha256": {name: sha256_file(path) for name, path in input_paths().items()}},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    return screened


if __name__ == "__main__":
    print(run_multisystem_screening().to_string(index=False))
