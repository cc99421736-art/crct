"""Exploratory Ag2SxTe1-x candidate screening in the 400-800 K range."""
from __future__ import annotations
import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from src.dcsa.baseline import PARAMS, _features, _load_points
from src.dcsa.data_source import PROJECT_ROOT
from src.features.element_property_features import element_features


def run_candidates(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    points, _ = _load_points(root)
    feature_columns = sorted(c for c in points.columns if c.startswith("f_"))
    model = lgb.LGBMRegressor(**PARAMS, random_state=20261004)
    model.fit(_features(points, feature_columns), points.target.to_numpy(float))
    existing = set(points.composition_normalized.astype(str))
    substitutions = [0.005, 0.01, 0.02, 0.05, 0.08, 0.10]
    temperatures = np.arange(400.0, 801.0, 10.0)
    rows = []
    for x in substitutions:
        formula = f"Ag2S{x:g}Te{1-x:g}"
        parsed = element_features(formula)
        features = {f"f_{k}": float(v) for k, v in parsed.items()
                    if isinstance(v, (int, float)) and np.isfinite(v)}
        frame = pd.DataFrame([{**features, "temperature": t} for t in temperatures])
        frame = frame.reindex(columns=feature_columns + ["temperature"], fill_value=0.0)
        prediction = model.predict(frame)
        best = int(np.argmax(prediction))
        rows.append({"formula": formula, "s_fraction": x,
                     "database_existing": formula in existing,
                     "peak_zt_400_800K": float(prediction[best]),
                     "peak_temperature_K": float(temperatures[best]),
                     "zt_400K": float(prediction[0]), "zt_800K": float(prediction[-1]),
                     "prediction_type": "zero-shot composition-temperature prior"})
    result = pd.DataFrame(rows)
    out = root / "results"
    result.to_csv(out / "ag_s_te_candidate_screening.csv", index=False)
    (out / "ag_s_te_candidate_screening_manifest.json").write_text(json.dumps({
        "base_family": "Ag2SxTe1-x", "s_fraction": substitutions,
        "temperature_range_K": [400, 800], "source": "all historical DOI data",
        "caveat": "exploratory zero-shot prior; no experimental validation",
    }, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(run_candidates().to_string(index=False))
