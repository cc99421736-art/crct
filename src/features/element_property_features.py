from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .composition_features import parse_composition

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "data" / "processed"

# Element property table (standard reference values)
ELEMENT_PROPS = {
    # element: (electronegativity, atomic_radius_pm, valence_electrons, ionization_eV, period, group, melting_K, density_gcm3)
    "H": (2.20, 25, 1, 13.60, 1, 1, 14, 0.09), "Li": (0.98, 145, 1, 5.39, 2, 1, 454, 0.53),
    "Be": (1.57, 105, 2, 9.32, 2, 2, 1560, 1.85), "B": (2.04, 85, 3, 8.30, 2, 13, 2349, 2.34),
    "C": (2.55, 70, 4, 11.26, 2, 14, 3823, 2.27), "N": (3.04, 65, 5, 14.53, 2, 15, 63, 0.00125),
    "O": (3.44, 60, 6, 13.62, 2, 16, 54, 0.00143), "F": (3.98, 50, 7, 17.42, 2, 17, 53, 0.0017),
    "Na": (0.93, 180, 1, 5.14, 3, 1, 371, 0.97), "Mg": (1.31, 150, 2, 7.65, 3, 2, 923, 1.74),
    "Al": (1.61, 125, 3, 5.99, 3, 13, 933, 2.70), "Si": (1.90, 110, 4, 8.15, 3, 14, 1687, 2.33),
    "P": (2.19, 100, 5, 10.49, 3, 15, 317, 1.82), "S": (2.58, 100, 6, 10.36, 3, 16, 388, 2.07),
    "Cl": (3.16, 100, 7, 12.97, 3, 17, 172, 0.0032), "K": (0.82, 220, 1, 4.34, 4, 1, 337, 0.86),
    "Ca": (1.00, 180, 2, 6.11, 4, 2, 1115, 1.55), "Sc": (1.36, 160, 3, 6.56, 4, 3, 1814, 2.99),
    "Ti": (1.54, 140, 4, 6.83, 4, 4, 1941, 4.51), "V": (1.63, 135, 5, 6.75, 4, 5, 2183, 6.11),
    "Cr": (1.66, 140, 6, 6.77, 4, 6, 2180, 7.15), "Mn": (1.55, 140, 7, 7.43, 4, 7, 1519, 7.21),
    "Fe": (1.83, 140, 8, 7.90, 4, 8, 1811, 7.87), "Co": (1.88, 135, 9, 7.88, 4, 9, 1768, 8.90),
    "Ni": (1.91, 135, 10, 7.64, 4, 10, 1728, 8.91), "Cu": (1.90, 135, 11, 7.73, 4, 11, 1358, 8.96),
    "Zn": (1.65, 135, 12, 9.39, 4, 12, 693, 7.14), "Ga": (1.81, 130, 3, 5.99, 4, 13, 303, 5.91),
    "Ge": (2.01, 125, 4, 7.90, 4, 14, 1211, 5.32), "As": (2.18, 115, 5, 9.79, 4, 15, 1090, 5.73),
    "Se": (2.55, 115, 6, 9.75, 4, 16, 494, 4.81), "Br": (2.96, 115, 7, 11.81, 4, 17, 266, 3.12),
    "Rb": (0.82, 235, 1, 4.18, 5, 1, 312, 1.53), "Sr": (0.95, 200, 2, 5.69, 5, 2, 1050, 2.64),
    "Y": (1.22, 180, 3, 6.22, 5, 3, 1799, 4.47), "Zr": (1.33, 155, 4, 6.63, 5, 4, 2128, 6.51),
    "Nb": (1.60, 145, 5, 6.76, 5, 5, 2750, 8.57), "Mo": (2.16, 145, 6, 7.09, 5, 6, 2896, 10.28),
    "Ru": (2.20, 130, 8, 7.36, 5, 8, 2607, 12.45), "Rh": (2.28, 135, 9, 7.46, 5, 9, 2237, 12.41),
    "Pd": (2.20, 140, 10, 8.34, 5, 10, 1828, 12.02), "Ag": (1.93, 160, 11, 7.58, 5, 11, 1235, 10.49),
    "Cd": (1.69, 155, 12, 8.99, 5, 12, 594, 8.65), "In": (1.78, 155, 3, 5.79, 5, 13, 430, 7.31),
    "Sn": (1.96, 145, 4, 7.34, 5, 14, 505, 7.29), "Sb": (2.05, 145, 5, 8.61, 5, 15, 904, 6.70),
    "Te": (2.10, 140, 6, 9.01, 5, 16, 723, 6.24), "I": (2.66, 140, 7, 10.45, 5, 17, 387, 4.93),
    "Cs": (0.79, 260, 1, 3.89, 6, 1, 302, 1.87), "Ba": (0.89, 215, 2, 5.21, 6, 2, 1000, 3.51),
    "La": (1.10, 195, 3, 5.58, 6, 3, 1193, 6.15), "Ce": (1.12, 185, 4, 5.54, 6, 3, 1068, 6.77),
    "Pr": (1.13, 185, 5, 5.47, 6, 3, 1208, 6.77), "Nd": (1.14, 185, 6, 5.53, 6, 3, 1297, 7.01),
    "Sm": (1.17, 185, 8, 5.64, 6, 3, 1345, 7.52), "Eu": (1.20, 185, 9, 5.67, 6, 3, 1099, 5.24),
    "Gd": (1.20, 180, 10, 6.15, 6, 3, 1585, 7.90), "Tb": (1.10, 175, 11, 5.86, 6, 3, 1629, 8.23),
    "Dy": (1.22, 175, 12, 5.94, 6, 3, 1680, 8.55), "Ho": (1.23, 175, 13, 6.02, 6, 3, 1734, 8.80),
    "Er": (1.24, 175, 14, 6.11, 6, 3, 1802, 9.07), "Tm": (1.25, 175, 15, 6.18, 6, 3, 1818, 9.32),
    "Yb": (1.10, 175, 16, 6.25, 6, 3, 1097, 6.90), "Lu": (1.27, 175, 3, 5.43, 6, 3, 1925, 9.84),
    "Hf": (1.30, 155, 4, 6.83, 6, 4, 2506, 13.31), "Ta": (1.50, 145, 5, 7.55, 6, 5, 3290, 16.65),
    "W": (2.36, 135, 6, 7.86, 6, 6, 3695, 19.25), "Re": (1.90, 135, 7, 7.83, 6, 7, 3459, 21.02),
    "Os": (2.20, 130, 8, 8.44, 6, 8, 3306, 22.59), "Ir": (2.20, 135, 9, 8.97, 6, 9, 2719, 22.56),
    "Pt": (2.28, 135, 10, 8.96, 6, 10, 2041, 21.45), "Au": (2.54, 135, 11, 9.23, 6, 11, 1337, 19.30),
    "Hg": (2.00, 150, 12, 10.44, 6, 12, 234, 13.53), "Tl": (1.62, 190, 3, 6.11, 6, 13, 577, 11.85),
    "Pb": (2.33, 180, 4, 7.42, 6, 14, 601, 11.34), "Bi": (2.02, 160, 5, 7.29, 6, 15, 545, 9.78),
    "Th": (1.30, 180, 4, 6.31, 7, 3, 2023, 11.72), "U": (1.38, 175, 6, 6.19, 7, 3, 1408, 19.05),
}

IDX = {"en": 0, "r": 1, "ve": 2, "ie": 3, "period": 4, "group": 5, "melt": 6, "dens": 7}


def element_features(formula):
    amounts = parse_composition(formula)
    out = {}
    if not amounts:
        return out
    total = sum(amounts.values())
    fracs = {e: n / total for e, n in amounts.items()}
    known = {e: ELEMENT_PROPS[e] for e in amounts if e in ELEMENT_PROPS}
    if not known:
        return out
    def wmean(k):
        vals = [ELEMENT_PROPS[e][IDX[k]] for e in known]
        w = [fracs[e] for e in known]
        return sum(v * f for v, f in zip(vals, w)) / sum(w)
    def wstd(k):
        vals = [ELEMENT_PROPS[e][IDX[k]] for e in known]
        w = [fracs[e] for e in known]
        m = wmean(k)
        return (sum(f * (v - m) ** 2 for v, f in zip(vals, w)) / sum(w)) ** 0.5
    def wrange(k):
        vals = [ELEMENT_PROPS[e][IDX[k]] for e in known]
        return max(vals) - min(vals)
    for k, label in [("en", "electronegativity"), ("r", "atomic_radius"),
                     ("ve", "valence_electrons"), ("ie", "ionization_energy"),
                     ("period", "period"), ("group", "group"),
                     ("melt", "melting_point"), ("dens", "element_density")]:
        out[f"wmean_{label}"] = wmean(k)
    out["range_electronegativity"] = wrange("en")
    out["std_electronegativity"] = wstd("en")
    out["total_valence_electrons"] = wmean("ve") * total
    return out


def main():
    src = OUT / "starrydata2_curves_long_clean.parquet"
    d = pd.read_parquet(src, columns=["composition_normalized"])
    unique = d["composition_normalized"].dropna().unique()
    rows = []
    for formula in unique:
        f = element_features(formula)
        if f:
            f["composition_normalized"] = formula
            rows.append(f)
    props = pd.DataFrame(rows)
    feat = pd.read_parquet(OUT / "te_features.parquet")
    merged = feat.merge(props, on="composition_normalized", how="left")
    merged.to_parquet(OUT / "te_features_B.parquet", index=False)
    meta = {"rows": len(merged), "element_feature_columns": list(props.columns),
            "coverage": float(merged["wmean_electronegativity"].notna().mean()),
            "source": "element periodic table reference values (no external API)"}
    (OUT / "feature_metadata_B.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("element features:", len(props), "rows; coverage:", meta["coverage"])
    print(list(props.columns))


if __name__ == "__main__":
    main()
