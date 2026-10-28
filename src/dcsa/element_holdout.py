"""Element and periodic-family novelty validation for 5-shot CRCT.

The split remains DOI-held-out.  Within each held-out DOI, five curves are
selected from composition features only.  Query curves are then restricted to
compositions containing at least one element (or periodic family) absent from
the support set.  CRCT and its temperature-Ridge comparator therefore use the
same support curves and the same query curves.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.dcsa.baseline import PARAMS, _features, _load_points, _split_dois
from src.dcsa.crct import FeatureScaler, evaluate_episode
from src.dcsa.data_source import PROJECT_ROOT, input_paths, sha256_file
from src.dcsa.masking_validation import SHOT, select_support
from src.features.composition_features import parse_composition


_LANTHANIDES = {"La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}
_ACTINIDES = {"Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"}


def composition_elements(formula: str) -> frozenset[str]:
    """Return parsed element symbols, or an empty set for an invalid formula."""
    return frozenset(parse_composition(formula).keys())


def element_family(element: str) -> str:
    """Map an element to a coarse periodic-table family used for holdout tests."""
    if element in _LANTHANIDES:
        return "lanthanide"
    if element in _ACTINIDES:
        return "actinide"
    # The periodic table used by element_property_features is deliberately
    # kept local so this audit does not depend on an external data source.
    groups = {
        "H": 1, "Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1,
        "Be": 2, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2,
    }
    group = groups.get(element)
    if group == 1:
        return "alkali"
    if group == 2:
        return "alkaline_earth"
    if element in {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
                   "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
                   "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"}:
        return "transition"
    if element in {"B", "Si", "Ge", "As", "Sb", "Te"}:
        return "metalloid"
    if element in {"C", "N", "O", "P", "S", "Se"}:
        return "nonmetal"
    if element in {"F", "Cl", "Br", "I"}:
        return "halogen"
    if element in {"Al", "Ga", "In", "Tl", "Sn", "Pb", "Bi"}:
        return "post_transition"
    return "unknown"


def composition_families(formula: str) -> frozenset[str]:
    return frozenset(element_family(element) for element in composition_elements(formula))


def novel_query_mask(query_meta: pd.DataFrame, support_meta: pd.DataFrame,
                     mode: str) -> np.ndarray:
    """Return query rows with an element/family absent from all support curves."""
    if mode not in {"element", "family"}:
        raise ValueError("mode must be 'element' or 'family'")
    support_sets = [
        composition_elements(value) if mode == "element" else composition_families(value)
        for value in support_meta.composition_normalized
    ]
    support_union = set().union(*support_sets) if support_sets else set()
    values = query_meta.composition_normalized.map(
        composition_elements if mode == "element" else composition_families)
    return values.map(lambda current: bool(set(current) - support_union)).to_numpy(dtype=bool)


def run_element_holdout(
    root: Path = PROJECT_ROOT, *, seeds=(11, 22, 33), shot: int = SHOT,
    support_strategy: str = "diverse_support", modes=("element", "family"),
    tau: float = 0.5, blend: float = 1.0, neighbor_k: int | None = None,
    gate_tau: float | None = None,
) -> pd.DataFrame:
    """Run novelty-restricted DOI-held-out validation and write DCSA results."""
    if shot < 1:
        raise ValueError("shot must be positive")
    if not modes or any(mode not in {"element", "family"} for mode in modes):
        raise ValueError("modes must contain only 'element' or 'family'")

    points, valid = _load_points(root)
    feature_columns = sorted(c for c in points.columns if c.startswith("f_"))
    feature_table = points[["material", *feature_columns]].drop_duplicates("material")
    valid = valid.merge(feature_table, on="material", how="inner")
    rows: list[dict[str, object]] = []
    for seed in seeds:
        train_dois, test_dois = _split_dois(points.doi.unique(), seed)
        train = points[points.doi.isin(train_dois)].copy()
        test = points[points.doi.isin(test_dois)]
        prior = lgb.LGBMRegressor(**PARAMS, random_state=seed)
        prior.fit(_features(train, feature_columns), train.target.to_numpy(dtype=float))
        scaler = FeatureScaler.fit(train, feature_columns)
        eligible = valid[valid.doi.isin(test_dois)].groupby("doi").filter(
            lambda group: group.material.nunique() > shot)
        for doi, curves in eligible.groupby("doi", sort=True):
            support_meta, query_meta = select_support(
                curves, shot, support_strategy, seed=seed)
            support = test[test.curve_row_id.isin(support_meta.curve_row_id)].copy()
            support_element_union = set().union(*[
                composition_elements(value)
                for value in support_meta.composition_normalized
            ])
            support_family_union = set().union(*[
                composition_families(value)
                for value in support_meta.composition_normalized
            ])
            for mode in modes:
                mask = novel_query_mask(query_meta, support_meta, mode)
                novel_meta = query_meta.loc[mask].copy()
                if novel_meta.empty:
                    continue
                query = test[test.curve_row_id.isin(novel_meta.curve_row_id)].copy()
                support_base = prior.predict(_features(support, feature_columns))
                query_base = prior.predict(_features(query, feature_columns))
                metrics = evaluate_episode(
                    support, query, feature_columns, support_base=support_base,
                    query_base=query_base, scaler=scaler, tau=tau, blend=blend,
                    neighbor_k=neighbor_k, gate_tau=gate_tau)
                for model, mae in (("crct", metrics["mae"]),
                                   ("temperature_ridge", metrics["ridge_mae"])):
                    rows.append({
                        "seed": seed, "doi": doi, "shot": shot,
                        "holdout_mode": mode, "model": model,
                        "support_selection": support_strategy,
                        "neighbor_k": neighbor_k,
                        "n_support_curves": int(len(support_meta)),
                        "n_query_curves": int(len(novel_meta)),
                        "n_query_points": int(len(query)),
                        "n_query_novel_units": int(sum(
                            len((composition_elements(value) if mode == "element" else
                                 composition_families(value)) -
                                (support_element_union if mode == "element" else
                                 support_family_union))
                            for value in novel_meta.composition_normalized
                        )),
                        "mae": float(mae),
                        "mean_nearest_distance": metrics["mean_nearest_distance"],
                        "fraction_outside_support_temperature": metrics[
                            "fraction_outside_support_temperature"],
                    })
        print(f"seed={seed}: train_dois={len(train_dois)}, test_dois={len(test_dois)}", flush=True)

    result = pd.DataFrame(rows)
    output = root / "results"
    result_path = output / "element_family_holdout.csv"
    result.to_csv(result_path, index=False)
    result["episode_id"] = result.seed.astype(str) + ":" + result.doi.astype(str)
    summary = (result.groupby(["shot", "holdout_mode", "model", "support_selection"], as_index=False)
               .agg(mae_mean=("mae", "mean"), mae_median=("mae", "median"),
                    n_episodes=("episode_id", "nunique"), n_query_curves=("n_query_curves", "sum"),
                    n_query_points=("n_query_points", "sum")))
    summary.to_csv(output / "element_family_holdout_summary.csv", index=False)
    manifest = {
        "seeds": list(seeds), "shot": shot, "modes": list(modes),
        "support_selection": support_strategy, "split": "DOI-held-out",
        "query_definition": "query contains an element/family absent from support union",
        "target": "ZT", "input_sha256": {
            name: sha256_file(path) for name, path in input_paths().items()
        },
    }
    (output / "element_family_holdout_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(run_element_holdout().to_string(index=False))
