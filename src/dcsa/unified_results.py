"""Build one manuscript-ready table from the completed validation analyses."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.dcsa.data_source import PROJECT_ROOT


def build_unified_results(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    out = root / "results"
    rows = []
    nested = pd.read_csv(out / "crct_nested_paired_improvement_summary.csv")
    overall = nested[nested.population.eq("overall")].iloc[0]
    rows.append({"protocol": "random DOI holdout", "subgroup": "overall",
                 "n_dois": overall.n_dois, "n_episodes": overall.n_episodes,
                 "crct_mae": overall.crct_mae, "baseline_mae": overall.ridge_mae,
                 "improvement": overall.mean_improvement, "win_rate": overall.win_rate,
                 "ci95_low": overall.ci95_low, "ci95_high": overall.ci95_high,
                 "interpretation": "primary masked validation"})
    temporal = pd.read_csv(out / "temporal_validation_stats.csv").query("population == 'overall'").iloc[0]
    rows.append({"protocol": "chronological database holdout", "subgroup": "overall",
                 "n_dois": temporal.n_dois, "n_episodes": temporal.n_episodes,
                 "crct_mae": temporal.crct_mae, "baseline_mae": temporal.ridge_mae,
                 "improvement": temporal.mean_improvement, "win_rate": temporal.win_rate,
                 "ci95_low": temporal.ci95_low, "ci95_high": temporal.ci95_high,
                 "interpretation": "supportive time-forward validation"})
    for _, row in nested[nested.population.isin(["near", "mid", "far"])].iterrows():
        rows.append({"protocol": "random DOI holdout", "subgroup": row.population,
                     "n_dois": row.n_dois, "n_episodes": row.n_episodes,
                     "crct_mae": row.crct_mae, "baseline_mae": row.ridge_mae,
                     "improvement": row.mean_improvement, "win_rate": row.win_rate,
                     "ci95_low": row.ci95_low, "ci95_high": row.ci95_high,
                     "interpretation": "composition-distance subgroup"})
    family = pd.read_csv(out / "element_family_holdout_paired_summary.csv")
    for _, row in family.iterrows():
        rows.append({"protocol": "novel element holdout", "subgroup": row.holdout_mode,
                     "n_dois": row.n_dois, "n_episodes": row.n_episodes,
                     "crct_mae": row.crct_mae, "baseline_mae": row.ridge_mae,
                     "improvement": row.mean_improvement, "win_rate": row.win_rate,
                     "ci95_low": row.ci95_low, "ci95_high": row.ci95_high,
                     "interpretation": "stress test; family result is a failure boundary"})
    result = pd.DataFrame(rows)
    result.to_csv(out / "unified_results_table.csv", index=False)
    return result


if __name__ == "__main__":
    print(build_unified_results().to_string(index=False))
