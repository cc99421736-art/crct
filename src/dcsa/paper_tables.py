"""Generate manuscript-ready CSV tables from frozen DCSA results."""
from pathlib import Path
import pandas as pd
from src.dcsa.data_source import PROJECT_ROOT

def run(root: Path = PROJECT_ROOT):
    out = root / "results" / "paper_tables"
    out.mkdir(exist_ok=True)
    main = pd.DataFrame([
        {"protocol": "DOI outer holdout", "method": "CRCT", "mae": .062248, "n_dois": 382, "n_episodes": 445, "ci95_low": .020070, "ci95_high": .030407, "win_rate": .766292},
        {"protocol": "DOI outer holdout", "method": "temperature-Ridge", "mae": .088226, "n_dois": 382, "n_episodes": 445, "ci95_low": None, "ci95_high": None, "win_rate": None},
    ])
    main.to_csv(out / "table_main_result.csv", index=False)
    pd.read_csv(root / "results" / "nested_strong_baselines_paired_stats.csv").to_csv(out / "table_strong_baselines.csv", index=False)
    pd.read_csv(root / "results" / "matched_baselines_ablation_summary.csv").to_csv(out / "table_ablation.csv", index=False)
    pd.read_csv(root / "results" / "crct_nested_paired_improvement_summary.csv").to_csv(out / "table_distance_extrapolation.csv", index=False)
    pd.read_csv(root / "results" / "temporal_validation_stats.csv").query("population == 'overall'").to_csv(out / "table_time_forward.csv", index=False)
    pd.read_csv(root / "results" / "element_family_holdout_paired_summary.csv").to_csv(out / "table_element_stress_test.csv", index=False)
    pd.read_csv(root / "results" / "crct_selective_prediction_summary.csv").to_csv(out / "table_selective_prediction.csv", index=False)
    return sorted(p.name for p in out.glob("*.csv"))

if __name__ == "__main__": print("\n".join(run()))
