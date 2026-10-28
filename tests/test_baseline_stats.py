import pandas as pd

from src.dcsa.baseline_stats import clustered_bootstrap


def test_bootstrap_clusters_seed_and_doi():
    frame = pd.DataFrame([
        {"shot": 1, "model": "a", "seed": 1, "doi": "x", "mae": 1.0},
        {"shot": 1, "model": "a", "seed": 1, "doi": "x", "mae": 3.0},
        {"shot": 1, "model": "a", "seed": 2, "doi": "x", "mae": 5.0},
        {"shot": 1, "model": "a", "seed": 2, "doi": "y", "mae": 7.0},
    ])
    out = clustered_bootstrap(frame, n_boot=100, seed=1)
    row = out.iloc[0]
    assert row.n_clusters == 3
    assert row.mae_mean == 14 / 3
