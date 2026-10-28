import pandas as pd

from src.dcsa.baseline import make_episode, paper_mean_offset


def test_episode_has_disjoint_support_and_query_materials():
    curves = pd.DataFrame({"doi": ["d"] * 3, "material": ["a", "b", "c"], "curve_row_id": [1, 2, 3]})
    episode = make_episode(curves, 2)
    assert set(episode.support.material) == {"a", "b"}
    assert set(episode.query.material) == {"c"}


def test_paper_offset_averages_curves_before_points():
    residuals = pd.DataFrame({"curve_row_id": [1, 1, 2], "residual": [1.0, 1.0, 3.0]})
    assert paper_mean_offset(residuals) == 2.0
