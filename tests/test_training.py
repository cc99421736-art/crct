import pandas as pd
import torch

from src.dcsa.train_dcsa import episode_tensors


def test_episode_tensors_have_fixed_support_shape():
    points = pd.DataFrame({
        "material": ["a"] * 3 + ["b"] * 3 + ["c"] * 3,
        "curve_row_id": [1] * 3 + [2] * 3 + [3] * 3,
        "temperature": [300, 400, 500] * 3,
        "target": [1., 2., 3., 2., 3., 4., 3., 4., 5.],
        "f_x": [0., 0., 0., 1., 1., 1., 2., 2., 2.],
    })
    q = points[points.material.eq("c")]
    s = points[~points.material.eq("c")]
    out = episode_tensors(q, s, ["f_x"], shot=2)
    assert out[0].shape == (3, 1)
    assert out[2].shape == (3, 2, 16, 1)
    assert out[3].shape == (3, 2, 16, 1)
    assert out[4].shape == (3, 2, 16, 1)
    assert out[6] is None
    with_base = episode_tensors(q, s, ["f_x"], shot=2,
                                query_base=[0.1, 0.2, 0.3],
                                support_base=[0.5] * len(s))
    assert with_base[6].shape == (3, 1)
    assert float(with_base[4][0, 0, 0, 0]) == 0.5
