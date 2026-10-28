import pandas as pd
import pytest

from src.dcsa.inner_selection import choose_best


def test_choose_best_prefers_lowest_mae_then_full_blend():
    frame = pd.DataFrame([
        {"tau": .5, "blend": .5, "mae": .1},
        {"tau": .25, "blend": 1., "mae": .1},
        {"tau": 1., "blend": 1., "mae": .2},
    ])
    assert choose_best(frame) == {"tau": .25, "blend": 1.0}


def test_choose_best_rejects_empty_frame():
    with pytest.raises(ValueError, match="must not be empty"):
        choose_best(pd.DataFrame(columns=["tau", "blend", "mae"]))
