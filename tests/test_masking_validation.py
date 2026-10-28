import pandas as pd

from src.dcsa.masking_validation import select_support


def test_selection_is_curve_disjoint_and_deterministic():
    curves = pd.DataFrame({
        "material": ["a", "b", "c", "d", "e", "f"],
        "curve_row_id": [1, 2, 3, 4, 5, 6],
        "f_x": [0., 1., 2., 3., 4., 5.],
    })
    support, query = select_support(curves, 5, "random", seed=3)
    assert len(support) == 5
    assert set(support.material).isdisjoint(query.material)
    a, b = select_support(curves, 5, "central_support")
    assert len(a) == 5 and len(b) == 1


def test_diverse_support_uses_composition_geometry_and_is_target_independent():
    curves = pd.DataFrame({
        "material": ["a", "b", "c", "d", "e"],
        "curve_row_id": [1, 2, 3, 4, 5],
        "target": [100., 1., 50., -20., 3.],
        "f_x": [0., 1., 10., 11., 5.],
        "f_y": [0., 0., 0., 0., 0.],
    })
    support, query = select_support(curves, 3, "diverse_support")

    assert set(support.material) == {"a", "d", "e"}
    assert set(support.material).isdisjoint(query.material)

    changed = curves.assign(target=[-999., 999., -999., 999., -999.])
    changed_support, _ = select_support(changed, 3, "diverse_support")
    assert set(changed_support.material) == set(support.material)


def test_diverse_support_requires_composition_features():
    curves = pd.DataFrame({"material": ["a", "b"], "curve_row_id": [1, 2]})
    try:
        select_support(curves, 1, "diverse_support")
    except ValueError as exc:
        assert "composition features" in str(exc)
    else:
        raise AssertionError("diverse support should require f_ composition features")
