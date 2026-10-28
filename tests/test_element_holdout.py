import numpy as np
import pandas as pd

from src.dcsa.element_holdout import (
    composition_elements,
    composition_families,
    element_family,
    novel_query_mask,
)


def _meta():
    support = pd.DataFrame({
        "composition_normalized": ["Bi2Te3", "Sb2Te3"],
        "material": ["a", "b"],
        "curve_row_id": [1, 2],
    })
    query = pd.DataFrame({
        "composition_normalized": ["Bi2Se3", "Bi2Te3", "PbTe"],
        "material": ["c", "d", "e"],
        "curve_row_id": [3, 4, 5],
    })
    return support, query


def test_element_and_family_parsing_is_deterministic():
    assert composition_elements("Bi2Se3") == frozenset({"Bi", "Se"})
    assert element_family("Bi") == "post_transition"
    assert element_family("Se") == "nonmetal"
    assert composition_families("Bi2Se3") == frozenset({"post_transition", "nonmetal"})


def test_novel_element_mask_excludes_support_compositions():
    support, query = _meta()
    mask = novel_query_mask(query, support, "element")
    assert np.array_equal(mask, np.array([True, False, True]))


def test_novel_family_mask_uses_support_family_union():
    support, query = _meta()
    mask = novel_query_mask(query, support, "family")
    # Bi2Se3 introduces a nonmetal family absent from Bi/Sb tellurides.
    assert np.array_equal(mask, np.array([True, False, False]))
