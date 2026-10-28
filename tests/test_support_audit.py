import pandas as pd

from src.dcsa.support_audit import audit_support_coverage


def curves(rows):
    return pd.DataFrame(rows, columns=[
        "SID", "sample_id", "curve_row_id", "composition_normalized",
        "property_class", "temperature_numeric", "y_numeric",
    ])


def samples(rows):
    return pd.DataFrame(rows, columns=["SID", "sample_id", "DOI"])


def test_shot_counts_require_k_support_and_one_query_composition():
    rows = []
    compositions = ["Bi2Te3", "Sb2Te3", "Bi2Se3", "PbTe", "SnSe", "GeTe"]
    for sample_id, formula in enumerate(compositions, 1):
        for temperature in (300.0, 400.0, 500.0):
            rows.append((1, sample_id, sample_id, formula, "ZT", temperature, 0.1))
    result = audit_support_coverage(
        curves(rows), samples([(1, i, "10.1000/example") for i in range(1, 7)])
    )
    assert result.counts["D_1"] == 1
    assert result.counts["D_2"] == 1
    assert result.counts["D_3"] == 1
    assert result.counts["D_5"] == 1


def test_short_unparseable_and_missing_doi_curves_are_excluded():
    rows = [
        (1, 1, 1, "Bi2Te3", "ZT", 300.0, 0.1),
        (1, 1, 1, "Bi2Te3", "ZT", 400.0, 0.1),
        (1, 2, 2, "Sb2Te3", "ZT", 300.0, 0.2),
        (1, 2, 2, "Sb2Te3", "ZT", 400.0, 0.2),
        (1, 3, 3, "not-a-formula!!", "ZT", 300.0, 0.3),
        (1, 3, 3, "not-a-formula!!", "ZT", 400.0, 0.3),
        (1, 3, 3, "not-a-formula!!", "ZT", 500.0, 0.3),
    ]
    result = audit_support_coverage(
        curves(rows), samples([(1, 1, "10.1000/short"), (1, 2, ""), (1, 3, "10.1000/short")])
    )
    assert result.counts["excluded_short_curves"] == 1
    assert result.counts["excluded_unparseable_composition_rows"] == 3
    assert result.counts["excluded_missing_doi_rows"] == 2
    assert result.doi_summary.empty
