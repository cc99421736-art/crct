import pandas as pd

from src.dcsa.multisystem_screening import make_candidate_frame, risk_label


def test_candidate_frame_builds_unmeasured_grid():
    frame = make_candidate_frame("Ag2S0.05Te0.95", [400.0, 500.0],
                                 ["f_wmean_atomic_radius"], curve_id="candidate")
    assert list(frame.formula.unique()) == ["Ag2S0.05Te0.95"]
    assert list(frame.temperature) == [400.0, 500.0]
    assert frame.target.isna().all()


def test_risk_label_is_ordered():
    assert risk_label(0.1, 0.2, 0.4) == "low"
    assert risk_label(0.3, 0.2, 0.4) == "medium"
    assert risk_label(0.9, 0.2, 0.4) == "high"
