import torch

from src.dcsa.model import DCSA


def test_dcsa_forward_is_permutation_invariant_for_support_set():
    torch.manual_seed(7)
    model = DCSA(composition_dim=4, hidden=16)
    qx = torch.randn(3, 4)
    qt = torch.randn(3, 1)
    sx = torch.randn(3, 2, 4)
    st = torch.randn(3, 2, 1)
    sy = torch.randn(3, 2, 1)
    a = model(qx, qt, sx, st, sy)
    b = model(qx, qt, sx[:, [1, 0]], st[:, [1, 0]], sy[:, [1, 0]])
    assert a.shape == (3,)
    assert torch.allclose(a, b)


def test_chemical_only_uses_query_support_interaction():
    model = DCSA(composition_dim=4, hidden=16)
    qx = torch.randn(2, 4, requires_grad=True)
    qt = torch.randn(2, 1)
    sx = torch.randn(2, 2, 4)
    st = torch.randn(2, 2, 1)
    sy = torch.randn(2, 2, 1)
    output = model(qx, qt, sx, st, sy, ablation="chemical_only")
    output.sum().backward()
    assert qx.grad is not None
    assert torch.count_nonzero(qx.grad) > 0
