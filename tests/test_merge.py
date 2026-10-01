import torch

from src.merge.merge import merge_tensor, ties


def test_avg_of_identical_specialists_is_that_specialist():
    base = torch.zeros(4)
    s = torch.tensor([1.0, -2.0, 0.0, 3.0])
    assert torch.allclose(merge_tensor(base, [s, s], "avg", 1.0, 1.0), s)


def test_task_arithmetic_sums():
    base = torch.ones(3)
    a, b = base + torch.tensor([1.0, 0, 0]), base + torch.tensor([0, 2.0, 0])
    assert torch.allclose(merge_tensor(base, [a, b], "task_arithmetic", 1.0, 1.0),
                          torch.tensor([2.0, 3.0, 1.0]))


def test_ties_sign_election():
    tvs = torch.tensor([[1.0, -1.0], [3.0, -2.0], [-1.0, 5.0]])
    out = ties(tvs, density=1.0)
    assert torch.allclose(out, torch.tensor([2.0, 5.0]))  # col0: +1,+3 agree; col1: +5 wins (sum>0)


def test_untouched_tensor_returns_base_object():
    base = torch.randn(5)
    assert merge_tensor(base, [base.clone(), base.clone()], "ties", 1.0, 0.2) is base
