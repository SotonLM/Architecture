# tests/unit/test_activations.py
import torch
from architecture.activations import gelu

def test_gelu_on_zero_is_zero():
    x = torch.tensor([0.0])
    y = gelu(x)
    assert torch.allclose(y, torch.tensor([0.0]))



def test_gelu_is_monotonic_on_positive_range():
    x1 = torch.tensor([0.5])
    x2 = torch.tensor([1.0])
    y1 = gelu(x1)
    y2 = gelu(x2)
    assert (y2 > y1).all()
