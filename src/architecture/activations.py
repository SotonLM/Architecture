# architecture/activations.py
import torch
import torch.nn.functional as F

def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x)
