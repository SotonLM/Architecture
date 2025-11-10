"""Positional encoding for transformer architecture."""

import torch
import math


def get_positional_encoding(seq_len, embed_dim):
    """
    Generate sinusoidal positional encodings.
    
    TODO: Implement this function (Problem 3).
    
    For position pos and dimension i:
    - If i is even: use sin(pos / (10000 ** (i / embed_dim)))
    - If i is odd: use cos(pos / (10000 ** ((i-1) / embed_dim)))
    
    Hints:
    - Use torch.arange() to create position indices
    - Use torch.sin() and torch.cos()
    - Use slicing like tensor[:, 0::2] for even indices
    - Use slicing like tensor[:, 1::2] for odd indices
    
    Args:
        seq_len: Length of the sequence
        embed_dim: Embedding dimension
        
    Returns:
        Positional encoding tensor of shape (seq_len, embed_dim)
    """
    # Your code here
    pass

