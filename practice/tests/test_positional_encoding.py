"""Tests for positional encoding implementation."""

import torch
import pytest
from architecture.positional_encoding import get_positional_encoding, PositionalEncoding


def test_positional_encoding_shape():
    """Test that positional encoding has correct shape."""
    seq_len = 20
    embed_dim = 64
    
    pos_encoding = get_positional_encoding(seq_len, embed_dim)
    
    assert pos_encoding.shape == (seq_len, embed_dim)


def test_different_positions_different():
    """Test that different positions have different encodings."""
    seq_len = 20
    embed_dim = 64
    
    pos_encoding = get_positional_encoding(seq_len, embed_dim)
    
    # First and second positions should be different
    assert not torch.allclose(pos_encoding[0], pos_encoding[1], atol=1e-6)
    
    # First and last positions should be different
    assert not torch.allclose(pos_encoding[0], pos_encoding[-1], atol=1e-6)


def test_positional_encoding_range():
    """Test that positional encoding values are in reasonable range."""
    seq_len = 100
    embed_dim = 128
    
    pos_encoding = get_positional_encoding(seq_len, embed_dim)
    
    # Values should be roughly in [-1, 1] range
    assert pos_encoding.min() >= -1.1
    assert pos_encoding.max() <= 1.1


def test_positional_encoding_module():
    """Test the PositionalEncoding module."""
    embed_dim = 64
    seq_len = 10
    batch_size = 2
    
    pos_encoder = PositionalEncoding(embed_dim=embed_dim, max_len=100)
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    output = pos_encoder(x)
    
    assert output.shape == x.shape
    # Output should be different from input (positional encoding added)
    assert not torch.allclose(output, x, atol=1e-5)

