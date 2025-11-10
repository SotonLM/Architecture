"""Tests for multi-head attention implementation."""

import torch
import pytest
from architecture.attention import MultiHeadAttention


def test_multi_head_attention_shapes():
    """Test that multi-head attention outputs correct shapes."""
    batch_size = 2
    seq_len = 10
    embed_dim = 64
    num_heads = 8
    
    attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    output, attn_weights = attention(x, x, x)
    
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)


def test_attention_weights_sum_to_one():
    """Test that attention weights sum to 1.0."""
    embed_dim = 64
    num_heads = 8
    seq_len = 10
    
    attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    x = torch.randn(1, seq_len, embed_dim)
    
    _, attn_weights = attention(x, x, x)
    
    # Check that weights sum to 1.0 along the last dimension
    attn_sum = attn_weights.sum(dim=-1)
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5)


def test_attention_with_mask():
    """Test that attention mask works correctly."""
    embed_dim = 64
    num_heads = 8
    seq_len = 5
    
    attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    x = torch.randn(1, seq_len, embed_dim)
    
    # Create a mask that masks out the last token
    mask = torch.ones(1, 1, seq_len, seq_len)
    mask[:, :, :, -1] = 0  # Mask last position
    
    output, attn_weights = attention(x, x, x, mask=mask)
    
    # Check that masked positions have very low attention weights
    assert attn_weights[:, :, :, -1].max() < 1e-5

