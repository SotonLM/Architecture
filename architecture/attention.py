"""Attention mechanisms for transformer architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(Q, K, V, scale_factor):
    """
    Compute scaled dot-product attention.
    
    TODO: Implement this function (Problem 2).
    
    Steps:
    1. Compute attention_scores = Q @ K^T (matrix multiplication)
    2. Scale by scale_factor
    3. Apply softmax to get attention_weights
    4. Multiply attention_weights with V to get output
    
    Args:
        Q: Query tensor of shape (batch_size, seq_len, embed_dim)
        K: Key tensor of shape (batch_size, seq_len, embed_dim)
        V: Value tensor of shape (batch_size, seq_len, embed_dim)
        scale_factor: Scaling factor (usually 1/sqrt(embed_dim))
        
    Returns:
        output: Attention output of shape (batch_size, seq_len, embed_dim)
        attention_weights: Attention weights of shape (batch_size, seq_len, seq_len)
    """
    # Your code here
    pass


class SimpleAttention(nn.Module):
    """
    Simple self-attention module.
    
    TODO: Implement this class (Problem 4).
    
    This should:
    1. Have linear layers for Q, K, V projections in __init__
    2. In forward(), project input to Q, K, V
    3. Use scaled_dot_product_attention with scale_factor = 1/sqrt(embed_dim)
    4. Return the attention output
    
    Args:
        embed_dim: Embedding dimension
    """
    
    def __init__(self, embed_dim):
        super().__init__()
        # Your code here
        pass
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Your code here
        pass

