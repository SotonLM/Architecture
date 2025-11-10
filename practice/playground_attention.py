"""
PLAYGROUND FILE - Feel free to experiment with ChatGPT or other AI tools!

This file is for practicing Problems 2 and 4: Attention Mechanisms
You can copy this file, modify it, and test your ideas here before implementing
the final solution in architecture/attention.py

Problems:
- Problem 2: Scaled Dot-Product Attention
- Problem 4: Simple Attention Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================================
# PROBLEM 2: Scaled Dot-Product Attention
# ============================================================================
#
# Task: Create a function scaled_dot_product_attention(Q, K, V, scale_factor)
#
# Steps:
#   1. Compute attention_scores = Q @ K^T (matrix multiplication)
#   2. Scale by scale_factor
#   3. Apply softmax to get attention_weights
#   4. Multiply attention_weights with V to get output
#
# Args:
#   Q: Query tensor of shape (batch_size, seq_len, embed_dim)
#   K: Key tensor of shape (batch_size, seq_len, embed_dim)
#   V: Value tensor of shape (batch_size, seq_len, embed_dim)
#   scale_factor: Scaling factor (usually 1/sqrt(embed_dim))
#
# Returns:
#   output: Attention output of shape (batch_size, seq_len, embed_dim)
#   attention_weights: Attention weights of shape (batch_size, seq_len, seq_len)
#
# ============================================================================

def scaled_dot_product_attention(Q, K, V, scale_factor):
    """
    Compute scaled dot-product attention.
    
    TODO: Implement this function!
    
    Hint: Use torch.matmul() for matrix multiplication and F.softmax() for softmax
    """
    # Your implementation here
    pass


# ============================================================================
# PROBLEM 4: Simple Attention Module
# ============================================================================
#
# Task: Create a class SimpleAttention that inherits from nn.Module
#
# Requirements:
#   1. Has linear layers for Q, K, V projections in __init__
#   2. In forward(), project input to Q, K, V
#   3. Use scaled_dot_product_attention with scale_factor = 1/sqrt(embed_dim)
#   4. Return the attention output
#
# ============================================================================

class SimpleAttention(nn.Module):
    """
    Simple self-attention module.
    
    TODO: Implement this class!
    """
    
    def __init__(self, embed_dim):
        super().__init__()
        # Your code here
        # Hint: Create linear layers using nn.Linear(embed_dim, embed_dim)
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
        # Hint: Use self-attention (Q=K=V=x after projection)
        pass


# ============================================================================
# TESTING AREA - Try out your implementations here!
# ============================================================================

def test_scaled_dot_product_attention():
    """Test Problem 2: Scaled dot-product attention."""
    print("Testing scaled_dot_product_attention...")
    print("-" * 50)
    
    batch_size = 2
    seq_len = 5
    embed_dim = 8
    
    # Create test tensors
    Q = torch.randn(batch_size, seq_len, embed_dim)
    K = torch.randn(batch_size, seq_len, embed_dim)
    V = torch.randn(batch_size, seq_len, embed_dim)
    scale_factor = 1.0 / math.sqrt(embed_dim)
    
    print(f"Input shapes: Q={Q.shape}, K={K.shape}, V={V.shape}")
    print(f"Scale factor: {scale_factor:.4f}")
    
    try:
        output, attn_weights = scaled_dot_product_attention(Q, K, V, scale_factor)
        
        # Check shapes
        print(f"\nOutput shape: {output.shape}")
        print(f"Attention weights shape: {attn_weights.shape}")
        
        # Check output shape
        expected_output_shape = (batch_size, seq_len, embed_dim)
        if output.shape == expected_output_shape:
            print(f"✓ Output shape correct: {output.shape}")
        else:
            print(f"✗ Output shape incorrect. Expected {expected_output_shape}, got {output.shape}")
        
        # Check attention weights shape
        expected_attn_shape = (batch_size, seq_len, seq_len)
        if attn_weights.shape == expected_attn_shape:
            print(f"✓ Attention weights shape correct: {attn_weights.shape}")
        else:
            print(f"✗ Attention weights shape incorrect. Expected {expected_attn_shape}, got {attn_weights.shape}")
        
        # Check that attention weights sum to 1.0
        attn_sum = attn_weights.sum(dim=-1)
        if torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5):
            print("✓ Attention weights sum to 1.0")
        else:
            print("✗ Attention weights should sum to 1.0 for each position")
            print(f"  Sums: {attn_sum[0, :5]}")  # Show first 5
        
        # Check that output is different from input (should be transformed)
        if not torch.allclose(output, V, atol=1e-5):
            print("✓ Output is transformed (different from V)")
        else:
            print("✗ Output should be different from V (attention should transform)")
        
        print("\n✓ Basic tests passed! (More tests in validate.py)")
        
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def test_simple_attention():
    """Test Problem 4: Simple attention module."""
    print("\n" + "=" * 50)
    print("Testing SimpleAttention module...")
    print("-" * 50)
    
    embed_dim = 32
    batch_size = 2
    seq_len = 8
    
    print(f"Embedding dimension: {embed_dim}")
    print(f"Input shape: (batch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim})")
    
    try:
        # Create module
        attention = SimpleAttention(embed_dim=embed_dim)
        print(f"✓ Module created successfully")
        
        # Check it's a PyTorch module
        if isinstance(attention, nn.Module):
            print("✓ Inherits from nn.Module")
        else:
            print("✗ Should inherit from nn.Module")
        
        # Test forward pass
        x = torch.randn(batch_size, seq_len, embed_dim)
        print(f"Input tensor shape: {x.shape}")
        
        output = attention(x)
        print(f"Output tensor shape: {output.shape}")
        
        # Check output shape
        if output.shape == (batch_size, seq_len, embed_dim):
            print("✓ Output shape correct")
        else:
            print(f"✗ Output shape incorrect. Expected {(batch_size, seq_len, embed_dim)}, got {output.shape}")
        
        # Check that output is different from input
        if not torch.allclose(output, x, atol=1e-5):
            print("✓ Output is transformed (different from input)")
        else:
            print("✗ Output should be different from input after attention")
        
        # Check that module has required attributes
        required_attrs = ['q_proj', 'k_proj', 'v_proj']
        for attr in required_attrs:
            if hasattr(attention, attr):
                print(f"✓ Has {attr} attribute")
            else:
                print(f"✗ Missing {attr} attribute")
        
        print("\n✓ Basic tests passed! (More tests in validate.py)")
        
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 50)
    print("ATTENTION MECHANISMS PLAYGROUND")
    print("=" * 50)
    print("\nThis file lets you experiment with attention implementations.")
    print("Try implementing the functions above and test them here!")
    print("\n" + "=" * 50)
    
    # Test Problem 2
    test_scaled_dot_product_attention()
    
    # Test Problem 4
    test_simple_attention()
    
    print("\n" + "=" * 50)
    print("When you're happy with your solutions, copy them to architecture/attention.py")
    print("Then run: python validate.py")

