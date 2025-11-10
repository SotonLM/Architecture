"""Hidden test implementations.

⚠️ INSTRUCTOR NOTE: This file contains the test solutions.
   For best learning outcomes, students should NOT see this file.
   Consider keeping this in a private branch or server, or instruct
   students not to read this file.

This module contains the actual test logic that students should not see.
Only the validate.py script should import from here.
"""

import torch
import torch.nn as nn
import math


def run_hidden_test(problem_num: int, module) -> tuple[bool, str]:
    """Run hidden tests for a specific problem."""
    try:
        if problem_num == 1:
            return test_problem_1(module)
        elif problem_num == 2:
            return test_problem_2(module)
        elif problem_num == 3:
            return test_problem_3(module)
        elif problem_num == 4:
            return test_problem_4(module)
        else:
            return False, "Unknown problem number"
    except Exception as e:
        return False, f"Error during testing: {str(e)}"


def test_problem_1(module) -> tuple[bool, str]:
    """Test Problem 1: Basic tensor scaling."""
    if not hasattr(module, 'scale_tensor'):
        return False, "Missing function: scale_tensor"
    
    # Test 1: Simple 1D tensor
    x = torch.tensor([1.0, 2.0, 3.0])
    result = module.scale_tensor(x, 0.5)
    expected = torch.tensor([0.5, 1.0, 1.5])
    if not torch.allclose(result, expected):
        return False, "Incorrect scaling for 1D tensor"
    
    # Test 2: 2D tensor
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = module.scale_tensor(x, 2.0)
    expected = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    if not torch.allclose(result, expected):
        return False, "Incorrect scaling for 2D tensor"
    
    # Test 3: Negative scale
    x = torch.tensor([1.0, -2.0, 3.0])
    result = module.scale_tensor(x, -1.0)
    expected = torch.tensor([-1.0, 2.0, -3.0])
    if not torch.allclose(result, expected):
        return False, "Incorrect scaling with negative factor"
    
    return True, "All test cases passed!"


def test_problem_2(module) -> tuple[bool, str]:
    """Test Problem 2: Scaled dot-product attention."""
    if not hasattr(module, 'scaled_dot_product_attention'):
        return False, "Missing function: scaled_dot_product_attention"
    
    batch_size = 2
    seq_len = 5
    embed_dim = 8
    
    # Create test tensors
    Q = torch.randn(batch_size, seq_len, embed_dim)
    K = torch.randn(batch_size, seq_len, embed_dim)
    V = torch.randn(batch_size, seq_len, embed_dim)
    scale_factor = 1.0 / math.sqrt(embed_dim)
    
    # Test function
    output, attn_weights = module.scaled_dot_product_attention(Q, K, V, scale_factor)
    
    # Check output shape
    if output.shape != (batch_size, seq_len, embed_dim):
        return False, f"Wrong output shape. Expected {(batch_size, seq_len, embed_dim)}, got {output.shape}"
    
    # Check attention weights shape
    if attn_weights.shape != (batch_size, seq_len, seq_len):
        return False, f"Wrong attention weights shape. Expected {(batch_size, seq_len, seq_len)}, got {attn_weights.shape}"
    
    # Check that attention weights sum to 1.0
    attn_sum = attn_weights.sum(dim=-1)
    if not torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5):
        return False, "Attention weights should sum to 1.0 for each position"
    
    # Check that scaling was applied (compare with unscaled version)
    manual_scores = torch.matmul(Q, K.transpose(-2, -1))
    if torch.allclose(manual_scores, attn_weights, atol=1e-5):
        return False, "Scale factor should be applied before softmax"
    
    return True, "All test cases passed!"


def test_problem_3(module) -> tuple[bool, str]:
    """Test Problem 3: Positional encoding."""
    if not hasattr(module, 'get_positional_encoding'):
        return False, "Missing function: get_positional_encoding"
    
    seq_len = 10
    embed_dim = 16
    
    # Test function
    pos_encoding = module.get_positional_encoding(seq_len, embed_dim)
    
    # Check shape
    if pos_encoding.shape != (seq_len, embed_dim):
        return False, f"Wrong shape. Expected {(seq_len, embed_dim)}, got {pos_encoding.shape}"
    
    # Check that different positions are different
    if torch.allclose(pos_encoding[0], pos_encoding[1], atol=1e-6):
        return False, "Different positions should have different encodings"
    
    # Check that values are in reasonable range
    if pos_encoding.min() < -2.0 or pos_encoding.max() > 2.0:
        return False, "Positional encoding values should be roughly in [-1, 1] range"
    
    # Check that even and odd indices have sin/cos pattern
    # First position, even indices should have sin-like values
    even_values = pos_encoding[0, 0::2]
    odd_values = pos_encoding[0, 1::2]
    
    # Values should alternate in a sinusoidal pattern
    if torch.allclose(even_values, torch.zeros_like(even_values), atol=1e-3):
        return False, "Even indices should use sin (non-zero values)"
    
    # Check second position is different from first
    if torch.allclose(pos_encoding[0], pos_encoding[-1], atol=1e-3):
        return False, "First and last positions should be significantly different"
    
    return True, "All test cases passed!"


def test_problem_4(module) -> tuple[bool, str]:
    """Test Problem 4: Simple attention module."""
    if not hasattr(module, 'SimpleAttention'):
        return False, "Missing class: SimpleAttention"
    
    embed_dim = 32
    batch_size = 2
    seq_len = 8
    
    # Create module
    try:
        attention = module.SimpleAttention(embed_dim=embed_dim)
    except Exception as e:
        return False, f"Error creating SimpleAttention: {str(e)}"
    
    # Check it's a PyTorch module
    if not isinstance(attention, nn.Module):
        return False, "SimpleAttention should inherit from nn.Module"
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    try:
        output = attention(x)
    except Exception as e:
        return False, f"Error in forward pass: {str(e)}"
    
    # Check output shape
    if output.shape != (batch_size, seq_len, embed_dim):
        return False, f"Wrong output shape. Expected {(batch_size, seq_len, embed_dim)}, got {output.shape}"
    
    # Check that output is different from input (should be transformed)
    if torch.allclose(output, x, atol=1e-5):
        return False, "Output should be different from input after attention"
    
    # Check that module has Q, K, V projections
    if not hasattr(attention, 'q_proj') or not hasattr(attention, 'k_proj') or not hasattr(attention, 'v_proj'):
        return False, "SimpleAttention should have q_proj, k_proj, and v_proj attributes"
    
    # Check projections are Linear layers
    if not isinstance(attention.q_proj, nn.Linear):
        return False, "q_proj should be a nn.Linear layer"
    
    return True, "All test cases passed!"

