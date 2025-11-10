"""
PLAYGROUND FILE - Feel free to experiment with ChatGPT or other AI tools!

This file is for practicing Problem 1: Basic Tensor Operations
You can copy this file, modify it, and test your ideas here before implementing
the final solution in architecture/basics.py

Problem: Create a function that scales a tensor by a factor.
"""

import torch

# ============================================================================
# PROBLEM 1: Basic Tensor Operations
# ============================================================================
# 
# Task: Create a function scale_tensor(x, scale_factor) that:
#   - Takes a PyTorch tensor x and a scale factor (float)
#   - Returns the tensor multiplied by the scale factor
#   - Should work with any tensor shape
#
# Example:
#   x = torch.tensor([1.0, 2.0, 3.0])
#   result = scale_tensor(x, 0.5)
#   # result should be tensor([0.5, 1.0, 1.5])
#
# ============================================================================

def scale_tensor(x, scale_factor):
    """
    Scale a tensor by a factor.
    
    TODO: Implement this function!
    
    Args:
        x: PyTorch tensor of any shape
        scale_factor: Float value to multiply the tensor by
        
    Returns:
        Scaled tensor of the same shape as x
    """
    # Your implementation here
    # Hint: PyTorch tensors support element-wise operations with scalars
    pass


# ============================================================================
# TESTING AREA - Try out your implementation here!
# ============================================================================

if __name__ == "__main__":
    print("Testing scale_tensor function...")
    print("-" * 50)
    
    # Test 1: Simple 1D tensor
    print("\nTest 1: 1D tensor")
    x1 = torch.tensor([1.0, 2.0, 3.0])
    print(f"Input: {x1}")
    print(f"Scale factor: 0.5")
    try:
        result1 = scale_tensor(x1, 0.5)
        print(f"Result: {result1}")
        print(f"Expected: tensor([0.5, 1.0, 1.5])")
        if torch.allclose(result1, torch.tensor([0.5, 1.0, 1.5])):
            print("✓ Test 1 PASSED")
        else:
            print("✗ Test 1 FAILED")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 2: 2D tensor
    print("\nTest 2: 2D tensor")
    x2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print(f"Input shape: {x2.shape}")
    print(f"Scale factor: 2.0")
    try:
        result2 = scale_tensor(x2, 2.0)
        print(f"Result shape: {result2.shape}")
        print(f"Result:\n{result2}")
        expected2 = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
        if torch.allclose(result2, expected2):
            print("✓ Test 2 PASSED")
        else:
            print("✗ Test 2 FAILED")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 3: Negative scale
    print("\nTest 3: Negative scale factor")
    x3 = torch.tensor([1.0, -2.0, 3.0])
    print(f"Input: {x3}")
    print(f"Scale factor: -1.0")
    try:
        result3 = scale_tensor(x3, -1.0)
        print(f"Result: {result3}")
        expected3 = torch.tensor([-1.0, 2.0, -3.0])
        if torch.allclose(result3, expected3):
            print("✓ Test 3 PASSED")
        else:
            print("✗ Test 3 FAILED")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "=" * 50)
    print("When you're happy with your solution, copy it to architecture/basics.py")
    print("Then run: python validate.py")

