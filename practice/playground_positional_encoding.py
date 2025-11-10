"""
PLAYGROUND FILE - Feel free to experiment with ChatGPT or other AI tools!

This file is for practicing Problem 3: Positional Encoding
You can copy this file, modify it, and test your ideas here before implementing
the final solution in architecture/positional_encoding.py

Problem: Create a function to generate sinusoidal positional encodings
"""

import torch
import math

# ============================================================================
# PROBLEM 3: Positional Encoding
# ============================================================================
#
# Task: Create a function get_positional_encoding(seq_len, embed_dim)
#
# Requirements:
#   - Takes sequence length and embedding dimension
#   - Returns a tensor of shape (seq_len, embed_dim)
#   - For position pos and dimension i:
#     * If i is even (0, 2, 4, ...): use sin(pos / (10000 ** (i / embed_dim)))
#     * If i is odd (1, 3, 5, ...): use cos(pos / (10000 ** ((i-1) / embed_dim)))
#
# Hints:
#   - Use torch.arange() to create position indices
#   - Use torch.sin() and torch.cos() for the functions
#   - You can use slicing like tensor[:, 0::2] for even indices
#   - You can use slicing like tensor[:, 1::2] for odd indices
#   - Consider using torch.exp() and math.log() for the division term
#
# ============================================================================

def get_positional_encoding(seq_len, embed_dim):
    """
    Generate sinusoidal positional encodings.
    
    TODO: Implement this function!
    
    Args:
        seq_len: Length of the sequence
        embed_dim: Embedding dimension
        
    Returns:
        Positional encoding tensor of shape (seq_len, embed_dim)
    """
    # Your implementation here
    # 
    # Step-by-step hint:
    # 1. Create position indices using torch.arange(seq_len)
    # 2. Create division term: 10000^(2i/d) for even indices
    #    (Hint: use exp(2i * -log(10000) / d))
    # 3. Initialize an empty tensor of shape (seq_len, embed_dim)
    # 4. Fill even columns with sin(position * div_term)
    # 5. Fill odd columns with cos(position * div_term)
    #
    pass


# ============================================================================
# TESTING AREA - Try out your implementation here!
# ============================================================================

def test_positional_encoding():
    """Test the positional encoding function."""
    print("Testing get_positional_encoding...")
    print("-" * 50)
    
    seq_len = 10
    embed_dim = 16
    
    print(f"Sequence length: {seq_len}")
    print(f"Embedding dimension: {embed_dim}")
    
    try:
        pos_encoding = get_positional_encoding(seq_len, embed_dim)
        
        # Check shape
        print(f"\nOutput shape: {pos_encoding.shape}")
        expected_shape = (seq_len, embed_dim)
        if pos_encoding.shape == expected_shape:
            print(f"✓ Shape correct: {pos_encoding.shape}")
        else:
            print(f"✗ Shape incorrect. Expected {expected_shape}, got {pos_encoding.shape}")
            return
        
        # Check that different positions are different
        print(f"\nChecking position differences...")
        if not torch.allclose(pos_encoding[0], pos_encoding[1], atol=1e-6):
            print("✓ Different positions have different encodings")
        else:
            print("✗ Different positions should have different encodings")
            print(f"  Position 0: {pos_encoding[0, :5]}")
            print(f"  Position 1: {pos_encoding[1, :5]}")
        
        # Check that values are in reasonable range
        min_val = pos_encoding.min().item()
        max_val = pos_encoding.max().item()
        print(f"\nValue range: [{min_val:.4f}, {max_val:.4f}]")
        if min_val >= -2.0 and max_val <= 2.0:
            print("✓ Values are in reasonable range (roughly [-1, 1])")
        else:
            print("⚠ Values might be outside expected range")
        
        # Check even/odd pattern
        print(f"\nChecking sin/cos pattern...")
        even_values = pos_encoding[0, 0::2]  # Even indices
        odd_values = pos_encoding[0, 1::2]   # Odd indices
        
        print(f"  Even indices (first 3): {even_values[:3]}")
        print(f"  Odd indices (first 3): {odd_values[:3]}")
        
        # Check that we're not getting all zeros
        if not torch.allclose(even_values, torch.zeros_like(even_values), atol=1e-3):
            print("✓ Even indices have non-zero values (sin pattern)")
        else:
            print("✗ Even indices should use sin (should be non-zero)")
        
        if not torch.allclose(odd_values, torch.zeros_like(odd_values), atol=1e-3):
            print("✓ Odd indices have non-zero values (cos pattern)")
        else:
            print("✗ Odd indices should use cos (should be non-zero)")
        
        # Check that first and last positions are different
        if not torch.allclose(pos_encoding[0], pos_encoding[-1], atol=1e-3):
            print("✓ First and last positions are different")
        else:
            print("✗ First and last positions should be significantly different")
        
        # Visual check: show first few positions and dimensions
        print(f"\nFirst 3 positions, first 8 dimensions:")
        print(pos_encoding[:3, :8])
        
        print("\n✓ Basic tests passed! (More tests in validate.py)")
        
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def test_different_sizes():
    """Test with different sequence lengths and embedding dimensions."""
    print("\n" + "=" * 50)
    print("Testing with different sizes...")
    print("-" * 50)
    
    test_cases = [
        (5, 8),
        (20, 32),
        (100, 64),
    ]
    
    for seq_len, embed_dim in test_cases:
        try:
            pos_encoding = get_positional_encoding(seq_len, embed_dim)
            if pos_encoding.shape == (seq_len, embed_dim):
                print(f"✓ seq_len={seq_len}, embed_dim={embed_dim}: shape correct")
            else:
                print(f"✗ seq_len={seq_len}, embed_dim={embed_dim}: shape incorrect")
        except Exception as e:
            print(f"✗ seq_len={seq_len}, embed_dim={embed_dim}: Error - {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("POSITIONAL ENCODING PLAYGROUND")
    print("=" * 50)
    print("\nThis file lets you experiment with positional encoding.")
    print("Try implementing the function above and test it here!")
    print("\n" + "=" * 50)
    
    # Main test
    test_positional_encoding()
    
    # Test different sizes
    test_different_sizes()
    
    print("\n" + "=" * 50)
    print("When you're happy with your solution, copy it to architecture/positional_encoding.py")
    print("Then run: python validate.py")
    
    # Bonus: Visualize if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        print("\n" + "=" * 50)
        print("Bonus: Visualizing positional encoding...")
        seq_len = 50
        embed_dim = 32
        pos_encoding = get_positional_encoding(seq_len, embed_dim)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(pos_encoding.T, cmap='coolwarm', aspect='auto')
        plt.colorbar(label='Encoding Value')
        plt.xlabel('Position in Sequence')
        plt.ylabel('Embedding Dimension')
        plt.title('Positional Encoding Visualization')
        plt.tight_layout()
        
        # Save visualization
        try:
            plt.savefig('positional_encoding_visualization.png')
            print("✓ Saved visualization to positional_encoding_visualization.png")
        except:
            print("(Could not save visualization, but you can view it)")
        
        print("Close the plot window to continue...")
        plt.show()
        
    except ImportError:
        print("\n(Install matplotlib to see a visualization of the positional encoding)")

