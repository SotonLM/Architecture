# Transformer Architecture Exercises - Questions

**Time Allocated: 30 minutes**

These exercises are designed to help you understand and implement key components of the transformer architecture. Start with the easy problems and work your way up!

---

## Problem 1: Basic Tensor Operations (5 minutes) ⭐ Easy

Create a function that performs a simple tensor operation commonly used in transformers.

**Task**: In `architecture/basics.py`, create a function `scale_tensor(x, scale_factor)` that:
- Takes a PyTorch tensor `x` and a scale factor (float)
- Returns the tensor multiplied by the scale factor
- The function should work with any tensor shape

**Example**:
```python
x = torch.tensor([1.0, 2.0, 3.0])
result = scale_tensor(x, 0.5)
# result should be tensor([0.5, 1.0, 1.5])
```

**💡 Experiment First**: Use `playground_basics.py` to experiment with ChatGPT or test your ideas before implementing the final solution!

**Test your solution**: Run `python validate.py` or `uv run python validate.py`

---

## Problem 2: Dot Product Attention (8 minutes) ⭐⭐ Medium-Easy

Implement the core of attention mechanism: scaled dot-product.

**Task**: In `architecture/attention.py`, create a function `scaled_dot_product_attention(Q, K, V, scale_factor)` that:
- Takes query `Q`, key `K`, and value `V` tensors (all shape: `batch_size, seq_len, embed_dim`)
- Takes a `scale_factor` (usually `1/sqrt(embed_dim)`)
- Computes: `attention_scores = (Q @ K.transpose(-2, -1)) * scale_factor`
- Applies softmax to get attention weights
- Returns: `(attention_weights @ V, attention_weights)`

**Hint**: Use `torch.matmul()` for matrix multiplication and `torch.softmax()` for softmax.

**💡 Experiment First**: Use `playground_attention.py` to experiment with ChatGPT or test your ideas!

**Test your solution**: Run `python validate.py`

---

## Problem 3: Positional Encoding Function (10 minutes) ⭐⭐ Medium

Create a function to generate positional encodings using sin and cos.

**Task**: In `architecture/positional_encoding.py`, create a function `get_positional_encoding(seq_len, embed_dim)` that:
- Takes sequence length and embedding dimension
- Returns a tensor of shape `(seq_len, embed_dim)`
- For position `pos` and dimension `i`:
  - If `i` is even (0, 2, 4, ...): use `sin(pos / (10000 ** (i / embed_dim)))`
  - If `i` is odd (1, 3, 5, ...): use `cos(pos / (10000 ** ((i-1) / embed_dim)))`

**Hint**: 
- Use `torch.arange()` to create position indices
- Use `torch.sin()` and `torch.cos()` for the functions
- You can use slicing like `tensor[:, 0::2]` for even indices and `tensor[:, 1::2]` for odd indices

**💡 Experiment First**: Use `playground_positional_encoding.py` to experiment with ChatGPT or test your ideas!

**Test your solution**: Run `python validate.py`

---

## Problem 4: Simple Attention Module (7 minutes) ⭐⭐⭐ Medium

Create a basic attention module class using PyTorch's `nn.Module`.

**Task**: In `architecture/attention.py`, create a class `SimpleAttention` that:
- Inherits from `nn.Module`
- Takes `embed_dim` in `__init__`
- Has linear layers for Q, K, V projections (use `nn.Linear`)
- In `forward()`, takes input `x` (shape: `batch_size, seq_len, embed_dim`)
- Performs self-attention (Q=K=V=x after projection)
- Uses scale factor `1/sqrt(embed_dim)`
- Returns the attention output

**Hint**: You can reuse your `scaled_dot_product_attention` function from Problem 2, or implement it inline.

**💡 Experiment First**: Use `playground_attention.py` to experiment with ChatGPT or test your ideas!

**Test your solution**: Run `python validate.py`

---

## Playground Files for Experimentation 🎮

Before implementing your final solutions, you can experiment with ChatGPT or other AI tools using these playground files:

- **`playground_basics.py`** - For Problem 1 (includes test code you can run)
- **`playground_attention.py`** - For Problems 2 and 4 (includes test code)
- **`playground_positional_encoding.py`** - For Problem 3 (includes test code and visualization)

These files have:
- Problem descriptions
- Starter code with TODOs
- Test functions you can run immediately
- Helpful error messages

**Workflow**:
1. Open a playground file (e.g., `playground_basics.py`)
2. Experiment with ChatGPT or try implementing yourself
3. Run the playground file to test: `python playground_basics.py`
4. When you're happy, copy your solution to the corresponding file in `architecture/`
5. Run the official validator: `python validate.py`

## Testing Your Work

To validate your solutions without seeing the test code, run:

```bash
uv run python validate.py
```

This will check each problem and tell you if it passes or fails, without revealing the test logic.

**Note**: The playground files have basic tests, but `validate.py` has more comprehensive hidden tests that will be used for grading.

## Submission Guidelines

- Complete Problems 1-3 at minimum (Problems 1-4 if you have time)
- All code should have basic comments
- Make sure `validate.py` shows all tests passing
- Focus on correctness over optimization for these exercises

## Tips

- Start with Problem 1 to get familiar with the setup
- Problem 2 is the foundation for more complex attention mechanisms
- Problem 3 is essential for transformers to understand word order
- Problem 4 combines everything into a reusable module

Good luck! 🚀

