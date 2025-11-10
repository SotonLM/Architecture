# Transformer Architecture Exercises

This repository contains hands-on exercises to implement key components of the transformer architecture.

## Exercises

See [QUESTIONS.md](QUESTIONS.md) for exercise questions.

**Time allocated: 30 minutes**

The exercises cover (from easy to medium):
1. Basic Tensor Operations
2. Dot Product Attention  
3. Positional Encoding
4. Simple Attention Module

### 🎮 Playground Files

Experiment with ChatGPT or test your ideas using the playground files:
- `playground_basics.py` - For Problem 1
- `playground_attention.py` - For Problems 2 and 4
- `playground_positional_encoding.py` - For Problem 3

These files include test code you can run immediately to verify your implementations!

Try to solve the problems yourself first! Solutions will be available after the exercise session.

## Setup

### Option 1: Using pip (Recommended)
```bash
pip install -r requirements.txt
```

### Option 2: Minimal setup (just PyTorch)
```bash
pip install torch
```

### Option 3: Using uv (if installed)
```bash
uv sync
```

**Note**: If `uv sync` doesn't work, see [SETUP_ALTERNATIVE.md](SETUP_ALTERNATIVE.md) for detailed setup instructions.

## Testing

**For experimentation**: Use the playground files (`playground_*.py`) to test your ideas. They include test code you can run immediately.

**For final validation**: Use `python validate.py` to test your final solutions. The test code is hidden to encourage learning through implementation rather than reverse-engineering from tests.

```bash
# Using uv (if installed)
uv run python validate.py

# Or using pip/python directly
python validate.py
```

**Note**: The `hidden_tests.py` file contains test implementations. For best learning, try not to look at it until after you've attempted the problems!

## Workflow

1. Read `QUESTIONS.md` to understand the problems
2. Use playground files to experiment and test your ideas
3. Implement solutions in `architecture/*.py` files
4. Test your implementations with `python validate.py`
5. Submit your solutions when complete

Good luck! 🚀

