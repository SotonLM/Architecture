# Transformer Architecture Exercises

This directory contains hands-on exercises to implement key components of the transformer architecture.

## Getting Started

1. Navigate into this directory:
   ```bash
   cd practice
   ```

2. Install dependencies (see Setup section below)

3. Read [QUESTIONS.md](QUESTIONS.md) for exercise questions

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

1. Make sure you're in the `practice/` directory
2. Read `QUESTIONS.md` to understand the problems
3. Use playground files to experiment and test your ideas
4. Implement solutions in `architecture/*.py` files
5. Test your implementations with `python validate.py`
6. Submit your solutions when complete

## Important Notes

- All commands should be run from within the `practice/` directory
- Make sure to `cd practice` first before running any scripts
- The `validate.py` script should be run from the `practice/` directory

Good luck! 🚀

