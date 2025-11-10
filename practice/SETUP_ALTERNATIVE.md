# Alternative Setup Instructions

If `uv sync` doesn't work, use one of these alternative methods:

## Option 1: Using pip with requirements.txt (Recommended)

1. Install Python 3.11 or later
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Option 2: Using pip with pyproject.toml

```bash
pip install -e .
```

Or install with dev dependencies:
```bash
pip install -e ".[dev]"
```

## Option 3: Install uv first, then use it

1. Install uv:
   ```bash
   # On Windows (PowerShell):
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # On macOS/Linux:
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Then run:
   ```bash
   uv sync
   ```

## Option 4: Install only essential dependencies

If you only need to run the exercises, install just PyTorch:

```bash
pip install torch
```

This is enough to run the playground files and solutions.

## Testing the Setup

After installation, test with:

```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

Or run the test file:
```bash
python test_solutions.py
```

## Troubleshooting

### "uv: command not found"
- Install uv using Option 3 above
- Or use pip instead (Option 1)

### "Python version not supported"
- Make sure you have Python 3.11 or later
- Check with: `python --version`

### "torch installation fails"
- Try installing from PyTorch website: https://pytorch.org/get-started/locally/
- Or use CPU-only version: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### "Permission denied"
- Use a virtual environment (recommended)
- Or use `--user` flag: `pip install --user -r requirements.txt`

## Quick Start (Minimal Setup)

For fastest setup with just what's needed:

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install PyTorch (essential)
pip install torch

# That's it! You can now run:
python playground_basics.py
python test_solutions.py
python validate.py
```

