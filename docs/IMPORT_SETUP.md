# Import Setup Guide

## Overview

This project uses a **flat module structure** where all source code resides in the `src/` directory. Files within `src/` use **direct imports** (without the `src.` prefix) because they're all in the same directory.

## Import Patterns

### ✅ Correct (files in src/)
```python
# src/train_phase1.py
from environment_phase1 import TradingEnvironmentPhase1
from model_utils import detect_models_in_folder
from market_specs import get_market_spec
```

### ✅ Also Correct (test files, external scripts)
```python
# tests/test_environment.py
from src.environment_phase1 import TradingEnvironmentPhase1
from src.model_utils import detect_models_in_folder
from src.market_specs import get_market_spec
```

### ❌ Incorrect
```python
# DON'T use relative imports without proper package structure
from .model_utils import X  # Only if running as: python -m src.script
```

## IDE Configuration

### VS Code / Pylance

The project includes `.vscode/settings.json` with proper configuration:
- `python.analysis.extraPaths` includes `src/`
- `python.autoComplete.extraPaths` includes `src/`
- Terminal PYTHONPATH includes `src/`

**To activate:**
1. Reload VS Code window: `Ctrl+Shift+P` → "Reload Window"
2. Or restart VS Code

### PyCharm

1. Right-click on `src/` directory
2. Select "Mark Directory as" → "Sources Root"

### Other IDEs

Add `src/` to the Python path in your IDE settings.

## Running Scripts

### From Command Line

**Option 1: With PYTHONPATH**
```bash
PYTHONPATH=src python src/train_phase1.py
```

**Option 2: Using the scripts directly** (they add src/ to path automatically)
```bash
python src/train_phase1.py
python src/train_phase2.py
python src/train_phase3_llm.py
```

**Option 3: Using main.py**
```bash
python main.py
```

### From Python

```python
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Now you can import
from model_utils import X
from market_specs import Y
```

## Why This Structure?

### Advantages
1. **Simple**: All modules in one directory
2. **Clear**: Direct imports show module relationships
3. **Works**: Runtime path manipulation is minimal
4. **Tested**: All training and evaluation scripts use this pattern

### Runtime vs IDE

- **Runtime**: Scripts add `src/` to `sys.path` automatically
- **IDE**: Needs configuration (`.vscode/settings.json`, `pyrightconfig.json`)

Both are now configured correctly.

## File Structure

```
AI Trainer/
├── .vscode/
│   └── settings.json          # VS Code Python path config
├── pyrightconfig.json         # Pylance/Pyright config
├── src/                       # All source code here
│   ├── __init__.py           # Makes src/ a package
│   ├── model_utils.py        # Utilities
│   ├── market_specs.py       # Market specifications
│   ├── environment_*.py      # Trading environments
│   ├── train_*.py           # Training scripts
│   └── ...
├── tests/                    # Test files
├── main.py                   # Entry point
└── ...
```

## Troubleshooting

### Import errors in IDE but code runs fine?

**Solution**: Reload your IDE or check that `.vscode/settings.json` and `pyrightconfig.json` are present.

### Import errors at runtime?

**Check**:
1. Are you running from project root?
2. Is PYTHONPATH set correctly?
3. For training scripts, they should handle path setup automatically

### Module not found errors?

**Verify**:
```bash
python -c "import sys; print('\\n'.join(sys.path))"
```

Should include `src/` directory or parent directory.

## Testing

```bash
# All tests
PYTHONPATH=src python -m pytest tests/

# Specific test
PYTHONPATH=src python tests/test_llm_integration.py

# With verbose output
PYTHONPATH=src python -m pytest tests/ -v
```

## Additional Notes

- The `src/__init__.py` file must exist (it does)
- Training scripts automatically configure paths
- Test files may use either import pattern
- Main.py adds src/ to path before importing

---

**Last Updated**: Based on cleanup performed after Phase 3 implementation
**Python Version**: 3.12+
**IDE Support**: VS Code (Pylance), PyCharm, and others
