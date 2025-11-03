# Smart Market Detection and Selection Implementation

## Overview
This implementation adds intelligent market detection and selection to the training scripts (Phase 1 and Phase 2). The system automatically detects all available market data files and prompts users to choose which market to train on when multiple datasets exist.

## Files Modified

### 1. `src/model_utils.py`
**New Functions Added:**

#### `detect_available_markets(data_dir='data')`
- Scans the data directory for all available market data files
- Looks for patterns: `{MARKET}_D1M.csv` and `D1M.csv`
- Returns a list of market dictionaries with metadata:
  - `market`: Market identifier (e.g., 'ES', 'NQ', 'GENERIC')
  - `minute_file`: Filename of minute-level data
  - `second_file`: Filename of second-level data (if exists)
  - `has_second`: Boolean indicating if second-level data is available
  - `path`: Full path to the minute data file

#### `select_market_for_training(markets, safe_print_func=print)`
- Prompts user to select a market when multiple options exist
- Automatically selects the market if only one dataset is available
- Returns selected market dictionary or None if cancelled
- Supports 'q' to quit, Ctrl+C to cancel
- Uses safe_print for Windows compatibility

### 2. `src/train_phase1.py`
**Changes Made:**

#### Import Statement (Line 55)
```python
from model_utils import get_model_save_name, detect_available_markets, select_market_for_training
```

#### `find_data_file(market=None)` (Lines 197-239)
- Added `market` parameter to support market-specific file lookups
- Prioritizes market-specific files (e.g., `ES_D1M.csv`) when market is specified
- Falls back to generic files and legacy names
- Maintains backward compatibility

#### `load_data(train_split=0.7, market=None)` (Lines 242-343)
- Added `market` parameter
- Passes market to `find_data_file()`
- Updated second-level data loading to check market-specific files first
- Improved candidate file search with duplicate removal

#### `train_phase1(continue_training=False, model_path=None)` (Lines 416-468)
- Added market detection and selection at the beginning of training
- Displays selected market information
- Exits gracefully if user cancels or no data is found
- Passes selected market to `load_data()`

### 3. `src/train_phase2.py`
**Changes Made:**

#### Import Statement (Line 56)
```python
from model_utils import detect_models_in_folder, detect_available_markets, select_market_for_training
```

#### `find_data_file(market=None)` (Lines 250-292)
- Same changes as train_phase1.py
- Added market parameter support
- Market-specific file prioritization

#### `load_data(train_split=0.7, market=None)` (Lines 295-384)
- Same changes as train_phase1.py
- Market parameter added
- Enhanced second-level data detection

#### `train_phase2()` (Lines 691-733)
- Added market detection and selection at start
- Displays selected market information
- Graceful exit on cancellation
- Passes market to `load_data()`

## Test Files Created

### 1. `test_market_selection.py`
- Basic functionality test script
- Tests market detection
- Tests auto-selection with single market
- Non-interactive test suite

### 2. `demo_market_selection.py`
- Comprehensive demonstration script
- Shows all scenarios:
  - Multiple markets (interactive selection)
  - Single market (auto-selection)
  - No data (error handling)
- Explains data file detection logic

## Behavior Examples

### Scenario 1: Multiple Markets Available
```
================================================================================
MARKET SELECTION
================================================================================

Detected 2 market datasets:

  1. ES       - ES_D1M.csv          [OK]
  2. NQ       - NQ_D1M.csv          [MINUTE ONLY]

================================================================================

Select market number (or 'q' to quit): 2

[DATA] Selected: NQ
[DATA] Minute data: NQ_D1M.csv
[DATA] Note: No second-level data available for NQ
[TRAINING] Market: NQ
```

### Scenario 2: Single Market Auto-Selection
```
[DATA] Auto-detected: ES (only dataset available)
[DATA] Using: ES_D1M.csv
[DATA] Second-level: ES_D1S.csv
[TRAINING] Market: ES
```

### Scenario 3: No Data Available
```
[ERROR] No market data files found!
[ERROR] Please run data processing first to create training data.
[ERROR] No market selected. Exiting training.
```

## Data File Priority Order

### Minute-Level Data
When `market='ES'`:
1. `data/ES_D1M.csv` (market-specific)
2. `data/D1M.csv` (generic)
3. `data/es_training_data_CORRECTED_CLEAN.csv` (legacy)
4. `data/es_training_data_CORRECTED.csv` (legacy)
5. `data/databento_es_training_data_processed_cleaned.csv` (legacy)
6. `data/databento_es_training_data_processed.csv` (legacy)
7. `data/*_D1M.csv` (wildcard - first match)

### Second-Level Data
When `market='ES'`:
1. `data/ES_D1S.csv` (market-specific)
2. `data/D1S.csv` (generic)
3. `data/*_D1S.csv` (wildcard - first match)

## Backward Compatibility

The implementation maintains full backward compatibility:
- If no market parameter is provided, old behavior is preserved
- Legacy filenames are still supported
- Existing training workflows are unaffected
- Generic `D1M.csv` and `D1S.csv` files still work

## Key Features

✅ **Auto-detection**: Scans data directory for all available markets
✅ **Smart selection**: Prompts only when multiple options exist
✅ **Auto-selection**: Uses single dataset automatically
✅ **Clear feedback**: Shows which data files are being used
✅ **Error handling**: Graceful handling of missing data
✅ **User control**: Can quit with 'q' or Ctrl+C
✅ **Windows compatible**: Uses safe_print for proper output
✅ **Second-level aware**: Detects and reports second-level data availability

## Testing

### Run the test suite:
```bash
python3 test_market_selection.py
```

### Run the demo:
```bash
python3 demo_market_selection.py
```

### Test interactively (requires user input):
```bash
python3 src/train_phase1.py --test
python3 src/train_phase2.py --test
```

### Verify syntax:
```bash
cd src
python3 -m py_compile train_phase1.py train_phase2.py model_utils.py
```

## Integration Notes

- The market selection happens **before** any heavy data loading
- Selection is done **once** at the start of training
- Selected market is passed through to all data loading functions
- No changes required to environment files or other components
- VecNormalize and checkpoint saving are unaffected

## Future Enhancements

Potential improvements for future versions:
- Save selected market in config file for continuation training
- Support for filtering markets by date range
- Display data statistics during selection (rows, date range, etc.)
- Support for multiple data directories
- Market-specific model naming conventions

## Summary of Changes

**Lines of Code Added:**
- `src/model_utils.py`: ~120 lines (2 new functions)
- `src/train_phase1.py`: ~40 lines modified
- `src/train_phase2.py`: ~40 lines modified
- Test files: ~200 lines

**Breaking Changes:** None

**Dependencies Added:** None

**Compatibility:** Fully backward compatible
