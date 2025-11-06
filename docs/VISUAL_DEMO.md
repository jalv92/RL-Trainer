# Visual Demo - Market Selection in Action

This document shows exactly what users will see when running the training scripts.

---

## Scenario 1: Multiple Markets Available (Interactive Selection)

When a user has both ES and NQ data:

```
$ python3 src/train_phase1.py

================================================================================
PHASE 1: FOUNDATIONAL TRADING PATTERNS
================================================================================

================================================================================
MARKET SELECTION
================================================================================

Detected 2 market datasets:

  1. ES       - ES_D1M.csv          [OK]
  2. NQ       - NQ_D1M.csv          [OK]

================================================================================

Select market number (or 'q' to quit): 1

[DATA] Selected: ES
[DATA] Minute data: ES_D1M.csv
[DATA] Second data: ES_D1S.csv

[TRAINING] Market: ES
[CONFIG] Total timesteps: 2,000,000
[CONFIG] Parallel envs (requested): 80
[CONFIG] Parallel envs (effective): 80
[SYSTEM] BLAS threads per process: 1
[CONFIG] Network: [512, 256, 128]
[CONFIG] Device: cuda
[CONFIG] Fixed SL: 1.5x ATR
[CONFIG] Fixed TP: 3.0x SL

[DATA] Loading minute-level data from /path/to/data/ES_D1M.csv
[DATA] Loaded 353,877 rows
[DATA] Full date range: 2023-01-03 18:00:00-05:00 to 2024-10-25 17:00:00-04:00

[SPLIT] Train/Val Split Applied:
[SPLIT] Train: 247,713 bars (70%) - 2023-01-03 18:00:00-05:00 to 2024-05-21 15:14:00-04:00
[SPLIT] Val:   106,164 bars (30%) - 2024-05-21 15:15:00-04:00 to 2024-10-25 17:00:00-04:00

...training continues...
```

---

## Scenario 2: Single Market Available (Auto-Selection)

When a user has only ES data:

```
$ python3 src/train_phase1.py

================================================================================
PHASE 1: FOUNDATIONAL TRADING PATTERNS
================================================================================

[DATA] Auto-detected: ES (only dataset available)
[DATA] Using: ES_D1M.csv
[DATA] Second-level: ES_D1S.csv

[TRAINING] Market: ES
[CONFIG] Total timesteps: 2,000,000
[CONFIG] Parallel envs (requested): 80
...

[DATA] Loading minute-level data from /path/to/data/ES_D1M.csv
[DATA] Loaded 353,877 rows

...training continues immediately without user interaction...
```

---

## Scenario 3: User Selects NQ

```
$ python3 src/train_phase1.py

================================================================================
PHASE 1: FOUNDATIONAL TRADING PATTERNS
================================================================================

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
[CONFIG] Total timesteps: 2,000,000
...

[DATA] Loading minute-level data from /path/to/data/NQ_D1M.csv
[DATA] Second-level data not found (expected NQ_D1S.csv or D1S.csv) - continuing without it

...training continues with NQ data...
```

---

## Scenario 4: User Cancels Selection

```
$ python3 src/train_phase1.py

================================================================================
PHASE 1: FOUNDATIONAL TRADING PATTERNS
================================================================================

================================================================================
MARKET SELECTION
================================================================================

Detected 2 market datasets:

  1. ES       - ES_D1M.csv          [OK]
  2. NQ       - NQ_D1M.csv          [OK]

================================================================================

Select market number (or 'q' to quit): q

[INFO] Training cancelled by user

[ERROR] No market selected. Exiting training.
```

Or with Ctrl+C:

```
Select market number (or 'q' to quit): ^C

[INFO] Training cancelled by user

[ERROR] No market selected. Exiting training.
```

---

## Scenario 5: No Data Available

```
$ python3 src/train_phase1.py

================================================================================
PHASE 1: FOUNDATIONAL TRADING PATTERNS
================================================================================

[ERROR] No market data files found!
[ERROR] Please run data processing first to create training data.

[ERROR] No market selected. Exiting training.
```

---

## Scenario 6: Invalid Input Handling

```
Select market number (or 'q' to quit): 5

[ERROR] Invalid choice. Please enter 1-2

Select market number (or 'q' to quit): abc

[ERROR] Invalid input. Please enter a number or 'q'

Select market number (or 'q' to quit): 1

[DATA] Selected: ES
...training continues...
```

---

## Test Mode Example

With --test flag, training runs quickly for verification:

```
$ python3 src/train_phase1.py --test

================================================================================
TEST MODE ENABLED - Quick Local Testing
================================================================================
[TEST] Timesteps:       2M -> 30,000 (1.5% for testing)
[TEST] Parallel envs:   80 -> 4 (local machine)
[TEST] Eval frequency:  Every 10,000 steps
[TEST] Early stopping:  DISABLED (test mode)
[TEST] Expected time:   ~5-10 minutes
[TEST] Purpose:         Verify pipeline works before full training
================================================================================

================================================================================
PHASE 1: FOUNDATIONAL TRADING PATTERNS
================================================================================

[DATA] Auto-detected: ES (only dataset available)
[DATA] Using: ES_D1M.csv
[DATA] Second-level: ES_D1S.csv

[TRAINING] Market: ES
[CONFIG] Total timesteps: 30,000
[CONFIG] Parallel envs (requested): 4
...
```

---

## Phase 2 Example

Phase 2 works identically:

```
$ python3 src/train_phase2.py

================================================================================
PHASE 2: POSITION MANAGEMENT MASTERY
================================================================================

================================================================================
MARKET SELECTION
================================================================================

Detected 2 market datasets:

  1. ES       - ES_D1M.csv          [OK]
  2. NQ       - NQ_D1M.csv          [OK]

================================================================================

Select market number (or 'q' to quit): 1

[DATA] Selected: ES
[DATA] Minute data: ES_D1M.csv
[DATA] Second data: ES_D1S.csv

[TRAINING] Market: ES
[CONFIG] Total timesteps: 5,000,000
[CONFIG] Parallel envs (requested): 80
[CONFIG] Network: [512, 256, 128]
[CONFIG] Action space: 6 (RL Fix #10: simplified from 9 to 6)
[CONFIG] Device: cuda
...
```

---

## Summary of User Experience

### What Users See:
✅ Clear market detection with file information
✅ Second-level data availability status
✅ Simple numbered menu for selection
✅ Option to quit at any time
✅ Immediate feedback on selection
✅ Graceful error handling

### What Users Don't See:
- Technical file path details (unless verbose)
- Complex detection logic
- Fallback mechanisms
- Error stack traces (unless actual errors)

### User Interaction:
- **Multiple markets:** Type number (1, 2, 3, etc.) or 'q' to quit
- **Single market:** No interaction needed (auto-selected)
- **No markets:** Clear error message with next steps

---

## Integration with Existing Workflow

The market selection happens **before** any heavy data loading or model initialization:

```
1. Training script starts
2. Display header (Phase 1 or Phase 2)
3. → MARKET SELECTION (NEW) ←
4. Load data
5. Create environments
6. Initialize model
7. Start training
```

This ensures:
- Fast feedback (detection is instant)
- No wasted resources if user quits
- Clean separation of concerns
- Easy to understand flow

---

*This visual demo shows the actual user interface for market selection.*
*All scenarios have been tested and verified to work as shown.*
