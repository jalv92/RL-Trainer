# Data Reprocessing Guide

## Table of Contents
1. [Overview](#overview)
2. [The Problem](#the-problem)
3. [The Solution](#the-solution)
4. [Quick Start](#quick-start)
5. [Detailed Usage](#detailed-usage)
6. [How It Works](#how-it-works)
7. [Verification](#verification)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This guide explains how to reprocess market data from source zip files with automatic corruption detection and fixing. The data processing pipeline now includes sophisticated validation and correction mechanisms to ensure clean, high-quality training data.

**When to Use This:**
- After discovering corrupted data in your training files
- When adding a new market to your pipeline
- To update existing market data with latest vendor data
- After pipeline improvements (to apply fixes retroactively)

---

## The Problem

### What Was Wrong?

During evaluation of an NQ model, we discovered severe data corruption:
- **Buy/Sell Imbalance:** 5 buys vs 259 sells (1:51.8 ratio)
- **Impossible Returns:** 19,771% returns that were mathematically impossible
- **Root Cause:** Vendor data (Databento) contained ~10% corrupted bars

### The Corruption Pattern

The vendor data had a "divide-by-100" error where some prices were incorrectly scaled:

```
Normal:    $22,655.00  ✅
Corrupted: $226.55     ❌ (divided by 100)
```

**Impact Statistics:**
- **NQ Data:** 9.2% of bars corrupted (3,039 out of 33,033 bars)
- **ES Data:** 11.7% of bars corrupted

This corruption caused the trained models to:
1. See artificially low prices as "buy" opportunities
2. Execute excessive sell orders (thinking prices were high)
3. Generate invalid performance metrics

---

## The Solution

### Automatic Corruption Detection & Fixing

The pipeline now includes three layers of protection:

#### 1. **Price Format Detection** (`src/data_validator.py`)
- Compares median price to expected range for each market
- Detects divide-by-100 errors automatically
- Multiplies corrupted bars by 100 to correct them

#### 2. **Market-Agnostic Statistical Validation**
- Uses IQR (Interquartile Range) method instead of hardcoded limits
- Works across all markets (ES, NQ, YM, RTY, etc.)
- Rejects statistical outliers (beyond 3 IQR from quartiles)

#### 3. **Comprehensive Data Cleaning**
- Removes NaN values
- Removes duplicate timestamps
- Validates OHLC consistency (high ≥ low)
- Detects anomalous price jumps (>5%)

### Files Modified

| File | Purpose | Changes |
|------|---------|---------|
| `src/data_validator.py` | **NEW** - Centralized validation module | All validation logic in one place |
| `src/update_training_data.py` | Minute data processing | Added price format detection |
| `src/process_second_data.py` | Second data processing | Added price format detection |
| `src/clean_second_data.py` | Data cleaning | Replaced ES-specific limits with IQR method |
| `src/reprocess_from_source.py` | **NEW** - Reprocessing utility | Easy way to reprocess data |

---

## Quick Start

### Reprocess a Single Market

```bash
# Reprocess NQ data (deletes old NQ files and reprocesses from zip)
python src/reprocess_from_source.py --market NQ
```

### Reprocess Multiple Markets

```bash
# Reprocess ES, NQ, and YM
python src/reprocess_from_source.py --market ES NQ YM
```

### Reprocess All Markets

```bash
# Reprocess all 8 supported markets
python src/reprocess_from_source.py --all
```

### Dry Run (Preview Changes)

```bash
# See what would be deleted without actually deleting
python src/reprocess_from_source.py --market NQ --dry-run
```

---

## Detailed Usage

### Step-by-Step: Reprocessing NQ Data

1. **Check Current Data**
   ```bash
   ls -lh data/NQ*.csv
   ```

   You should see files like:
   - `NQ_D1M.csv` - Minute-level training data
   - `NQ_D1S.csv` - Second-level data (optional)

2. **Run Reprocessing Script**
   ```bash
   python src/reprocess_from_source.py --market NQ
   ```

3. **Confirm Deletion**
   ```
   ⚠️  WARNING: This will DELETE existing data files for these markets!
   Data will be reprocessed from source zip files with corruption fixes.

   Continue? (yes/no): yes
   ```

4. **Monitor Progress**

   The script will:
   - Delete old NQ data files
   - Extract data from zip files
   - **Automatically detect and fix** corrupted prices
   - Validate data statistically
   - Generate clean training files

5. **Review Output**

   Look for these key indicators:

   ```
   [PRICE FORMAT CHECK]
     Market: NQ
     Expected median: $18,000.00
     Actual median: $240.00
     ⚠️  Median is 74.8x too low - format issue detected!
     Found 3,039 bars (9.2%) below $3,600.00
     Multiplying corrupted bars by 100...
     ✅ Fix successful! New median: $18,245.50
   ```

6. **Verify Results**

   The script automatically verifies output files:

   ```
   VERIFICATION
   ✅ NQ_D1M.csv: 156.23 MB
   ✅ NQ_D1S.csv: 1,234.56 MB

   ✅ NQ reprocessing completed successfully!
   ```

---

## How It Works

### Data Processing Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Load Raw Vendor Data (from zip files)                   │
│    - Databento OHLCV CSV format                            │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 2. Filter Market Contracts                                  │
│    - Extract specific market (ES, NQ, YM, etc.)            │
│    - Handle contract rollovers                             │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 3. Detect & Fix Price Format ⚡ NEW                        │
│    - Compare median to expected price                      │
│    - Detect divide-by-100 errors                           │
│    - Auto-correct corrupted bars                           │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 4. Statistical Validation ⚡ IMPROVED                      │
│    - Market-agnostic IQR method                            │
│    - Works for all markets (not just ES)                   │
│    - Rejects statistical outliers                          │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 5. Comprehensive Cleaning                                   │
│    - Remove NaN values                                      │
│    - Remove duplicates                                      │
│    - Validate OHLC consistency                             │
│    - Detect price jumps                                     │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 6. Generate Training-Ready Data                             │
│    - Clean OHLCV CSV files                                  │
│    - Ready for Phase 1 & Phase 2 training                  │
└─────────────────────────────────────────────────────────────┘
```

### Price Format Detection Algorithm

```python
# Pseudocode for price format detection
expected_median = MARKET_MEDIANS[market]  # e.g., NQ = $18,000
actual_median = data['close'].median()

if actual_median < expected_median * 0.20:  # Suspiciously low
    ratio = expected_median / actual_median

    if 50 < ratio < 150:  # Divide-by-100 error
        # Find corrupted bars (below 20% threshold)
        threshold = expected_median * 0.20
        corrupted_mask = data['close'] < threshold

        # Multiply by 100 to fix
        for col in ['open', 'high', 'low', 'close']:
            data.loc[corrupted_mask, col] *= 100

        # Verify fix worked
        new_median = data['close'].median()
        if 0.80 * expected_median < new_median < 1.50 * expected_median:
            print("✅ Fix successful!")
```

### Market-Agnostic Validation

**Old Method (ES-specific):**
```python
min_price = 500      # ❌ Hardcoded
max_price = 15000    # ❌ Rejects NQ > $15K
```

**New Method (Market-agnostic):**
```python
q1 = data['close'].quantile(0.25)
q3 = data['close'].quantile(0.75)
iqr = q3 - q1

min_price = max(0, q1 - 3*iqr)  # ✅ Statistical
max_price = q3 + 3*iqr           # ✅ Adaptive
```

---

## Verification

### How to Verify Data Quality

After reprocessing, check these indicators:

#### 1. **File Sizes**

Minute data should be substantial:
```bash
ls -lh data/*_D1M.csv
```

Expected sizes:
- ES: ~80-150 MB
- NQ: ~100-200 MB
- YM: ~60-100 MB

#### 2. **Price Statistics**

Check median prices using Python:
```python
import pandas as pd

# Load processed data
df = pd.read_csv('data/NQ_D1M.csv', index_col=0, parse_dates=True)

# Check price statistics
print(f"Median: ${df['close'].median():,.2f}")
print(f"Min:    ${df['close'].min():,.2f}")
print(f"Max:    ${df['close'].max():,.2f}")

# Expected for NQ:
# Median: ~$18,000
# Min:    ~$15,000
# Max:    ~$22,000
```

#### 3. **Corruption Check**

Look for suspiciously low prices:
```python
# Check for prices below expected minimum
threshold = 10000  # For NQ
low_prices = df[df['close'] < threshold]

print(f"Bars below ${threshold}: {len(low_prices)}")
# Should be 0 or very close to 0
```

#### 4. **Train a Test Model**

The ultimate verification is to train and evaluate a model:
```bash
# Train Phase 1 on reprocessed data
python src/train_phase1.py --market NQ --test

# Evaluate (look for balanced buy/sell orders)
python src/evaluate_phase2.py --market NQ
```

Expected: Roughly balanced buy/sell ratio (not 1:51 like before!)

---

## Troubleshooting

### Problem: "No market data found in zip file"

**Cause:** The market you're trying to process doesn't exist in the vendor data

**Solution:**
1. Check what markets are in your zip file:
   ```bash
   python src/update_training_data.py --market ES --help
   ```
2. Only process markets that exist in your data

### Problem: "Fix may not be complete. New median: $..."

**Cause:** The corruption pattern doesn't match the divide-by-100 error

**Solution:**
1. Check the logs to see actual vs expected median
2. The data may have a different corruption pattern
3. Manual inspection may be required:
   ```python
   import pandas as pd
   df = pd.read_csv('data/NQ_D1M.csv', index_col=0, parse_dates=True)
   print(df.describe())
   ```

### Problem: "File seems too small"

**Cause:** Very little data passed validation

**Solutions:**
1. Check if the source zip file has sufficient data
2. Review validation thresholds in `src/data_validator.py`
3. Check logs for excessive filtering

### Problem: Reprocessing takes too long

**Expected Times:**
- ES: ~2-5 minutes
- NQ: ~3-7 minutes
- All 8 markets: ~20-40 minutes

**If much slower:**
1. Check disk I/O (WSL2 can be slow on Windows filesystem)
2. Move data to Linux filesystem if using WSL2
3. Check CPU usage

### Problem: Still seeing buy/sell imbalance after reprocessing

**Diagnostic Steps:**

1. **Verify clean data:**
   ```python
   import pandas as pd
   df = pd.read_csv('data/NQ_D1M.csv', index_col=0, parse_dates=True)

   # Should be around expected median
   print(f"Median: ${df['close'].median():,.2f}")

   # Should have no values below 20% of expected
   low_count = (df['close'] < 3600).sum()  # 20% of $18,000
   print(f"Suspiciously low prices: {low_count}")
   ```

2. **Check model files:**
   - Delete old trained models: `rm -rf models/*NQ*`
   - Retrain from scratch on clean data

3. **Verify evaluation is using new data:**
   - Check timestamps in evaluation logs
   - Ensure evaluation script loaded the new CSV file

---

## Expected Output Example

### Successful NQ Reprocessing

```
======================================================================
REPROCESSING NQ FROM SOURCE
======================================================================

Deleting NQ data files:
  Deleting: NQ_D1M.csv (156.23 MB)
    ✅ Deleted
  Deleting: NQ_D1S.csv (1234.56 MB)
    ✅ Deleted

Deleted 2 files (1390.79 MB total)

======================================================================
PROCESSING NQ MINUTE DATA
======================================================================

[1/4] Loading and filtering data...
      Loaded: 50,000 bars

[2/4] Feature engineering...
      Added 15 technical indicators

[3/4] Detecting and fixing price format...
  [PRICE FORMAT CHECK]
    Market: NQ
    Expected median: $18,000.00
    Actual median: $240.00
    ⚠️  Median is 74.8x too low - format issue detected!
    Found 3,039 bars (9.2%) below $3,600.00
    Multiplying corrupted bars by 100...
    ✅ Fix successful! New median: $18,245.50

[4/4] Saving training data...
      Saved: data/NQ_D1M.csv (156.45 MB)

======================================================================
VERIFICATION
======================================================================
✅ NQ_D1M.csv: 156.45 MB

✅ NQ reprocessing completed successfully!
```

---

## Next Steps After Reprocessing

Once you've reprocessed your data:

### 1. **Delete Old Models**
```bash
# Remove models trained on corrupted data
rm -rf models/*NQ*
```

### 2. **Retrain Models**
```bash
# Train fresh Phase 1 model on clean data
python src/train_phase1.py --market NQ

# Phase 2 will start automatically after Phase 1
```

### 3. **Evaluate New Models**
```bash
# Evaluate on clean data
python src/evaluate_phase2.py --market NQ
```

### 4. **Verify Results**

You should now see:
- ✅ Balanced buy/sell ratio (not 1:51!)
- ✅ Realistic returns (not 19,771%!)
- ✅ Median prices around $18,000 for NQ
- ✅ Training metrics that make sense

---

## Reference: Supported Markets

| Market | Name | Expected Median | Typical Range |
|--------|------|----------------|---------------|
| ES | E-mini S&P 500 | $5,500 | $4,000-$7,000 |
| NQ | E-mini Nasdaq-100 | $18,000 | $15,000-$22,000 |
| YM | E-mini Dow | $35,000 | $30,000-$40,000 |
| RTY | E-mini Russell 2000 | $2,100 | $1,800-$2,400 |
| MES | Micro E-mini S&P 500 | $5,500 | $4,000-$7,000 |
| MNQ | Micro E-mini Nasdaq-100 | $18,000 | $15,000-$22,000 |
| M2K | Micro E-mini Russell 2000 | $2,100 | $1,800-$2,400 |
| MYM | Micro E-mini Dow | $35,000 | $30,000-$40,000 |

---

## Summary

The data processing pipeline now includes:
- ✅ **Automatic corruption detection** - Finds divide-by-100 errors
- ✅ **Automatic correction** - Multiplies corrupted bars by 100
- ✅ **Market-agnostic validation** - Works for all futures markets
- ✅ **Statistical outlier rejection** - IQR method instead of hardcoded limits
- ✅ **Easy reprocessing** - One command to clean and rebuild data

**You can now confidently reprocess data from source knowing that corruption will be automatically detected and fixed.**

---

## Questions?

If you encounter issues not covered in this guide:
1. Check the logs for detailed error messages
2. Review the source code in `src/data_validator.py`
3. Run with `--dry-run` first to preview changes
4. Verify your vendor data format matches expectations

**Remember:** The pipeline is designed to be defensive and will skip processing if it detects unexpected data patterns.
