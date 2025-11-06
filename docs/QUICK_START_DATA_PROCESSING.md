# Quick Start: Data Processing Guide

## 🎯 The Simple Answer

**Just downloaded new data from Databento?**

```bash
python src/process_new_data.py --market NQ
```

**That's it!** The script automatically:
- ✅ Detects if data is corrupted
- ✅ Fixes corruption automatically
- ✅ Validates data quality
- ✅ Saves clean training-ready data

---

## 📋 Which Script Should I Use?

### 🆕 For NEW Data (Just Downloaded)

**Use:** `process_new_data.py`

```bash
# Process new NQ data from Databento
python src/process_new_data.py --market NQ
```

**When:**
- You just downloaded new data from Databento (.zip file)
- You're adding a new market to your system
- You're updating existing data with newer data

**What it does:**
- Automatically detects corruption (you don't need to check!)
- Automatically fixes corruption (divide-by-100 errors)
- Shows you clear warnings if issues are detected
- Generates clean data ready for training

---

### 🔄 For OLD Data (Already Processed Before)

**Use:** `reprocess_from_source.py`

```bash
# Reprocess old corrupted NQ data
python src/reprocess_from_source.py --market NQ
```

**When:**
- You have OLD data that was processed BEFORE we added auto-fixing
- You want to clean up existing corrupted training files
- You're sure the data is corrupted and want to start fresh

**What it does:**
- Deletes old corrupted files
- Reprocesses from original zip files
- Applies all the new corruption fixes
- Generates fresh clean data

---

## 💡 Common Scenarios

### Scenario 1: "I just downloaded NQ data from Databento"

```bash
# Simple! Just run this:
python src/process_new_data.py --market NQ
```

**What you'll see if data is good:**
```
[PRE-FLIGHT CHECK] Analyzing data quality...
  ✅ Data quality looks good (median: $18,245.50)
```

**What you'll see if data is corrupted:**
```
[PRE-FLIGHT CHECK] Analyzing data quality...
  ⚠️  WARNING: Data appears corrupted!
  Expected median: $18,000.00
  Actual median: $240.00 (74.8x too low)
  → Auto-correction will be applied...

[PRICE FORMAT FIX]
  Found 3,039 bars (9.2%) below $3,600.00
  Multiplying corrupted bars by 100...
  ✅ Fix successful! New median: $18,245.50
```

**Either way, you get clean data!** No manual intervention needed.

---

### Scenario 2: "I have old NQ data that's corrupted"

```bash
# Reprocess from source
python src/reprocess_from_source.py --market NQ
```

This will:
1. Delete `NQ_D1M.csv` (old corrupted file)
2. Delete `NQ_D1S.csv` (if exists)
3. Reprocess from original zip file with auto-fixing
4. Generate fresh clean data

---

### Scenario 3: "I want to check if my data is corrupted"

```bash
# Just run the normal processing - it checks automatically
python src/process_new_data.py --market NQ
```

The script will tell you immediately if there's corruption and fix it automatically.

---

### Scenario 4: "I want to process multiple markets at once"

```bash
# Process ES, NQ, and YM all at once
python src/process_new_data.py --market ES NQ YM
```

Each market will be processed with automatic corruption detection.

---

## 🔍 How to Know If It Worked

After processing, check for these signs:

### ✅ Good Signs

```
✅ Data quality looks good (median: $18,245.50)
✅ Fix successful! New median: $18,245.50
✅ Clean data ready for training!
```

### ⚠️ Warning Signs

```
⚠️  Fix may not be complete. New median: $...
❌ Processing failed
```

If you see warnings:
1. Check the logs for details
2. Verify your zip file is correct
3. Try reprocessing: `python src/reprocess_from_source.py --market NQ`

---

## 📊 Expected Price Ranges

To verify your data looks correct:

| Market | Expected Median | Typical Range |
|--------|----------------|---------------|
| ES | ~$5,500 | $4,000 - $7,000 |
| NQ | ~$18,000 | $15,000 - $22,000 |
| YM | ~$35,000 | $30,000 - $40,000 |
| RTY | ~$2,100 | $1,800 - $2,400 |

After processing, your data should have median prices in these ranges.

---

## 🚀 Complete Workflow

### When You Get New Data from Databento

1. **Extract zip file to data folder**
   ```bash
   # Your zip file should be in data/
   ls data/*.zip
   ```

2. **Process the data**
   ```bash
   python src/process_new_data.py --market NQ
   ```

3. **Verify output**
   ```bash
   ls -lh data/NQ_D1M.csv
   # Should be 100-200 MB for NQ
   ```

4. **Train your model**
   ```bash
   python src/train_phase1.py --market NQ
   ```

5. **Evaluate**
   ```bash
   python src/evaluate_phase2.py --market NQ
   ```

---

## ❓ FAQ

### Q: Do I need to check if data is corrupted before processing?

**A: No!** The script automatically detects and fixes corruption. Just run it!

### Q: What if I'm not sure if my data is good or bad?

**A: It doesn't matter!** The script checks for you and fixes any issues automatically.

### Q: Can corruption detection fail?

**A: Unlikely.** The detection looks for prices that are 80% below expected values. If your data has the divide-by-100 error (like Databento data), it will be detected and fixed.

### Q: What if I want to see what would happen without actually processing?

**A: Use dry-run mode:**
```bash
python src/process_new_data.py --market NQ --dry-run
```

### Q: Should I delete my old corrupted data first?

**A: No need!**
- For NEW downloads: Just run `process_new_data.py` (overwrites old file)
- For OLD corrupted data: Use `reprocess_from_source.py` (deletes and reprocesses)

### Q: How long does processing take?

**Typical times:**
- ES: 2-5 minutes
- NQ: 3-7 minutes
- YM: 2-4 minutes
- All 8 markets: 20-40 minutes

---

## 🎯 Remember

**The key takeaway:** You don't need to worry about data corruption anymore!

Just run:
```bash
python src/process_new_data.py --market NQ
```

And you'll get clean, validated, training-ready data. No manual checking required! 🎉

---

## 📖 More Information

- **Detailed guide:** See `docs/DATA_REPROCESSING_GUIDE.md`
- **Technical details:** See `src/data_validator.py` source code
- **All validation functions:** Check `src/data_validator.py` docstrings
