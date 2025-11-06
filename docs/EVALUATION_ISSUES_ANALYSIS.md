# Evaluation Issues Analysis

**Document Purpose:** Comprehensive analysis of evaluation and training issues preventing normal trading behavior
**Created:** November 5, 2025
**Status:** Issues identified, solutions pending implementation

---

## Executive Summary

The Phase 2 model evaluation is experiencing critical issues resulting in:
- **Early episode termination** at step 41 out of 5000 (0.82% completion)
- **Minimal trading activity** (1 buy, 0 sells, 40 holds, 0 position management actions)
- **Episode ends before any meaningful trading** can occur

Multiple root causes have been identified through log analysis, including data mismatches, undertrained models, and potential configuration errors.

---

## Current Evaluation State

**Latest Evaluation Results** (`results/phase2_metrics.json`):
```json
{
  "total_steps": 41,
  "final_equity": 49993.51782246879,
  "total_return_pct": -0.012964355062416872,
  "sharpe_ratio": -50.19960159204454,
  "max_drawdown_pct": 0.012964355062416872,
  "pm_actions_used": 0,
  "pm_usage_pct": 0.0,
  "action_distribution": {
    "0": 40,  // Hold action
    "1": 1    // Buy action
  }
}
```

**Key Observations:**
- Episode terminates at step 41 (should run for up to 5000 steps)
- Only 2 unique actions taken across entire episode (Hold: 40x, Buy: 1x)
- No sell orders executed
- No position management actions used (Move to BE, Enable/Disable Trail)
- Slightly negative return (-0.013%)
- Extremely negative Sharpe ratio (-50.2) indicates poor risk-adjusted performance

---

## Historical Evaluation Comparison

Analysis of `logs/evaluation.log` reveals 4 distinct evaluation runs with very different behaviors:

### Evaluation 1: October 25, 2025 (ES Market - Baseline)
- **Market:** ES (E-mini S&P 500)
- **Steps:** 49 (early termination)
- **Actions:** 5 buys, 4 closes
- **Behavior:** Some trading activity before early termination

### Evaluation 2: November 3, 2025 (NQ Market - Corrupted Data)
- **Market:** NQ (E-mini NASDAQ-100)
- **Steps:** 5000 (full episode)
- **Actions:** 5 buys, 259 sells (1:51.8 ratio - extreme sell bias)
- **Returns:** +19,771% (physically impossible)
- **Issue:** Data corruption with divide-by-100 errors ($22,655 → $226.55)
- **Impact:** Model learned on corrupted data, extreme behavioral bias

### Evaluation 3: November 4, 2025 (NQ Market - After Partial Fix)
- **Market:** NQ
- **Steps:** 41 (early termination)
- **Actions:** 1 buy, 0 sells
- **Issue:** Episode ends before meaningful trading

### Evaluation 4: November 4, 2025 (NQ Market - After Full Fix) **← CURRENT STATE**
- **Market:** NQ
- **Steps:** 41 (early termination)
- **Actions:** 1 buy, 0 sells
- **Issue:** Same as Evaluation 3 - no improvement

**Pattern Analysis:**
- All NQ evaluations after corruption fix terminate at exactly step 41
- ES evaluation terminated at step 49 (similar early termination pattern)
- Only the corrupted data evaluation ran full 5000 steps (but with incorrect behavior)
- Suggests evaluation termination issue is independent of data corruption

---

## Critical Issues Identified

### Issue 1: Data Mismatch in Evaluator (CRITICAL BUG)

**Evidence:** `logs/evaluation.log` lines 369-370
```
[DATA] Loading second-level data from C:\Users\javlo\Documents\Code Projects\tensortrade\AI Trainer\data\ES_D1S.csv
```

**Context:** The evaluation is for NQ market (line 358 shows "NQ_D1M.csv" for minute data), but the evaluator loads ES second-level data.

**Location:** Likely in `src/evaluate_phase2.py` or environment initialization

**Impact:**
- Model trained on NQ data
- Evaluated with NQ minute data BUT ES second-level data
- Second-level data used for precise execution timing and spread calculations
- Mismatch causes incorrect order execution, spreads, and market microstructure
- May trigger unexpected episode termination conditions

**Why This Matters:**
- ES tick size: $12.50 per tick ($50 multiplier × 0.25)
- NQ tick size: $5.00 per tick ($20 multiplier × 0.25)
- Different price levels ($6,400 vs $24,000)
- Different volatility characteristics
- Wrong spread calculations and slippage models

**Required Fix:** Evaluator must load second-level data matching the market being evaluated (NQ_D1S.csv for NQ, ES_D1S.csv for ES)

---

### Issue 2: Undertrained Model (CRITICAL)

**Evidence:** `logs/training_phase1_test.log` (end of file)
```
[TRAIN] Starting Phase 1 training for 30,000 timesteps...
...
Total timesteps: 32,768
Training time: 0.01 hours
```

**Analysis:**
- Model trained for only **32,768 timesteps**
- Training was done in **TEST MODE** (indicated by 30,000 target and 0.01 hours)
- Production Phase 1 training should be **2,000,000 timesteps** (~61x more)
- Phase 2 training should be **1,000,000 timesteps**

**Impact:**
- Model has not learned proper trading patterns
- Insufficient exploration of state-action space
- No meaningful policy convergence
- Explains minimal trading activity (model hasn't learned when to trade)

**Why This Matters:**
- RL models require extensive training to learn complex trading strategies
- Test mode is designed for debugging, not for creating production models
- A model trained for <1% of required timesteps cannot be expected to perform well

**Required Fix:** Train Phase 1 model with full 2,000,000 timesteps, then train Phase 2 with 1,000,000 timesteps

---

### Issue 3: VecNormalize Statistics Mismatch (LIKELY)

**Evidence:** Model loads "phase2_position_mgmt_final_vecnorm.pkl" (normalization statistics)

**Hypothesis:**
- If VecNormalize was created during ES training but evaluation uses NQ data
- Observation normalization would use wrong mean/std statistics
- Price levels: ES ~$6,400 vs NQ ~$24,000 (3.75x difference)
- Normalized observations would be outside trained distribution

**Impact:**
- Model receives observations that look "out of distribution"
- May trigger unexpected behavior or conservative actions (holds)
- Could contribute to early episode termination if observations are extreme

**How to Verify:**
- Check if phase2_position_mgmt_final_vecnorm.pkl was created during ES or NQ training
- Check if current evaluation market matches vecnorm training market
- Look for observation statistics in logs during episode termination

**Required Fix:**
- Ensure VecNormalize statistics match the market being evaluated
- Train separate models for ES and NQ, or
- Retrain VecNormalize statistics when switching markets
- Note: As of Nov 2025, naming convention updated to match model names (e.g., phase2_position_mgmt_test_vecnorm.pkl)

---

### Issue 4: Early Episode Termination Pattern (UNKNOWN ROOT CAUSE)

**Evidence:**
- Oct 25 (ES): Terminates at step 49
- Nov 4 (NQ): Terminates at step 41 (both evaluations)

**Observations:**
- Termination occurs very early (0.82-0.98% of max episode length)
- No obvious error messages in logs
- Pattern is consistent (same step count across multiple runs)
- Different markets terminate at different steps (ES: 49, NQ: 41)

**Possible Causes:**

1. **Data-related termination:**
   - Not enough valid bars in evaluation dataset
   - Data validation failing early in episode
   - Data mismatch (Issue #1) triggering safety checks

2. **Episode done condition triggered:**
   - Account blown (equity < margin requirement)
   - Maximum drawdown exceeded
   - Invalid state detected by environment

3. **Data indexing issue:**
   - Environment trying to access data beyond available range
   - Off-by-one error in episode length calculation
   - Second-level data shorter than minute-level data

4. **Position-related termination:**
   - Open position at end of available data
   - Unable to close position due to data/market mismatch
   - Position management logic error

**Terminal Observation Analysis:**
The terminal observation in logs shows a complex state vector with multiple NaN values, suggesting potential state corruption or uninitialized features.

**Required Investigation:**
- Add detailed logging to environment's `step()` and `reset()` methods
- Log the exact reason for episode termination (done=True)
- Check data lengths for both minute and second-level data
- Verify episode length calculation matches available data

---

### Issue 5: No Position Management Actions Used

**Evidence:** `pm_actions_used: 0, pm_usage_pct: 0.0`

**Context:**
- Phase 2 has 6 actions: Hold, Buy, Sell, Move to BE, Enable Trail, Disable Trail
- Only Hold (40x) and Buy (1x) were used
- No position management actions attempted

**Possible Causes:**

1. **Episode too short:**
   - Terminates at step 41 before position can be managed
   - After buy at some step, episode ends before PM actions needed

2. **Action masking too restrictive:**
   - PM actions masked out when they should be available
   - Mask logic error preventing PM action selection

3. **Undertrained model:**
   - Model hasn't learned when to use PM actions
   - Only learned simplest actions (Hold, Buy)

4. **No position to manage:**
   - If position never opened properly, PM actions would be masked
   - Data mismatch might prevent proper position tracking

**Required Investigation:**
- Log action masks at each step to see what actions are available
- Verify position is properly opened and tracked after buy action
- Check if PM actions are ever unmasked during episode
- Verify mask logic in `environment_phase2.py`

---

### Issue 6: Observation State Quality

**Evidence:** Terminal observation in logs contains multiple NaN values and extreme values

**Impact:**
- NaN values in observations can cause model to output unexpected actions
- May trigger safety mechanisms causing episode termination
- Indicates feature engineering or normalization issues

**Possible Causes:**
- Division by zero in feature calculations
- Invalid market data (empty bars, missing values)
- VecNormalize encountering out-of-distribution values
- Data mismatch causing incompatible feature calculations

**Required Investigation:**
- Review feature engineering in `src/feature_engineering.py`
- Check for division by zero or log(0) operations
- Add NaN detection and handling in observation generation
- Verify all technical indicators handle edge cases

---

## Data Quality History

### Data Corruption Issue (RESOLVED)

**Problem:** 9-12% of NQ data had divide-by-100 errors from Databento vendor

**Examples:**
- $22,655.00 → $226.55 (divided by 100)
- Created bimodal distribution (90% clean + 10% corrupted)

**Detection Challenge:**
- Initial detector used median-only check
- With 90% clean data, median was correct
- Corruption in minority (bottom 10%) was invisible to median

**Solution Implemented:**
- Switched to percentile-based detection (P1, P5)
- Centralized detection in `src/data_validator.py`
- Updated expected median prices to Nov 2025 market levels (NQ: $18K → $24K, ES: $5.5K → $6.4K)

**Current Status:** Data corruption detection and fixing is working correctly

---

## Environment Configuration

### Market Specifications

**NQ (E-mini NASDAQ-100):**
- Current Price: ~$24,000
- Contract Multiplier: $20
- Tick Size: 0.25 points
- Tick Value: $5.00
- Commission: ~$2.50 per side

**ES (E-mini S&P 500):**
- Current Price: ~$6,400
- Contract Multiplier: $50
- Tick Size: 0.25 points
- Tick Value: $12.50
- Commission: ~$2.50 per side

### Action Space (Phase 2)
- Action 0: Hold
- Action 1: Buy
- Action 2: Sell
- Action 3: Move Stop Loss to Breakeven
- Action 4: Enable Trailing Stop
- Action 5: Disable Trailing Stop

### Phase 2 Curriculum Learning
- Agent has full control over position management
- Can dynamically adjust stop loss and trailing stops
- Should learn when to secure profits and minimize losses
- Requires sophisticated strategy learning (needs full training)

---

## Impact Assessment

### Immediate Impact
- **Model is completely non-functional** for live trading
- Cannot complete evaluation episodes
- No meaningful trading strategy demonstrated
- Risk metrics cannot be properly assessed

### Training Impact
- Undertrained model wastes evaluation time
- Cannot assess curriculum learning effectiveness
- Cannot validate Phase 2 position management features
- Cannot compare ES vs NQ performance

### Development Impact
- Cannot proceed with next development phases
- Cannot validate recent code changes work correctly
- Blocks testing of multi-market support
- Prevents validation of data quality improvements

---

## Recommendations for Resolution

### Priority 1 (Must Fix Immediately)

1. **Fix Data Mismatch in Evaluator**
   - Location: `src/evaluate_phase2.py`
   - Action: Pass market parameter to second-level data loader
   - Expected: Load NQ_D1S.csv when evaluating NQ, ES_D1S.csv when evaluating ES
   - Validation: Check logs show correct second-level file loaded

2. **Train Model with Full Timesteps**
   - Phase 1: Train for 2,000,000 timesteps (not 32,768)
   - Phase 2: Train for 1,000,000 timesteps
   - Use production settings, not test mode
   - Validation: Check training logs show full timestep completion

3. **Verify VecNormalize Matches Evaluation Market**
   - Check which market was used to create phase2_vecnorm.pkl
   - If mismatch, retrain model on correct market
   - Or train separate models for each market

### Priority 2 (Important for Diagnosis)

4. **Add Episode Termination Logging**
   - Log exact reason for done=True in environment
   - Add data length checks and log available vs requested bars
   - Log account equity, position status, and done conditions at each step
   - Location: `src/environment_phase2.py` step() method

5. **Verify Episode Length Configuration**
   - Check evaluation data has at least 5000 minute bars
   - Verify second-level data matches minute-level data length
   - Add assertions to catch length mismatches early

6. **Add Action Mask Logging**
   - Log available actions (action_mask) at each step
   - Track when PM actions become available
   - Verify mask logic is correct for position states

### Priority 3 (Data Quality)

7. **Add Observation Quality Checks**
   - Add NaN detection in observation generation
   - Add range checks for normalized observations
   - Log warning when observations are out of expected range
   - Add safeguards in feature engineering

8. **Validate Data Loading**
   - Add file existence checks before loading
   - Verify data dimensions match expected shapes
   - Log data statistics at environment initialization

---

## Testing Protocol After Fixes

### Step 1: Verify Data Loading
```bash
# Check correct files are loaded
python src/evaluate_phase2.py --market NQ
# Should see: Loading NQ_D1M.csv AND NQ_D1S.csv
```

### Step 2: Quick Test with Existing Model
```bash
# Run evaluation with detailed logging
python src/evaluate_phase2.py --market NQ --verbose
# Check: Episode length, action distribution, termination reason
```

### Step 3: Full Phase 1 Training
```bash
# Train Phase 1 with full timesteps
python src/train_phase1.py --market NQ --timesteps 2000000
# Expected: ~10-20 hours training time
# Validation: Check final mean reward > 0
```

### Step 4: Full Phase 2 Training
```bash
# Train Phase 2 on Phase 1 checkpoint
python src/train_phase2.py --market NQ --timesteps 1000000
# Expected: ~5-10 hours training time
# Validation: Check PM actions are used, reward improves
```

### Step 5: Production Evaluation
```bash
# Evaluate fully trained model
python src/evaluate_phase2.py --market NQ
# Expected:
#   - Episode completes (close to 5000 steps)
#   - Multiple action types used
#   - PM actions used for position management
#   - Positive or near-zero returns
#   - Sharpe ratio > -1
```

### Success Criteria
- [ ] Episode runs for >1000 steps (at least 20% completion)
- [ ] At least 4 different action types used
- [ ] PM actions used >5% of the time
- [ ] No data mismatch errors in logs
- [ ] Sharpe ratio > -1.0
- [ ] Return % between -10% and +20% (reasonable range)

---

## Code Locations for Investigation

### Evaluation Code
- `src/evaluate_phase2.py` - Main evaluation script, likely contains data mismatch bug
- `src/environment_phase2.py` - Environment step() and reset() methods, episode termination logic

### Training Code
- `src/train_phase1.py` - Phase 1 training, check timestep configuration
- `src/train_phase2.py` - Phase 2 training, check curriculum learning setup
- `src/kl_callback.py` - Training callbacks, check for test mode flags

### Data Processing
- `src/data_validator.py` - Corruption detection (working correctly)
- `src/update_training_data.py` - Minute data processing (fixed)
- `src/process_second_data.py` - Second-level data processing (working correctly)

### Environment
- `src/environment_phase1.py` - Phase 1 environment logic
- `src/environment_phase2.py` - Phase 2 environment logic, action masking, episode termination
- `src/feature_engineering.py` - Observation generation, check for NaN sources
- `src/technical_indicators.py` - Indicator calculations, check for division by zero

---

## Conclusion

The evaluation system is experiencing multiple critical issues that prevent normal operation:

1. **Data mismatch bug** (wrong second-level data loaded) - highest priority
2. **Undertrained model** (test mode training insufficient) - must retrain
3. **Early episode termination** (unknown root cause) - needs investigation
4. **No position management usage** (may be caused by above issues)

The most likely cascade of issues:
1. Model undertrained → learns basic hold/buy only
2. Data mismatch → causes unexpected state observations
3. Unexpected observations → triggers episode termination
4. Episode terminates early → no chance to use PM actions

**Recommended approach:**
1. Fix data mismatch bug first (highest confidence, easiest fix)
2. Add detailed logging for episode termination
3. Run quick test to see if data fix resolves early termination
4. If termination persists, investigate further with new logs
5. Once evaluation runs properly, retrain models with full timesteps
6. Validate with production evaluation

All issues are solvable with targeted fixes. The system architecture is sound; these are configuration and training issues, not fundamental design problems.

---

## Implementation Update (November 2025)

- Training scripts now write metadata alongside models/VecNormalize dumps, recording market, timestep budget, and test-mode status. Evaluation refuses to run when metadata signals a market mismatch or test-mode artifact.
- Phase 2 evaluation validates that minute and second-level datasets share the same market symbol and comparable coverage before starting an episode.
- Trading environments emit structured diagnostics (done reasons, action masks, observation quality), enabling evaluation logs to pinpoint early termination causes without manual log spelunking.
