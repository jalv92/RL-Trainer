# AI Validation Prompt: NinjaTrader 8 "AiBridge" Strategy Compliance Check

**Purpose:** Use this prompt to ask another AI (like Claude or ChatGPT) to verify that your NinjaTrader 8 strategy implements all required functionality for the Phase 2 trading model.

---

## Prompt to Send to AI

```
I have a NinjaTrader 8 C# strategy called "AiBridge" that needs to receive trading actions from an AI model and execute them.

Please analyze the attached strategy code and verify it meets ALL of the following requirements:

---

## REQUIRED ACTIONS (Must Handle All 6)

The strategy must correctly handle these 6 actions from the AI model:

### Action 0: HOLD
- ✅ No action taken
- ✅ Maintains current position if any
- ✅ Does not generate errors

### Action 1: BUY (Open Long)
Requirements:
- ✅ Only executes when position is FLAT (no current position)
- ✅ Only executes during Regular Trading Hours (9:30 AM - 4:59 PM ET)
- ✅ Enters Market Buy order for 1 contract
- ✅ Adds +0.25 points (1 tick) slippage
- ✅ Sets Stop Loss at: Entry Price - (1.5 × ATR)
- ✅ Sets Take Profit at: Entry Price + (4.5 × ATR)
- ✅ Records entry price, SL price, TP price
- ✅ Sets position direction to LONG (1)
- ✅ Initializes trailing stop flag to FALSE
- ✅ Resets BE move counter to 0

### Action 2: SELL (Open Short)
Requirements:
- ✅ Only executes when position is FLAT (no current position)
- ✅ Only executes during Regular Trading Hours (9:30 AM - 4:59 PM ET)
- ✅ Enters Market Sell Short order for 1 contract
- ✅ Subtracts -0.25 points (1 tick) slippage
- ✅ Sets Stop Loss at: Entry Price + (1.5 × ATR)
- ✅ Sets Take Profit at: Entry Price - (4.5 × ATR)
- ✅ Records entry price, SL price, TP price
- ✅ Sets position direction to SHORT (-1)
- ✅ Initializes trailing stop flag to FALSE
- ✅ Resets BE move counter to 0

### Action 3: MOVE_TO_BE (Move Stop Loss to Break-Even)
Requirements:
- ✅ Only executes when position is OPEN (long or short)
- ✅ Only executes when position is PROFITABLE (unrealized PnL > 0)
- ✅ Rejects if already at break-even
- ✅ For LONG: Moves SL to entry price (or entry + 1 tick)
- ✅ For SHORT: Moves SL to entry price (or entry - 1 tick)
- ✅ Increments BE move counter
- ✅ Logs successful move

### Action 4: ENABLE_TRAIL (Activate Trailing Stop)
Requirements:
- ✅ Only executes when position is OPEN (long or short)
- ✅ Only executes when profit ≥ 1R (1 × initial risk)
- ✅ Rejects if trailing is already active
- ✅ Sets trailing stop active flag to TRUE
- ✅ Records highest profit point
- ✅ For LONG: Sets SL to Current Price - (1.0 × ATR)
- ✅ For SHORT: Sets SL to Current Price + (1.0 × ATR)
- ✅ Logs activation

### Action 5: DISABLE_TRAIL (Deactivate Trailing Stop)
Requirements:
- ✅ Only executes when position is OPEN (long or short)
- ✅ Only executes when trailing is currently ACTIVE
- ✅ Sets trailing stop active flag to FALSE
- ✅ KEEPS current SL level (does NOT reset to original)
- ✅ Stops updating SL on future bars
- ✅ Logs deactivation

---

## TRAILING STOP UPDATE LOGIC

When trailing stop is ACTIVE, the strategy must:
- ✅ Update stop loss on EVERY new bar (OnBarUpdate)
- ✅ Only move SL in profit direction (NEVER against)
- ✅ For LONG: SL can only move UP (max function)
- ✅ For SHORT: SL can only move DOWN (min function)
- ✅ Trail distance is exactly 1.0 × ATR
- ✅ Track highest profit point
- ✅ Only update SL when new profit high is reached
- ✅ Do NOT update when trailing is disabled

---

## STATE MANAGEMENT

The strategy must maintain these state variables:

Position State:
- ✅ entryPriceRecorded (double)
- ✅ positionDirection (int: 0=flat, 1=long, -1=short)
- ✅ stopLossPrice (double)
- ✅ takeProfitPrice (double)

Position Management State:
- ✅ trailingStopActive (bool)
- ✅ highestProfitPoint (double)
- ✅ beMovesCount (int)

Timing State:
- ✅ entryTime (DateTime)
- ✅ entryBarIndex (int)

All state must RESET when position closes.

---

## VALIDATION RULES

The strategy must validate and REJECT invalid actions:

❌ REJECT Action 1 (BUY) if:
- Position is already open (long or short)
- Outside Regular Trading Hours

❌ REJECT Action 2 (SELL) if:
- Position is already open (long or short)
- Outside Regular Trading Hours

❌ REJECT Action 3 (MOVE_TO_BE) if:
- No position is open (flat)
- Position is in LOSS (unrealized PnL ≤ 0)
- SL already at or past break-even

❌ REJECT Action 4 (ENABLE_TRAIL) if:
- No position is open (flat)
- Position profit < 1R (initial risk)
- Trailing already active

❌ REJECT Action 5 (DISABLE_TRAIL) if:
- No position is open (flat)
- Trailing already disabled

All rejections must be LOGGED with reason.

---

## TECHNICAL REQUIREMENTS

Market Specifications:
- ✅ Symbol: NQ (E-mini Nasdaq-100)
- ✅ Contract Multiplier: $20 per point
- ✅ Tick Size: 0.25 points
- ✅ Tick Value: $5.00
- ✅ Commission: $2.50 per side
- ✅ Position Size: Exactly 1 contract

Indicators:
- ✅ ATR(14) must be calculated correctly
- ✅ Used for SL/TP calculations
- ✅ Used for trailing distance

Timing:
- ✅ Regular Trading Hours: 9:30 AM - 4:59 PM ET
- ✅ No entries outside RTH
- ✅ Exits allowed any time

---

## ERROR HANDLING

The strategy must:
- ✅ Log all actions received with timestamp
- ✅ Log all order executions
- ✅ Log all invalid actions with reason
- ✅ Log all SL/TP modifications
- ✅ Handle model disconnection gracefully
- ✅ Not crash on invalid input

---

## ANALYSIS INSTRUCTIONS

Please analyze the strategy code and provide:

1. **COMPLIANCE REPORT:**
   - List each of the 6 actions and whether it's correctly implemented ✅ or ❌
   - For each action, check ALL requirements listed above
   - Note any missing functionality

2. **STATE MANAGEMENT REVIEW:**
   - Verify all required state variables exist
   - Check they are initialized properly
   - Verify they reset when position closes

3. **VALIDATION LOGIC REVIEW:**
   - Check if invalid actions are rejected
   - Verify rejection conditions match requirements
   - Confirm error logging exists

4. **TRAILING STOP REVIEW:**
   - Verify trailing updates on each bar
   - Check SL only moves in profit direction
   - Confirm trail distance = 1.0 × ATR
   - Validate highest profit tracking

5. **CRITICAL ISSUES:**
   - List any bugs or logic errors
   - Identify missing functionality
   - Flag any violations of requirements

6. **RECOMMENDATIONS:**
   - Suggest fixes for issues found
   - Propose improvements
   - Highlight risk areas

7. **OVERALL VERDICT:**
   - Ready for testing? YES/NO
   - Confidence level: 1-10
   - Required fixes before deployment

---

## OUTPUT FORMAT

Please structure your response as:

```
# AiBridge Strategy Compliance Report

## Executive Summary
[PASS/FAIL] - Overall verdict
[X/10] - Confidence score

## Action Implementation Status
Action 0 (HOLD): ✅/❌ [notes]
Action 1 (BUY): ✅/❌ [notes]
Action 2 (SELL): ✅/❌ [notes]
Action 3 (MOVE_TO_BE): ✅/❌ [notes]
Action 4 (ENABLE_TRAIL): ✅/❌ [notes]
Action 5 (DISABLE_TRAIL): ✅/❌ [notes]

## Detailed Findings
[Component-by-component analysis]

## Critical Issues
[List of blocking issues]

## Recommendations
[Prioritized list of fixes]

## Test Scenarios
[Suggested test cases to validate fixes]

## Final Verdict
[Ready/Not Ready + reasoning]
```

---

## STRATEGY CODE TO ANALYZE

[Paste your NinjaTrader 8 AiBridge strategy C# code here]

---

## REFERENCE DOCUMENTATION

The AI model this strategy interfaces with:
- Trained on E-mini Nasdaq-100 (NQ)
- 5 million timesteps of training
- Phase 2: Position Management model
- Expected performance: Sharpe 21.71, +11% returns
- Uses 6-action discrete action space (0-5)
- Position management usage: ~13.5% of actions

The model outputs actions as integers 0-5 based on real-time market data and expects the strategy to execute them immediately.

---

END OF VALIDATION PROMPT
```

---

## How to Use This Prompt

### Step 1: Copy the Entire Prompt Above
Copy everything between the triple backticks (```) starting with "I have a NinjaTrader 8 C# strategy..."

### Step 2: Paste Your Strategy Code
Add your AiBridge strategy C# code after the section "[Paste your NinjaTrader 8 AiBridge strategy C# code here]"

### Step 3: Send to AI
Paste the complete prompt (with your code) into:
- Claude (Anthropic)
- ChatGPT (OpenAI)
- Any other code-analysis AI

### Step 4: Review the Response
The AI will provide a detailed compliance report showing:
- ✅ What's correctly implemented
- ❌ What's missing or wrong
- 🔧 How to fix issues
- ⚠️ Potential risks

### Step 5: Iterate
Fix the issues identified, then re-run the validation until you get a "PASS" verdict with confidence 8/10 or higher.

---

## Example Validation Conversation

**You:**
```
[Paste full prompt with your strategy code]
```

**AI Response:**
```
# AiBridge Strategy Compliance Report

## Executive Summary
FAIL - Strategy has critical missing functionality
Confidence: 3/10

## Action Implementation Status
Action 0 (HOLD): ✅ Correctly implemented
Action 1 (BUY): ⚠️ Partially implemented - missing RTH check
Action 2 (SELL): ❌ NOT implemented - no handler found
Action 3 (MOVE_TO_BE): ❌ NOT implemented
Action 4 (ENABLE_TRAIL): ❌ NOT implemented
Action 5 (DISABLE_TRAIL): ❌ NOT implemented

## Critical Issues
1. Actions 2-5 have no implementation
2. Action 1 missing RTH validation
3. No state management variables found
...

## Recommendations
1. Add handlers for actions 2-5 (CRITICAL)
2. Add RTH check: TimeSpan.FromHours(9.5) to TimeSpan.FromHours(16.983)
...
```

**You:** (After fixes)
```
[Paste updated strategy code]
```

**AI Response:**
```
# AiBridge Strategy Compliance Report

## Executive Summary
PASS - All core functionality implemented
Confidence: 9/10

## Action Implementation Status
Action 0 (HOLD): ✅ Correctly implemented
Action 1 (BUY): ✅ Correctly implemented
Action 2 (SELL): ✅ Correctly implemented
Action 3 (MOVE_TO_BE): ✅ Correctly implemented
Action 4 (ENABLE_TRAIL): ✅ Correctly implemented
Action 5 (DISABLE_TRAIL): ✅ Correctly implemented

Ready for paper trading!
```

---

## Quick Validation Checklist

Use this as a pre-check before sending to AI:

### Before Sending Code to AI:
- [ ] Strategy compiles without errors
- [ ] All 6 action cases exist in code (search for "case 0:" through "case 5:")
- [ ] State variables are declared (entryPriceRecorded, trailingStopActive, etc.)
- [ ] ATR(14) indicator is added
- [ ] OnBarUpdate() method exists
- [ ] Order execution methods are used (EnterLong, EnterShort, etc.)

### After AI Validation:
- [ ] AI reports PASS with confidence ≥ 8/10
- [ ] All 6 actions marked as ✅
- [ ] No CRITICAL issues reported
- [ ] Recommended fixes are minor/cosmetic
- [ ] Ready for backtesting/paper trading

---

## Alternative: Step-by-Step Validation

If you prefer to validate piece by piece, ask the AI these questions separately:

1. **"Does this strategy correctly implement Action 1 (BUY) with all requirements?"**
   - Paste just the BUY action handler code
   - Get focused feedback

2. **"Does this strategy correctly implement Action 3 (MOVE_TO_BE)?"**
   - Paste just the MOVE_TO_BE code
   - Verify logic

3. **"Does this trailing stop update logic work correctly?"**
   - Paste OnBarUpdate() method
   - Check trailing implementation

4. **"Are these state variables sufficient for the requirements?"**
   - Paste state variable declarations
   - Verify completeness

This approach helps you fix issues incrementally.

---

## Common Issues AI Will Find

Based on typical NinjaTrader strategy implementations:

### Issue 1: Missing Action Handlers
```csharp
// ❌ WRONG: Only handles HOLD, BUY
switch(action) {
    case 0: break;
    case 1: EnterLong(); break;
    // Missing cases 2-5!
}

// ✅ CORRECT: Handles all 6 actions
switch(action) {
    case 0: break;
    case 1: /* BUY logic */ break;
    case 2: /* SELL logic */ break;
    case 3: /* MOVE_TO_BE logic */ break;
    case 4: /* ENABLE_TRAIL logic */ break;
    case 5: /* DISABLE_TRAIL logic */ break;
}
```

### Issue 2: No Validation
```csharp
// ❌ WRONG: No validation
case 1:
    EnterLong();
    break;

// ✅ CORRECT: Validates before entry
case 1:
    if (Position.MarketPosition == MarketPosition.Flat && IsRTH()) {
        EnterLong();
    } else {
        Print("BUY rejected: " + reason);
    }
    break;
```

### Issue 3: Trailing Stop Never Updates
```csharp
// ❌ WRONG: Trailing set once, never updated
case 4:
    trailingStopActive = true;
    SetStopLoss(currentPrice - ATR[0]);
    // No OnBarUpdate logic!

// ✅ CORRECT: Updates on each bar
protected override void OnBarUpdate() {
    if (trailingStopActive && Position.MarketPosition != MarketPosition.Flat) {
        // Update SL logic here
    }
}
```

### Issue 4: State Doesn't Reset
```csharp
// ❌ WRONG: State persists across trades
// No reset logic

// ✅ CORRECT: Reset on position close
protected override void OnPositionUpdate(...) {
    if (Position.MarketPosition == MarketPosition.Flat) {
        entryPriceRecorded = 0;
        trailingStopActive = false;
        beMovesCount = 0;
    }
}
```

---

## Expected AI Analysis Quality

A good validation should provide:

### Detailed Code Review:
- Line-by-line analysis of each action handler
- Logic flow verification
- Edge case identification

### Test Scenarios:
- "What happens if action 1 received twice?"
- "What if action 3 sent when position is losing?"
- "What if trailing enabled when profit < 1R?"

### Risk Assessment:
- Potential for double entries
- SL modification race conditions
- State corruption scenarios

### Performance Impact:
- Execution speed considerations
- Resource usage
- Scalability concerns

---

**END OF VALIDATION GUIDE**
