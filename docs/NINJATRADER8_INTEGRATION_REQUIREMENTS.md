# NinjaTrader 8 Integration Requirements for Phase 2 AI Model

**Document Version:** 1.0
**Created:** November 6, 2025
**Target:** NinjaTrader 8 Strategy "AiBridge" (or similar)
**Model:** Phase 2 Position Management Model (`phase2_position_mgmt_final.zip`)

---

## Executive Summary

This document specifies the **exact requirements** for a NinjaTrader 8 strategy to receive and execute orders from your trained Phase 2 AI model. The strategy must handle **6 distinct actions** and maintain **position management state** to properly execute the model's trading decisions.

---

## Phase 2 Model Action Space

Your trained model outputs **6 actions** (discrete integers 0-5):

| Action ID | Action Name | Description | NinjaTrader Requirement |
|-----------|-------------|-------------|------------------------|
| **0** | `HOLD` | No action, maintain current position | Do nothing |
| **1** | `BUY` | Open LONG position | Execute Market BUY order |
| **2** | `SELL` | Open SHORT position | Execute Market SELL SHORT order |
| **3** | `MOVE_TO_BE` | Move Stop Loss to Break-Even | Modify existing Stop Loss to entry price |
| **4** | `ENABLE_TRAIL` | Activate Trailing Stop | Enable trailing stop mechanism |
| **5** | `DISABLE_TRAIL` | Deactivate Trailing Stop | Disable trailing stop, revert to fixed SL |

---

## Detailed Action Specifications

### Action 0: HOLD
**Model Output:** `0`
**NinjaTrader Requirement:**
- No order execution
- Maintain current position (if any)
- Continue monitoring market data
- Send data back to model for next decision

**Implementation:**
```csharp
case 0: // HOLD
    // No action required
    // Continue monitoring
    break;
```

---

### Action 1: BUY (Open Long)
**Model Output:** `1`
**NinjaTrader Requirement:**

**Pre-Conditions:**
- ✅ No current position (flat)
- ✅ Within Regular Trading Hours (RTH): 9:30 AM - 4:59 PM ET
- ✅ Sufficient margin available

**Execution Sequence:**
1. **Enter Market Order:**
   - Direction: LONG (BUY)
   - Quantity: 1 contract (for NQ)
   - Order Type: Market
   - Slippage: +0.25 points (1 tick)

2. **Set Initial Stop Loss (SL):**
   - Distance: `1.5 × ATR` below entry price
   - Type: Stop Market
   - Quantity: 1 contract

3. **Set Initial Take Profit (TP):**
   - Distance: `4.5 × ATR` above entry price (3:1 R:R ratio)
   - Type: Limit
   - Quantity: 1 contract

4. **Record Entry State:**
   - Entry Price
   - Entry Time
   - Initial SL Price
   - Initial TP Price
   - Position Direction (LONG)

**Implementation:**
```csharp
case 1: // BUY
    if (Position.MarketPosition == MarketPosition.Flat && IsRegularTradingHours())
    {
        double entryPrice = Close[0] + (1 * TickSize); // Add slippage
        double atr = ATR(14)[0];

        // Calculate SL and TP
        double slDistance = atr * 1.5;
        double tpDistance = slDistance * 3.0;

        stopLossPrice = entryPrice - slDistance;
        takeProfitPrice = entryPrice + tpDistance;

        // Enter Market
        EnterLong(1, "AI_LONG");

        // Set protective orders (OCO)
        ExitLongStopMarket(1, true, 1, stopLossPrice, "SL", "AI_LONG");
        ExitLongLimit(1, true, 1, takeProfitPrice, "TP", "AI_LONG");

        // Record state
        entryPriceRecorded = entryPrice;
        positionDirection = 1; // LONG
        trailingStopActive = false;
        beMovesCount = 0;
    }
    break;
```

---

### Action 2: SELL (Open Short)
**Model Output:** `2`
**NinjaTrader Requirement:**

**Pre-Conditions:**
- ✅ No current position (flat)
- ✅ Within Regular Trading Hours (RTH): 9:30 AM - 4:59 PM ET
- ✅ Sufficient margin available

**Execution Sequence:**
1. **Enter Market Order:**
   - Direction: SHORT (SELL)
   - Quantity: 1 contract
   - Order Type: Market
   - Slippage: -0.25 points (1 tick)

2. **Set Initial Stop Loss (SL):**
   - Distance: `1.5 × ATR` above entry price
   - Type: Stop Market
   - Quantity: 1 contract

3. **Set Initial Take Profit (TP):**
   - Distance: `4.5 × ATR` below entry price (3:1 R:R ratio)
   - Type: Limit
   - Quantity: 1 contract

4. **Record Entry State:**
   - Entry Price
   - Entry Time
   - Initial SL Price
   - Initial TP Price
   - Position Direction (SHORT)

**Implementation:**
```csharp
case 2: // SELL SHORT
    if (Position.MarketPosition == MarketPosition.Flat && IsRegularTradingHours())
    {
        double entryPrice = Close[0] - (1 * TickSize); // Subtract slippage
        double atr = ATR(14)[0];

        // Calculate SL and TP
        double slDistance = atr * 1.5;
        double tpDistance = slDistance * 3.0;

        stopLossPrice = entryPrice + slDistance;
        takeProfitPrice = entryPrice - tpDistance;

        // Enter Market
        EnterShort(1, "AI_SHORT");

        // Set protective orders (OCO)
        ExitShortStopMarket(1, true, 1, stopLossPrice, "SL", "AI_SHORT");
        ExitShortLimit(1, true, 1, takeProfitPrice, "TP", "AI_SHORT");

        // Record state
        entryPriceRecorded = entryPrice;
        positionDirection = -1; // SHORT
        trailingStopActive = false;
        beMovesCount = 0;
    }
    break;
```

---

### Action 3: MOVE_TO_BE (Move Stop Loss to Break-Even)
**Model Output:** `3`
**NinjaTrader Requirement:**

**Pre-Conditions:**
- ✅ Position is OPEN (long or short)
- ✅ Position is PROFITABLE (unrealized PnL > 0)
- ✅ Stop Loss is NOT already at break-even

**Execution Sequence:**
1. **Calculate Unrealized P&L:**
   - Long: `(Current Price - Entry Price) × $20`
   - Short: `(Entry Price - Current Price) × $20`

2. **Validate Move:**
   - If unrealized PnL ≤ 0 → REJECT (not profitable)
   - If SL already at or past entry → REJECT (already protected)

3. **Update Stop Loss:**
   - **Long:** Set SL to entry price (or entry + 1 tick for safety)
   - **Short:** Set SL to entry price (or entry - 1 tick for safety)

4. **Increment BE Move Counter:**
   - Track how many times BE was used for this trade

**Implementation:**
```csharp
case 3: // MOVE_TO_BE
    if (Position.MarketPosition != MarketPosition.Flat)
    {
        double currentPrice = Close[0];
        double unrealizedPnl = 0;

        if (positionDirection == 1) // LONG
        {
            unrealizedPnl = (currentPrice - entryPriceRecorded) * 20;

            if (unrealizedPnl > 0 && stopLossPrice < entryPriceRecorded)
            {
                // Move SL to break-even (plus 1 tick for safety)
                stopLossPrice = entryPriceRecorded + TickSize;
                ExitLongStopMarket(1, true, 1, stopLossPrice, "SL_BE", "AI_LONG");
                beMovesCount++;

                Print("✓ Stop Loss moved to Break-Even: " + stopLossPrice);
            }
        }
        else if (positionDirection == -1) // SHORT
        {
            unrealizedPnl = (entryPriceRecorded - currentPrice) * 20;

            if (unrealizedPnl > 0 && stopLossPrice > entryPriceRecorded)
            {
                // Move SL to break-even (minus 1 tick for safety)
                stopLossPrice = entryPriceRecorded - TickSize;
                ExitShortStopMarket(1, true, 1, stopLossPrice, "SL_BE", "AI_SHORT");
                beMovesCount++;

                Print("✓ Stop Loss moved to Break-Even: " + stopLossPrice);
            }
        }
    }
    break;
```

---

### Action 4: ENABLE_TRAIL (Activate Trailing Stop)
**Model Output:** `4`
**NinjaTrader Requirement:**

**Pre-Conditions:**
- ✅ Position is OPEN (long or short)
- ✅ Position is PROFITABLE (≥ 1R profit, where R = initial risk)
- ✅ Trailing stop is NOT already active

**Execution Sequence:**
1. **Validate Activation:**
   - Calculate unrealized profit in R multiples
   - Minimum profit: 1R (1 × initial SL distance)
   - If profit < 1R → REJECT

2. **Enable Trailing Mechanism:**
   - Set flag: `trailingStopActive = true`
   - Calculate initial trail distance: `1.0 × ATR`
   - Record highest profit point

3. **Update Stop Loss to Trailing:**
   - **Long:** `SL = Current Price - (1 × ATR)`
   - **Short:** `SL = Current Price + (1 × ATR)`

4. **Monitor on Each Bar:**
   - If price moves in profit direction → Update SL
   - If price moves against → SL stays (locked in profit)

**Implementation:**
```csharp
case 4: // ENABLE_TRAIL
    if (Position.MarketPosition != MarketPosition.Flat && !trailingStopActive)
    {
        double currentPrice = Close[0];
        double atr = ATR(14)[0];
        double unrealizedPnl = 0;

        if (positionDirection == 1) // LONG
        {
            unrealizedPnl = (currentPrice - entryPriceRecorded) * 20;
        }
        else if (positionDirection == -1) // SHORT
        {
            unrealizedPnl = (entryPriceRecorded - currentPrice) * 20;
        }

        double initialRisk = Math.Abs(entryPriceRecorded - stopLossPrice) * 20;

        // Check if profit >= 1R
        if (unrealizedPnl >= initialRisk)
        {
            trailingStopActive = true;
            highestProfitPoint = unrealizedPnl;

            // Set initial trailing stop
            double trailDistance = atr * 1.0;

            if (positionDirection == 1) // LONG
            {
                stopLossPrice = currentPrice - trailDistance;
                ExitLongStopMarket(1, true, 1, stopLossPrice, "SL_TRAIL", "AI_LONG");
            }
            else if (positionDirection == -1) // SHORT
            {
                stopLossPrice = currentPrice + trailDistance;
                ExitShortStopMarket(1, true, 1, stopLossPrice, "SL_TRAIL", "AI_SHORT");
            }

            Print("✓ Trailing Stop ENABLED at: " + stopLossPrice);
        }
    }
    break;
```

---

### Action 5: DISABLE_TRAIL (Deactivate Trailing Stop)
**Model Output:** `5`
**NinjaTrader Requirement:**

**Pre-Conditions:**
- ✅ Position is OPEN (long or short)
- ✅ Trailing stop IS currently active

**Execution Sequence:**
1. **Validate Deactivation:**
   - Check if trailing is active
   - If not active → REJECT (already disabled)

2. **Disable Trailing Mechanism:**
   - Set flag: `trailingStopActive = false`
   - Lock current SL price (no more updates)

3. **Keep Current Stop Loss:**
   - Do NOT move SL back to original
   - Maintain profit protection at current level
   - Stop will remain FIXED at last trailing position

**Implementation:**
```csharp
case 5: // DISABLE_TRAIL
    if (Position.MarketPosition != MarketPosition.Flat && trailingStopActive)
    {
        trailingStopActive = false;

        // SL remains at current level (locked in profit)
        // No need to modify stop order - just stop updating it

        Print("✓ Trailing Stop DISABLED. SL locked at: " + stopLossPrice);
    }
    break;
```

---

## Trailing Stop Update Logic (OnBarUpdate)

When `trailingStopActive == true`, the strategy must update the stop loss on **each new bar**:

```csharp
protected override void OnBarUpdate()
{
    // ... other logic ...

    // Update trailing stop if active
    if (trailingStopActive && Position.MarketPosition != MarketPosition.Flat)
    {
        double currentPrice = Close[0];
        double atr = ATR(14)[0];
        double trailDistance = atr * 1.0;
        double unrealizedPnl = 0;

        if (positionDirection == 1) // LONG
        {
            unrealizedPnl = (currentPrice - entryPriceRecorded) * 20;

            // Update SL only if profit increased
            if (unrealizedPnl > highestProfitPoint)
            {
                highestProfitPoint = unrealizedPnl;
                double newSL = currentPrice - trailDistance;

                // Only move SL up (never down)
                if (newSL > stopLossPrice)
                {
                    stopLossPrice = newSL;
                    ExitLongStopMarket(1, true, 1, stopLossPrice, "SL_TRAIL", "AI_LONG");
                    Print("↑ Trailing SL updated: " + stopLossPrice);
                }
            }
        }
        else if (positionDirection == -1) // SHORT
        {
            unrealizedPnl = (entryPriceRecorded - currentPrice) * 20;

            // Update SL only if profit increased
            if (unrealizedPnl > highestProfitPoint)
            {
                highestProfitPoint = unrealizedPnl;
                double newSL = currentPrice + trailDistance;

                // Only move SL down (never up)
                if (newSL < stopLossPrice)
                {
                    stopLossPrice = newSL;
                    ExitShortStopMarket(1, true, 1, stopLossPrice, "SL_TRAIL", "AI_SHORT");
                    Print("↓ Trailing SL updated: " + stopLossPrice);
                }
            }
        }
    }
}
```

---

## Required State Variables

Your NinjaTrader 8 strategy **MUST** maintain the following state variables:

```csharp
// Position tracking
private double entryPriceRecorded = 0;
private int positionDirection = 0;  // 0=flat, 1=long, -1=short
private double stopLossPrice = 0;
private double takeProfitPrice = 0;

// Position management state
private bool trailingStopActive = false;
private double highestProfitPoint = 0;
private int beMovesCount = 0;

// Timing
private DateTime entryTime;
private int entryBarIndex = 0;

// Performance tracking
private int totalTrades = 0;
private int winningTrades = 0;
private int losingTrades = 0;
private double totalPnL = 0;

// Model communication
private int lastActionReceived = -1;
private DateTime lastActionTimestamp;
```

---

## Market Specifications (NQ)

Your model was trained on **E-mini Nasdaq-100 (NQ)** with these specifications:

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Symbol** | NQ | E-mini Nasdaq-100 |
| **Contract Multiplier** | $20 per point | 1 point move = $20 |
| **Tick Size** | 0.25 points | Minimum price increment |
| **Tick Value** | $5.00 | 0.25 × $20 = $5 |
| **Commission** | $2.50 per side | Round-trip = $5.00 |
| **Position Size** | 1 contract | Fixed for Apex compliance |
| **RTH Hours** | 9:30 AM - 5:00 PM ET | Regular Trading Hours |
| **Initial Margin** | ~$18,000 | Check with broker |
| **Trailing DD Limit** | $2,500 | Apex Trader Funding rule |

---

## Data Feed Requirements

The model expects **real-time minute bars** with these features:

### Required OHLCV Data:
- ✅ **Open** price
- ✅ **High** price
- ✅ **Low** price
- ✅ **Close** price
- ✅ **Volume**

### Required Technical Indicators:
The model uses these indicators - NinjaTrader must calculate them:

1. **ATR (14)** - Average True Range
2. **SMA (20, 50, 200)** - Simple Moving Averages
3. **EMA (9, 21)** - Exponential Moving Averages
4. **RSI (14)** - Relative Strength Index
5. **MACD (12, 26, 9)** - Moving Average Convergence Divergence
6. **Bollinger Bands (20, 2)** - Price bands
7. **ADX (14)** - Average Directional Index
8. **VWAP** - Volume Weighted Average Price

### Time Synchronization:
- Timestamp format: `YYYY-MM-DD HH:MM:SS` (ET timezone)
- Bar close time (not open time)
- Must match model training data format

---

## Communication Protocol: AI Model ↔ NinjaTrader

### 1. Data Flow: NinjaTrader → Model

On each new bar close, send this JSON to the model:

```json
{
  "timestamp": "2025-11-06 10:30:00",
  "symbol": "NQ",
  "open": 21450.25,
  "high": 21455.50,
  "low": 21448.00,
  "close": 21452.75,
  "volume": 1234,
  "indicators": {
    "atr_14": 45.5,
    "sma_20": 21445.0,
    "sma_50": 21440.0,
    "sma_200": 21350.0,
    "ema_9": 21450.0,
    "ema_21": 21442.0,
    "rsi_14": 52.3,
    "macd": 12.5,
    "macd_signal": 10.2,
    "macd_hist": 2.3,
    "bb_upper": 21480.0,
    "bb_middle": 21450.0,
    "bb_lower": 21420.0,
    "adx_14": 25.6,
    "vwap": 21451.5
  },
  "position": {
    "direction": 0,
    "entry_price": 0,
    "unrealized_pnl": 0,
    "sl_price": 0,
    "tp_price": 0
  }
}
```

### 2. Data Flow: Model → NinjaTrader

Model returns action as JSON:

```json
{
  "action": 1,
  "action_name": "BUY",
  "confidence": 0.87,
  "timestamp": "2025-11-06 10:30:00"
}
```

Where `action` is one of: `0, 1, 2, 3, 4, 5`

---

## Action Validation Rules

The strategy **MUST** validate actions before execution:

### Validation Table:

| Action | Current Position | Valid? | Reason |
|--------|-----------------|--------|--------|
| **0 (HOLD)** | Any | ✅ Always | No-op |
| **1 (BUY)** | Flat | ✅ Yes | Can open long |
| **1 (BUY)** | Long/Short | ❌ No | Position already open |
| **2 (SELL)** | Flat | ✅ Yes | Can open short |
| **2 (SELL)** | Long/Short | ❌ No | Position already open |
| **3 (MOVE_TO_BE)** | Flat | ❌ No | No position to manage |
| **3 (MOVE_TO_BE)** | Long/Short (profitable) | ✅ Yes | Can protect profit |
| **3 (MOVE_TO_BE)** | Long/Short (losing) | ❌ No | Not profitable |
| **4 (ENABLE_TRAIL)** | Flat | ❌ No | No position to trail |
| **4 (ENABLE_TRAIL)** | Long/Short (≥1R profit) | ✅ Yes | Can enable trail |
| **4 (ENABLE_TRAIL)** | Long/Short (<1R profit) | ❌ No | Not enough profit |
| **5 (DISABLE_TRAIL)** | Flat | ❌ No | No position |
| **5 (DISABLE_TRAIL)** | Long/Short (trail active) | ✅ Yes | Can disable |
| **5 (DISABLE_TRAIL)** | Long/Short (trail off) | ❌ No | Already disabled |

---

## Testing Checklist for "AiBridge" Strategy

Use this checklist to verify your NinjaTrader 8 strategy implements all requirements:

### ✅ Basic Order Execution
- [ ] Can receive action integers (0-5) from external source
- [ ] Action 1 (BUY) opens LONG position with Market order
- [ ] Action 2 (SELL) opens SHORT position with Market order
- [ ] SL and TP orders are placed automatically on entry
- [ ] Position size is exactly 1 contract
- [ ] Slippage is applied (+/- 1 tick)
- [ ] Commission is tracked ($2.50 per side)

### ✅ Position Management Actions
- [ ] Action 3 (MOVE_TO_BE) moves SL to entry price
- [ ] MOVE_TO_BE only works when position is profitable
- [ ] Action 4 (ENABLE_TRAIL) activates trailing stop
- [ ] ENABLE_TRAIL only works when profit ≥ 1R
- [ ] Action 5 (DISABLE_TRAIL) deactivates trailing stop
- [ ] DISABLE_TRAIL locks SL at current level (doesn't reset)

### ✅ Trailing Stop Mechanism
- [ ] Trailing stop updates on each bar when active
- [ ] Trail distance is 1 × ATR behind price
- [ ] SL only moves in profit direction (never against)
- [ ] Highest profit point is tracked correctly
- [ ] Trailing stops when disabled (doesn't auto-resume)

### ✅ State Management
- [ ] Entry price is recorded on position open
- [ ] Position direction is tracked (0/1/-1)
- [ ] Current SL and TP prices are tracked
- [ ] Trailing stop active flag is maintained
- [ ] BE move count is incremented correctly
- [ ] State resets when position closes

### ✅ Risk Management
- [ ] Only trades during RTH (9:30 AM - 5:00 PM ET)
- [ ] Only one position open at a time
- [ ] Entry actions rejected when position exists
- [ ] PM actions rejected when no position exists
- [ ] Invalid actions are logged (not executed silently)

### ✅ Data & Indicators
- [ ] ATR(14) is calculated correctly
- [ ] All required indicators are available
- [ ] Data is timestamped in ET timezone
- [ ] Minute bars are closed bars (not forming bars)
- [ ] Historical data matches training data format

### ✅ Communication
- [ ] Strategy can send market data to model
- [ ] Strategy can receive actions from model
- [ ] JSON format is correct (see protocol above)
- [ ] Errors are logged and handled gracefully
- [ ] Model disconnection doesn't crash strategy

### ✅ Logging & Monitoring
- [ ] All actions received are logged with timestamp
- [ ] Entry/exit executions are logged
- [ ] SL/TP modifications are logged
- [ ] Invalid actions are logged with reason
- [ ] Trade P&L is calculated and logged

### ✅ Apex Compliance
- [ ] Maximum 1 contract position
- [ ] Trailing drawdown is monitored
- [ ] Account balance tracking
- [ ] Daily loss limits enforced (if applicable)
- [ ] Profit target tracking

---

## Example: Complete Action Sequence

Here's what a typical trading sequence looks like:

```
Step 1: Model sends action 1 (BUY)
  → NinjaTrader: Enter Long at 21450.25
  → NinjaTrader: Set SL at 21382.00 (68.25 points = 1.5×ATR)
  → NinjaTrader: Set TP at 21654.75 (204.50 points = 4.5×ATR)
  → State: position=1, entry=21450.25, trailing=false

Step 2: Model sends action 0 (HOLD) [bars 2-10]
  → NinjaTrader: No action, monitor position
  → Price moves to 21480 (+29.75 points = +$595 unrealized)

Step 3: Model sends action 3 (MOVE_TO_BE)
  → NinjaTrader: Check profit (✓ positive)
  → NinjaTrader: Move SL from 21382.00 to 21450.50 (entry + 1 tick)
  → State: SL now at break-even, be_moves=1

Step 4: Model sends action 0 (HOLD) [bars 11-15]
  → Price continues to 21520 (+69.75 points = +$1,395 unrealized)

Step 5: Model sends action 4 (ENABLE_TRAIL)
  → NinjaTrader: Check profit ≥ 1R (✓ yes, 69.75 > 68.25)
  → NinjaTrader: Enable trailing, set SL to 21474.5 (price - ATR)
  → State: trailing=true, highest_profit=1395

Step 6: Price moves to 21535 [OnBarUpdate triggered]
  → NinjaTrader: New high profit ($1,695)
  → NinjaTrader: Update SL to 21489.5 (new price - ATR)
  → State: SL trails upward

Step 7: Price retraces to 21510
  → NinjaTrader: Profit decreased, SL stays at 21489.5
  → State: SL locked (doesn't move down)

Step 8: Model sends action 5 (DISABLE_TRAIL)
  → NinjaTrader: Trailing disabled
  → NinjaTrader: SL remains locked at 21489.5
  → State: trailing=false, SL fixed

Step 9: Price hits TP at 21654.75
  → NinjaTrader: Exit at TP
  → Realized P&L: +$4,090 (204.5 points × $20)
  → Trade logged, position closed
  → State reset: position=0, trailing=false, be_moves=0
```

---

## Error Handling Requirements

The strategy must handle these error scenarios:

### Invalid Actions:
```csharp
// Example: Receive action 1 (BUY) when already in position
if (action == 1 && Position.MarketPosition != MarketPosition.Flat)
{
    Print("❌ ERROR: BUY action rejected - position already open");
    SendErrorToModel("INVALID_ACTION", "Cannot BUY when position exists");
    return; // Don't execute
}
```

### Connection Loss:
```csharp
// If model connection lost
if (TimeSinceLastAction() > TimeSpan.FromMinutes(5))
{
    Print("⚠️ WARNING: No action received for 5 minutes");
    // Option 1: Close position at market
    // Option 2: Continue with last known stops
    // Option 3: Alert trader
}
```

### Market Closed:
```csharp
// Reject entries outside RTH
if (!IsRegularTradingHours() && (action == 1 || action == 2))
{
    Print("❌ ERROR: Entry rejected - outside RTH hours");
    SendErrorToModel("MARKET_CLOSED", "No entries allowed outside 9:30-16:59 ET");
    return;
}
```

---

## Performance Monitoring

The strategy should track and report these metrics:

```csharp
// Real-time metrics to send back to model
public class PerformanceMetrics
{
    public int TotalTrades { get; set; }
    public int WinningTrades { get; set; }
    public int LosingTrades { get; set; }
    public double TotalPnL { get; set; }
    public double WinRate { get; set; }
    public double AveragePnL { get; set; }
    public double MaxDrawdown { get; set; }
    public double CurrentDrawdown { get; set; }
    public int ConsecutiveWins { get; set; }
    public int ConsecutiveLosses { get; set; }
    public DateTime LastTradeTime { get; set; }

    // Position Management stats
    public int TotalBEMoves { get; set; }
    public int TotalTrailEnables { get; set; }
    public int TradesWithPM { get; set; }
    public double AvgPnLWithPM { get; set; }
    public double AvgPnLWithoutPM { get; set; }
}
```

---

## Summary: Minimum Requirements Checklist

For the "AiBridge" strategy to work with your Phase 2 model:

### ✅ MUST HAVE:
1. Handle 6 actions (0-5)
2. Execute Market BUY/SELL orders
3. Set SL and TP on entry
4. Implement Move-to-BE logic
5. Implement Enable/Disable Trail logic
6. Update trailing stops on each bar
7. Track position state accurately
8. Validate actions before execution
9. Calculate ATR(14) for SL/TP
10. Operate only during RTH

### ✅ SHOULD HAVE:
11. JSON communication protocol
12. Error handling and logging
13. Performance metrics tracking
14. Connection loss recovery
15. Invalid action rejection with logs

### ✅ NICE TO HAVE:
16. Real-time dashboard
17. Alert notifications
18. Trade journal export
19. Backtesting replay mode
20. Model confidence threshold filtering

---

## Testing the Integration

### Phase 1: Replay Mode Testing
1. Load historical 1-minute NQ data
2. Run model in "replay" mode
3. Send each action to strategy
4. Verify strategy executes correctly
5. Compare P&L to model's evaluation

### Phase 2: Paper Trading
1. Connect to live market data feed
2. Run model in real-time
3. Execute on paper/sim account
4. Monitor for 1 week minimum
5. Verify no execution errors

### Phase 3: Live Micro Trading
1. Switch to MNQ (micro contract)
2. Run for 2 weeks on live account
3. Validate all actions work correctly
4. Check Apex compliance rules
5. Verify consistent with backtests

### Phase 4: Production (NQ)
1. Deploy to full NQ contract
2. Start with conservative sizing
3. Monitor daily performance
4. Track PM action usage (should be ~13%)
5. Ensure Sharpe ratio matches evaluation

---

## Questions to Ask When Validating "AiBridge"

When checking if the strategy is complete:

1. **Does it handle all 6 actions?**
   - Test each action individually
   - Check action 0 does nothing
   - Check actions 1-2 open positions
   - Check actions 3-5 modify positions

2. **Does it maintain state correctly?**
   - Entry price stored?
   - Position direction tracked?
   - Trailing flag persists?
   - BE count increments?

3. **Does it validate properly?**
   - Rejects BUY when already long?
   - Rejects MOVE_TO_BE when losing?
   - Rejects ENABLE_TRAIL when profit < 1R?
   - Logs invalid actions?

4. **Does it calculate correctly?**
   - ATR used for SL/TP?
   - Trailing distance = 1 × ATR?
   - Unrealized PnL accurate?
   - Commission included?

5. **Does it communicate properly?**
   - Sends market data to model?
   - Receives actions from model?
   - JSON format correct?
   - Timestamps synchronized?

---

## Contact & Support

For questions about model integration, refer to:
- Model documentation: `/docs/IMPLEMENTATION_SUMMARY.md`
- Evaluation results: `/results/phase2_metrics.json`
- Training logs: `/logs/training_phase2_production.log`

**Model Version:** Phase 2 Position Management (5M timesteps)
**Training Date:** November 6, 2025
**Market:** E-mini Nasdaq-100 (NQ)
**Expected Performance:** Sharpe 21.71, +11% returns, 13.5% PM usage

---

**END OF DOCUMENT**
