# Hybrid RL + LLM Trading Agent - Architecture Documentation

## Overview

The Phase 3 Hybrid Trading Agent combines the strengths of **Reinforcement Learning (RL)** and **Large Language Models (LLM)** to create a more robust and context-aware trading system.

### Key Components

1. **RL Agent (PPO)**: Fast pattern recognition and execution (5M training timesteps)
2. **LLM Advisor (Phi-3-mini)**: Context-aware reasoning about market conditions  
3. **Decision Fusion**: Intelligent combination of both recommendations
4. **Risk Management**: Automatic veto of high-risk actions

### Architecture Benefits

- **Context Awareness**: 261D observations vs 228D in Phase 2
- **Explicit Reasoning**: LLM provides human-readable analysis
- **Risk-Aware Decisions**: Considers account state and recent performance
- **Improved Robustness**: Two independent systems reduce single points of failure

## System Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Phase 3 Hybrid Agent                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐         ┌──────────────────┐              │
│  │   RL Agent   │         │    LLM Advisor   │              │
│  │   (PPO)      │         │ (Phi-3-mini)     │              │
│  └──────┬───────┘         └────────┬─────────┘              │
│         │                          │                         │
│         │ 228D features            │ 261D features           │
│         │ Action+Value             │ Action+Confidence       │
│         ↓                          ↓                         │
│  ┌──────────────────────────────────────────┐              │
│  │         Decision Fusion Module          │              │
│  │  • Agreement detection                   │              │
│  │  • Confidence weighting                  │              │
│  │  • Risk veto application               │              │
│  └───────────────┬─────────────────────────┘              │
│                  │                                          │
│                  ↓                                          │
│  ┌──────────────────────────────────────────┐              │
│  │          Final Trading Action           │              │
│  │          (0-5: HOLD/BUY/SELL/etc)      │              │
│  └──────────────────────────────────────────┘              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Observation Building**: Environment creates 261D observation (228 base + 33 LLM features)
2. **Parallel Processing**: RL and LLM process observation independently
3. **Decision Fusion**: Hybrid agent combines recommendations with risk management
4. **Action Execution**: Final action sent to trading environment
5. **Feedback Loop**: Results tracked for both RL training and LLM context

## Component Details

### 1. Enhanced Observation Space (261D)

The Phase 3 environment extends the 228D Phase 2 observation space with 33 additional features:

#### Extended Market Context (10 features)
- **ADX Slope**: Trend strength momentum
- **VWAP Distance**: Price relative to volume-weighted average
- **Volatility Regime**: Current volatility environment
- **Volume Regime**: Relative volume analysis
- **Price Momentum (20/60min)**: Multi-timeframe momentum
- **Efficiency Ratio**: Price movement efficiency
- **Spread Ratio**: Current spread relative to price
- **Session Trend**: Intraday session trend
- **Market Regime**: Overall market condition

#### Multi-Timeframe Indicators (8 features)
- **SMA-50/200 Slopes**: Long-term trend detection
- **RSI (15/60min)**: Multi-timeframe momentum
- **Volume Ratios (5/20min)**: Volume analysis across timeframes
- **Price Change (60min)**: Hourly price change
- **Support/Resistance Distance**: Distance to key levels

#### Pattern Recognition (10 features)
- **Higher/Lower Highs/Lows**: Trend pattern detection
- **Double Top/Bottom**: Reversal pattern detection
- **Support/Resistance Levels**: Key price levels
- **Breakout/Breakdown Signals**: Trend continuation signals

#### Risk Context (5 features)
- **Unrealized P&L**: Current position P&L
- **Current Drawdown**: Drawdown from peak
- **Consecutive Losses**: Recent losing streak
- **Recent Win Rate**: Performance over last 10 trades
- **MAE/MFE Ratio**: Maximum adverse vs favorable excursion

### 2. LLM Reasoning Module

#### Model Configuration
- **Model**: `microsoft/Phi-3-mini-4k-instruct` (3.8B parameters)
- **Quantization**: INT8 (4GB VRAM) or INT4 (2GB VRAM)
- **Inference Time**: ~15-20ms per query on RTX 3060
- **Context Window**: 4,096 tokens

#### Prompt Engineering

**System Prompt**:
```
You are a professional futures trader analyzing NQ (Nasdaq-100 E-mini).
Provide concise trading advice based on market context and risk management.

Available actions: HOLD, BUY, SELL, MOVE_TO_BE, ENABLE_TRAIL, DISABLE_TRAIL
Respond format: ACTION | confidence (0-1) | brief_reason (max 10 words)

Guidelines:
- Consider trend strength, momentum, and support/resistance
- Factor in recent performance and risk metrics  
- Be decisive but cautious with low confidence
- Prioritize capital preservation on losing streaks
```

**Context Template**:
```
Market: {market_name}
Time: {current_time} ET
Price: ${current_price:.2f}

Trend Analysis:
- ADX: {adx:.1f} (trend strength)
- Price vs VWAP: {vwap_distance:+.2%}
- RSI: {rsi:.1f}
- Momentum: {momentum:.1f}

Position: {position_status}
Unrealized P&L: ${unrealized_pnl:+.0f}

Recent Performance:
- Win Rate: {win_rate:.1%}
- Consecutive Losses: {consecutive_losses}
- Account Balance: ${balance:.0f}

What is your recommended action?
```

#### Response Parsing

LLM responses are parsed using the format:
```
ACTION | confidence | reason
```

**Examples**:
```
BUY | 0.85 | Strong uptrend with volume confirmation
SELL | 0.92 | Bearish divergence on RSI
HOLD | 0.60 | Wait for clearer signal
MOVE_TO_BE | 0.75 | Protect profits at breakeven
ENABLE_TRAIL | 0.80 | Strong momentum, trail profits
```

### 3. Decision Fusion Strategies

#### Fusion Methods

1. **Agreement (Priority 1)**
   - When RL and LLM agree → Take the action
   - Highest confidence in decision
   - Statistics tracked for analysis

2. **High Confidence Override (Priority 2)**
   - RL confidence > 0.9 + LLM confidence < 0.6 → Follow RL
   - LLM confidence > 0.9 + RL confidence < 0.6 → Follow LLM
   - Allows expert override when one system is very confident

3. **Weighted Decision (Priority 3)**
   - Both systems moderately confident
   - Weight by confidence scores
   - LLM weight from config (default 0.3)

4. **Uncertainty Hold (Priority 4)**
   - Both systems uncertain (< 0.5 confidence)
   - Default to HOLD for safety
   - Prevents low-confidence trades

#### Risk Veto System

**Veto Conditions**:

1. **Consecutive Losses**
   - Block new entries after 3+ consecutive losses
   - Protects against losing streaks
   - Allows exits and risk management

2. **Drawdown Proximity**
   - Block aggressive actions near trailing DD limit
   - DD buffer < 20% triggers veto
   - Preserves capital near limits

3. **Low Win Rate**
   - Block entries if win rate < 40%
   - Indicates poor recent performance
   - Forces system to "reset"

**Veto Actions**:
- Only allow: HOLD, MOVE_TO_BE, DISABLE_TRAIL
- Block: BUY, SELL, ENABLE_TRAIL (new entries)

### 4. Selective Querying

To reduce latency, LLM is only queried when:

1. **RL Uncertainty**: RL confidence < threshold (0.7)
2. **Entry Decision**: Position flat (critical decision point)
3. **Time Interval**: Every N bars (configurable, default 5)
4. **State Change**: Significant change in position/balance/win rate

**Caching**:
- LLM responses cached for 5 minutes
- Confidence decays by 20% per reuse
- Reduces redundant queries

## Implementation Details

### File Structure

```
src/
├── llm_features.py          # Feature calculation (33 new features)
├── environment_phase3_llm.py # Phase 3 environment (261D)
├── llm_reasoning.py         # LLM inference engine
├── hybrid_agent.py          # Decision fusion logic
├── llm_callback.py          # Training monitoring
├── train_phase3_llm.py      # Training pipeline
└── evaluate_phase3_llm.py   # Evaluation script

config/
└── llm_config.yaml          # LLM and fusion configuration

tests/
└── test_llm_integration.py  # Comprehensive test suite
```

### Configuration (llm_config.yaml)

```yaml
# LLM Model Selection
llm_model:
  name: "microsoft/Phi-3-mini-4k-instruct"
  quantization: "int8"  # int8, int4, or none
  device: "auto"
  max_new_tokens: 50
  temperature: 0.1
  top_p: 0.9

# Decision Fusion Parameters
fusion:
  llm_weight: 0.3  # Trust in LLM (0.0 = RL only, 1.0 = LLM only)
  confidence_threshold: 0.7  # Minimum confidence to override
  use_selective_querying: true
  query_interval: 5

# Risk Management
risk:
  max_consecutive_losses: 3
  min_win_rate_threshold: 0.4
  dd_buffer_threshold: 0.2
  enable_risk_veto: true
```

### Performance Characteristics

#### Resource Requirements
- **GPU**: RTX 3060 or better (8GB+ VRAM recommended)
- **CPU**: 4+ cores for parallel environments
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 10GB for model and data

#### Training Time
- **Phase 3 Test**: ~30 minutes (50K timesteps)
- **Phase 3 Production**: ~12-16 hours (5M timesteps)
- **Inference**: ~15-20ms per LLM query

#### Model Size
- **Phi-3-mini INT8**: ~4GB VRAM
- **Phi-3-mini INT4**: ~2GB VRAM
- **PPO Policy**: ~50MB
- **VecNormalize**: ~1MB

## Integration with Existing System

### Backward Compatibility

Phase 3 maintains full backward compatibility:

```python
# Can run as Phase 2 (228D)
env = TradingEnvironmentPhase3LLM(data, use_llm_features=False)

# Or as Phase 3 (261D)  
env = TradingEnvironmentPhase3LLM(data, use_llm_features=True)
```

### Training Pipeline

```bash
# Phase 1: Entry learning
python src/train_phase1.py --market NQ

# Phase 2: Position management  
python src/train_phase2.py --market NQ

# Phase 3: Hybrid LLM agent
python src/train_phase3_llm.py --market NQ
```

### Evaluation

```bash
# Evaluate hybrid agent
python src/evaluate_phase3_llm.py \
    --model models/phase3_hybrid_final \
    --market NQ \
    --episodes 20
```

## Monitoring and Debugging

### TensorBoard Integration

The LLM monitoring callback logs to TensorBoard:

```bash
tensorboard --logdir ./tensorboard_logs/phase3_hybrid
```

**Tracked Metrics**:
- `llm/total_queries` - Total LLM queries
- `llm/avg_latency_ms` - Average inference latency
- `llm/error_rate` - LLM error percentage
- `fusion/agreement_pct` - RL-LLM agreement rate
- `fusion/risk_veto_pct` - Risk veto frequency
- `confidence/rl_avg` - Average RL confidence
- `confidence/llm_avg` - Average LLM confidence

### Logging

**LLM queries** are logged with:
- Prompt sent to LLM
- Response received
- Parsed action/confidence/reasoning
- Latency and any errors

**Decision fusion** logs:
- RL and LLM recommendations
- Confidence scores
- Fusion method used
- Risk veto decisions

### Debugging Tips

1. **High LLM Error Rate**
   - Check GPU memory (need 4GB+ free)
   - Verify model files downloaded correctly
   - Try mock mode to isolate issues

2. **Low Agreement Rate**
   - Check feature normalization
   - Verify LLM prompt includes all context
   - Adjust confidence thresholds

3. **Slow Inference**
   - Enable selective querying
   - Reduce query interval
   - Use INT4 quantization

4. **Risk Veto Too Aggressive**
   - Adjust thresholds in config
   - Check consecutive losses calculation
   - Verify drawdown tracking

## Future Enhancements

### Potential Improvements

1. **Multi-LLM Ensemble**
   - Use multiple LLMs with different strengths
   - Weight by historical accuracy
   - Reduce single-model bias

2. **Adaptive Fusion**
   - Learn fusion weights during training
   - Adjust based on market regime
   - Dynamic confidence thresholds

3. **Advanced Prompting**
   - Few-shot examples in prompts
   - Chain-of-thought reasoning
   - Multi-step analysis

4. **Performance Optimization**
   - Model quantization (INT4)
   - TensorRT inference
   - Batch processing

5. **Expanded Context**
   - News sentiment analysis
   - Economic calendar events
   - Correlated market analysis

## Conclusion

The Phase 3 Hybrid RL + LLM Trading Agent represents a significant advancement in algorithmic trading by combining:

- **RL's speed and pattern recognition** with **LLM's reasoning and context awareness**
- **Systematic execution** with **adaptive decision making**
- **Quantitative analysis** with **qualitative insights**

This hybrid approach addresses many limitations of pure RL systems while maintaining the benefits of systematic trading. The modular design allows for easy experimentation with different LLMs, fusion strategies, and risk management techniques.

The system's ability to provide **human-readable reasoning** for each decision also enables:
- Better debugging and analysis
- Regulatory compliance documentation
- Human oversight and intervention
- Performance attribution

Overall, the hybrid architecture provides a robust foundation for building increasingly sophisticated trading systems that can adapt to changing market conditions while maintaining strict risk controls.