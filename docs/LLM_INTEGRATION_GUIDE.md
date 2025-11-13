# LLM Integration Guide

## Overview

This guide provides detailed instructions for integrating and customizing the LLM components in the Phase 3 Hybrid Trading Agent.

## Prerequisites

### Hardware Requirements

**Minimum**:
- GPU: 6GB VRAM (GTX 1060 / RTX 2060)
- CPU: 4 cores
- RAM: 16GB
- Storage: 10GB free

**Recommended**:
- GPU: 8GB+ VRAM (RTX 3060 / RTX 4060)
- CPU: 8+ cores
- RAM: 32GB
- Storage: 20GB free

### Software Requirements

```bash
# Core dependencies
pip install torch>=2.1.0
cuda-version  # Match your GPU (11.8 or 12.1)

# LLM dependencies
pip install transformers>=4.35.0
pip install accelerate>=0.24.0
pip install bitsandbytes>=0.41.0
pip install sentencepiece>=0.1.99

# Optional but recommended
pip install tensorrt  # For inference optimization
```

### GPU Setup

**NVIDIA GPU**:
```bash
# Install CUDA toolkit (if not already installed)
# Download from: https://developer.nvidia.com/cuda-downloads

# Verify installation
nvidia-smi
nvcc --version
```

**AMD GPU**:
```bash
# Install ROCm (Linux only)
# Follow: https://rocmdocs.amd.com/

# PyTorch with ROCm support
pip install torch --index-url https://download.pytorch.org/whl/rocm5.6
```

## Quick Start

### 1. Install Dependencies

```bash
cd RL-Trainner-Executor-System
pip install -r requirements.txt

# Install LLM-specific packages
pip install transformers torch accelerate bitsandbytes sentencepiece
```

### 2. Verify Installation

```bash
python -c "
import torch
from transformers import AutoModelForCausalLM
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"
```

### 3. Test LLM Integration

```bash
python tests/test_llm_integration.py
```

Expected output:
```
LLM INTEGRATION TEST SUITE
================================
✓ Phase 3 observation shape correct (261D)
✓ LLM features builder working correctly
✓ LLM prompt generation working
✓ LLM response parsing working
✓ Decision fusion working correctly
✓ Risk veto mechanism working
✓ LLM context generation working
✓ Selective querying working

All tests passed! ✓
```

### 4. Run Training

```bash
# Test mode (reduced timesteps, mock LLM)
python main.py
# Select: Training → Phase 3 Hybrid (Test Mode)

# Production mode (full training)
python main.py
# Select: Training → Phase 3 Hybrid (Production)
```

## Configuration

### LLM Configuration File (`config/llm_config.yaml`)

```yaml
# LLM Model Selection
llm_model:
  name: "microsoft/Phi-3-mini-4k-instruct"  # 3.8B parameters
  quantization: "int8"  # Options: int8, int4, none
  device: "auto"  # auto, cuda, cpu
  max_new_tokens: 50  # Response length
  temperature: 0.1  # 0.1-0.3 recommended for trading
  top_p: 0.9  # Nucleus sampling

# Decision Fusion
fusion:
  llm_weight: 0.3  # 0.0 = RL only, 1.0 = LLM only
  confidence_threshold: 0.7  # Minimum confidence to override
  use_selective_querying: true  # Reduce LLM calls
  query_interval: 5  # Query every N bars

# Risk Management
risk:
  max_consecutive_losses: 3  # Block entries after N losses
  min_win_rate_threshold: 0.4  # Block if win rate < N%
  dd_buffer_threshold: 0.2  # Block near DD limit
  enable_risk_veto: true

# Prompt Templates
prompts:
  system: |
    You are a professional futures trader analyzing {market}.
    Provide concise trading advice based on:
    - Technical indicators
    - Market context
    - Risk metrics
    - Recent performance
    
    Available actions: HOLD, BUY, SELL, MOVE_TO_BE, ENABLE_TRAIL, DISABLE_TRAIL
    Format: ACTION | confidence (0-1) | brief_reason
    
  context_template: |
    Market: {market_name} | Time: {current_time} ET | Price: ${current_price:.2f}
    
    Trend: ADX={adx:.1f} RSI={rsi:.1f} VWAP={vwap_distance:+.2%}
    Position: {position_status} | Unrealized P&L: ${unrealized_pnl:+.0f}
    Performance: Win Rate={win_rate:.1%} | Consecutive Losses={consecutive_losses}
    Account: Balance=${balance:.0f}
    
    Recommendation?
```

### Model Selection Guide

| Model | Size | VRAM (INT8) | Speed | Quality | Use Case |
|-------|------|-------------|-------|---------|----------|
| Phi-3-mini | 3.8B | 4GB | Fast | Good | Default, balanced |
| Phi-3-small | 7B | 7GB | Medium | Better | Higher quality needed |
| Llama-2-7B | 7B | 7GB | Medium | Better | Alternative to Phi-3 |
| Mistral-7B | 7B | 7GB | Medium | Better | Good reasoning |

**Recommendation**: Start with Phi-3-mini for testing, upgrade if needed.

## Customization

### Custom Prompt Engineering

#### Adding Market-Specific Context

```yaml
prompts:
  context_template: |
    Market: {market_name} | Contract: {contract_specs}
    Session: {session_time} | Volatility: {vol_regime:.1f}
    
    Technical Analysis:
    - Trend: ADX={adx:.1f} (Strength: {trend_strength})
    - Momentum: RSI={rsi:.1f} MACD={macd:.2f}
    - Support/Resistance: {support:.0f} / {resistance:.0f}
    
    Position Management:
    - Current Position: {position_status}
    - Entry Price: ${entry_price:.2f}
    - Current Price: ${current_price:.2f}
    - Unrealized P&L: ${unrealized_pnl:+.0f}
    
    Risk Metrics:
    - Account Balance: ${balance:.0f}
    - Peak Balance: ${peak_balance:.0f}
    - Drawdown: {drawdown:.1%}
    - Consecutive Losses: {consecutive_losses}
    - Recent Win Rate: {win_rate:.1%}
    
    Based on this analysis, what action do you recommend?
```

#### Adding Trading Rules

```yaml
prompts:
  system: |
    You are a professional futures trader with strict risk management rules:
    
    TRADING RULES:
    1. MAX 3 consecutive losses before stopping
    2. MIN 40% win rate to continue trading
    3. NEVER risk more than 2% per trade
    4. Always use stop losses
    5. Take profits at 2:1 risk/reward minimum
    
    MARKET ANALYSIS:
    - Focus on trend following in strong trends (ADX>25)
    - Fade extremes in ranging markets (ADX<20)
    - Confirm with volume and momentum
    
    RESPONSE FORMAT:
    ACTION | confidence | reason (max 10 words)
    
    ACTIONS: HOLD, BUY, SELL, MOVE_TO_BE, ENABLE_TRAIL, DISABLE_TRAIL
```

### Custom Feature Engineering

#### Adding New LLM Features

Edit `src/llm_features.py`:

```python
def _add_custom_features(self, obs: np.ndarray, data: pd.DataFrame, idx: int):
    """Add your custom features here."""
    
    # Example: Add correlation with correlated market
    if 'es_close' in data.columns:
        # NQ/ES correlation feature
        nq_return = (data['close'].iloc[idx] / data['close'].iloc[idx-20]) - 1
        es_return = (data['es_close'].iloc[idx] / data['es_close'].iloc[idx-20]) - 1
        obs[261] = nq_return - es_return  # Divergence
    
    # Example: Economic event proximity
    if 'fed_minutes_days' in data.columns:
        obs[262] = max(0, 5 - data['fed_minutes_days'].iloc[idx]) / 5
    
    # Example: Options expiration effect
    if 'days_to_opex' in data.columns:
        obs[263] = 1.0 if data['days_to_opex'].iloc[idx] < 2 else 0.0
```

Update feature indices:

```python
self.feature_indices = {
    # ... existing features ...
    'market_divergence': 261,
    'fed_event_proximity': 262,
    'opex_effect': 263,
}
```

### Custom Fusion Logic

#### Implementing New Fusion Strategy

Edit `src/hybrid_agent.py`:

```python
def _fuse_decisions(self, rl_action, rl_conf, llm_action, llm_conf, 
                   action_mask, position_state):
    """Custom fusion logic."""
    
    # Your custom fusion strategy here
    # Example: Market regime-aware fusion
    
    market_regime = position_state.get('market_regime', 0)
    
    if market_regime == 1:  # Strong trend
        # Trust RL more in strong trends
        rl_weight = 0.8
        llm_weight = 0.2
    elif market_regime == -1:  # Ranging
        # Trust LLM more in ranging markets
        rl_weight = 0.3
        llm_weight = 0.7
    else:  # Neutral
        # Balanced approach
        rl_weight = 0.7
        llm_weight = 0.3
    
    # Weight by confidence
    rl_score = rl_conf * rl_weight
    llm_score = llm_conf * llm_weight
    
    if llm_score > rl_score:
        return llm_action, {'fusion_method': 'custom_llm_weighted'}
    else:
        return rl_action, {'fusion_method': 'custom_rl_weighted'}
```

### Custom Risk Management

#### Adding New Risk Rules

Edit `src/hybrid_agent.py`:

```python
def _apply_risk_veto(self, action, position_state, market_context):
    """Apply custom risk-based veto."""
    
    # Standard vetoes
    action, vetoed = super()._apply_risk_veto(action, position_state, market_context)
    if vetoed:
        return action, True
    
    # Custom risk rules
    
    # Rule 1: Veto if VIX is high (market stress)
    vix_level = market_context.get('vix_level', 20)
    if vix_level > 30 and action in [1, 2]:  # BUY or SELL
        self.logger.info(f"[RISK_VETO] High VIX ({vix_level}), blocking entry")
        return 0, True  # Force HOLD
    
    # Rule 2: Veto if near major support/resistance
    price = market_context.get('current_price', 0)
    support = market_context.get('support_level', 0)
    resistance = market_context.get('resistance_level', 999999)
    
    if support > 0 and abs(price - support) / price < 0.002:
        self.logger.info(f"[RISK_VETO] Near support ({support}), blocking short")
        if action == 2:  # SELL
            return 0, True
    
    if resistance < 999999 and abs(price - resistance) / price < 0.002:
        self.logger.info(f"[RISK_VETO] Near resistance ({resistance}), blocking long")
        if action == 1:  # BUY
            return 0, True
    
    # Rule 3: Time-based restrictions
    current_time = market_context.get('current_time', '10:00')
    hour = int(current_time.split(':')[0])
    
    # Avoid trading during lunch (12:00-13:00 ET)
    if 12 <= hour < 13 and action in [1, 2]:
        self.logger.info(f"[RISK_VETO] Lunch time, blocking entry")
        return 0, True
    
    return action, False
```

## Performance Optimization

### Reducing LLM Latency

#### 1. Enable Selective Querying

```yaml
fusion:
  use_selective_querying: true
  query_interval: 10  # Query every 10 bars instead of 5
```

#### 2. Use INT4 Quantization

```yaml
llm_model:
  quantization: "int4"  # Reduces VRAM to ~2GB
```

#### 3. Cache Optimization

```yaml
fusion:
  cache_decay_rate: 0.5  # More aggressive decay, more frequent refresh
```

#### 4. Batch Inference

For multiple simultaneous queries:

```python
def batch_query(self, observations, position_states, market_contexts):
    """Process multiple queries in batch."""
    prompts = [
        self._build_prompt(obs, state, context)
        for obs, state, context in zip(observations, position_states, market_contexts)
    ]
    
    # Tokenize all prompts together
    inputs = self.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(self.device)
    
    # Generate all responses
    with torch.no_grad():
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config['max_new_tokens'],
            temperature=self.config['temperature'],
            top_p=self.config['top_p'],
            do_sample=True
        )
    
    # Parse all responses
    responses = []
    for i, output in enumerate(outputs):
        response = self.tokenizer.decode(output, skip_special_tokens=True)
        response = response[len(prompts[i]):].strip()
        responses.append(response)
    
    return [self._parse_response(r) for r in responses]
```

### Memory Optimization

#### Reducing GPU Memory Usage

```python
# Load with CPU offloading
self.model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
    max_memory={0: "3GiB", "cpu": "16GiB"}  # Limit GPU memory
)

# Clear cache periodically
torch.cuda.empty_cache()
```

#### Gradient Checkpointing (for fine-tuning)

```python
self.model.gradient_checkpointing_enable()
```

## Troubleshooting

### Common Issues

#### 1. "CUDA out of memory"

**Solutions**:
- Use INT4 quantization instead of INT8
- Reduce batch size in training
- Close other GPU applications
- Use CPU offloading

```yaml
llm_model:
  quantization: "int4"  # Use smaller quantization
  
# Or in code
self.model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # Instead of load_in_8bit
    device_map="auto"
)
```

#### 2. "Model not found" or "Connection error"

**Solutions**:
- Check internet connection
- Use local model cache
- Download model manually

```bash
# Download model manually
huggingface-cli download microsoft/Phi-3-mini-4k-instruct

# Or in code
from huggingface_hub import snapshot_download

model_path = snapshot_download(
    "microsoft/Phi-3-mini-4k-instruct",
    cache_dir="./models/llm_cache"
)
```

#### 3. Slow inference

**Solutions**:
- Enable selective querying
- Use faster model (Phi-3-mini vs larger models)
- Optimize with TensorRT
- Reduce max_new_tokens

```yaml
llm_model:
  max_new_tokens: 30  # Reduce from 50
  temperature: 0.1  # Lower temperature for faster generation
```

#### 4. Poor LLM responses

**Solutions**:
- Improve prompt engineering
- Add more context
- Adjust temperature/top_p
- Try different model

```yaml
llm_model:
  temperature: 0.1  # Lower for more focused responses
  top_p: 0.95  # Higher for more variety if needed
```

### Debug Mode

Enable verbose logging:

```yaml
logging:
  log_llm_queries: true
  log_decision_fusion: true
  log_risk_veto: true
  log_latency: true

development:
  verbose_logging: true
```

Check logs:
```bash
tail -f logs/llm_queries.log
tail -f logs/decision_fusion.log
```

## Advanced Topics

### Fine-tuning LLM for Trading

#### Prepare Training Data

```python
training_data = [
    {
        "prompt": """Market: NQ | Time: 10:30 | ADX: 32 | RSI: 65
Position: FLAT | Win Rate: 55% | Consecutive Losses: 0
Recommendation?""",
        "completion": "BUY | 0.85 | Strong uptrend with momentum"
    },
    # ... more examples
]
```

#### Fine-tune with LoRA

```python
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./llm_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=1e-4,
    fp16=True,
    save_steps=100,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

### Using Different LLM Models

#### OpenAI GPT Models

```python
import openai

class OpenAILLM:
    def __init__(self, api_key):
        openai.api_key = api_key
    
    def query(self, observation, position_state, market_context):
        prompt = self._build_prompt(observation, position_state, market_context)
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.config['prompts']['system']},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        return self._parse_response(response.choices[0].message.content)
```

#### Anthropic Claude

```python
import anthropic

class ClaudeLLM:
    def __init__(self, api_key):
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def query(self, observation, position_state, market_context):
        prompt = self._build_prompt(observation, position_state, market_context)
        
        response = self.client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=50,
            temperature=0.1,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return self._parse_response(response.content[0].text)
```

### Ensemble LLM Approach

```python
class EnsembleLLM:
    def __init__(self, llm_models, weights=None):
        self.llm_models = llm_models
        self.weights = weights or [1.0/len(llm_models)] * len(llm_models)
    
    def query(self, observation, position_state, market_context):
        # Query all LLMs
        responses = []
        for llm in self.llm_models:
            try:
                action, confidence, reasoning = llm.query(
                    observation, position_state, market_context
                )
                responses.append((action, confidence, reasoning))
            except Exception as e:
                print(f"LLM {llm} failed: {e}")
        
        if not responses:
            return 0, 0.0, "All LLMs failed"
        
        # Weighted voting
        action_scores = {}
        for (action, conf, _), weight in zip(responses, self.weights):
            action_scores[action] = action_scores.get(action, 0) + conf * weight
        
        # Choose action with highest weighted score
        final_action = max(action_scores.items(), key=lambda x: x[1])[0]
        avg_confidence = np.mean([r[1] for r in responses])
        
        # Combine reasoning
        all_reasons = " | ".join([r[2] for r in responses])
        
        return final_action, avg_confidence, f"Ensemble: {all_reasons}"
```

## Best Practices

### 1. Start Simple
- Begin with Phi-3-mini and default settings
- Test thoroughly before customizing
- Gradually add complexity

### 2. Monitor Performance
- Track LLM query latency
- Monitor agreement rates
- Analyze risk veto frequency

### 3. Iterate on Prompts
- Test different prompt formats
- A/B test system prompts
- Include relevant context only

### 4. Risk Management First
- Set conservative risk parameters
- Test risk vetoes thoroughly
- Monitor drawdown carefully

### 5. Keep Records
- Log all LLM queries and responses
- Track decision fusion statistics
- Document configuration changes

## Support and Resources

### Documentation
- [Hybrid Architecture](./HYBRID_ARCHITECTURE.md)
- [API Reference](./API_REFERENCE.md)
- [Configuration Guide](./CONFIGURATION.md)

### Community
- GitHub Issues: [Report bugs](https://github.com/yourrepo/issues)
- Discussions: [Ask questions](https://github.com/yourrepo/discussions)
- Discord: [Chat with developers](https://discord.gg/yourserver)

### Model Resources
- [Phi-3 Models](https://huggingface.co/microsoft)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

**Remember**: LLM integration adds significant complexity. Always test thoroughly in simulation before live trading, and start with small position sizes when deploying to production.