# AI - TRAINER - AI Trading System

![RL TRAINER Main Menu](img/Screenshot_105.png)

A comprehensive reinforcement learning trading system with an interactive command-line interface for managing the complete ML pipeline from data processing to model evaluation.

## Overview

This project implements a **three-phase curriculum learning approach** for training trading agents that comply with Apex Trader Funding rules. The system uses PPO (Proximal Policy Optimization) algorithm with transfer learning and optional LLM integration to develop robust trading strategies.

## Features

- **Interactive CLI Menu System** - Easy-to-use interface for all operations
- **Three-Phase Training Pipeline** - Progressive curriculum learning
  - Phase 1: Entry signal quality learning (fixed SL/TP)
  - Phase 2: Position management with dynamic SL/TP and action masking
  - Phase 3: Hybrid RL + LLM agent with context-aware reasoning
- **Continue Training from Checkpoint** - Resume training from any saved model
- **Smart Model Detection** - Automatically finds and loads the newest models
- **Automatic Data Corruption Detection** - Detects and fixes divide-by-100 errors
- **Multi-Market Support** - 8 major futures instruments with automatic contract specs
- **Action Masking** - Prevents invalid actions in Phase 2 & 3
- **Comprehensive Testing Framework** - Hardware-maximized and pipeline testing modes
- **Thread Pool Management** - Stable training on constrained systems
- **Model Evaluation** - Comprehensive performance metrics
- **Compliance Enforcement** - 100% Apex Trader Funding rules compliance
- **Progress Tracking** - Colored output and detailed logging

## Supported Instruments

![Data Processing Menu](img/Screenshot_106.png)

- **NQ** - Nasdaq 100 E-mini
- **ES** - S&P 500 E-mini
- **YM** - Dow Jones E-mini
- **RTY** - Russell 2000 E-mini
- **MNQ** - Micro Nasdaq 100
- **MES** - Micro S&P 500
- **M2K** - Micro Russell 2000
- **MYM** - Micro Dow Jones

## Project Structure

```
AI Trainer/
├── main.py                      # Interactive menu system (main entry point)
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # TensorTrade package setup
│
├── config/                      # Configuration files
│   ├── llm_config.yaml          # Phase 3 LLM configuration
│   ├── test_hardware_maximized.yaml  # Hardware testing config
│   └── test_pipeline.yaml       # Pipeline testing config
│
├── src/                         # Core source code (30 files)
│   ├── Training & Environments
│   │   ├── train_phase1.py         # Phase 1 training (2M timesteps)
│   │   ├── train_phase2.py         # Phase 2 training (5M timesteps)
│   │   ├── train_phase3_llm.py     # Phase 3 hybrid training
│   │   ├── environment_phase1.py   # Base trading environment
│   │   ├── environment_phase2.py   # Position management environment
│   │   ├── environment_phase3_llm.py  # LLM-enhanced environment
│   │   └── environment_phase1_simplified.py  # Simplified reward function
│   │
│   ├── Evaluation
│   │   ├── evaluate_phase2.py      # Phase 2 evaluation
│   │   └── evaluate_phase3_llm.py  # Phase 3 evaluation
│   │
│   ├── Data Processing
│   │   ├── update_training_data.py      # Main data pipeline (auto corruption fix)
│   │   ├── process_new_data.py          # Fast processing with corruption detection
│   │   ├── reprocess_from_source.py     # Reprocess corrupted data
│   │   ├── clean_second_data.py         # Second-level data cleaning
│   │   ├── process_second_data.py       # Second-level data processing
│   │   ├── data_validator.py            # Centralized validation functions
│   │   └── validate_data_quality.py     # Data quality validation
│   │
│   ├── Feature Engineering
│   │   ├── feature_engineering.py       # Market regime features
│   │   ├── technical_indicators.py      # 11+ technical indicators
│   │   └── llm_features.py              # Extended LLM features
│   │
│   ├── Trading Logic & Compliance
│   │   ├── market_specs.py              # Market specifications (8 futures)
│   │   ├── apex_compliance_checker.py   # Post-training compliance validation
│   │   └── metadata_utils.py            # Model metadata management
│   │
│   ├── RL & LLM Integration
│   │   ├── hybrid_agent.py              # Decision fusion module
│   │   ├── llm_reasoning.py             # LLM reasoning module
│   │   ├── llm_callback.py              # LLM training callbacks
│   │   ├── kl_callback.py               # KL divergence monitoring
│   │   └── model_utils.py               # Model detection and loading utilities
│   │
│   └── Utilities
│       ├── testing_framework.py         # Optimized testing framework
│       └── diagnose_environment.py      # Environment diagnostics
│
├── data/                        # Training data (CSV files)
│   ├── NQ_D1M.csv              # Example: NQ minute data
│   ├── NQ_D1S.csv              # Example: NQ second data
│   └── .gitkeep
│
├── models/                      # Saved model checkpoints
│   ├── phase1_foundational_final.zip
│   ├── phase2_position_mgmt_final.zip
│   ├── phase3_hybrid_final.zip
│   └── .gitkeep
│
├── results/                     # Evaluation results
│   └── .gitkeep
│
├── logs/                        # Training and execution logs
│   └── .gitkeep
│
├── tensorboard_logs/            # TensorBoard monitoring
│   └── .gitkeep
│
├── img/                         # Screenshots and images
│   ├── Screenshot_105.png       # Main menu
│   ├── Screenshot_106.png       # Data processing
│   └── Screenshot_107.png       # Evaluator
│
├── tests/                       # Test suite (8 files)
│   ├── test_setup.py            # Installation verification
│   ├── test_environment.py      # Environment tests
│   ├── test_integration.py      # Integration tests
│   ├── test_market_selection.py # Market selection tests
│   ├── test_action_masking.py   # Action masking tests
│   ├── test_llm_minimal.py      # Minimal LLM tests
│   ├── test_llm_integration.py  # Full LLM integration tests
│   └── __init__.py
│
├── docs/                        # Comprehensive documentation
│   ├── Apex-Rules.md            # Apex Trader Funding compliance rules
│   ├── QUICK_START_DATA_PROCESSING.md  # Quick start data guide
│   ├── HYBRID_ARCHITECTURE.md   # Phase 3 architecture
│   ├── LLM_INTEGRATION_GUIDE.md # LLM setup and customization
│   ├── TESTING_FRAMEWORK_README.md     # Testing framework guide
│   ├── FIXES_SUMMARY.md         # Recent fixes and improvements
│   └── ...
│
└── scripts/                     # Utility scripts
    ├── run_pipeline.py          # Automated pipeline execution
    ├── benchmark_optimizations.py  # Performance benchmarking
    └── validate_testing_framework.py  # Test validation
```

## Installation

### Prerequisites

- Python 3.11.9 or higher (Python 3.13 supported)
- pip (Python package manager)
- (Optional) CUDA-compatible GPU for faster training

### Step 1: Install Dependencies

Run the main menu and select option 1, or install manually:

```bash
# Using main.py menu (recommended)
python main.py
# Select option 1: Requirements Installation

# Or install manually
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
# Test imports
python -c "import gymnasium, stable_baselines3, pandas, numpy; print('✓ All core dependencies installed')"

# Or run the verification script
python tests/test_setup.py
```

## Quick Start

### Using the Interactive Menu (Recommended)

```bash
python main.py
```

The interactive menu provides 5 main options:

1. **Requirements Installation** - Install all dependencies
2. **Data Processing** - Process market data for training
3. **Training Model** - Train Phase 1, 2, and 3 models
   - Training Test (Local Testing)
   - Training Pod (Production)
   - **Continue from Existing Model** ⭐ NEW
4. **Evaluator** - Evaluate trained models
5. **Exit** - Close the program

![Evaluator Menu](img/Screenshot_107.png)

### Continue Training Feature ⭐ NEW

Resume training from any previously saved model:

```bash
python main.py
# Select 3: Training Model
# Select 3: Continue from Existing Model
# Choose from list of available models
# Select test or production mode
```

**Benefits:**
- Resume interrupted training sessions
- Extend training for models that haven't converged
- Continue training with different hyperparameters
- No need to remember model names - automatic detection
- Preserves timestep count and training progress

**Command-line usage:**
```bash
# Continue training from a specific model
python src/train_phase1.py --continue --model-path models/phase1_foundational_final.zip

# Continue in test mode
python src/train_phase1.py --continue --model-path models/phase1_foundational_final.zip --test
```

### Manual Workflow

#### 1. Process Data

**NEW: Automatic corruption detection and fixing!** 🎉

When you download new data from Databento, just run:

```bash
# Process new data with automatic corruption detection & fixing
python src/process_new_data.py --market NQ
```

The script automatically:
- ✅ Detects corrupted data (divide-by-100 errors)
- ✅ Fixes corruption automatically
- ✅ Validates data quality
- ✅ Generates clean training-ready data

**Alternative (direct processing):**
```bash
# Also has automatic corruption detection built-in
python src/update_training_data.py --market ES
```

**Reprocess old corrupted data:**
```bash
# Deletes old files and reprocesses from source
python src/reprocess_from_source.py --market NQ
```

**Available markets:** NQ, ES, YM, RTY, MNQ, MES, M2K, MYM

📖 **See:** `docs/QUICK_START_DATA_PROCESSING.md` for detailed guide

#### 2. Train Models

```bash
# Phase 1: Entry Signal Learning (2M timesteps)
python src/train_phase1.py

# Phase 2: Position Management (5M timesteps with transfer learning)
python src/train_phase2.py

# Phase 3: Hybrid RL + LLM Agent (5M timesteps)
python src/train_phase3_llm.py
```

For quick testing (reduced dataset):

```bash
python src/train_phase1.py --test
python src/train_phase2.py --test
python src/train_phase3_llm.py --test
```

#### 3. Evaluate Model

```bash
# Phase 2 evaluation
python src/evaluate_phase2.py

# Phase 3 evaluation
python src/evaluate_phase3_llm.py --model models/phase3_hybrid_final --market NQ
```

#### 4. View Training Progress

```bash
tensorboard --logdir tensorboard_logs/
```

## 🌍 Multi-Market Support

The RL Trainer supports **8 different futures markets** with automatic contract specification handling:

### Supported Markets

| Symbol | Name | Contract Size | Tick Size | Tick Value | Default Commission | Type |
|--------|------|--------------|-----------|-----------|-------------------|------|
| **ES** | E-mini S&P 500 | $50 | 0.25 | $12.50 | $2.50/side | E-mini |
| **NQ** | E-mini Nasdaq-100 | $20 | 0.25 | $5.00 | $2.50/side | E-mini |
| **YM** | E-mini Dow Jones | $5 | 1.0 | $5.00 | $2.50/side | E-mini |
| **RTY** | E-mini Russell 2000 | $50 | 0.10 | $5.00 | $2.50/side | E-mini |
| **MNQ** | Micro E-mini Nasdaq | $2 | 0.25 | $0.50 | $0.60/side | Micro |
| **MES** | Micro E-mini S&P 500 | $5 | 0.25 | $1.25 | $0.60/side | Micro |
| **M2K** | Micro E-mini Russell | $5 | 0.10 | $0.50 | $0.60/side | Micro |
| **MYM** | Micro E-mini Dow | $0.50 | 1.0 | $0.50 | $0.60/side | Micro |

### How It Works

1. **Process data** for your desired market (e.g., NQ)
2. **Start training** - the system automatically:
   - Detects available market data files
   - Prompts you to select a market (if multiple exist)
   - Loads correct contract specifications
   - Applies market-specific P&L calculations, commissions, and slippage

### Example: Training on Multiple Markets

```bash
# Process NQ data
python main.py
# Select: Data Processing → NQ

# Train on NQ
python main.py
# Select: Training Model → Training Pod
# System shows: "Detected 2 markets: ES, NQ"
# Select: NQ
# Training uses NQ specs automatically!
```

### Custom Commission Override

You can override default commissions in training configs:

```python
# train_phase1.py or train_phase2.py
PHASE1_CONFIG = {
    # ... other config ...
    'commission_override': 1.50  # Use custom commission instead of market default
}
```

### Market-Dependent Features

- **P&L Calculations**: Automatically use correct contract multiplier
- **Slippage Modeling**:
  - Liquid markets (ES, NQ, MNQ, MES) = 1 tick slippage
  - Less liquid (YM, RTY, M2K, MYM) = 2 ticks slippage
- **Commissions**:
  - E-mini default = $2.50/side
  - Micro default = $0.60/side
  - Fully configurable per training run

## System Architecture

### Three-Phase Curriculum Learning

```
Phase 1: Entry Signal Learning
  ├─ Fixed SL/TP (1.5x ATR SL, 3:1 ratio)
  ├─ 3 actions: Hold, Buy, Sell
  ├─ Focus: Entry signal quality
  ├─ Constraints: Relaxed for learning ($15K trailing DD)
  └─ Duration: 2M timesteps (~6-8 hours on RTX 4000)

Phase 2: Position Management (Transfer Learning)
  ├─ Auto-loads newest Phase 1 model
  ├─ Inherit Phase 1 weights
  ├─ Dynamic SL/TP adjustment
  ├─ 6 actions: Streamlined position management
  ├─ Focus: Risk management
  ├─ Constraints: Strict Apex compliance ($2.5K trailing DD)
  └─ Duration: 5M timesteps (~8-10 hours)

Phase 3: Hybrid RL + LLM Agent
  ├─ Action Space: 6 actions (same as Phase 2)
  ├─ LLM: Phi-3-mini (3.8B params, INT8 quantized)
  ├─ Decision Fusion: Confidence-weighted voting
  ├─ Observation Space: 261D (extended context)
  ├─ Hardware: GPU with 8GB+ VRAM
  └─ Duration: 5M timesteps (~12-16 hours)
```

### Multi-Layer Safety System

```
Layer 1: Environment Level (Primary)
  └─ Apex rules enforced in reward + done signal

Layer 2: Wrapper Level (Secondary)
  └─ Safety validation before action execution

Layer 3: Validation Level (Verification)
  └─ Post-training compliance checks
```

### Thread Pool Management

The system includes automatic thread pool management to prevent OpenBLAS pthread creation failures:

```python
# Automatic BLAS thread limiting
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Automatic environment count alignment with CPU cores
# Override with: TRAINER_NUM_ENVS=40
```

**Benefits:**
- Stable training on constrained systems
- Prevents "pthread_create failed" errors
- Automatic CPU core detection and optimization
- Configurable overrides for advanced users

## Apex Trader Funding Compliance

The system enforces 100% compliance with Apex rules:

- 4:59 PM ET mandatory position close
- $2,500 max trailing drawdown limit
- No overnight positions
- Position size constraints
- Trade frequency limits

See `docs/Apex-Rules.md` for complete compliance details.

## Configuration

### Phase 1 Configuration (src/train_phase1.py)

```python
PHASE1_CONFIG = {
    # Training
    'total_timesteps': 2_000_000,
    'num_envs': 80,  # Auto-adjusted based on CPU cores
    
    # Network
    'policy_layers': [512, 256, 128],
    
    # PPO parameters
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 512,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    
    # Device
    'device': 'cuda'  # or 'cpu'
}
```

### Phase 2 Configuration (src/train_phase2.py)

```python
PHASE2_CONFIG = {
    # Training
    'total_timesteps': 5_000_000,
    'num_envs': 80,  # Auto-adjusted based on CPU cores
    
    # Network (reduced capacity to prevent overfitting)
    'policy_layers': [512, 256, 128],
    
    # PPO parameters
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 512,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'target_kl': 0.01,
    
    # Device
    'device': 'cuda'  # or 'cpu'
}
```

**Phase 2 Actions (6 total):**
- Actions 0-2: Entry/Exit (Hold, Buy, Sell)
- Action 3: Move SL to Break-Even
- Action 4: Enable Trailing Stop
- Action 5: Disable Trailing Stop

*Note: Simplified from original 9 actions for improved sample efficiency and reduced overfitting*

### Phase 3: Hybrid RL + LLM Agent (Advanced)

**NEW** - Combines reinforcement learning with LLM reasoning for intelligent trading decisions.

```bash
# Phase 3: Hybrid LLM Agent (requires GPU with 8GB+ VRAM)
python main.py
# Select: Training → Phase 3 Hybrid (Test Mode)  # ~30 min test run
# Select: Training → Phase 3 Hybrid (Production)  # ~12-16 hours
```

**Phase 3 Features:**
- **Context Awareness**: 261D observations (vs 228D in Phase 2)
  - Extended market context (ADX slope, VWAP distance, volatility regime)
  - Multi-timeframe indicators (SMA-50/200 slopes, multi-timeframe RSI)
  - Pattern recognition (higher/lower highs/lows, support/resistance)
  - Risk context (unrealized P&L, drawdown tracking, consecutive losses)
- **LLM Reasoning**: Phi-3-mini provides explicit analysis before decisions
- **Decision Fusion**: Intelligent combination of RL + LLM recommendations
- **Risk Management**: Automatic veto of high-risk actions
- **Selective Querying**: Queries LLM only when needed (reduces latency)

**Hardware Requirements:**
- GPU: RTX 3060 or better (8GB+ VRAM recommended)
- RAM: 16GB minimum, 32GB recommended
- Storage: 10GB additional space

**Key Files:**
- `src/train_phase3_llm.py` - Training pipeline
- `src/hybrid_agent.py` - Decision fusion logic
- `src/llm_reasoning.py` - LLM reasoning module
- `src/llm_features.py` - Extended feature calculation
- `config/llm_config.yaml` - LLM configuration
- `docs/HYBRID_ARCHITECTURE.md` - Complete architecture documentation

**Quick Start:**
```bash
# Install LLM dependencies
pip install transformers torch accelerate bitsandbytes sentencepiece

# Run test training
python main.py
# Select: Training → Phase 3 Hybrid (Test Mode)
```

**Configuration:**
Edit `config/llm_config.yaml` to customize:
- LLM model selection (Phi-3-mini, Phi-3-small, etc.)
- Decision fusion parameters (LLM weight: 0.0-1.0, confidence thresholds)
- Prompt templates for different market conditions
- Risk veto thresholds
- Performance optimization settings

**Performance:**
- Test mode: ~30 minutes (50K timesteps)
- Production: ~12-16 hours (5M timesteps)
- LLM inference: ~15-20ms per query (RTX 3060)
- VRAM usage: ~4GB with INT8 quantization

**Evaluation:**
```bash
python src/evaluate_phase3_llm.py \
    --model models/phase3_hybrid_final \
    --market NQ \
    --episodes 20
```

**Documentation:**
- [Hybrid Architecture Guide](docs/HYBRID_ARCHITECTURE.md) - Complete system overview
- [LLM Integration Guide](docs/LLM_INTEGRATION_GUIDE.md) - Setup and customization

## Testing Framework

The system includes an **optimized testing framework** with two modes:

### 1. Hardware-Maximized Validation Mode
- Full GPU utilization with reduced timesteps
- Cached LLM feature calculations
- Vectorized decision fusion
- Performance benchmarking

```bash
python src/testing_framework.py --mode hardware_maximized --market NQ
```

### 2. Automated Sequential Pipeline Mode
- Continuous execution with checkpointing
- Automated validation
- Progress tracking

```bash
python src/testing_framework.py --mode pipeline --market NQ
```

### Run Standard Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific tests
python -m pytest tests/test_environment.py -v
python -m pytest tests/test_integration.py -v
python -m pytest tests/test_llm_integration.py -v
python -m pytest tests/test_action_masking.py -v
```

**Test Coverage:**
- Environment validation
- Action masking correctness
- Integration workflows
- LLM integration
- Market selection
- Apex compliance

## Data Format

Training data should be CSV files with the following columns:

**Minute Data (D1M):**
- DateTime
- Open, High, Low, Close
- Volume

**Second Data (D1S):**
- DateTime
- Open, High, Low, Close
- Volume

Data files should be placed in `data/` directory with naming convention:
- `{INSTRUMENT}_D1M.csv` (minute data)
- `{INSTRUMENT}_D1S.csv` (second data)

### Data Processing Pipeline

```bash
# Process new data (auto-corruption detection)
python src/process_new_data.py --market NQ

# Reprocess old corrupted data
python src/reprocess_from_source.py --market NQ

# Validate data quality
python src/validate_data_quality.py --market NQ
```

**Corruption Detection:** Automatically detects and fixes divide-by-100 errors using statistical analysis and percentile-based validation.

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or number of parallel environments:

```python
# In train_phase1.py or train_phase2.py
'batch_size': 256,  # Down from 512
'num_envs': 40,     # Down from 80
```

Or use CPU:

```python
'device': 'cpu'
```

### Training Divergence

- Reduce learning rate: `3e-4` → `1e-4`
- Increase clip range: `0.2` → `0.3`
- Check reward scaling
- Enable KL divergence monitoring

### Import Errors

Ensure TensorTrade is installed:

```bash
python setup.py install
```

### Model Not Found (Phase 2)

The system auto-detects the newest Phase 1 model. Ensure:
- Phase 1 training completed
- Model exists in `models/` directory
- Model filename contains "phase1"

### Thread Pool Errors (OpenBLAS)

The system automatically limits thread pools. If you encounter issues:

```bash
# Explicitly set thread limits
set TRAINER_MAX_BLAS_THREADS=1
set TRAINER_NUM_ENVS=40

# Or use environment variables
export TRAINER_MAX_BLAS_THREADS=1
export TRAINER_NUM_ENVS=40
```

### Data Corruption

Run reprocessing:

```bash
python src/reprocess_from_source.py --market NQ
```

## Performance Metrics

Target metrics for successful training:

| Metric | Target | Status |
|--------|--------|--------|
| Apex Compliance | 100% | Enforced |
| Sharpe Ratio | > 2.5 | Phase 2 target |
| Win Rate | > 50% | Target |
| Max Drawdown | < 5% | Enforced |
| GPU Utilization | 85%+ | Optimized |
| LLM Query Latency | < 20ms | Phase 3 |

## Logs and Outputs

- **Training logs**: `logs/`
- **TensorBoard logs**: `tensorboard_logs/`
- **Model checkpoints**: `models/`
- **Evaluation results**: `results/`
- **Testing logs**: `logs/testing/`

## Tips

- All operations are logged for debugging
- Press Ctrl+C to cancel any operation
- Check logs directory for detailed output
- Ensure sufficient disk space (>20GB for full training)
- Use test mode (`--test` flag) for quick validation
- Monitor GPU/CPU usage during training
- Use "Continue Training" to resume interrupted sessions
- Phase 2 automatically finds your newest Phase 1 model
- Run tests before committing changes: `pytest tests/ -v`

## Common Workflows

### 1. First Time Setup

```bash
python main.py
# Select 1: Requirements Installation
# Select 2: Data Processing (choose instrument)
# Wait for completion
```

### 2. Complete Training Pipeline (Production)

```bash
python main.py
# Select 3: Training Model → 2: Training Pod (Production)
# Wait for completion (~26-34 hours total)
# Select 4: Evaluator
```

### 3. Quick Test Run

```bash
python main.py
# Select 3: Training Model → 1: Training Test (Local)
# Wait for completion (~30-45 minutes)
# Select 4: Evaluator
```

### 4. Resume Training from Checkpoint

```bash
python main.py
# Select 3: Training Model → 3: Continue from Existing Model
# Select model from list
# Choose test or production mode
# Training continues from checkpoint
```

### 5. Data Processing Workflow

```bash
# Download new data from Databento
# Process with automatic corruption detection
python src/process_new_data.py --market NQ

# Validate data quality
python src/validate_data_quality.py --market NQ

# Start training
python src/train_phase1.py --market NQ
```

### 6. Testing Workflow

```bash
# Run hardware-maximized test
python src/testing_framework.py --mode hardware_maximized --market NQ

# Run pipeline test
python src/testing_framework.py --mode pipeline --market NQ

# Run unit tests
python -m pytest tests/ -v
```

## Technology Stack

- **RL Framework**: Stable Baselines3 (PPO algorithm)
- **Action Masking**: sb3-contrib (MaskablePPO)
- **Environment**: Gymnasium API
- **Deep Learning**: PyTorch 2.0+, TensorFlow 2.15+
- **LLM Integration**: Hugging Face Transformers (Phi-3-mini)
- **Quantization**: bitsandbytes (INT8/INT4)
- **Data Processing**: Pandas, NumPy, SciPy, Scikit-Learn
- **Visualization**: Matplotlib, Plotly, TensorBoard
- **UI**: Colorama, tqdm, rich
- **Testing**: pytest

## License

Apache License 2.0 (inherited from TensorTrade)

## Contributing

1. Run tests before committing: `pytest tests/ -v`
2. Follow existing code style
3. Update documentation for new features
4. Ensure Apex compliance for any trading logic changes
5. Update CHANGELOG.md with changes

## Support & Contact

**Author**: Javier  
**X (Twitter)**: [@javiertradess](https://x.com/javiertradess)

For technical issues:
1. Check logs in `logs/` directory
2. Review `docs/Apex-Rules.md` for compliance questions
3. Run tests to diagnose issues: `pytest tests/ -v`
4. Review `docs/FIXES_SUMMARY.md` for recent bug fixes

## Version

**RL TRAINER v1.1.0** - November 2025  
Based on TensorTrade v1.0.4-dev1

**Recent Updates:**
- ✅ Three-phase curriculum learning (Phase 3 Hybrid LLM)
- ✅ Continue training from any checkpoint
- ✅ Action space optimization (9→6 actions in Phase 2)
- ✅ Automatic data corruption detection & fixing
- ✅ Thread pool management for stable training
- ✅ Comprehensive testing framework
- ✅ Smart model detection and auto-loading
- ✅ Multi-market support with auto contract specs

---

**Getting Started**: Run `python main.py` and follow the interactive menu!

**Star this repo if you find it useful!** ⭐
