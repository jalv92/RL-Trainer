#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2: Position Management Training with Transfer Learning
- Load Phase 1 weights
- Expand action space from 3 -> 9
- Train dynamic SL/TP management
- Duration: 15M timesteps (~12-16 hours on RTX 4000 Ada 20GB)

Based on: Transfer Learning + OpenAI Spinning Up PPO
Optimized for: RunPod RTX 4000 Ada deployment
"""

import os
import sys
import glob
import torch
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
# RL FIX #4: Import MaskablePPO for action masking support
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnNoModelImprovement
from environment_phase2 import TradingEnvironmentPhase2
from kl_callback import KLDivergenceCallback
from feature_engineering import add_market_regime_features

# Set UTF-8 encoding for Windows compatibility
if os.name == 'nt':  # Windows
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')
    except:
        pass


def safe_print(message=""):
    """Print with fallback for encoding errors on Windows."""
    try:
        print(message)
    except UnicodeEncodeError:
        # Fallback: replace Unicode characters with ASCII equivalents
        replacements = {
            '✓': '[OK]',
            '✅': '[OK]',
            '✗': '[X]',
            '❌': '[X]',
            '→': '->',
            '⚠': '[WARN]',
            '⚠️': '[WARN]',
            '—': '-',
            '–': '-',
            '’': "'",
            '“': '"',
            '”': '"',
        }
        ascii_message = message
        for src, target in replacements.items():
            ascii_message = ascii_message.replace(src, target)
        ascii_message = ascii_message.encode('ascii', errors='ignore').decode('ascii')
        print(ascii_message)


# Check progress bar availability
PROGRESS_BAR_AVAILABLE = False
try:
    import tqdm
    import rich
    PROGRESS_BAR_AVAILABLE = True
except ImportError:
    safe_print("[WARNING] tqdm or rich not installed - progress bar will be disabled")
    safe_print("[WARNING] Install with: pip install 'stable-baselines3[extra]'")
    safe_print("[WARNING] Training will continue without progress bar visualization")

# COMPREHENSIVE OVERFITTING FIXES + RTX 4000 Ada OPTIMIZATIONS
# STRATEGY B: Data-aware training with early stopping
PHASE2_CONFIG = {
    # Training - Data-constrained budget (22,984 unique episodes)
    'total_timesteps': 5_000_000,  # 5M max - early stopping will find optimal point
    'num_envs': 80,  # INCREASED for RTX 4000 Ada 20GB: excellent parallelization

    # Network architecture - REDUCED to prevent overfitting
    'policy_layers': [512, 256, 128],  # DOWN from [1024,512,256,128,64,32] - less capacity = less memorization

    # PPO parameters - MULTIPLE FIXES APPLIED
    'learning_rate': 3e-4,  # FIX #7: UP from 1e-4 - match Phase 1, allow faster adaptation
    'n_steps': 2048,
    # OPTIMIZED batch size for RTX 4000 Ada
    # Calculation: 80 envs × 2048 = 163,840 samples per update
    # 163,840 / 512 = 320 minibatches (excellent balance)
    # Larger batch = better GPU utilization while maintaining generalization
    'batch_size': 512,  # INCREASED from 256 for 20GB GPU
    'n_epochs': 5,  # FIX: DOWN from 10 - reduce overfitting on same batch
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.06,  # FIXED: Increased for 9-action exploration (was 0.015/0.03)
    'vf_coef': 0.25,  # FIX: DOWN from 0.5 - reduce value function overfitting
    'max_grad_norm': 0.5,

    # Environment parameters
    'window_size': 20,
    'initial_balance': 50000,
    'initial_sl_multiplier': 1.5,
    'initial_tp_ratio': 3.0,
    'position_size': 1.0,  # Full size (0.5 in Phase 1)
    'trailing_dd_limit': 2500,  # Strict Apex rules ($5k in Phase 1)
    'tighten_sl_step': 0.5,
    'extend_tp_step': 1.0,

    # Evaluation - More frequent for better early stopping
    'eval_freq': 50_000,  # Every 50K = 100 evals max (5M / 50K)
    'n_eval_episodes': 10,  # Better estimate of true performance

    # Early stopping - Aggressive to prevent overfitting with limited data
    'use_early_stopping': True,
    'early_stop_max_no_improvement': 5,  # DOWN from 10 - stop faster
    'early_stop_min_evals': 3,  # DOWN from 5 - require fewer evals

    # Device configuration
    # SWITCHED TO GPU FOR TESTING (heavy environment bottleneck detected)
    # Heavy feature engineering (33 features) + dual data sources (minute + second-level)
    # CPU environment overhead (~80-100ms) > GPU data transfer overhead (~20-30ms)
    # GPU freed CPU to handle environment simulation better
    'device': 'cuda',  # Testing GPU for heavy environment scenarios

    # Transfer learning
    'phase1_model_path': 'models/phase1_foundational_final.zip',
    'phase1_vecnorm_path': 'models/phase1_vecnorm.pkl',

    # Small-World Rewiring (Watts-Strogatz inspired)
    # Based on: Watts & Strogatz (1998) "Collective dynamics of 'small-world' networks"
    # Theory: 1-5% random rewiring creates small-world properties
    # - Preserves local clustering (Phase 1 patterns)
    # - Creates shortcuts (faster Phase 2 adaptation)
    'use_smallworld_rewiring': True,  # Enable small-world transfer
    'rewiring_probability': 0.05,  # 5% of weights rewired (optimal per Watts-Strogatz)

    # NEW: Early stopping with KL monitoring
    'target_kl': 0.01,
    'use_kl_callback': True,

    # NEW: Learning rate scheduling (matching Phase 1)
    'use_lr_schedule': True,
    'lr_final_fraction': 0.2,  # End at 20% of initial LR

    # RL FIX #7: Entropy decay schedule for exploration->exploitation
    # NOTE: MaskablePPO doesn't support entropy schedules - use fixed value instead
    'use_ent_schedule': False,  # Disabled for MaskablePPO compatibility
    'ent_coef_initial': 0.02,
    'ent_coef_final': 0.005
}

def create_lr_schedule(initial_lr, final_fraction, total_timesteps):
    """Create linear learning rate schedule."""
    def lr_schedule(progress):
        # progress goes from 0 to 1
        return initial_lr * (1 - progress * (1 - final_fraction))
    return lr_schedule


def get_learning_rate(config):
    """Get learning rate, with schedule if enabled."""
    learning_rate = config["learning_rate"]
    if config.get("use_lr_schedule", False):
        learning_rate = create_lr_schedule(
            config["learning_rate"],
            config["lr_final_fraction"],
            config["total_timesteps"]
        )
    return learning_rate



def create_entropy_schedule(initial_ent, final_ent, total_timesteps):
    """
    Create linear entropy coefficient schedule.

    RL FIX #7: Entropy decay for better exploration->exploitation transition.
    Start with high exploration (0.02), end with low exploration (0.005).

    Args:
        initial_ent: Starting entropy coefficient (higher = more exploration)
        final_ent: Final entropy coefficient (lower = more exploitation)
        total_timesteps: Total training timesteps

    Returns:
        Callable schedule function
    """
    def ent_schedule(progress):
        # Linear decay from initial_ent to final_ent
        return initial_ent * (1 - progress) + final_ent * progress
    return ent_schedule


def find_data_file():
    """Find training data file with priority order."""
    # Get project root directory (parent of src/)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(script_dir, 'data')

    filenames_to_try = [
        'D1M.csv',  # New generic 1-minute data format
        'ES_D1M.csv',  # Instrument-prefixed 1-minute data
        'es_training_data_CORRECTED_CLEAN.csv',  # Legacy ES format
        'es_training_data_CORRECTED.csv',
        'databento_es_training_data_processed_cleaned.csv',
        'databento_es_training_data_processed.csv'
    ]

    for filename in filenames_to_try:
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            return path

    # Fallback: look for instrument-prefixed files like ES_D1M.csv, NQ_D1M.csv, etc.
    pattern = os.path.join(data_dir, '*_D1M.csv')
    instrument_files = sorted(glob.glob(pattern))
    if instrument_files:
        return instrument_files[0]

    raise FileNotFoundError(
        f"Training data not found in {data_dir}. "
        f"Expected one of: {filenames_to_try} or any '*_D1M.csv' file"
    )


def load_data(train_split=0.7):
    """
    Load and prepare training data with proper train/val split.

    FIX #1: CRITICAL - Separate train and validation data to prevent overfitting

    Args:
        train_split: Fraction of data to use for training (default 0.7 = 70%)

    Returns:
        train_data, val_data, train_second_data, val_second_data
    """
    data_path = find_data_file()
    safe_print(f"[DATA] Loading minute-level data from {data_path}")

    try:
        data = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
    except Exception:
        df_tmp = pd.read_csv(data_path)
        time_col = 'timestamp' if 'timestamp' in df_tmp.columns else 'datetime'
        df_tmp[time_col] = pd.to_datetime(df_tmp[time_col])
        data = df_tmp.set_index(time_col)

    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC').tz_convert('America/New_York')
    elif str(data.index.tz) != 'America/New_York':
        data.index = data.index.tz_convert('America/New_York')

    safe_print(f"[DATA] Loaded {len(data):,} rows")
    safe_print(f"[DATA] Full date range: {data.index.min()} to {data.index.max()}")

    # FIX #1: Chronological train/val split
    train_end_idx = int(len(data) * train_split)
    train_data = data.iloc[:train_end_idx].copy()
    val_data = data.iloc[train_end_idx:].copy()

    safe_print(f"\n[SPLIT] Train/Val Split Applied:")
    safe_print(f"[SPLIT] Train: {len(train_data):,} bars ({train_split*100:.0f}%) - {train_data.index.min()} to {train_data.index.max()}")
    safe_print(f"[SPLIT] Val:   {len(val_data):,} bars ({(1-train_split)*100:.0f}%) - {val_data.index.min()} to {val_data.index.max()}")

    # Add market regime features to BOTH splits separately (prevent leakage)
    safe_print("[DATA] Adding market regime features to train data...")
    train_data = add_market_regime_features(train_data)
    safe_print("[DATA] Adding market regime features to val data...")
    val_data = add_market_regime_features(val_data)
    safe_print(f"[DATA] Feature count: {len(train_data.columns)}")

    # Load and split second-level data
    train_second_data = None
    val_second_data = None
    # Get project root directory (parent of src/)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    second_data_candidates = [
        os.path.join(script_dir, 'data', 'D1S.csv'),
        os.path.join(script_dir, 'data', 'ES_D1S.csv')
    ]
    if not any(os.path.exists(path) for path in second_data_candidates):
        pattern = os.path.join(script_dir, 'data', '*_D1S.csv')
        instrument_seconds = sorted(glob.glob(pattern))
        second_data_candidates.extend(instrument_seconds)

    second_data_path = next((path for path in second_data_candidates if os.path.exists(path)), None)

    if second_data_path and os.path.exists(second_data_path):
        try:
            safe_print(f"[DATA] Loading second-level data from {second_data_path}")
            second_data = pd.read_csv(second_data_path, index_col='ts_event', parse_dates=True)
            if second_data.index.tz is None:
                second_data.index = second_data.index.tz_localize('UTC').tz_convert('America/New_York')
            elif str(second_data.index.tz) != 'America/New_York':
                second_data.index = second_data.index.tz_convert('America/New_York')

            # Split second-level data by time range
            train_second_data = second_data[(second_data.index >= train_data.index[0]) &
                                           (second_data.index <= train_data.index[-1])].copy()
            val_second_data = second_data[(second_data.index >= val_data.index[0]) &
                                         (second_data.index <= val_data.index[-1])].copy()

            safe_print(f"[SPLIT] Train second-level: {len(train_second_data):,} bars")
            safe_print(f"[SPLIT] Val second-level: {len(val_second_data):,} bars")
        except Exception as e:
            safe_print(f"[DATA] Warning: Could not load second-level data: {e}")
    else:
        missing_path = os.path.join(script_dir, 'data', 'D1S.csv')
        safe_print(f"[DATA] Second-level data not found (looked for D1S/ES_D1S/_D1S.csv). Optional feature. Missing reference: {missing_path}")

    return train_data, val_data, train_second_data, val_second_data


def make_env(data, second_data, env_id, config):
    """
    Create Phase 2 environment factory with random episode starts to prevent temporal leakage.

    RL FIX #3: Removed fixed env_id-based seeding to eliminate temporal correlation.
    Each environment now samples truly random episodes on every reset.
    """
    def _init():
        # Calculate episode parameters
        episode_length = config.get('episode_length', 390)  # 1 trading day
        window_size = config['window_size']

        # RL FIX #3: TRUE random start position (no fixed seeding)
        # Previous: np.random.seed(env_id + 42) caused same episodes each run
        # New: Use global random state - different episodes every time
        max_start = len(data) - episode_length - window_size
        if max_start <= window_size:
            start_idx = window_size
        else:
            # NO SEEDING - use global random state for true randomization
            # This prevents temporal overfitting and improves generalization
            start_idx = np.random.randint(0, max_start)

        end_idx = start_idx + episode_length

        # Extract episode data - this creates independent episodes
        env_data = data.iloc[start_idx:end_idx].copy()

        # Filter second-level data for this episode
        env_second_data = None
        if second_data is not None:
            start_time = data.index[start_idx]
            end_time = data.index[end_idx-1]
            mask = (second_data.index >= start_time) & (second_data.index < end_time)
            env_second_data = second_data[mask].copy()

        if env_id == 0:
            safe_print(f"[ENV] Env {env_id}: Random start at index {start_idx}, "
                  f"period {data.index[start_idx]} to {data.index[end_idx-1]}")
            safe_print(f"[ENV] Episode length: {len(env_data)} bars")
            safe_print(f"[ENV] Data range: {len(data)} total bars, using episodes of {episode_length}")
            safe_print(f"[ENV] RL FIX #3: Using true random sampling (no fixed seeding)")
            if env_second_data is not None:
                safe_print(f"[ENV] Second-level data: {len(env_second_data)} bars")

        env = TradingEnvironmentPhase2(
            data=env_data,
            window_size=config['window_size'],
            initial_balance=config['initial_balance'],
            second_data=env_second_data,  # Pass second-level data
            initial_sl_multiplier=config['initial_sl_multiplier'],
            initial_tp_ratio=config['initial_tp_ratio'],
            position_size_contracts=config['position_size'],
            trailing_drawdown_limit=config['trailing_dd_limit'],
            tighten_sl_step=config['tighten_sl_step'],
            extend_tp_step=config['extend_tp_step']
        )

        return Monitor(env)

    return _init


def apply_smallworld_rewiring(weight_tensor, bias_tensor, rewiring_prob, device='cuda'):
    """
    Apply Watts-Strogatz inspired rewiring to neural network weights.

    Theory (Watts & Strogatz 1998):
    - Regular lattice: High clustering, long paths
    - Random rewiring: Creates shortcuts while preserving local structure
    - Result: Small-world properties = fast propagation + pattern preservation

    For neural networks:
    - Keep (1-p)% of learned connections (clustering = Phase 1 patterns)
    - Rewire p% randomly (shortcuts = faster Phase 2 adaptation)

    Args:
        weight_tensor: Original weight matrix from Phase 1
        bias_tensor: Original bias vector from Phase 1
        rewiring_prob: Probability of rewiring each weight (0.01-0.10 typical)
        device: torch device

    Returns:
        Rewired weight tensor, original bias tensor
    """
    rewired_weight = weight_tensor.clone()

    # Create rewiring mask: True = rewire, False = keep
    rewiring_mask = torch.rand_like(rewired_weight) < rewiring_prob
    num_rewired = rewiring_mask.sum().item()
    total_weights = rewiring_mask.numel()

    # Rewire selected connections with random values
    # Use Xavier/Glorot initialization for rewired connections
    fan_in, fan_out = rewired_weight.shape[1], rewired_weight.shape[0]
    std = np.sqrt(2.0 / (fan_in + fan_out))

    # Generate random rewiring values
    random_weights = torch.randn_like(rewired_weight) * std

    # Apply rewiring: keep original where mask=False, use random where mask=True
    rewired_weight = torch.where(rewiring_mask, random_weights, weight_tensor)

    return rewired_weight, bias_tensor, num_rewired, total_weights


def load_phase1_and_transfer(config, env):
    """
    Load Phase 1 model and perform transfer learning with optional small-world rewiring.

    Strategy:
    1. Check if Phase 1 model exists
    2. Create new Phase 2 model with expanded action space (9 actions)
    3. Load Phase 1 weights into shared layers
    4. OPTIONAL: Apply small-world rewiring (Watts-Strogatz)
       - Preserves 95% of Phase 1 connections (high clustering)
       - Rewires 5% randomly (creates shortcuts)
       - Result: Faster adaptation while preventing catastrophic forgetting
    5. Initialize new action heads with small random weights
    6. Return Phase 2 model ready for training

    Returns:
        PPO model with transferred Phase 1 knowledge
    """
    phase1_path = config['phase1_model_path']

    if not os.path.exists(phase1_path):
        safe_print(f"\n[WARNING] Phase 1 model not found at {phase1_path}")
        safe_print("[WARNING] Starting Phase 2 from scratch (not recommended)")
        safe_print("[WARNING] For best results, train Phase 1 first!")
        return None

    safe_print(f"\n[TRANSFER] Loading Phase 1 model from {phase1_path}")

    try:
        # Load Phase 1 model (3 actions)
        phase1_model = PPO.load(phase1_path, device=config['device'])
        safe_print("[TRANSFER] [OK] Phase 1 model loaded")

        # Create new Phase 2 model (9 actions with action masking)
        safe_print("[TRANSFER] Creating Phase 2 model with expanded action space (9 actions + masking)...")
        safe_print("[TRANSFER] RL FIX #4: Using MaskablePPO for efficient exploration")

        learning_rate = get_learning_rate(config)

        # RL FIX #7: Use fixed entropy coefficient (MaskablePPO doesn't support schedules)
        ent_coef = 0.06  # FIXED: 4x higher for 9-action space exploration

        # RL FIX #4: Use MaskablePPO instead of standard PPO
        phase2_model = MaskablePPO(
            'MlpPolicy',
            env,
            learning_rate=learning_rate,
            n_steps=config['n_steps'],
            batch_size=config['batch_size'],
            n_epochs=config['n_epochs'],
            gamma=config['gamma'],
            gae_lambda=config['gae_lambda'],
            clip_range=config['clip_range'],
            ent_coef=ent_coef,  # Use schedule instead of fixed value
            vf_coef=config['vf_coef'],
            max_grad_norm=config['max_grad_norm'],
            policy_kwargs={
                'net_arch': dict(
                    pi=config['policy_layers'],
                    vf=config['policy_layers']
                ),
                'activation_fn': torch.nn.ReLU
            },
            device=config['device'],
            verbose=1,
            tensorboard_log='./tensorboard_logs/phase2/'
        )

        # Transfer weights from Phase 1 to Phase 2
        use_rewiring = config.get('use_smallworld_rewiring', False)
        rewiring_prob = config.get('rewiring_probability', 0.05)

        if use_rewiring:
            safe_print("[TRANSFER] Transferring Phase 1 knowledge with SMALL-WORLD REWIRING...")
            safe_print(f"[TRANSFER] Rewiring probability: {rewiring_prob:.1%} (Watts-Strogatz model)")
            safe_print(f"[TRANSFER] This preserves {(1-rewiring_prob)*100:.1f}% of learned patterns")
        else:
            safe_print("[TRANSFER] Transferring Phase 1 knowledge (standard copy)...")

        total_rewired = 0
        total_weights_transferred = 0

        with torch.no_grad():
            # Get extractors
            phase1_extractor = phase1_model.policy.mlp_extractor
            phase2_extractor = phase2_model.policy.mlp_extractor

            # Transfer policy network weights (shared layers)
            try:
                for i, (p1_layer, p2_layer) in enumerate(zip(
                    phase1_extractor.policy_net,
                    phase2_extractor.policy_net
                )):
                    if hasattr(p1_layer, 'weight'):
                        if p1_layer.weight.shape == p2_layer.weight.shape:
                            if use_rewiring:
                                # Apply small-world rewiring
                                rewired_weight, bias, num_rewired, total_w = apply_smallworld_rewiring(
                                    p1_layer.weight,
                                    p1_layer.bias,
                                    rewiring_prob,
                                    device=config['device']
                                )
                                p2_layer.weight.copy_(rewired_weight)
                                p2_layer.bias.copy_(bias)
                                total_rewired += num_rewired
                                total_weights_transferred += total_w
                                safe_print(f"  [REWIRE] Policy layer {i}: {p1_layer.weight.shape} "
                                      f"({num_rewired:,}/{total_w:,} rewired = {num_rewired/total_w*100:.2f}%)")
                            else:
                                # Standard transfer (copy all)
                                p2_layer.weight.copy_(p1_layer.weight)
                                p2_layer.bias.copy_(p1_layer.bias)
                                safe_print(f"  [OK] Transferred policy layer {i}: {p1_layer.weight.shape}")
            except Exception as e:
                safe_print(f"  [!] Warning: Could not transfer all policy layers: {e}")

            # Transfer value network weights (shared layers)
            try:
                for i, (p1_layer, p2_layer) in enumerate(zip(
                    phase1_extractor.value_net,
                    phase2_extractor.value_net
                )):
                    if hasattr(p1_layer, 'weight'):
                        if p1_layer.weight.shape == p2_layer.weight.shape:
                            if use_rewiring:
                                # Apply small-world rewiring
                                rewired_weight, bias, num_rewired, total_w = apply_smallworld_rewiring(
                                    p1_layer.weight,
                                    p1_layer.bias,
                                    rewiring_prob,
                                    device=config['device']
                                )
                                p2_layer.weight.copy_(rewired_weight)
                                p2_layer.bias.copy_(bias)
                                total_rewired += num_rewired
                                total_weights_transferred += total_w
                                safe_print(f"  [REWIRE] Value layer {i}: {p1_layer.weight.shape} "
                                      f"({num_rewired:,}/{total_w:,} rewired = {num_rewired/total_w*100:.2f}%)")
                            else:
                                # Standard transfer (copy all)
                                p2_layer.weight.copy_(p1_layer.weight)
                                p2_layer.bias.copy_(p1_layer.bias)
                                safe_print(f"  [OK] Transferred value layer {i}: {p1_layer.weight.shape}")
            except Exception as e:
                safe_print(f"  [!] Warning: Could not transfer all value layers: {e}")

            # Note: Action head (3->9 actions) is left with random initialization
            # This allows the model to learn new actions while preserving pattern knowledge

        safe_print("[TRANSFER] [OK] Transfer learning complete!")
        if use_rewiring:
            safe_print(f"[TRANSFER] Small-world rewiring applied: {total_rewired:,}/{total_weights_transferred:,} "
                  f"weights rewired ({total_rewired/total_weights_transferred*100:.2f}%)")
            safe_print("[TRANSFER] Phase 1 patterns preserved in high-clustering regions")
            safe_print("[TRANSFER] Random shortcuts created for faster Phase 2 adaptation")
        else:
            safe_print("[TRANSFER] Phase 1 knowledge (entry patterns) preserved")
        safe_print("[TRANSFER] New actions (3-8) initialized for learning")

        return phase2_model

    except Exception as e:
        safe_print(f"\n[ERROR] Transfer learning failed: {e}")
        safe_print("[ERROR] Starting Phase 2 from scratch...")
        return None


def train_phase2():
    """Execute Phase 2 training with transfer learning."""
    safe_print("=" * 80)
    safe_print("PHASE 2: POSITION MANAGEMENT MASTERY")
    safe_print("=" * 80)
    safe_print(f"[CONFIG] Total timesteps: {PHASE2_CONFIG['total_timesteps']:,}")
    safe_print(f"[CONFIG] Parallel envs: {PHASE2_CONFIG['num_envs']}")
    safe_print(f"[CONFIG] Network: {PHASE2_CONFIG['policy_layers']}")
    safe_print(f"[CONFIG] Action space: 9 (RL Fix #9: split toggle -> enable/disable)")
    safe_print(f"[CONFIG] Device: {PHASE2_CONFIG['device']}")
    safe_print(f"[CONFIG] Position size: {PHASE2_CONFIG['position_size']} contracts (full)")
    safe_print(f"[CONFIG] Trailing DD: ${PHASE2_CONFIG['trailing_dd_limit']:,} (strict Apex)")
    safe_print()

    # Create directories
    os.makedirs('models/phase2', exist_ok=True)
    os.makedirs('models/phase2/checkpoints', exist_ok=True)
    os.makedirs('logs/phase2', exist_ok=True)
    os.makedirs('tensorboard_logs/phase2', exist_ok=True)

    # Load data with train/val split - FIX #1
    train_data, val_data, train_second_data, val_second_data = load_data(train_split=0.7)

    # Create vectorized TRAINING environments (use TRAIN data only)
    safe_print(f"\n[ENV] Creating {PHASE2_CONFIG['num_envs']} Phase 2 TRAINING environments...")
    env_fns = [make_env(train_data, train_second_data, i, PHASE2_CONFIG) for i in range(PHASE2_CONFIG['num_envs'])]

    if PHASE2_CONFIG['num_envs'] > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    safe_print("[ENV] [OK] Phase 2 training environments created")

    # Create VALIDATION environment (use VAL data - CRITICAL FIX)
    safe_print("[EVAL] Creating VALIDATION environment (unseen data)...")
    eval_env = DummyVecEnv([lambda: Monitor(TradingEnvironmentPhase2(
        data=val_data,  # CHANGED: Use validation data
        window_size=PHASE2_CONFIG['window_size'],
        initial_balance=PHASE2_CONFIG['initial_balance'],
        second_data=val_second_data,  # CHANGED: Use val second-level data
        initial_sl_multiplier=PHASE2_CONFIG['initial_sl_multiplier'],
        initial_tp_ratio=PHASE2_CONFIG['initial_tp_ratio'],
        position_size_contracts=PHASE2_CONFIG['position_size'],
        trailing_drawdown_limit=PHASE2_CONFIG['trailing_dd_limit']
    ))])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    # Transfer learning from Phase 1
    model = load_phase1_and_transfer(PHASE2_CONFIG, env)

    if model is None:
        learning_rate = get_learning_rate(PHASE2_CONFIG)
        safe_print("\n[MODEL] Creating Phase 2 model from scratch...")
        safe_print("[MODEL] RL FIX #4: Using MaskablePPO for efficient exploration")

        # RL FIX #7: Use fixed entropy coefficient (MaskablePPO doesn't support schedules)
        ent_coef = 0.06  # FIXED: 4x higher for 9-action space exploration

        # RL FIX #4: Use MaskablePPO instead of standard PPO
        model = MaskablePPO(
            'MlpPolicy',
            env,
            learning_rate=learning_rate,
            n_steps=PHASE2_CONFIG['n_steps'],
            batch_size=PHASE2_CONFIG['batch_size'],
            n_epochs=PHASE2_CONFIG['n_epochs'],
            gamma=PHASE2_CONFIG['gamma'],
            gae_lambda=PHASE2_CONFIG['gae_lambda'],
            clip_range=PHASE2_CONFIG['clip_range'],
            ent_coef=ent_coef,  # Use schedule
            vf_coef=PHASE2_CONFIG['vf_coef'],
            max_grad_norm=PHASE2_CONFIG['max_grad_norm'],
            policy_kwargs={
                'net_arch': dict(
                    pi=PHASE2_CONFIG['policy_layers'],
                    vf=PHASE2_CONFIG['policy_layers']
                ),
                'activation_fn': torch.nn.ReLU
            },
            device=PHASE2_CONFIG['device'],
            verbose=1,
            tensorboard_log='./tensorboard_logs/phase2/'
        )

    # Callbacks
    # STRATEGY B: Aggressive early stopping to prevent overfitting with limited data
    if PHASE2_CONFIG.get('use_early_stopping', False):
        early_stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=PHASE2_CONFIG['early_stop_max_no_improvement'],
            min_evals=PHASE2_CONFIG['early_stop_min_evals'],
            verbose=1
        )
        safe_print(f"\n[TRAIN] Early stopping enabled:")
        safe_print(f"        - Stop after {PHASE2_CONFIG['early_stop_max_no_improvement']} evals with no improvement")
        safe_print(f"        - Minimum {PHASE2_CONFIG['early_stop_min_evals']} evals required")
        safe_print(f"        - Evaluation every {PHASE2_CONFIG['eval_freq']:,} timesteps")
        safe_print(f"        - Could stop as early as: {(PHASE2_CONFIG['early_stop_min_evals'] + PHASE2_CONFIG['early_stop_max_no_improvement']) * PHASE2_CONFIG['eval_freq']:,} timesteps")
    else:
        early_stop_callback = None
        safe_print("[TRAIN] Early stopping disabled")

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=PHASE2_CONFIG['eval_freq'],
        n_eval_episodes=PHASE2_CONFIG['n_eval_episodes'],
        best_model_save_path='./models/phase2/',
        log_path='./logs/phase2/',
        deterministic=True,
        render=False,
        verbose=1,
        callback_after_eval=early_stop_callback  # Pass early stopping here
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path='./models/phase2/checkpoints/',
        name_prefix='phase2',
        save_replay_buffer=False,
        save_vecnormalize=True
    )

    # NEW: KL divergence monitoring callback
    callbacks = [eval_callback, checkpoint_callback]  # Don't add early_stop separately
    
    if PHASE2_CONFIG.get('use_kl_callback', False):
        kl_callback = KLDivergenceCallback(
            target_kl=PHASE2_CONFIG['target_kl'],
            verbose=1,
            log_freq=1000
        )
        callbacks.append(kl_callback)
        safe_print("[TRAIN] KL divergence monitoring enabled")

    # Train Phase 2
    safe_print("\n" + "=" * 80)
    safe_print(f"[TRAIN] Starting Phase 2 training for {PHASE2_CONFIG['total_timesteps']:,} timesteps...")
    safe_print("=" * 80)
    safe_print("\n[TRAIN] Position management actions (RL Fix #9 - Markovian):")
    safe_print("        Action 3: Close position (manual exit)")
    safe_print("        Action 4: Tighten SL (reduce risk)")
    safe_print("        Action 5: Move SL to break-even (risk-free)")
    safe_print("        Action 6: Extend TP (ride trends)")
    safe_print("        Action 7: Enable trailing stop (explicit enable)")
    safe_print("        Action 8: Disable trailing stop (explicit disable)")
    safe_print("\n[TRAIN] Monitor progress:")
    safe_print("        - TensorBoard: tensorboard --logdir tensorboard_logs/phase2/")
    safe_print("        - Logs: logs/phase2/evaluations.npz")
    safe_print()

    import time
    start_time = time.time()

    # Use progress bar if available, otherwise disable
    use_progress_bar = PROGRESS_BAR_AVAILABLE
    if use_progress_bar:
        safe_print("[TRAIN] Progress bar enabled (tqdm + rich available)")
    else:
        safe_print("[TRAIN] Progress bar disabled (missing tqdm/rich)")
        safe_print("[TRAIN] Monitor via TensorBoard: tensorboard --logdir tensorboard_logs/phase2/")

    try:
        model.learn(
            total_timesteps=PHASE2_CONFIG['total_timesteps'],
            callback=callbacks,
            progress_bar=use_progress_bar
        )
    except KeyboardInterrupt:
        safe_print("\n[TRAIN] Training interrupted by user")
        safe_print("[TRAIN] Saving current model state...")
    except Exception as e:
        safe_print(f"\n[ERROR] Training failed: {e}")
        raise

    elapsed = time.time() - start_time

    # Save final model
    safe_print("\n[SAVE] Saving final Phase 2 model...")
    model.save('models/phase2_position_mgmt_final')
    env.save('models/phase2_vecnorm.pkl')

    safe_print("\n" + "=" * 80)
    safe_print("PHASE 2 TRAINING COMPLETE!")
    safe_print("=" * 80)
    safe_print(f"[RESULTS] Training time: {elapsed/3600:.2f} hours")
    safe_print(f"[SAVE] Model: models/phase2_position_mgmt_final.zip")
    safe_print(f"[SAVE] VecNorm: models/phase2_vecnorm.pkl")
    safe_print(f"[SAVE] Best model: models/phase2/best_model.zip")
    safe_print()
    safe_print("[SUCCESS] Two-phase training complete!")
    safe_print()
    safe_print("[NEXT STEPS]")
    safe_print("  1. Evaluate Phase 2: python3 evaluate_phase2.py")
    safe_print("  2. Compare TensorBoard: tensorboard --logdir tensorboard_logs/")
    safe_print("  3. Review position management usage in logs")
    safe_print()


if __name__ == '__main__':
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Phase 2 Training')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode with reduced timesteps (50K for quick local testing)')
    args = parser.parse_args()

    # Override config for test mode (quick local testing)
    if args.test:
        safe_print("\n" + "=" * 80)
        safe_print("TEST MODE ENABLED - Quick Local Testing")
        safe_print("=" * 80)
        PHASE2_CONFIG['total_timesteps'] = 50_000  # Small test run
        PHASE2_CONFIG['num_envs'] = 4  # Minimal for local CPU/small GPU
        PHASE2_CONFIG['eval_freq'] = 10_000  # Eval 5 times
        PHASE2_CONFIG['n_eval_episodes'] = 3  # Faster eval

        # Disable early stopping in test mode (run full 50K)
        PHASE2_CONFIG['use_early_stopping'] = False

        safe_print(f"[TEST] Timesteps:       5M -> {PHASE2_CONFIG['total_timesteps']:,} (1% for testing)")
        safe_print(f"[TEST] Parallel envs:   80 -> {PHASE2_CONFIG['num_envs']} (local machine)")
        safe_print(f"[TEST] Eval frequency:  Every {PHASE2_CONFIG['eval_freq']:,} steps")
        safe_print(f"[TEST] Early stopping:  DISABLED (test mode)")
        safe_print(f"[TEST] Expected time:   ~10-15 minutes")
        safe_print(f"[TEST] Purpose:         Verify pipeline works before full training")
        safe_print("=" * 80 + "\n")

    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)

    train_phase2()
