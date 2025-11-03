#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1: Foundational Trading Patterns Training
- Fixed SL/TP (1.5x ATR SL, 3:1 TP ratio)
- 3 actions: Hold, Buy, Sell
- Focus: Entry signal quality
- Duration: 5M timesteps (~6-8 hours on RTX 4000 Ada 20GB)

Based on: OpenAI Spinning Up PPO + Stable Baselines3
Optimized for: RunPod RTX 4000 Ada deployment
"""

import os

# Limit math/BLAS thread pools before importing numpy/torch to prevent
# pthread/resource exhaustion on constrained hosts.
try:
    _THREAD_LIMIT_INT = max(1, int(os.environ.get("TRAINER_MAX_BLAS_THREADS", "1")))
except ValueError:
    _THREAD_LIMIT_INT = 1
_THREAD_LIMIT_STR = str(_THREAD_LIMIT_INT)

for _env_var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_env_var, _THREAD_LIMIT_STR)

import sys
import glob
import torch
import numpy as np
import pandas as pd

_TORCH_THREADS_OVERRIDE = os.environ.get("PYTORCH_NUM_THREADS")
try:
    if _TORCH_THREADS_OVERRIDE is not None:
        torch.set_num_threads(max(1, int(_TORCH_THREADS_OVERRIDE)))
    else:
        torch.set_num_threads(_THREAD_LIMIT_INT)
except (TypeError, ValueError):
    # Fall back to deterministic single-thread execution if override is invalid
    torch.set_num_threads(1)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnNoModelImprovement
from environment_phase1 import TradingEnvironmentPhase1
from kl_callback import KLDivergenceCallback
from feature_engineering import add_market_regime_features
from model_utils import get_model_save_name, detect_available_markets, select_market_for_training

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


def get_effective_num_envs(requested_envs: int) -> int:
    """Determine a safe number of parallel environments for the host."""
    override = os.environ.get("TRAINER_NUM_ENVS")
    if override:
        try:
            value = max(1, int(override))
            safe_print(f"[CONFIG] TRAINER_NUM_ENVS override detected: {value}")
            return value
        except ValueError:
            safe_print(f"[WARN] Ignoring invalid TRAINER_NUM_ENVS='{override}'")

    cpu_count = os.cpu_count()
    if cpu_count:
        cpu_aligned = max(1, cpu_count)
        if requested_envs > cpu_aligned:
            safe_print(
                f"[SYSTEM] Reducing parallel envs to {cpu_aligned} (requested {requested_envs}, {cpu_count} CPU cores)"
            )
            return cpu_aligned

    return requested_envs


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

# PPO Hyperparameters (optimized for RTX 4000 Ada 20GB VRAM)
# Production config for RunPod deployment
# STRATEGY B: Data-aware training with early stopping
PHASE1_CONFIG = {
    # Training - Data-constrained budget (22,984 unique episodes)
    'total_timesteps': 2_000_000,  # 2M max - early stopping will find optimal point
    'num_envs': 80,  # INCREASED for RTX 4000 Ada 20GB: excellent parallelization

    # Network architecture - REDUCED to prevent overfitting
    'policy_layers': [512, 256, 128],  # DOWN from [1024, 512, 256, 128] - FIX #3

    # PPO parameters (from OpenAI Spinning Up)
    'learning_rate': 3e-4,
    'n_steps': 2048,  # Rollout length
    # OPTIMIZED batch size for RTX 4000 Ada
    # Calculation: 80 envs × 2048 steps = 163,840 samples per update
    # 163,840 / 512 = 320 minibatches (excellent balance)
    # Larger batch = better GPU utilization while maintaining generalization
    'batch_size': 512,  # INCREASED from 256 for 20GB GPU
    'n_epochs': 10,  # Optimization epochs per update
    'gamma': 0.99,  # Discount factor
    'gae_lambda': 0.95,  # GAE parameter
    'clip_range': 0.2,  # PPO clip range
    'ent_coef': 0.01,  # Entropy coefficient (exploration)
    'vf_coef': 0.5,  # Value function coefficient
    'max_grad_norm': 0.5,  # Gradient clipping

    # NEW: Learning rate scheduling
    'use_lr_schedule': True,
    'lr_final_fraction': 0.2,  # End at 20% of initial LR

    # NEW: Early stopping with KL monitoring
    'target_kl': 0.01,
    'use_kl_callback': True,

    # Environment parameters
    'window_size': 20,
    'initial_balance': 50000,
    'fixed_sl_multiplier': 1.5,
    'fixed_tp_ratio': 3.0,
    'position_size': 0.5,
    'trailing_dd_limit': 5000,  # Relaxed for Phase 1
    'episode_length': 390,  # NEW: 1 trading day

    # Evaluation - More frequent for better early stopping
    'eval_freq': 50_000,  # Every 50K = 40 evals max (2M / 50K)
    'n_eval_episodes': 10,  # Increased from 5 for more reliable estimates

    # Early stopping - Prevent overfitting with limited data
    'use_early_stopping': True,
    'early_stop_max_no_improvement': 5,  # Stop after 5 evals with no improvement
    'early_stop_min_evals': 3,  # Require at least 3 evals before stopping can trigger

    # Device configuration
    # SWITCHED TO GPU FOR TESTING (heavy environment bottleneck detected)
    # Heavy feature engineering (33 features) + dual data sources (minute + second-level)
    # CPU environment overhead (~80-100ms) > GPU data transfer overhead (~20-30ms)
    # GPU freed CPU to handle environment simulation better
    'device': 'cuda'  # Testing GPU for heavy environment scenarios
}


def create_lr_schedule(initial_lr, final_fraction, total_timesteps):
    """Create linear learning rate schedule."""
    def lr_schedule(progress):
        # progress goes from 0 to 1
        return initial_lr * (1 - progress * (1 - final_fraction))
    return lr_schedule

def find_data_file(market=None):
    """Find training data file with priority order.

    Args:
        market: Market identifier (e.g., 'ES', 'NQ') or None for auto-detect

    Returns:
        Path to data file
    """
    # Get project root directory (parent of src/)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(script_dir, 'data')

    # If market is specified and not GENERIC, look for market-specific file first
    if market and market != 'GENERIC':
        market_file = os.path.join(data_dir, f'{market}_D1M.csv')
        if os.path.exists(market_file):
            return market_file

    # Priority order: D1M (generic 1-minute) > old ES-specific names (backward compat)
    filenames_to_try = [
        'D1M.csv',  # New generic 1-minute data format
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


def load_data(train_split=0.7, market=None):
    """
    Load and prepare training data with proper train/val split.

    Args:
        train_split: Fraction of data to use for training (default 0.7 = 70%)
        market: Market to load data for (e.g., 'ES', 'NQ', or None for auto-detect)

    Returns:
        train_data, val_data, train_second_data, val_second_data
    """
    data_path = find_data_file(market=market)
    safe_print(f"[DATA] Loading minute-level data from {data_path}")

    # Load minute-level data
    try:
        data = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
    except Exception:
        df_tmp = pd.read_csv(data_path)
        time_col = 'timestamp' if 'timestamp' in df_tmp.columns else 'datetime'
        df_tmp[time_col] = pd.to_datetime(df_tmp[time_col])
        data = df_tmp.set_index(time_col)

    # Convert to America/New_York timezone
    if data.index.tz is None:
        safe_print("[DATA] Converting timezone: UTC -> America/New_York")
        data.index = data.index.tz_localize('UTC').tz_convert('America/New_York')
    elif str(data.index.tz) != 'America/New_York':
        safe_print(f"[DATA] Converting timezone: {data.index.tz} -> America/New_York")
        data.index = data.index.tz_convert('America/New_York')

    safe_print(f"[DATA] Loaded {len(data):,} rows")
    safe_print(f"[DATA] Full date range: {data.index.min()} to {data.index.max()}")

    # CRITICAL FIX: Chronological train/val split to prevent overfitting
    train_end_idx = int(len(data) * train_split)
    train_data = data.iloc[:train_end_idx].copy()
    val_data = data.iloc[train_end_idx:].copy()

    safe_print(f"\n[SPLIT] Train/Val Split Applied:")
    safe_print(f"[SPLIT] Train: {len(train_data):,} bars ({train_split*100:.0f}%) - {train_data.index.min()} to {train_data.index.max()}")
    safe_print(f"[SPLIT] Val:   {len(val_data):,} bars ({(1-train_split)*100:.0f}%) - {val_data.index.min()} to {val_data.index.max()}")
    safe_print(f"[SPLIT] Train stats: Close {train_data['close'].min():.2f}-{train_data['close'].max():.2f}, ATR {train_data['atr'].min():.4f}-{train_data['atr'].max():.4f}")

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
    second_data_candidates = []

    # Try market-specific file first, then generic
    if market and market != 'GENERIC':
        second_data_candidates.append(os.path.join(script_dir, 'data', f'{market}_D1S.csv'))

    # Add generic file
    second_data_candidates.append(os.path.join(script_dir, 'data', 'D1S.csv'))

    # Also check for wildcards in case market wasn't detected
    filename = os.path.basename(data_path)
    if filename.endswith('_D1M.csv') and len(filename) > len('_D1M.csv'):
        prefix = filename[:-len('_D1M.csv')]
        if prefix:
            second_data_candidates.append(os.path.join(script_dir, 'data', f"{prefix}_D1S.csv"))

    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for candidate in second_data_candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)
    second_data_candidates = unique_candidates

    second_data_path = None
    for candidate in second_data_candidates:
        if os.path.exists(candidate):
            second_data_path = candidate
            break

    if second_data_path:
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
        fallback_list = " or ".join(second_data_candidates)
        safe_print(f"[DATA] Second-level data not found (expected {fallback_list}) - continuing without it")

    return train_data, val_data, train_second_data, val_second_data


def make_env(data, second_data, env_id, config):
    """
    Create environment factory with random episode starts to prevent temporal leakage.

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

        env = TradingEnvironmentPhase1(
            data=env_data,
            window_size=config['window_size'],
            initial_balance=config['initial_balance'],
            second_data=env_second_data,  # Pass second-level data
            fixed_sl_atr_multiplier=config['fixed_sl_multiplier'],
            fixed_tp_to_sl_ratio=config['fixed_tp_ratio'],
            position_size_contracts=config['position_size'],
            trailing_drawdown_limit=config['trailing_dd_limit']
        )

        return Monitor(env)

    return _init


def train_phase1(continue_training=False, model_path=None):
    """Execute Phase 1 training.

    Args:
        continue_training: If True, load and continue training from existing model
        model_path: Path to existing model to continue from
    """
    if continue_training:
        safe_print("=" * 80)
        safe_print("PHASE 1: CONTINUING TRAINING FROM EXISTING MODEL")
        safe_print("=" * 80)
        safe_print(f"[MODEL] Loading model from: {model_path}")
    else:
        safe_print("=" * 80)
        safe_print("PHASE 1: FOUNDATIONAL TRADING PATTERNS")
        safe_print("=" * 80)

    # Detect and select market
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(script_dir, 'data')
    available_markets = detect_available_markets(data_dir)
    selected_market = select_market_for_training(available_markets, safe_print)

    if selected_market is None:
        safe_print("\n[ERROR] No market selected. Exiting training.")
        return  # User cancelled or no data

    market_name = selected_market['market']
    safe_print(f"\n[TRAINING] Market: {market_name}")

    safe_print(f"[CONFIG] Total timesteps: {PHASE1_CONFIG['total_timesteps']:,}")
    requested_envs = PHASE1_CONFIG['num_envs']
    num_envs = get_effective_num_envs(requested_envs)
    safe_print(f"[CONFIG] Parallel envs (requested): {requested_envs}")
    if num_envs != requested_envs:
        safe_print(f"[CONFIG] Parallel envs (effective): {num_envs} (adjusted for host limits)")
    else:
        safe_print(f"[CONFIG] Parallel envs (effective): {num_envs}")
    safe_print(f"[SYSTEM] BLAS threads per process: {os.environ.get('OPENBLAS_NUM_THREADS', 'unknown')}")
    safe_print(f"[CONFIG] Network: {PHASE1_CONFIG['policy_layers']}")
    safe_print(f"[CONFIG] Device: {PHASE1_CONFIG['device']}")
    safe_print(f"[CONFIG] Fixed SL: {PHASE1_CONFIG['fixed_sl_multiplier']}x ATR")
    safe_print(f"[CONFIG] Fixed TP: {PHASE1_CONFIG['fixed_tp_ratio']}x SL")
    safe_print("")

    # Create directories
    os.makedirs('models/phase1', exist_ok=True)
    os.makedirs('models/phase1/checkpoints', exist_ok=True)
    os.makedirs('logs/phase1', exist_ok=True)
    os.makedirs('tensorboard_logs/phase1', exist_ok=True)

    # Load data with train/val split
    train_data, val_data, train_second_data, val_second_data = load_data(train_split=0.7, market=market_name)

    # Create vectorized training environments (use TRAIN data only)
    safe_print(f"\n[ENV] Creating {num_envs} parallel TRAINING environments...")
    env_fns = [make_env(train_data, train_second_data, i, PHASE1_CONFIG) for i in range(num_envs)]

    if num_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    # Add normalization (critical for PPO stability)
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )

    safe_print("[ENV] Training environments created with VecNormalize")

    # Create evaluation environment (use VAL data only - CRITICAL FIX)
    safe_print("[EVAL] Creating VALIDATION environment (unseen data)...")
    eval_env = DummyVecEnv([lambda: Monitor(TradingEnvironmentPhase1(
        data=val_data,  # CHANGED: Use validation data
        window_size=PHASE1_CONFIG['window_size'],
        initial_balance=PHASE1_CONFIG['initial_balance'],
        second_data=val_second_data,  # CHANGED: Use val second-level data
        fixed_sl_atr_multiplier=PHASE1_CONFIG['fixed_sl_multiplier'],
        fixed_tp_to_sl_ratio=PHASE1_CONFIG['fixed_tp_ratio'],
        position_size_contracts=PHASE1_CONFIG['position_size'],
        trailing_drawdown_limit=PHASE1_CONFIG['trailing_dd_limit']
    ))])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Create learning rate schedule if enabled
    learning_rate = PHASE1_CONFIG['learning_rate']
    if PHASE1_CONFIG.get('use_lr_schedule', False):
        learning_rate = create_lr_schedule(
            PHASE1_CONFIG['learning_rate'],
            PHASE1_CONFIG['lr_final_fraction'],
            PHASE1_CONFIG['total_timesteps']
        )
        safe_print("[TRAIN] Using linear learning rate schedule")
    
    # Create or load PPO model
    if continue_training and model_path:
        safe_print("\n[MODEL] Loading existing PPO model...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        # Load the existing model
        model = PPO.load(model_path, device=PHASE1_CONFIG['device'])

        # Update the environment (important for VecNormalize)
        model.set_env(env)

        # Update tensorboard log directory
        model.tensorboard_log = './tensorboard_logs/phase1/'

        safe_print(f"[MODEL] Model loaded successfully from {model_path}")
        safe_print(f"[MODEL] Current timesteps: {model.num_timesteps:,}")
        safe_print(f"[MODEL] Will train for additional {PHASE1_CONFIG['total_timesteps']:,} timesteps")
    else:
        safe_print("\n[MODEL] Creating NEW PPO model...")
        safe_print(f"[MODEL] Policy network: {PHASE1_CONFIG['policy_layers']}")
        safe_print(f"[MODEL] Initial learning rate: {PHASE1_CONFIG['learning_rate']}")
        safe_print(f"[MODEL] Batch size: {PHASE1_CONFIG['batch_size']} (INCREASED)")
        safe_print(f"[MODEL] PPO clip range: {PHASE1_CONFIG['clip_range']}")

        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=learning_rate,
            n_steps=PHASE1_CONFIG['n_steps'],
            batch_size=PHASE1_CONFIG['batch_size'],
            n_epochs=PHASE1_CONFIG['n_epochs'],
            gamma=PHASE1_CONFIG['gamma'],
            gae_lambda=PHASE1_CONFIG['gae_lambda'],
            clip_range=PHASE1_CONFIG['clip_range'],
            ent_coef=PHASE1_CONFIG['ent_coef'],
            vf_coef=PHASE1_CONFIG['vf_coef'],
            max_grad_norm=PHASE1_CONFIG['max_grad_norm'],
            policy_kwargs={
                'net_arch': dict(
                    pi=PHASE1_CONFIG['policy_layers'],
                    vf=PHASE1_CONFIG['policy_layers']
                ),
                'activation_fn': torch.nn.ReLU
            },
            device=PHASE1_CONFIG['device'],
            verbose=1,
            tensorboard_log='./tensorboard_logs/phase1/'
        )

        safe_print("[MODEL] PPO model created")

    # Callbacks
    # STRATEGY B: Early stopping to prevent overfitting with limited data
    callbacks_list = []

    if PHASE1_CONFIG.get('use_early_stopping', False):
        early_stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=PHASE1_CONFIG['early_stop_max_no_improvement'],
            min_evals=PHASE1_CONFIG['early_stop_min_evals'],
            verbose=1
        )
        safe_print(f"[TRAIN] Early stopping enabled:")
        safe_print(f"        - Stop after {PHASE1_CONFIG['early_stop_max_no_improvement']} evals with no improvement")
        safe_print(f"        - Minimum {PHASE1_CONFIG['early_stop_min_evals']} evals required")
        safe_print(f"        - Evaluation every {PHASE1_CONFIG['eval_freq']:,} timesteps")
        safe_print(f"        - Could stop as early as: {(PHASE1_CONFIG['early_stop_min_evals'] + PHASE1_CONFIG['early_stop_max_no_improvement']) * PHASE1_CONFIG['eval_freq']:,} timesteps")
    else:
        early_stop_callback = None
        safe_print("[TRAIN] Early stopping disabled")

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=PHASE1_CONFIG['eval_freq'],
        n_eval_episodes=PHASE1_CONFIG['n_eval_episodes'],
        best_model_save_path='./models/phase1/',
        log_path='./logs/phase1/',
        deterministic=True,
        render=False,
        verbose=1,
        callback_after_eval=early_stop_callback  # Add early stopping
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path='./models/phase1/checkpoints/',
        name_prefix='phase1',
        save_replay_buffer=False,
        save_vecnormalize=True
    )

    # Build callbacks list (early_stop is already in EvalCallback via callback_after_eval)
    callbacks = [eval_callback, checkpoint_callback]
    
    if PHASE1_CONFIG.get('use_kl_callback', False):
        kl_callback = KLDivergenceCallback(
            target_kl=PHASE1_CONFIG['target_kl'],
            verbose=1,
            log_freq=1000
        )
        callbacks.append(kl_callback)
        safe_print("[TRAIN] KL divergence monitoring enabled")

    # Train
    safe_print("\n" + "=" * 80)
    safe_print(f"[TRAIN] Starting Phase 1 training for {PHASE1_CONFIG['total_timesteps']:,} timesteps...")
    safe_print("=" * 80)
    safe_print("\n[TRAIN] Monitor progress:")
    safe_print("        - TensorBoard: tensorboard --logdir tensorboard_logs/phase1/")
    safe_print("        - Logs: logs/phase1/evaluations.npz")
    safe_print("        - Checkpoints: models/phase1/checkpoints/")
    safe_print("")

    import time
    start_time = time.time()

    # Use progress bar if available, otherwise disable
    use_progress_bar = PROGRESS_BAR_AVAILABLE
    if use_progress_bar:
        safe_print("[TRAIN] Progress bar enabled (tqdm + rich available)")
    else:
        safe_print("[TRAIN] Progress bar disabled (missing tqdm/rich)")
        safe_print("[TRAIN] Monitor via TensorBoard: tensorboard --logdir tensorboard_logs/phase1/")

    try:
        model.learn(
            total_timesteps=PHASE1_CONFIG['total_timesteps'],
            callback=callbacks,
            progress_bar=use_progress_bar,
            reset_num_timesteps=not continue_training  # Don't reset if continuing
        )
    except KeyboardInterrupt:
        safe_print("\n[TRAIN] Training interrupted by user")
        safe_print("[TRAIN] Saving current model state...")
    except Exception as e:
        safe_print(f"\n[ERROR] Training failed: {e}")
        raise

    elapsed = time.time() - start_time

    # Save final model with user-chosen name
    safe_print("\n[SAVE] Saving final Phase 1 model...")

    # Determine default name
    if continue_training:
        # Extract base name from loaded model path
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        default_name = f"{base_name}_continued"
    else:
        default_name = "phase1_foundational_final"

    # Get save name from user (interactive mode only)
    if sys.stdin.isatty():  # Check if running interactively
        try:
            save_name = get_model_save_name(default_name)
        except:
            # Fallback if there's an issue with the prompt
            save_name = default_name
            safe_print(f"[SAVE] Using default name: {save_name}")
    else:
        # Non-interactive mode (e.g., called from menu script)
        save_name = default_name
        safe_print(f"[SAVE] Non-interactive mode - using default name: {save_name}")

    model_save_path = f'models/{save_name}'
    vecnorm_save_path = f'models/{save_name}_vecnorm.pkl'

    model.save(model_save_path)
    env.save(vecnorm_save_path)

    safe_print("\n" + "=" * 80)
    safe_print("PHASE 1 TRAINING COMPLETE!")
    safe_print("=" * 80)
    safe_print(f"[RESULTS] Training time: {elapsed/3600:.2f} hours")
    safe_print(f"[RESULTS] Total timesteps: {model.num_timesteps:,}")
    safe_print(f"[SAVE] Model: {model_save_path}.zip")
    safe_print(f"[SAVE] VecNorm: {vecnorm_save_path}")
    safe_print(f"[SAVE] Best model: models/phase1/best_model.zip")
    safe_print("")
    safe_print("[NEXT STEPS]")
    safe_print("  1. Review training logs: logs/phase1/evaluations.npz")
    safe_print("  2. Check TensorBoard: tensorboard --logdir tensorboard_logs/phase1/")
    safe_print("  3. Run Phase 2: python3 train_phase2.py")
    safe_print("")


if __name__ == '__main__':
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Phase 1 Training')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode with reduced timesteps (30K for quick local testing)')
    parser.add_argument('--continue', dest='continue_training', action='store_true',
                       help='Continue training from an existing model')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to the model to continue training from')
    args = parser.parse_args()

    # Override config for test mode (quick local testing)
    if args.test:
        safe_print("\n" + "=" * 80)
        safe_print("TEST MODE ENABLED - Quick Local Testing")
        safe_print("=" * 80)
        PHASE1_CONFIG['total_timesteps'] = 30_000  # Small test run
        PHASE1_CONFIG['num_envs'] = 4  # Minimal for local CPU/small GPU
        PHASE1_CONFIG['eval_freq'] = 10_000  # Eval 3 times (10K, 20K, 30K)
        PHASE1_CONFIG['n_eval_episodes'] = 3  # Faster eval

        # Disable early stopping in test mode (run full 30K)
        PHASE1_CONFIG['use_early_stopping'] = False

        safe_print(f"[TEST] Timesteps:       2M -> {PHASE1_CONFIG['total_timesteps']:,} (1.5% for testing)")
        safe_print(f"[TEST] Parallel envs:   80 -> {PHASE1_CONFIG['num_envs']} (local machine)")
        safe_print(f"[TEST] Eval frequency:  Every {PHASE1_CONFIG['eval_freq']:,} steps")
        safe_print(f"[TEST] Early stopping:  DISABLED (test mode)")
        safe_print(f"[TEST] Expected time:   ~5-10 minutes")
        safe_print(f"[TEST] Purpose:         Verify pipeline works before full training")
        safe_print("=" * 80 + "\n")

    # Validate continuation arguments
    if args.continue_training and not args.model_path:
        safe_print("\n[ERROR] --continue requires --model-path to be specified")
        sys.exit(1)

    if args.continue_training:
        safe_print(f"\n[INFO] Continuation mode enabled")
        safe_print(f"[INFO] Will load model from: {args.model_path}")

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    train_phase1(continue_training=args.continue_training, model_path=args.model_path)
