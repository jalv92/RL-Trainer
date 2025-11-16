#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3: Hybrid RL + LLM Trading Agent Training

Combines:
- RL Agent (PPO): Fast pattern recognition from 5M training timesteps  
- LLM Advisor (Phi-3-mini): Context-aware reasoning about market conditions
- Decision Fusion: Intelligent combination of both for robust decisions

Key Features:
- 261D observations (vs 228D in Phase 2)
- LLM reasoning integration
- Risk-aware decision fusion
- Selective querying for performance

Based on: train_phase2.py with LLM integration
"""

import os
import platform
import shutil
from pathlib import Path
from typing import Optional


def _detect_thread_cap() -> int:
    """Auto-detect a reasonable BLAS thread cap based on CPU size."""
    try:
        # Prefer affinity-aware measurement when available.
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count() or 2
    return max(1, cpu_count - 1)


# Limit math/BLAS thread pools before heavy numerical imports. Allow override via env var.
try:
    _THREAD_LIMIT_INT = max(
        1,
        int(os.environ.get("TRAINER_MAX_BLAS_THREADS", str(_detect_thread_cap()))),
    )
except ValueError:
    _THREAD_LIMIT_INT = _detect_thread_cap()
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
import yaml
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
    torch.set_num_threads(1)
TORCH_THREADS_EFFECTIVE = torch.get_num_threads()

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

# Phase 3 imports
from environment_phase3_llm import TradingEnvironmentPhase3LLM
from llm_reasoning import LLMReasoningModule
from hybrid_agent import HybridTradingAgent
from llm_callback import LLMMonitoringCallback
from kl_callback import KLDivergenceCallback
from feature_engineering import add_market_regime_features
from model_utils import detect_models_in_folder, detect_available_markets, select_market_for_training
from market_specs import get_market_spec
from metadata_utils import read_metadata, write_metadata
from action_mask_utils import get_action_masks, ensure_vecenv_action_masks
from self_correcting_init import init_self_correcting_system

# CHECKPOINT MANAGEMENT (replaces old SafeCheckpointCallback)
from checkpoint_manager import DynamicCheckpointManager, MetricTrackingEvalCallback, EvalMetricHook
from checkpoint_retention import CheckpointRetentionManager
from stable_baselines3.common.callbacks import CallbackList

# Set UTF-8 encoding for Windows compatibility
if os.name == 'nt':
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
        replacements = {
            '✓': '[OK]', '✅': '[OK]', '✗': '[X]', '❌': '[X]', '→': '->',
            '⚠': '[WARN]', '⚠️': '[WARN]', '—': '-', '–': '-', '’': "'", '“': '"', '”': '"',
        }
        ascii_message = message
        for src, target in replacements.items():
            ascii_message = ascii_message.replace(src, target)
        ascii_message = ascii_message.encode('ascii', errors='ignore').decode('ascii')
        print(ascii_message)


def get_effective_num_envs(requested_envs: int) -> int:
    """Determine a safe number of parallel environments for this host."""
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
    
    effective = max(1, min(requested_envs, cpu_count - 1))
    return effective


def supports_subproc_vec_env() -> bool:
    """
    Detect whether SubprocVecEnv is safe on this host.

    Windows / WSL builds often struggle with forked environments, so we disable
    it automatically there while keeping it enabled for native Linux pods.
    """
    if os.name == 'nt':
        return False
    release = platform.release().lower()
    return 'microsoft' not in release and 'wsl' not in release


def compute_env_start_index(data_length: int, config: dict, rank: int) -> Optional[int]:
    """
    Compute deterministic start index for an environment when randomization is disabled.

    Returns:
        int or None if not enough data.
    """
    min_start = config.get('window_size', 20)
    min_episode_bars = max(config.get('min_episode_bars', 1500), 10)
    max_start = data_length - min_episode_bars
    if max_start <= min_start:
        return max(min_start, 0)

    if config.get('deterministic_env_offsets', False):
        n_envs = max(1, config.get('n_envs', 1))
        spacing = max(1, (max_start - min_start) // n_envs or 1)
        start = min_start + spacing * rank
        return min(max(start, min_start), max_start)

    seed = (config.get('start_offset_seed', 0) + rank) or None
    rng = np.random.default_rng(seed)
    return int(rng.integers(min_start, max_start + 1))


def _log_disk_warning(label: str, path: str, exc: Exception) -> None:
    """Log disk usage when checkpoint/eval saving fails."""
    safe_print(f"[{label}] Failed to save model: {exc}")
    try:
        target = Path(path)
        disk_root = target if target.is_dir() else target.parent
        usage = shutil.disk_usage(disk_root)
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        safe_print(f"[{label}] Disk usage at {disk_root}: {free_gb:.2f} GB free / {total_gb:.2f} GB total")
    except Exception:
        pass


class SafeEvalCallback(EvalCallback):
    """Eval callback that tolerates disk full errors."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._disable_saves = False

    def _save_model(self) -> None:
        if self._disable_saves:
            return
        try:
            super()._save_model()
        except (OSError, RuntimeError) as exc:
            if "No space left" in str(exc):
                _log_disk_warning("EVAL", self.best_model_save_path, exc)
                self._disable_saves = True
            else:
                raise


# DEPRECATED: SafeCheckpointCallback replaced by DynamicCheckpointManager
# The new checkpoint manager includes disk-full protection plus:
# - Adaptive save intervals (grows as training stabilizes)
# - Event-driven triggers (best metric, phase boundaries)
# - Rich metadata with metrics embedded in filenames
# - Automatic retention cleanup
# Import from checkpoint_manager for compatibility:
# from checkpoint_manager import DynamicCheckpointManager as SafeCheckpointCallback


# Phase 3 Configuration
PHASE3_CONFIG = {
    'total_timesteps': 5_000_000,  # 5M timesteps for hybrid training
    'learning_rate': 3e-4,  # Slightly lower for stability with LLM
    'n_steps': 2048,  # PPO batch size
    'batch_size': 256,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,  # Slightly higher for exploration with LLM
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'target_kl': 0.02,  # KL divergence target
    'tensorboard_log': "./tensorboard_logs/phase3_hybrid",
    'policy_kwargs': {
        'net_arch': dict(
            pi=[512, 512, 256],  # Larger network for 261D input
            vf=[512, 512, 256]
        ),
        'activation_fn': torch.nn.ReLU,
    },
    'device': 'auto',  # Allow explicit override (e.g., 'cuda' or 'cpu')
    # Environment
    'window_size': 20,  # Same as Phase 2
    'second_data_enabled': False,
    # Parallel environments
    'n_envs': 8,  # Higher default to better utilize large pods (auto-capped per host)
    'vec_env_cls': 'subproc',  # 'subproc' or 'dummy' (auto-fallback on Windows/WSL)
    'min_episode_bars': 1500,  # Minimum remaining bars per episode after start offset
    'randomize_start_offsets': True,  # Randomize episode starts each reset
    'deterministic_env_offsets': False,  # If True, spread envs evenly instead of random
    'start_offset_seed': 42,  # Base seed for deterministic offsets
    # Callbacks
    'eval_freq': 10000,
    'save_freq': 50000,
    'llm_log_freq': 1000,  # LLM-specific logging frequency
    # Model paths
    'model_save_path': "./models/phase3_hybrid",
    'vecnormalize_path': "./models/vecnormalize/phase3_hybrid.pkl",
    # Transfer learning (CRITICAL FIX #1)
    'phase2_model_path': 'models/phase2_position_mgmt_final.zip',
    'phase2_vecnorm_path': 'models/vecnormalize/phase2.pkl',
    # LLM-specific
    'llm_config_path': "./config/llm_config.yaml",
    'use_llm_features': True,  # Enable 261D observations

    # Adapter-specific (NEW - for Phase 2 → Phase 3 transfer learning)
    'freeze_phase2_initially': True,  # Freeze Phase 2 weights during adapter warmup
    'adapter_warmup_steps': 100_000,  # Train only adapter for first 100K steps
    'unfreeze_after_warmup': True,    # Unfreeze Phase 2 after warmup for full training
}


def normalize_device(device):
    """Convert SB3 device strings/objects into torch.device."""
    if device is None:
        return torch.device('cpu')
    if isinstance(device, str):
        return torch.device(device)
    return device


def move_policy_to_device(policy, device):
    """Move policy (and cached attribute) to the specified device."""
    if policy is None:
        return
    torch_device = normalize_device(device)
    policy.to(torch_device)


def summarize_policy_devices(policy):
    """Return count of parameters per device for debugging."""
    if policy is None:
        return {}
    counts = {}
    for _, param in policy.named_parameters():
        dev = str(param.device)
        counts[dev] = counts.get(dev, 0) + 1
    return counts


def log_device_summary(model, hybrid_agent=None, label=""):
    """Log device placement for policy, adapter, and hybrid components."""
    policy = getattr(model, 'policy', None)
    if policy is None:
        return

    counts = summarize_policy_devices(policy)
    safe_print(f"[DEVICE] Policy parameter devices{f' ({label})' if label else ''}: {counts or 'n/a'}")
    model_device = getattr(model, 'device', 'unknown')
    safe_print(f"[DEVICE] Model device{f' ({label})' if label else ''}: {model_device}")

    if hasattr(policy, 'adapter'):
        safe_print(f"[DEVICE] Adapter module device: {policy.adapter.weight.device}")

    if len(counts) > 1:
        safe_print("[DEVICE] [WARN] Policy parameters span multiple devices. "
                   "This can trigger runtime errors during training.")

    if hybrid_agent is not None and hasattr(hybrid_agent, 'fusion_network') and hybrid_agent.fusion_network is not None:
        fusion_device = next(hybrid_agent.fusion_network.parameters()).device
        safe_print(f"[DEVICE] Fusion network device: {fusion_device}")


def create_phase3_env(env_data, second_data=None, market_name=None, config=None, rank=0, hybrid_agent=None):
    """
    Create Phase 3 training environment.
    
    Args:
        env_data: Training data DataFrame
        second_data: Optional second-level data
        market_name: Market name for specifications
        config: Configuration dictionary
        rank: Environment rank (for vectorized envs)
        hybrid_agent: HybridTradingAgent instance for outcome tracking
    
    Returns:
        TradingEnvironmentPhase3LLM instance
    """
    if config is None:
        config = PHASE3_CONFIG
    
    # Get market specification
    market_spec = get_market_spec(market_name) if market_name else None
    
    # Determine deterministic start index (used when randomization disabled)
    deterministic_start = compute_env_start_index(len(env_data), config, rank)

    # Create environment
    env = TradingEnvironmentPhase3LLM(
        data=env_data,
        use_llm_features=config['use_llm_features'],  # Enable 261D observations
        initial_balance=50000,
        window_size=config['window_size'],
        second_data=second_data,
        market_spec=market_spec,
        commission_override=None,  # Use market spec default
        initial_sl_multiplier=1.5,
        initial_tp_ratio=3.0,
        position_size_contracts=1.0,  # Full size for Apex
        trailing_drawdown_limit=2500,  # Apex rules
        tighten_sl_step=0.5,
        extend_tp_step=1.0,
        trailing_activation_profit=1.0,
        hybrid_agent=hybrid_agent,  # PHASE 1 & 2: For outcome tracking
        start_index=deterministic_start,
        randomize_start_offsets=config.get('randomize_start_offsets', True),
        min_episode_bars=config.get('min_episode_bars', 1500)
    )

    # Validation: Ensure environment has randomization support
    assert hasattr(env, 'randomize_start_offsets'), \
        "Environment missing randomization support"
    assert hasattr(env, 'min_episode_bars'), \
        "Environment missing min_episode_bars attribute"
    assert hasattr(env, '_determine_episode_start'), \
        "Environment missing _determine_episode_start method"

    # Wrap with Monitor first for logging/episode stats
    env = Monitor(env)

    # Ensure ActionMasker is the outermost layer so MaskablePPO (and Gym)
    # can access action_masks without deprecated attribute hops.
    return ActionMasker(env, lambda env: get_action_masks(env))


def _unwrap_gym_env(env):
    """Recursively unwrap gym-style wrappers to reach the base environment."""
    current = env
    depth = 0
    while hasattr(current, 'env') and depth < 20:
        current = current.env
        depth += 1
    return current


def _extract_vec_envs(vec_env):
    """
    Retrieve the list of individual environments from a VecEnv wrapper.

    Works for DummyVecEnv (possibly wrapped in VecNormalize). Returns None for
    SubprocVecEnv where direct access is unavailable.
    """
    current = vec_env
    depth = 0
    while hasattr(current, 'venv') and depth < 20:
        current = current.venv
        depth += 1
    return getattr(current, 'envs', None)


def register_training_envs(vec_env, expected_envs, hybrid_agent=None) -> bool:
    """
    Register the actual gym environments with the hybrid policy so it can read
    live position and market state instead of falling back to defaults.
    """
    envs = _extract_vec_envs(vec_env)
    if not envs:
        safe_print("[ENV] [WARN] Unable to access underlying envs for registry "
                   "(likely SubprocVecEnv). Hybrid policy will use fallback state.")
        return False

    try:
        from .hybrid_policy import register_environment
    except ImportError:
        from hybrid_policy import register_environment

    registered = 0
    for idx, env in enumerate(envs):
        base_env = _unwrap_gym_env(env)
        if base_env is None:
            continue
        # Ensure the environment keeps a reference to the live hybrid agent
        if hybrid_agent is not None and getattr(base_env, 'hybrid_agent', None) is None:
            base_env.hybrid_agent = hybrid_agent
        register_environment(idx, base_env)
        registered += 1

    safe_print(f"[ENV] Registered {registered} training environments for hybrid policy state access")
    if registered < expected_envs:
        safe_print(f"[ENV] [WARN] Expected {expected_envs} envs but only registered {registered}.")
    return registered > 0


def load_data_for_training(market_name, data_path_pattern, use_second_data=False):
    """
    Load and prepare training data.
    
    Args:
        market_name: Market name (e.g., 'NQ', 'ES')
        data_path_pattern: Path pattern for data files
        use_second_data: Whether to load second-level data
    
    Returns:
        Tuple of (env_data, second_data)
    """
    safe_print(f"[DATA] Loading data for {market_name}...")
    
    # Find data files
    data_files = glob.glob(data_path_pattern)
    if not data_files:
        raise ValueError(f"No data files found matching pattern: {data_path_pattern}")
    
    # Load and combine data
    all_data = []
    for file in sorted(data_files):
        safe_print(f"[DATA] Loading {file}...")
        df = pd.read_csv(file, index_col=0, parse_dates=True)
        
        # Ensure timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
        
        all_data.append(df)
    
    # Combine all data
    env_data = pd.concat(all_data, axis=0)
    env_data.sort_index(inplace=True)
    
    # Add market regime features (including LLM features)
    safe_print("[DATA] Adding market regime features...")
    env_data = add_market_regime_features(env_data)
    
    # CRITICAL FIX #4: Validate LLM features (fail-fast, not just warning)
    required_features = ['sma_50', 'sma_200', 'rsi_15min', 'rsi_60min',
                        'volume_ratio_5min', 'support_20', 'resistance_20']
    missing = [f for f in required_features if f not in env_data.columns]
    if missing:
        raise ValueError(
            f"Missing required LLM features: {missing}\n"
            f"Phase 3 requires these features for 261D observations.\n"
            f"Run feature_engineering.py or update_training_data.py to generate them."
        )
    else:
        safe_print(f"[OK] All required LLM features present")
    
    safe_print(f"[DATA] Final dataset: {len(env_data)} rows, {len(env_data.columns)} features")
    safe_print(f"[DATA] Date range: {env_data.index[0]} to {env_data.index[-1]}")
    
    # Second data (optional)
    second_data = None
    if use_second_data:
        safe_print("[DATA] Loading second-level data...")
        # Add second data loading logic if needed
        pass
    
    return env_data, second_data


def load_phase2_and_transfer(config, env):
    """
    Load Phase 2 model for transfer learning to Phase 3 using a robust
    dummy environment to prevent loading errors.
    """
    phase2_path = config.get('phase2_model_path', 'models/phase2_position_mgmt_final.zip')
    if not os.path.exists(phase2_path):
        safe_print(f"\n[INFO] Configured Phase 2 model not found at {phase2_path}")
        safe_print("[INFO] Auto-detecting newest Phase 2 model...")
        phase2_models = detect_models_in_folder(phase='phase2')
        if not phase2_models:
            safe_print("[WARNING] No Phase 2 models found. Cannot perform transfer learning.")
            return None
        phase2_path = phase2_models[0]['path']
        safe_print(f"[INFO] Using newest: {phase2_models[0]['name']}")

    safe_print(f"\n[TRANSFER] Loading Phase 2 model from {phase2_path}")
    try:
        from stable_baselines3.common.save_util import load_from_zip_file
        from stable_baselines3.common.vec_env import DummyVecEnv
        import gymnasium as gym

        data, _, _ = load_from_zip_file(phase2_path)
        phase2_obs_space = data['observation_space']
        phase2_action_space = data['action_space']

        class _DummyGymEnv(gym.Env):
            def __init__(self, obs_space, act_space):
                self.observation_space = obs_space
                self.action_space = act_space
            def reset(self, seed=None): return self.observation_space.sample(), {}
            def step(self, action): return self.observation_space.sample(), 0.0, False, False, {}

        dummy_vec_env = DummyVecEnv([lambda: _DummyGymEnv(phase2_obs_space, phase2_action_space)] * env.num_envs)
        
        phase2_model = MaskablePPO.load(phase2_path, env=dummy_vec_env, device=config.get('device', 'auto'))
        safe_print("[TRANSFER] [OK] Phase 2 model loaded successfully using a temporary environment.")
        return phase2_model
    except Exception as e:
        safe_print(f"[TRANSFER] [ERROR] Failed to load Phase 2 model: {e}")
        import traceback
        traceback.print_exc()
        return None


def setup_hybrid_model(env, hybrid_agent, config=None, load_path=None, base_model=None):
    """
    Setup PPO model with hybrid policy for Phase 3, including the critical
    fix of recreating the rollout buffer for the new observation space.
    """
    if config is None: config = PHASE3_CONFIG
    from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
    try:
        from hybrid_policy_with_adapter import HybridAgentPolicyWithAdapter
    except ImportError:
        from .hybrid_policy_with_adapter import HybridAgentPolicyWithAdapter

    if base_model is not None: # This is the transfer learning path
        safe_print("\n[ADAPTER] Wrapping Phase 2 model with adapter policy...")
        
        phase2_net_arch = base_model.policy.net_arch
        safe_print(f"[ADAPTER] Detected Phase 2 architecture: {phase2_net_arch}")

        policy_kwargs = {
            'net_arch': phase2_net_arch,
            'activation_fn': base_model.policy.activation_fn,
            'hybrid_agent': hybrid_agent,
            'base_obs_dim': 228
        }
        
        phase2_state_dict = base_model.policy.state_dict()

        base_model.policy = HybridAgentPolicyWithAdapter(
            observation_space=env.observation_space,
            action_space=base_model.action_space,
            lr_schedule=base_model.lr_schedule,
            **policy_kwargs
        )
        move_policy_to_device(base_model.policy, getattr(base_model, 'device', 'cpu'))
        base_model.policy.load_state_dict(phase2_state_dict, strict=False)
        
        if config.get('freeze_phase2_initially', False):
            safe_print("\n[ADAPTER] Freezing Phase 2 weights for adapter warmup...")
            frozen_count = 0
            for name, param in base_model.policy.named_parameters():
                if 'adapter' not in name:
                    param.requires_grad = False
                    frozen_count += 1
            safe_print(f"[ADAPTER] Frozen {frozen_count} Phase 2 parameters.")

        safe_print("\n[ADAPTER] ✅ Transfer learning complete!")
        
        # CRITICAL FIX: The loaded model has a buffer sized for 228D observations.
        # We MUST replace it with a new buffer correctly sized for the 261D env.
        safe_print("[FIX] Recreating rollout buffer to match Phase 3 observation space (261D)...")
        base_model.env = env
        base_model.observation_space = env.observation_space # Ensure model's space is updated
        
        base_model.rollout_buffer = MaskableRolloutBuffer(
            base_model.n_steps,
            env.observation_space,
            env.action_space,
            base_model.device,
            gamma=base_model.gamma,
            gae_lambda=base_model.gae_lambda,
            n_envs=base_model.n_envs,
        )
        safe_print("[OK] Rollout buffer correctly sized for Phase 3.")
        return base_model

    if load_path and os.path.exists(load_path):
        # Logic for continuing a Phase 3 training run
        safe_print(f"[MODEL] Loading existing Phase 3 model from {load_path}...")
        model = MaskablePPO.load(
            load_path,
            env=env,
            tensorboard_log=config['tensorboard_log'],
            device=config.get('device', 'auto')
        )
        if hasattr(model.policy, 'hybrid_agent'):
            model.policy.hybrid_agent = hybrid_agent
        move_policy_to_device(model.policy, getattr(model, 'device', 'cpu'))
        return model
    else:
        # Logic for creating a new Phase 3 model from scratch
        safe_print("[MODEL] Creating new Phase 3 model from scratch...")
        policy_kwargs = config['policy_kwargs'].copy()
        policy_kwargs['hybrid_agent'] = hybrid_agent
        policy_kwargs['base_obs_dim'] = 228

        model = MaskablePPO(
            policy=HybridAgentPolicyWithAdapter,
            env=env,
            learning_rate=config['learning_rate'],
            n_steps=config['n_steps'],
            batch_size=config['batch_size'],
            n_epochs=config['n_epochs'],
            gamma=config['gamma'],
            gae_lambda=config['gae_lambda'],
            clip_range=config['clip_range'],
            ent_coef=config['ent_coef'],
            vf_coef=config['vf_coef'],
            max_grad_norm=config['max_grad_norm'],
            target_kl=config['target_kl'],
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=config['tensorboard_log'],
            device=config.get('device', 'auto')
        )
        move_policy_to_device(model.policy, getattr(model, 'device', 'cpu'))
        return model


def setup_model(env, config=None, load_path=None):
    """
    Setup PPO model for Phase 3 training (legacy, kept for backward compatibility).
    
    Args:
        env: Training environment
        config: Configuration dictionary
        load_path: Optional path to load existing model

    Returns:
        MaskablePPO model instance
    """
    if config is None:
        config = PHASE3_CONFIG

    if load_path and os.path.exists(load_path):
        safe_print(f"[MODEL] Loading existing model from {load_path}...")
        model = MaskablePPO.load(
            load_path,
            env=env,
            tensorboard_log=config['tensorboard_log'],
            print_system_info=False,
            device=config.get('device', 'auto')
        )
        safe_print("[OK] Model loaded successfully")
    else:
        safe_print("[MODEL] Creating new Phase 3 model...")

        # Check observation space
        obs_shape = env.observation_space.shape
        safe_print(f"[MODEL] Observation space: {obs_shape}")

        # CRITICAL FIX #3: Validate observation dimensions
        if obs_shape[0] != 261:
            raise ValueError(
                f"Phase 3 requires 261D observations, but environment provides {obs_shape[0]}D. "
                f"Expected: 228D (Phase 2 base) + 33D (LLM features) = 261D. "
                f"Check that use_llm_features=True in environment creation."
            )

        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=config['learning_rate'],
            n_steps=config['n_steps'],
            batch_size=config['batch_size'],
            n_epochs=config['n_epochs'],
            gamma=config['gamma'],
            gae_lambda=config['gae_lambda'],
            clip_range=config['clip_range'],
            ent_coef=config['ent_coef'],
            vf_coef=config['vf_coef'],
            max_grad_norm=config['max_grad_norm'],
            target_kl=config['target_kl'],
            policy_kwargs=config['policy_kwargs'],
            verbose=1,
            tensorboard_log=config['tensorboard_log'],
            device=config.get('device', 'auto')
        )

        safe_print("[OK] Model created successfully")

    move_policy_to_device(model.policy, getattr(model, 'device', 'cpu'))
    return model


def create_callbacks(model, eval_env, hybrid_agent, market_name, config=None):
    """
    Create training callbacks.
    
    Args:
        model: Training model
        eval_env: Evaluation environment
        hybrid_agent: HybridTradingAgent instance
        market_name: Market symbol being trained
        config: Configuration dictionary
    
    Returns:
        List of callbacks
    """
    if config is None:
        config = PHASE3_CONFIG
    
    callbacks = []

    # Initialize self-correcting components shared across phases
    registry, forecaster, metric_tracker, policy_controller, corrective_manager = init_self_correcting_system(
        market=market_name,
        phase=3,
        config_path='config/checkpoint_config.yaml',
        verbose=True
    )
    metric_tracker.set_seed(42)  # Match training seed
    metric_hook = EvalMetricHook(metric_tracker)

    # Evaluation callback with metric hook
    # Note: Phase 3 doesn't use early stopping, so only metric_hook is needed
    eval_callback = SafeEvalCallback(
        eval_env,
        best_model_save_path=config['model_save_path'],
        log_path="./logs/phase3_eval",
        eval_freq=config['eval_freq'],
        deterministic=True,
        render=False,
        verbose=1,
        callback_after_eval=metric_hook
    )
    callbacks.append(eval_callback)

    # IMPROVED: Dynamic checkpoint manager with adaptive intervals and event-driven saves
    checkpoint_manager = DynamicCheckpointManager(
        market=market_name,
        phase=3,
        seed=42,
        config_path='config/checkpoint_config.yaml',
        metric_tracker=metric_tracker,
        target_timesteps=config['total_timesteps'],
        verbose=True,
        registry=registry
    )
    safe_print(f"[CHECKPOINT] Dynamic checkpoint manager initialized")
    safe_print(f"[CHECKPOINT] Base interval: 10K steps (adaptive, LLM-optimized)")
    safe_print(f"[CHECKPOINT] Event triggers: periodic, best, phase_end, interrupt")
    safe_print(f"[CHECKPOINT] Disk-full protection: enabled")
    callbacks.append(checkpoint_manager)
    
    # KL Divergence callback
    kl_callback = KLDivergenceCallback(log_freq=1000, target_kl=config['target_kl'])
    callbacks.append(kl_callback)
    
    # LLM Monitoring callback
    llm_callback = LLMMonitoringCallback(
        hybrid_agent=hybrid_agent,
        log_freq=config['llm_log_freq'],
        verbose=1
    )
    callbacks.append(llm_callback)

    # Optional self-correcting callbacks (policy controller + corrective manager)
    if policy_controller is not None:
        callbacks.append(policy_controller)
    if corrective_manager is not None:
        callbacks.append(corrective_manager)
    
    # ============================================
    # PHASE 1 IMPLEMENTATION: Adaptive Fusion
    # ============================================
    
    # Import fusion components
    try:
        from .fusion_network import FusionNetwork, FusionExperienceBuffer, FusionTrainer
    except ImportError:
        from fusion_network import FusionNetwork, FusionExperienceBuffer, FusionTrainer
    
    # Initialize fusion network
    if hybrid_agent.use_neural_fusion:
        safe_print("[FUSION] Initializing adaptive fusion network...")
        fusion_network = FusionNetwork(
            input_dim=20,
            hidden_dims=[128, 64, 32],
            num_actions=6
        )
        fusion_trainer = FusionTrainer(fusion_network, learning_rate=3e-4, device=config.get('device', 'auto'))
        fusion_buffer = FusionExperienceBuffer(max_size=100000)
        
        # Attach to hybrid agent
        hybrid_agent.fusion_network = fusion_network
        hybrid_agent.fusion_buffer = fusion_buffer
        
        safe_print("[FUSION] ✅ Adaptive fusion network initialized")
        
        # ============================================
        # Fusion Training Callback
        # ============================================
        
        class FusionTrainingCallback(BaseCallback):
            """
            Callback to train fusion network during RL training.

            Strategy:
            - Collect fusion decisions during rollouts
            - Every N steps, train fusion network on successful outcomes
            - Monitor fusion accuracy and trust scores
            """

            def __init__(self, fusion_trainer, fusion_buffer, train_freq=1000, verbose=0):
                super().__init__(verbose)
                self.fusion_trainer = fusion_trainer
                self.fusion_buffer = fusion_buffer
                self.train_freq = train_freq
                self.fusion_losses = []
                self.fusion_accuracies = []

            def _on_step(self) -> bool:
                # Train fusion network periodically
                if self.num_timesteps % self.train_freq == 0 and len(self.fusion_buffer) >= 256:
                    # Sample and train
                    inputs, actions, weights = self.fusion_buffer.sample(batch_size=256)
                    loss, accuracy = self.fusion_trainer.train_step(inputs, actions, weights)

                    self.fusion_losses.append(loss)
                    self.fusion_accuracies.append(accuracy)

                    # Log
                    if self.num_timesteps % (self.train_freq * 10) == 0:
                        avg_loss = np.mean(self.fusion_losses[-10:])
                        avg_acc = np.mean(self.fusion_accuracies[-10:])
                        print(f"[FUSION] Step {self.num_timesteps}: Loss={avg_loss:.4f}, Acc={avg_acc:.2%}")

                return True
        
        fusion_callback = FusionTrainingCallback(fusion_trainer, fusion_buffer, train_freq=1000)
        callbacks.append(fusion_callback)
        
        # ============================================
        # PHASE 2 IMPLEMENTATION: LLM Fine-Tuning
        # ============================================
        
        class LLMFineTuningCallback(BaseCallback):
            """
            Callback to fine-tune LLM during training.
            
            Strategy:
            - Collect LLM queries and outcomes during episodes
            - Every N steps, fine-tune LLM on successful trades
            - Monitor LLM improvement via agreement rate with optimal actions
            """
            
            def __init__(self, llm_model, train_freq=5000, batch_size=8, verbose=0):
                super().__init__(verbose)
                self.llm_model = llm_model
                self.train_freq = train_freq
                self.batch_size = batch_size
                self.steps = 0
                self.llm_losses = []
                self.llm_accuracies = []
            
            def _on_step(self) -> bool:
                self.steps += 1
                
                # Fine-tune LLM periodically
                if self.steps % self.train_freq == 0:
                    if (hasattr(self.llm_model, 'experience_buffer') and 
                        len(self.llm_model.experience_buffer) >= self.batch_size):
                        # Fine-tune
                        loss, accuracy = self.llm_model.fine_tune_step(
                            batch_size=self.batch_size,
                            learning_rate=5e-5
                        )
                        
                        if loss is not None:
                            self.llm_losses.append(loss)
                            self.llm_accuracies.append(accuracy)
                            
                            # Log
                            print(f"[LLM FINE-TUNE] Step {self.steps}: Loss={loss:.4f}, Acc={accuracy:.2%}")
                
                return True
            
            def _on_rollout_end(self):
                """Called after each rollout - update outcomes."""
                # Note: Outcome updating happens in hybrid_agent during episode
                pass
        
        if hybrid_agent.llm_advisor.enable_fine_tuning:
            llm_finetune_callback = LLMFineTuningCallback(
                llm_model=hybrid_agent.llm_advisor,
                train_freq=5000,
                batch_size=8
            )
            callbacks.append(llm_finetune_callback)
            
            # Save LoRA adapters with model
            def save_with_lora(model, path):
                """Save model with LoRA adapters."""
                model.save(path)
                lora_path = path.replace('.zip', '_lora')
                hybrid_agent.llm_advisor.save_lora_adapters(lora_path)
                print(f"[SAVE] Model + LoRA adapters saved to {path}")

    # ============================================
    # ADAPTER WARMUP CALLBACK
    # ============================================
    # Unfreezes Phase 2 weights after adapter warmup period

    class AdapterWarmupCallback(BaseCallback):
        """
        Unfreezes Phase 2 weights after adapter warmup period.

        Strategy:
        - During warmup: Only adapter is trainable, Phase 2 frozen
        - After warmup: Unfreeze all weights for full training
        - This allows adapter to learn optimal projection first

        Args:
            warmup_steps: Number of steps to train only adapter
            verbose: Verbosity level
        """

        def __init__(self, warmup_steps, verbose=0):
            super().__init__(verbose)
            self.warmup_steps = warmup_steps
            self.unfrozen = False
            self.reported = False

        def _on_step(self):
            """Called after each training step."""
            if not self.unfrozen and self.num_timesteps >= self.warmup_steps:
                if not self.reported:
                    safe_print(f"\n{'='*70}")
                    safe_print(f"[ADAPTER] Warmup complete ({self.warmup_steps:,} steps)")
                    safe_print(f"[ADAPTER] Unfreezing Phase 2 weights for full training...")

                    # Unfreeze all parameters
                    trainable_before = sum(p.numel() for p in self.model.policy.parameters() if p.requires_grad)

                    for param in self.model.policy.parameters():
                        param.requires_grad = True

                    trainable_after = sum(p.numel() for p in self.model.policy.parameters() if p.requires_grad)

                    safe_print(f"[ADAPTER] Trainable parameters:")
                    safe_print(f"  - Before: {trainable_before:,} (adapter only)")
                    safe_print(f"  - After:  {trainable_after:,} (full network)")
                    safe_print(f"[ADAPTER] ✅ All weights now trainable!")
                    safe_print(f"{'='*70}\n")

                    self.unfrozen = True
                    self.reported = True

            return True

    # Add adapter warmup callback if enabled
    if config.get('freeze_phase2_initially') and config.get('unfreeze_after_warmup'):
        # Check if policy has adapter (transfer learning case)
        if hasattr(model.policy, 'adapter'):
            adapter_callback = AdapterWarmupCallback(
                warmup_steps=config['adapter_warmup_steps'],
                verbose=1
            )
            callbacks.append(adapter_callback)
            safe_print(f"[CALLBACK] Adapter warmup enabled: {config['adapter_warmup_steps']:,} steps")
        else:
            safe_print("[CALLBACK] Adapter warmup skipped (no adapter in policy)")

    return callbacks, checkpoint_manager


def train_phase3(
    market_name=None,
    test_mode=False,
    continue_training=False,
    model_path=None,
    n_envs: Optional[int] = None,
    vec_env_cls: Optional[str] = None,
):
    """
    Main Phase 3 training function.
    
    Args:
        market_name: Market to train on (e.g., 'NQ', 'ES')
        test_mode: If True, use reduced timesteps for testing
        continue_training: If True, continue from existing model
        model_path: Path to model to continue from
    """
    safe_print("=" * 70)
    safe_print("Phase 3: Hybrid RL + LLM Trading Agent Training")
    safe_print("=" * 70)
    
    # Configuration
    config = PHASE3_CONFIG.copy()
    if test_mode:
        config['total_timesteps'] = 50_000  # Reduced for testing
        config['eval_freq'] = 1000
        config['save_freq'] = 5000
        config['n_envs'] = 2  # Reduce parallel envs
        safe_print("[CONFIG] Test mode enabled - reduced settings")

    if n_envs is not None:
        config['n_envs'] = max(1, n_envs)
    if vec_env_cls is not None:
        config['vec_env_cls'] = vec_env_cls

    requested_envs = config.get('n_envs', 1)
    config['n_envs'] = get_effective_num_envs(requested_envs)
    if config['n_envs'] != requested_envs:
        safe_print(
            f"[ENV] Requested {requested_envs} envs, capped to {config['n_envs']} based on host CPU"
        )

    if config.get('vec_env_cls') == 'subproc' and not supports_subproc_vec_env():
        safe_print("[ENV] SubprocVecEnv unsupported on this host, falling back to DummyVecEnv")
        config['vec_env_cls'] = 'dummy'

    safe_print(
        f"[THREADS] BLAS threads={_THREAD_LIMIT_INT} | PyTorch threads={TORCH_THREADS_EFFECTIVE} "
        "(override via TRAINER_MAX_BLAS_THREADS / PYTORCH_NUM_THREADS)"
    )
    
    # Market selection
    if market_name is None:
        available_markets = detect_available_markets()
        if not available_markets:
            safe_print("[ERROR] No training data found")
            return None
        selected_market, market_spec = select_market_for_training(available_markets, safe_print)
        if selected_market is None:
            safe_print("[ERROR] No market selected")
            return None
        market_name = selected_market['market']
    
    safe_print(f"[MARKET] Training on: {market_name}")
    
    # Data loading
    data_pattern = f"./data/{market_name}_D1M.csv"
    try:
        env_data, second_data = load_data_for_training(
            market_name, data_pattern, 
            use_second_data=config['second_data_enabled']
        )
    except Exception as e:
        safe_print(f"[ERROR] Failed to load data: {e}")
        return None
    
    # Initialize LLM first (needed for hybrid agent)
    safe_print("[LLM] Initializing LLM advisor...")
    try:
        llm_model = LLMReasoningModule(
            config_path=config['llm_config_path']
        )
        safe_print("[OK] LLM advisor initialized")
    except Exception as e:
        safe_print(f"[ERROR] Failed to initialize LLM: {e}")
        safe_print("[INFO] Ensure Phi-3-mini-4k-instruct model is downloaded to project folder")
        return None

    # CRITICAL FIX: Create hybrid agent BEFORE environments
    # This allows the hybrid agent to be passed to environment constructors
    safe_print("[HYBRID] Creating hybrid agent...")
    # Create hybrid agent with rl_model=None (will be set after model creation)
    hybrid_agent = HybridTradingAgent(
        rl_model=None,
        llm_model=llm_model,
        config=config
    )
    safe_print("[OK] Hybrid agent created (RL model will be set after environment creation)")

    # Create environments with hybrid agent reference
    safe_print(f"[ENV] Creating {config['n_envs']} parallel environments...")

    # CRITICAL FIX: Don't pass hybrid_agent to SubprocVecEnv (unpicklable thread objects)
    # The hybrid_agent is only needed in main process via HybridAgentPolicy, not in subprocesses
    # Subprocesses only execute environment steps; LLM decisions happen in main process policy
    if config['vec_env_cls'] == 'subproc':
        # SubprocVecEnv: Pass None for hybrid_agent (avoids pickling ThreadPoolExecutor)
        def make_env(rank):
            return lambda: create_phase3_env(
                env_data, second_data, market_name, config, rank, hybrid_agent=None
            )
        train_env = SubprocVecEnv([make_env(i) for i in range(config['n_envs'])])
        safe_print("[ENV] Using SubprocVecEnv (multiprocess) - hybrid_agent in main process only")
    else:
        # DummyVecEnv: Can pass hybrid_agent (single process, no pickling needed)
        def make_env(rank):
            return lambda: create_phase3_env(
                env_data, second_data, market_name, config, rank, hybrid_agent=hybrid_agent
            )
        train_env = DummyVecEnv([make_env(i) for i in range(config['n_envs'])])
        safe_print("[ENV] Using DummyVecEnv (single process) - hybrid_agent enabled")

    # Normalization with Phase 2 transfer (CRITICAL FIX #5)
    safe_print("[ENV] Setting up normalization...")

    # Try to load Phase 2 VecNormalize stats
    phase2_vecnorm_path = config.get('phase2_vecnorm_path')
    loaded_phase2_vecnorm = False

    if phase2_vecnorm_path and os.path.exists(phase2_vecnorm_path):
        try:
            safe_print(f"[VECNORM] Loading Phase 2 normalization stats from {phase2_vecnorm_path}")
            train_env = VecNormalize.load(phase2_vecnorm_path, train_env)
            train_env = ensure_vecenv_action_masks(train_env)
            train_env.training = True  # Enable training mode
            train_env.norm_obs = True
            train_env.norm_reward = True
            loaded_phase2_vecnorm = True
            safe_print("[VECNORM] ✅ Phase 2 normalization stats loaded successfully")
            safe_print("[VECNORM] This provides ~20% faster convergence by reusing Phase 2 statistics")
        except Exception as e:
            safe_print(f"[VECNORM] [WARNING] Could not load Phase 2 VecNormalize: {e}")
            safe_print("[VECNORM] Creating fresh normalization (will take ~100K steps to stabilize)")
            loaded_phase2_vecnorm = False

    if not loaded_phase2_vecnorm:
        # Create fresh VecNormalize if Phase 2 stats not available
        train_env = VecNormalize(
            train_env,
            training=True,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=config['gamma'],
            epsilon=1e-8
        )
        train_env = ensure_vecenv_action_masks(train_env)
        safe_print("[VECNORM] Created fresh normalization wrapper")

    registry_ready = register_training_envs(train_env, config['n_envs'], hybrid_agent)
    if not registry_ready:
        safe_print("[ENV] [WARN] Hybrid policy will use fallback position/market state data.")

    # Evaluation environment (must match training env wrapper structure)
    eval_env = DummyVecEnv([lambda: create_phase3_env(env_data, second_data, market_name, config, rank=999, hybrid_agent=None)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    eval_env = ensure_vecenv_action_masks(eval_env)

    # CRITICAL FIX: Load or create model with hybrid policy (enables LLM during training)
    if continue_training and model_path:
        # Continue from existing Phase 3 model
        safe_print(f"[MODEL] Continuing training from {model_path}...")
        model = setup_hybrid_model(train_env, hybrid_agent, config, load_path=model_path)
        # Update hybrid agent with actual model
        hybrid_agent.set_rl_model(model)
        safe_print("[OK] Hybrid agent updated with loaded model")
        log_device_summary(model, hybrid_agent, label="after-continue-load")
    else:
        # Try transfer learning from Phase 2
        safe_print("\n[MODEL] Attempting Phase 2 → Phase 3 transfer learning...")

        # First create base model (will be wrapped with hybrid policy)
        base_model = load_phase2_and_transfer(config, train_env)

        if base_model is None:
            # Fallback: Create from scratch if no Phase 2 model found
            safe_print("\n[MODEL] No Phase 2 model found, creating Phase 3 from scratch...")
            safe_print("[MODEL] [WARNING] Training without Phase 2 transfer learning is NOT recommended!")
            safe_print("[MODEL] [WARNING] For best results, train Phase 2 first!")
            base_model = setup_model(train_env, config)

        # Wrap with hybrid policy (CRITICAL: enables LLM during training)
        safe_print("\n[MODEL] Wrapping model with hybrid policy for LLM integration...")
        # Pass base_model to preserve transfer learning (CRITICAL FIX for dimension mismatch)
        model = setup_hybrid_model(train_env, hybrid_agent, config, base_model=base_model)
        # Update hybrid agent with final wrapped model
        hybrid_agent.set_rl_model(model)
        log_device_summary(model, hybrid_agent, label="after-transfer-wrap")
        safe_print("[OK] Model wrapped with hybrid policy - LLM integration enabled!")
        
        # DEBUG: Log buffer creation details
        safe_print(f"[DEBUG] Model created with n_envs={getattr(model, 'n_envs', 'unknown')}")
        safe_print(f"[DEBUG] About to start learning with train_env.n_envs={train_env.num_envs}")

    # Recreate evaluation environment with hybrid agent
    eval_env = DummyVecEnv([lambda: create_phase3_env(env_data, second_data, market_name, config, rank=999, hybrid_agent=hybrid_agent)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    eval_env = ensure_vecenv_action_masks(eval_env)

    safe_print("[ENV] Hybrid agent communication channels established")

    # Create callbacks
    callbacks, checkpoint_manager = create_callbacks(model, eval_env, hybrid_agent, market_name, config)

    # Training
    safe_print("=" * 70)
    safe_print("Starting Phase 3 Hybrid Training")
    safe_print(f"Total timesteps: {config['total_timesteps']:,}")
    safe_print(f"Parallel environments: {config['n_envs']}")
    safe_print(f"LLM features: {'Enabled' if config['use_llm_features'] else 'Disabled'}")
    safe_print("=" * 70)
    
    try:
        model.learn(
            total_timesteps=config['total_timesteps'],
            callback=callbacks,
            progress_bar=True,
            tb_log_name=f"phase3_hybrid_{market_name}"
        )

        # Save phase_end checkpoint after successful training
        checkpoint_manager.on_phase_end()

        safe_print("\n" + "=" * 70)
        safe_print("✅ Phase 3 Training Completed Successfully!")
        safe_print("=" * 70)

        # Run checkpoint retention cleanup
        safe_print("\n[CHECKPOINT] Running checkpoint retention cleanup...")
        try:
            retention_manager = CheckpointRetentionManager('config/checkpoint_config.yaml')
            checkpoint_dir = f'./models/phase3_hybrid/{market_name}/checkpoints/'
            retention_manager.prune_checkpoints(checkpoint_dir, dry_run=False, verbose=False)
            safe_print("[CHECKPOINT] Retention cleanup completed")
        except Exception as e:
            safe_print(f"[WARNING] Checkpoint retention cleanup failed: {e}")

        # Save final model
        final_model_dir = config['model_save_path']
        os.makedirs(final_model_dir, exist_ok=True)
        final_model_path = f"{final_model_dir}/phase3_hybrid_final"
        model.save(final_model_path)
        vecnorm_path = config['vecnormalize_path']
        vecnorm_dir = os.path.dirname(vecnorm_path)
        if vecnorm_dir:
            os.makedirs(vecnorm_dir, exist_ok=True)
        train_env.save(vecnorm_path)

        safe_print(f"[SAVE] Model saved to: {final_model_path}")
        safe_print(f"[SAVE] VecNormalize saved to: {config['vecnormalize_path']}")

        # Print final statistics
        final_stats = hybrid_agent.get_stats()
        safe_print("\n[STATS] Final Hybrid Agent Statistics:")
        safe_print(f"  Total decisions: {final_stats.get('total_decisions', 0)}")
        safe_print(f"  Agreement rate: {final_stats.get('agreement_pct', 0):.1f}%")
        safe_print(f"  Risk veto rate: {final_stats.get('risk_veto_pct', 0):.1f}%")
        safe_print(f"  LLM query rate: {final_stats.get('llm_query_rate', 0):.1f}%")

        # Print LLM statistics
        llm_stats = hybrid_agent.get_llm_stats()
        if llm_stats.get('total_queries', 0) > 0:
            safe_print("\n[STATS] LLM Performance:")
            safe_print(f"  Total queries: {llm_stats.get('total_queries', 0)}")
            safe_print(f"  Average latency: {llm_stats.get('avg_latency_ms', 0):.1f}ms")
            safe_print(f"  Error rate: {llm_stats.get('error_rate', 0):.1f}%")
            safe_print(f"  Cache hit rate: {llm_stats.get('cache_hit_rate', 0):.1f}%")

        return model

    except KeyboardInterrupt:
        safe_print("\n[INTERRUPT] Training interrupted by user")
        safe_print("[CHECKPOINT] Saving interrupt checkpoint...")
        checkpoint_manager.on_interrupt()

        # Also save final model (legacy support)
        checkpoint_path = f"{config['model_save_path']}/phase3_hybrid_interrupted"
        model.save(checkpoint_path)
        train_env.save(config['vecnormalize_path'])

        safe_print(f"[SAVE] Legacy checkpoint saved to: {checkpoint_path}")
        return model

    except Exception as e:
        safe_print(f"\n[ERROR] Training failed: {e}")
        # Save interrupt checkpoint on exception
        try:
            checkpoint_manager.on_interrupt()
        except Exception:
            pass
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Cleanup
        try:
            train_env.close()
            eval_env.close()
        except:
            pass


def main():
    """Main entry point for Phase 3 training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 3 Hybrid RL + LLM Training')
    parser.add_argument('--market', type=str, help='Market to train on (e.g., NQ, ES)')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--continue', dest='continue_training', action='store_true',
                       help='Continue from existing model')
    parser.add_argument('--model-path', type=str, help='Path to model to continue from')
    parser.add_argument('--non-interactive', action='store_true', help='Run without prompts')
    parser.add_argument('--n-envs', type=int, help='Override number of parallel environments')
    parser.add_argument('--vec-env', choices=['dummy', 'subproc'], help='Select vectorized env type')

    args = parser.parse_args()
    
    # Run training
    model = train_phase3(
        market_name=args.market,
        test_mode=args.test,
        continue_training=args.continue_training,
        model_path=args.model_path,
        n_envs=args.n_envs,
        vec_env_cls=args.vec_env
    )
    
    if model is None:
        sys.exit(1)
    else:
        safe_print("\n✅ Phase 3 training script completed successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()
