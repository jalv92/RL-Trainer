#!/usr/bin/env python3
"""
Debug script to investigate action masking issue in Phase 3 training.

This script will help us understand:
1. What action masks are being generated
2. What dimensions the buffer expects
3. Where the mismatch occurs
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib.common.wrappers import ActionMasker

# Add src to path
sys.path.append('src')

from environment_phase3_llm import TradingEnvironmentPhase3LLM
from feature_engineering import add_market_regime_features
from market_specs import get_market_spec

def create_debug_env():
    """Create a minimal environment for debugging."""
    print("[DEBUG] Creating test environment...")
    
    # Create minimal test data
    dates = pd.date_range('2024-01-01 09:30', periods=100, freq='1min', tz='America/New_York')
    test_data = pd.DataFrame({
        'open': np.random.uniform(4000, 4100, 100),
        'high': np.random.uniform(4100, 4150, 100),
        'low': np.random.uniform(3950, 4000, 100),
        'close': np.random.uniform(4000, 4100, 100),
        'volume': np.random.randint(100, 1000, 100),
        'atr': np.random.uniform(10, 30, 100),
        'sma_5': np.random.uniform(4000, 4100, 100),
        'sma_20': np.random.uniform(4000, 4100, 100),
        'sma_50': np.random.uniform(4000, 4100, 100),
        'sma_200': np.random.uniform(4000, 4100, 100),
        'rsi': np.random.uniform(30, 70, 100),
        'rsi_15min': np.random.uniform(30, 70, 100),
        'rsi_60min': np.random.uniform(30, 70, 100),
        'adx': np.random.uniform(20, 40, 100),
        'vwap': np.random.uniform(4000, 4100, 100),
        'price_to_vwap': np.random.uniform(-0.02, 0.02, 100),
        'vol_regime': np.random.uniform(0.3, 0.7, 100),
        'trend_strength': np.random.choice([-1, 0, 1], 100),
        'support_20': np.random.uniform(3950, 4050, 100),
        'resistance_20': np.random.uniform(4050, 4150, 100),
        'volume_ratio_5min': np.random.uniform(0.5, 2.0, 100),
        'volume_ratio_20min': np.random.uniform(0.5, 2.0, 100),
        'price_change_60min': np.random.uniform(-0.01, 0.01, 100),
        'efficiency_ratio': np.random.uniform(0.2, 0.8, 100)
    }, index=dates)
    
    # Add market regime features
    test_data = add_market_regime_features(test_data)
    
    # Create environment
    market_spec = get_market_spec('NQ')
    env = TradingEnvironmentPhase3LLM(
        data=test_data,
        use_llm_features=True,
        initial_balance=50000,
        window_size=20,
        market_spec=market_spec,
        commission_override=None,
        initial_sl_multiplier=1.5,
        initial_tp_ratio=3.0,
        position_size_contracts=1.0,
        trailing_drawdown_limit=2500,
        tighten_sl_step=0.5,
        extend_tp_step=1.0,
        trailing_activation_profit=1.0,
        hybrid_agent=None
    )
    
    # Add action masking wrapper
    env = ActionMasker(env, lambda env: env.action_masks())
    
    return env

def test_single_env_action_masks():
    """Test action masks from a single environment."""
    print("\n" + "="*60)
    print("TEST 1: Single Environment Action Masks")
    print("="*60)
    
    env = create_debug_env()
    
    # Reset environment
    obs, info = env.reset()
    print(f"[DEBUG] Observation shape: {obs.shape}")
    print(f"[DEBUG] Action space: {env.action_space}")
    
    # Get action masks
    action_masks = env.action_masks()
    print(f"[DEBUG] Action masks shape: {action_masks.shape}")
    print(f"[DEBUG] Action masks: {action_masks}")
    print(f"[DEBUG] Action masks dtype: {action_masks.dtype}")
    
    # Test a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        action_masks = env.action_masks()
        print(f"[DEBUG] Step {i+1}: action_masks shape={action_masks.shape}, values={action_masks}")
        
        if terminated:
            break
    
    env.close()

def test_vec_env_action_masks():
    """Test action masks from vectorized environment."""
    print("\n" + "="*60)
    print("TEST 2: Vectorized Environment Action Masks")
    print("="*60)
    
    # Create vectorized environment with 2 environments (like test mode)
    n_envs = 2
    
    def make_env():
        return create_debug_env()
    
    vec_env = DummyVecEnv([make_env for _ in range(n_envs)])
    
    # Reset vectorized environment
    obs = vec_env.reset()
    print(f"[DEBUG] Vectorized observation shape: {obs.shape}")
    print(f"[DEBUG] Number of environments: {n_envs}")
    
    # Test action masks from each environment
    for env_idx in range(n_envs):
        # Get the underlying environment
        env = vec_env.envs[env_idx]
        action_masks = env.action_masks()
        print(f"[DEBUG] Env {env_idx}: action_masks shape={action_masks.shape}, values={action_masks}")
    
    # Test a few steps
    for i in range(5):
        actions = [vec_env.action_space.sample() for _ in range(n_envs)]
        obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)
        
        print(f"[DEBUG] Step {i+1}:")
        for env_idx in range(n_envs):
            env = vec_env.envs[env_idx]
            action_masks = env.action_masks()
            print(f"  Env {env_idx}: action_masks shape={action_masks.shape}, values={action_masks}")
        
        if all(terminateds):
            break
    
    vec_env.close()

def test_buffer_expectations():
    """Test what dimensions the buffer expects."""
    print("\n" + "="*60)
    print("TEST 3: Buffer Dimension Analysis")
    print("="*60)
    
    # This will help us understand what the buffer expects
    from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer
    
    # Create a small buffer to test
    buffer = MaskableDictRolloutBuffer(
        buffer_size=10,
        observation_space=create_debug_env().observation_space,
        action_space=create_debug_env().action_space,
        device='cpu',
        n_envs=2  # Test with 2 environments
    )
    
    print(f"[DEBUG] Buffer n_envs: {buffer.n_envs}")
    print(f"[DEBUG] Buffer mask_dims: {buffer.mask_dims}")
    print(f"[DEBUG] Expected action mask shape: ({buffer.n_envs}, {buffer.mask_dims})")
    print(f"[DEBUG] Expected total elements: {buffer.n_envs * buffer.mask_dims}")

def main():
    """Run all debug tests."""
    print("Phase 3 Action Masking Debug Script")
    print("="*60)
    
    try:
        # Test 1: Single environment
        test_single_env_action_masks()
        
        # Test 2: Vectorized environment
        test_vec_env_action_masks()
        
        # Test 3: Buffer expectations
        test_buffer_expectations()
        
        print("\n" + "="*60)
        print("DEBUG COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"\n[ERROR] Debug script failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()