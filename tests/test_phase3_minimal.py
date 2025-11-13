#!/usr/bin/env python3
"""Minimal Phase 3 test to isolate segfault issue."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("[TEST] Starting minimal Phase 3 test...")

# Test 1: Import modules
print("[TEST] Step 1: Importing modules...")
try:
    from train_phase3_llm import load_data_for_training, create_phase3_env
    from llm_reasoning import LLMReasoningModule
    from hybrid_agent import HybridTradingAgent
    print("[OK] Imports successful")
except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Load data
print("[TEST] Step 2: Loading data...")
try:
    env_data, second_data = load_data_for_training(
        'NQ', './data/NQ_D1M.csv', use_second_data=False
    )
    print(f"[OK] Data loaded: {len(env_data)} rows")
except Exception as e:
    print(f"[ERROR] Data loading failed: {e}")
    sys.exit(1)

# Test 3: Initialize mock LLM
print("[TEST] Step 3: Initializing mock LLM...")
try:
    llm_model = LLMReasoningModule(
        config_path='config/llm_config.yaml',
        mock_mode=True
    )
    print("[OK] Mock LLM initialized")
except Exception as e:
    print(f"[ERROR] LLM initialization failed: {e}")
    sys.exit(1)

# Test 4: Create hybrid agent
print("[TEST] Step 4: Creating hybrid agent...")
try:
    config = {
        'llm_weight': 0.3,
        'confidence_threshold': 0.7,
        'risk_veto_enabled': True,
    }
    hybrid_agent = HybridTradingAgent(
        rl_model=None,
        llm_model=llm_model,
        config=config
    )
    print("[OK] Hybrid agent created")
except Exception as e:
    print(f"[ERROR] Hybrid agent creation failed: {e}")
    sys.exit(1)

# Test 5: Create environment
print("[TEST] Step 5: Creating environment...")
try:
    config = {
        'window_size': 20,
        'second_data_enabled': False,
        'mock_llm': True,
        'use_llm_features': True,  # Enable 261D observations
    }
    env = create_phase3_env(
        env_data, second_data, 'NQ', config, rank=0, hybrid_agent=hybrid_agent
    )
    print(f"[OK] Environment created")
    print(f"[OK] Observation space: {env.observation_space.shape}")
    print(f"[OK] Action space: {env.action_space.n}")
except Exception as e:
    print(f"[ERROR] Environment creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Reset environment
print("[TEST] Step 6: Resetting environment...")
try:
    obs, info = env.reset()
    print(f"[OK] Environment reset successful")
    print(f"[OK] Observation shape: {obs.shape}")
except Exception as e:
    print(f"[ERROR] Environment reset failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test VecEnv wrapper
print("[TEST] Step 7: Testing DummyVecEnv wrapper...")
try:
    from stable_baselines3.common.vec_env import DummyVecEnv

    def make_env():
        return create_phase3_env(
            env_data, second_data, 'NQ', config, rank=0, hybrid_agent=hybrid_agent
        )

    vec_env = DummyVecEnv([make_env])
    print("[OK] DummyVecEnv created")

    obs = vec_env.reset()
    print(f"[OK] VecEnv reset successful")
    print(f"[OK] Observation shape: {obs.shape}")
except Exception as e:
    print(f"[ERROR] VecEnv test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Test VecNormalize
print("[TEST] Step 8: Testing VecNormalize...")
try:
    from stable_baselines3.common.vec_env import VecNormalize

    vec_env = VecNormalize(
        vec_env,
        training=True,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )
    print("[OK] VecNormalize created")

    obs = vec_env.reset()
    print(f"[OK] VecNormalize reset successful")
    print(f"[OK] Normalized observation shape: {obs.shape}")
except Exception as e:
    print(f"[ERROR] VecNormalize test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[SUCCESS] All tests passed!")
print("The crash is likely happening during model loading or training loop.")
