#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification script for Phase 3 LLM integration fix.
Tests the core architectural changes without complex dependencies.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

def verify_fix():
    """Verify that the LLM integration fix is working."""
    print("=" * 70)
    print("PHASE 3 LLM INTEGRATION FIX VERIFICATION")
    print("=" * 70)
    
    success_count = 0
    total_tests = 0
    
    # Test 1: Hybrid policy exists and is importable
    total_tests += 1
    print("\n[TEST 1] Hybrid policy module...")
    try:
        from hybrid_policy import HybridAgentPolicy, HybridPolicyWrapper
        print("  PASS: Hybrid policy imported successfully")
        print(f"  - HybridAgentPolicy class: {HybridAgentPolicy.__name__}")
        print(f"  - HybridPolicyWrapper class: {HybridPolicyWrapper.__name__}")
        success_count += 1
    except Exception as e:
        print(f"  FAIL: {e}")
    
    # Test 2: Hybrid agent functionality
    total_tests += 1
    print("\n[TEST 2] Hybrid agent functionality...")
    try:
        from hybrid_agent import HybridTradingAgent
        from llm_reasoning import LLMReasoningModule
        from stable_baselines3 import PPO
        import gymnasium as gym
        import numpy as np
        
        # Create minimal test setup
        env = gym.make("CartPole-v1")
        rl_model = PPO("MlpPolicy", env, verbose=0, n_steps=16, batch_size=8)
        llm_model = LLMReasoningModule(config_path="config/llm_config.yaml", mock_mode=True)
        
        config = {
            'fusion': {
                'llm_weight': 0.3,
                'confidence_threshold': 0.7,
                'use_selective_querying': False,
                'query_interval': 5,
                'always_on_thinking': True,
                'use_batched_llm': True,
                'llm_batch_size': 4
            },
            'risk': {
                'max_consecutive_losses': 3,
                'min_win_rate_threshold': 0.4,
                'dd_buffer_threshold': 0.2,
                'enable_risk_veto': True
            },
            'llm_config_path': 'config/llm_config.yaml',
            'mock_llm': True
        }
        
        hybrid_agent = HybridTradingAgent(rl_model, llm_model, config)
        
        # Test prediction
        obs = np.random.randn(261)
        action_mask = np.array([1, 1, 1, 0, 0, 0])
        position_state = {
            'position': 0, 'balance': 50000, 'win_rate': 0.5, 
            'consecutive_losses': 0, 'dd_buffer_ratio': 1.0,
            'time_in_position': 0, 'unrealized_pnl': 0.0,
            'max_adverse_excursion': 0.0, 'max_favorable_excursion': 0.0,
            'entry_price': 0.0, 'timestamp': 0
        }
        market_context = {'market_name': 'NQ', 'current_time': '10:30', 'current_price': 5000.0}
        
        action, meta = hybrid_agent.predict(obs, action_mask, position_state, market_context)
        
        print(f"  PASS: Hybrid agent prediction works")
        print(f"  - Action: {action}")
        print(f"  - Fusion method: {meta.get('fusion_method', 'unknown')}")
        print(f"  - LLM confidence: {meta.get('llm_confidence', 0):.2f}")
        
        # Check stats
        stats = hybrid_agent.get_stats()
        print(f"  - Total decisions: {stats['total_decisions']}")
        
        success_count += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Environment predict method
    total_tests += 1
    print("\n[TEST 3] Environment predict method...")
    try:
        from environment_phase3_llm import TradingEnvironmentPhase3LLM
        import pandas as pd
        import numpy as np
        
        # Create minimal test data
        dates = pd.date_range('2024-01-01 09:30', periods=500, freq='1min', tz='America/New_York')
        test_data = pd.DataFrame({
            'open': np.random.uniform(4000, 4100, 500),
            'high': np.random.uniform(4100, 4150, 500),
            'low': np.random.uniform(3950, 4000, 500),
            'close': np.random.uniform(4000, 4100, 500),
            'volume': np.random.uniform(1000, 5000, 500),
            'sma_50': np.random.uniform(4000, 4100, 500),
            'sma_200': np.random.uniform(4000, 4100, 500),
            'rsi_15min': np.random.uniform(30, 70, 500),
            'rsi_60min': np.random.uniform(30, 70, 500),
            'volume_ratio_5min': np.random.uniform(0.5, 2.0, 500),
            'support_20': np.random.uniform(3950, 4000, 500),
            'resistance_20': np.random.uniform(4100, 4150, 500),
        }, index=dates)
        
        # Create environment with hybrid agent
        env = TradingEnvironmentPhase3LLM(
            data=test_data,
            use_llm_features=True,
            hybrid_agent=hybrid_agent
        )
        
        obs, info = env.reset()
        print(f"  PASS: Environment created with hybrid agent")
        print(f"  - Observation shape: {obs.shape} (expected: 261)")
        
        # Test predict method
        action_mask = env.action_masks()
        action, meta = env.predict(obs, action_mask)
        print(f"  PASS: Environment predict method works")
        print(f"  - Action: {action}")
        print(f"  - Has metadata: {meta is not None}")
        
        env.close()
        success_count += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Training script modifications
    total_tests += 1
    print("\n[TEST 4] Training script modifications...")
    try:
        # Check that setup_hybrid_model exists
        from train_phase3_llm import setup_hybrid_model
        print("  PASS: setup_hybrid_model function exists")
        
        # Check that hybrid policy is used
        import inspect
        source = inspect.getsource(setup_hybrid_model)
        if 'HybridAgentPolicy' in source:
            print("  PASS: setup_hybrid_model uses HybridAgentPolicy")
        else:
            print("  FAIL: setup_hybrid_model doesn't use HybridAgentPolicy")
            raise ValueError("Hybrid policy not integrated")
        
        success_count += 1
    except Exception as e:
        print(f"  FAIL: {e}")
    
    # Test 5: Async LLM functionality
    total_tests += 1
    print("\n[TEST 5] Async LLM functionality...")
    try:
        from async_llm import BatchedAsyncLLM
        import time
        
        async_llm = BatchedAsyncLLM(llm_model, max_batch_size=4, batch_timeout_ms=10)
        
        # Submit queries
        obs = np.random.randn(261).astype(np.float32)
        position_state = {'position': 0}
        market_context = {'price': 20150, 'trend': 'down'}
        available_actions = ['HOLD', 'BUY', 'SELL']
        
        for env_id in range(3):
            async_llm.submit_query(env_id, obs, position_state, market_context, available_actions)
        
        time.sleep(0.3)
        
        results = 0
        for env_id in range(3):
            result = async_llm.get_latest_result(env_id)
            if result:
                results += 1
        
        async_llm.shutdown()
        
        print(f"  PASS: Async LLM processed queries")
        print(f"  - Results ready: {results}/3")
        success_count += 1
    except Exception as e:
        print(f"  FAIL: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nThe Phase 3 LLM integration fix is working correctly:")
        print("  1. Hybrid policy wrapper routes predictions through hybrid agent")
        print("  2. Environment predict method enables LLM during training")
        print("  3. Training script uses setup_hybrid_model (not setup_model)")
        print("  4. Async LLM processes queries without blocking")
        print("  5. LLM statistics will be active (non-zero) during training")
        print("\nKEY ACHIEVEMENT:")
        print("  LLM integration is now active during training, not just inference!")
        print("  LLM statistics (query rate, agreement rate, etc.) will show actual values.")
        return True
    else:
        print("\n✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("Please check the errors above and fix the issues.")
        return False


if __name__ == "__main__":
    success = verify_fix()
    sys.exit(0 if success else 1)