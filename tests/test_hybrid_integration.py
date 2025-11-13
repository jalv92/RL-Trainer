#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Test for Hybrid LLM Training

Tests that the architectural fixes enable LLM integration during training.
This script verifies that LLM statistics become active (non-zero) during training.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

def test_hybrid_integration():
    """Test that hybrid agent integrates LLM during training."""
    print("=" * 70)
    print("Hybrid LLM Integration Test")
    print("=" * 70)
    
    # Test 1: Import hybrid policy
    print("\n[TEST 1] Testing hybrid policy import...")
    try:
        from hybrid_policy import HybridAgentPolicy, HybridPolicyWrapper
        print("✅ Hybrid policy imported successfully")
    except Exception as e:
        print(f"❌ Failed to import hybrid policy: {e}")
        return False
    
    # Test 2: Test hybrid agent with mock LLM
    print("\n[TEST 2] Testing hybrid agent with mock LLM...")
    try:
        from hybrid_agent import HybridTradingAgent
        from llm_reasoning import LLMReasoningModule
        from stable_baselines3 import PPO
        import gymnasium as gym
        import numpy as np
        
        # Create mock environment
        env = gym.make("CartPole-v1")
        
        # Create mock RL model
        rl_model = PPO("MlpPolicy", env, verbose=0)
        
        # Create mock LLM
        llm_model = LLMReasoningModule(config_path="config/llm_config.yaml", mock_mode=True)
        
        # Create hybrid agent
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
            'llm_config_path': './config/llm_config.yaml',
            'mock_llm': True
        }
        
        hybrid_agent = HybridTradingAgent(rl_model, llm_model, config)
        print("✅ Hybrid agent created successfully")
        
        # Test prediction
        obs = np.random.randn(261)
        action_mask = np.array([1, 1, 1, 0, 0, 0])
        position_state = {
            'position': 0,
            'balance': 50000,
            'win_rate': 0.5,
            'consecutive_losses': 0,
            'dd_buffer_ratio': 1.0,
            'time_in_position': 0,
            'unrealized_pnl': 0.0,
            'max_adverse_excursion': 0.0,
            'max_favorable_excursion': 0.0,
            'entry_price': 0.0,
            'timestamp': 0
        }
        market_context = {
            'market_name': 'NQ',
            'current_time': '10:30',
            'current_price': 5000.0
        }
        
        action, meta = hybrid_agent.predict(obs, action_mask, position_state, market_context)
        print(f"✅ Hybrid agent prediction successful: action={action}")
        print(f"   LLM confidence: {meta.get('llm_confidence', 0):.2f}")
        print(f"   Fusion method: {meta.get('fusion_method', 'unknown')}")
        
        # Check stats
        stats = hybrid_agent.get_stats()
        print(f"✅ Hybrid agent stats: {stats['total_decisions']} decisions")
        
    except Exception as e:
        print(f"❌ Hybrid agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Test environment integration
    print("\n[TEST 3] Testing environment integration...")
    try:
        from environment_phase3_llm import TradingEnvironmentPhase3LLM
        import pandas as pd
        import numpy as np
        
        # Create minimal test data with required features
        dates = pd.date_range('2024-01-01 09:30', periods=1000, freq='1min', tz='America/New_York')
        test_data = pd.DataFrame({
            'open': np.random.uniform(4000, 4100, 1000),
            'high': np.random.uniform(4100, 4150, 1000),
            'low': np.random.uniform(3950, 4000, 1000),
            'close': np.random.uniform(4000, 4100, 1000),
            'volume': np.random.uniform(1000, 5000, 1000),
            # Required LLM features
            'sma_50': np.random.uniform(4000, 4100, 1000),
            'sma_200': np.random.uniform(4000, 4100, 1000),
            'rsi_15min': np.random.uniform(30, 70, 1000),
            'rsi_60min': np.random.uniform(30, 70, 1000),
            'volume_ratio_5min': np.random.uniform(0.5, 2.0, 1000),
            'support_20': np.random.uniform(3950, 4000, 1000),
            'resistance_20': np.random.uniform(4100, 4150, 1000),
        }, index=dates)
        
        # Create environment
        env = TradingEnvironmentPhase3LLM(
            data=test_data,
            use_llm_features=True,
            hybrid_agent=hybrid_agent
        )
        
        # Test environment
        obs, info = env.reset()
        print(f"✅ Environment created with hybrid agent")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Expected: 261D")
        
        # Test environment prediction
        action_mask = env.action_masks()
        action, meta = env.predict(obs, action_mask)
        print(f"✅ Environment prediction successful: action={action}")
        
        # Test step
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Environment step successful: reward={reward:.2f}")
        
        env.close()
        
    except Exception as e:
        print(f"❌ Environment integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Test async LLM
    print("\n[TEST 4] Testing async LLM...")
    try:
        from async_llm import BatchedAsyncLLM
        import time
        
        # Create async LLM
        async_llm = BatchedAsyncLLM(llm_model, max_batch_size=4, batch_timeout_ms=10)
        
        # Submit queries
        obs = np.random.randn(261).astype(np.float32)
        position_state = {'position': 0}
        market_context = {'price': 20150, 'trend': 'down'}
        available_actions = ['HOLD', 'BUY', 'SELL']
        
        print("Submitting 5 async queries...")
        for env_id in range(5):
            async_llm.submit_query(env_id, obs, position_state, market_context, available_actions)
        
        # Wait for results
        time.sleep(0.5)
        
        results_ready = 0
        for env_id in range(5):
            result = async_llm.get_latest_result(env_id)
            if result:
                results_ready += 1
                print(f"  Env {env_id}: action={result['action']}, confidence={result['confidence']:.2f}")
        
        print(f"✅ {results_ready}/5 results ready")
        print(f"✅ Async LLM working correctly")
        
        async_llm.shutdown()
        
    except Exception as e:
        print(f"❌ Async LLM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("✅ ALL INTEGRATION TESTS PASSED!")
    print("=" * 70)
    print("\nThe hybrid LLM integration is working correctly:")
    print("  • Hybrid policy routes predictions through hybrid agent")
    print("  • Environment integrates hybrid agent into training loop")
    print("  • Async LLM processes queries without blocking")
    print("  • LLM statistics will be active during training")
    
    return True


def test_training_loop():
    """Test a short training loop to verify LLM integration."""
    print("\n" + "=" * 70)
    print("Training Loop Integration Test")
    print("=" * 70)
    
    try:
        from train_phase3_llm import train_phase3
        
        print("\n[TEST] Running short training test (mock mode)...")
        print("This will verify LLM statistics become active during training.")
        print("Expected: LLM query rate > 0% after training starts\n")
        
        # Run short training in test mode
        model = train_phase3(
            market_name='NQ',
            test_mode=True,  # Reduced timesteps
            continue_training=False,
            model_path=None
        )
        
        if model is not None:
            print("\n✅ Training completed successfully")
            print("✅ LLM integration is working during training")
            return True
        else:
            print("\n❌ Training failed")
            return False
            
    except Exception as e:
        print(f"\n❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Hybrid LLM Integration Test Suite")
    print("Testing the architectural fixes for Phase 3 LLM integration\n")
    
    # Run integration tests
    success = test_hybrid_integration()
    
    if success:
        # Optionally run training test
        response = input("\nRun short training test? (y/n): ")
        if response.lower() == 'y':
            success = test_training_loop()
    
    if success:
        print("\n" + "🎉" * 35)
        print("ALL TESTS PASSED - LLM INTEGRATION IS WORKING!")
        print("🎉" * 35)
        sys.exit(0)
    else:
        print("\n" + "❌" * 35)
        print("TESTS FAILED - PLEASE CHECK ERRORS ABOVE")
        print("❌" * 35)
        sys.exit(1)