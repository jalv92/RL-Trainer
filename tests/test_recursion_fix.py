#!/usr/bin/env python3
"""
Test for Infinite Recursion Fix

This test verifies that the hybrid agent can make predictions without
entering an infinite recursion loop.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

def test_recursion_fix():
    """Test that hybrid agent doesn't cause infinite recursion."""
    print("=" * 70)
    print("Testing Infinite Recursion Fix")
    print("=" * 70)
    
    try:
        # Import required modules
        from hybrid_agent import HybridTradingAgent
        from hybrid_policy import HybridAgentPolicy
        from sb3_contrib import MaskablePPO
        from gymnasium import spaces
        import numpy as np
        import torch
        import torch.nn as nn
        
        print("\n[SETUP] Creating mock components...")
        
        # Create a simple mock environment spec
        obs_dim = 261  # Phase 3 observation dimension
        action_dim = 6
        
        # Create mock observation space
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_dim,), dtype=np.float32
        )
        action_space = spaces.Discrete(action_dim)
        
        # Create a simple mock policy that simulates the hybrid policy structure
        class MockPolicy:
            def __init__(self):
                self.observation_space = observation_space
                self.action_space = action_space
                # Simulate having parameters for device detection
                self._dummy_param = nn.Parameter(torch.tensor(1.0))
                
            def _rl_only_predict(self, observation, action_mask):
                """Mock RL-only prediction that should be called instead of predict."""
                print("  [OK] MockPolicy._rl_only_predict() called successfully")
                return 1, {'fusion_method': 'rl_only_test'}  # BUY action
                
            def parameters(self):
                return [self._dummy_param]
        
        # Create mock RL model
        class MockRLModel:
            def __init__(self):
                self.policy = MockPolicy()
                
            def predict(self, obs, action_masks=None):
                """This should NOT be called for hybrid policies (would cause recursion)."""
                raise RuntimeError("INFINITE RECURSION: MockRLModel.predict() was called!")
        
        # Create mock LLM
        class MockLLM:
            def get_stats(self):
                return {'total_queries': 0}
        
        print("\n[TEST] Creating hybrid agent with mock RL model...")
        
        # Configuration
        config = {
            'fusion': {
                'llm_weight': 0.3,
                'confidence_threshold': 0.7,
                'use_selective_querying': False,
                'query_interval': 5,
                'always_on_thinking': False
            },
            'risk': {
                'max_consecutive_losses': 3,
                'min_win_rate_threshold': 0.4,
                'dd_buffer_threshold': 0.2,
                'enable_risk_veto': True
            }
        }
        
        # Create hybrid agent
        rl_model = MockRLModel()
        llm_model = MockLLM()
        hybrid_agent = HybridTradingAgent(rl_model, llm_model, config)
        
        print("\n[TEST] Making prediction (this should NOT cause recursion)...")
        
        # Create test data
        observation = np.random.randn(obs_dim)
        action_mask = np.array([1, 1, 1, 0, 0, 0])  # HOLD, BUY, SELL allowed
        position_state = {
            'position': 0,
            'balance': 50000.0,
            'win_rate': 0.5,
            'consecutive_losses': 0,
            'dd_buffer_ratio': 1.0,
            'time_in_position': 0,
            'unrealized_pnl': 0.0,
            'timestamp': 0
        }
        market_context = {
            'market_name': 'NQ',
            'current_time': '10:30',
            'current_price': 15000.0
        }
        
        # This should call _rl_only_predict() NOT predict()
        try:
            action, meta = hybrid_agent.predict(
                observation, action_mask, position_state, market_context, env_id=0
            )
            
            print(f"\n[SUCCESS] Prediction completed without recursion!")
            print(f"   Action: {action}")
            print(f"   Fusion method: {meta['fusion_method']}")
            print(f"   RL action: {meta['rl_action']}")
            print(f"   LLM action: {meta['llm_action']}")
            
            return True
            
        except RuntimeError as e:
            if "INFINITE RECURSION" in str(e):
                print(f"\n[FAILED] {e}")
                return False
            else:
                raise
        
    except Exception as e:
        print(f"\n[FAILED] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """Test that the fix works with standard (non-hybrid) policies too."""
    print("\n" + "=" * 70)
    print("Testing Backward Compatibility")
    print("=" * 70)
    
    try:
        from hybrid_agent import HybridTradingAgent
        import numpy as np
        
        print("\n[SETUP] Creating mock standard policy...")
        
        # Create mock RL model with standard policy (no _rl_only_predict)
        class MockStandardPolicy:
            pass  # No _rl_only_predict method
        
        class MockStandardRLModel:
            def __init__(self):
                self.policy = MockStandardPolicy()
                self.predict_call_count = 0
                
            def predict(self, obs, action_masks=None):
                """Standard predict method for non-hybrid policies."""
                self.predict_call_count += 1
                print("  [OK] MockStandardRLModel.predict() called successfully")
                return 1, 0.8  # BUY action, 0.8 value
        
        # Create mock LLM
        class MockLLM:
            def get_stats(self):
                return {'total_queries': 0}
        
        print("\n[TEST] Creating hybrid agent with standard RL model...")
        
        # Configuration
        config = {
            'fusion': {
                'llm_weight': 0.3,
                'confidence_threshold': 0.7,
                'use_selective_querying': False,
                'query_interval': 5,
                'always_on_thinking': False
            },
            'risk': {
                'max_consecutive_losses': 3,
                'min_win_rate_threshold': 0.4,
                'dd_buffer_threshold': 0.2,
                'enable_risk_veto': True
            }
        }
        
        # Create hybrid agent
        rl_model = MockStandardRLModel()
        llm_model = MockLLM()
        hybrid_agent = HybridTradingAgent(rl_model, llm_model, config)
        
        print("\n[TEST] Making prediction with standard policy...")
        
        # Create test data
        observation = np.random.randn(261)
        action_mask = np.array([1, 1, 1, 0, 0, 0])
        position_state = {
            'position': 0,
            'balance': 50000.0,
            'win_rate': 0.5,
            'consecutive_losses': 0,
            'dd_buffer_ratio': 1.0,
            'timestamp': 0
        }
        market_context = {
            'market_name': 'NQ',
            'current_time': '10:30',
            'current_price': 15000.0
        }
        
        # This should call the standard predict() method
        action, meta = hybrid_agent.predict(
            observation, action_mask, position_state, market_context, env_id=0
        )
        
        if rl_model.predict_call_count == 1:
            print(f"\n[SUCCESS] Standard policy predict() called correctly!")
            print(f"   Action: {action}")
            print(f"   Fusion method: {meta['fusion_method']}")
            return True
        else:
            print(f"\n[FAILED] Standard predict() was called {rl_model.predict_call_count} times (expected 1)")
            return False
        
    except Exception as e:
        print(f"\n[FAILED] Backward compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("Testing Infinite Recursion Fix for Hybrid Agent")
    print("=" * 70)
    
    success1 = test_recursion_fix()
    success2 = test_backward_compatibility()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Recursion fix test: {'[PASSED]' if success1 else '[FAILED]'}")
    print(f"Backward compatibility test: {'[PASSED]' if success2 else '[FAILED]'}")
    
    if success1 and success2:
        print("\n[SUCCESS] All tests passed! The infinite recursion fix is working correctly.")
        sys.exit(0)
    else:
        print("\n[FAILED] Some tests failed. Please check the implementation.")
        sys.exit(1)
