#!/usr/bin/env python3
"""
Minimal LLM integration tests - ASCII only
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd

def create_test_data():
    dates = pd.date_range('2024-01-01 09:30', periods=500, freq='1min', tz='America/New_York')
    data = pd.DataFrame({
        'open': np.random.uniform(4000, 4100, 500),
        'high': np.random.uniform(4100, 4150, 500),
        'low': np.random.uniform(3950, 4000, 500),
        'close': np.random.uniform(4000, 4100, 500),
        'volume': np.random.randint(100, 1000, 500),
        'atr': np.random.uniform(10, 30, 500),
        'sma_5': np.random.uniform(4000, 4100, 500),
        'sma_20': np.random.uniform(4000, 4100, 500),
        'sma_50': np.random.uniform(4000, 4100, 500),
        'sma_200': np.random.uniform(4000, 4100, 500),
        'rsi': np.random.uniform(30, 70, 500),
        'rsi_15min': np.random.uniform(30, 70, 500),
        'rsi_60min': np.random.uniform(30, 70, 500),
        'macd': 0,
        'momentum': 0,
        'adx': np.random.uniform(20, 40, 500),
        'vwap': np.random.uniform(4000, 4100, 500),
        'price_to_vwap': np.random.uniform(-0.02, 0.02, 500),
        'vol_regime': np.random.uniform(0.3, 0.7, 500),
        'trend_strength': np.random.choice([-1, 0, 1], 500),
        'support_20': np.random.uniform(3950, 4050, 500),
        'resistance_20': np.random.uniform(4050, 4150, 500),
        'volume_ratio_5min': np.random.uniform(0.5, 2.0, 500),
        'volume_ratio_20min': np.random.uniform(0.5, 2.0, 500),
        'price_change_60min': np.random.uniform(-0.01, 0.01, 500),
        'efficiency_ratio': np.random.uniform(0.2, 0.8, 500)
    }, index=dates)
    return data

def test_environment():
    from environment_phase3_llm import TradingEnvironmentPhase3LLM
    data = create_test_data()
    env = TradingEnvironmentPhase3LLM(data, use_llm_features=True)
    obs, _ = env.reset()
    assert obs.shape == (261,), f"Expected (261,), got {obs.shape}"
    obs, _, _, _, _ = env.step(0)
    assert obs.shape == (261,), f"Expected (261,), got {obs.shape}"
    env.close()
    print("PASS: Environment test")

def test_llm_features():
    from environment_phase3_llm import TradingEnvironmentPhase3LLM
    from llm_features import LLMFeatureBuilder
    data = create_test_data()
    env = TradingEnvironmentPhase3LLM(data, use_llm_features=True)
    env.reset()
    env.current_step = 100
    base_obs = np.random.randn(228)
    builder = LLMFeatureBuilder()
    enhanced_obs = builder.build_enhanced_observation(env, base_obs)
    assert enhanced_obs.shape == (261,), f"Expected (261,), got {enhanced_obs.shape}"
    env.close()
    print("PASS: LLM features test")

def test_hybrid_agent():
    from hybrid_agent import HybridTradingAgent
    from llm_reasoning import LLMReasoningModule
    
    class MockRL:
        def predict(self, obs, action_masks=None, deterministic=True):
            return 1, 15.0
    
    llm = LLMReasoningModule(mock_mode=True)
    config = {
        'fusion': {
            'llm_weight': 0.3,
            'confidence_threshold': 0.7,
            'use_selective_querying': False,
            'query_interval': 5,
            'cache_decay_rate': 0.8
        },
        'risk': {
            'max_consecutive_losses': 3,
            'min_win_rate_threshold': 0.4,
            'dd_buffer_threshold': 0.2,
            'enable_risk_veto': True
        }
    }
    
    hybrid = HybridTradingAgent(MockRL(), llm, config)
    obs = np.random.randn(261)
    action_mask = np.array([1, 1, 1, 0, 0, 0])
    position_state = {'position': 0, 'balance': 50000, 'win_rate': 0.5, 'consecutive_losses': 0, 'dd_buffer_ratio': 0.8}
    market_context = {}
    action, meta = hybrid.predict(obs, action_mask, position_state, market_context)
    assert action == 1
    print("PASS: Hybrid agent test")

if __name__ == "__main__":
    print("=" * 70)
    print("MINIMAL LLM INTEGRATION TESTS")
    print("=" * 70)
    try:
        test_environment()
        test_llm_features()
        test_hybrid_agent()
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED!")
        print("=" * 70)
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)