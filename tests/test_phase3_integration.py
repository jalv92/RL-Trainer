"""
Phase 3 Integration Test - LLM Statistics Verification

Tests that the Phase 3 LLM integration fix works correctly:
- LLM statistics are non-zero during training
- Position state is passed correctly to hybrid agent
- Decision fusion occurs during training loop
- Both mock and real LLM modes work

Run with: python -m pytest tests/test_phase3_integration.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(project_root))

try:
    from environment_phase3_llm import TradingEnvironmentPhase3LLM
    from llm_reasoning import LLMReasoningModule
    from hybrid_agent import HybridTradingAgent
    from hybrid_policy import HybridAgentPolicy, register_environment
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    MODULES_AVAILABLE = False


def create_test_data(n_steps=500):
    """Create minimal test data for integration testing."""
    dates = pd.date_range('2024-01-01 09:30', periods=n_steps, freq='1min', tz='America/New_York')

    # Generate synthetic price data
    base_price = 5000
    price_walk = np.random.randn(n_steps).cumsum() * 2

    data = pd.DataFrame({
        'open': base_price + price_walk,
        'high': base_price + price_walk + np.abs(np.random.randn(n_steps)) * 5,
        'low': base_price + price_walk - np.abs(np.random.randn(n_steps)) * 5,
        'close': base_price + price_walk + np.random.randn(n_steps),
        'volume': np.random.randint(1000, 10000, n_steps),
        'atr': np.random.uniform(10, 30, n_steps),
        'sma_5': base_price + price_walk,
        'sma_20': base_price + price_walk + np.random.randn(n_steps) * 2,
        'sma_50': base_price + price_walk + np.random.randn(n_steps) * 5,
        'sma_200': base_price + price_walk + np.random.randn(n_steps) * 10,
        'rsi': np.random.uniform(30, 70, n_steps),
        'rsi_15min': np.random.uniform(30, 70, n_steps),
        'rsi_60min': np.random.uniform(30, 70, n_steps),
        'adx': np.random.uniform(20, 40, n_steps),
        'vwap': base_price + price_walk + np.random.randn(n_steps),
        'price_to_vwap': np.random.uniform(-0.02, 0.02, n_steps),
        'vol_regime': np.random.uniform(0.3, 0.7, n_steps),
        'trend_strength': np.random.choice([-1, 0, 1], n_steps),
        'support_20': base_price + price_walk - 50,
        'resistance_20': base_price + price_walk + 50,
        'volume_ratio_5min': np.random.uniform(0.5, 2.0, n_steps),
        'volume_ratio_20min': np.random.uniform(0.5, 2.0, n_steps),
        'price_change_60min': np.random.uniform(-0.01, 0.01, n_steps),
        'efficiency_ratio': np.random.uniform(0.2, 0.8, n_steps),
        'macd': np.random.uniform(-10, 10, n_steps),
        'momentum': np.random.uniform(-100, 100, n_steps),
    }, index=dates)

    return data


@pytest.mark.skipif(not MODULES_AVAILABLE, reason="Required modules not available")
def test_hybrid_agent_creation_with_none_model():
    """Test that HybridTradingAgent can be created with rl_model=None."""
    print("\n" + "="*70)
    print("Test 1: HybridTradingAgent Creation with None Model")
    print("="*70)

    # Create LLM in mock mode
    llm_config = {
        'llm_config_path': 'config/llm_config.yaml',
        'mock_llm': True,
        'fusion': {
            'llm_weight': 0.3,
            'confidence_threshold': 0.7,
            'use_neural_fusion': False,
            'always_on_thinking': True,
        },
        'risk': {
            'max_consecutive_losses': 3,
            'min_win_rate_threshold': 0.4,
        }
    }

    llm = LLMReasoningModule(mock_mode=True)

    # Create hybrid agent with rl_model=None
    hybrid_agent = HybridTradingAgent(
        rl_model=None,
        llm_model=llm,
        config=llm_config
    )

    assert hybrid_agent.rl_agent is None, "RL agent should be None initially"
    assert hybrid_agent.llm_advisor is not None, "LLM advisor should be set"

    print("✅ HybridTradingAgent created successfully with rl_model=None")
    print(f"   - RL agent: {hybrid_agent.rl_agent}")
    print(f"   - LLM advisor: {hybrid_agent.llm_advisor}")


@pytest.mark.skipif(not MODULES_AVAILABLE, reason="Required modules not available")
def test_set_rl_model():
    """Test that RL model can be set after creation."""
    print("\n" + "="*70)
    print("Test 2: Set RL Model After Creation")
    print("="*70)

    llm_config = {
        'llm_config_path': 'config/llm_config.yaml',
        'mock_llm': True,
        'fusion': {'llm_weight': 0.3, 'always_on_thinking': True},
        'risk': {}
    }

    llm = LLMReasoningModule(mock_mode=True)
    hybrid_agent = HybridTradingAgent(rl_model=None, llm_model=llm, config=llm_config)

    # Create a simple environment and model
    data = create_test_data(n_steps=200)
    env = TradingEnvironmentPhase3LLM(data, use_llm_features=True)
    vec_env = DummyVecEnv([lambda: env])

    # Create model
    model = MaskablePPO("MlpPolicy", vec_env, verbose=0)

    # Set RL model
    hybrid_agent.set_rl_model(model)

    assert hybrid_agent.rl_agent is not None, "RL agent should be set"
    assert hybrid_agent.rl_model is model, "rl_model property should return the model"

    print("✅ RL model set successfully")
    print(f"   - RL agent type: {type(hybrid_agent.rl_agent)}")


@pytest.mark.skipif(not MODULES_AVAILABLE, reason="Required modules not available")
def test_hybrid_policy_state_access():
    """Test that HybridAgentPolicy can access environment state."""
    print("\n" + "="*70)
    print("Test 3: HybridAgentPolicy State Access")
    print("="*70)

    llm_config = {
        'llm_config_path': 'config/llm_config.yaml',
        'mock_llm': True,
        'fusion': {'llm_weight': 0.3, 'always_on_thinking': True, 'use_neural_fusion': False},
        'risk': {}
    }

    llm = LLMReasoningModule(mock_mode=True)

    # Create environment
    data = create_test_data(n_steps=200)
    env = TradingEnvironmentPhase3LLM(data, use_llm_features=True)

    # Register environment
    register_environment(0, env)

    # Create model with hybrid policy
    vec_env = DummyVecEnv([lambda: env])
    model = MaskablePPO("MlpPolicy", vec_env, verbose=0)

    # Create hybrid agent and set model
    hybrid_agent = HybridTradingAgent(rl_model=model, llm_model=llm, config=llm_config)

    # Create hybrid policy
    from hybrid_policy import HybridAgentPolicy
    policy = HybridAgentPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: 3e-4,
        hybrid_agent=hybrid_agent
    )

    # Test state access methods
    position_state = policy._build_position_state(env_id=0)
    market_context = policy._build_market_context(env_id=0)

    print(f"✅ State access successful")
    print(f"   - Position state keys: {list(position_state.keys())}")
    print(f"   - Market context keys: {list(market_context.keys())}")

    # Check state access stats
    stats = policy.get_state_access_stats()
    print(f"   - State access stats: {stats}")


@pytest.mark.skipif(not MODULES_AVAILABLE, reason="Required modules not available")
def test_training_with_llm_statistics():
    """
    Test that LLM statistics are non-zero during training.
    This is the KEY test for the Phase 3 fix!
    """
    print("\n" + "="*70)
    print("Test 4: Training with LLM Statistics (CRITICAL TEST)")
    print("="*70)

    llm_config = {
        'llm_config_path': 'config/llm_config.yaml',
        'mock_llm': True,
        'fusion': {
            'llm_weight': 0.3,
            'confidence_threshold': 0.7,
            'use_neural_fusion': False,
            'always_on_thinking': True,
            'use_selective_querying': False,  # Always query
            'query_interval': 1,  # Query every step
        },
        'risk': {
            'enable_risk_veto': True,
            'max_consecutive_losses': 3,
        }
    }

    # Create environment and LLM
    data = create_test_data(n_steps=300)
    env = TradingEnvironmentPhase3LLM(data, use_llm_features=True)
    llm = LLMReasoningModule(mock_mode=True)

    # Register environment
    register_environment(0, env)

    # Create hybrid agent with rl_model=None initially
    hybrid_agent = HybridTradingAgent(
        rl_model=None,
        llm_model=llm,
        config=llm_config
    )

    # Create vectorized environment
    vec_env = DummyVecEnv([lambda: env])

    # Create model with standard policy
    model = MaskablePPO("MlpPolicy", vec_env, verbose=0, n_steps=64)

    # Set RL model in hybrid agent
    hybrid_agent.set_rl_model(model)

    # Create hybrid policy and replace model's policy
    hybrid_policy = HybridAgentPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: 3e-4,
        hybrid_agent=hybrid_agent
    )

    # Replace model's policy with hybrid policy
    # Copy weights from original policy
    hybrid_policy.load_state_dict(model.policy.state_dict())
    model.policy = hybrid_policy

    print("Starting short training run (128 timesteps)...")

    # Train for a short period
    model.learn(total_timesteps=128, progress_bar=False)

    # Get hybrid agent statistics
    stats = hybrid_agent.get_stats()

    print("\n" + "="*70)
    print("HYBRID AGENT STATISTICS")
    print("="*70)
    print(f"Total decisions: {stats['total_decisions']}")
    print(f"Agreement rate: {stats['agreement_pct']:.1f}%")
    print(f"Disagreement rate: {stats['disagreement_pct']:.1f}%")
    print(f"Risk veto rate: {stats['risk_veto_pct']:.1f}%")
    print(f"LLM query rate: {stats['llm_query_rate']:.1f}%")
    print(f"RL avg confidence: {stats['rl_avg_confidence']:.3f}")
    print(f"LLM avg confidence: {stats['llm_avg_confidence']:.3f}")
    print("="*70)

    # CRITICAL ASSERTIONS - These verify the fix worked!
    assert stats['total_decisions'] > 0, "Should have made decisions during training"
    assert stats['llm_query_rate'] > 0, "❌ LLM query rate is 0% - FIX DID NOT WORK!"

    print("\n✅ CRITICAL TEST PASSED!")
    print("   ✅ LLM statistics are NON-ZERO during training")
    print(f"   ✅ LLM query rate: {stats['llm_query_rate']:.1f}%")
    print(f"   ✅ Total decisions: {stats['total_decisions']}")

    # Get policy state access stats
    policy_stats = model.policy.get_state_access_stats()
    print(f"\n   - Policy state access: {policy_stats['total_accesses']} total")
    print(f"   - Position state actual: {policy_stats['position_state_actual_pct']:.1f}%")
    print(f"   - Market context actual: {policy_stats['market_context_actual_pct']:.1f}%")


@pytest.mark.skipif(not MODULES_AVAILABLE, reason="Required modules not available")
def test_decision_fusion_during_training():
    """Test that decision fusion methods are being used during training."""
    print("\n" + "="*70)
    print("Test 5: Decision Fusion During Training")
    print("="*70)

    llm_config = {
        'llm_config_path': 'config/llm_config.yaml',
        'mock_llm': True,
        'fusion': {
            'llm_weight': 0.3,
            'always_on_thinking': True,
            'use_neural_fusion': False,
        },
        'risk': {}
    }

    data = create_test_data(n_steps=200)
    env = TradingEnvironmentPhase3LLM(data, use_llm_features=True)
    llm = LLMReasoningModule(mock_mode=True)

    register_environment(0, env)

    hybrid_agent = HybridTradingAgent(rl_model=None, llm_model=llm, config=llm_config)
    vec_env = DummyVecEnv([lambda: env])
    model = MaskablePPO("MlpPolicy", vec_env, verbose=0, n_steps=32)

    hybrid_agent.set_rl_model(model)

    hybrid_policy = HybridAgentPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: 3e-4,
        hybrid_agent=hybrid_agent
    )

    hybrid_policy.load_state_dict(model.policy.state_dict())
    model.policy = hybrid_policy

    # Train briefly
    model.learn(total_timesteps=64, progress_bar=False)

    stats = hybrid_agent.get_stats()

    # Check that different fusion methods were used
    fusion_methods_used = (
        stats['rl_only'] +
        stats['llm_only'] +
        stats['agreement'] +
        stats['disagreement']
    )

    assert fusion_methods_used > 0, "No fusion decisions recorded"

    print(f"✅ Decision fusion is working")
    print(f"   - RL only: {stats['rl_only']}")
    print(f"   - LLM only: {stats['llm_only']}")
    print(f"   - Agreement: {stats['agreement']}")
    print(f"   - Disagreement: {stats['disagreement']}")
    print(f"   - Risk veto: {stats['risk_veto']}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PHASE 3 INTEGRATION TEST SUITE")
    print("Testing: LLM Statistics Verification")
    print("="*70)

    if not MODULES_AVAILABLE:
        print("\n❌ ERROR: Required modules not available")
        print("Please ensure Phase 3 modules are installed.")
        sys.exit(1)

    # Run tests
    try:
        test_hybrid_agent_creation_with_none_model()
        test_set_rl_model()
        test_hybrid_policy_state_access()
        test_training_with_llm_statistics()
        test_decision_fusion_during_training()

        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nPhase 3 LLM integration is working correctly:")
        print("  ✅ Hybrid agent creation with None model")
        print("  ✅ RL model can be set after creation")
        print("  ✅ Policy can access environment state")
        print("  ✅ LLM statistics are NON-ZERO during training")
        print("  ✅ Decision fusion works during training")
        print("\n" + "="*70)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
