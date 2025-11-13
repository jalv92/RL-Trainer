#!/usr/bin/env python3
"""
Standalone test for Async LLM functionality.

Tests BatchedAsyncLLM independently to verify it works correctly.
"""

import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import time
from src.llm_reasoning import LLMReasoningModule
from src.async_llm import BatchedAsyncLLM

def test_async_llm():
    """Test async LLM functionality."""
    print("Testing Async LLM Functionality")
    print("=" * 50)

    # Initialize LLM (mock mode for testing)
    print("1. Initializing LLM...")
    llm = LLMReasoningModule(config_path='config/llm_config.yaml', mock_mode=True)
    print("   ✅ LLM initialized")

    # Initialize BatchedAsyncLLM
    print("2. Initializing BatchedAsyncLLM...")
    async_llm = BatchedAsyncLLM(
        llm_model=llm,
        max_batch_size=4,
        batch_timeout_ms=100  # Short timeout for testing
    )
    print("   ✅ BatchedAsyncLLM initialized")

    # Test data
    obs = np.random.randn(261).astype(np.float32)
    position_state = {'position': 0, 'balance': 50000}
    market_context = {'price': 20150, 'trend': 'up'}
    available_actions = ['HOLD', 'BUY', 'SELL']

    print("3. Submitting test queries...")

    # Submit queries for multiple environments
    for env_id in range(3):
        print(f"   Submitting query for env {env_id}...")
        async_llm.submit_query(
            env_id, obs, position_state, market_context, available_actions
        )

    print("   ✅ All queries submitted")

    # Wait for results
    print("4. Waiting for results...")
    time.sleep(0.5)  # Give time for processing

    results_received = 0
    for env_id in range(3):
        result = async_llm.get_latest_result(env_id, timeout_ms=10)
        if result:
            print(f"   ✅ Env {env_id}: action={result['action']}, confidence={result['confidence']:.2f}, success={result['success']}")
            results_received += 1
        else:
            print(f"   ❌ Env {env_id}: No result received")

    print(f"5. Results: {results_received}/3 received")

    # Cleanup
    async_llm.shutdown()
    print("   ✅ Async LLM shutdown")

    # Verify success
    if results_received == 3:
        print("\n🎉 SUCCESS: Async LLM test passed!")
        return True
    else:
        print(f"\n❌ FAILURE: Only {results_received}/3 results received")
        return False

if __name__ == '__main__':
    success = test_async_llm()
    sys.exit(0 if success else 1)
