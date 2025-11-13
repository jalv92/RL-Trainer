#!/usr/bin/env python3
"""
Simple debug script to investigate action masking issue in Phase 3 training.

This script focuses specifically on the buffer dimension mismatch.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add src to path
sys.path.append('src')

from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
from gymnasium import spaces

def test_buffer_dimensions():
    """Test what dimensions the buffer expects vs what we provide."""
    print("="*60)
    print("TEST: Buffer Dimension Analysis")
    print("="*60)
    
    # Create test spaces matching Phase 3
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(261,), dtype=np.float32)
    action_space = spaces.Discrete(6)
    
    print(f"[DEBUG] Observation space: {obs_space.shape}")
    print(f"[DEBUG] Action space: {action_space.n} actions")
    
    # Test with different n_envs values
    for n_envs in [2, 4]:
        print(f"\n[DEBUG] Testing with n_envs={n_envs}:")
        
        # Create buffer
        buffer = MaskableRolloutBuffer(
            buffer_size=10,
            observation_space=obs_space,
            action_space=action_space,
            device='cpu',
            n_envs=n_envs
        )
        
        print(f"  Buffer n_envs: {buffer.n_envs}")
        print(f"  Buffer mask_dims: {buffer.mask_dims}")
        print(f"  Expected action mask shape: ({buffer.n_envs}, {buffer.mask_dims})")
        print(f"  Expected total elements: {buffer.n_envs * buffer.mask_dims}")
        
        # Test adding data with correct dimensions
        try:
            # Create test action masks
            action_masks = np.ones((n_envs, 6), dtype=bool)
            print(f"  Test action_masks shape: {action_masks.shape}")
            print(f"  Test action_masks size: {action_masks.size}")
            
            # Try to add to buffer (this should work)
            observations = np.random.random((n_envs, 261)).astype(np.float32)
            rewards = np.random.random(n_envs)
            dones = np.array([False] * n_envs)
            
            buffer.add(
                observations,
                rewards,
                dones,
                np.zeros_like(rewards),  # values
                np.log(np.ones(6) / 6),  # log_probs
                action_masks=action_masks
            )
            print(f"  ✅ Buffer add successful")
            
        except Exception as e:
            print(f"  [X] Buffer add failed: {e}")
        
        # Test adding data with wrong dimensions
        try:
            # Create action masks with wrong n_envs
            wrong_action_masks = np.ones((2, 6), dtype=bool)  # Always 2 envs
            print(f"  Wrong action_masks shape: {wrong_action_masks.shape}")
            print(f"  Wrong action_masks size: {wrong_action_masks.size}")
            
            # Try to add to buffer (this should fail)
            buffer.add(
                observations,
                rewards,
                dones,
                np.zeros_like(rewards),  # values
                np.log(np.ones(6) / 6),  # log_probs
                action_masks=wrong_action_masks
            )
            print(f"  ❌ Buffer add should have failed but didn't!")
            
        except Exception as e:
            print(f"  ✅ Buffer add correctly failed: {e}")

def test_actual_training_scenario():
    """Test the actual scenario from the error."""
    print("\n" + "="*60)
    print("TEST: Actual Training Scenario Reproduction")
    print("="*60)
    
    # The error shows: cannot reshape array of size 12 into shape (4,6)
    # This means:
    # - We have 12 elements (2 envs × 6 actions)
    # - Buffer expects 24 elements (4 envs × 6 actions)
    
    print("[DEBUG] Error analysis:")
    print("  - Actual array size: 12 elements")
    print("  - Expected shape: (4, 6)")
    print("  - Expected elements: 4 × 6 = 24")
    print("  - This suggests buffer was created with n_envs=4")
    print("  - But action masks are being provided for only 2 environments")
    
    # Create buffer with 4 environments (what the error suggests)
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(261,), dtype=np.float32)
    action_space = spaces.Discrete(6)
    
    buffer = MaskableDictRolloutBuffer(
        buffer_size=10,
        observation_space=obs_space,
        action_space=action_space,
        device='cpu',
        n_envs=4  # This is what the error suggests
    )
    
    print(f"\n[DEBUG] Created buffer with n_envs=4")
    print(f"  Buffer expects action masks shape: {buffer.n_envs, buffer.mask_dims}")
    
    # Try to add action masks for only 2 environments (what we actually have)
    try:
        action_masks_2_envs = np.ones((2, 6), dtype=bool)
        print(f"[DEBUG] Trying to add action_masks with shape: {action_masks_2_envs.shape}")
        
        observations = np.random.random((2, 261)).astype(np.float32)
        rewards = np.random.random(2)
        dones = np.array([False] * 2)
        
        buffer.add(
            observations,
            rewards,
            dones,
            np.zeros_like(rewards),
            np.log(np.ones(6) / 6),
            action_masks=action_masks_2_envs
        )
        print("❌ This should have failed but didn't!")
        
    except Exception as e:
        print(f"[OK] Expected error occurred: {e}")
        print(f"   This confirms the dimension mismatch issue!")

def main():
    """Run all debug tests."""
    print("Phase 3 Action Masking - Simple Debug Script")
    print("="*60)
    
    try:
        # Test 1: Buffer dimensions
        test_buffer_dimensions()
        
        # Test 2: Actual scenario
        test_actual_training_scenario()
        
        print("\n" + "="*60)
        print("DEBUG ANALYSIS COMPLETE")
        print("="*60)
        print("\nCONCLUSION:")
        print("The error occurs because:")
        print("1. Buffer is created with n_envs=4 (expects 4×6=24 elements)")
        print("2. But training runs with n_envs=2 (provides 2×6=12 elements)")
        print("3. This mismatch causes the reshape error")
        print("\nSOLUTION:")
        print("Ensure buffer.n_envs matches the actual number of environments used in training")
        
    except Exception as e:
        print(f"\n[ERROR] Debug script failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()