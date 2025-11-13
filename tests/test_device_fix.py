#!/usr/bin/env python3
"""
Test for Device Mismatch Fix

This test verifies that the adapter layer is on the correct device
and doesn't cause device mismatches during forward passes.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

def test_adapter_device_placement():
    """Test that adapter is on correct device after model creation."""
    print("=" * 70)
    print("Testing Adapter Device Placement")
    print("=" * 70)
    
    try:
        import torch
        import numpy as np
        from gymnasium import spaces
        from hybrid_policy_with_adapter import HybridAgentPolicyWithAdapter
        from sb3_contrib import MaskablePPO
        
        print("\n[SETUP] Creating policy with adapter...")
        
        # Create observation and action spaces
        obs_dim = 261  # Phase 3 dimension
        action_dim = 6
        
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_dim,), dtype=np.float32
        )
        action_space = spaces.Discrete(action_dim)
        
        # Create learning rate schedule (dummy)
        def lr_schedule(progress):
            return 0.0003
        
        # Create policy
        print("\n[TEST] Creating HybridAgentPolicyWithAdapter...")
        policy = HybridAgentPolicyWithAdapter(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            hybrid_agent=None  # Not needed for this test
        )
        
        # Check initial device (should be CPU by default)
        adapter_device = policy.adapter.weight.device
        print(f"\n[CHECK] Adapter device after creation: {adapter_device}")
        
        # Check device of other parameters
        try:
            first_param_device = next(policy.parameters()).device
            print(f"[CHECK] First parameter device: {first_param_device}")
        except StopIteration:
            print("[CHECK] No parameters found (unexpected)")
            return False
        
        # Test 1: Adapter should be on same device as other parameters
        if adapter_device == first_param_device:
            print("[OK] Adapter is on same device as other parameters")
        else:
            print(f"[WARNING] Adapter on {adapter_device}, parameters on {first_param_device}")
        
        # Test 2: Move to GPU if available
        if torch.cuda.is_available():
            print("\n[TEST] Moving policy to CUDA...")
            policy.to('cuda')
            
            adapter_device = policy.adapter.weight.device
            first_param_device = next(policy.parameters()).device
            
            print(f"[CHECK] Adapter device after .to('cuda'): {adapter_device}")
            print(f"[CHECK] First parameter device: {first_param_device}")
            
            if adapter_device.type == 'cuda' and first_param_device.type == 'cuda':
                print("[OK] Adapter successfully moved to CUDA")
            else:
                print("[FAILED] Adapter or parameters not on CUDA")
                return False
                
            # Test 3: Forward pass with CUDA tensors
            print("\n[TEST] Running forward pass with CUDA tensors...")
            batch_size = 2
            obs_tensor = torch.randn(batch_size, obs_dim, device='cuda')
            action_masks = torch.ones(batch_size, action_dim, dtype=torch.bool, device='cuda')
            
            try:
                with torch.no_grad():
                    actions, values, log_probs = policy(obs_tensor, deterministic=False, action_masks=action_masks)
                
                print(f"[OK] Forward pass successful!")
                print(f"  Actions shape: {actions.shape}")
                print(f"  Values shape: {values.shape}")
                print(f"  Log probs shape: {log_probs.shape}")
                
                # Check that outputs are on correct device
                if actions.device.type == 'cuda':
                    print("[OK] Outputs on CUDA device")
                else:
                    print("[WARNING] Outputs not on CUDA device")
                
                return True
                
            except RuntimeError as e:
                if "Expected all tensors to be on the same device" in str(e):
                    print(f"[FAILED] Device mismatch error: {e}")
                    return False
                else:
                    raise
        else:
            print("\n[SKIP] CUDA not available, skipping GPU tests")
            
            # Test forward pass on CPU
            print("\n[TEST] Running forward pass with CPU tensors...")
            batch_size = 2
            obs_tensor = torch.randn(batch_size, obs_dim)
            action_masks = torch.ones(batch_size, action_dim, dtype=torch.bool)
            
            with torch.no_grad():
                actions, values, log_probs = policy(obs_tensor, deterministic=False, action_masks=action_masks)
            
            print(f"[OK] Forward pass successful on CPU!")
            return True
        
    except Exception as e:
        print(f"\n[FAILED] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_adapter_with_mock_model():
    """Test adapter with a mock MaskablePPO model."""
    print("\n" + "=" * 70)
    print("Testing Adapter with Mock Model")
    print("=" * 70)
    
    try:
        import torch
        import numpy as np
        from gymnasium import spaces
        from hybrid_policy_with_adapter import HybridAgentPolicyWithAdapter
        
        print("\n[SETUP] Creating mock model with adapter policy...")
        
        # Create environment spaces
        obs_dim = 261
        action_dim = 6
        
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_dim,), dtype=np.float32
        )
        action_space = spaces.Discrete(action_dim)
        
        # Create policy
        def lr_schedule(progress):
            return 0.0003
        
        policy = HybridAgentPolicyWithAdapter(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule
        )
        
        # Test with CUDA if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n[TEST] Using device: {device}")
        
        policy.to(device)
        
        # Create test observation (261D as expected in Phase 3)
        obs = np.random.randn(obs_dim).astype(np.float32)
        action_mask = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
        
        print(f"\n[TEST] Running prediction...")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Action mask: {action_mask}")
        
        # Use the policy's predict method
        try:
            actions, _ = policy.predict(obs, deterministic=False, action_masks=action_mask)
            print(f"[OK] Prediction successful!")
            print(f"  Action: {actions}")
            
            # Check adapter stats
            stats = policy.get_adapter_stats()
            print(f"\n[OK] Adapter stats retrieved:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            return True
            
        except RuntimeError as e:
            if "Expected all tensors to be on the same device" in str(e):
                print(f"[FAILED] Device mismatch: {e}")
                return False
            else:
                raise
        
    except Exception as e:
        print(f"\n[FAILED] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("Device Mismatch Fix Verification")
    print("=" * 70)
    
    success1 = test_adapter_device_placement()
    success2 = test_adapter_with_mock_model()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Adapter device test: {'[PASSED]' if success1 else '[FAILED]'}")
    print(f"Mock model test: {'[PASSED]' if success2 else '[FAILED]'}")
    
    if success1 and success2:
        print("\n[SUCCESS] All device placement tests passed!")
        sys.exit(0)
    else:
        print("\n[FAILED] Some tests failed.")
        sys.exit(1)
