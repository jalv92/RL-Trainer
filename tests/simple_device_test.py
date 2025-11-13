#!/usr/bin/env python3
"""
Simple Device Fix Verification

Quick test to verify the adapter device fix works.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

def test_adapter_device_fix():
    """Test that adapter moves to correct device."""
    print("=" * 70)
    print("Simple Adapter Device Fix Test")
    print("=" * 70)
    
    try:
        import torch
        from hybrid_policy_with_adapter import HybridAgentPolicyWithAdapter
        from gymnasium import spaces
        import numpy as np
        
        print("\n[1] Creating HybridAgentPolicyWithAdapter...")
        
        # Create spaces
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(261,), dtype=np.float32
        )
        action_space = spaces.Discrete(6)
        
        def lr_schedule(progress): return 0.0003
        
        # Create policy
        policy = HybridAgentPolicyWithAdapter(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule
        )
        
        print("[2] Checking initial device placement...")
        adapter_device = policy.adapter.weight.device
        param_device = next(policy.parameters()).device
        print(f"    Adapter: {adapter_device}")
        print(f"    Parameters: {param_device}")
        
        if adapter_device != param_device:
            print("[FAILED] Adapter not on same device as parameters initially!")
            return False
        print("[OK] Initial device placement correct")
        
        # Test CUDA if available
        if torch.cuda.is_available():
            print("\n[3] Testing CUDA device movement...")
            
            policy.to('cuda')
            
            adapter_device = policy.adapter.weight.device
            param_device = next(policy.parameters()).device
            print(f"    Adapter: {adapter_device}")
            print(f"    Parameters: {param_device}")
            
            if adapter_device.type != 'cuda' or param_device.type != 'cuda':
                print("[FAILED] Not all components moved to CUDA!")
                return False
            print("[OK] All components moved to CUDA")
            
            # Test forward pass
            print("\n[4] Testing forward pass on CUDA...")
            obs = torch.randn(2, 261, device='cuda')
            masks = torch.ones(2, 6, dtype=torch.bool, device='cuda')
            
            try:
                with torch.no_grad():
                    actions, values, log_probs = policy(obs, action_masks=masks)
                print("[OK] Forward pass successful on CUDA")
            except RuntimeError as e:
                if "Expected all tensors to be on the same device" in str(e):
                    print(f"[FAILED] Device mismatch error: {e}")
                    return False
                raise
        else:
            print("\n[3] CUDA not available, skipping GPU tests")
            print("[4] Testing forward pass on CPU...")
            
            obs = torch.randn(2, 261)
            masks = torch.ones(2, 6, dtype=torch.bool)
            
            with torch.no_grad():
                actions, values, log_probs = policy(obs, action_masks=masks)
            print("[OK] Forward pass successful on CPU")
        
        print("\n" + "=" * 70)
        print("SUCCESS: Adapter device fix is working correctly!")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n[FAILED] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_adapter_device_fix()
    sys.exit(0 if success else 1)
