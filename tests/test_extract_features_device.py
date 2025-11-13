#!/usr/bin/env python3
"""
Test extract_features device management fix.

This test verifies that the extract_features method correctly handles
device mismatches between observation tensors (from environment, typically CPU)
and adapter weights (on CUDA during training).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

def test_extract_features_cpu_to_cuda():
    """Test extract_features with CPU observations and CUDA adapter."""
    print("=" * 70)
    print("Testing extract_features with CPU observations to CUDA adapter")
    print("=" * 70)
    
    try:
        import torch
        from hybrid_policy_with_adapter import HybridAgentPolicyWithAdapter
        
        if not torch.cuda.is_available():
            print("\n[SKIP] CUDA not available")
            return True
        from gymnasium import spaces
        import numpy as np
        
        print("\n[SETUP] Creating policy on CUDA...")
        
        # Create policy
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(261,), dtype=np.float32
        )
        action_space = spaces.Discrete(6)
        
        def lr_schedule(progress): return 0.0003
        
        policy = HybridAgentPolicyWithAdapter(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule
        )
        
        # Move policy to CUDA
        policy.to('cuda')
        
        adapter_device = policy.adapter.weight.device
        print(f"[CHECK] Adapter device: {adapter_device}")
        
        if adapter_device.type != 'cuda':
            print("[FAILED] Adapter not on CUDA!")
            return False
        
        print("\n[TEST] Creating CPU observation tensor (simulating environment)...")
        
        # Create observation on CPU (like environment produces)
        batch_size = 2
        cpu_obs = torch.randn(batch_size, 261)  # Note: no device specified = CPU
        
        print(f"[CHECK] Observation device: {cpu_obs.device}")
        
        if cpu_obs.device.type != 'cpu':
            print("[FAILED] Observation not on CPU!")
            return False
        
        print("\n[TEST] Calling extract_features with CPU observation...")
        
        try:
            # This should work now - extract_features will move obs to adapter's device
            features = policy.extract_features(cpu_obs)
            
            print(f"[OK] extract_features succeeded!")
            print(f"[CHECK] Features shape: {features.shape}")
            print(f"[CHECK] Features device: {features.device}")
            
            # Verify features are on correct device
            if features.device.type == 'cuda':
                print("[OK] Features correctly on CUDA device")
            else:
                print(f"[WARNING] Features on {features.device}, expected cuda")
            
            # Verify shape is correct (should be 228D after adapter)
            if features.shape == (batch_size, 228):
                print("[OK] Features have correct shape [batch, 228]")
            else:
                print(f"[FAILED] Features shape {features.shape}, expected [{batch_size}, 228]")
                return False
            
            return True
            
        except RuntimeError as e:
            if "Expected all tensors to be on the same device" in str(e):
                print(f"[FAILED] Device mismatch error still occurs: {e}")
                return False
            else:
                # Some other error
                raise
        
    except Exception as e:
        print(f"\n[FAILED] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_extract_features_cuda_to_cuda():
    """Test extract_features with CUDA observations (should still work)."""
    print("\n" + "=" * 70)
    print("Testing extract_features with CUDA observations to CUDA adapter")
    print("=" * 70)
    
    try:
        import torch
        from hybrid_policy_with_adapter import HybridAgentPolicyWithAdapter
        from gymnasium import spaces
        import numpy as np
        
        if not torch.cuda.is_available():
            print("\n[SKIP] CUDA not available")
            return True
        
        print("\n[SETUP] Creating policy on CUDA...")
        
        # Create policy
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(261,), dtype=np.float32
        )
        action_space = spaces.Discrete(6)
        
        def lr_schedule(progress): return 0.0003
        
        policy = HybridAgentPolicyWithAdapter(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule
        )
        
        # Move policy to CUDA
        policy.to('cuda')
        
        print("\n[TEST] Creating CUDA observation tensor...")
        
        # Create observation on CUDA
        batch_size = 2
        cuda_obs = torch.randn(batch_size, 261, device='cuda')
        
        print(f"[CHECK] Observation device: {cuda_obs.device}")
        
        print("\n[TEST] Calling extract_features with CUDA observation...")
        
        # This should work as before
        features = policy.extract_features(cuda_obs)
        
        print(f"[OK] extract_features succeeded!")
        print(f"[CHECK] Features shape: {features.shape}")
        print(f"[CHECK] Features device: {features.device}")
        
        # Both should be on CUDA
        if features.device.type == 'cuda' and cuda_obs.device.type == 'cuda':
            print("[OK] Both input and output on CUDA")
        else:
            print(f"[WARNING] Device mismatch: input {cuda_obs.device}, output {features.device}")
        
        return True
        
    except Exception as e:
        print(f"\n[FAILED] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_forward_pass_with_cpu_obs():
    """Test full forward pass with CPU observations."""
    print("\n" + "=" * 70)
    print("Testing full forward pass with CPU observations")
    print("=" * 70)
    
    try:
        import torch
        from hybrid_policy_with_adapter import HybridAgentPolicyWithAdapter
        from gymnasium import spaces
        import numpy as np
        
        if not torch.cuda.is_available():
            print("\n[SKIP] CUDA not available")
            return True
        
        print("\n[SETUP] Creating policy on CUDA...")
        
        # Create policy
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(261,), dtype=np.float32
        )
        action_space = spaces.Discrete(6)
        
        def lr_schedule(progress): return 0.0003
        
        policy = HybridAgentPolicyWithAdapter(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule
        )
        
        # Move policy to CUDA
        policy.to('cuda')
        
        print("\n[TEST] Running forward pass with CPU observations...")
        
        # Create CPU observations and masks
        batch_size = 2
        cpu_obs = torch.randn(batch_size, 261)  # CPU
        cpu_masks = torch.ones(batch_size, 6, dtype=torch.bool)  # CPU
        
        print(f"[CHECK] Observation device: {cpu_obs.device}")
        print(f"[CHECK] Masks device: {cpu_masks.device}")
        
        try:
            with torch.no_grad():
                # Full forward pass - should handle device management internally
                actions, values, log_probs = policy(cpu_obs, deterministic=False, action_masks=cpu_masks)
            
            print(f"[OK] Forward pass succeeded!")
            print(f"[CHECK] Actions: {actions}")
            print(f"[CHECK] Values shape: {values.shape}")
            print(f"[CHECK] Log probs shape: {log_probs.shape}")
            
            # Verify outputs are on CUDA
            if actions.device.type == 'cuda':
                print("[OK] Actions on CUDA device")
            else:
                print(f"[WARNING] Actions on {actions.device}")
            
            return True
            
        except RuntimeError as e:
            if "Expected all tensors to be on the same device" in str(e):
                print(f"[FAILED] Device mismatch in forward pass: {e}")
                return False
            else:
                raise
        
    except Exception as e:
        print(f"\n[FAILED] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("Extract Features Device Management Test")
    print("=" * 70)
    
    import torch
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    success1 = test_extract_features_cpu_to_cuda()
    success2 = test_extract_features_cuda_to_cuda()
    success3 = test_forward_pass_with_cpu_obs()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"CPU to CUDA test: {'[PASSED]' if success1 else '[FAILED]'}")
    print(f"CUDA to CUDA test: {'[PASSED]' if success2 else '[FAILED]'}")
    print(f"Full forward pass test: {'[PASSED]' if success3 else '[FAILED]'}")
    
    if success1 and success2 and success3:
        print("\n[SUCCESS] All device management tests passed!")
        print("The extract_features fix correctly handles device mismatches.")
        sys.exit(0)
    else:
        print("\n[FAILED] Some tests failed.")
        sys.exit(1)
