#!/usr/bin/env python3
"""
Integration Test: Phase 3 Training Device Fix

Tests that Phase 3 training can start without device mismatch errors
after the adapter device placement fix.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

def test_phase3_training_start():
    """Test that Phase 3 training can start without device errors."""
    print("=" * 70)
    print("Phase 3 Training Device Fix Integration Test")
    print("=" * 70)
    
    try:
        # Import training script components
        from train_phase3_llm import setup_phase3_training, load_config
        from hybrid_policy_with_adapter import HybridAgentPolicyWithAdapter
        import torch
        
        print("\n[SETUP] Loading configuration...")
        
        # Load config
        config = load_config('config/phase3_config.yaml')
        
        # Override for quick test
        config['test_mode'] = True
        config['market'] = 'NQ'
        config['non_interactive'] = True
        config['total_timesteps'] = 100  # Very short run
        config['n_envs'] = 1  # Single environment
        config['device'] = 'auto'  # Auto-detect device
        
        print(f"[CONFIG] Test mode: {config['test_mode']}")
        print(f"[CONFIG] Device: {config['device']}")
        print(f"[CONFIG] Timesteps: {config['total_timesteps']}")
        
        print("\n[TEST] Setting up Phase 3 training...")
        
        # Setup training (this creates the model and environments)
        result = setup_phase3_training(config)
        
        if result is None:
            print("[FAILED] setup_phase3_training returned None")
            return False
            
        model, env, hybrid_agent, vec_norm = result
        
        print(f"\n[CHECK] Model device: {model.device}")
        print(f"[CHECK] Policy type: {type(model.policy).__name__}")
        
        # Verify adapter is on correct device
        if isinstance(model.policy, HybridAgentPolicyWithAdapter):
            adapter_device = model.policy.adapter.weight.device
            print(f"[CHECK] Adapter device: {adapter_device}")
            
            # Check if adapter is on same device as model
            if adapter_device.type == model.device.type:
                print("[OK] Adapter on same device as model")
            else:
                print(f"[WARNING] Adapter on {adapter_device}, model on {model.device}")
        else:
            print("[WARNING] Policy is not HybridAgentPolicyWithAdapter")
        
        print("\n[TEST] Attempting short training run...")
        
        # Try to run a few training steps
        try:
            model.learn(
                total_timesteps=10,  # Just a few steps
                callback=None,
                tb_log_name="test_device_fix"
            )
            
            print("[SUCCESS] Training started without device errors!")
            print("[OK] The device mismatch fix is working")
            
            # Cleanup
            env.close()
            
            return True
            
        except RuntimeError as e:
            if "Expected all tensors to be on the same device" in str(e):
                print(f"[FAILED] Device mismatch error still occurring: {e}")
                return False
            else:
                # Some other error, re-raise
                raise
        
    except Exception as e:
        print(f"\n[FAILED] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_adapter_device_with_cuda():
    """Specifically test adapter device placement with CUDA."""
    print("\n" + "=" * 70)
    print("CUDA Device Placement Test")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("\n[SKIP] CUDA not available")
        return True
    
    try:
        import torch
        from hybrid_policy_with_adapter import HybridAgentPolicyWithAdapter
        from gymnasium import spaces
        import numpy as np
        
        print("\n[SETUP] Creating policy...")
        
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
        
        print("[CHECK] Initial adapter device:", policy.adapter.weight.device)
        
        # Move to CUDA
        print("\n[TEST] Moving to CUDA...")
        policy.to('cuda')
        
        adapter_device = policy.adapter.weight.device
        print("[CHECK] Adapter device after .to('cuda'):", adapter_device)
        
        if adapter_device.type == 'cuda':
            print("[OK] Adapter successfully moved to CUDA")
        else:
            print("[FAILED] Adapter not on CUDA")
            return False
        
        # Test forward pass
        print("\n[TEST] Forward pass with CUDA...")
        obs_tensor = torch.randn(2, 261, device='cuda')
        action_masks = torch.ones(2, 6, dtype=torch.bool, device='cuda')
        
        with torch.no_grad():
            actions, values, log_probs = policy(obs_tensor, action_masks=action_masks)
        
        print("[OK] Forward pass successful on CUDA")
        print(f"  Actions device: {actions.device}")
        print(f"  Values device: {values.device}")
        
        return True
        
    except Exception as e:
        print(f"\n[FAILED] CUDA test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("Phase 3 Device Fix Integration Tests")
    print("=" * 70)
    
    import torch
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    success1 = test_phase3_training_start()
    success2 = test_adapter_device_with_cuda()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Phase 3 training test: {'[PASSED]' if success1 else '[FAILED]'}")
    print(f"CUDA adapter test: {'[PASSED]' if success2 else '[FAILED]'}")
    
    if success1 and success2:
        print("\n[SUCCESS] All integration tests passed!")
        print("The device mismatch fix should resolve the training error.")
        sys.exit(0)
    else:
        print("\n[FAILED] Some integration tests failed.")
        sys.exit(1)
