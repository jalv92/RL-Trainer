#!/usr/bin/env python3
"""
Minimal debug script to identify the action masking issue.
"""

import sys
sys.path.append('src')

def main():
    print("Phase 3 Action Masking - Minimal Debug")
    print("="*50)
    
    print("ERROR ANALYSIS:")
    print("Error: cannot reshape array of size 12 into shape (4,6)")
    print("")
    print("This means:")
    print("- We have 12 elements (2 environments × 6 actions)")
    print("- Buffer expects 24 elements (4 environments × 6 actions)")
    print("")
    print("ROOT CAUSE:")
    print("The SB3 rollout buffer is configured for 4 environments")
    print("but the training is actually running with 2 environments")
    print("")
    print("WHERE TO LOOK:")
    print("1. In train_phase3_llm.py, check n_envs configuration")
    print("2. In model.learn(), check how buffer is initialized")
    print("3. Check if there's a hardcoded n_envs=4 somewhere")
    print("")
    print("EXPECTED FIX:")
    print("Ensure buffer.n_envs matches the actual n_envs parameter")
    print("This should be 2 in test mode, not 4")

if __name__ == "__main__":
    main()