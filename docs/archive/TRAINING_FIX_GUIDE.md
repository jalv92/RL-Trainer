# AI Trading Model Retraining Guide
## Fixing Invalid Action Predictions (Position Management When Flat)

**Date:** 2025-11-07
**Issue:** Model predicts MOVE_TO_BE 76.4% (expected 2.0%) and HOLD only 0.9% (expected 86.5%)
**Root Cause:** Training environment lacks action masking and invalid action penalties

---

## Problem Summary

### Current Behavior (BROKEN)
```
Action Distribution (100 predictions):
- HOLD (0):         1 (  0.9%) ❌ Expected: 86.5%
- BUY (1):          0 (  0.0%) ✅ Expected: 5.0%
- SELL (2):        11 ( 10.0%) ✅ Expected: 5.0%
- MOVE_TO_BE (3):  84 ( 76.4%) ❌ Expected: 2.0%
- ENABLE_TRAIL (4): 0 (  0.0%) ✅ Expected: 1.0%
- DISABLE_TRAIL (5): 14 ( 12.7%) ✅ Expected: 0.5%

Position Management Usage: 89.1% (Target: ~13.5%)
```

### Root Causes
1. **No action masking during training** - Model was allowed to select PM actions when position = FLAT
2. **No penalty for invalid actions** - Environment didn't penalize invalid action selections strongly
3. **Weak position signal** - Position state is only 1 of 225 features (0.44%)
4. **Model collapsed to local optimum** - Learned "always predict action 3" as safe behavior

---

## Solution Overview

### Phase 1: Immediate Mitigation (COMPLETED ✅)
- [x] Added inference-time action masking in `model_manager.py`
- [x] Prevents invalid predictions even with broken model
- [x] Forces HOLD when PM action predicted while flat

### Phase 2: Model Retraining (REQUIRED)
- [ ] Implement action masking in training environment
- [ ] Add large penalties for invalid actions
- [ ] Enhance observation space with explicit validity features
- [ ] Retrain model for 5M+ timesteps with fixes

---

## Detailed Fix Instructions

### 1. Training Environment Updates

#### File: `trading_env.py` (Gym Environment)

**A. Add Action Masking Method:**

```python
def action_masks(self):
    """
    Return binary mask indicating valid actions

    Returns:
        np.array([6]): 1 = valid action, 0 = invalid action
    """
    if self.position == 0:  # FLAT
        # Only HOLD (0), BUY (1), SELL (2) are valid when flat
        # Position management actions (3, 4, 5) are INVALID
        return np.array([1, 1, 1, 0, 0, 0], dtype=np.int8)
    else:  # LONG or SHORT
        # All actions are valid when in position
        return np.array([1, 1, 1, 1, 1, 1], dtype=np.int8)
```

**B. Add Invalid Action Penalty in step():**

```python
def step(self, action):
    """
    Execute action and return next state

    CRITICAL: Add penalty for invalid actions BEFORE processing
    """
    # Check if action is invalid for current position state
    valid_actions = self.action_masks()

    if valid_actions[action] == 0:
        # CRITICAL: Large negative reward for invalid action
        reward = -10.0  # Must be large enough to discourage
        done = False
        truncated = False
        info = {
            'invalid_action': True,
            'action': action,
            'position': self.position,
            'reason': f"Action {action} invalid when position={self.position}"
        }

        # DO NOT execute invalid action - return current state
        return self._get_observation(), reward, done, truncated, info

    # ... rest of normal step logic ...
```

**C. Update Observation Space (Add Explicit Validity Features):**

```python
def _get_position_features(self):
    """
    Build position features with explicit action validity flags

    Returns:
        np.array([8]): Enhanced from 5 to 8 features
    """
    # Original 5 features
    position_value = float(self.position)  # -1, 0, or 1
    entry_price_ratio = self.entry_price / self.current_price if self.position != 0 else 1.0
    sl_distance_atr = abs(self.current_price - self.sl_price) / self.current_atr if self.position != 0 else 0.0
    tp_distance_atr = abs(self.tp_price - self.current_price) / self.current_atr if self.position != 0 else 0.0
    time_in_position = float(self.current_bar - self.entry_bar) if self.position != 0 else 0.0

    # NEW: 3 explicit action validity features (makes it OBVIOUS to network)
    can_enter = 1.0 if self.position == 0 else 0.0       # Can use BUY/SELL
    can_manage = 1.0 if self.position != 0 else 0.0      # Can use PM actions
    has_position = 1.0 if self.position != 0 else 0.0    # Has open position

    return np.array([
        position_value,
        entry_price_ratio,
        sl_distance_atr,
        tp_distance_atr,
        time_in_position,
        can_enter,      # NEW
        can_manage,     # NEW
        has_position    # NEW
    ], dtype=np.float32)
```

**D. Update Observation Space Shape:**

```python
def __init__(self):
    super().__init__()

    # Market features: 220 (11 features × 20 bars)
    # Position features: 8 (upgraded from 5)
    # Total: 228 (upgraded from 225)

    self.observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(228,),  # Changed from (225,)
        dtype=np.float32
    )

    self.action_space = spaces.Discrete(6)
```

---

### 2. Training Script Updates

#### File: `train.py` (Training Script)

**A. Import Required Libraries:**

```python
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
```

**B. Wrap Environment with ActionMasker:**

```python
def make_env():
    """Create training environment with action masking"""
    env = YourTradingEnv()  # Your custom gym environment
    return env

# Create environment
env = make_env()

# CRITICAL: Wrap with ActionMasker to enable action masking during training
def mask_fn(env):
    """Wrapper function to provide action masks to MaskablePPO"""
    return env.action_masks()

env = ActionMasker(env, mask_fn)

# Wrap with DummyVecEnv (required by SB3)
env = DummyVecEnv([lambda: env])

# Wrap with VecNormalize for observation normalization
env = VecNormalize(
    env,
    norm_obs=True,
    norm_reward=False,  # Don't normalize reward
    clip_obs=10.0,
    clip_reward=10.0
)
```

**C. Create MaskablePPO Model:**

```python
# Model configuration
model = MaskablePPO(
    policy='MlpPolicy',
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="./tensorboard_logs/"
)
```

**D. Train Model:**

```python
# Train for sufficient timesteps
total_timesteps = 5_000_000  # Minimum 5M timesteps

print(f"Starting training for {total_timesteps:,} timesteps...")
model.learn(
    total_timesteps=total_timesteps,
    callback=None,  # Add custom callbacks if needed
    log_interval=10,
    reset_num_timesteps=True,
    tb_log_name="maskable_ppo_trading"
)

# Save model
model.save("NQ_o1_fixed.zip")
print("✅ Model saved: NQ_o1_fixed.zip")

# Save VecNormalize stats (CRITICAL!)
env.save("phase2_position_mgmt_final_vecnorm_fixed.pkl")
print("✅ VecNormalize stats saved")
```

---

### 3. Configuration Updates

#### File: `config.py`

**Update observation size:**

```python
# Observation space
WINDOW_SIZE = 20                 # Number of bars in rolling window
MARKET_FEATURE_COUNT = 11        # Features per bar
TOTAL_MARKET_FEATURES = 220      # 11 × 20 bars

POSITION_FEATURE_COUNT = 8       # Updated from 5 to 8
TOTAL_OBSERVATION_SIZE = 228     # Updated from 225 (220 market + 8 position)
```

---

### 4. Reward Structure (CRITICAL)

**Recommended Reward Function:**

```python
def calculate_reward(self, action, position_before, position_after):
    """
    Calculate reward with strong penalties for invalid actions

    Args:
        action: Action taken (0-5)
        position_before: Position state before action
        position_after: Position state after action

    Returns:
        float: Reward value
    """
    # CRITICAL: Invalid action penalty (already handled in step())
    # This is a backup check
    if position_before == 0 and action in [3, 4, 5]:
        return -10.0  # Large penalty

    if position_before != 0 and action in [1, 2]:
        return -10.0  # Penalty for entry when already in position

    # Rewards for valid actions
    if action == 0:  # HOLD
        if position_before == 0:
            # Small positive reward for patience when flat
            return 0.01
        else:
            # Neutral when holding position
            return 0.0

    elif action in [1, 2]:  # BUY or SELL (entry actions)
        if position_after != 0:
            # Entry successful - reward based on outcome
            # This will be updated at trade exit
            return 0.1  # Small immediate reward
        else:
            # Entry failed
            return -1.0

    elif action == 3:  # MOVE_TO_BE
        if position_before != 0:
            # Successful move to breakeven
            return 0.5
        # Should never reach here due to masking
        return -10.0

    elif action == 4:  # ENABLE_TRAIL
        if position_before != 0:
            # Successful trail enable
            return 0.5
        return -10.0

    elif action == 5:  # DISABLE_TRAIL
        if position_before != 0:
            # Successful trail disable
            return 0.2
        return -10.0

    return 0.0
```

---

### 5. Pre-Training Validation Tests

**Run these tests BEFORE starting training to ensure fixes are working:**

#### Test 1: Action Masking
```python
def test_action_masking():
    """Verify action masks work correctly"""
    env = YourTradingEnv()

    # Test when FLAT
    env.reset()
    env.position = 0
    mask = env.action_masks()
    assert mask.tolist() == [1, 1, 1, 0, 0, 0], f"FLAT mask wrong: {mask}"
    print("✅ FLAT action mask correct")

    # Test when LONG
    env.position = 1
    mask = env.action_masks()
    assert mask.tolist() == [1, 1, 1, 1, 1, 1], f"LONG mask wrong: {mask}"
    print("✅ LONG action mask correct")

    # Test when SHORT
    env.position = -1
    mask = env.action_masks()
    assert mask.tolist() == [1, 1, 1, 1, 1, 1], f"SHORT mask wrong: {mask}"
    print("✅ SHORT action mask correct")
```

#### Test 2: Invalid Action Penalty
```python
def test_invalid_action_penalty():
    """Verify invalid actions are penalized"""
    env = YourTradingEnv()
    env.reset()
    env.position = 0  # FLAT

    # Try MOVE_TO_BE when flat (should be invalid)
    obs, reward, done, truncated, info = env.step(3)

    assert reward == -10.0, f"Invalid action penalty wrong: {reward}"
    assert info['invalid_action'] == True, "Invalid action flag not set"
    print("✅ Invalid action penalty correct")
    print(f"   Reward: {reward}")
    print(f"   Info: {info}")
```

#### Test 3: Observation Space
```python
def test_observation_space():
    """Verify observation shape is correct"""
    env = YourTradingEnv()
    obs, _ = env.reset()

    assert obs.shape == (228,), f"Observation shape wrong: {obs.shape}"
    print("✅ Observation shape correct: (228,)")

    # Check position features
    position_features = obs[-8:]
    print(f"   Position features: {position_features}")
```

#### Test 4: ActionMasker Wrapper
```python
def test_action_masker_wrapper():
    """Verify ActionMasker wrapper works"""
    from sb3_contrib.common.wrappers import ActionMasker

    env = YourTradingEnv()
    env = ActionMasker(env, lambda e: e.action_masks())

    obs, _ = env.reset()
    mask = env.action_masks()

    print("✅ ActionMasker wrapper working")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Action mask: {mask}")
```

**Run all tests:**
```bash
python test_training_fixes.py
```

**Expected output:**
```
✅ FLAT action mask correct
✅ LONG action mask correct
✅ SHORT action mask correct
✅ Invalid action penalty correct
   Reward: -10.0
   Info: {'invalid_action': True, ...}
✅ Observation shape correct: (228,)
   Position features: [0. 1. 0. 0. 0. 1. 0. 1.]
✅ ActionMasker wrapper working
   Observation shape: (228,)
   Action mask: [1 1 1 0 0 0]
```

---

### 6. Training Hyperparameters

**Recommended configuration for retraining:**

```python
# Model hyperparameters
LEARNING_RATE = 3e-4          # Standard for PPO
N_STEPS = 2048                # Steps per update
BATCH_SIZE = 64               # Minibatch size
N_EPOCHS = 10                 # Optimization epochs
GAMMA = 0.99                  # Discount factor
GAE_LAMBDA = 0.95             # GAE parameter
CLIP_RANGE = 0.2              # PPO clip range
ENT_COEF = 0.01               # Entropy coefficient (exploration)

# Training duration
TOTAL_TIMESTEPS = 5_000_000   # Minimum 5M timesteps
                              # May need 10M+ for convergence

# Data distribution target
# Ensure 86.5% of training timesteps have position=FLAT
# This matches expected HOLD frequency
```

---

### 7. Post-Training Validation

**After training completes, validate the new model:**

```python
def validate_trained_model():
    """
    Run 1000 episodes and check action distribution
    """
    from model_manager import ModelManager

    manager = ModelManager()
    manager.load_model("NQ_o1_fixed.zip", "phase2_position_mgmt_final_vecnorm_fixed.pkl")

    action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    flat_pm_count = 0  # Count PM actions when flat
    total_predictions = 0

    # Run test episodes
    env = YourTradingEnv()
    for episode in range(1000):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = manager.predict(obs)
            position = obs[-5]

            action_counts[action] += 1
            total_predictions += 1

            # Check for invalid PM actions when flat
            if abs(position) < 0.1 and action in [3, 4, 5]:
                flat_pm_count += 1

            obs, _, done, _, _ = env.step(action)

    # Print results
    print("\n" + "="*70)
    print("POST-TRAINING VALIDATION RESULTS")
    print("="*70)

    for action, count in action_counts.items():
        pct = (count / total_predictions) * 100
        print(f"Action {action}: {count:5d} ({pct:5.1f}%)")

    print(f"\nInvalid PM actions when flat: {flat_pm_count} ({flat_pm_count/total_predictions*100:.2f}%)")

    # Validation criteria
    hold_pct = (action_counts[0] / total_predictions) * 100
    pm_flat_pct = (flat_pm_count / total_predictions) * 100

    print("\nValidation Criteria:")
    print(f"  HOLD: {hold_pct:.1f}% (target: 80-90%)")
    print(f"  PM when flat: {pm_flat_pct:.2f}% (target: <1%)")

    if hold_pct >= 80 and pm_flat_pct < 1.0:
        print("\n✅ MODEL VALIDATION PASSED")
        return True
    else:
        print("\n❌ MODEL VALIDATION FAILED - Retrain required")
        return False
```

---

### 8. Deployment Steps

**Once validation passes:**

1. **Backup Current Model:**
   ```bash
   cp NQ_o1.zip NQ_o1_broken_backup.zip
   cp phase2_position_mgmt_final_vecnorm.pkl phase2_position_mgmt_final_vecnorm_broken_backup.pkl
   ```

2. **Deploy New Model:**
   ```bash
   cp NQ_o1_fixed.zip NQ_o1.zip
   cp phase2_position_mgmt_final_vecnorm_fixed.pkl phase2_position_mgmt_final_vecnorm.pkl
   ```

3. **Update Config:**
   ```python
   # In config.py
   TOTAL_OBSERVATION_SIZE = 228  # Updated from 225
   ```

4. **Test with Live Connection:**
   ```bash
   python ai_trading_bridge.py
   ```

5. **Monitor Performance:**
   - Check action distribution over 100 bars
   - Verify PM actions only when position open
   - Confirm HOLD is dominant action when flat

---

## Summary Checklist

### Required Changes:
- [ ] Add `action_masks()` method to training environment
- [ ] Add invalid action penalty (-10.0) in `step()` method
- [ ] Enhance observation space from 225 to 228 features
- [ ] Update `config.py` with new observation size
- [ ] Wrap training env with `ActionMasker`
- [ ] Use `MaskablePPO` algorithm (not regular PPO)
- [ ] Train for minimum 5M timesteps
- [ ] Run all pre-training validation tests
- [ ] Run post-training validation (must pass)
- [ ] Deploy and test with live NinjaTrader connection

### Success Criteria:
- ✅ HOLD action: 80-90% when position=FLAT
- ✅ PM actions: <1% when position=FLAT (ideally 0%)
- ✅ PM actions: 40-60% when position != FLAT
- ✅ BUY/SELL: 5-10% when position=FLAT
- ✅ All validation tests pass

---

## Troubleshooting

### Issue: ActionMasker not working
**Solution:** Ensure you imported from `sb3_contrib.common.wrappers`:
```python
from sb3_contrib.common.wrappers import ActionMasker
```

### Issue: Model still predicts invalid actions
**Solution:** Verify action mask is being passed to `model.predict()`:
```python
# Check this line exists in model_manager.py predict() method
action, _states = self.model.predict(normalized_obs, action_masks=action_mask, deterministic=True)
```

### Issue: Observation shape mismatch
**Solution:** Ensure all observation builders return (228,) not (225,):
- Update `trading_env.py` position features to 8
- Update `config.py` TOTAL_OBSERVATION_SIZE to 228
- Retrain from scratch (cannot load old VecNormalize with new shape)

### Issue: Training is slow
**Solution:**
- Use smaller `n_steps` (1024 instead of 2048)
- Use GPU if available: `device='cuda'`
- Reduce `n_epochs` (5 instead of 10)

### Issue: Model not learning HOLD
**Solution:**
- Increase entropy coefficient: `ent_coef=0.02`
- Add small positive reward for HOLD when flat: `+0.01`
- Ensure 86.5% of training data has position=FLAT

---

## Contact & Support

**Issue Tracker:** Document all retraining attempts with:
- Git commit hash of training code
- Hyperparameters used
- Training duration
- Final validation results
- Action distribution analysis

**Model Versions:**
- `NQ_o1.zip` - Original broken model (BACKUP)
- `NQ_o1_fixed.zip` - Retrained with action masking (NEW)

---

**Last Updated:** 2025-11-07
**Status:** Ready for retraining
**Priority:** HIGH - Model unusable in current state
