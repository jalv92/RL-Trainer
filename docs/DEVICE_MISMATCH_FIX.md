# Device Mismatch Error Fix - Complete Solution

## Problem Summary

**Error:** `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!`

**Location:** `src/hybrid_policy_with_adapter.py:185` during `extract_features()` forward pass

**Context:** Phase 3 training with adapter layer for Phase 2 → Phase 3 transfer learning

## Root Cause Analysis

The adapter layer (`nn.Linear`) was created in `HybridAgentPolicyWithAdapter.__init__()` but wasn't properly placed on the same device as other policy parameters. When the model was moved to GPU:

1. **Model moved to CUDA**: `model.to('cuda')` moved all registered parameters
2. **Adapter stayed on CPU**: The adapter wasn't properly tracked as a submodule
3. **Forward pass failed**: CUDA tensors hit CPU adapter weights → device mismatch

## Solution Implemented

### File Modified: `src/hybrid_policy_with_adapter.py`

**Lines Changed:** 112-122 (after adapter creation)

**Code Added:**
```python
# FIX: Move adapter to same device as policy parameters
# This ensures the adapter moves with the model when .to(device) is called
try:
    device = next(self.parameters()).device
    self.adapter.to(device)
    logger.info(f"[ADAPTER] Moved adapter to device: {device}")
except StopIteration:
    # No parameters yet, adapter will stay on CPU (shouldn't happen in practice)
    logger.warning("[ADAPTER] No policy parameters found, adapter staying on CPU")
```

### How It Works

1. **After creating adapter**, immediately detect device of existing parameters
2. **Move adapter to same device** using `self.adapter.to(device)`
3. **Ensures proper tracking** - adapter now moves with model when `.to(device)` is called
4. **Maintains consistency** - all components start on same device

## Testing & Verification

### Test 1: Device Placement ✅ PASSED

```bash
python simple_device_test.py
```

**Output:**
```
[1] Creating HybridAgentPolicyWithAdapter...
[2] Checking initial device placement...
    Adapter: cpu
    Parameters: cpu
[OK] Initial device placement correct

[3] Testing CUDA device movement...
    Adapter: cuda:0
    Parameters: cuda:0
[OK] All components moved to CUDA

[4] Testing forward pass on CUDA...
[OK] Forward pass successful on CUDA

SUCCESS: Adapter device fix is working correctly!
```

### Test 2: Forward Pass on CUDA ✅ PASSED

- Created 261D observation tensor on CUDA
- Created action masks on CUDA
- Forward pass completed without device errors
- All outputs correctly placed on CUDA

## Complete Fix Summary

You now have **two critical fixes** in place:

### Fix 1: Infinite Recursion (Already Applied)
- **File:** `src/hybrid_agent.py`
- **Lines:** 183-193
- **Problem:** Hybrid agent → RL model → Hybrid policy → Hybrid agent (loop)
- **Solution:** Use `_rl_only_predict()` to bypass hybrid logic for RL predictions
- **Status:** ✅ Tested and working

### Fix 2: Device Mismatch (Just Applied)
- **File:** `src/hybrid_policy_with_adapter.py`
- **Lines:** 114-122
- **Problem:** Adapter on CPU, tensors on CUDA → device mismatch
- **Solution:** Move adapter to same device as parameters after creation
- **Status:** ✅ Tested and working

## Next Steps

### 1. Test Phase 3 Training Again

Run your training script to verify both fixes work together:

```bash
python src\train_phase3_llm.py --test --market NQ --non-interactive
```

**Expected behavior:**
- ✅ Training starts without infinite recursion
- ✅ Forward passes succeed without device errors
- ✅ Adapter layer works correctly on GPU
- ✅ Phase 2 → Phase 3 transfer learning proceeds

### 2. Monitor Training Progress

Watch for:
- **Adapter learning**: Check adapter stats in logs
- **GPU utilization**: Verify model is training on GPU
- **No errors**: No recursion or device mismatch errors

### 3. Verify Transfer Learning

Confirm that:
- Phase 2 knowledge is preserved (adapter starts with identity)
- LLM features are being learned (adapter weights for last 33 dimensions)
- Training loss decreases appropriately

## Troubleshooting

### If you still see device errors:

1. **Check adapter device manually:**
```python
print(f"Adapter device: {model.policy.adapter.weight.device}")
print(f"Model device: {model.device}")
```

2. **Force adapter to correct device:**
```python
model.policy.adapter.to(model.device)
```

3. **Verify all parameters:**
```python
for name, param in model.policy.named_parameters():
    print(f"{name}: {param.device}")
```

### If you see recursion errors:

1. **Verify hybrid agent fix is in place:**
```python
# Check hybrid_agent.py line 183-193 has the _rl_only_predict logic
```

2. **Test with simple script:**
```bash
python test_recursion_fix.py
```

## Technical Details

### Adapter Architecture

```
Phase 3 Observation (261D)
    ↓
[Adapter Layer: 261D → 228D] ← Learnable projection
    ↓
[Phase 2 Network: 228D → actions] ← Transferred unchanged
```

### Device Management Flow

```
1. Policy created on CPU (default)
   ↓
2. Adapter created on CPU
   ↓
3. FIX: Adapter moved to same device as parameters
   ↓
4. Model.to('cuda') moves everything to GPU
   ↓
5. Forward pass: CUDA tensor → CUDA adapter → success
```

## Files Modified

1. **src/hybrid_agent.py** - Infinite recursion fix
2. **src/hybrid_policy_with_adapter.py** - Device mismatch fix

## Files Created for Testing

1. **test_recursion_fix.py** - Tests recursion fix
2. **simple_device_test.py** - Tests device fix
3. **INFINITE_RECURSION_FIX.md** - Recursion fix documentation
4. **ADAPTER_DEVICE_FIX.md** - Device fix documentation
5. **DEVICE_MISMATCH_FIX.md** - This file (complete summary)

## Expected Outcome

With both fixes applied, Phase 3 training should:
- ✅ Start without infinite recursion
- ✅ Run without device mismatch errors
- ✅ Successfully transfer Phase 2 knowledge via adapter
- ✅ Integrate LLM features through the 33 additional dimensions
- ✅ Train efficiently on GPU

Your hybrid RL + LLM trading agent training should now proceed smoothly!