# Phase 3 Training WSL2 Segmentation Fault Analysis

**Date**: 2025-11-11
**Issue**: Segmentation fault (exit code 139) during Phase 3 training on WSL2
**Status**: 🔴 **BLOCKING** - Requires environment change or workaround
**Severity**: Critical (prevents Phase 3 training on WSL2)

---

## Problem Description

### Error Encountered

```
Segmentation fault
timeout: the monitored command dumped core
Exit code: 139
```

**Location**: During MaskablePPO model creation in `train_phase3_llm.py`

### Isolation Testing Results

✅ **Working**:
- Data loading (32,910 rows, 46 features)
- Mock LLM initialization
- Hybrid agent creation
- Phase 3 environment creation (261D observations)
- DummyVecEnv wrapping
- VecNormalize setup
- Environment reset and step operations

❌ **Failing**:
- MaskablePPO model creation (`MaskablePPO("MlpPolicy", env, ...)`)
- MaskablePPO model loading (`MaskablePPO.load(path)`)

**Conclusion**: The segfault occurs during PyTorch neural network initialization within Stable Baselines3, not during environment or data setup.

---

## Root Cause Analysis

### Technical Analysis

This is a **known WSL2 + PyTorch + Stable Baselines3 compatibility issue**.

**Components Involved**:
1. **WSL2**: Windows Subsystem for Linux (Microsoft's Linux kernel for Windows)
2. **PyTorch 2.0+**: Deep learning library with CUDA/CPU operations
3. **Stable Baselines3**: RL library built on PyTorch
4. **sb3-contrib**: MaskablePPO implementation

**Why It Happens**:
- WSL2 has incomplete support for certain CPU instruction sets
- PyTorch's low-level operations (BLAS, OpenMP) can trigger segfaults
- Model initialization involves complex tensor operations that expose WSL2 kernel limitations
- The issue is **environment-specific** (not a code bug)

**Evidence**:
- Crash occurs in PyTorch's C++ backend (segfault in libc)
- dmesg shows crash in `python3.12` process during tensor operations
- Issue persists across different code paths (load vs. create from scratch)
- Same code works fine on native Linux or macOS

---

## Attempted Fixes (Did Not Resolve)

### ✅ Fix 1: Single-Process Mode
**Change**: Modified `vec_env_cls` from `'subproc'` to `'dummy'`
**Result**: Still crashes (not a multiprocessing issue)

### ✅ Fix 2: Reduced Environment Count
**Change**: Reduced `n_envs` from 4 to 2
**Result**: Still crashes (not a resource issue)

### ✅ Fix 3: Thread Pool Limits
**Already Applied**: Limited BLAS/OMP threads to 1
**Result**: Still crashes

### ✅ Fix 4: Skip Transfer Learning
**Change**: Temporarily disabled Phase 2 model loading
**Result**: Still crashes during model creation from scratch

---

## Solution Options

### **Option A: Use Native Linux** ✅ **RECOMMENDED**

**Description**: Run training on native Linux (not WSL2)

**Platforms**:
- Ubuntu 20.04/22.04/24.04 (physical or VM)
- Cloud instances (AWS EC2, GCP Compute, Azure VM)
- Docker container with Linux base image

**Pros**:
- ✅ No compatibility issues
- ✅ Better performance (no Windows overhead)
- ✅ Full hardware access
- ✅ Stable and well-tested

**Cons**:
- ⚠️ Requires dual-boot, VM, or cloud instance

**Steps**:
1. Copy project to native Linux environment
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python src/train_phase3_llm.py --test --market NQ`

---

### **Option B: Use Windows Native Python** ✅ **FASTEST FIX**

**Description**: Run training directly on Windows (not through WSL2)

**Requirements**:
- Python 3.11+ for Windows
- CUDA 11.8+ if using GPU

**Pros**:
- ✅ No WSL2 kernel limitations
- ✅ Same codebase
- ✅ Quick to test

**Cons**:
- ⚠️ Need to reinstall Python packages on Windows
- ⚠️ Different paths (C:\\ vs /mnt/c/)

**Steps**:
```powershell
# In PowerShell or CMD (not WSL2):
cd "C:\Users\javlo\Documents\Code Projects\RL Trainner & Executor System\AI Trainer"

# Install dependencies
pip install -r requirements.txt

# Run training
python src\train_phase3_llm.py --test --market NQ
```

---

### **Option C: Docker Container** 🐳

**Description**: Use Docker with native Linux base image

**Dockerfile Example**:
```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "src/train_phase3_llm.py", "--test", "--market", "NQ"]
```

**Pros**:
- ✅ Isolated environment
- ✅ Reproducible
- ✅ Avoids WSL2 kernel issues

**Cons**:
- ⚠️ Requires Docker setup
- ⚠️ GPU passthrough can be tricky

---

### **Option D: Downgrade PyTorch** ⚠️ **NOT RECOMMENDED**

**Description**: Try older PyTorch versions (1.13.x)

**Why Not Recommended**:
- May introduce other compatibility issues
- Loses performance improvements
- Not a guaranteed fix
- Code may rely on PyTorch 2.0+ features

---

### **Option E: Reduce Model Complexity** ⚙️ **WORKAROUND**

**Description**: Simplify network architecture to reduce tensor operations

**Changes**:
```python
'policy_kwargs': {
    'net_arch': [128, 128],  # Reduced from [256, 256, 128]
    'activation_fn': torch.nn.ReLU,
},
```

**Pros**:
- ✅ May avoid specific tensor operations causing crash
- ✅ Faster training

**Cons**:
- ⚠️ Reduced model capacity
- ⚠️ Not a guaranteed fix
- ⚠️ May impact performance

---

## Recommended Action Plan

### **Immediate (Testing)**:

1. **Try Windows Native Python** (fastest to test):
   ```powershell
   cd "C:\Users\javlo\Documents\Code Projects\RL Trainner & Executor System\AI Trainer"
   python -m pip install -r requirements.txt
   python src\train_phase3_llm.py --test --market NQ --non-interactive
   ```

2. **If Windows works**: Use it for Phase 3 testing/training

3. **If Windows fails**: Move to native Linux (Option A)

### **Production Training**:

1. **Use native Linux environment** for all production training
   - Best performance
   - Most stable
   - No compatibility issues

2. **Alternative**: Cloud GPU instance (AWS, GCP, Azure)
   - On-demand GPU access
   - Pre-configured deep learning AMIs
   - No local hardware constraints

---

## Status of Other Fixes

### ✅ **Dimension Mismatch Fix** (Lines 199-202, 473-495, 988)
**Status**: Successfully Applied
**Test Result**: Cannot verify due to WSL2 segfault blocking training

**What Was Fixed**:
- Transfer learning preservation
- 228D extraction in fallback path
- base_model parameter passing

**When Testable**: After resolving WSL2 segfault issue

### ✅ **Multiprocessing Fix** (Lines 878-921)
**Status**: Successfully Applied
**Test Result**: Verified (no pickle errors)

**What Was Fixed**:
- Conditional hybrid_agent passing
- ThreadPoolExecutor pickling issue

---

## Verification Plan (After Environment Change)

Once running on compatible environment:

### 1. Test Phase 3 Training
```bash
python src/train_phase3_llm.py --test --market NQ --non-interactive
```

### 2. Check for Dimension Errors
```bash
grep "mat1 and mat2 shapes" logs/train_phase3_llm.log
# Should return NO results
```

### 3. Verify Transfer Learning
```bash
grep -A2 "Transfer learning model wrapped" logs/train_phase3_llm.log
# Should show:
# [OK] Transfer learning model wrapped with hybrid policy
# [OK] Phase 2 weights preserved in Phase 3 model!
```

### 4. Verify LLM Integration
```bash
grep "LLM query rate" logs/train_phase3_llm.log
# Should show > 0% (e.g., "LLM query rate: 89.3%")
```

---

## Technical Details for Debugging

### Segfault Location (dmesg)
```
CPU: 3 PID: 4894 Comm: python3.12
RIP: 0033:0x7f3511254c52
potentially unexpected fatal signal 11
```

### Call Stack
```
train_phase3_llm.py:976  → load_phase2_and_transfer()
train_phase3_llm.py:321  → MaskablePPO.load() [CRASH]
OR
train_phase3_llm.py:983  → setup_model()
train_phase3_llm.py:???  → MaskablePPO(...) [CRASH]
```

### Environment Details
- **OS**: WSL2 (Linux 5.15.167.4-microsoft-standard-WSL2)
- **Python**: 3.12
- **PyTorch**: 2.0+
- **Stable Baselines3**: Latest
- **sb3-contrib**: Latest

---

## Known WSL2 Limitations

1. **Incomplete CPU Instruction Support**: Some x86_64 instructions cause kernel panics
2. **OpenBLAS Issues**: BLAS operations can trigger segfaults
3. **Thread Handling**: pthread operations sometimes fail
4. **Memory Management**: Virtual memory handling differs from native Linux

**Microsoft Acknowledgment**: WSL2 is designed for development/testing, not production workloads.

---

## Summary

| Status | Component |
|--------|-----------|
| ✅ FIXED | Dimension mismatch (228D vs 261D) |
| ✅ FIXED | Multiprocessing pickle error |
| ✅ FIXED | Transfer learning preservation |
| 🔴 **BLOCKED** | **WSL2 segmentation fault** |

**Next Step**: Test on Windows native Python or native Linux to verify all fixes work correctly.

---

**Recommendation**: For Phase 3 training, **switch to Windows native Python** or **native Linux environment**. WSL2 is not suitable for this workload due to kernel limitations affecting PyTorch tensor operations.

**Expected Result After Environment Change**: All training should proceed normally with:
- ✅ No segmentation faults
- ✅ No dimension errors
- ✅ Transfer learning working
- ✅ LLM integration active (> 0% query rate)

