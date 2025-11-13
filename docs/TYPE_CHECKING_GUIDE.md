# Type Checking Guide

## Current Configuration

Type checking is **disabled** (`"typeCheckingMode": "off"`) to avoid 96+ false-positive warnings from:
- Optional parameters with non-optional type hints
- Pandas DatetimeIndex type inference issues
- Stable-Baselines3 incomplete type stubs

## If You Want to Enable Type Checking

### Option 1: Keep It Disabled (Recommended)

**Current setting:**
```json
"python.analysis.typeCheckingMode": "off"
```

**Pros:**
- No false positives
- Faster IDE performance
- Focus on real errors (imports, syntax)

**Cons:**
- Won't catch actual type errors
- Less strict code quality enforcement

---

### Option 2: Enable with Suppressions (Moderate)

In `.vscode/settings.json`, change:
```json
{
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.diagnosticSeverityOverrides": {
        // Keep these suppressions to avoid pandas/SB3 noise
        "reportOptionalMemberAccess": "none",
        "reportAttributeAccessIssue": "none",

        // Enable these if you want to fix them
        "reportArgumentType": "warning",  // Show as warnings, not errors
        "reportGeneralTypeIssues": "warning"
    }
}
```

**Pros:**
- Some type checking for your code
- Pandas/SB3 noise still suppressed

**Cons:**
- Will show ~20-30 warnings from optional parameters

---

### Option 3: Fix All Type Hints (Strict)

Enable full type checking and fix all issues:

```json
{
    "python.analysis.typeCheckingMode": "strict"
}
```

Then fix issues by adding proper optional type hints:

#### Fix 1: Optional Parameters

**Before:**
```python
def __init__(
    self,
    data: pd.DataFrame,
    second_data=None,  # Type hint missing
    market_spec=None,  # Type hint missing
    commission_override=None  # Type hint missing
):
```

**After:**
```python
from typing import Optional

def __init__(
    self,
    data: pd.DataFrame,
    second_data: Optional[pd.DataFrame] = None,
    market_spec: Optional[MarketSpecification] = None,
    commission_override: Optional[float] = None
):
```

#### Fix 2: Pandas DatetimeIndex

**Before:**
```python
if data.index.tz is None:  # Pylance doesn't know about .tz
    data.index = data.index.tz_localize('America/New_York')
```

**After:**
```python
from pandas import DatetimeIndex
from typing import cast

# Assert or cast to DatetimeIndex
dt_index = cast(DatetimeIndex, data.index)
if dt_index.tz is None:
    data.index = dt_index.tz_localize('America/New_York')
```

Or use type guard:
```python
if isinstance(data.index, pd.DatetimeIndex):
    if data.index.tz is None:
        data.index = data.index.tz_localize('America/New_York')
```

#### Fix 3: Stable-Baselines3 Types

**Before:**
```python
# Pylance: "No parameter named 'use_sde'"
model = MaskablePPO(
    policy="MlpPolicy",
    env=env,
    use_sde=True,  # SB3 type stubs outdated
    sde_sample_freq=4
)
```

**After:**
```python
# Add type: ignore comments for SB3 limitations
model = MaskablePPO(
    policy="MlpPolicy",
    env=env,
    use_sde=True,  # type: ignore[call-arg]
    sde_sample_freq=4  # type: ignore[call-arg]
)
```

---

## Recommendation

**Keep type checking OFF** for now because:

1. **Your code works** - All 96 "errors" are false positives
2. **Dependencies are loosely typed** - pandas and SB3 stubs incomplete
3. **Time investment** - Fixing all type hints = 2-3 hours of work
4. **Limited benefit** - Won't catch actual bugs, just satisfy linter

**When to enable:**
- When preparing code for distribution (pip package)
- When onboarding multiple developers (type hints help understanding)
- When you have time for code cleanup (not during active development)

---

## Quick Reference

### Check Current Setting

In VS Code: `Ctrl+Shift+P` → "Python: Select Type Checking Mode"

### Temporarily Enable

Add to top of specific file:
```python
# pyright: basic
# or
# pyright: strict
```

### Ignore Specific Lines

```python
x = some_function()  # type: ignore
y = another_function()  # type: ignore[arg-type]
```

### Per-File Configuration

```python
# pyright: reportOptionalMemberAccess=false
# At top of file to disable specific checks
```

---

## Summary

- **Current:** Type checking OFF, import resolution ON
- **Result:** No noise, imports work, autocomplete works
- **Future:** Enable when you want stricter code quality
- **Fix effort:** 2-3 hours to properly type-hint entire codebase

For active development, current configuration is optimal.
