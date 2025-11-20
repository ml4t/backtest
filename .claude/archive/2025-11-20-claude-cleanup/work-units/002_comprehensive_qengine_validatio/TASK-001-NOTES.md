# TASK-001: VectorBT Pro Installation Notes

## Status: ⚠️ Partially Complete with Issues

### What Was Completed
✅ Virtual environment `.venv-vectorbt` created successfully
✅ VectorBT Pro base dependencies installed (numpy, pandas, numba, etc.)
✅ VectorBT Pro installed in editable mode from `resources/vectorbt.pro-main/`
✅ Data dependencies installed (`[data-base]` extras)
✅ Fixed missing type imports (`IndexSlice`, `MultiIndex`) in `_typing.py`

### Critical Issue Discovered
❌ **VectorBT Pro source code is incomplete - missing `vectorbtpro.data` module**

**Error Details:**
- `vectorbtpro/ohlcv/accessors.py:102` tries to import `from vectorbtpro.data.base import OHLCDataMixin`
- The `data` directory does not exist in `resources/vectorbt.pro-main/`
- Auto-import cannot be disabled before this import occurs

**Impact:**
- Cannot run `import vectorbtpro` successfully
- Cannot use VectorBT Pro's high-level APIs
- May still be able to use low-level numba functions directly

### Fixes Applied to Source
1. **Added missing pandas imports to `_typing.py`:**
   ```python
   # Line 37-38 added:
   from pandas import IndexSlice, MultiIndex
   ```

### Next Steps / Resolution Options

#### Option 1: Get Complete VectorBT Pro Distribution
- **Recommended**: Obtain complete VectorBT Pro source/package
- Resources mentioned in requirements:
  - Use `quantgpt.chat` for VectorBT Pro questions
  - May need to contact user for proper installation source
  - Check if there's a pip-installable version: `pip install vectorbtpro`

#### Option 2: Create Stub Data Module (Temporary Workaround)
- Create minimal `vectorbtpro/data/` directory with `__init__.py` and `base.py`
- Add stub `OHLCDataMixin` class to satisfy imports
- **Risk**: May break functionality that depends on real data module

#### Option 3: Patch Import Chain (Complex)
- Comment out problematic imports in `ohlcv/accessors.py` and other files
- **Risk**: May break OHLCV functionality, cascade to other modules

#### Option 4: Use Alternative VectorBT (Not Pro)
- Install open-source `vectorbt` instead of `vectorbtpro`
- Command: `pip install vectorbt`
- **Limitation**: May not have all Pro features needed for validation

### Recommendation
**BLOCK TASK-001** until we can:
1. Get complete VectorBT Pro source/package from proper source
2. OR get guidance from user on how to properly install VectorBT Pro
3. OR pivot to open-source vectorbt as baseline comparison

### Current Virtual Environment State
- Location: `.venv-vectorbt/`
- Python: 3.12.3
- VectorBT Pro: 2025.7.27 (incomplete installation)
- Dependencies: Fully installed (numpy, pandas, numba, yfinance, etc.)
- Ready for reinstallation once proper source is obtained

### Questions for User
1. Where is the complete VectorBT Pro distribution?
2. Should we use `pip install vectorbtpro` from PyPI (if available)?
3. Is the open-source `vectorbt` an acceptable alternative for validation?
4. Can quantgpt.chat provide installation guidance?
