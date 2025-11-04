# FINAL Handoff Before Project Move to ~/ml4t/software/

**Date**: 2025-10-05
**Work Unit**: 002_comprehensive_qengine_validatio
**Phase**: Phase 0 - Infrastructure Setup
**Status**: CRITICAL BUG FIXED ✅

---

## CRITICAL: Bug Fix Completed

**Session Summary**: Successfully identified and fixed a critical MA calculation bug in Zipline adapter that was causing 7.95% return variance. Framework validation infrastructure now working correctly.

### The Bug (FIXED)

**Root Cause**: Zipline adapter was calling `history.mean()` on `(long_window + 1)` days of data, calculating MA(31) instead of MA(30).

**Location**: `tests/validation/frameworks/zipline_adapter.py` lines 98 and 103

**Fix Applied**:
```python
# BEFORE (WRONG):
ma_long = history.mean()  # Averages ALL 31 days instead of 30!

# AFTER (CORRECT):
ma_long = history[-context.long_window:].mean()  # Only last 30 days
```

**Verification**:
- Diagnostic script confirmed ALL 8 signals now match perfectly between frameworks
- Cross-framework alignment test suite created: 7/7 tests passing
- When using same data source: Perfect 12.52% return match
- Zipline adapter tests: Still 11/11 passing

**Commits**:
- 74a51d1: "fix: Correct MA calculation bug in Zipline adapter"
- 124f167: "docs: Update LEARNINGS.md with bug fix details"

---

## Current State

### Progress: 6/38 Tasks Complete (15.8%) - TASK-005 NOW TRULY COMPLETE ✅

**Completed**:
- ✅ TASK-001: VectorBT Pro installed and working (2025.7.27)
- ✅ TASK-002: Zipline-Reloaded installed (3.1.1)
- ✅ TASK-003: Backtrader installed
- ✅ TASK-004: VectorBTProAdapter - 18/18 tests passing, reliable baseline
- ✅ TASK-007: UniversalDataLoader - 21/21 tests passing
- ✅ TASK-005: ZiplineAdapter - 11/11 tests passing + signals verified correct

**Next Available**:
- TASK-006: Backtrader adapter (can proceed)
- TASK-008: Baseline verification test (now unblocked - all adapters working)

### Key Files

**Critical Test Files**:
- `tests/validation/test_cross_framework_alignment.py` - **NEW**: 271 lines, 7/7 passing
  - Verifies signal detection logic matches between frameworks
  - Tests MA calculations, signal dates, trade execution
  - Documents that signals now align perfectly on same data source

**Adapters** (All Working):
- `tests/validation/frameworks/vectorbtpro_adapter.py` - 345 lines, 18/18 tests ✅
- `tests/validation/frameworks/zipline_adapter.py` - 204 lines, 11/11 tests ✅ (FIXED)
- `tests/validation/frameworks/base.py` - Base class for all adapters

**Data Loading**:
- `tests/validation/data_loader.py` - 376 lines, 21/21 tests ✅

**Diagnostic Scripts** (in `.claude/diagnostics/`):
- `debug_signal_alignment.py` - Compare MA calculations date-by-date
- `analyze_trade_difference.py` - Trade-level reconciliation
- `verify_alignment.py` - Full cross-framework validation

**Documentation**:
- `.claude/work/current/002_comprehensive_qengine_validatio/LEARNINGS.md` - Comprehensive learnings including bug fix
- `.claude/work/current/002_comprehensive_qengine_validatio/ZIPLINE_VECTORBT_RECONCILIATION.md` - Original investigation (pre-fix)

---

## Key Learnings for New Location

### 1. Test Coverage is Necessary But Insufficient

**What Happened**: All 11 Zipline adapter tests passed, but the MA calculation was wrong (MA(31) vs MA(30)). Tests validated in isolation but didn't catch cross-framework discrepancies.

**Solution**: Created cross-framework alignment test suite that compares outputs between frameworks on identical data.

**Lesson**: **Cross-framework validation is ESSENTIAL**, not optional, for validation infrastructure.

### 2. User Feedback is Gold

**What Happened**: Initially claimed 7.95% variance was "acceptable" (within 10% tolerance). User correctly pushed back: "That's a 4x difference, reconcile it."

**Impact**: This triggered deep investigation that found the critical bug.

**Lesson**: When building validation infrastructure, **perfect agreement is the goal**, not "acceptable variance."

### 3. Trade-Level Comparison Required

**What Happened**: Overall metrics (return, Sharpe) masked the underlying issue. Only by comparing entry/exit dates and prices trade-by-trade did the 1-day signal offset become visible.

**Method**:
```python
# Compare not just:
assert abs(return_a - return_b) < tolerance

# But also:
assert entry_date_a == entry_date_b
assert exit_date_a == exit_date_b
```

**Lesson**: For validation work, **compare intermediate signals**, not just final outputs.

### 4. Data Source Variance Still Exists

**Remaining Issue**: When Zipline uses quandl bundle vs VectorBT uses Wiki parquet, minor OHLCV differences may cause 1-3 day signal variance even with correct logic.

**User Insight**: "The wiki parquet data *should be* identical to the quandl bundle" - this variance needs further investigation.

**Future Work**: Implement signal-based adapter interface where pre-computed signals are passed to both frameworks, eliminating data source variance entirely.

---

## Architecture Decisions Made

### 1. Direct `data.history()` Over Pipeline API

**Decision**: Use Zipline's simple `data.history()` in `handle_data()` instead of Pipeline API

**Rationale**:
- Pipeline: 322 lines, 0 trades, complex debugging
- Direct: 204 lines, trades working, easy to debug
- Pipeline is for large-scale factor research, overkill for validation

**Outcome**: Correct decision, confirmed by successful implementation.

### 2. Unified Data Loader Over Framework-Specific

**Decision**: Single `UniversalDataLoader` with framework-specific conversion methods

**Rationale**: Single source of truth, centralized timezone handling, less duplication

**Outcome**: Working well, 21/21 tests passing.

### 3. No Subprocess Isolation (Yet)

**Decision**: Import frameworks directly in main venv instead of isolated subprocesses

**Rationale**: Simpler, faster, easier debugging. Add isolation only if version conflicts arise.

**Status**: Working fine so far.

---

## Known Issues / Future Work

### 1. Signal-Based Adapter Interface (IMPORTANT)

**Problem**: Each adapter computes signals independently, leading to data source variance.

**Solution**: Extend `BaseFrameworkAdapter` with:
```python
def run_backtest_with_signals(
    self,
    data: pd.DataFrame,
    entry_signals: pd.Series,  # Boolean series
    exit_signals: pd.Series,   # Boolean series
    initial_capital: float
) -> ValidationResult
```

**Benefit**: Eliminates MA calculation and data source variance, tests pure execution logic.

**Priority**: Medium - current approach works but this would be more robust.

### 2. Quandl Bundle vs Wiki Parquet Investigation

**Problem**: User noted these should be identical (same source) but Zipline shows signals 1-3 days different.

**Investigation Needed**:
- Extract exact OHLCV from quandl bundle for same dates
- Compare to Wiki parquet values
- Identify if adjustment differences (splits, dividends) or data quality issues

**Priority**: Low - doesn't block current work, but important for understanding.

### 3. Backtrader Adapter (Next Task)

**Apply Learnings**:
1. Write cross-framework alignment tests FROM THE START
2. Verify signals match VectorBT before claiming "done"
3. Research Backtrader's simplest API first (avoid overcomplication)
4. Watch for similar windowing bugs in MA calculations

---

## Testing Philosophy Established

### What We Test

**Unit Tests** (per adapter):
- Initialization works
- Executes without errors
- Returns valid ValidationResult
- Handles different parameters
- Error cases

**Integration Tests**:
- Data loader produces compatible formats
- Strategies generate trades
- Metrics calculated correctly

**Cross-Framework Alignment Tests** (NEW - CRITICAL):
- Signal count matches
- Signal dates match
- MA values match on critical dates
- Trade execution logic correct
- No spurious signals

**Validation Flow**:
```
Unit Tests → Integration Tests → Cross-Framework Tests → Production Use
     ↓              ↓                      ↓                     ↓
   Passes        Passes             MUST MATCH           Ready for QEngine
```

### Test Coverage Targets

- Adapter unit tests: 10-15 tests each
- Cross-framework: 5-10 tests per strategy
- Target: >80% code coverage
- **But coverage % is not enough - cross-framework verification is mandatory**

---

## Environment / Dependencies

**Virtual Environment**: `.venv/` (Python 3.13.5)

**Key Packages**:
- vectorbtpro==2025.7.27 (commercial, Pro license)
- zipline-reloaded==3.1.1
- backtrader==1.9.78.123 (installed but adapter not yet implemented)
- pandas, numpy, pytest

**Data Sources**:
- Wiki parquet: `~/ml4t/projects/daily_us_equities/wiki_prices.parquet`
- Zipline quandl bundle: Managed by Zipline, ingested separately

**Git Status**: Clean working tree, all changes committed

---

## File Locations (BEFORE MOVE)

**Current Path**: `/home/stefan/ml4t/backtest/`

**After Move Will Be**: `/home/stefan/ml4t/software/backtest/` (or similar)

**Critical Paths to Update** (if hardcoded):
- Data loader default path: `~/ml4t/projects/` (uses expanduser, should be OK)
- Any absolute paths in tests (grep for `/home/stefan/ml4t/backtest`)

**Git Remote**: Check if remote needs updating after move

---

## Quick Start After Move

### 1. Verify Environment
```bash
cd ~/ml4t/software/backtest  # Or wherever moved to
source .venv/bin/activate
python -c "import vectorbtpro, zipline; print('OK')"
```

### 2. Run Test Suite
```bash
# All validation tests
pytest tests/validation/ -v

# Just cross-framework alignment
pytest tests/validation/test_cross_framework_alignment.py -v

# Individual adapters
pytest tests/validation/test_vectorbtpro_adapter.py -v
pytest tests/validation/test_zipline_adapter.py -v
```

### 3. Review Documentation
```bash
# Comprehensive learnings
cat .claude/work/current/002_comprehensive_qengine_validatio/LEARNINGS.md

# Bug fix details (search for "2025-10-05")
cat .claude/work/current/002_comprehensive_qengine_validatio/LEARNINGS.md | grep -A 30 "2025-10-05"
```

### 4. Continue Work
```bash
# Next task: Backtrader adapter (TASK-006)
# Or: Baseline verification test (TASK-008) - now unblocked
```

---

## Most Important Takeaway

**The validation framework's purpose is to validate QEngine correctness.**

**To validate QEngine, the validation framework itself must be correct.**

**To ensure the validation framework is correct, we must cross-validate between established frameworks (VectorBT, Zipline, Backtrader).**

**Cross-framework validation caught a bug that 100% test coverage missed.**

**This is the most important lesson from this entire session.**

---

## Contact Points / References

**VectorBT Pro**: Commercial license, extensive documentation at vectorbt.pro

**Zipline-Reloaded**: Fork of Quantopian's Zipline, maintained community project
- Issue: Pipeline API complex, use simple `data.history()` instead
- Workaround: Timezone-naive dates for Python 3.13 compatibility

**Backtrader**: Next to implement, research simple API first

**QEngine**: The system being validated (not yet integrated with validation framework)

---

**Created**: 2025-10-05
**Last Commit**: 124f167 "docs: Update LEARNINGS.md with bug fix details"
**Git Branch**: main
**Next Session Should Start With**: Review this handoff document, then proceed to TASK-006 (Backtrader) or TASK-008 (Baseline verification)
