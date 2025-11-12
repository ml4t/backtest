# Test 2.1 Critical Finding: Signal Timing Discrepancy

**Date**: 2025-11-11
**Test**: Test 2.1 - Percentage Commission (0.1%)
**Status**: 🔴 BLOCKED - Requires investigation

## Problem Summary

Test 2.1 revealed that **qengine and VectorBT execute trades at different times** when commission is enabled, despite using identical signals.

### Evidence

**Test Configuration**:
- Signals: Entry every 50 bars (bars 10, 60, 110...), exit after 10 bars (bars 20, 70, 120...)
- 20 entry/exit pairs expected
- Real BTC spot data (1000 minute bars from 2021-01-01)
- Fees: 0.1% per trade
- Slippage: 0.0

**Results**:
| Metric | qengine | VectorBT | Difference |
|--------|---------|----------|------------|
| Num Trades | 20 | 20 | ✅ Match |
| Final Value | $98,689.13 | $97,299.63 | ❌ $1,389.50 diff |
| Total PnL | -$1,310.87 | -$2,700.37 | ❌ $1,389.50 diff |

**Trade Comparison (First 5)**:

| Trade | qengine Entry Time | qengine Entry Price | VectorBT Entry Time | VectorBT Entry Price | Entry Time Diff | Price Diff |
|-------|-------------------|---------------------|---------------------|----------------------|-----------------|------------|
| 1 | None (NaT) | $28,894.68 | 2021-01-01 00:10 | $28,894.68 | ❓ | ✅ $0.00 |
| 2 | 2021-01-01 00:21 | $28,835.08 | 2021-01-01 01:00 | $29,044.99 | ❌ 39 min | ❌ $209.91 |
| 3 | None (NaT) | $29,421.60 | 2021-01-01 01:50 | $29,421.60 | ❓ | ✅ $0.00 |
| 4 | 2021-01-01 02:01 | $29,507.11 | 2021-01-01 02:40 | $29,282.82 | ❌ 39 min | ❌ $224.29 |
| 5 | None (NaT) | $29,243.35 | 2021-01-01 03:30 | $29,243.35 | ❓ | ✅ $0.00 |

**Pattern Observed**:
- Trade 2: qengine enters at bar 21 (00:21), VectorBT at bar 60 (01:00)
- Trade 4: qengine enters at bar 121 (02:01), VectorBT at bar 160 (02:40)
- **qengine appears to be re-entering immediately after exit** (bar 20→21, bar 120→121)
- **VectorBT correctly waits for the next signal** (bar 60, bar 160)

## Root Cause Hypothesis

### Most Likely: Signal Processing During Same Bar

When an exit occurs at bar N, and the strategy checks for new entries:
- **qengine**: May be re-entering at bar N+1 even without an explicit entry signal
- **VectorBT**: Only enters when an explicit entry signal exists

This suggests:
1. qengine's `SignalStrategy` may have a bug in signal handling
2. The wrapper may not be correctly translating boolean signals
3. There may be a state management issue after exits

### Alternative Hypothesis: Bar Timing/Fill Execution

- qengine might be executing at "open of next bar" while VectorBT executes at "close of signal bar"
- This would explain why some trades match (odd entries) but others don't (even entries)

## Impact Assessment

### Severity: CRITICAL 🔴

This is not a commission calculation issue - it's a **fundamental signal execution difference**.

**Why Critical**:
1. **Invalidates all fee/slippage tests**: Can't validate commission if trades don't match
2. **Questions baseline results**: Phase 1 tests passed, but were they correctly aligned?
3. **Affects trust**: If signals aren't processed identically, what else differs?

### Scope

- ❌ **Phase 2 (Fees)**: Cannot proceed until resolved
- ❌ **Phase 3 (Slippage)**: Depends on Phase 2
- ❌ **Phase 4+ (Advanced)**: All subsequent tests affected

## Investigation Required

### Step 1: Verify Baseline Tests (1.1-1.3)

**Action**: Re-examine baseline tests to check if timing matches
- Review Test 1.1 (entries only) - did entry times match?
- Review Test 1.2 (entry/exit pairs) - were trades at correct indices?
- Review Test 1.3 (multiple round trips) - 40 trades at correct bars?

**Hypothesis**: Baseline tests may have hidden the issue if:
- No overlapping entry/exit on same bar
- Signal spacing prevented ambiguous cases

### Step 2: Isolate Signal Processing

**Create diagnostic test**:
```python
# Test with explicit signal indices
entries_at = [10, 60, 110, 160, 210]  # Explicit bars
exits_at = [20, 70, 120, 170, 220]    # Explicit bars

# Run both engines
# Compare: Do both execute at EXACTLY these bars?
```

### Step 3: Inspect QEngineWrapper Signal Translation

**File**: `tests/validation/common/engine_wrappers.py:58-280`

**Check**:
- How are pandas boolean Series converted to qengine signals?
- Does `SignalStrategy` correctly respect the signal timing?
- Is there state leakage after exits?

### Step 4: Compare with VectorBT Documentation

**Reference**: VectorBT Pro fill model documentation
- How does VBT handle same-bar entry/exit?
- What is the exact fill timing for signals?
- Are there configurable options we're missing?

## Immediate Actions

1. ✅ **Document finding** (this file)
2. ⏳ **Pause TASK-002** (mark as blocked)
3. ⏳ **Create TASK-002B**: "Investigate signal timing discrepancy"
4. ⏳ **Review baseline tests**: Verify Phase 1 results are valid
5. ⏳ **Fix signal processing**: Correct qengine wrapper or SignalStrategy
6. ⏳ **Re-run Test 2.1**: Verify fix resolves discrepancy

## Acceptance Criteria for Resolution

Before continuing with Phase 2:

- ✅ Diagnostic test confirms both engines execute at identical bars
- ✅ Test 2.1 shows <$5 final value difference (rounding tolerance)
- ✅ All 20 trades have matching entry/exit times (within 1 bar tolerance)
- ✅ Entry prices match within 0.01%
- ✅ Root cause understood and documented
- ✅ Fix applied and validated

## References

- Test file: `tests/validation/test_2_1_percentage_commission.py`
- Wrapper: `tests/validation/common/engine_wrappers.py`
- Baseline tests: `test_1_1_baseline_entries.py`, `test_1_2_entry_exit_pairs.py`, `test_1_3_multiple_round_trips.py`
- Handoff: `.claude/transitions/2025-11-11/225220.md`

## Notes for Next Session

**User directive**: "Please do not use synthetic data"
- ✅ All tests now use real BTC spot data
- ✅ Data loader implemented: `load_real_crypto_data()`

**Current state**:
- Phase 1: 3/3 tests passing (but may need re-validation)
- Phase 2: 0/3 tests (TASK-002 blocked by this issue)
- Overall: 1/15 tasks complete (7%)

**Priority**: Resolve signal timing before continuing validation roadmap.
