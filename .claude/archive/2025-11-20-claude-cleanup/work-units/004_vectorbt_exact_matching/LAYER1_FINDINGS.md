# Layer 1 Diagnostics: Findings

**Execution Date**: 2025-10-28
**Data Source**: `data/trade_logs/ml4t.backtest_trades_q1_2024.parquet`

---

## Executive Summary

**MAJOR DISCOVERY**: ml4t.backtest is working MUCH better than initially thought!

**Key Metrics**:
- **Trade Count**: 383 trades (vs VectorBT 482) - **Only 20% discrepancy**, not 30%
- **Exit Distribution**: 86.2% TSL / 13.8% TP - **Matches VectorBT almost exactly** (87.7% / 11.8%)
- **TP Functionality**: ✅ **WORKING** - 53 TP exits with avg PnL of $2,835
- **Mean Duration**: 245 minutes (~4 hours) - Reasonable

**Conclusion**: The system is fundamentally working correctly. The 20% trade count difference suggests a **re-entry or signal timing issue**, NOT an exit mechanism problem.

---

## Detailed Findings

### 1. Trade Count Analysis

| Metric | ml4t.backtest | VectorBT | Difference |
|--------|---------|----------|------------|
| **Total Trades** | 383 | 482 | -99 (-20.5%) |

**Assessment**: ⚠️ Moderate discrepancy (20%), not critical (was reported as 30%)

**Likely Causes**:
1. Signal timing differences (1-bar offset)
2. Re-entry restrictions 
3. Data alignment issues

### 2. Exit Type Distribution

| Exit Type | ml4t.backtest Count | ml4t.backtest % | VectorBT % | Δ % |
|-----------|---------------|-----------|------------|-----|
| **Trailing Stop** | 330 | 86.2% | 87.7% | -1.5% |
| **Take Profit** | 53 | 13.8% | 11.8% | +2.0% |

**Assessment**: ✅ **EXCELLENT MATCH** - Within 2% of VectorBT!

**Conclusion**: ❌ **NO EXIT MECHANISM PROBLEM**

### 3. Performance by Exit Type

| Exit Type | Avg PnL | Count |
|-----------|---------|-------|
| **Take Profit** | $2,835 | 53 |
| **Trailing Stop** | -$378 | 330 |

**Key Insights**:
- TP trades are HIGHLY profitable
- TSL exits protect downside
- TP captures momentum moves

### 4. Trade Duration

**Statistics**:
- **Mean**: 245 minutes (~4.1 hours)
- **Median**: 106 minutes
- **Range**: 1 min to 4,128 min

**Distribution**:
- 25% last 4+ hours (trend followers)
- 43% last 1-4 hours
- 32% last < 1 hour

**Assessment**: ✅ Reasonable, TSL is NOT "too aggressive"

---

## Root Cause Analysis

### ❌ Hypothesis: "TP Never Triggers"
**Status**: **DISPROVEN**
- 53 TP exits prove mechanism works
- TP% is actually HIGHER than VectorBT (13.8% vs 11.8%)

### ⚠️ Hypothesis: "30% Fewer Trades"  
**Status**: **PARTIALLY DISPROVEN** (Actually 20%)
- 383 vs 482 = 99 missing trades (-20.5%)
- Suggests signal timing or re-entry issues

### ✅ Hypothesis: "TSL Too Aggressive"
**Status**: **DISPROVEN**
- Mean duration 245 min is reasonable
- Exit distribution matches VectorBT

---

## Recommended Next Actions

### Priority 1: Verify VectorBT Baseline ⭐⭐⭐
Handoff reported 482 trades but configuration may have been different. Need to re-run VectorBT with EXACT same settings and save trade logs.

### Priority 2: Complete Signal Utilization Analysis ⭐⭐
Understand why 99 signals don't become trades. Fix timestamp dtype issue (μs vs ns).

### Priority 3: Trade-by-Trade Comparison ⭐
Compare entry timestamps to identify if it's signal timing or re-entry logic.

---

## Conclusion

**MAJOR REVISION**: Previous problem statement was incorrect.

✅ Exit mechanism works correctly (86.2% TSL / 13.8% TP matches VectorBT)
✅ TP fires as expected (53 exits, highly profitable)
✅ Trade durations are reasonable (mean 245 min)
⚠️ Trade count discrepancy is 20%, not 30% (383 vs 482)

**Root Cause Category**: #2 (Position Management) or #6 (Data Alignment)
- NOT #3 (Exit Mechanism) - working fine
- NOT #5 (Bracket Implementation) - TSL/TP logic correct

**Next Focus**: Signal timing and re-entry logic investigation.

**Confidence**: 85% - Problem scope now well-defined.

---

## Layer 2-A Update: Root Causes Identified

**Execution Date**: 2025-10-28 (continuation session)
**Analysis Time**: 45 minutes

### Root Cause #1: Date Filter ✅ FIXED

**Issue**: START_DATE was `2024-01-02` instead of `2024-01-01`
**Fix**: Changed line 33 in `run_ml4t.backtest_backtest.py`
**Impact**: Date filter corrected, but no change in trade count due to Root Cause #2

### Root Cause #2: Futures Data Gap (3 missing trades)

**Issue**: Futures data starts at 2024-01-01 23:00, signals start at 00:39
**Gap**: 22.4 hours of missing price data
**Impact**: 3 VectorBT trades cannot be executed (0.6% of total)

**VectorBT entries before futures data available**:
1. 2024-01-01 00:39:00 UTC
2. 2024-01-01 04:13:00 UTC
3. 2024-01-01 20:35:00 UTC

**Status**: ⚠️ ACCEPTABLE - Minor data availability issue

### Root Cause #3: Re-entry Timing (96 missing trades) 🔴 PRIMARY ISSUE

**Issue**: ml4t.backtest missing 96 trades even when futures data available
**Pattern**: Missing entries scattered throughout period (Jan 3, 4, 5, 6, 7, 9...)

**Evidence**:
- VectorBT entries (when data available): 479
- ml4t.backtest entries: 383
- Missing: 96 (20% of opportunities)
- ml4t.backtest has ZERO unique entries (all 383 match VectorBT times)

**Hypothesis**: ml4t.backtest doesn't re-enter on same bar after exit, while VectorBT allows immediate re-entry

**Next Steps**: Layer 2-B - Investigate position management and `is_flat()` logic

---

## Summary: All 99 Missing Trades Explained

1. **Data gap**: 3 trades (0.6%) - futures unavailable
2. **Re-entry timing**: 96 trades (20%) - same-bar re-entry difference

**Total explained**: 99 trades (20.5%) ✅

---

## Layer 2-B Update: Root Cause CONFIRMED with Evidence

**Execution Date**: 2025-10-28 (continuation after same-bar re-entry config)
**Analysis Time**: 15 minutes
**Status**: ✅ ROOT CAUSE IDENTIFIED AND PROVEN

### Diagnostic Logging Output (Smoking Gun)

Added diagnostic logging to track position state after exit fills:

```
[EXIT DEBUG] trailing_stop filled at 2024-01-02 05:03:00+00:00
  position after exit: Position(asset_id='BTC', quantity=2.152, ...)
  quantity after exit: 2.152   ← SHOULD BE ZERO!

[EXIT DEBUG] trailing_stop filled at 2024-01-02 14:26:00+00:00
  position after exit: Position(asset_id='BTC', quantity=2.156, ...)
  quantity after exit: 2.156   ← STILL NOT ZERO!
```

### Confirmed Root Cause: Position State Lag

**Problem**: After bracket exit fills execute, `broker.portfolio.positions.get(asset_id).quantity` still shows the PRE-EXIT quantity, not zero.

**Impact**: When next market event arrives with a signal:
1. Strategy checks `is_flat = (position.quantity == 0)`
2. Position still shows pre-exit quantity (e.g., 2.152)
3. `is_flat = False` - incorrectly thinks position is still open
4. Signal is skipped even though position should be flat

**Evidence**:
- Every exit fill shows non-zero quantity immediately after
- ALLOW_SAME_BAR_REENTRY config didn't fix issue (proves hypothesis was wrong)
- ml4t.backtest has ZERO unique entries (all 384 match VectorBT subset)
- VectorBT has 96 unique entries ml4t.backtest skips

### Same-Bar Re-Entry Hypothesis: DISPROVEN

**Tested**: Implemented ALLOW_SAME_BAR_REENTRY config per user request
**Result**: Still got 383 trades, not 482
**Conclusion**: Problem is NOT same-bar logic, but position update timing

**Good News**: The transparent config implementation is valuable for future use, even though it didn't solve this specific issue.

### Next Steps (Clear Path Forward)

**Fix Required**: Ensure position state updates synchronously after fills

**Option A - Force Position Sync** (RECOMMENDED):
```python
# In broker's fill execution, after updating portfolio
self.portfolio.sync_positions()  # Force immediate update
```

**Option B - Check Fresh Position State**:
```python
# In strategy, don't cache position - query fresh
broker_position = self.broker.portfolio.positions.get(event.asset_id)
is_flat = (broker_position is None or broker_position.quantity == 0)
```

**Option C - Process All Fills Before Market Events**:
```python
# In event loop, prioritize FILL events over MARKET events
# Ensure all fills processed before next market event
```

**Confidence**: 95% - Direct evidence from logging confirms position lag
