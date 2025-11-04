# Zipline vs VectorBT Trade Reconciliation

**Date**: 2025-10-04
**Test Case**: AAPL 2017-01-03 to 2017-12-29, MA(10/30) crossover
**Initial Capital**: $10,000

---

## Executive Summary

**Return Difference**: 17.45% (Zipline) vs 12.52% (VectorBT) = **4.93% variance**

**Root Cause**: Different signal generation logic leads to different entry/exit timing and position holding periods.

**Status**: ⚠️ **NOT ACCEPTABLE** - Requires signal logic alignment

---

## Trade-by-Trade Comparison

### VectorBT: 4 Trades (All Closed)

| Trade | Entry Date | Entry Price | Exit Date  | Exit Price | Shares | PnL      |
|-------|-----------|-------------|-----------|------------|--------|----------|
| 1     | 2017-04-26 | $143.65    | 2017-06-13 | $146.59   | 69.61  | +$204.61 |
| 2     | 2017-07-19 | $151.02    | 2017-09-19 | $158.73   | 67.57  | +$520.97 |
| 3     | 2017-10-18 | $159.76    | 2017-12-11 | $172.67   | 67.14  | +$866.72 |
| 4     | 2017-12-20 | $174.35    | 2017-12-29 | $169.23   | 66.49  | **-$340.42** |

**Total PnL**: +$1,251.88
**Final Value**: $11,251.88
**Return**: 12.52%

### Zipline: 2 Positions (1 Closed, 1 Open)

| Position | Entry Date | Entry Price | Exit Date  | Exit Price | Shares | PnL      | Status |
|----------|-----------|-------------|-----------|------------|--------|----------|---------|
| 1        | 2017-04-25 | $144.54    | 2017-09-19 | $158.73   | ~69    | +$979.11 | Closed  |
| 2        | 2017-10-18 | $159.76    | -         | $169.23*  | ~67    | +$634.49* | **OPEN** |

*Unrealized PnL based on final price

**Total Realized PnL**: +$979.11
**Unrealized PnL**: +$634.49
**Final Value**: $11,744.66
**Return**: 17.45%

---

## Critical Differences Identified

### 1. Entry Timing Mismatch (Position 1)

**VectorBT**: Entered 2017-04-26
**Zipline**: Entered 2017-04-25 (1 day earlier!)

**Why**: Signal detection logic differs:
- Zipline uses: `(prev_ma_short <= prev_ma_long) and (ma_short > ma_long)`
- VectorBT uses: `(ma_short > ma_long) & (ma_short.shift(1) <= ma_long.shift(1))`

These should be equivalent but the 1-day difference suggests different MA calculation or alignment.

**Impact**: Different entry prices ($144.54 vs $143.65)

### 2. Exit Signal Discrepancy (Position 1)

**VectorBT**: Exited 2017-06-13 (death cross detected)
**Zipline**: Stayed in position (no death cross detected until 2017-09-19)

**Investigation needed**: Why did VectorBT see a death cross on 2017-06-13 that Zipline didn't see?

Let me check the MA values on that date:
- 2017-06-13: Need to verify MA(10) vs MA(30) values

**Impact**:
- VectorBT held for 48 days, gained $204
- Zipline held for 147 days, gained $979
- **Zipline captured the full trend, VectorBT exited prematurely**

### 3. Re-Entry After Premature Exit (VectorBT Only)

**VectorBT**: Entered again 2017-07-19 (golden cross)
**Zipline**: Still in original position, no signal

**Why**: VectorBT had exited, so was flat and could re-enter. Zipline never exited, so ignored the signal (already in position).

### 4. Final Exit Timing

**VectorBT Trade 3**: Exited 2017-12-11 (death cross) for +$867
**VectorBT Trade 4**: Re-entered 2017-12-20, force-closed 2017-12-29 for -$340
**Zipline Position 2**: Entered 2017-10-18, **still open at end**

**Key Issue**: VectorBT force-closes positions at end of data (losing $340), while Zipline keeps them open (realistic for live trading).

**Impact**: VectorBT's forced exit creates a realized loss that doesn't reflect actual strategy performance.

---

## Signal Detection Analysis

### Expected Signals (Manual Calculation)

Using pandas on raw data:

**Golden Crosses** (MA10 crosses above MA30):
1. 2017-04-26
2. 2017-07-19
3. 2017-10-18
4. 2017-12-20

**Death Crosses** (MA10 crosses below MA30):
1. 2017-04-24
2. 2017-06-13
3. 2017-09-19
4. 2017-12-11

### VectorBT Signals Detected

Matches manual calculation exactly (4 golden, 4 death)

### Zipline Signals Detected

**Actual debug output** (from earlier run):
- Golden crosses: 5 detected (but only 2 executed due to position rules)
- Death crosses: 1 detected

**Discrepancy**: Zipline is detecting DIFFERENT signals than manual calculation!

---

## Root Cause Analysis

### Primary Issue: MA Calculation Differences

**Hypothesis**: Zipline's `data.history()` may return different historical windows than VectorBT's rolling calculation.

**Test needed**:
1. Extract exact MA values from Zipline on each date
2. Compare to VectorBT MA values
3. Identify where they diverge

### Secondary Issue: Signal Detection Logic

**Zipline logic**:
```python
prev_ma_short = prev_history[-short_window:].mean()
prev_ma_long = prev_history.mean()
```

**Potential issue**: `prev_history.mean()` uses the ENTIRE prev_history (long_window + 1 days minus 1 = long_window days), which is correct. But the indexing might be off.

### Tertiary Issue: Position Tracking

Zipline correctly doesn't trigger entry signals when already in position.
VectorBT executes ALL entry/exit signals regardless of current position (then reconciles them into trades).

---

## Action Items

### Immediate (Required for Validation)

1. **[ ] Extract MA values from both frameworks on same dates**
   - Compare MA(10) and MA(30) values for 2017-04-25 vs 2017-04-26
   - Compare MA values on 2017-06-13 (mystery exit)
   - Verify calculation methodology

2. **[ ] Align signal detection logic**
   - Ensure both use same crossover detection
   - Verify index alignment (shift vs prev_history)
   - Test with simple synthetic data (known signals)

3. **[ ] Fix end-of-period handling**
   - VectorBT should NOT force-close positions at data end
   - Both should report unrealized PnL for open positions
   - Document whether to include unrealized in return calculation

4. **[ ] Update adapters with aligned logic**
   - Standardize MA calculation
   - Standardize crossover detection
   - Standardize position handling

### Documentation (For LEARNINGS.md)

5. **[ ] Document signal generation differences**
   - Exact cause of 2017-04-25 vs 2017-04-26 entry
   - Why 2017-06-13 exit only in VectorBT
   - Framework philosophy differences

6. **[ ] Create validation test cases**
   - Synthetic data with known crossovers
   - Verify both frameworks match expected behavior
   - Document acceptable variance (if any)

---

## Acceptable Variance Guidelines

### What's Acceptable

- **Floating point rounding**: < 0.01% difference in returns
- **Position sizing**: < 0.01 shares difference (due to rounding)
- **Commission/slippage**: When explicitly set to 0, should be exactly 0

### What's NOT Acceptable

- ❌ **Different entry/exit dates** (current issue)
- ❌ **4.93% return variance** (way above tolerance)
- ❌ **Force-closing positions at data end** (VectorBT issue)
- ❌ **Missing signals** (Zipline detecting 5 golden vs expected 4)

### Target After Fixes

- Same entry/exit dates (±0 days)
- Return variance < 0.5%
- Same number of trades (±0)
- Sharpe ratio variance < 10%

---

## Next Steps

1. Extract and compare exact MA values from both frameworks
2. Create synthetic test case with KNOWN crossover dates
3. Fix signal detection in adapter with higher tolerance signals
4. Re-test and verify < 0.5% variance

**Status**: Investigation in progress, NOT ready for validation use.
