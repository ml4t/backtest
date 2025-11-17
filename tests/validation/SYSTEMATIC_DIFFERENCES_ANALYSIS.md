# Systematic Differences Analysis - 4-Way Framework Validation

**Date**: 2025-11-16
**Variance**: 2.45% maximum (ml4t.backtest vs Zipline)
**Status**: **UNACCEPTABLE** - Systematic differences identified and must be fixed

## Executive Summary

Trade-by-trade analysis of all 67 trades reveals **three critical systematic differences** causing the 2.4% variance:

1. **Fractional shares not respected** (Backtrader, Zipline)
2. **Different fill prices** (VectorBT using Close instead of Open)
3. **Margin/leverage enabled** (Zipline buying more than available cash)

## Cumulative Cash Flow Differences (vs ml4t.backtest)

| Framework | Cumulative Difference | As % of Initial Capital |
|-----------|----------------------:|------------------------:|
| Backtrader | -$3,664.28 | -3.66% |
| VectorBT | -$4,289.87 | -4.29% |
| Zipline | -$8,516.40 | -8.52% |

**Final Portfolio Values:**
- ml4t.backtest: $194,240.20 (baseline)
- Backtrader: $194,036.69 (-0.10%)
- VectorBT: $198,785.87 (+2.34%)
- Zipline: $198,991.33 (+2.45%)

## Issue #1: Fractional Shares Not Respected

**Config Setting**: `fractional_shares=True` in `FrameworkConfig.for_zipline_matching()`

### Backtrader

**Trade #1 (2020-04-08 BUY):**
- **Expected**: 1503.213399 shares (to invest ~$100k at $66.5175)
- **Actual**: 1500.000000 shares (integer!)
- **Impact**: -$213.75 vs ml4t.backtest

**Pattern**: Backtrader is rounding to integer shares on EVERY trade despite `fractional_shares=True`

**Root Cause**: Our Backtrader adapter is not enabling fractional shares properly. Backtrader needs:
```python
cerebro.broker.set_coc(True)  # Already doing this
cerebro.broker.set_eosbar(True)  # Missing?
# OR need to handle fractional shares differently
```

**Cumulative Impact**: -$3,664 total (-3.66%)

### Zipline

**Trade #1 (2020-04-08 BUY):**
- **Expected**: ~1503 shares fractional (to invest ~$100k at $66.518)
- **Actual**: 1541.000000 shares (integer!)
- **Worse**: Spending $102,504 when only $100,000 available!

**Pattern**: Zipline is:
1. Using integer shares (ignoring `fractional_shares=True`)
2. Allowing purchases beyond available cash (margin/leverage enabled?)

**Root Cause**: Zipline adapter issues:
1. Not configuring fractional share support
2. Margin may be enabled by default in TradingAlgorithm

**Cumulative Impact**: -$8,516 total (-8.52%)

## Issue #2: Different Fill Prices (VectorBT)

**Config Setting**: `fill_timing="next_open"` (all frameworks should fill at next bar's Open)

### VectorBT Price Divergence

**Trade #1 (2020-04-08 BUY):**
- **ml4t.backtest/Backtrader/Zipline**: $66.5175 (next bar Open ✓)
- **VectorBT**: $67.1750 (DIFFERENT PRICE)
- **Difference**: +$0.6575/share (+0.99%)

**Trade #4 (2020-10-29 SELL) - HUGE DIVERGENCE:**
- **ml4t.backtest/Backtrader/Zipline**: $115.3200
- **VectorBT**: $111.0600 (DIFFERENT PRICE)
- **Difference**: -$4.26/share (-3.69%!)
- **Impact on this trade**: -$10,779.77

**Pattern**: VectorBT is using **different fill prices** than other frameworks, often several dollars different.

**Hypothesis**: VectorBT may be:
1. Using Close prices instead of Open
2. Using previous bar's price instead of next bar
3. Not respecting the signal shift we're applying

**Root Cause**: Need to investigate VectorBT adapter's price selection logic. With `fill_timing="next_open"` and signal shifting, VectorBT should use:
```python
# After signal shift of 1 day:
fill_price = data.loc[signal_date, 'Open']  # Signal date is already T+1
```

**Cumulative Impact**: -$4,290 total (-4.29%)

## Issue #3: Margin/Leverage (Zipline)

**Trade #1 (2020-04-08 BUY):**
- **Available Cash**: $100,000
- **Zipline Purchase**: $102,504.24
- **Excess**: +$2,504 (2.5% leverage!)

**Problem**: Zipline is buying MORE than available capital on the first trade!

**Root Cause**: Zipline's TradingAlgorithm may have margin enabled by default. Need to check:
```python
algo.set_margin(leverage=1.0)  # No leverage
# OR disable margin entirely
```

## Configuration Comparison

### Current Config (for_zipline_matching):
```python
{
    "fill_timing": "next_open",
    "commission_pct": 0.0,
    "slippage_pct": 0.0,
    "fractional_shares": True,  # NOT WORKING for Backtrader/Zipline
    "backtrader_coc": True,
    "close_final_position": False
}
```

### What Each Framework Is Actually Doing:

| Framework | Fractional Shares | Fill Price | Margin |
|-----------|:-----------------:|:----------:|:------:|
| ml4t.backtest | ✓ (1503.213) | Open ✓ | No ✓ |
| Backtrader | ✗ (1500.000) | Open ✓ | No ✓ |
| VectorBT | ✓ (1488.649) | Close? ✗ | No ✓ |
| Zipline | ✗ (1541.000) | Open ✓ | YES ✗ |

## Required Fixes

### Priority 1: Fix Backtrader Fractional Shares

**File**: `tests/validation/frameworks/backtrader_adapter.py`

**Investigation needed**:
1. Check if Backtrader supports fractional shares at all
2. If yes, find the configuration parameter
3. If no, we need to adjust test expectations or exclude Backtrader from fractional share tests

**Reference**: Search Backtrader source code for "fractional" or "decimals"

### Priority 2: Fix Zipline Margin and Fractional Shares

**File**: `tests/validation/frameworks/zipline_adapter.py`

**Fixes needed**:
1. Disable margin/leverage:
   ```python
   # In TradingAlgorithm initialize():
   context.set_leverage(1.0)  # No leverage
   ```
2. Enable fractional shares (if Zipline supports it)
3. If Zipline doesn't support fractional shares, adjust position sizing to respect cash constraints

### Priority 3: Fix VectorBT Fill Price Selection

**File**: `tests/validation/frameworks/vectorbt_adapter.py`

**Investigation needed**:
1. Print actual dates and prices being used for fills
2. Verify signal shift is being applied correctly
3. Check if we're using `from_signals()` parameters correctly
4. Ensure `price` parameter points to the right column

**Debug approach**:
```python
# Add logging to see what VectorBT is doing:
print(f"Signal date: {signal_date}")
print(f"Fill price: {data.loc[signal_date, 'Open']}")
print(f"Close price: {data.loc[signal_date, 'Close']}")
```

## Expected Outcome After Fixes

If all three issues are fixed:
- Backtrader should match ml4t.backtest within <0.1%
- VectorBT should match ml4t.backtest within <0.1%
- Zipline should match ml4t.backtest within <0.1%

**Target variance**: <0.1% maximum across all frameworks

## Next Steps

1. ✅ **Document findings** (this file)
2. ⏳ **Fix Backtrader**: Enable fractional shares or document limitation
3. ⏳ **Fix Zipline**: Disable margin, fix fractional shares
4. ⏳ **Fix VectorBT**: Correct fill price selection
5. ⏳ **Re-run validation**: Verify <0.1% variance
6. ⏳ **Update config presets**: Document working configurations

---

**Analysis Date**: 2025-11-16
**Script**: `detailed_trade_comparison.py`
**Output**: `trade_comparison_fixed.txt`
