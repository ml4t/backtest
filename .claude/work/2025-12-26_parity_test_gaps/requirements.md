# Parity Test Gaps: Commission, Slippage, Trailing Stops, Bracket Orders

**Source**: Book repo review (2025-12-26)
**Priority**: High - These are core backtester features that need cross-framework validation

## Current Validation Status

From `validation/README.md`:

| Feature | VectorBT Pro | VectorBT OSS | Backtrader | Zipline |
|---------|--------------|--------------|------------|---------|
| Long only | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS |
| Long/Short | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS |
| Multi-asset (500×10yr) | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS |
| Stop-loss | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS |
| Take-profit | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS |
| **% Commission** | ⬜ | ⬜ | ⬜ | ⬜ |
| **Per-share commission** | ⬜ | ⬜ | ⬜ | ⬜ |
| **Fixed slippage** | ⬜ | ⬜ | ⬜ | ⬜ |
| **% Slippage** | ⬜ | ⬜ | ⬜ | ⬜ |
| **Trailing stop** | ⬜ | ⬜ | ⬜ | ⬜ |

**Bracket orders (OCO)** are also untested but not in the matrix.

## Required New Validation Scripts

### Scenario 03: Percentage Commission

Test 0.1% commission on each trade.

**Expected behavior**: Commission = trade_value × 0.001

**Files to create**:
- `validation/vectorbt_pro/scenario_03_commission_pct.py`
- `validation/vectorbt_oss/scenario_03_commission_pct.py`
- `validation/backtrader/scenario_03_commission_pct.py`
- `validation/zipline/scenario_03_commission_pct.py`

**Framework config mapping**:
```python
# VectorBT Pro/OSS
pf = vbt.Portfolio.from_signals(..., fees=0.001)

# Backtrader
cerebro.broker.setcommission(commission=0.001)

# Zipline
set_commission(PerTrade(cost=0.001))  # or PerShare with commission_rate

# ml4t.backtest
engine = Engine(..., commission=PercentageCommission(rate=0.001))
```

### Scenario 04: Per-Share Commission

Test $0.005 per share (IB-like).

**Expected behavior**: Commission = shares × 0.005

**Files to create**:
- `validation/vectorbt_pro/scenario_04_commission_per_share.py`
- `validation/vectorbt_oss/scenario_04_commission_per_share.py`
- `validation/backtrader/scenario_04_commission_per_share.py`
- `validation/zipline/scenario_04_commission_per_share.py`

**Framework config mapping**:
```python
# VectorBT Pro/OSS
# Requires custom function or fees as absolute amount

# Backtrader
cerebro.broker.setcommission(commission=0.005, commtype=bt.CommInfoBase.COMM_FIXED)

# Zipline
set_commission(PerShare(cost=0.005))

# ml4t.backtest
engine = Engine(..., commission=PerShareCommission(rate=0.005))
```

### Scenario 05: Fixed Slippage

Test $0.01 fixed slippage (execution worse than signal price).

**Expected behavior**: Buy fills at signal_price + 0.01, Sell fills at signal_price - 0.01

**Files to create**:
- `validation/vectorbt_pro/scenario_05_slippage_fixed.py`
- `validation/vectorbt_oss/scenario_05_slippage_fixed.py`
- `validation/backtrader/scenario_05_slippage_fixed.py`
- `validation/zipline/scenario_05_slippage_fixed.py`

**Framework config mapping**:
```python
# VectorBT Pro/OSS
pf = vbt.Portfolio.from_signals(..., slippage=0.01)  # Absolute

# Backtrader
cerebro.broker.set_slippage_fixed(0.01)

# Zipline
set_slippage(FixedSlippage(spread=0.02))  # Total spread = 2 × one-way

# ml4t.backtest
engine = Engine(..., slippage=FixedSlippage(amount=0.01))
```

### Scenario 06: Percentage Slippage

Test 0.1% slippage (common for illiquid assets).

**Expected behavior**: Buy fills at signal_price × 1.001, Sell fills at signal_price × 0.999

**Files to create**:
- `validation/vectorbt_pro/scenario_06_slippage_pct.py`
- `validation/vectorbt_oss/scenario_06_slippage_pct.py`
- `validation/backtrader/scenario_06_slippage_pct.py`
- `validation/zipline/scenario_06_slippage_pct.py`

### Scenario 07: Trailing Stop

Test 5% trailing stop (stop price follows highest high).

**Expected behavior**: Stop price = highest_since_entry × 0.95

**Files to create**:
- `validation/vectorbt_pro/scenario_07_trailing_stop.py`
- `validation/vectorbt_oss/scenario_07_trailing_stop.py`
- `validation/backtrader/scenario_07_trailing_stop.py`
- `validation/zipline/scenario_07_trailing_stop.py`

**Note**: This is the most complex scenario. May require:
- Tracking high-water mark per position
- Different implementations across frameworks
- Careful documentation of semantic differences

### Scenario 08: Bracket Orders (OCO)

Test simultaneous stop-loss (-5%) and take-profit (+10%) that cancel each other.

**Expected behavior**: First triggered order executes, other is cancelled

**Note**: Check if all frameworks support true OCO (one-cancels-other) semantics.

## Implementation Order

Recommended priority based on complexity:

1. **Scenario 03: % Commission** - Simplest, well-supported everywhere
2. **Scenario 04: Per-share Commission** - Similar pattern
3. **Scenario 05: Fixed Slippage** - Straightforward
4. **Scenario 06: % Slippage** - Similar to fixed
5. **Scenario 07: Trailing Stop** - Most complex state tracking
6. **Scenario 08: Bracket Orders** - May have semantic differences

## Acceptance Criteria

For each scenario:
1. All four frameworks produce results within 0.01% of each other (or document why they differ)
2. README.md updated with results
3. Configuration required for matching documented

## Test Data

Use the same synthetic signal data from existing scenarios:
- `validation/data/signals_long_only.parquet`
- `validation/data/signals_long_short.parquet`

Or create new signal files if needed for trailing/bracket tests.
