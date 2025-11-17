# Test 2.2: Combined Fees Baseline

**Date**: 2025-11-15
**Test**: `test_2_2_combined_fees.py`
**Status**: PASSED
**Engine**: ml4t.backtest only (VectorBT Pro not installed)

## Configuration

- **Asset**: BTC (real CryptoCompare spot data)
- **Data**: 1000 minute bars from 2021-01-01
- **Initial Cash**: $100,000.00
- **Commission**: 0.1% (0.001) + $2.00 fixed per trade
- **Slippage**: 0.0%
- **Order Type**: Market orders

## Signals

- **Pattern**: Entry every 50 bars, hold for 10 bars
- **Start offset**: Bar 10
- **Expected trades**: 20
- **Entry indices**: [10, 60, 110, 160, ..., 960]
- **Exit indices**: [20, 70, 120, 170, ..., 970]

## Results

### ml4t.backtest Engine

- **Total trades**: 20
- **Final value**: $95,493.32
- **Final cash**: $95,493.32
- **Final position**: 0 (flat)
- **Total PnL**: -$4,506.68

### Commission Analysis

- **Entry commission**: $1,994.51
- **Exit commission**: $1,993.98
- **Total commission**: $3,988.49
- **Commission as % of initial capital**: 3.988%
- **Average commission per round trip**: $199.42

### Commission Structure Breakdown

Each trade incurs combined fees: **percentage component (0.1% of notional) + fixed component ($2.00)**

#### Sample Round Trips

**Trade 0:**
- Entry notional: $99,888.11
- Entry commission: $101.89 = $99.89 (0.1%) + $2.00 (fixed)
- Exit notional: $99,580.01
- Exit commission: $101.58 = $99.58 (0.1%) + $2.00 (fixed)
- **Round-trip total**: $203.47

**Trade 1:**
- Entry notional: $99,377.10
- Entry commission: $101.38 = $99.38 (0.1%) + $2.00 (fixed)
- Exit notional: $99,572.11
- Exit commission: $101.57 = $99.57 (0.1%) + $2.00 (fixed)
- **Round-trip total**: $202.95

**Trade 2:**
- Entry notional: $99,369.17
- Entry commission: $101.37 = $99.37 (0.1%) + $2.00 (fixed)
- Exit notional: $99,474.93
- Exit commission: $101.47 = $99.47 (0.1%) + $2.00 (fixed)
- **Round-trip total**: $202.84

### Comparison with Test 2.1 (Percentage Only)

| Metric | Test 2.1 (0.1% only) | Test 2.2 (0.1% + $2) | Difference |
|--------|---------------------|---------------------|------------|
| Total commission | $3,910.08 | $3,988.49 | +$78.41 |
| Expected fixed fee component | N/A | $80.00 | (40 fills × $2) |
| Variance from expected | N/A | $1.59 | (very small) |
| Final value | $95,571.53 | $95,493.32 | -$78.21 |

**Observation**: The difference in total commission ($78.41) is very close to the expected fixed fee component ($80.00), with only $1.59 variance. This confirms correct combined fee calculation.

## Validation Criteria

| Criterion | Status | Details |
|-----------|--------|---------|
| Total commission > 0 | ✅ PASS | $3,988.49 |
| Combined fee calculation correct | ✅ PASS | Each exit = (notional × 0.001) + $2.00 |
| Commission reasonable for 20 trades | ✅ PASS | ~$199.42 per round trip |
| Fixed fee component correct | ✅ PASS | $78.41 actual vs $80.00 expected (1.6% variance) |
| Test passes pytest | ✅ PASS | All assertions passed |

## Commission Formula Verification

### Exit Commission (Exact Calculation)

For each exit:
```
exit_commission = (exit_price × exit_quantity × 0.001) + $2.00
```

Sample verifications (within $0.10 tolerance):
- Trade 0: $101.58 = ($99.58) + $2.00 ✅
- Trade 1: $101.57 = ($99.57) + $2.00 ✅
- Trade 2: $101.47 = ($99.47) + $2.00 ✅

### Entry Commission

Entry commissions follow the same formula but are deducted from available cash before the fill, affecting the actual filled quantity. The recorded commission represents the total fee paid (percentage + fixed) on the entry order.

## Notes

1. **VectorBT Comparison**: Not available in this environment (VectorBT Pro not installed)
2. **Fixed Fee Validation**: The $78.41 difference vs Test 2.1 matches the expected $80.00 fixed fee component (40 fills × $2) within 1.6% variance
3. **Commission Structure**: Both percentage and fixed components are correctly applied to each fill (entry and exit)
4. **Total Commission**: The sum of entry + exit commissions ($3,988.49) is reasonable for 20 round trips with combined fees

## Future Work

When VectorBT Pro becomes available:
- Compare total commission amounts (should match within $5)
- Compare final portfolio values (should match within $10)
- Validate combined fee formula matches VectorBT's implementation
- Verify both percentage and fixed components are applied correctly
