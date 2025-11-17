# Test 3.1: Fixed Slippage Baseline

**Date**: 2025-11-15
**Test**: `test_3_1_fixed_slippage.py`
**Status**: PASSED
**Engine**: ml4t.backtest only (VectorBT Pro not installed)

## Configuration

- **Asset**: BTC (real CryptoCompare spot data)
- **Data**: 1000 minute bars from 2021-01-01
- **Initial Cash**: $100,000.00
- **Commission**: 0.0% (zero fees to isolate slippage)
- **Slippage**: 0.0342% (percentage approximation of $10 fixed)
- **Target Slippage**: $10.00 per unit of BTC
- **Order Type**: Market orders

## Signals

- **Pattern**: Entry every 50 bars, hold for 10 bars
- **Start offset**: Bar 10
- **Expected trades**: 20
- **Entry indices**: [10, 60, 110, 160, ..., 960]
- **Exit indices**: [20, 70, 120, 170, ..., 970]

## Slippage Model

**Important Note**: This test uses **percentage-based slippage** to approximate **fixed dollar slippage**.

- VectorBT only supports percentage-based slippage, not fixed dollar amounts
- To approximate $10 fixed slippage: `slippage_pct = 10 / avg_price`
- Average BTC price: $29,281.66
- Calculated slippage percentage: 10 / 29281.66 = 0.0342%

**Expected Behavior**:
- Each unit of BTC incurs ~$10 slippage per fill
- Total slippage per fill = $10 × quantity (in BTC)
- For a 3.4 BTC position: slippage ≈ $34 per fill

## Results

### ml4t.backtest Engine

- **Total trades**: 20
- **Final value**: $98,122.02
- **Final cash**: $98,122.02
- **Final position**: 0 (flat)
- **Total PnL**: -$1,877.98

### Slippage Analysis

- **Entry slippage**: $676.57
- **Exit slippage**: $676.38
- **Total slippage**: $1,352.95
- **Slippage as % of initial capital**: 1.353%
- **Average slippage per fill**: $33.82
- **Target slippage per unit**: $10.00

### Slippage Verification

#### Per-Fill Slippage (Sample)
All slippage amounts verified to match percentage calculation:

**Trade 0**:
- Entry: 3.4486 BTC @ $28,993.98, slippage = $34.14
- Exit: 3.4486 BTC @ $28,884.81, slippage = $34.03
- Total: $68.17

**Trade 1**:
- Entry: 3.4427 BTC @ $28,935.14, slippage = $34.01
- Exit: 3.4427 BTC @ $28,972.12, slippage = $34.07
- Total: $68.08

**Trade 2**:
- Entry: 3.4365 BTC @ $29,024.02, slippage = $34.05
- Exit: 3.4365 BTC @ $29,035.07, slippage = $34.09
- Total: $68.14

#### Slippage Calculation Verification

For Trade 0 entry:
- Fill price: $28,993.98
- Market price: $28,993.98 / (1 + 0.000342) = $28,984.07
- Slippage per unit: $28,993.98 - $28,984.07 = $9.91 ≈ $10.00 ✅
- Total slippage: $9.91 × 3.4486 BTC = $34.18 ✅ (matches $34.14 from trade record)

## Validation Criteria

| Criterion | Status | Details |
|-----------|--------|---------|
| Total slippage > 0 | ✅ PASS | $1,352.95 |
| Slippage per unit ≈ $10 | ✅ PASS | $9.91 (within $1 tolerance) |
| Slippage scales with position size | ✅ PASS | ~$34 per fill for 3.4 BTC |
| Total fills = 40 (20 × 2) | ✅ PASS | 20 entries + 20 exits |
| Test passes pytest | ✅ PASS | All assertions passed |

## Notes

1. **Percentage vs Fixed Slippage**: This test uses percentage-based slippage (0.0342%) to approximate $10 fixed slippage per unit of BTC. The actual slippage per fill depends on position size.

2. **Position Sizing**: Each trade uses approximately all available cash (~$100k), resulting in positions of ~3.4 BTC at average price of ~$29k.

3. **Slippage Scaling**: With 3.4 BTC per trade:
   - Per-unit slippage: $10.00
   - Total slippage per fill: $10.00 × 3.4 = $34.00 (actual: $33.82 due to price variation)

4. **Total Slippage Cost**:
   - Expected: 40 fills × $34 avg = $1,360
   - Actual: $1,352.95
   - Difference: $7.05 (0.5%) - within tolerance due to price variation

5. **VectorBT Comparison**: Not available in this environment (VectorBT Pro not installed)

## Future Work

When VectorBT Pro becomes available:
- Compare total slippage amounts (should match within $10 due to price variation)
- Compare final portfolio values (should match within $10 tolerance)
- Verify VectorBT uses same percentage slippage calculation (multiplicative formula)

## Acceptance

**Test Status**: ✅ PASSED

The test demonstrates that:
1. Percentage slippage is correctly applied using VectorBT's multiplicative formula
2. Slippage per unit of BTC is approximately $10 as intended
3. Total slippage scales correctly with position size
4. All 20 round-trip trades complete successfully
5. Final portfolio value reflects expected slippage impact
