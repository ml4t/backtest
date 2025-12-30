# Cash Constraint Fix Documentation

## Problem Statement

The broker's partial fill logic could produce negative fill quantities when cash constraints limited the fill amount. This occurred due to a circular dependency in the commission calculation:

1. Commission was estimated based on the full requested quantity
2. Available quantity was then reduced based on (cash - commission) / price
3. If commission was high relative to available cash, this could result in negative quantities

## Root Cause

The issue was in the cash constraint calculation in `SimulationBroker._try_fill_order()`:

```python
# OLD CODE (BUGGY):
estimated_commission = self._calculate_commission(order, fill_quantity, fill_price)
required_cash = fill_quantity * fill_price + estimated_commission
if required_cash > self.cash:
    # This could be negative if estimated_commission > self.cash
    fill_quantity = (self.cash - estimated_commission) / fill_price
```

The commission was calculated for the original `fill_quantity`, but then used to reduce the `fill_quantity`, creating an inconsistency.

## Solution Implemented

### 1. Proper Commission Rate Calculation

The fix properly calculates the maximum affordable quantity considering commission:

```python
# Calculate effective commission rate
if self.commission_model:
    # Estimate commission rate as a fraction of notional
    test_quantity = 1.0
    test_commission = self._calculate_commission(order, test_quantity, fill_price)
    commission_rate = test_commission / (test_quantity * fill_price)
else:
    # Use asset-specific fee or default
    commission_rate = getattr(asset_spec, "taker_fee", 0.001)

# Calculate max affordable quantity
# Solving: quantity * price * (1 + commission_rate) <= cash
max_affordable_quantity = self.cash / (fill_price * (1 + commission_rate))
```

### 2. Binary Search for Non-Linear Commission Models

For complex commission models that aren't linear, the fix includes a binary search to find the exact maximum affordable quantity:

```python
if required_cash > self.cash:
    # Binary search for the right quantity
    low_qty = 0.0
    high_qty = fill_quantity
    for _ in range(10):  # Max 10 iterations
        mid_qty = (low_qty + high_qty) / 2
        mid_commission = self._calculate_commission(order, mid_qty, fill_price)
        mid_cost = mid_qty * fill_price + mid_commission

        if mid_cost <= self.cash:
            low_qty = mid_qty
        else:
            high_qty = mid_qty

    fill_quantity = low_qty
```

### 3. Safety Checks

Multiple safety checks ensure fill quantities are always valid:
- Minimum fill size check (0.01 shares)
- Non-negative quantity guarantee
- Total cost verification

## Testing

Comprehensive test suite in `test_cash_constraints.py` covers:

1. **High flat commission**: Ensures no negative quantities with fixed fees
2. **Percentage commission**: Tests proportional commission models
3. **Insufficient cash**: Verifies orders are rejected when can't afford commission
4. **Exact cash match**: Edge case where cash exactly matches requirements
5. **No commission model**: Tests default behavior
6. **Progressive depletion**: Multiple orders depleting cash
7. **Minimum size constraint**: Orders below minimum are rejected

## Impact

This fix ensures:
- **No negative quantities**: Fill quantities are always >= 0
- **Cash constraints respected**: Total cost never exceeds available cash
- **Commission accuracy**: Commission calculated correctly for actual fill quantity
- **Robust handling**: Works with all commission models (flat, percentage, custom)

## Examples

### Example 1: High Commission Relative to Cash

```python
broker = SimulationBroker(
    initial_cash=1000.0,
    commission_model=FlatCommission(fee=10.0)
)

# Want to buy 100 shares at $100 = $10,000
# But only have $1000
# With $10 commission, can afford 9 shares: 9 * $100 + $10 = $910
order = Order(asset_id="AAPL", quantity=100, price=100)

# Result: Partial fill of 9 shares (not negative!)
```

### Example 2: Percentage Commission

```python
broker = SimulationBroker(
    initial_cash=1000.0,
    commission_model=PercentageCommission(rate=0.01)  # 1%
)

# Want to buy at $50 per share
# Can afford: $1000 / ($50 * 1.01) ≈ 19.8 shares
order = Order(asset_id="MSFT", quantity=50, price=50)

# Result: Partial fill of ~19.8 shares
```

## Migration Guide

No changes required for existing code. The fix is backward compatible and transparent to strategies. The broker now correctly handles cash constraints in all scenarios.

## Performance Considerations

- Linear commission models: O(1) calculation
- Non-linear commission models: O(log n) with binary search (max 10 iterations)
- Negligible performance impact in practice

## Related Issues

This fix addresses the P1 issue from code review:
- **Issue**: Negative fill quantities bug when cash is low
- **Status**: ✅ Resolved
- **Impact**: Medium - Prevented incorrect position calculations

## Future Enhancements

Consider adding:
1. **Configurable minimum fill sizes** per asset class
2. **Smart order splitting** to maximize fill within constraints
3. **Reserve cash settings** to maintain minimum cash balance
4. **Multi-currency cash management** for international trading
