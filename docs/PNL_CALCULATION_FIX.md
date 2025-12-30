# P&L Calculation Fix Documentation

## Problem Statement

The P&L calculations had several issues:
1. **Options P&L Ambiguity**: The `calculate_pnl` method for options was unclear whether `entry_price` and `exit_price` parameters represented option premiums or underlying asset prices
2. **Misleading Documentation**: Comments suggested "falling back to intrinsic value calculation" when the actual behavior was using prices as premiums
3. **Missing Functionality**: No method to calculate option P&L at expiry based on intrinsic value

## Root Cause

The core issue was documentation and API clarity rather than incorrect calculations. The math was correct, but the parameter names and documentation created confusion about what values should be passed for options.

## Solution Implemented

### 1. Clarified Documentation

Updated all methods to explicitly state that for options:
- `entry_price` and `exit_price` must be option premiums, NOT underlying prices
- This applies to both `calculate_pnl()` and `calculate_pnl_enhanced()`

### 2. Added Expiry P&L Method

Created `calculate_option_pnl_at_expiry()` for options held to expiry:
```python
def calculate_option_pnl_at_expiry(
    self,
    entry_premium: Price,
    underlying_price_at_expiry: Price,
    quantity: float,
    option_type: str = "call",  # "call" or "put"
    include_costs: bool = True,
) -> float:
```

This method:
- Calculates intrinsic value at expiry based on strike and underlying price
- Computes P&L as the difference between intrinsic value and entry premium
- Handles both calls and puts, long and short positions

### 3. Comprehensive Testing

Added extensive tests covering:
- All asset classes (Equity, Future, Option, FX, Crypto)
- Premium-based P&L for options (trading before expiry)
- Intrinsic value-based P&L for options (held to expiry)
- Long and short positions
- With and without trading costs
- Edge cases (zero quantity, zero price change)

## API Usage Examples

### Options Trading (Before Expiry)

```python
# Using premium prices
spec = AssetSpec(asset_id="AAPL_CALL_150", asset_class=AssetClass.OPTION,
                 contract_size=100, strike=150.0)

# Long 1 call: bought at $2.00 premium, sold at $3.50 premium
pnl = spec.calculate_pnl(
    entry_price=2.00,    # Premium, NOT underlying price
    exit_price=3.50,     # Premium, NOT underlying price
    quantity=1.0
)
# Result: (3.50 - 2.00) * 1 * 100 = $150 profit
```

### Options Held to Expiry

```python
# Long call expires in the money
pnl = spec.calculate_option_pnl_at_expiry(
    entry_premium=2.00,
    underlying_price_at_expiry=155.0,  # Underlying > Strike (150)
    quantity=1.0,
    option_type="call"
)
# Intrinsic value: max(0, 155 - 150) = $5.00
# P&L: (5.00 - 2.00) * 1 * 100 = $300 profit
```

### FX Trading

```python
spec = AssetSpec(asset_id="EUR/USD", asset_class=AssetClass.FX)

# Long 10,000 EUR
pnl = spec.calculate_pnl(
    entry_price=1.1000,
    exit_price=1.1050,
    quantity=10000.0
)
# Result: 10000 * (1.1050 - 1.1000) = $50 profit
```

## Migration Guide

### For Existing Code

1. **Options P&L**: Ensure you're passing premium values, not underlying prices
2. **New Functionality**: Use `calculate_option_pnl_at_expiry()` for expiry scenarios
3. **No Breaking Changes**: All existing methods maintain backward compatibility

### Best Practices

1. Always use premium prices for options in `calculate_pnl()`
2. Use `calculate_pnl_premium_based()` for explicit premium-based calculations
3. Use `calculate_option_pnl_at_expiry()` for options held to expiry
4. Include trading costs for realistic P&L (`include_costs=True`)

## Impact

This fix ensures:
- **Clear API Contract**: No ambiguity about what prices represent
- **Complete Functionality**: Both trading and expiry P&L calculations available
- **Industry Standard**: Calculations match standard financial practices
- **Comprehensive Testing**: All asset classes thoroughly tested

## Related Issues

This fix addresses the P1 issue from code review:
- **Issue**: Domain Correctness Errors in P&L calculations
- **Status**: ✅ Resolved
- **Impact**: Medium - Affects accuracy of backtesting results for options

## Future Enhancements

Consider adding:
1. **Greeks-based P&L**: Decompose option P&L by Greeks (delta, gamma, theta, vega)
2. **Path-dependent Options**: Support for American options with early exercise
3. **Complex Strategies**: Built-in P&L for spreads, straddles, etc.
4. **Currency Conversion**: Automatic P&L conversion to base currency
