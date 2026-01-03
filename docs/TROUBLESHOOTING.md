# ml4t.backtest Troubleshooting Guide

Common issues and their solutions when using ml4t.backtest.

---

## Order Issues

### Orders Not Filling

**Symptom**: Orders are submitted but never execute.

**Possible Causes**:

1. **Limit price not reached**
   - Limit buy orders only fill when `low <= limit_price`
   - Limit sell orders only fill when `high >= limit_price`
   - Check your data has proper OHLC values

2. **Stop price not triggered**
   - Stop buy orders trigger when `high >= stop_price`
   - Stop sell orders trigger when `low <= stop_price`

3. **NEXT_BAR execution mode**
   - Orders submitted on bar N fill on bar N+1
   - This is realistic but may surprise users expecting same-bar fills

4. **Insufficient capital**
   - Cash accounts reject orders that would result in negative cash
   - Check `broker.get_buying_power()` before ordering

**Solution**: Enable debug logging or check `broker.get_rejected_orders()`:
```python
rejected = engine.broker.get_rejected_orders()
for order in rejected:
    print(f"{order.asset}: {order.rejection_reason}")
```

### Same-Bar Re-Entry Blocked

**Symptom**: After a stop-loss exit, new entry orders on the same bar return `None`.

**Cause**: VectorBT Pro compatibility mode prevents same-bar re-entry after stop exits.

**Solution**: This is intentional behavior. If you need same-bar re-entry:
- Use a custom strategy that handles this case
- Or wait for the next bar to re-enter

---

## Position Issues

### Position Shows Wrong Quantity

**Symptom**: `broker.positions` shows unexpected quantities.

**Possible Causes**:

1. **Partial fills** (if using `ExecutionLimits`)
   - Check `order.filled_quantity` vs `order.quantity`

2. **Position flipping**
   - When you go from long to short (or vice versa), the position flips
   - Check trade history for the closing trade

**Solution**: Use the accessor methods:
```python
# Get single position
pos = broker.get_position("AAPL")
if pos:
    print(f"Quantity: {pos.quantity}")

# Get all positions
for asset, pos in broker.get_positions().items():
    print(f"{asset}: {pos.quantity}")
```

### "Positions" vs "get_positions()"

**Issue**: Direct `.positions` access vs `.get_positions()` method.

**Answer**: Both work, but `get_positions()` is the recommended API:
```python
# Recommended
positions = broker.get_positions()

# Also works but less explicit
positions = broker.positions
```

---

## Cash and Margin Issues

### Order Rejected Due to Insufficient Funds

**Symptom**: Order rejected with "Insufficient cash" or "Insufficient buying power".

**For Cash Accounts**:
- You cannot buy more than your cash allows
- You cannot short sell (requires margin account)

**For Margin Accounts**:
- Check your maintenance margin isn't exceeded
- Use `broker.get_buying_power()` to see available capital

**Solution**:
```python
# Check available buying power
print(f"Buying Power: ${broker.get_buying_power():,.2f}")

# Use target percent/value methods for safer sizing
broker.order_target_percent("AAPL", 0.10)  # 10% of portfolio
```

### Short Selling Not Working

**Symptom**: Sell orders for assets you don't own are rejected.

**Cause**: Cash accounts don't allow short selling.

**Solution**: Use a margin account:
```python
engine = Engine(
    feed=feed,
    strategy=strategy,
    account_type="margin",  # Enable margin/shorting
    initial_margin=0.5,     # 50% initial margin
)
```

---

## Data Issues

### "No data for asset" Warnings

**Symptom**: Strategy receives no data for some assets.

**Possible Causes**:

1. **Asset not in DataFrame**
   - Check your data includes the asset symbol

2. **Missing timestamps**
   - Ensure data covers the entire backtest period

3. **Column naming**
   - DataFeed expects: `timestamp`, `asset`, `open`, `high`, `low`, `close`, `volume`

**Solution**: Verify your data:
```python
import polars as pl

df = pl.read_parquet("data.parquet")
print(df.columns)  # Check column names
print(df["asset"].unique())  # Check asset symbols
print(df.select(["timestamp"]).min(), df.select(["timestamp"]).max())
```

### Look-Ahead Bias Concerns

**Symptom**: Backtest results seem too good to be true.

**Possible Causes**:

1. **Using close price for same-bar orders**
   - `ExecutionMode.SAME_BAR` fills at close, which may not be realistic
   - Consider `ExecutionMode.NEXT_BAR` for more realistic fills

2. **Signal leakage**
   - Signals computed using future data
   - Ensure features are computed with proper time shifts

**Solution**: Use realistic execution settings:
```python
from ml4t.backtest import Engine, ExecutionMode

engine = Engine(
    feed=feed,
    strategy=strategy,
    execution_mode=ExecutionMode.NEXT_BAR,  # More realistic
)
```

---

## Risk Management Issues

### Stop-Loss Not Triggering

**Symptom**: Stop-loss orders don't trigger when expected.

**Possible Causes**:

1. **Stop price vs fill price confusion**
   - Stop orders trigger at `stop_price` but may fill at a different price (gap)

2. **NEXT_BAR_OPEN mode**
   - Some configurations defer stop fills to next bar's open

3. **High/Low data missing**
   - Stop checking requires accurate high/low prices

**Solution**: Check your stop configuration:
```python
from ml4t.backtest import StopFillMode

# Fill at stop price (default)
engine = Engine(..., stop_fill_mode=StopFillMode.STOP_PRICE)

# Fill at bar open (for gaps)
engine = Engine(..., stop_fill_mode=StopFillMode.BAR_OPEN)
```

### Trailing Stop Not Updating

**Symptom**: Trailing stop stays at initial level.

**Cause**: Trailing stops update based on `trail_hwm_source` configuration.

**Solution**: Check your trailing stop setup:
```python
from ml4t.backtest import TrailHwmSource

engine = Engine(
    ...,
    trail_hwm_source=TrailHwmSource.CLOSE,  # Update on close
    # or TrailHwmSource.HIGH  # Update on high (tighter)
)
```

---

## Performance Issues

### Backtest Running Slowly

**Possible Causes**:

1. **Large dataset without optimization**
   - Use Polars lazy mode for data preparation
   - Consider sampling for initial testing

2. **Complex risk rules**
   - Simplify rules during development

3. **Many assets**
   - Start with fewer assets, scale up once working

**Solution**:
```python
# Use Polars lazy mode for data prep
df = pl.scan_parquet("large_data.parquet").collect()

# Sample for testing
df_sample = df.sample(n=10000)
```

---

## Framework Compatibility Issues

### Results Don't Match VectorBT

**Possible Causes**:

1. **Different execution modes**
   - VectorBT uses vectorized execution
   - ml4t.backtest uses event-driven

2. **Same-bar re-entry behavior**
   - VectorBT Pro's `accumulate=False` prevents re-entry

**Solution**: Use VectorBT-compatible preset:
```python
from ml4t.backtest import BacktestConfig, PRESETS

config = BacktestConfig.from_yaml(PRESETS["vectorbt_pro"])
engine = Engine.from_config(feed, strategy, config)
```

### Results Don't Match Backtrader

**Possible Causes**:

1. **Order-On-Close vs Order-On-Open**
   - Backtrader uses COC (cheat-on-close) or COO flags
   - ml4t.backtest has similar but differently named options

**Solution**: Use Backtrader-compatible preset:
```python
config = BacktestConfig.from_yaml(PRESETS["backtrader"])
engine = Engine.from_config(feed, strategy, config)
```

---

## Common Error Messages

### "Cannot execute fill without current time"

**Cause**: Broker state not properly initialized before fill execution.

**Solution**: Ensure you're running through the Engine, not calling broker methods directly.

### "Unknown account_type"

**Cause**: Invalid account type specified.

**Solution**: Use `"cash"` or `"margin"`:
```python
engine = Engine(..., account_type="cash")  # or "margin"
```

### "Insufficient buying power for order"

**Cause**: Order would exceed available capital.

**Solution**: Check buying power before ordering:
```python
if broker.get_buying_power() > order_value * 1.1:  # 10% buffer
    broker.submit_order(asset, quantity)
```

---

## Getting Help

If you can't find a solution here:

1. **Check the API Reference**: See [docs/api/complete_reference.md](api/complete_reference.md)
2. **Review Examples**: See [examples/](../examples/) directory
3. **Enable Debug Mode**: Add print statements to track order flow
4. **Check Trade History**: Examine `engine.broker.trades` after backtest

---

*Last updated: January 2026*
