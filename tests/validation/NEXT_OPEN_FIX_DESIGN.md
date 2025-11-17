# Design: Fix ml4t.backtest to Fill at Next Bar's Open

**Date**: 2025-11-16
**Problem**: ml4t.backtest fills orders at same bar's CLOSE with double-delay, not next bar's OPEN
**Impact**: 2.4% variance vs other frameworks; unrealistic daily backtesting

## Current Execution Flow (BROKEN)

### With `execution_delay=True` (current default):

**Day T** (Signal fires):
1. Strategy receives MarketEvent(T) with T's OHLC
2. Strategy sees signal, places order
3. `broker.submit_order()` called
4. `order_router.route_order()` → order goes to `_pending_orders[asset_id]`

**Day T+1**:
5. `broker.on_market_event(MarketEvent(T+1))` called
6. Line 612-613: `price = event.close` (T+1's CLOSE!)
7. Line 622-646: Process `_open_orders` (empty - no orders here yet)
8. Line 1008-1017: **Move `_pending_orders` → `_open_orders`** (WRONG TIMING!)
9. Orders now in open, waiting for NEXT event

**Day T+2**:
10. `broker.on_market_event(MarketEvent(T+2))` called
11. Line 612-613: `price = event.close` (T+2's CLOSE!)
12. Line 637-646: Fill orders at T+2's CLOSE

**Result**: Order placed on T fills at T+2's CLOSE (double delay, wrong price)

## What We Need (Industry Standard)

**Day T** (Signal fires):
1. Strategy receives MarketEvent(T) with T's OHLC
2. Strategy sees signal, places order
3. Order goes to pending

**Day T+1**:
4. `broker.on_market_event(MarketEvent(T+1))` called
5. **IMMEDIATELY move pending → open** (at START of event, not end)
6. **Determine fill price = event.OPEN** (not close!)
7. Fill orders at T+1's OPEN

**Result**: Order placed on T fills at T+1's OPEN (correct!)

## Required Code Changes

### Fix #1: Move Pending Orders at START of Event

**File**: `src/ml4t/backtest/execution/broker.py`
**Method**: `on_market_event()`
**Line**: Move lines 1006-1017 to BEGINNING of method (before line 622)

```python
def on_market_event(self, event: MarketEvent) -> list[FillEvent]:
    """Process market event and generate fills."""
    # Increment bar counter
    self.trade_tracker.on_bar()

    fills = []
    asset_id = event.asset_id

    # FIX #1: Move pending orders to open FIRST (at START of event)
    # This allows them to be filled on THIS event (next bar's open)
    if self.execution_delay and asset_id in self._pending_orders:
        for order, _ in self._pending_orders[asset_id]:
            self._open_orders[asset_id].append(order)
        self._pending_orders[asset_id].clear()

    # Determine execution price (FIX #2 will modify this)
    if event.close is not None:
        price = event.close
    ...
```

### Fix #2: Use Open Price for Fills (When Configured)

**Problem**: Broker always uses `event.close` (line 612-613)
**Solution**: Add configuration to use `event.open` for next-open fills

**Option A**: Add parameter to broker `__init__`:
```python
def __init__(
    self,
    ...
    fill_price_source: Literal["open", "close"] = "close",  # NEW
):
    self.fill_price_source = fill_price_source
```

Then in `on_market_event()`:
```python
# Determine execution price based on configuration
if self.fill_price_source == "open" and event.open is not None:
    price = event.open
elif event.close is not None:
    price = event.close
elif event.price is not None:
    price = event.price
else:
    return fills  # No price available
```

**Option B**: Detect from execution_delay:
```python
# If execution_delay enabled, use open (realistic)
# If execution_delay disabled, use close (legacy/testing)
if self.execution_delay and event.open is not None:
    price = event.open
elif event.close is not None:
    price = event.close
...
```

**Recommendation**: **Option B** - simpler, automatically correct behavior when `execution_delay=True`

### Fix #3: Update BacktestWrapper to Enable execution_delay

**File**: `tests/validation/common/engine_wrappers.py`

**Current Issue**: BacktestWrapper creates broker but doesn't pass `execution_delay` parameter

**Check**: Line 120-150 of engine_wrappers.py - does it create SimulationBroker?

Let me check this...

## Expected Behavior After Fixes

### Test Case: Signal on 2020-04-07

**Data**:
- 2020-04-07: Open=$67.70, Close=$64.86 (signal fires here)
- 2020-04-08: Open=$65.685, Close=$66.5175

**Current (BROKEN)**:
- Signal on 2020-04-07
- Pending on 2020-04-08 (no fill)
- Fill on 2020-04-09 at close (NOT SHOWN IN DATA)

**After Fix**:
- Signal on 2020-04-07 (strategy sees this bar's data, decides to trade)
- Fill on 2020-04-08 at OPEN = **$65.685** ✓

**All frameworks should match**:
- ml4t.backtest: $65.685 (after fix)
- VectorBT: $65.685 (already fixed in adapter)
- Backtrader: $65.685 (needs verification)
- Zipline: $65.685 (needs verification)

## Testing Plan

1. **Unit test**: Test broker fills at open with execution_delay=True
2. **Integration test**: Run validation with all 4 frameworks
3. **Verify**: All frameworks fill first trade at $65.685 on 2020-04-08
4. **Verify**: Maximum variance < 0.1% after fractional shares also fixed

## Implementation Order

1. ✅ **Document design** (this file)
2. ⏳ **Implement Fix #1**: Move pending→open at start
3. ⏳ **Implement Fix #2**: Use event.open for fill price
4. ⏳ **Test**: Verify ml4t.backtest fills at correct price
5. ⏳ **Fix Backtrader fractional shares**
6. ⏳ **Fix Zipline margin + fractional shares**
7. ⏳ **Run full validation**: Verify <0.1% variance

## Alternative: Keep Current Delay, Add Open Price

If we want to preserve the double-delay behavior for some reason (legacy compatibility), we could:
- Keep pending→open move at END of event (current behavior)
- Still change fill price from close to open

But this would mean:
- Signal on T → Fill on T+2's open (still wrong for daily data!)
- Not recommended - industry standard is T+1

## Conclusion

The fixes are straightforward:
1. Move 11 lines of code (pending→open) from end to start of method
2. Change 1 line to use `event.open` instead of `event.close`
3. Total impact: ~15 lines changed in broker.py

This will make ml4t.backtest match industry-standard daily backtesting behavior.
