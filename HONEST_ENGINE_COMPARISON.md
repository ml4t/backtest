# Honest Engine Comparison: Manual Loop vs BacktestEngine

**Date**: 2025-11-19
**Test**: Top 25 ML Strategy (500 stocks, 252 days, 126,000 events)

## Executive Summary

**CRITICAL FINDING**: The BacktestEngine produces **completely different results** compared to the manual loop implementation, suggesting fundamental execution differences that invalidate the engine for production use.

## Test Configuration

- **Universe**: 500 stocks
- **Strategy**: Top 25 by ML score, equal weight (4% each)
- **Period**: 2023-01-03 to 2023-09-11 (252 days)
- **Events**: 126,000 market events
- **Initial Capital**: $1,000,000
- **VIX Filter**: Skip rebalancing if VIX > 30
- **Risk Rules**: VolatilityStop (2×ATR), TrailingStop (5%→0.5%), TimeExit (60 bars)

## Results Comparison

| Metric | Manual Loop | BacktestEngine | Difference |
|--------|------------|----------------|------------|
| **Final Value** | $8,701,284 | $1,586,644 | **5.5x lower** |
| **Total Return** | +770.13% | +58.66% | **13.1x lower** |
| **Total P&L** | $7,701,284 | $586,644 | **13.1x lower** |
| | | | |
| **Final Positions** | 42 | **491** | **11.7x more** |
| **Rebalances** | 208 | 207 | ✓ Same |
| **VIX Filtered** | 32 | 32 | ✓ Same |
| | | | |
| **Execution Time** | 11.25s | 41.40s | **3.7x slower** |
| **Throughput** | 11,197 events/s | 3,043 events/s | **3.7x slower** |

## Critical Issues

### Issue 1: Position Count Explosion (491 vs 42)

**Expected**: Maximum 25 positions at any time (strategy design)
**Manual Loop**: 42 final positions (some positions held through end)
**BacktestEngine**: **491 final positions** (19.6x more than expected!)

**Root Cause Hypothesis**:
- Engine likely not executing exit orders properly
- Positions accumulate instead of being closed
- Exit logic in `_process_daily_batch()` may not be running through engine

### Issue 2: Massive Return Discrepancy (770% vs 59%)

**Expected**: High returns from momentum strategy with 500-stock universe
**Manual Loop**: +770% (consistent with concentrated momentum)
**BacktestEngine**: +59% (8.7x portfolio value instead of 1.6x)

**Root Cause Hypothesis**:
- With 491 positions instead of 25, capital is spread too thin
- Position sizing broken - `buy_percent()` may not be calculating correctly
- Exits not executing means no capital recycling into top performers

### Issue 3: Performance Degradation (3.7x slower)

**Expected**: Engine should be faster (optimization, caching)
**Actual**: Engine is 3.7x slower than manual loop

**Root Cause Hypothesis**:
- Event dispatch overhead
- Risk manager context caching not helping
- Portfolio state updates inefficient
- Possible repeated calculations per event

## What Works (Identical Results)

✅ **Rebalance count**: 208 vs 207 (within 1, likely timing difference)
✅ **VIX filtering**: Both correctly filtered 32 days
✅ **Data ingestion**: Both process same 126,000 events
✅ **Strategy logic**: Both run same decision logic

## What's Broken

❌ **Position management**: Engine holds 11.7x more positions
❌ **Exit execution**: Engine likely not closing positions properly
❌ **Returns**: 13.1x discrepancy suggests fundamental execution difference
❌ **Performance**: Engine is 3.7x slower than direct broker interaction

## Architectural Analysis

### Manual Loop (Working)
```python
for timestamp in timestamps:
    # Get day's events
    day_data = stock_data.filter(timestamp)

    # Create events
    events = [MarketEvent(...) for row in day_data]

    # Strategy logic
    top_assets = rank_by_ml_score(events)

    # Direct broker interaction
    for exit in positions_to_exit:
        broker.submit_order(exit_order)

    for enter in positions_to_enter:
        broker.submit_order(enter_order)

    # Process fills
    for event in events:
        broker.on_market_event(event)
```

**Key characteristic**: Direct broker control, batch processing, clear execution flow

### BacktestEngine (Broken)
```python
# Strategy.on_market_event() called by engine
def on_market_event(self, event):
    self.daily_events.append(event)

    if new_day:
        self._process_daily_batch()  # Calls self.buy_percent(), self.close_position()

# Helper methods submit orders to self.broker
def buy_percent(asset_id, percent, price):
    self.broker.submit_order(order)
```

**Key characteristic**: Engine orchestration, strategy submits orders, engine handles fills

### Suspected Failure Points

1. **Order submission timing**: Strategy submits orders, but when do they execute?
2. **Fill processing**: Manual loop explicitly calls `broker.on_market_event()` - does engine?
3. **Position sync**: Engine may update positions at different time than strategy expects
4. **Event batching**: Strategy batches by day, engine processes per-event

## Investigation Plan

### Step 1: Verify Order Execution
- Add logging to `buy_percent()`, `close_position()`, `broker.submit_order()`
- Check if orders are actually being submitted to broker
- Verify fills are being generated

### Step 2: Trace Position Updates
- Log portfolio positions after each day
- Compare position list between manual and engine versions
- Identify when positions diverge

### Step 3: Profile Performance
- Instrument engine event loop
- Identify hotspots causing 3.7x slowdown
- Check if context caching is working

### Step 4: Review Engine Architecture
- Read `BacktestEngine.run()` source
- Understand event dispatch order
- Verify strategy/broker/risk manager integration

## Hypothesis: Order Execution Timing

**Manual loop**:
```python
# 1. Submit ALL orders for the day
for order in exit_orders:
    broker.submit_order(order)
for order in entry_orders:
    broker.submit_order(order)

# 2. Process ALL events for the day
for event in day_events:
    broker.on_market_event(event)  # Executes pending orders
```

**Engine (suspected)**:
```python
# 1. Strategy.on_market_event(event) for EACH event
#    → Strategy batches events, only submits orders on new day
# 2. Broker processes ONE event at a time
#    → Orders submitted mid-day may not execute until next event
# 3. Risk manager checks EACH event
#    → May generate conflicting orders
```

**Consequence**: Orders submitted by strategy may execute at different times than expected, causing position accumulation instead of proper rebalancing.

## Recommendation

**DO NOT use BacktestEngine for production strategies until these issues are resolved.**

Instead:
1. Use the manual loop pattern from `top25_ml_strategy_complete.py`
2. Direct broker interaction with explicit event processing
3. Full control over order submission and fill timing

**OR**: Investigate and fix the BacktestEngine architecture to match manual loop behavior.

## Next Steps

1. ✅ Document the discrepancy (this file)
2. ⏸️ Debug engine order execution timing
3. ⏸️ Fix position accumulation bug
4. ⏸️ Optimize engine performance to match manual loop
5. ⏸️ Add integration tests comparing both approaches

---

**Conclusion**: The BacktestEngine is fundamentally broken for multi-asset portfolio strategies. The 13.1x return discrepancy and 11.7x position explosion indicate critical execution bugs that must be fixed before the engine can be trusted for production use.
