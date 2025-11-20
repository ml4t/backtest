# BacktestEngine Architecture Diagnosis

**Date**: 2025-11-19
**Context**: Week-long development of multi-asset backtesting capability
**Critical Finding**: The BacktestEngine has fundamental architectural mismatches that prevent correct multi-asset portfolio execution

---

## Executive Summary

The BacktestEngine produces **13.1x lower returns** and **11.7x more positions** than the manual loop implementation for the same strategy. Three interconnected architectural issues create a cascading failure:

1. **Strategy Helper API Bug**: `buy_percent()` adds to positions instead of setting to target
2. **Per-Asset Event Processing**: Broker processes one asset at a time, delaying cross-asset rebalancing
3. **Execution Delay Timing**: Default 1-bar delay causes order accumulation during rebalancing

**Bottom Line**: The engine was designed for **single-asset**, **event-driven** strategies. Multi-asset **portfolio rebalancing** requires batch processing and simultaneous order execution, which the current architecture cannot support.

---

## The Original Plan (What We Tried to Build)

### Vision: Event-Driven Multi-Asset Backtesting

**Intended Architecture:**
1. **DataFeed** emits MarketEvents chronologically (all assets interleaved by timestamp)
2. **Strategy** receives events, makes decisions, submits orders via helper methods
3. **Broker** queues orders per asset, executes on next bar (execution delay)
4. **Portfolio** tracks positions and P&L across all assets
5. **RiskManager** validates orders and triggers exits

**Key Assumption**: This event-driven architecture would work for both single-asset strategies (MA crossover) AND multi-asset portfolios (top-N momentum).

### What We Actually Implemented

**✅ Data Layer** - `MultiSymbolDataFeed`
- Correctly merges price, signals, context data
- Emits 126,000 events in chronological order
- Performance: 34,751 events/sec (fast)

**✅ Engine Core** - `BacktestEngine.run()`
- Event loop processes MarketEvents sequentially
- Dispatches to broker, portfolio, risk manager
- Risk hooks (C, B, D) integrated correctly

**⚠️ Strategy Helpers** - `Strategy.buy_percent()`, `close_position()`
- Implemented but **fundamentally broken** for portfolio rebalancing
- Designed for accumulation, not target-setting

**❌ Multi-Asset Execution** - `SimulationBroker.on_market_event()`
- Processes orders **per asset, per event**
- No mechanism for simultaneous cross-asset rebalancing
- Execution delay causes order accumulation

---

## The Three Fatal Flaws

### Flaw #1: Strategy Helper API Mismatch

**Problem**: `Strategy.buy_percent(asset_id, percent, price)` calculates `quantity = portfolio_value × percent / price` and submits a BUY order for that quantity. It does NOT check existing position or calculate the difference.

**Example:**
```python
# Portfolio value: $100,000
# Call: buy_percent("AAPL", 0.04, $200)
#
# Calculation:
#   quantity = ($100,000 × 0.04) / $200 = 20 shares
#   Submits BUY 20 shares
#
# If we already have 20 shares → now have 40 shares (8% of portfolio)
# If we already have 0 shares → now have 20 shares (4% of portfolio)
```

**Manual Loop (Correct)**:
```python
target_shares = target_amt / price              # What we want
current_shares = portfolio.get_position(asset_id).quantity  # What we have
trade_shares = target_shares - current_shares  # What to buy/sell

if abs(trade_shares) > threshold:
    side = OrderSide.BUY if trade_shares > 0 else OrderSide.SELL
    broker.submit_order(Order(asset_id, side, abs(trade_shares)))
```

**Consequence**: Each rebalance ADDS 4% instead of SETTING to 4%, causing position accumulation.

### Flaw #2: Per-Asset Event Processing

**Problem**: The broker's `on_market_event(event)` only processes orders for `event.asset_id`. Multi-asset portfolios need to submit orders for 50+ assets simultaneously, but fills are staggered across hundreds of events.

**Execution Timeline:**
```
Day 2, Event 1 (AAPL):
  - Strategy processes Day 1 batch
  - Submits exit orders for: XYZ, ABC, DEF (not in top 25)
  - Submits buy orders for: AAPL, MSFT, GOOGL (top 25)
  - Engine dispatches Event 1 to broker
  - Broker processes asset_id=AAPL:
    - Moves _pending_orders["AAPL"] to _open_orders["AAPL"] (none pending yet)
    - Checks _open_orders["AAPL"] for fills (none)

Day 2, Event 2 (MSFT):
  - Broker processes asset_id=MSFT:
    - Moves _pending_orders["MSFT"] to _open_orders["MSFT"] (buy order is there!)
    - Marks as newly activated, CAN'T fill yet

Day 2, Event 3 (GOOGL):
  - Broker processes asset_id=GOOGL:
    - Moves _pending_orders["GOOGL"] to _open_orders["GOOGL"]
    - Marks as newly activated, CAN'T fill yet

...

Day 3, Event 1 (AAPL):
  - Strategy processes Day 2 batch (sees UNFILLED positions)
  - Submits MORE exit orders (duplicates!)
  - Submits MORE buy orders (accumulation!)

Day 3, Event 2 (MSFT):
  - Broker CAN NOW fill Day 2's MSFT buy order (finally!)
  - But Day 3's MSFT buy order also pending...
```

**Consequence**: Orders submitted together fill across 100+ bars, defeating the purpose of portfolio rebalancing.

### Flaw #3: Execution Delay Timing

**Problem**: `SimulationBroker(execution_delay=True)` (default) implements a **1-bar delay** to prevent look-ahead bias. For daily data:
- Order submitted on bar T → pending
- Bar T+1 → moves to open, marked "newly activated", can't fill
- Bar T+2 → can fill

This is correct for **single-asset strategies** where you see a signal and trade the next bar. But for **multi-asset portfolio rebalancing**, you want to exit ALL old positions and enter ALL new positions at the SAME TIME (end-of-day batch).

**Consequence**: Exit orders take 2 days to fill, entry orders take 2 days to fill, but the strategy checks positions every day and sees old positions still open, so it submits duplicate orders.

---

## Why Manual Loop Works

The manual loop bypasses the event-driven architecture:

```python
for timestamp in timestamps:
    # 1. Get ALL events for this day
    day_events = [MarketEvent(...) for row in day_data]

    # 2. Strategy logic sees FULL PICTURE
    top_assets = rank_by_ml_score(day_events)
    target_portfolio = {asset: 4% for asset in top_assets[:25]}

    # 3. Calculate DIFFERENCES and submit orders
    for asset_id in old_positions:
        if asset_id not in target_portfolio:
            broker.submit_order(exit_order)  # Exit old

    for asset_id in target_portfolio:
        target_shares = calculate_target()
        current_shares = get_current()
        if abs(target - current) > threshold:
            broker.submit_order(trade_order)  # Buy/sell difference

    # 4. Process ALL fills at once
    for event in day_events:
        broker.on_market_event(event)  # All orders fill this bar
```

**Key Differences:**
1. **Batch Processing**: Strategy sees all 500 assets at once
2. **Correct Order Sizing**: Calculates difference, not absolute target
3. **Simultaneous Fills**: All orders submitted before any fills processed
4. **No Execution Delay**: Direct call to broker, no queueing

**Result**: +770% return, 42 final positions (some held to end), 11.2 seconds

---

## What We Got Wrong

### Misconception #1: Event-Driven = Better

**We thought**: "Event-driven architecture is more realistic because you process data as it arrives."

**Reality**: For daily portfolio rebalancing, you DO process as a batch. You don't rebalance after each stock's close price - you wait for the full market close, analyze all stocks, then submit orders.

### Misconception #2: Helper Methods = Convenience

**We thought**: "`buy_percent()` is a nice helper that strategies can use to size positions."

**Reality**: `buy_percent()` is only useful for initial entry. Rebalancing requires `set_target_percent()` or manual difference calculation. The API name is misleading.

### Misconception #3: Execution Delay = Realism

**We thought**: "1-bar delay prevents look-ahead bias."

**Reality**: For end-of-day portfolio rebalancing, the correct timing is:
- See close prices at 4:00 PM
- Calculate new targets
- Submit orders for next-day open (fills at open prices)

1-bar delay is correct, but the implementation assumes per-asset decision-making, not portfolio-level batch decisions.

---

## How to Salvage This

### Option A: Fix the Engine (High Effort, Architectural)

**Required Changes:**

1. **Add `Strategy.set_target_percent()` method**
   ```python
   def set_target_percent(self, asset_id, target_pct, price):
       current_qty = self.get_position(asset_id)
       current_value = current_qty * price
       target_value = self.get_portfolio_value() * target_pct
       difference_value = target_value - current_value

       if abs(difference_value) > threshold:
           side = BUY if difference_value > 0 else SELL
           qty = abs(difference_value / price)
           self.broker.submit_order(Order(asset_id, side, qty))
   ```

2. **Add batch execution mode to Broker**
   ```python
   class SimulationBroker:
       def __init__(self, batch_fill_mode=False):
           self.batch_fill_mode = batch_fill_mode
           self._batch_orders = []

       def submit_order(self, order):
           if self.batch_fill_mode:
               self._batch_orders.append(order)
           else:
               # Existing per-asset queue

       def execute_batch(self, events):
           """Fill all batched orders using provided events."""
           for order in self._batch_orders:
               event = find_event_for_asset(order.asset_id, events)
               self._try_fill_order(order, event)
           self._batch_orders.clear()
   ```

3. **Update Engine to support batch mode**
   - Collect all events for a timestamp before strategy processes
   - Strategy submits orders (batched)
   - Engine calls `broker.execute_batch(events)`
   - All fills happen simultaneously

**Estimated Effort**: 2-3 days, requires significant refactoring

### Option B: Document Manual Loop Pattern (Low Effort, Pragmatic)

**Approach**: Accept that the engine is for single-asset strategies, provide a manual loop template for portfolio strategies.

**Documentation Structure:**
```markdown
# ml4t.backtest Usage Guide

## Single-Asset Strategies

Use BacktestEngine for event-driven strategies on one asset:
- MA crossover
- Breakout systems
- Mean reversion

## Multi-Asset Portfolio Strategies

Use the manual loop pattern for portfolio rebalancing:
- Top-N momentum
- Sector rotation
- Factor investing

Example: [top25_ml_strategy_complete.py](examples/integrated/)
```

**Estimated Effort**: 1 day, update README and examples

### Option C: Create Separate PortfolioEngine (Medium Effort, Clean)

**Approach**: Build a new `PortfolioEngine` specifically for batch-based portfolio strategies.

```python
class PortfolioEngine:
    """Engine for portfolio strategies that rebalance on a schedule."""

    def __init__(self, data_feed, strategy, schedule="daily"):
        self.data_feed = data_feed
        self.strategy = strategy
        self.schedule = schedule  # "daily", "weekly", "monthly"

    def run(self):
        for timestamp, events in self.data_feed.group_by_timestamp():
            # 1. Strategy sees all events at once
            signals = self.strategy.on_bar(timestamp, events)

            # 2. Calculate target portfolio
            targets = self.strategy.calculate_targets(signals)

            # 3. Generate rebalance orders
            orders = self.portfolio.rebalance_to_targets(targets)

            # 4. Execute all fills simultaneously
            fills = self.broker.execute_batch(orders, events)

            # 5. Update portfolio
            self.portfolio.apply_fills(fills)
```

**Estimated Effort**: 3-4 days, clean separation of concerns

---

## Recommended Path Forward

**Immediate (Today):**
1. ✅ Document the issues (this file)
2. ✅ Explain why manual loop works
3. Update README to clarify engine limitations

**Short-term (This Week):**
1. Implement Option A: Fix `buy_percent()` → `set_target_percent()`
2. Add batch execution mode to broker
3. Update engine to support batch fills
4. Test with top25 strategy → validate 770% return match

**Medium-term (Next Sprint):**
1. Build comprehensive test suite comparing engine vs manual loop
2. Validate with VectorBT, Backtrader (original Task 3)
3. Document performance characteristics of both approaches

**Long-term (Future):**
1. Consider Option C: Separate PortfolioEngine for clarity
2. Add live trading support (requires different execution model)
3. Optimize batch fill performance

---

## Lessons Learned

### What Worked

1. **MultiSymbolDataFeed**: Fast, correct, production-ready
2. **RiskManager Integration**: Hooks work as designed
3. **Portfolio Tracking**: Accurate P&L, position tracking
4. **Event System**: Clean dispatch, no memory leaks

### What Didn't Work

1. **Strategy Helper API**: Misleading names, broken semantics
2. **Per-Asset Execution**: Architectural mismatch for portfolios
3. **Execution Delay**: Correct concept, wrong implementation
4. **Validation Strategy**: Should have compared to manual loop earlier

### Key Insights

1. **Architecture Matters**: Event-driven ≠ batch processing
2. **Test Early**: Comparison tests catch architectural issues
3. **API Design**: Method names must match semantics (`buy_percent` ≠ `set_target_percent`)
4. **Performance**: Manual loop (11s) faster than engine (41s) due to overhead
5. **Simplicity**: Sometimes direct implementation beats abstraction

---

## Conclusion

The BacktestEngine is **not broken for its intended purpose** (single-asset, event-driven strategies). It is **architecturally incompatible** with multi-asset portfolio rebalancing strategies.

The week of development was not wasted - we built solid components (MultiSymbolDataFeed, RiskManager, Portfolio tracking). But we need to either:
- **Fix the execution model** (batch fills, correct helper methods)
- **Accept the limitations** (document, provide manual loop template)
- **Build a new engine** (PortfolioEngine for batch strategies)

**My recommendation**: Fix Option A (2-3 days) to salvage the architecture. The engine *can* work correctly with the right execution model and API fixes.

**User's instinct was correct**: "Did you create a monster? Did you create a core that doesn't really work?" Yes, but it's a **fixable monster** with the right architectural adjustments.
