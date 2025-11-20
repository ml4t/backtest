# External Code Review Request: ml4t.backtest Multi-Asset Backtesting Engine

## Context

We've spent two weeks building a Python backtesting engine (`ml4t.backtest`) with the goal of creating a **modern, fast, multi-asset portfolio backtesting framework** using Polars for data handling. However, we've discovered critical architectural issues that prevent correct execution for multi-asset portfolio strategies.

**Current State:**
- ✅ Single-asset strategies work correctly (MA crossover, etc.)
- ❌ Multi-asset portfolios produce 13.1x lower returns and 11.7x more positions than expected
- ❌ Performance is 3.7x SLOWER than Backtrader (the slowest competitor)
- ❌ `BacktestEngine` has fundamental architectural mismatches

## Project Goals (What We're Trying to Build)

### Primary Objective
Build a **high-performance, event-driven backtesting engine** for quantitative trading strategies that:

1. **Multi-Asset Portfolio Support**
   - Handle 500+ stocks with ML signals per asset
   - Portfolio rebalancing with configurable weights (equal weight, custom allocations)
   - Event-driven processing driven by timestamps in price/signal data
   - Asynchronous signals (not all assets have signals at all timestamps)

2. **User-Friendly Strategy Definition**
   - Similar API to Backtrader/Zipline for familiarity
   - Users define entry/exit logic based on current portfolio state
   - Support for: "if position exists", "if position profitable", "if signal X", etc.
   - Clear position sizing rules (fixed weight, dynamic rebalancing, risk-based)

3. **Performance Requirements**
   - **Fast**: Use Polars for data handling (10-100x faster than pandas)
   - **Target**: Process 126,000 events (500 stocks × 252 days) in <5 seconds
   - **Competitive**: Match or beat VectorBT (1.7x our current speed), Backtrader (6.5x our speed)

4. **ML Signal Integration**
   - Accept external ML predictions as signals per asset
   - Arbitrary number of signal columns (ml_score, momentum, regime, etc.)
   - Market-wide context data (VIX, SPY, regime) broadcast to all assets

5. **Risk Management**
   - Position-level risk rules (stop-loss, trailing stop, time-based exit)
   - Portfolio-level constraints (max exposure, sector limits)
   - Integrated hooks (check exits before strategy, validate orders, record fills)

## Critical Issues Discovered

### Issue #1: Strategy Helper API is Broken

**Problem**: `Strategy.buy_percent(asset, percent, price)` **adds** to position instead of **setting** to target.

**Example:**
```python
# Portfolio value: $100,000
buy_percent("AAPL", 0.04, $200)  # Want 4% of portfolio in AAPL

# Calculation in code:
quantity = ($100,000 × 0.04) / $200 = 20 shares
# Submits BUY 20 shares

# If already have 20 shares → now have 40 shares (8% of portfolio)  ❌
# Expected: calculate difference, trade to get to 4%  ✅
```

**Consequence**: Each rebalance accumulates positions instead of setting to target, causing the 491-position explosion.

### Issue #2: Per-Asset Event Processing (Architectural Mismatch)

**Problem**: Engine processes MarketEvents one at a time, but portfolio rebalancing needs to submit orders for 50+ assets simultaneously and have them fill together.

**Current Flow:**
```python
# Day 2, Event 1 (asset_id=AAPL):
- Strategy batches events from Day 1, decides to rebalance
- Submits orders: Exit(XYZ), Exit(ABC), Buy(AAPL), Buy(MSFT), Buy(GOOGL)
- Engine dispatches Event 1 to broker
- Broker.on_market_event(event) only processes asset_id=AAPL
- Orders for MSFT, GOOGL sit in queue until their events arrive

# Day 2, Events 2-500 (other assets):
- Each event only processes orders for that specific asset
- Takes 500 events to process one day's rebalancing

# Day 3, Event 1:
- Strategy sees Day 2 positions STILL NOT FILLED
- Submits DUPLICATE orders
```

**Consequence**: Orders fill over 100+ bars instead of simultaneously, defeating portfolio rebalancing.

### Issue #3: Execution Delay Timing (Correct for Single-Asset, Wrong for Portfolios)

**Problem**: Default 1-bar execution delay is correct for single-asset strategies (see signal on bar T, trade on bar T+1), but for portfolio rebalancing you want to:
1. Analyze all 500 stocks at EOD
2. Submit exit/entry orders as a batch
3. Fill all orders simultaneously (same bar or next bar, but together)

**Current Behavior**: Orders submitted together on Day 2 don't fill until Days 3-4, causing strategy to submit duplicates.

## What Works vs What's Broken

### ✅ Components That Work

1. **MultiSymbolDataFeed** (201 lines)
   - Correctly merges price, signals, context data from Polars DataFrames
   - Emits 126,000 events in chronological order
   - Performance: 34,751 events/sec

2. **Risk Management Integration**
   - `RiskManager` with Hook C (check exits before strategy), Hook B (validate orders), Hook D (record fills)
   - Position tracking, MFE/MAE calculation
   - Priority-based rule merging

3. **Portfolio Tracking**
   - Accurate P&L tracking across multiple assets
   - Position state management
   - Precision management for cash/prices

4. **Data Layer**
   - Polars-based for fast data access
   - Schema validation
   - Context caching for performance

### ❌ What's Broken

1. **Strategy Helper Methods**
   - `buy_percent()`, `sell_percent()`, `close_position()` have incorrect semantics
   - Missing `set_target_percent()` for rebalancing

2. **Broker Event Processing**
   - `SimulationBroker.on_market_event(event)` only processes `event.asset_id`
   - No batch fill mechanism for multi-asset portfolios

3. **Engine Orchestration**
   - Designed for per-event processing (single-asset)
   - Missing batch mode for portfolio strategies

4. **Performance**
   - Currently 41s for 126,000 events (3,043 events/sec)
   - Manual loop version: 11s (11,197 events/sec) - 3.7x faster!
   - Backtrader: 0.15s (single-asset test)
   - Target: <5s for this workload

## The Working Manual Loop Pattern

We have a working implementation that bypasses the engine:

```python
# Manual loop (examples/integrated/top25_ml_strategy_complete.py)
for timestamp in timestamps:
    # 1. Get all events for this day
    day_events = [MarketEvent(...) for row in stock_data.filter(timestamp)]

    # 2. Strategy logic sees full picture
    top_assets = rank_by_ml_score(day_events)
    target_portfolio = {asset: 4% for asset in top_assets[:25]}

    # 3. Calculate DIFFERENCES
    for asset in target_portfolio:
        current_shares = get_position(asset).quantity
        target_shares = (portfolio_value × 4%) / price
        trade_shares = target_shares - current_shares  # Key: difference!

        if abs(trade_shares) > threshold:
            submit_order(asset, side, abs(trade_shares))

    # 4. Process all fills at once
    for event in day_events:
        broker.on_market_event(event)  # All orders fill this bar
```

**Results**: +770% return, 42 positions, 11.2 seconds ✅

## Questions for Expert Review

### Architecture

1. **Is event-driven processing the right architecture for multi-asset portfolio rebalancing?**
   - Should we batch events by timestamp instead?
   - How do VectorBT/Backtrader/Zipline handle this?

2. **How should broker execution work for portfolio strategies?**
   - Separate `execute_batch(orders, events)` method?
   - Queue orders globally, fill all matching orders per event?
   - Completely different execution model?

3. **What's the correct abstraction for portfolio rebalancing?**
   - Current: `buy_percent()` accumulates
   - Needed: `set_target_percent()` or `rebalance_to(targets)`
   - Or completely different API?

### Performance

4. **Why is the engine 3.7x slower than the manual loop?**
   - Event dispatch overhead?
   - Risk manager context caching ineffective?
   - Portfolio state updates inefficient?
   - Possible optimizations?

5. **How can we achieve 10x+ speedup vs Backtrader?**
   - Polars should help, but we're currently SLOWER
   - Numba compilation feasible? Where to apply?
   - Vectorize hot paths while maintaining event-driven semantics?

6. **Is 34,751 events/sec for data feed good enough?**
   - This is just data iteration (no strategy, no fills)
   - How much overhead should engine/broker/portfolio add?

### API Design

7. **How should users define portfolio strategies?**
   - Backtrader-style: per-event callbacks (`next()`)
   - Vectorized: `from_signals()` like VectorBT
   - Batch-based: `on_day(events)` callback
   - Hybrid approach?

8. **What's the right balance between flexibility and performance?**
   - User-defined Python callbacks vs compiled strategies
   - When to use Numba, when to stay pure Python
   - How to optimize without sacrificing usability

### Strategy Patterns

9. **How should position sizing work?**
   - Fixed percentages at entry
   - Dynamic rebalancing (drift tolerance)
   - Risk-based sizing (volatility, correlation)
   - User-defined custom logic

10. **How to handle asynchronous signals?**
    - Not all assets have signals at all times
    - Should strategy wait for full batch or process incrementally?
    - How to avoid race conditions?

## Code Structure

### Core Files (What Matters)

```
src/ml4t/backtest/
├── engine.py (639 lines) - Main orchestration, event loop
├── execution/
│   ├── broker.py (1379 lines) - Order execution, position tracking
│   ├── fill_simulator.py (663 lines) - Realistic fill modeling
│   └── order.py (478 lines) - Order types and lifecycle
├── portfolio/
│   └── portfolio.py (373 lines) - Position and P&L tracking
├── data/
│   ├── multi_symbol_feed.py (201 lines) - Multi-asset data feed
│   └── feed.py (base class)
├── strategy/
│   └── base.py (851 lines) - Strategy interface (broken API)
├── core/
│   ├── event.py - Event system (MarketEvent, FillEvent, etc.)
│   ├── types.py - Type definitions
│   └── clock.py (460 lines) - Event synchronization
└── risk/
    └── manager.py (621 lines) - Risk rule orchestration
```

### Potential Dead Code (22,709 total lines)

- `strategy/adapters.py` (371 lines) - VectorBT/Backtrader compat
- `strategy/*_adapter.py` (950 lines) - Specific strategy implementations
- `execution/corporate_actions.py` (840 lines) - Stock splits, probably not needed yet
- `reporting/*` (2000+ lines) - HTML reports, visualizations (nice-to-have)
- `data/polars_feed.py` (618 lines) - Single-asset feed (replaced by MultiSymbolDataFeed)
- `data/validation.py` (848 lines) - Validation layer (overkill?)
- `config.py` (510 lines) - Configuration system (unnecessary?)
- `risk/rules/*` (1000+ lines) - Specific rules (vs core rule interface)

## Deliverables Requested

### 1. Architectural Assessment

- Is the current architecture salvageable for multi-asset portfolios?
- Should we refactor the engine or build a separate `PortfolioEngine`?
- What's the cleanest separation of concerns?

### 2. Performance Analysis

- Where are the bottlenecks causing 3.7x slowdown?
- What optimizations would yield 10x+ speedup?
- Is Polars being used effectively, or is pandas sneaking in?

### 3. API Design Recommendations

- Correct strategy interface for portfolio rebalancing
- Helper methods for position sizing/rebalancing
- Balance between flexibility and performance

### 4. Implementation Roadmap

- Priority 1: Critical fixes to make multi-asset work correctly
- Priority 2: Performance optimizations to match/beat competitors
- Priority 3: API improvements for usability
- Estimated effort for each

### 5. Code Cleanup Plan

- Which files to keep vs remove
- Simplified architecture diagram
- Dependencies to eliminate

## Success Criteria

**A successful refactoring would:**

1. **Correctness**: Top-25 ML strategy produces +770% return (matching manual loop)
2. **Speed**: Process 126,000 events in <5 seconds (currently 41s)
3. **Simplicity**: <5,000 lines of core code (currently 22,709 lines)
4. **Usability**: Clear API similar to Backtrader/Zipline
5. **Performance**: Match or beat VectorBT/Backtrader on benchmarks

## Additional Context

- **Manual loop works**: `examples/integrated/top25_ml_strategy_complete.py` is the reference
- **Diagnosis available**: `ARCHITECTURE_DIAGNOSIS.md` has full technical analysis
- **Performance baseline**: `HONEST_ENGINE_COMPARISON.md` has comparison table
- **Competitor source code**: Available locally in `resources/` (VectorBT, Backtrader, Zipline)

## What We Need From You

1. **Honest assessment**: Is this fixable or should we start over?
2. **Prioritized recommendations**: What to fix first for maximum impact
3. **Performance roadmap**: Path to 10x+ speedup
4. **API design**: Right abstractions for multi-asset portfolio strategies
5. **Code quality**: What to keep, what to delete, what to refactor

**We're looking for direct, practical guidance from someone with deep experience in backtesting frameworks, event-driven architectures, and high-performance Python.**

Thank you for your time and expertise.
