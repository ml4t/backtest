# Strategy API Examples

This directory contains example strategies demonstrating the unified Strategy API with both **Simple Mode** and **Batch Mode** execution.

## Execution Modes

The ml4t.backtest Strategy class supports two execution modes:

### 1. Simple Mode (Default)

**When to use:**
- Single-asset strategies
- Independent decision making per asset
- Lower memory overhead
- Most strategies fit this pattern

**How it works:**
- Override `on_market_event(event, context)`
- Called once per market event
- Access OHLCV via `event.open`, `event.close`, etc.
- Access ML signals via `event.signals` dict
- Access indicators via `event.indicators` dict
- Access market context via `context` dict parameter

**Example:**
```python
class MyStrategy(Strategy):
    def on_market_event(self, event, context=None):
        # Single event at a time
        ml_score = event.signals.get('ml_score', 0.0)
        vix = context.get('VIX', 0.0) if context else 0.0

        if ml_score > 0.7 and vix < 30:
            self.buy_percent(event.asset_id, 0.10, event.close)
```

### 2. Batch Mode

**When to use:**
- Multi-asset strategies (10+ assets)
- Cross-asset ranking and selection
- Portfolio optimization
- Pairs trading, statistical arbitrage
- Any strategy that needs to see all assets simultaneously

**How it works:**
- Override `on_timestamp_batch(timestamp, events, context)`
- Called once per timestamp with ALL events at that time
- Access list of MarketEvent objects
- Make decisions across all assets simultaneously
- Memory efficient: context shared across all events

**Example:**
```python
class MyStrategy(Strategy):
    def on_timestamp_batch(self, timestamp, events, context=None):
        # All events at this timestamp
        scores = {e.asset_id: e.indicators.get('momentum', 0) for e in events}
        top_5 = sorted(scores, key=scores.get, reverse=True)[:5]

        # Rebalance to top 5 assets
        weights = {asset: 0.20 for asset in top_5}
        prices = {e.asset_id: e.close for e in events}
        self.rebalance_to_weights(weights, prices)
```

## Mode Detection

Mode is **automatically detected** based on which methods are overridden:
- If `on_timestamp_batch()` is overridden → **Batch Mode**
- Otherwise → **Simple Mode** (default)

Check the detected mode:
```python
strategy = MyStrategy()
print(strategy.execution_mode)  # "simple" or "batch"
```

## Helper Methods

Both modes have access to these trading helpers:

### Position Information
```python
position = self.get_position(asset_id)  # Current quantity
cash = self.get_cash()                  # Available cash
value = self.get_portfolio_value()      # Total equity
```

### Position Sizing
```python
# Buy 10% of portfolio
self.buy_percent(asset_id, 0.10, price)

# Sell 50% of position
self.sell_percent(asset_id, 0.50)

# Close entire position
self.close_position(asset_id)
```

### ML-Specific Sizing
```python
# Size by ML confidence (Kelly-like)
self.size_by_confidence(
    asset_id=asset_id,
    confidence=0.80,      # ML confidence score
    max_percent=0.20,     # Max position at full confidence
    price=event.close
)
```

### Portfolio Rebalancing
```python
# Rebalance to target weights
target_weights = {"AAPL": 0.30, "GOOGL": 0.25, "MSFT": 0.20}
current_prices = {"AAPL": 150.0, "GOOGL": 2800.0, "MSFT": 380.0}

self.rebalance_to_weights(
    target_weights=target_weights,
    current_prices=current_prices,
    tolerance=0.02  # 2% tolerance before rebalancing
)
```

### P&L Tracking
```python
# Get unrealized P&L as percentage
pnl_pct = self.get_unrealized_pnl_pct(asset_id)
if pnl_pct and pnl_pct > 0.20:  # 20% gain
    self.close_position(asset_id)  # Take profit
```

## Example Strategies

### Simple Mode Examples

#### 1. `simple_ma_crossover.py` - Moving Average Crossover
- **Type:** Single-asset trend following
- **Indicators:** SMA-10, SMA-50
- **Logic:** Buy on golden cross, sell on death cross
- **Context:** VIX filtering (no trades when VIX > 30)
- **Demonstrates:**
  - Reading indicators from `event.indicators`
  - Using `buy_percent()` and `close_position()`
  - Context-dependent risk management

**Run it:**
```bash
cd examples/strategies
python simple_ma_crossover.py
```

### Batch Mode Examples

#### 2. `multi_asset_momentum.py` - Multi-Asset Momentum
- **Type:** Multi-asset portfolio (10 assets)
- **Indicators:** 20-day momentum
- **Logic:** Rank by momentum, hold top 5 equal-weighted
- **Context:** VIX-based position scaling
- **Demonstrates:**
  - Processing all events at once
  - Cross-asset ranking
  - Using `rebalance_to_weights()`
  - Dynamic position sizing based on regime

**Run it:**
```bash
cd examples/strategies
python multi_asset_momentum.py
```

## Data Requirements

### MarketEvent Structure

Strategies receive MarketEvent objects with three auxiliary data dictionaries:

```python
class MarketEvent:
    # Price data
    open: float
    high: float
    low: float
    close: float
    volume: int

    # ML predictions and signals
    signals: dict[str, float]  # e.g., {'ml_score': 0.85, 'confidence': 0.90}

    # Per-asset indicators
    indicators: dict[str, float]  # e.g., {'sma_10': 450.2, 'rsi': 65.3}

    # Market-wide context (same for all events at timestamp)
    context: dict[str, float]  # e.g., {'VIX': 18.5, 'SPY': 485.0}
```

### Data Feed Configuration

Configure which columns map to which dicts:

```python
data_feed = ParquetDataFeed(
    path="data.parquet",
    asset_id="SPY",
    timestamp_column="timestamp",
    signal_columns=["ml_score", "confidence"],      # → event.signals
    indicator_columns=["sma_10", "sma_50", "rsi"],  # → event.indicators
)

# Multi-asset version
data_feed = PolarsDataFeed(
    path="multi_asset_data.parquet",
    timestamp_column="timestamp",
    asset_id_column="asset_id",                     # Multi-asset support
    indicator_columns=["momentum", "volatility"],    # → event.indicators
)
```

### Context Data

Market-wide context is passed separately to the engine:

```python
context_data = {
    datetime(2024, 1, 15): {'VIX': 18.5, 'SPY': 485.0, 'regime': 'bull'},
    datetime(2024, 1, 16): {'VIX': 17.2, 'SPY': 487.3, 'regime': 'bull'},
    # ...
}

engine = BacktestEngine(
    data_feed=data_feed,
    strategy=strategy,
    context_data=context_data,  # Shared across all assets
    initial_capital=100000.0
)
```

## Performance Considerations

### Memory Efficiency

**Simple Mode:**
- Processes one event at a time
- Minimal memory overhead
- Suitable for large backtests (years of data)

**Batch Mode:**
- Collects all events for same timestamp
- Higher memory if many assets (100+)
- Context dict is shared (not duplicated)

**Recommendation:**
- Use Simple Mode for 1-10 assets
- Use Batch Mode for 10+ assets when cross-asset logic is needed
- For 100+ assets with independent logic, stick with Simple Mode

### Execution Speed

Both modes have similar performance:
- Simple Mode: 100k+ events/sec
- Batch Mode: 100k+ events/sec (slightly slower due to batch collection)

The engine automatically detects mode and optimizes accordingly.

## Best Practices

### 1. Start with Simple Mode
Most strategies don't need batch mode. Start simple:
```python
class MyStrategy(Strategy):
    def on_market_event(self, event, context=None):
        # Your logic here
        pass
```

### 2. Use Batch Mode for Cross-Asset Logic
Only switch to batch mode if you need to:
- Rank assets against each other
- Maintain portfolio-level constraints
- Implement pairs trading logic

### 3. Keep Context Lightweight
Context dict is shared across all events:
```python
# Good: Small, frequently-updated values
context = {'VIX': 18.5, 'SPY': 485.0}

# Bad: Large, rarely-used data
context = {'historical_prices': huge_dataframe}  # Don't do this!
```

### 4. Use Helper Methods
Don't manually create orders. Use helpers:
```python
# Good
self.buy_percent(asset_id, 0.10, price)

# Bad (verbose and error-prone)
from ml4t.backtest.execution.order import Order
order = Order(asset_id=asset_id, side=OrderSide.BUY, ...)
self.broker.submit_order(order)
```

### 5. Implement on_event() for Backward Compatibility
Always implement the required abstract method:
```python
def on_event(self, event):
    """Required abstract method."""
    pass  # Logic in on_market_event or on_timestamp_batch
```

## Common Patterns

### Pattern 1: ML Signal-Based Entry
```python
def on_market_event(self, event, context=None):
    ml_score = event.signals.get('ml_score', 0.0)
    confidence = event.signals.get('confidence', 0.0)

    if ml_score > 0.7 and confidence > 0.8:
        self.size_by_confidence(
            asset_id=event.asset_id,
            confidence=confidence,
            max_percent=0.20,
            price=event.close
        )
```

### Pattern 2: Indicator-Based Exit
```python
def on_market_event(self, event, context=None):
    position = self.get_position(event.asset_id)
    if position > 0:
        rsi = event.indicators.get('rsi', 50)
        if rsi > 70:  # Overbought
            self.close_position(event.asset_id)
```

### Pattern 3: Context-Dependent Risk
```python
def on_market_event(self, event, context=None):
    vix = context.get('VIX', 0.0) if context else 0.0

    if vix > 30:
        # High volatility - close positions
        if self.get_position(event.asset_id) != 0:
            self.close_position(event.asset_id)
    else:
        # Normal trading logic
        # ...
```

### Pattern 4: Portfolio Rebalancing
```python
def on_timestamp_batch(self, timestamp, events, context=None):
    # Rank by some metric
    scores = {e.asset_id: e.indicators.get('score', 0) for e in events}
    top_n = sorted(scores, key=scores.get, reverse=True)[:5]

    # Equal weight top N
    weights = {asset: 1.0 / len(top_n) for asset in top_n}
    prices = {e.asset_id: e.close for e in events}

    self.rebalance_to_weights(weights, prices, tolerance=0.05)
```

## Testing Your Strategy

Use the same pattern as the examples:

```python
# 1. Create sample data
data_path, context_data = create_sample_data()

# 2. Configure data feed
data_feed = ParquetDataFeed(
    path=data_path,
    asset_id="SPY",
    timestamp_column="timestamp",
    indicator_columns=["sma_10", "sma_50"]
)

# 3. Create strategy
strategy = MyStrategy()

# 4. Verify mode
print(f"Execution mode: {strategy.execution_mode}")

# 5. Run backtest
engine = BacktestEngine(
    data_feed=data_feed,
    strategy=strategy,
    context_data=context_data,
    initial_capital=100000.0
)
results = engine.run()

# 6. Analyze results
print(f"Total Return: {results['total_return']:.2f}%")
```

## Advanced Topics

### Custom Position Sizing
```python
def custom_size(self, asset_id, signal_strength, price):
    """Custom position sizing logic."""
    base_size = 0.10  # 10% base

    # Scale by signal strength
    adjusted_size = base_size * signal_strength

    # Apply portfolio-level constraints
    if self.get_portfolio_value() < 50000:
        adjusted_size *= 0.5  # Half size when equity low

    self.buy_percent(asset_id, adjusted_size, price)
```

### Dynamic Stop Losses
```python
def check_stops(self, asset_id, atr):
    """Volatility-adjusted stop losses."""
    pnl_pct = self.get_unrealized_pnl_pct(asset_id)
    if pnl_pct is None:
        return

    # Stop loss at 2 ATR
    stop_threshold = -2.0 * atr / self.get_position(asset_id)

    if pnl_pct < stop_threshold:
        self.close_position(asset_id)
```

### Regime-Dependent Logic
```python
def on_market_event(self, event, context=None):
    regime = context.get('regime', 'neutral') if context else 'neutral'

    if regime == 'bull':
        # Aggressive: larger positions
        position_size = 0.20
    elif regime == 'bear':
        # Defensive: smaller positions or cash
        position_size = 0.05
    else:
        # Neutral: moderate
        position_size = 0.10

    # Apply logic with regime-adjusted sizing
    # ...
```

## Troubleshooting

### Problem: Strategy not receiving events
**Solution:** Check that `on_event()` is implemented (required abstract method).

### Problem: Mode not detected correctly
**Solution:** Ensure you're overriding the right method name:
- Simple: `on_market_event(self, event, context=None)`
- Batch: `on_timestamp_batch(self, timestamp, events, context=None)`

### Problem: Helper methods raise "Broker not initialized"
**Solution:** Broker is injected by engine. Don't call helpers in `__init__()`.

### Problem: Batch mode receiving events one at a time
**Solution:** Check that events have identical timestamps. Sort data by timestamp.

### Problem: Context dict is empty
**Solution:** Pass `context_data` parameter to `BacktestEngine()`.

## Further Reading

- **CLAUDE.md**: Project development guidelines
- **src/ml4t/backtest/strategy/base.py**: Strategy class source code
- **src/ml4t/backtest/engine.py**: Engine batch collection logic
- **tests/unit/test_strategy_api.py**: Comprehensive test suite

---

**Questions or issues?** Check the test suite for additional examples and edge cases.
