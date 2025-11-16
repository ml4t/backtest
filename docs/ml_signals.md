# ML Signal Integration Guide

**Complete guide to using machine learning predictions in ml4t.backtest strategies.**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Signal Workflow](#signal-workflow)
4. [Context Integration](#context-integration)
5. [Helper Methods Reference](#helper-methods-reference)
6. [Advanced Patterns](#advanced-patterns)
7. [Performance Characteristics](#performance-characteristics)
8. [Complete Working Example](#complete-working-example)
9. [Best Practices](#best-practices)

---

## Introduction

ml4t.backtest treats **ML predictions as first-class citizens** through its signals architecture. Instead of computing features inside your strategy (which risks look-ahead bias), you pre-compute ML predictions and pass them alongside market data.

**Key Benefits:**
- **No look-ahead bias** - Predictions computed externally with proper time alignment
- **Clean separation** - ML inference separate from execution logic
- **Scalable** - Pre-compute predictions once, backtest many parameter sets
- **Realistic** - Mirrors production ML pipelines (model API → execution engine)

**What's New (Phase 1 & 1b - November 2025):**
- ✅ Signals dict in MarketEvent for unlimited ML predictions
- ✅ Context integration for market-wide indicators (VIX, regime)
- ✅ 9 helper methods for clean strategy code
- ✅ Memory-efficient ContextCache (2-5x savings)
- ✅ Comprehensive test fixtures for rapid development

---

## Quick Start

**1. Prepare data with ML signals:**

```python
import polars as pl

# Your ML predictions (pre-computed)
data = pl.DataFrame({
    "timestamp": [...],
    "open": [...],
    "high": [...],
    "low": [...],
    "close": [...],
    "volume": [...],
    "prediction": [0.72, 0.45, 0.89, ...],  # ML signal: prob of up move
    "confidence": [0.85, 0.60, 0.92, ...],  # Model confidence
})

data.write_parquet("ml_data.parquet")
```

**2. Create data feed with signal columns:**

```python
from ml4t.backtest.data import ParquetDataFeed

feed = ParquetDataFeed(
    path="ml_data.parquet",
    asset_id="AAPL",
    signal_columns=["prediction", "confidence"],  # Extract as signals
)
```

**3. Write ML strategy using helper methods:**

```python
from ml4t.backtest import Strategy

class MLStrategy(Strategy):
    def on_market_event(self, event, context=None):
        # Access ML signals
        prediction = event.signals["prediction"]
        confidence = event.signals["confidence"]

        # Simple logic with helper methods
        if prediction > 0.7 and confidence > 0.8:
            self.buy_percent(event.asset_id, percent=0.10, price=event.close)
        elif prediction < 0.3:
            self.close_position(event.asset_id)
```

**4. Run backtest:**

```python
from ml4t.backtest import BacktestEngine

engine = BacktestEngine(
    data_feed=feed,
    strategy=MLStrategy(),
    initial_capital=100_000,
)

results = engine.run()
print(f"Return: {results['total_return']:.2%}")
```

---

## Signal Workflow

### How Signals Work

**1. Data Preparation (Offline):**
```python
# Train ML model
model = train_your_model(features, labels)

# Generate predictions for backtest period
predictions = model.predict_proba(features)[:, 1]  # Probability of class 1
confidences = calculate_confidence(model, features)

# Add to market data
df = df.with_columns([
    pl.Series("ml_pred", predictions),
    pl.Series("ml_conf", confidences),
])
```

**2. Signal Extraction (DataFeed):**
```python
feed = ParquetDataFeed(
    path="data.parquet",
    asset_id="AAPL",
    signal_columns=["ml_pred", "ml_conf"],  # These become signals
)

# DataFeed automatically extracts these columns into event.signals dict
```

**3. Signal Access (Strategy):**
```python
class MyStrategy(Strategy):
    def on_market_event(self, event, context=None):
        # event.signals is a dict[str, float]
        ml_pred = event.signals.get("ml_pred", 0.0)
        ml_conf = event.signals.get("ml_conf", 0.0)

        # Use in trading logic
        if ml_pred > 0.65 and ml_conf > 0.75:
            self.buy_percent("AAPL", percent=0.15, price=event.close)
```

### Multiple Signals

You can have unlimited signals per event:

```python
# Data with multiple ML models
df = pl.DataFrame({
    "timestamp": [...],
    "open": [...], "high": [...], "low": [...], "close": [...], "volume": [...],

    # Entry/exit predictions
    "entry_signal": [...],      # Probability to enter long
    "exit_signal": [...],       # Probability to exit

    # Model confidence
    "entry_confidence": [...],
    "exit_confidence": [...],

    # Auxiliary signals
    "volatility_forecast": [...],  # Predicted vol for position sizing
    "regime_prob": [...],          # Probability of bull regime
})

feed = ParquetDataFeed(
    path="data.parquet",
    asset_id="AAPL",
    signal_columns=[
        "entry_signal", "exit_signal",
        "entry_confidence", "exit_confidence",
        "volatility_forecast", "regime_prob",
    ],
)
```

---

## Context Integration

**Context** provides market-wide indicators that apply to all assets (VIX, SPY price, regime indicators).

### Why Context?

**Problem:** In a 500-stock universe, duplicating VIX across all 500 DataFrames wastes memory.

**Solution:** Shared context object with timestamp-based caching.

**Memory Savings:** 2-5x reduction (measured with tracemalloc, see Performance section).

### Using Context

**1. Prepare context data:**

```python
# Context is a dict[timestamp, dict[str, float]]
context_data = {
    datetime(2023, 1, 3, 9, 30): {"VIX": 18.5, "SPY": 380.2, "regime": 1.0},
    datetime(2023, 1, 4, 9, 30): {"VIX": 19.1, "SPY": 378.5, "regime": 1.0},
    datetime(2023, 1, 5, 9, 30): {"VIX": 22.3, "SPY": 375.0, "regime": 0.0},
    # ... one entry per timestamp
}
```

**2. Pass to BacktestEngine:**

```python
engine = BacktestEngine(
    data_feed=feed,
    strategy=strategy,
    context_data=context_data,  # Add this
    initial_capital=100_000,
)
```

**3. Access in strategy:**

```python
class ContextAwareStrategy(Strategy):
    def on_market_event(self, event, context=None):
        # Context is dict[str, float] for current timestamp
        vix = context.get("VIX", 0.0) if context else 0.0
        regime = context.get("regime", 0.0) if context else 0.0

        # Risk-off during high volatility
        if vix > 30:
            self.close_position(event.asset_id)
            return

        # Normal trading logic
        prediction = event.signals.get("prediction", 0.0)
        if prediction > 0.7 and regime > 0.5:  # Bull regime
            self.buy_percent(event.asset_id, percent=0.10, price=event.close)
```

### Context Best Practices

**DO:**
- Use for market-wide indicators (VIX, interest rates, market regime)
- Use for indicators computed once per timestamp
- Use for multi-asset strategies (100+ assets)

**DON'T:**
- Use for asset-specific data (put in signals instead)
- Use for single-asset strategies (join into signals DataFrame instead)
- Mutate context dict (it's shared across assets)

---

## Helper Methods Reference

ml4t.backtest provides 9 helper methods to simplify strategy code.

### Standard Trading Helpers (6 methods)

**Position & Portfolio State:**

```python
# Get current position (shares held)
position = self.get_position("AAPL")  # Returns int (negative for short)

# Get available cash
cash = self.get_cash()  # Returns float

# Get total portfolio value (cash + positions)
value = self.get_portfolio_value()  # Returns float
```

**Order Submission:**

```python
# Buy as percentage of portfolio
self.buy_percent(
    asset_id="AAPL",
    percent=0.10,        # 10% of portfolio value
    price=150.0,         # Current price for quantity calculation
)

# Sell as percentage of portfolio
self.sell_percent(
    asset_id="AAPL",
    percent=0.05,        # 5% of portfolio value
    price=150.0,
)

# Close entire position (liquidate)
self.close_position("AAPL")
```

### ML-Specific Helpers (3 methods)

**Confidence-Based Position Sizing:**

```python
# Size position based on ML confidence (Kelly-like)
self.size_by_confidence(
    asset_id="AAPL",
    confidence=0.85,     # ML model confidence (0-1)
    max_percent=0.20,    # Maximum position size
    price=150.0,
)

# Calculation: position_size = max_percent * confidence
# Example: 20% * 0.85 = 17% of portfolio
```

**Portfolio Rebalancing:**

```python
# Rebalance to target weights
target_weights = {
    "AAPL": 0.30,  # 30% of portfolio
    "MSFT": 0.25,  # 25%
    "GOOGL": 0.20, # 20%
    # Remaining 25% in cash
}

current_prices = {
    "AAPL": 150.0,
    "MSFT": 320.0,
    "GOOGL": 2800.0,
}

self.rebalance_to_weights(target_weights, current_prices)
```

**P&L Tracking:**

```python
# Get unrealized P&L as percentage
pnl_pct = self.get_unrealized_pnl_pct("AAPL")

# Use for exit logic
if pnl_pct and pnl_pct >= 0.15:  # 15% profit
    self.close_position("AAPL")
elif pnl_pct and pnl_pct <= -0.05:  # 5% loss
    self.close_position("AAPL")
```

### Error Handling

All helpers raise `ValueError` if broker not initialized:

```python
try:
    position = self.get_position("AAPL")
except ValueError as e:
    # "Broker not initialized. Helper methods only available during backtest."
    pass
```

This is expected behavior - helpers only work inside `on_market_event()` during backtest execution.

---

## Advanced Patterns

### Confidence-Based Position Sizing

**Pattern:** Size positions proportional to ML confidence (similar to Kelly Criterion).

```python
class KellyStrategy(Strategy):
    def on_market_event(self, event, context=None):
        prediction = event.signals.get("prediction", 0.0)
        confidence = event.signals.get("confidence", 0.0)

        # Only trade high-conviction signals
        if prediction > 0.65 and confidence > 0.70:
            # Size by confidence: high confidence → larger position
            self.size_by_confidence(
                asset_id=event.asset_id,
                confidence=confidence,
                max_percent=0.25,  # Never exceed 25% of portfolio
                price=event.close,
            )
```

**Example sizing:**
- Confidence 0.70 → 25% × 0.70 = 17.5% position
- Confidence 0.85 → 25% × 0.85 = 21.25% position
- Confidence 0.95 → 25% × 0.95 = 23.75% position

### Multi-Signal Decision Logic

**Pattern:** Combine multiple ML models or signals for robust decisions.

```python
class EnsembleStrategy(Strategy):
    def on_market_event(self, event, context=None):
        # Multiple ML models
        model_a = event.signals.get("model_a_pred", 0.0)
        model_b = event.signals.get("model_b_pred", 0.0)
        model_c = event.signals.get("model_c_pred", 0.0)

        # Ensemble vote (majority)
        signals = [model_a, model_b, model_c]
        bullish_votes = sum(1 for s in signals if s > 0.5)

        # Require 2/3 agreement
        if bullish_votes >= 2:
            avg_confidence = sum(signals) / len(signals)
            self.size_by_confidence(
                asset_id=event.asset_id,
                confidence=avg_confidence,
                max_percent=0.15,
                price=event.close,
            )
```

### Regime-Dependent Trading

**Pattern:** Adjust strategy based on market regime (bull/bear/high-vol).

```python
class RegimeStrategy(Strategy):
    def on_market_event(self, event, context=None):
        # Get market regime from context
        vix = context.get("VIX", 0.0) if context else 0.0
        regime = context.get("regime", "unknown") if context else "unknown"

        prediction = event.signals.get("prediction", 0.0)

        # Risk-off during high volatility
        if vix > 30:
            self.close_position(event.asset_id)
            return

        # Aggressive in bull markets
        if regime == "bull" and prediction > 0.60:
            self.buy_percent(event.asset_id, percent=0.20, price=event.close)

        # Conservative in bear markets (higher threshold)
        elif regime == "bear" and prediction > 0.75:
            self.buy_percent(event.asset_id, percent=0.10, price=event.close)
```

### Dynamic Stop Loss / Take Profit

**Pattern:** Exit based on realized P&L percentage.

```python
class StopLossStrategy(Strategy):
    def __init__(self, profit_target=0.15, stop_loss=-0.05):
        super().__init__()
        self.profit_target = profit_target
        self.stop_loss = stop_loss

    def on_market_event(self, event, context=None):
        position = self.get_position(event.asset_id)

        # Check exit conditions if in position
        if position != 0:
            pnl_pct = self.get_unrealized_pnl_pct(event.asset_id)

            if pnl_pct and pnl_pct >= self.profit_target:
                self.close_position(event.asset_id)
                return  # Exit taken

            if pnl_pct and pnl_pct <= self.stop_loss:
                self.close_position(event.asset_id)
                return  # Stop loss hit

        # Entry logic
        prediction = event.signals.get("prediction", 0.0)
        if position == 0 and prediction > 0.70:
            self.buy_percent(event.asset_id, percent=0.12, price=event.close)
```

### Portfolio Rebalancing Strategy

**Pattern:** Maintain target weights across multiple assets.

```python
class RebalanceStrategy(Strategy):
    def __init__(self, rebalance_frequency_days=30):
        super().__init__()
        self.rebalance_frequency = rebalance_frequency_days
        self.last_rebalance = None

    def on_market_event(self, event, context=None):
        # Rebalance monthly
        if self._should_rebalance(event.timestamp):
            # Get ML predictions for all assets
            predictions = self._get_all_predictions()  # Your logic

            # Allocate based on signal strength
            target_weights = self._calculate_weights(predictions)

            # Get current prices
            current_prices = self._get_current_prices()  # Your logic

            # Rebalance
            self.rebalance_to_weights(target_weights, current_prices)
            self.last_rebalance = event.timestamp

    def _should_rebalance(self, timestamp):
        if not self.last_rebalance:
            return True
        days_since = (timestamp - self.last_rebalance).days
        return days_since >= self.rebalance_frequency
```

---

## Performance Characteristics

### ContextCache Memory Savings

**Benchmark Setup:** 252 trading days, varying asset counts and context richness.

**Minimal Context** (8 indicators: VIX, SPY, 6 regime flags):

| Assets | With ContextCache | Without (dict copy) | Savings |
|--------|-------------------|---------------------|---------|
| 10     | 1.03 MB           | 1.94 MB             | 1.9x    |
| 100    | 9.91 MB           | 19.30 MB            | 1.9x    |
| 500    | 49.31 MB          | 96.39 MB            | 2.0x    |

**Large Context** (50+ indicators: ML features, multiple indices, sector indicators):

| Assets | With ContextCache | Without (dict copy) | Savings |
|--------|-------------------|---------------------|---------|
| 10     | 1.03 MB           | 5.09 MB             | 5.0x    |
| 100    | 9.91 MB           | 50.83 MB            | 5.1x    |
| 500    | 49.31 MB          | 254.04 MB           | 5.2x    |

**Key Findings:**
- Memory savings scale with context richness (2x → 5x)
- Larger benefit for ML strategies with many features (50-100 indicators)
- Consistent efficiency across universe sizes
- **Recommendation:** Use ContextCache for multi-asset (100+ stocks) with rich ML context

**Source:** `tests/benchmarks/test_context_memory.py` (measured with `tracemalloc`)

### Event Processing Performance

**Backtest execution speed** (from ml_strategy_example.py):

```
Sample Backtest: 504 events (2 years daily)
- Events/Second: ~8,000-12,000
- Total Duration: ~0.05-0.08 seconds
```

Performance scales linearly with complexity:
- Simple strategies (buy/hold): 15,000+ events/sec
- Complex ML strategies (9 helper calls): 8,000-10,000 events/sec
- Multi-asset (100 stocks): 5,000-8,000 events/sec

---

## Complete Working Example

**Full ML strategy with all features:**

```python
"""Complete ML Strategy Example."""
import polars as pl
from datetime import datetime
from ml4t.backtest import BacktestEngine, Strategy
from ml4t.backtest.data import ParquetDataFeed

# 1. Generate ML predictions (normally from your trained model)
data = pl.DataFrame({
    "timestamp": pl.datetime_range(
        datetime(2023, 1, 1, 9, 30),
        datetime(2023, 12, 31, 16, 0),
        interval="1d",
        eager=True,
    )[:252],
    "open": [100 + i * 0.1 for i in range(252)],
    "high": [100.5 + i * 0.1 for i in range(252)],
    "low": [99.5 + i * 0.1 for i in range(252)],
    "close": [100 + i * 0.1 for i in range(252)],
    "volume": [1_000_000] * 252,

    # ML signals (normally from your model.predict())
    "prediction": [0.6 + (i % 10) * 0.04 for i in range(252)],
    "confidence": [0.7 + (i % 5) * 0.05 for i in range(252)],
})

data.write_parquet("ml_data.parquet")

# 2. Create context data
context_data = {
    ts: {
        "VIX": 15.0 + i % 20,
        "regime": "bull" if i < 150 else "bear",
    }
    for i, ts in enumerate(data["timestamp"])
}

# 3. Define ML strategy
class MyMLStrategy(Strategy):
    def on_market_event(self, event, context=None):
        # Extract signals
        prediction = event.signals.get("prediction", 0.0)
        confidence = event.signals.get("confidence", 0.0)

        # Extract context
        vix = context.get("VIX", 0.0) if context else 0.0

        # Get position
        position = self.get_position(event.asset_id)

        # Exit logic
        if position != 0:
            pnl_pct = self.get_unrealized_pnl_pct(event.asset_id)
            if pnl_pct and pnl_pct >= 0.15:  # 15% profit
                self.close_position(event.asset_id)
                return
            if pnl_pct and pnl_pct <= -0.05:  # 5% loss
                self.close_position(event.asset_id)
                return

        # Entry logic
        if position == 0 and vix < 30:
            if prediction > 0.65 and confidence > 0.75:
                self.size_by_confidence(
                    asset_id=event.asset_id,
                    confidence=confidence,
                    max_percent=0.20,
                    price=event.close,
                )

# 4. Create data feed
feed = ParquetDataFeed(
    path="ml_data.parquet",
    asset_id="AAPL",
    signal_columns=["prediction", "confidence"],
)

# 5. Run backtest
engine = BacktestEngine(
    data_feed=feed,
    strategy=MyMLStrategy(),
    context_data=context_data,
    initial_capital=100_000,
)

results = engine.run()
print(f"Return: {results['total_return']:.2%}")
print(f"Final Value: ${results['final_value']:,.2f}")
```

---

## Best Practices

### Signal Generation

**DO:**
- Pre-compute ML predictions offline (proper train/test split)
- Use point-in-time features (no future data)
- Validate predictions align with timestamps
- Include confidence/uncertainty estimates
- Save predictions in same DataFrame as OHLCV

**DON'T:**
- Compute predictions inside `on_market_event()` (slow, risks look-ahead)
- Use future data for predictions (data leakage)
- Forget to handle missing/NaN signals

### Context Usage

**DO:**
- Use for market-wide indicators (VIX, SPY, rates)
- Pre-compute context offline (same as signals)
- Provide fallback values (`context.get("VIX", 0.0)`)
- Use ContextCache for multi-asset strategies (100+ stocks)

**DON'T:**
- Duplicate asset-specific data (use signals instead)
- Mutate context dict (shared across assets)
- Use context for single-asset strategies (join into signals)

### Helper Methods

**DO:**
- Use helpers for clean, readable code
- Handle `None` returns from `get_unrealized_pnl_pct()`
- Check position before sizing (`get_position() == 0`)
- Set reasonable `max_percent` limits (10-25% typical)

**DON'T:**
- Call helpers outside `on_market_event()` (broker not available)
- Assume helpers never fail (catch `ValueError`)
- Over-leverage (respect position limits)

### Testing

**DO:**
- Use test fixtures (`tests/fixtures/ml_signal_data.py`)
- Test edge cases (high VIX, low confidence, zero position)
- Validate P&L calculations
- Benchmark memory usage for large universes

**DON'T:**
- Skip testing with realistic data
- Ignore memory usage (track with tracemalloc)
- Assume signals always present (use `.get()` with defaults)

### Performance

**DO:**
- Pre-compute all signals and context offline
- Use ContextCache for large universes (100+ assets)
- Profile memory usage (`tests/benchmarks/test_context_memory.py`)
- Optimize hot paths in strategy logic

**DON'T:**
- Compute features inside strategy (bottleneck)
- Copy context dicts unnecessarily (use ContextCache)
- Run heavy ML inference in `on_market_event()`

---

## Next Steps

**1. Try the example:**
```bash
cd /home/stefan/ml4t/software/backtest
python examples/ml_strategy_example.py
```

**2. Use test fixtures:**
```python
# In your tests
from tests.fixtures import ml_signal_data, context_data

def test_my_strategy(ml_signal_data, context_data):
    # Use pre-built realistic data
    pass
```

**3. Read fixture documentation:**
- `tests/fixtures/README.md` - Comprehensive fixture guide
- `tests/fixtures/USAGE_EXAMPLE.md` - Quick examples

**4. Review advanced tests:**
- `tests/unit/test_strategy_helpers.py` - Helper method usage
- `tests/benchmarks/test_context_memory.py` - Performance validation

---

**Questions or issues?** See main documentation in `/home/stefan/ml4t/software/backtest/README.md` or raise an issue on GitHub.
