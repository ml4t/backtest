# Migration Guide to ml4t.backtest

## Philosophy

ml4t.backtest takes a fundamentally different approach to backtesting:

**Separation of Concerns**: Unlike Zipline and Backtrader which compute indicators on-the-fly during the backtest, ml4t.backtest separates feature engineering from execution. You pre-compute your signals/indicators once, then the backtest is pure execution logic.

This provides:
- **Speed**: Compute expensive indicators once, not on every backtest iteration
- **Reproducibility**: Same signals produce same results
- **ML Integration**: Natural fit for ML models that produce signals offline
- **Simpler Strategies**: No indicator boilerplate in your trading logic

## Quick Comparison

| Feature | Zipline | Backtrader | ml4t.backtest |
|---------|---------|------------|---------------|
| Strategy Method | `handle_data()` | `next()` | `on_data()` |
| Indicators | Computed on-the-fly | Computed on-the-fly | **Pre-computed** |
| Data Access | `data.current()` | `self.data[0]` | `data[asset]['close']` |
| Signals Access | Custom columns | Custom lines | `data[asset]['signals']` |
| Order Submission | `order_target_percent()` | `self.buy()` | `broker.submit_order()` |
| Performance | Moderate | Slow | Fast (Polars-based) |

## The ml4t.backtest Workflow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   prices_df     │     │   signals_df    │     │   context_df    │
│ (OHLCV data)    │     │ (indicators,    │     │ (regime, VIX,   │
│                 │     │  ML signals)    │     │  market state)  │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────┬───────┴───────────────────────┘
                         ▼
              ┌─────────────────────┐
              │      DataFeed       │
              │ (merges all data)   │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │       Engine        │
              │  (event loop)       │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │      Strategy       │
              │ on_data(ts, data,   │
              │   context, broker)  │
              └─────────────────────┘
```

---

## Migrating from Zipline

### Example 1: Single-Asset Moving Average Crossover

**Zipline (on-the-fly indicators):**
```python
from zipline.api import symbol, order_target_percent

def initialize(context):
    context.asset = symbol('AAPL')
    context.short_period = 50
    context.long_period = 200

def handle_data(context, data):
    # Compute indicators on every bar (slow!)
    short_mavg = data.history(
        context.asset, 'price', context.short_period, '1d'
    ).mean()
    long_mavg = data.history(
        context.asset, 'price', context.long_period, '1d'
    ).mean()

    if short_mavg > long_mavg:
        order_target_percent(context.asset, 1.0)
    else:
        order_target_percent(context.asset, 0.0)
```

**ml4t.backtest (pre-computed signals):**

Step 1: Prepare your data with pre-computed indicators
```python
import polars as pl

# Load price data
prices_df = pl.read_parquet('aapl_prices.parquet')
# Columns: timestamp, asset, open, high, low, close, volume

# Compute signals ONCE (use Polars, pandas, ta-lib, whatever you prefer)
signals_df = prices_df.select([
    'timestamp', 'asset',
    pl.col('close').rolling_mean(50).over('asset').alias('sma_50'),
    pl.col('close').rolling_mean(200).over('asset').alias('sma_200'),
])

# Add the crossover signal
signals_df = signals_df.with_columns([
    (pl.col('sma_50') > pl.col('sma_200')).cast(pl.Int8).alias('long_signal')
])
```

Step 2: Define your strategy (pure execution logic)
```python
from ml4t.backtest import Strategy, DataFeed, Engine

class MovingAverageCrossover(Strategy):
    def on_data(self, timestamp, data, context, broker):
        for asset, bar in data.items():
            signal = bar['signals'].get('long_signal', 0)
            pos = broker.get_position(asset)

            if signal == 1 and pos is None:
                # Enter long
                cash = broker.get_cash()
                price = bar['close']
                qty = int(cash * 0.95 / price)  # 95% allocation
                if qty > 0:
                    broker.submit_order(asset, qty)

            elif signal == 0 and pos is not None:
                # Exit
                broker.close_position(asset)

# Run backtest
feed = DataFeed(prices_df=prices_df, signals_df=signals_df)
strategy = MovingAverageCrossover()
engine = Engine(feed, strategy, initial_cash=100000)
result = engine.run()
```

### Example 2: Multi-Asset Universe

**Zipline (Pipeline):**
```python
from zipline.api import attach_pipeline, pipeline_output, order_target_percent
from zipline.pipeline import Pipeline
from zipline.pipeline.factors import SimpleMovingAverage

def make_pipeline():
    sma_10 = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=10)
    sma_30 = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=30)
    return Pipeline(columns={'sma_10': sma_10, 'sma_30': sma_30})

def initialize(context):
    attach_pipeline(make_pipeline(), 'my_pipeline')

def before_trading_start(context, data):
    context.output = pipeline_output('my_pipeline')

def handle_data(context, data):
    for asset in context.output.index:
        sma_10 = context.output.loc[asset, 'sma_10']
        sma_30 = context.output.loc[asset, 'sma_30']
        if sma_10 > sma_30:
            order_target_percent(asset, 0.1)  # 10% each
        else:
            order_target_percent(asset, 0.0)
```

**ml4t.backtest (multi-asset):**

Step 1: Prepare multi-asset data
```python
import polars as pl

# prices_df has multiple assets
# Columns: timestamp, asset, open, high, low, close, volume
prices_df = pl.read_parquet('universe_prices.parquet')

# Compute signals per asset using group_by
signals_df = prices_df.group_by('asset').agg([
    pl.col('timestamp'),
    pl.col('close').rolling_mean(10).alias('sma_10'),
    pl.col('close').rolling_mean(30).alias('sma_30'),
]).explode(['timestamp', 'sma_10', 'sma_30'])

signals_df = signals_df.with_columns([
    (pl.col('sma_10') > pl.col('sma_30')).cast(pl.Int8).alias('long_signal')
])
```

Step 2: Multi-asset strategy
```python
class MultiAssetMomentum(Strategy):
    def __init__(self, max_positions: int = 10, position_size: float = 0.1):
        self.max_positions = max_positions
        self.position_size = position_size

    def on_data(self, timestamp, data, context, broker):
        # Collect signals
        longs = []
        for asset, bar in data.items():
            signal = bar['signals'].get('long_signal', 0)
            if signal == 1:
                longs.append(asset)

        # Exit positions no longer in signal
        for asset in list(broker.positions.keys()):
            if asset not in longs:
                broker.close_position(asset)

        # Enter new positions (up to max)
        current_positions = len(broker.positions)
        account_value = broker.get_account_value()

        for asset in longs[:self.max_positions]:
            if broker.get_position(asset) is None:
                if current_positions < self.max_positions:
                    price = data[asset]['close']
                    allocation = account_value * self.position_size
                    qty = int(allocation / price)
                    if qty > 0:
                        broker.submit_order(asset, qty)
                        current_positions += 1

# Run
feed = DataFeed(prices_df=prices_df, signals_df=signals_df)
strategy = MultiAssetMomentum(max_positions=10, position_size=0.1)
engine = Engine(feed, strategy, initial_cash=1000000)
result = engine.run()
```

### Key Zipline Migration Notes

1. **No `context` object** - Use class attributes instead
2. **No `data.history()`** - Pre-compute and pass via `signals_df`
3. **No Pipeline** - Use Polars/pandas transformations beforehand
4. **Explicit order quantities** - `submit_order(asset, qty)` instead of `order_target_percent()`
5. **Position sizing** - Calculate in strategy or pre-compute in signals

---

## Migrating from Backtrader

### Example 1: Single-Asset with Stop-Loss

**Backtrader (on-the-fly indicators):**
```python
import backtrader as bt

class SmaCross(bt.Strategy):
    params = (
        ('sma_period', 20),
        ('stop_pct', 0.02),  # 2% stop-loss
    )

    def __init__(self):
        self.sma = bt.indicators.SMA(period=self.params.sma_period)
        self.order = None
        self.entry_price = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.data.close[0] > self.sma[0]:
                self.order = self.buy()
                self.entry_price = self.data.close[0]
        else:
            # Check stop-loss
            stop_price = self.entry_price * (1 - self.params.stop_pct)
            if self.data.close[0] < stop_price:
                self.order = self.close()
            elif self.data.close[0] < self.sma[0]:
                self.order = self.close()

cerebro = bt.Cerebro()
cerebro.addstrategy(SmaCross)
cerebro.adddata(bt.feeds.PandasData(dataname=df))
cerebro.run()
```

**ml4t.backtest:**

Step 1: Pre-compute signals
```python
import polars as pl

prices_df = pl.read_parquet('prices.parquet')

signals_df = prices_df.select([
    'timestamp', 'asset',
    pl.col('close').rolling_mean(20).alias('sma_20'),
]).with_columns([
    (pl.col('close') > pl.col('sma_20')).cast(pl.Int8).alias('above_sma')
])
```

Step 2: Strategy with stop-loss
```python
from ml4t.backtest import Strategy, DataFeed, Engine

class SmaWithStop(Strategy):
    def __init__(self, stop_pct: float = 0.02):
        self.stop_pct = stop_pct
        self.entry_prices = {}  # Track entry prices

    def on_data(self, timestamp, data, context, broker):
        for asset, bar in data.items():
            pos = broker.get_position(asset)
            price = bar['close']
            above_sma = bar['signals'].get('above_sma', 0)

            if pos is None:
                # Entry logic
                if above_sma == 1:
                    cash = broker.get_cash()
                    qty = int(cash * 0.95 / price)
                    if qty > 0:
                        broker.submit_order(asset, qty)
                        self.entry_prices[asset] = price
            else:
                # Exit logic
                entry_price = self.entry_prices.get(asset, pos.entry_price)
                stop_price = entry_price * (1 - self.stop_pct)

                if price < stop_price or above_sma == 0:
                    broker.close_position(asset)
                    self.entry_prices.pop(asset, None)

feed = DataFeed(prices_df=prices_df, signals_df=signals_df)
engine = Engine(feed, SmaWithStop(stop_pct=0.02), initial_cash=100000)
result = engine.run()
```

### Example 2: Multi-Asset Portfolio

**Backtrader (multiple data feeds):**
```python
class MultiAsset(bt.Strategy):
    def __init__(self):
        self.sma = {}
        for d in self.datas:
            self.sma[d._name] = bt.indicators.SMA(d, period=20)

    def next(self):
        for d in self.datas:
            if d.close[0] > self.sma[d._name][0]:
                if not self.getposition(d):
                    self.buy(data=d, size=100)
            else:
                if self.getposition(d):
                    self.close(data=d)

cerebro = bt.Cerebro()
for ticker in ['AAPL', 'GOOGL', 'MSFT']:
    data = bt.feeds.PandasData(dataname=dfs[ticker], name=ticker)
    cerebro.adddata(data)
cerebro.addstrategy(MultiAsset)
```

**ml4t.backtest (single multi-asset DataFrame):**
```python
# All assets in one DataFrame - much simpler!
prices_df = pl.concat([
    aapl_df.with_columns(pl.lit('AAPL').alias('asset')),
    googl_df.with_columns(pl.lit('GOOGL').alias('asset')),
    msft_df.with_columns(pl.lit('MSFT').alias('asset')),
])

signals_df = prices_df.group_by('asset').agg([
    pl.col('timestamp'),
    (pl.col('close') > pl.col('close').rolling_mean(20)).cast(pl.Int8).alias('above_sma'),
]).explode(['timestamp', 'above_sma'])

class SimpleMultiAsset(Strategy):
    def on_data(self, timestamp, data, context, broker):
        for asset, bar in data.items():
            pos = broker.get_position(asset)
            above_sma = bar['signals'].get('above_sma', 0)

            if above_sma == 1 and pos is None:
                broker.submit_order(asset, 100)
            elif above_sma == 0 and pos is not None:
                broker.close_position(asset)

feed = DataFeed(prices_df=prices_df, signals_df=signals_df)
engine = Engine(feed, SimpleMultiAsset(), initial_cash=100000)
result = engine.run()
```

### Key Backtrader Migration Notes

1. **No `self.data[0]` indexing** - Data is passed to `on_data()` directly
2. **No indicator classes** - Pre-compute everything
3. **No `cerebro`** - Use `Engine` directly
4. **No `params` tuple** - Use `__init__` parameters
5. **Single DataFrame** - All assets in one DataFrame with `asset` column
6. **Position access** - `broker.get_position(asset)` instead of `self.getposition(data)`

---

## Pre-Computing Signals: Best Practices

### Using Polars (Recommended)
```python
import polars as pl

# Fast, memory-efficient
signals_df = prices_df.with_columns([
    pl.col('close').rolling_mean(20).over('asset').alias('sma_20'),
    pl.col('close').rolling_std(20).over('asset').alias('std_20'),
    pl.col('close').pct_change().over('asset').alias('returns'),
])
```

### Using pandas with TA-Lib
```python
import pandas as pd
import talib

# For complex indicators
df = prices_df.to_pandas()
df['rsi'] = df.groupby('asset')['close'].transform(lambda x: talib.RSI(x, 14))
df['macd'], df['macd_signal'], _ = talib.MACD(df['close'])
signals_df = pl.from_pandas(df[['timestamp', 'asset', 'rsi', 'macd', 'macd_signal']])
```

### Using ML Models
```python
import polars as pl
from sklearn.ensemble import RandomForestClassifier

# Train model offline
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Generate predictions
predictions = model.predict_proba(X_test)[:, 1]

# Add to signals DataFrame
signals_df = prices_df.select(['timestamp', 'asset']).with_columns([
    pl.Series('ml_signal', predictions),
    pl.Series('confidence', predictions),  # Use probability as confidence
])
```

---

## Common Patterns

### Order Types

```python
# Market order
broker.submit_order('AAPL', 100)  # Buy 100 shares

# Limit order
broker.submit_order('AAPL', 100, order_type=OrderType.LIMIT, limit_price=150.0)

# Stop order
broker.submit_order('AAPL', -100, order_type=OrderType.STOP, stop_price=145.0)

# Bracket order (entry + take-profit + stop-loss)
entry, tp, sl = broker.submit_bracket(
    'AAPL', 100,
    take_profit=160.0,
    stop_loss=145.0
)
```

### Position Information

```python
# Get position
pos = broker.get_position('AAPL')
if pos:
    print(f"Quantity: {pos.quantity}")
    print(f"Entry price: {pos.entry_price}")
    print(f"Bars held: {pos.bars_held}")
    print(f"Unrealized P&L: {pos.unrealized_pnl(current_price)}")

# Get all positions
for asset, pos in broker.positions.items():
    print(f"{asset}: {pos.quantity} @ {pos.entry_price}")

# Portfolio value
cash = broker.get_cash()
total_value = broker.get_account_value()
```

### Risk Management

```python
from ml4t.backtest.risk import StopLoss, TakeProfit, TimeExit, RuleChain

# Define rules
rules = RuleChain([
    StopLoss(pct=0.02),           # 2% stop-loss
    TakeProfit(pct=0.05),         # 5% take-profit
    TimeExit(max_bars=20),        # Exit after 20 bars
])

# Apply to broker
broker.set_position_rules(rules)
# Or per-asset
broker.set_position_rules(rules, asset='AAPL')
```

---

## Why ml4t.backtest?

### Performance
- 10-100x faster than pure Python frameworks
- Polars for efficient data handling
- Pre-computed signals eliminate redundant computation

### Correctness
- No look-ahead bias by design
- Point-in-time correct data access
- Validated against VectorBT, Backtrader

### Simplicity
- Clean separation: data prep → backtest → analysis
- No magic or metaclasses
- Standard Python patterns

### ML-First
- Natural fit for ML signals
- Confidence scores in signals
- Easy integration with sklearn, PyTorch, etc.

### Live Trading Ready
- Same Strategy interface works for live trading
- Broker abstraction matches real brokers
- Easy transition from backtest to production
