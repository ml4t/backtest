# Data Feed

`DataFeed` converts a Polars DataFrame into per-bar data for the engine. It handles partitioning by timestamp, multi-asset iteration, optional signals/context data, and additive quote caches for execution-aware workloads.

## Required Columns

The prices DataFrame must always include:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | Datetime | Bar timestamp |
| `asset` | String | Asset identifier |

Standard OHLCV feeds usually provide:

| Column | Type | Description |
|--------|------|-------------|
| `open` | Float | Opening price |
| `high` | Float | High price |
| `low` | Float | Low price |
| `close` | Float | Closing price |
| `volume` | Float | Trading volume |

`DataFeed` also exposes a normalized `bar["price"]` field. By default it follows `close`, but if your `FeedSpec` or constructor sets `price_col`, that column becomes the broker reference price.

Optional quote columns are carried through when present:

| Column | Description |
|--------|-------------|
| `bid_col` | Best bid price |
| `ask_col` | Best ask price |
| `mid_col` | Explicit midpoint if your data provides one |
| `bid_size_col` | Bid-side available size |
| `ask_size_col` | Ask-side available size |

## Basic Usage

```python
import polars as pl
from ml4t.backtest import DataFeed

prices = pl.DataFrame({
    "timestamp": [...],
    "asset": [...],
    "open": [...],
    "high": [...],
    "low": [...],
    "close": [...],
    "volume": [...],
})

feed = DataFeed(prices_df=prices)
```

Inside `on_data()`, each asset bar contains `price`, `open`, `high`, `low`, `close`, `volume`, plus any available quote fields and `signals`.

## FeedSpec and Column Overrides

Use `FeedSpec` or explicit keyword overrides when your schema differs from OHLCV defaults:

```python
from ml4t.backtest import DataFeed
from ml4t.specs import FeedSpec

feed = DataFeed(
    prices_df=quotes,
    feed_spec=FeedSpec(
        timestamp_col="ts",
        entity_col="symbol",
        price_col="mid_price",
        close_col="last_trade",
        bid_col="bid",
        ask_col="ask",
        bid_size_col="bid_size",
        ask_size_col="ask_size",
    ),
)
```

Constructor keyword arguments override `FeedSpec` fields, so you can keep a shared spec and specialize it for a single backtest.

## Multi-Asset Data

Stack all assets in a single DataFrame. The engine handles partitioning by timestamp automatically:

```python
# Two assets, same timestamps
prices = pl.DataFrame({
    "timestamp": [t1, t1, t2, t2, t3, t3],
    "asset":     ["AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT"],
    "open":      [150.0, 280.0, 151.0, 281.0, 152.0, 282.0],
    "high":      [152.0, 282.0, 153.0, 283.0, 154.0, 284.0],
    "low":       [149.0, 279.0, 150.0, 280.0, 151.0, 281.0],
    "close":     [151.0, 281.0, 152.0, 282.0, 153.0, 283.0],
    "volume":    [1e6, 2e6, 1e6, 2e6, 1e6, 2e6],
})
```

## Signals

Pass pre-computed signals (ML predictions, indicators, etc.) as a separate DataFrame:

```python
signals = pl.DataFrame({
    "timestamp": [...],
    "asset":     [...],
    "prediction": [...],
    "momentum":   [...],
})

feed = DataFeed(prices_df=prices, signals_df=signals)
```

Signals appear in `on_data` under the `"signals"` key:

```python
def on_data(self, timestamp, data, context, broker):
    for asset, bar in data.items():
        pred = bar.get("signals", {}).get("prediction", 0)
```

Any column in the signals DataFrame (other than `timestamp` and `asset`) becomes a signal.

## Quote-Aware Execution Inputs

Quote columns are additive: you can keep OHLCV behavior unchanged, or opt into quote-aware execution in config:

```python
from ml4t.backtest import BacktestConfig
from ml4t.backtest.config import ExecutionPrice

config = BacktestConfig(
    execution_price=ExecutionPrice.QUOTE_SIDE,
    mark_price=ExecutionPrice.QUOTE_SIDE,
)
```

When quotes are present:

- `ExecutionPrice.PRICE` uses `FeedSpec.price_col`
- `ExecutionPrice.BID` and `ExecutionPrice.ASK` use the best quote on that side
- `ExecutionPrice.QUOTE_MID` uses the explicit midpoint or derives `(bid + ask) / 2`
- `ExecutionPrice.QUOTE_SIDE` buys at ask and sells at bid

If a quote field is missing, the broker falls back to the reference price or OHLC value for the configured source.

Those quote inputs also flow into the reporting layer:

- `result.to_fills_dataframe()` preserves fill-level quote context
- `result.to_trades_dataframe()` preserves entry/exit quote summaries
- `result.to_portfolio_state_dataframe()` reflects the configured mark source

## Context Data

Context provides per-bar metadata that isn't tied to individual assets:

```python
context = pl.DataFrame({
    "timestamp": [...],
    "vix":       [...],
    "regime":    [...],
})

feed = DataFeed(prices_df=prices, context_df=context)
```

Context is passed as the third argument to `on_data`:

```python
def on_data(self, timestamp, data, context, broker):
    vix = context.get("vix", 0)
    if vix > 30:
        return  # Don't trade in high-vol regimes
```

## Loading from Files

DataFeed accepts Parquet file paths:

```python
feed = DataFeed(
    prices_path="data/prices.parquet",
    signals_path="data/signals.parquet",
    context_path="data/context.parquet",
)
```

Or mix paths and DataFrames:

```python
feed = DataFeed(
    prices_df=prices,
    signals_path="data/signals.parquet",
)
```

## Using with run_backtest

The convenience function handles DataFeed creation:

```python
from ml4t.backtest import run_backtest

# DataFrames
result = run_backtest(prices, strategy, signals=signals_df)

# File paths
result = run_backtest("data/prices.parquet", strategy, signals="data/signals.parquet")
```

## Performance

DataFeed pre-partitions data by timestamp at initialization and pre-extracts column indices for O(1) per-bar access. For 1M bars, this uses roughly 100 MB (10x less than converting everything to Python dicts upfront). Quote columns are cached additively, so the legacy OHLCV path stays unchanged unless you provide quote data.

## See It in Action

The [Machine Learning for Trading](https://github.com/stefan-jansen/machine-learning-for-trading) book prepares DataFeed inputs in every Engine case study:

- **Ch16 case studies** — each case study loads OHLCV from Parquet, constructs a signals DataFrame from ML predictions, and passes both to DataFeed
- **Ch16 / NB13** (`futures_backtesting`) — multi-contract futures data with session boundaries and overnight gaps
- The common pattern: `prices_df` is a stacked multi-asset OHLCV DataFrame, `signals_df` contains prediction columns aligned by (timestamp, asset)

## Next Steps

- [Book Guide](../book-guide/index.md) -- chapter and case-study map for data preparation patterns
- [Quickstart](../getting-started/quickstart.md) -- end-to-end examples
- [Strategies](strategies.md) -- how to use data in strategy callbacks
- [Rebalancing](rebalancing.md) -- multi-asset weight-based strategies
