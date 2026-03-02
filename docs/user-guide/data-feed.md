# Data Feed

`DataFeed` converts a Polars DataFrame into per-bar data for the engine. It handles partitioning by timestamp, multi-asset iteration, and optional signals/context data.

## Required Columns

The prices DataFrame must have these columns:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | Datetime | Bar timestamp |
| `asset` | String | Asset identifier |
| `open` | Float | Opening price |
| `high` | Float | High price |
| `low` | Float | Low price |
| `close` | Float | Closing price |
| `volume` | Float | Trading volume |

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

DataFeed pre-partitions data by timestamp at initialization and pre-extracts column indices for O(1) per-bar access. For 1M bars, this uses roughly 100 MB (10x less than converting everything to Python dicts upfront).

## Next Steps

- [Quickstart](../getting-started/quickstart.md) -- end-to-end examples
- [Strategies](strategies.md) -- how to use data in strategy callbacks
- [Rebalancing](rebalancing.md) -- multi-asset weight-based strategies
