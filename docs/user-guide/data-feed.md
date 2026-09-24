# Data Feed

The [multi-asset rebalancing tutorial](../tutorials/multiasset-rebalancing.md) runs equities, ETFs, futures, and FX from bundled inputs.

`DataFeed` converts a Polars DataFrame into per-bar data for the engine. It handles partitioning by timestamp, multi-asset iteration, optional signals/context data, and additive quote caches for execution-aware workloads.

Schema snippets using `[...]` stand for user-supplied columns; they are not standalone datasets. Run the [bundled data examples](../tutorials/data.md) for complete panels and checked output.

## Required Columns

With the default `FeedSpec`, the prices DataFrame must include:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | Datetime | Bar timestamp |
| `asset` | String | Asset identifier |
| `close` | Numeric | Reference and closing price |

The entity column may instead be `symbol`, `product`, or `ticker`, or an explicitly
configured name. Standard OHLCV feeds also provide:

| Column | Type | Description |
|--------|------|-------------|
| `open` | Float | Opening price |
| `high` | Float | High price |
| `low` | Float | Low price |
| `close` | Float | Closing price |
| `volume` | Float | Trading volume |

Open, high, low, and volume are optional. Missing OHLC values fall back to the
configured close; missing volume becomes zero. These fallbacks permit close-only
feeds, but they do not reconstruct intrabar paths or liquidity.

`DataFeed` also exposes a normalized `bar["price"]` field. By default it follows
`close`, but if your `FeedSpec` or constructor sets `price_col`, that column becomes
the broker reference price. If that name is absent but the configured close exists,
the close becomes the resolved reference price. Construction raises `ValueError`
when neither exists. For a one-price derived-bar schema, map both `price_col` and
`close_col` to that column so the contract is explicit.

For each source, pass either its Parquet path or its DataFrame, not both. Conflicting
sources raise `ValueError` instead of applying implicit precedence.

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

## Daily decisions across different close times

By default, the feed emits one event per exact timestamp. If ES closes at 21:00 UTC
and ZC closes at 21:15 UTC, the strategy sees two callbacks, each with one asset.
The first callback has no ZC bar. A cross-sectional rebalance built from that
partial mapping can submit an unintended close for a held asset.

Add a Polars `Date` session column to each price row and pass its name to
`DataFeed` to request one decision per completed session:

```python
from datetime import date, datetime
import polars as pl
from ml4t.backtest import BacktestConfig, DataFeed, Engine
from ml4t.backtest.types import ExecutionMode

prices = pl.DataFrame({
    "timestamp": [datetime(2024, 1, 2, 21), datetime(2024, 1, 2, 21, 15)],
    "session_date": [date(2024, 1, 2), date(2024, 1, 2)],
    "asset": ["ES", "ZC"],
    "open": [100.0, 100.0],
    "close": [100.0, 100.0],
})
feed = DataFeed(prices_df=prices, session_col="session_date")
engine = Engine(feed, strategy, BacktestConfig(execution_mode=ExecutionMode.NEXT_BAR))
```

The engine registers and marks each real bar at its own timestamp. At the final
bar of a session, it calls `on_data` once with both assets and the timestamp of
that final bar. Each asset's `signals` come from its own bar. Context values from
earlier events in the session remain available, with later values taking
precedence for duplicate keys. Market orders from this decision can fill only
on a later bar for that asset. Market-on-close orders also wait for a later
matching close. Same-bar execution is rejected for session decisions.

A decision session must contain exactly one price bar for every asset in the
feed's asset set. Missing or duplicate asset bars, mixed session dates at one
timestamp, and signal-only or context-only timestamps raise before the run.
A holiday with no rows produces no callback; a partial holiday session raises
rather than silently rebalancing an incomplete portfolio. Supply exchange-local
session dates explicitly when bars cross midnight or daylight-saving boundaries.
The timestamp remains the actual bar close in its original timezone.

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

The feed iterates the sorted union of price, signal, and context timestamps. A
signal-only or context-only timestamp therefore invokes the strategy with an empty
`data` mapping. Signal rows require the resolved timestamp and entity columns.

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

Context is timestamp-level data. When several rows share a context timestamp, the
first row is used.

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

DataFeed pre-partitions data by timestamp at initialization and pre-extracts column indices for O(1) per-bar access. Quote columns are cached additively, so the OHLCV path stays unchanged unless you provide quote data. The release benchmark records setup separately from engine runtime and measures memory over the complete child process.

## In the book

Chapter 16, Section 16.3, [Futures backtesting](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/16_strategy_simulation/02_futures_backtesting.ipynb) works through contract and session inputs. The [FX pairs backtest](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/case_studies/fx_pairs/13_backtest.ipynb) extends feed alignment to a prediction stream.

## Next Steps

- [Book Guide](../book-guide/index.md) -- chapter and case-study map for data preparation patterns
- [Quickstart](../getting-started/quickstart.md) -- end-to-end examples
- [Strategies](strategies.md) -- how to use data in strategy callbacks
- [Rebalancing](rebalancing.md) -- multi-asset weight-based strategies
