# ML4T Backtest

Event-driven backtesting engine with configurable execution semantics, validated against four independent frameworks.

Use `ml4t-backtest` when notebook research is no longer enough and you need explicit,
reproducible answers to practical execution questions: when orders fill, how stops trigger,
how cash is reserved, and how results change when you match another framework's behavior.

<div class="grid cards" markdown>

-   :material-play-circle:{ .lg .middle } __Run Your First Backtest__
    ---
    Define a strategy, pick a config profile, get results in 10 lines.
    [:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md)

-   :material-tune:{ .lg .middle } __40+ Configurable Knobs__
    ---
    Fill ordering, stop modes, cash policy, settlement, and account behavior
    are explicit named parameters.
    [:octicons-arrow-right-24: Configuration](user-guide/configuration.md)

-   :material-check-all:{ .lg .middle } __Validated Against 4 Frameworks__
    ---
    Compare fills and terminal values against VectorBT, Backtrader, Zipline,
    and LEAN on the same benchmark scenarios.
    [:octicons-arrow-right-24: Profiles](user-guide/profiles.md)

-   :material-book-open-variant:{ .lg .middle } __Chapters 16-19__
    ---
    The book develops the ideas in notebooks. This library turns them into
    reusable execution and reporting workflows.
    [:octicons-arrow-right-24: Book Guide](book-guide/index.md)

</div>

## Overview

`ml4t-backtest` is the simulation layer in the ML4T stack. It sits between research and
deployment:

- `ml4t-data` prepares canonical market datasets
- `ml4t-engineer` produces labels and features
- `ml4t-diagnostic` validates signals, models, and portfolio behavior
- `ml4t-backtest` simulates execution with explicit, configurable semantics
- `ml4t-live` reuses the same strategy surface for paper and live rollout

## Quick Example

```python
import polars as pl
from ml4t.backtest import Engine, DataFeed, Strategy, BacktestConfig

class BuyAndHold(Strategy):
    def on_data(self, timestamp, data, context, broker):
        for asset, bar in data.items():
            if broker.get_position(asset) is None:
                broker.submit_order(asset, 100)

feed = DataFeed(prices_df=prices)
engine = Engine(feed=feed, strategy=BuyAndHold())
result = engine.run()

print(f"Total Return: {result.metrics['total_return_pct']:.1f}%")
print(f"Sharpe Ratio: {result.metrics['sharpe']:.2f}")
```

Or use the convenience function:

```python
from ml4t.backtest import run_backtest

result = run_backtest(prices, BuyAndHold(), config="backtrader")
```

## Why ML4T Backtest?

**Configurable execution semantics.** Every behavioral difference between backtesting frameworks (fill ordering, stop modes, cash policies, settlement) is a named config parameter. Switch profiles to replicate any framework exactly.

**Quote-aware when you need it.** The feed can cache bid, ask, midpoint, and quote sizes additively. Market execution and position marking can use `price`, `bid`, `ask`, `quote_mid`, or `quote_side`.

**Validated at scale.** 225,000+ trades verified trade-by-trade against VectorBT Pro, Backtrader, Zipline, and LEAN on real market data (250 assets x 20 years).

| Feature | Description |
|---------|-------------|
| Event-driven | Point-in-time correctness, no look-ahead bias |
| 40+ behavioral knobs | Every execution detail is configurable |
| Quote-aware execution | Side-aware fills and separate mark pricing |
| 10 framework profiles | Match VectorBT, Backtrader, Zipline, LEAN exactly |
| Risk management | Stop-loss, take-profit, trailing stops, portfolio limits |
| Multi-asset | Rebalancing, weight targets, exit-first ordering |
| Rich persistence | Export trades, fills, equity, portfolio state, and daily P&L to Parquet |

## Parity Validation

Each comparison runs the same benchmark scenario on real OHLCV data and checks fill
counts, trade gaps, and terminal portfolio value on the matched execution surface.

| Profile | Trades Compared | Trade Gap | Value Gap |
|---------|----------------|-----------|-----------|
| `zipline_strict` | 225,583 | 0 (0.00%) | $19 (0.0001%) |
| `backtrader_strict` | 216,980 | 1 (0.0005%) | $503 (0.004%) |
| `vectorbt_strict` | 210,352 | 91 (0.04%) | $0 (0.00%) |
| `lean` | 428,459 fills | 0 (0.00%) | $1.55 (0.0002%) |

On the same workloads, `ml4t-backtest` processes 40,000+ bars/second and runs about
19x faster than Backtrader, 8x faster than Zipline, and 5x faster than LEAN.

## Installation

```bash
pip install ml4t-backtest
```

## Next Steps

- **New here?** Start with the [Quickstart](getting-started/quickstart.md)
- **Coming from the book?** Use the [Book Guide](book-guide/index.md)
- **Debugging execution behavior?** Read [Execution Semantics](user-guide/execution-semantics.md)
- **Need exact interfaces?** See the [API Reference](api/index.md)

## From Book to Library

If you are reading *Machine Learning for Trading, Third Edition*, use the docs in this order:

1. learn the execution or reporting concept in the notebook
2. use the [Book Guide](book-guide/index.md) to find the matching production workflow
3. move to the relevant user-guide page for the reusable API
4. finish in the [API Reference](api/index.md) for exact call signatures

This is especially important for quote-aware execution, realistic reporting, rebalancing,
and strategy portability into `ml4t-live`.

## Part of the ML4T Ecosystem

```
ml4t-data --> ml4t-engineer --> ml4t-diagnostic --> ml4t-backtest --> ml4t-live
```

The same `Strategy` class works in both backtest and live trading via `ml4t-live`.
