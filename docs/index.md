# ML4T Backtest

Event-driven backtesting engine with configurable execution semantics, validated against four independent frameworks.

## Overview

`ml4t-backtest` is the simulation layer in the ML4T stack. It sits between research and
deployment:

- `ml4t-data` prepares canonical market datasets
- `ml4t-engineer` produces labels and features
- `ml4t-diagnostic` validates signals, models, and portfolio behavior
- `ml4t-backtest` simulates execution with explicit, configurable semantics
- `ml4t-live` reuses the same strategy surface for paper and live rollout

The docs are structured as one integrated site:

1. Getting Started for installation and first runs
2. User Guide for execution semantics and workflows
3. API Reference for exact public interfaces
4. [Book Guide](book-guide/index.md) for chapter, notebook, and case-study mapping

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

**Fast.** 19x faster than Backtrader, 8x faster than Zipline, 5x faster than LEAN on identical workloads. Processes 40,000+ bars/second across 250 assets.

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

| Profile | Trades Compared | Trade Gap | Value Gap |
|---------|----------------|-----------|-----------|
| `zipline_strict` | 225,583 | 0 (0.00%) | $19 (0.0001%) |
| `backtrader_strict` | 216,980 | 1 (0.0005%) | $503 (0.004%) |
| `vectorbt_strict` | 210,352 | 91 (0.04%) | $0 (0.00%) |
| `lean_strict` | 226,172 | 589 (0.26%) | $7.2K (0.66%) |

## Installation

```bash
pip install ml4t-backtest
```

## Documentation

- [Installation](getting-started/installation.md) -- setup and verification
- [Quickstart](getting-started/quickstart.md) -- your first backtest in 5 minutes
- [Book Guide](book-guide/index.md) -- map the book and case studies to reusable APIs
- [How It Works](concepts/how-it-works.md) -- architecture and execution flow
- [Execution Semantics](user-guide/execution-semantics.md) -- fill ordering, stops, timing
- [Configuration](user-guide/configuration.md) -- all 40+ knobs explained
- [Profiles](user-guide/profiles.md) -- framework parity and presets
- [Strategies](user-guide/strategies.md) -- writing strategies and templates
- [Stateful Strategies](user-guide/stateful-strategies.md) -- advanced event-driven patterns
- [Risk Management](user-guide/risk-management.md) -- stops, trails, portfolio limits
- [Rebalancing](user-guide/rebalancing.md) -- weight-based portfolio management
- [Data Feed](user-guide/data-feed.md) -- preparing price and signal data
- [Results & Analysis](user-guide/results.md) -- metrics, trades, equity export
- [Market Impact](user-guide/market-impact.md) -- commission, slippage, and impact models
- [Orders](user-guide/orders.md) -- order types and bracket orders
- [Accounts](user-guide/accounts.md) -- cash, crypto, and margin accounts
- [API Reference](api/index.md) -- full API documentation

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
