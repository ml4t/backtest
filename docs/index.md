# ML4T Backtest

Event-driven backtesting engine with configurable execution semantics, validated against four independent frameworks.

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

**Validated at scale.** 225,000+ trades verified trade-by-trade against VectorBT Pro, Backtrader, Zipline, and LEAN on real market data (250 assets x 20 years).

**Fast.** 19x faster than Backtrader, 8x faster than Zipline, 5x faster than LEAN on identical workloads. Processes 40,000+ bars/second across 250 assets.

| Feature | Description |
|---------|-------------|
| Event-driven | Point-in-time correctness, no look-ahead bias |
| 40+ behavioral knobs | Every execution detail is configurable |
| 10 framework profiles | Match VectorBT, Backtrader, Zipline, LEAN exactly |
| Risk management | Stop-loss, take-profit, trailing stops, portfolio limits |
| Multi-asset | Rebalancing, weight targets, exit-first ordering |

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

## Part of the ML4T Ecosystem

```
ml4t-data --> ml4t-engineer --> ml4t-diagnostic --> ml4t-backtest --> ml4t-live
```

The same `Strategy` class works in both backtest and live trading via `ml4t-live`.
