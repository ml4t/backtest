# ML4T Backtest

Event-driven backtesting engine with point-in-time correctness, validated against VectorBT Pro.

## Overview

ML4T Backtest provides a minimal, high-performance backtesting engine that:

- Processes **100,000+ events per second**
- Uses **exit-first order processing** matching real broker behavior
- Is **validated against VectorBT Pro** for exact result matching
- Supports **cash and margin accounts** with configurable leverage

## Quick Example

```python
from ml4t.backtest import Engine, Strategy, BacktestConfig

class MyStrategy(Strategy):
    def on_bar(self, bar):
        if self.position == 0:
            self.buy(size=100)

config = BacktestConfig(initial_cash=100_000)
engine = Engine(data, MyStrategy(), config)
result = engine.run()

print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe Ratio: {result.metrics['sharpe']:.2f}")
```

## Key Features

| Feature | Description |
|---------|-------------|
| Event-Driven | Point-in-time correctness, no look-ahead bias |
| Exit-First | Processes exits before entries, matching real brokers |
| VectorBT Validated | Results match VectorBT Pro exactly |
| High Performance | 100,000+ events/second |
| Minimal Core | ~2,800 lines of focused code |

## Order Types

- `MARKET` - Execute at next bar open
- `LIMIT` - Execute if price crosses limit
- `STOP` - Trigger market order at stop price
- `STOP_LIMIT` - Trigger limit order at stop price

## Installation

```bash
pip install ml4t-backtest
```

## Next Steps

- [Installation Guide](getting-started/installation.md) - Setup instructions
- [Quickstart](getting-started/quickstart.md) - Build your first strategy
- [Strategy Guide](user-guide/strategies.md) - Advanced strategy patterns
- [API Reference](api/index.md) - Complete API documentation

## Part of the ML4T Library Suite

ML4T Backtest integrates seamlessly with other ML4T libraries:

```
ml4t-data → ml4t-engineer → ml4t-diagnostic → ml4t-backtest → ml4t-live
```

The same Strategy class works in both backtest and live trading via ml4t-live.
