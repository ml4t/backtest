# ml4t-backtest

Event-driven backtesting engine for ML4T quantitative trading strategies.

## Features

- **Event-Driven Architecture**: Point-in-time correctness with no look-ahead bias
- **Exit-First Processing**: Matches real broker order execution behavior
- **VectorBT Validation**: Results validated against VectorBT Pro
- **Account Policies**: Cash and margin account support
- **Minimal Core**: ~2,800 lines of focused, maintainable code
- **100k+ events/sec**: High-performance event processing

## Installation

```bash
pip install ml4t-backtest
```

## Quick Start

```python
from ml4t.backtest import Engine, Strategy, BacktestConfig

class SimpleMovingAverage(Strategy):
    def __init__(self, fast=10, slow=30):
        self.fast = fast
        self.slow = slow
        
    def on_bar(self, bar):
        fast_ma = bar.close_ma(self.fast)
        slow_ma = bar.close_ma(self.slow)
        
        if fast_ma > slow_ma and self.position == 0:
            self.buy(size=100)
        elif fast_ma < slow_ma and self.position > 0:
            self.close()

config = BacktestConfig(
    initial_cash=100_000,
    commission=0.001,
)

engine = Engine(data, SimpleMovingAverage(), config)
result = engine.run()

print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe Ratio: {result.metrics['sharpe']:.2f}")
```

## Documentation

- **[AGENT.md](AGENT.md)**: Comprehensive API reference for agents and developers
- **[api.yaml](api.yaml)**: Machine-readable API specification

## Part of ML4T

This library is part of the ML4T quantitative trading toolkit:

- **ml4t-data**: Market data acquisition and storage
- **ml4t-engineer**: Feature engineering and indicators
- **ml4t-diagnostic**: Statistical validation and evaluation
- **ml4t-backtest**: Event-driven backtesting (this library)
- **ml4t-live**: Live trading platform

## License

MIT License - see LICENSE for details.
