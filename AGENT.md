# ml4t-backtest

Minimal event-driven backtesting engine.

## Structure

| Directory | Purpose |
|-----------|---------|
| src/ml4t/backtest/ | Package root |
| tests/ | 154 tests |
| validation/ | VectorBT/Backtrader matching |

## Key Modules

| Module | Purpose |
|--------|---------|
| engine.py | Event loop orchestration |
| broker.py | Order execution, positions |
| strategy.py | Strategy base class |
| datafeed.py | Price + signal iteration |

## Entry Point

```python
from ml4t.backtest import Engine, Strategy
```
