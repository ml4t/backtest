# ml4t-backtest

Event-driven backtesting engine with cross-framework parity validation.

## Structure

| Directory | Purpose |
|-----------|---------|
| src/ml4t/backtest/ | Package root (~13.8k lines, 40 modules) |
| tests/ | 1,367 tests |
| validation/ | Cross-framework parity (VBT, Backtrader, Zipline, LEAN) |

## Key Modules

| Module | Lines | Purpose |
|--------|-------|---------|
| broker.py | 1,463 | Order execution, positions |
| result.py | 1,047 | BacktestResult container |
| config.py | 848 | BacktestConfig (40+ knobs) |
| calendar.py | 786 | Trading calendar, sessions |
| types.py | 625 | Order, Position, Fill, Trade, cost decomposition |
| engine.py | 419 | Event loop orchestration |
| profiles.py | 384 | 6 core + 4 strict profiles |
| export.py | 312 | Result export (Parquet, YAML, JSON) |
| sessions.py | 279 | Session handling |
| models.py | 248 | Commission/slippage models |
| datafeed.py | 224 | Price/signal iteration |
| strategy.py | 28 | Strategy base class |

## Subpackages

| Directory | Lines | Purpose |
|-----------|-------|---------|
| execution/ | 1,351 | Fill executor, rebalancer, impact |
| core/ | 1,314 | Order book, execution engine, fill engine, risk engine |
| accounting/ | 1,076 | Cash/margin policies, gatekeeper |
| analytics/ | 970 | Metrics, equity, trades, cost decomposition, diagnostic bridge |
| risk/ | 1,906 | Position rules, portfolio limits |
| strategies/ | 417 | Strategy templates |

## Entry Point

```python
from ml4t.backtest import Engine, Strategy, BacktestConfig, run_backtest
```
