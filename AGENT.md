# ml4t-backtest

Event-driven backtesting engine with cross-framework parity validation.

## Structure

| Directory | Purpose |
|-----------|---------|
| src/ml4t/backtest/ | Package root (~14.3k lines, 40 modules) |
| tests/ | 1,083 tests |
| validation/ | Cross-framework parity (VBT, Backtrader, Zipline, LEAN) |

## Key Modules

| Module | Lines | Purpose |
|--------|-------|---------|
| engine.py | 491 | Event loop orchestration |
| broker.py | 1,438 | Order execution, positions |
| config.py | 937 | BacktestConfig (40+ knobs) |
| result.py | 1,025 | BacktestResult container |
| types.py | 578 | Order, Position, Fill, Trade |
| profiles.py | 375 | 6 core + 4 strict profiles |
| calendar.py | 786 | Trading calendar, sessions |

## Subpackages

| Directory | Lines | Purpose |
|-----------|-------|---------|
| core/ | 1,365 | Order book, execution engine, fill engine, risk engine |
| accounting/ | 1,180 | Cash/margin policies, gatekeeper |
| analytics/ | 917 | Metrics, equity, trades, diagnostic bridge |
| execution/ | 1,328 | Fill executor, rebalancer, impact |
| risk/ | 1,876 | Position rules, portfolio limits |
| strategies/ | 417 | Strategy templates |

## Entry Point

```python
from ml4t.backtest import Engine, Strategy, BacktestConfig, run_backtest
```
