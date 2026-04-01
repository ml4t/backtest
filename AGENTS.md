# ml4t-backtest

Event-driven backtesting engine with cross-framework parity validation.

## Structure

| Directory | Purpose |
|-----------|---------|
| src/ml4t/backtest/ | Package root (~17.0k lines, 40+ modules) |
| tests/ | 1,443 passing tests, 13 skipped |
| validation/ | Cross-framework parity (VectorBT, Backtrader, Zipline, LEAN) |

## Key Modules

| Module | Lines | Purpose |
|--------|-------|---------|
| broker.py | 1,678 | Order execution, positions |
| result.py | 1,057 | BacktestResult container |
| config.py | 1,044 | BacktestConfig (40+ knobs) |
| calendar.py | 794 | Trading calendar, sessions |
| types.py | 647 | Order, Position, Fill, Trade, cost decomposition |
| engine.py | 578 | Event loop orchestration |
| profiles.py | 384 | 6 core + 4 strict profiles |
| export.py | 320 | Result export (Parquet, YAML, JSON) |
| sessions.py | 279 | Session handling |
| models.py | 248 | Commission/slippage models |
| datafeed.py | 394 | Price/signal iteration |
| strategy.py | 38 | Strategy base class |

## Subpackages

| Directory | Lines | Purpose |
|-----------|-------|---------|
| execution/ | 1,894 | Fill executor, rebalancer, impact, schedule |
| core/ | 1,538 | Order book, execution engine, fill engine, risk engine |
| accounting/ | 1,123 | Cash/margin policies, gatekeeper |
| analytics/ | 1,220 | Metrics, equity, trades, diagnostic bridge |
| risk/ | 1,790 | Position rules, portfolio limits |
| strategies/ | 492 | Strategy templates |

## Entry Point

```python
from ml4t.backtest import Engine, Strategy, BacktestConfig, run_backtest
```
