# ml4t.backtest - Package Index

## Core Modules

| File | Lines | Purpose |
|------|-------|---------|
| broker.py | 1,463 | Order execution, positions, risk eval |
| result.py | 1,047 | BacktestResult container |
| config.py | 848 | BacktestConfig, 40+ behavioral knobs |
| calendar.py | 786 | Trading calendar, overnight sessions |
| types.py | 625 | Order, Position, Fill, Trade, cost decomposition |
| engine.py | 419 | Event loop orchestration |
| profiles.py | 384 | 6 core + 4 strict framework profiles |
| export.py | 312 | Result export (Parquet, YAML, JSON) |
| sessions.py | 279 | Session handling |
| models.py | 248 | Commission/slippage models |
| datafeed.py | 224 | Price/signal iteration |
| strategy.py | 28 | Strategy base class |

## Subpackages

| Directory | Lines | Purpose |
|-----------|-------|---------|
| execution/ | 1,351 | Fill executor, rebalancer, impact, limits |
| core/ | 1,314 | Order book, execution engine, fill engine, risk engine |
| accounting/ | 1,076 | Unified account policy, gatekeeper |
| analytics/ | 970 | Metrics, equity, trades, cost decomposition, diagnostic bridge |
| risk/ | 1,906 | Position rules (stop/trail/TP), portfolio limits |
| strategies/ | 417 | Strategy templates |

## Key

`Engine`, `Broker`, `Strategy`, `BacktestConfig`, `BacktestResult`
