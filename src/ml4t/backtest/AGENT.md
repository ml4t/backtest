# ml4t.backtest - Package Index

## Core Modules

| File | Lines | Purpose |
|------|-------|---------|
| engine.py | 491 | Event loop orchestration |
| broker.py | 1,438 | Order execution, positions, risk eval |
| config.py | 937 | BacktestConfig, 40+ behavioral knobs |
| result.py | 1,025 | BacktestResult container |
| types.py | 578 | Order, Position, Fill, Trade |
| profiles.py | 375 | 6 core + 4 strict framework profiles |
| calendar.py | 786 | Trading calendar, overnight sessions |
| datafeed.py | 224 | Price/signal iteration |
| strategy.py | 28 | Strategy base class |
| models.py | 245 | Commission/slippage models |
| sessions.py | 279 | Session handling |
| export.py | 312 | Result export (Parquet, YAML, JSON) |

## Subpackages

| Directory | Lines | Purpose |
|-----------|-------|---------|
| core/ | 1,365 | Order book, execution engine, fill engine, risk engine |
| accounting/ | 1,180 | Cash/margin/crypto policies, gatekeeper |
| analytics/ | 917 | Metrics, equity, trades, diagnostic bridge |
| execution/ | 1,328 | Fill executor, rebalancer, impact, limits |
| risk/ | 1,876 | Position rules (stop/trail/TP), portfolio limits |
| strategies/ | 417 | Strategy templates |

## Key

`Engine`, `Broker`, `Strategy`, `BacktestConfig`, `BacktestResult`
