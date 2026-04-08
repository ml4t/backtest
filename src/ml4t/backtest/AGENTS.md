# ml4t.backtest - Package Index

## Core Modules

| File | Lines | Purpose |
|------|-------|---------|
| broker.py | 1,678 | Order execution, positions, risk eval |
| result.py | 1,057 | BacktestResult container |
| config.py | 1,044 | BacktestConfig, 40+ behavioral knobs |
| calendar.py | 794 | Trading calendar, overnight sessions |
| types.py | 647 | Order, Position, Fill, Trade, cost decomposition |
| engine.py | 578 | Event loop orchestration |
| profiles.py | 384 | 6 core + 4 strict framework profiles |
| export.py | 320 | Result export (Parquet, YAML, JSON) |
| sessions.py | 279 | Session handling |
| models.py | 248 | Commission/slippage models |
| datafeed.py | 394 | Price/signal iteration |
| strategy.py | 38 | Strategy base class |

## Subpackages

| Directory | Lines | Purpose |
|-----------|-------|---------|
| execution/ | 1,894 | Fill executor, rebalancer, impact, limits, schedule |
| core/ | 1,538 | Order book, execution engine, fill engine, risk engine |
| accounting/ | 1,123 | Unified account policy, gatekeeper |
| analytics/ | 1,220 | Metrics, equity, trades, diagnostic bridge |
| risk/ | 1,790 | Position rules (stop/trail/TP), portfolio limits |
| strategies/ | 492 | Strategy templates |

## Key

`Engine`, `Broker`, `Strategy`, `BacktestConfig`, `BacktestResult`
