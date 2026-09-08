# ml4t-backtest

Event-driven backtesting for quantitative strategies with explicit execution, accounting, and risk
semantics. The package uses runtime-neutral lifecycle and order contracts from `ml4t.specs` so a
strategy can be compared with other frameworks and moved toward live execution without changing its
decision logic.

## Public entry points

```python
from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy, run_backtest
```

Prefer exports from `ml4t.backtest` for user-facing code. `Engine` coordinates the event loop,
`DataFeed` supplies time-ordered market data, `Strategy` defines decisions, and `BacktestConfig`
controls execution assumptions. `BacktestResult` contains portfolio, order, trade, and diagnostic
outputs.

## Source map

| Path | Responsibility |
|---|---|
| `src/ml4t/backtest/engine.py` | Event-loop orchestration and top-level execution |
| `src/ml4t/backtest/broker.py` | Orders, fills, positions, and broker state |
| `src/ml4t/backtest/datafeed.py` | Price and signal iteration |
| `src/ml4t/backtest/config.py` | Backtest and execution configuration |
| `src/ml4t/backtest/execution/` | Rebalancing, fill execution, impact, and schedules |
| `src/ml4t/backtest/accounting/` | Cash, margin, and accounting policies |
| `src/ml4t/backtest/risk/` | Position and portfolio risk rules |
| `src/ml4t/backtest/analytics/` | Metrics, equity, trades, and diagnostic integration |
| `validation/` | Cross-framework parity and retained performance evidence |
| `tests/` | Unit, integration, contract, and regression coverage |

Read the nearest nested `AGENTS.md` before changing a subsystem. Those guides identify the local
contracts and entry points without repeating this repository overview.

## Behavioral constraints

- Preserve causal information boundaries: a decision cannot use a completed price and fill at that
  same price unless the configured lifecycle contract explicitly permits it.
- Keep runtime-neutral lifecycle, intent, execution-policy, and position-rule contracts in
  `ml4t-specs`; this repository owns their backtest implementation.
- Treat parity claims as claims about explicit configurations, not framework defaults. Update the
  retained validation evidence when behavior or a comparison configuration changes.
- Keep public result artifacts versioned and readable across compatible releases.

## Quality commands

```bash
uv sync
uv run ruff check src tests
uv run ruff format --check src tests
uv run ty check
uv run pytest
pre-commit run --all-files
```
