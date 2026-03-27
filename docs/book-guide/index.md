# Book Guide

Use this guide to move between *Machine Learning for Trading, Third Edition* and
`ml4t-backtest` without guessing which notebook or case-study file maps to which
production API.

The book teaches the concepts step by step. `ml4t-backtest` packages those ideas into
reusable execution, reporting, and portfolio-simulation workflows.

## How To Use This Guide

- start in the book when you want intuition, derivations, and side-by-side experiments
- start in the library docs when you want stable APIs and configurable workflows
- use this page to jump from a chapter or case study to the corresponding guide page

The most common reader journey is:

1. read the notebook that explains the trading or execution idea
2. use this guide to find the matching library workflow
3. read the user-guide page for the reusable configuration pattern
4. use the [API Reference](../api/index.md) for exact classes and signatures

## Chapter Map

### Chapter 16: Strategy Simulation

| Book path | What the book teaches | Library workflow | Docs page |
|-----------|-----------------------|------------------|-----------|
| `16_strategy_simulation/02_order_types.py` | order semantics, stop/limit behavior, and timing choices | order types and trigger behavior | [Order Types](../user-guide/orders.md) |
| `16_strategy_simulation/03_execution_semantics.py` | next-bar vs same-bar timing, stop handling, and execution assumptions | execution-mode, trigger, and stop configuration | [Execution Semantics](../user-guide/execution-semantics.md) |
| `16_strategy_simulation/04_configuration_profiles.py` | framework-style presets and reproducible backtest behavior | named config profiles and `BacktestConfig` | [Configuration](../user-guide/configuration.md), [Profiles](../user-guide/profiles.md) |
| `16_strategy_simulation/05_performance_reporting.py` | trade analytics, tearsheets, and gross-vs-net reporting | `BacktestResult`, trade/fill/equity export, reporting helpers | [Results & Analysis](../user-guide/results.md) |
| `16_strategy_simulation/06_framework_parity.py` | why configurable semantics matter for reproducibility | strict profiles and parity validation philosophy | [Profiles](../user-guide/profiles.md) |
| `16_strategy_simulation/08_rebalancing.py` | weight-based portfolio management | target weights and rebalancing executors | [Rebalancing](../user-guide/rebalancing.md) |

What changes when you move to the library:

- the book shows concepts one execution assumption at a time
- the library exposes those assumptions as named configuration knobs
- results become exportable, replayable, and comparable across frameworks

### Chapter 17: Portfolio Construction

| Book path | What the book teaches | Library workflow | Docs page |
|-----------|-----------------------|------------------|-----------|
| `17_portfolio_construction/01_portfolio_metrics.py` | allocator comparison and portfolio-level behavior | portfolio accounting and result metrics | [Results & Analysis](../user-guide/results.md), [Account Policies](../user-guide/accounts.md) |
| `17_portfolio_construction/08_library_comparison.py` | comparing portfolio construction workflows | feed preparation, rebalancing, and configuration | [Data Feed](../user-guide/data-feed.md), [Rebalancing](../user-guide/rebalancing.md) |
| `17_portfolio_construction/09_allocation_horse_race.py` | realistic allocator comparison under costs | cost-aware execution and turnover analysis | [Results & Analysis](../user-guide/results.md), [Market Impact](../user-guide/market-impact.md) |

What changes when you move to the library:

- allocators become reusable strategies or weight targets
- fills, turnover, and portfolio state become first-class reporting surfaces
- account policy becomes explicit instead of hidden in notebook assumptions

### Chapter 18: Costs

| Book path | What the book teaches | Library workflow | Docs page |
|-----------|-----------------------|------------------|-----------|
| `18_costs/01_transaction_costs.py` | commissions, slippage, and cost decomposition | commission and slippage configuration | [Configuration](../user-guide/configuration.md), [Market Impact](../user-guide/market-impact.md) |
| `18_costs/02_market_impact.py` | realistic impact assumptions and sensitivity to size | market-impact models and execution drag | [Market Impact](../user-guide/market-impact.md) |
| `18_costs/03_quote_aware_execution.py` | bid/ask-aware execution and quote-side marking | quote-aware fills, bid/ask mark prices, quote-side execution | [Execution Semantics](../user-guide/execution-semantics.md), [Data Feed](../user-guide/data-feed.md) |

What changes when you move to the library:

- manual cost adjustments become configuration-backed execution models
- quote-aware assumptions become explicit and reproducible
- fills and trades carry enough context to audit cost drag directly

### Chapter 19: Risk Management

| Book path | What the book teaches | Library workflow | Docs page |
|-----------|-----------------------|------------------|-----------|
| `19_risk_management/02_exit_strategies.py` | stop-loss, take-profit, and trailing-stop design | position-rule composition | [Risk Management](../user-guide/risk-management.md) |
| `19_risk_management/03_position_sizing.py` | position sizing and constraints | account policy, sizing, and exposure limits | [Account Policies](../user-guide/accounts.md), [Risk Management](../user-guide/risk-management.md) |
| `19_risk_management/06_portfolio_limits.py` | portfolio-level guardrails | portfolio limits and manager coordination | [Risk Management](../user-guide/risk-management.md) |

What changes when you move to the library:

- notebook exits become reusable rule objects
- portfolio-wide guardrails become configuration, not ad hoc if-statements
- risk and execution interact through explicit event ordering

## Case-Study Pipeline Map

Most case studies use `ml4t-backtest` in the final simulation and reporting stages.

| Case-study step | Typical file | Library workflow | Docs page |
|-----------------|--------------|------------------|-----------|
| Backtest setup | `case_studies/<study>/code/14_backtest.py` | `BacktestConfig`, `DataFeed`, `Engine`, strategy bridge | [Quickstart](../getting-started/quickstart.md), [Configuration](../user-guide/configuration.md), [Data Feed](../user-guide/data-feed.md) |
| Performance reporting | `case_studies/<study>/code/15_evaluation.py` or equivalent | `BacktestResult`, trade/fill/equity export, downstream diagnostics | [Results & Analysis](../user-guide/results.md) |
| Portfolio/risk comparison | allocator or risk notebooks in Ch17-Ch19 | account policy, turnover, limits, and rebalancing behavior | [Account Policies](../user-guide/accounts.md), [Rebalancing](../user-guide/rebalancing.md), [Risk Management](../user-guide/risk-management.md) |

Particularly strong bridges in the current materials:

- Ch16 case studies use `setup.yaml` and `get_backtest_config()` to turn notebook assumptions
  into reusable configuration
- NASDAQ-100 microstructure work demonstrates quote-aware execution and richer reporting
- portfolio-construction studies rely on rebalancing and turnover-aware result surfaces
- live-trading materials depend on keeping the same strategy shape before migrating to `ml4t-live`

## From Notebook Code to Library API

Use this translation when moving from the book to reusable code:

| Book pattern | Library equivalent |
|--------------|--------------------|
| inline execution assumptions | `BacktestConfig(...)` or named profile |
| notebook-specific fill logic | `ExecutionMode`, fill-ordering, stop-fill configuration |
| manual portfolio loops | rebalancers and weight-target executors |
| ad hoc reporting frames | `BacktestResult` exports for trades, fills, equity, and portfolio state |
| cost tweaks in separate analysis cells | slippage, commission, and market-impact models in config |
| one-off state carried across notebook cells | stateful strategy patterns in reusable `Strategy` subclasses |

## What To Prioritize

These are the workflows most readers should prioritize:

- execution semantics and configuration profiles
- data-feed preparation and signal alignment
- result export and performance reporting
- risk-management rules and rebalancing
- quote-aware execution and market-impact modeling when realism matters

## Related Docs

- [Quickstart](../getting-started/quickstart.md)
- [How It Works](../concepts/how-it-works.md)
- [Execution Semantics](../user-guide/execution-semantics.md)
- [Results & Analysis](../user-guide/results.md)
- [API Reference](../api/index.md)
