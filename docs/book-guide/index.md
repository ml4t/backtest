# Book Guide

This map connects selected *Machine Learning for Trading, Third Edition*
companion notebooks to the library workflows that implement the corresponding
backtest operations. For the library task guide, start at the
[User Guide](../user-guide/index.md). The [tutorials](../getting-started/quickstart.md) work with
bundled synthetic data, so the book and its datasets are optional. All book
links below point to one checked companion revision.

The **Notebook role** column distinguishes direct library examples from
research that supplies inputs or interprets outputs. A case study that uses a
book-specific helper is labeled as such; its helper and datasets are not part
of the installed `ml4t-backtest` package. Start with the linked library workflow
for a standalone example.

## Chapters and workflows

| Book section and notebook | Notebook role | What it adds | Library workflow |
|---|---|---|---|
| [Futures backtesting](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/16_strategy_simulation/02_futures_backtesting.ipynb) | Calls `DataFeed` and `Engine` | Prepare futures bars and contract specifications | [Example data](../tutorials/data.md) |
| [16.3 Vectorized and event-driven backtesting](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/16_strategy_simulation/04_single_asset_ml4t_backtest.ipynb) | Calls `DataFeed` and `Engine` | Run one strategy, reconcile fills and trades | [First backtest](../getting-started/quickstart.md) |
| [16.3 Stateful strategies](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/16_strategy_simulation/05_stateful_strategies.ipynb) | Calls `DataFeed` and `Engine` | Carry realized state into later decisions | [Risk and state](../tutorials/risk-and-state.md) |
| [16.3 Vectorized and event-driven backtesting](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/16_strategy_simulation/07_engine_divergence_anatomy.ipynb) | Calls `Engine` with controlled settings | Change one execution assumption at a time | [Profiles and parity](../tutorials/profiles-and-parity.md); [Zipline migration](../user-guide/migrate-from-zipline.md) |
| [16.5 Understanding performance metrics](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/16_strategy_simulation/09_performance_reporting.ipynb) | Calls `Engine` and `ml4t-diagnostic` | Read returns and drawdowns | [Result exports](../tutorials/results-and-analysis.md) |
| [Case-study LEAN parity](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/16_strategy_simulation/16_case_study_lean_parity.ipynb) | Reads retained comparison evidence | Interpret the bounded framework audit | [Profiles and parity](../tutorials/profiles-and-parity.md) |
| [Portfolio performance analysis](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/17_portfolio_construction/01_portfolio_metrics.ipynb) | Calls `ml4t-diagnostic` for downstream analysis | Analyze returns and drawdowns | [Diagnostic handoff](../tutorials/diagnostic-handoff.md) |
| [17.4 Defining Baseline Allocators](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/17_portfolio_construction/07_conformal_position_sizing.ipynb) | Teaches sizing from prediction uncertainty | Turn uncertainty into position sizes | [Accounts and constraints](../tutorials/accounts-and-constraints.md) |
| [17.7 Comparing Allocator Performance](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/17_portfolio_construction/08_library_comparison.ipynb) | Calls `Engine` for allocator comparison | Compare allocators with matched inputs | [Multi-asset rebalancing](../tutorials/multiasset-rebalancing.md) |
| [Market impact scenarios](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/18_transaction_costs/03_market_impact_calibration.ipynb) | Teaches calibration from market panels | Assess size and capacity assumptions | [Market impact](../user-guide/market-impact.md) |
| [18.7 Transaction Cost Analysis and Model Validation](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/18_transaction_costs/10_gross_vs_net_performance.ipynb) | Teaches cost analysis from return series | Reconcile gross and net performance | [Costs and funding](../tutorials/costs-and-funding.md) |
| [19.4 Drawdowns, Path Risk, and Time-to-Recovery](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/19_risk_management/02_exit_strategies.ipynb) | Uses library risk and trade types in a research comparison | Compare fixed and trailing exits | [Risk and state](../tutorials/risk-and-state.md) |
| [19.4 Drawdowns, Path Risk, and Time-to-Recovery](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/19_risk_management/10_ml4t_backtest_risk_demo.ipynb) | Calls `ml4t.backtest.risk` directly | Use library position rules and portfolio limits | [Risk management](../user-guide/risk-management.md) |

The book develops research questions, statistical interpretation, and
larger datasets. The library pages specify feed contracts, order timing,
account behavior, executable examples, and result schemas. Follow the library
reference when a notebook and the current API differ.

## Case studies

| Companion example | Notebook role | What it adds | Library workflow |
|---|---|---|---|
| [ETF backtest](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/case_studies/etfs/14_backtest.ipynb) | Calls the book helper `backtest_runner`, which uses `Engine` | Weight targets from a prediction stream | [Multi-asset rebalancing](../tutorials/multiasset-rebalancing.md) |
| [CME futures backtest](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/case_studies/cme_futures/13_backtest.ipynb) | Uses a book research workflow | Futures prediction selection and equal-weight baseline | [Example data](../tutorials/data.md) |
| [FX pairs backtest](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/case_studies/fx_pairs/13_backtest.ipynb) | Uses a book research workflow | FX prediction population and strategy grid | [Data Feed](../user-guide/data-feed.md) |
| [Crypto perpetual funding](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/case_studies/crypto_perps_funding/16_costs.ipynb) | Uses a book research workflow | Funding and transaction-cost assumptions | [Costs and funding](../tutorials/costs-and-funding.md) |
| [ETF risk controls](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/case_studies/etfs/16_risk_management.ipynb) | Calls the book helper `backtest_runner`, which uses `Engine` | Position exits in a full strategy | [Risk management](../tutorials/risk-and-state.md) |

## Move from a notebook to a reusable run

1. Start with the [first backtest](../getting-started/quickstart.md) and the
   [bundled price panels](../tutorials/data.md).
2. Reproduce the notebook's decision and fill timing with the
   [orders tutorial](../tutorials/orders-and-timing.md).
3. Specify capital, share precision, and exposure limits with the
   [account tutorial](../tutorials/accounts-and-constraints.md).
4. Add costs or risk rules only after checking the baseline fills.
5. Export the result and join it to the original timestamps with the
   [result tutorial](../tutorials/results-and-analysis.md).

The [API reference](../api/index.md) gives current signatures for each step.
