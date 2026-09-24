# Book Guide

This map connects selected *Machine Learning for Trading, Third Edition*
companion notebooks to the library workflows that implement the corresponding
backtest operations. For the library task guide, start at the
[User Guide](../user-guide/index.md). The [tutorials](../getting-started/quickstart.md) work with
bundled synthetic data, so the book and its datasets are optional. All book
links below point to one checked companion revision.

## Chapters and workflows

| Book section and notebook | What it adds | Library workflow |
|---|---|---|
| [16.3 Vectorized and event-driven backtesting](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/16_strategy_simulation/04_single_asset_ml4t_backtest.ipynb) | Run one strategy, reconcile fills and trades | [First backtest](../getting-started/quickstart.md) |
| [16.3 Stateful strategies](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/16_strategy_simulation/05_stateful_strategies.ipynb) | Carry realized state into later decisions | [Risk and state](../tutorials/risk-and-state.md) |
| [16.5 Understanding performance metrics](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/16_strategy_simulation/09_performance_reporting.ipynb) | Read returns and drawdowns | [Result exports](../tutorials/results-and-analysis.md) |
| [16.3 Vectorized and event-driven backtesting](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/16_strategy_simulation/07_engine_divergence_anatomy.ipynb) | Change one execution assumption at a time | [Profiles and parity](../tutorials/profiles-and-parity.md) |
| [17.4 Defining Baseline Allocators](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/17_portfolio_construction/07_conformal_position_sizing.ipynb) | Turn uncertainty into position sizes | [Accounts and constraints](../tutorials/accounts-and-constraints.md) |
| [17.7 Comparing Allocator Performance](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/17_portfolio_construction/08_library_comparison.ipynb) | Compare allocators with matched inputs | [Multi-asset rebalancing](../tutorials/multiasset-rebalancing.md) |
| [18.7 Transaction Cost Analysis and Model Validation](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/18_transaction_costs/10_gross_vs_net_performance.ipynb) | Reconcile gross and net performance | [Costs and funding](../tutorials/costs-and-funding.md) |
| [19.4 Drawdowns, Path Risk, and Time-to-Recovery](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/19_risk_management/02_exit_strategies.ipynb) | Compare fixed and trailing exits | [Risk and state](../tutorials/risk-and-state.md) |
| [19.4 Drawdowns, Path Risk, and Time-to-Recovery](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/19_risk_management/10_ml4t_backtest_risk_demo.ipynb) | Use library position rules and portfolio limits | [Risk management](../user-guide/risk-management.md) |

The book develops research questions, statistical interpretation, and
larger datasets. The library pages specify feed contracts, order timing,
account behavior, executable examples, and result schemas. Follow the library
reference when a notebook and the current API differ.

## Case studies

| Companion example | What it adds | Library workflow |
|---|---|---|
| [ETF backtest](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/case_studies/etfs/14_backtest.ipynb) | Weight targets from a prediction stream | [Multi-asset rebalancing](../tutorials/multiasset-rebalancing.md) |
| [CME futures backtest](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/case_studies/cme_futures/13_backtest.ipynb) | Contract multipliers and futures sessions | [Example data](../tutorials/data.md) |
| [FX pairs backtest](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/case_studies/fx_pairs/13_backtest.ipynb) | USD-quoted pairs and signal alignment | [Data Feed](../user-guide/data-feed.md) |
| [Crypto perpetual funding](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/case_studies/crypto_perps_funding/16_costs.ipynb) | Funding and transaction-cost assumptions | [Costs and funding](../tutorials/costs-and-funding.md) |
| [ETF risk controls](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/case_studies/etfs/16_risk_management.ipynb) | Position exits in a full strategy | [Risk management](../tutorials/risk-and-state.md) |

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
