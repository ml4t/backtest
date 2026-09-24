# Send a backtest to ml4t-diagnostic

`ml4t-diagnostic` is an optional dependency for post-backtest analysis. This
example was verified with `ml4t-diagnostic==0.1.4` in addition to the installed
`ml4t-backtest` wheel. Install that version in an analysis environment before
running the code. The core [result export tutorial](results-and-analysis.md)
runs without it.

The helper `portfolio_analysis_from_result()` reads daily returns from the
backtest result. `calendar="crypto"` selects the crypto annualization calendar.
`compute_summary_stats()` returns a metrics object; the displayed total
return and maximum drawdown come from that object. This example has no funding
or trading costs, so its figures differ from the funded run in the core
result export tutorial.

<!-- ml4t-doc-test: tutorial-diagnostic-handoff -->
```python
from importlib.metadata import version

import polars as pl
from ml4t.backtest import AssetClass, BacktestConfig, ContractSpec, DataFeed, Engine
from ml4t.backtest.example_data import ExampleRoundTrip, load_example_prices
from ml4t.diagnostic.integration import portfolio_analysis_from_result

prices = load_example_prices("crypto_perp").filter(pl.col("asset") == "BTC-PERP")
result = Engine(
    DataFeed(prices_df=prices), ExampleRoundTrip("BTC-PERP", 1),
    BacktestConfig(initial_cash=100_000, allow_leverage=True),
    contract_specs={"BTC-PERP": ContractSpec("BTC-PERP", AssetClass.FUTURE, margin=4000)},
).run()
analysis = portfolio_analysis_from_result(result, calendar="crypto")
stats = analysis.compute_summary_stats()
print("ml4t-backtest " + version("ml4t-backtest"))
print(f"daily returns={len(analysis.returns)} total={stats.total_return:.6f} "
      f"max drawdown={stats.max_drawdown:.6f}")
```

<!-- ml4t-doc-output: tutorial-diagnostic-handoff -->
```text
ml4t-backtest {package_version}
daily returns=9 total=0.007000 max drawdown=-0.002986
```

The example reports nine daily-return observations, not nine closed trades.
For a larger analysis, pass a benchmark to the helper and inspect the
[Results & Analysis](../user-guide/results.md#integration-with-ml4t-diagnostic)
reference for trade, fill, and portfolio-state handoffs.

## In the book

Chapter 17, Section 17.3, [Portfolio metrics](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/17_portfolio_construction/01_portfolio_metrics.ipynb) applies `ml4t-diagnostic` to a larger ETF allocation. The small example above tests the bridge before adding benchmark and rolling analyses.
