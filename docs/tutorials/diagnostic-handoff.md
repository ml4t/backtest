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
    BacktestConfig(initial_cash=100_000, allow_leverage=True, calendar="crypto"),
    contract_specs={"BTC-PERP": ContractSpec("BTC-PERP", AssetClass.FUTURE, margin=4000)},
).run()
assert result.config.resolved_calendar == "crypto"
assert result.equity.periods_per_year == 365.0
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

Chapter 17, Section 17.3, [Portfolio metrics](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/17_portfolio_construction/01_portfolio_metrics.ipynb) applies `ml4t-diagnostic` to a larger ETF allocation. The small example above tests the bridge before adding benchmark and rolling analyses.

## Send observed intraday accounts and holdings

The upcoming Backtest successor release can capture actual holdings with
`holdings_sampling="bar"`. The example below uses the public `PortfolioAnalysis`
constructor from Diagnostic 0.1.9. It preserves the three minute instants and
uses an explicitly chosen annualization scale. Returns include the first fill's
commission. Holdings come from account observations, including positions still
open at the end of the run.

<!-- ml4t-doc-test: tutorial-observed-diagnostic-handoff -->
```python
from datetime import UTC, datetime, timedelta

import polars as pl
from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.config import CommissionType, DataFrequency, ExecutionMode
from ml4t.diagnostic.evaluation import PortfolioAnalysis

class BuyOnce(Strategy):
    def on_data(self, timestamp, data, context, broker):
        if not broker.positions:
            broker.submit_order("TEST", 10)

timestamps = [
    datetime(2025, 1, 2, 15, tzinfo=UTC) + timedelta(minutes=i)
    for i in range(3)
]
prices = pl.DataFrame({
    "timestamp": timestamps, "asset": ["TEST"] * 3, "close": [100.0] * 3,
})
result = Engine(
    DataFeed(prices_df=prices), BuyOnce(),
    BacktestConfig(
        initial_cash=10_000, calendar=None,
        data_frequency=DataFrequency.MINUTE_1, periods_per_year=252 * 390,
        execution_mode=ExecutionMode.SAME_BAR,
        commission_type=CommissionType.PER_TRADE, commission_per_trade=10,
    ),
    holdings_sampling="bar",
).run()
accounts = result.to_portfolio_state_dataframe()
holdings = result.to_holdings_dataframe()
assert accounts["holdings_complete"].all()
assert accounts["cash"].to_list() == [8990.0] * 3
assert accounts["equity"].to_list() == [9990.0] * 3
positions = holdings.select(
    pl.col("timestamp").alias("date"), "asset",
    pl.col("liquidation_value").alias("value"),
)
analysis = PortfolioAnalysis(
    returns=result.to_returns_series(), dates=pl.Series("date", timestamps),
    positions=positions, periods_per_year=252 * 390,
)
assert abs(analysis.compute_summary_stats().total_return + 0.001) < 1e-12
assert positions["date"].to_list() == timestamps
print("Observed minute accounts: 3; investment return: -0.1%")
```

<!-- ml4t-doc-output: tutorial-observed-diagnostic-handoff -->
```text
Observed minute accounts: 3; investment return: -0.1%
```

The account table supplies `cash` and `equity` separately from positions.
For a multi-asset position panel, an absent asset can be filled with zero only
at observations marked `holdings_complete=True`. An all-cash account has real
account observations and an empty holdings table; it needs no invented asset.
Keep `observation_id` when storing fill-level snapshots: several economically
distinct observations can share one timestamp.

Diagnostic's planned holdings-analysis API has also been exercised with these
producer tables, using its pinned source candidate. That API is not part of the
published Diagnostic 0.1.9 package. Use the public constructor above for published
return analysis; Backtest continues to supply accounts, holdings and fills rather
than implementing a second analytics library.
