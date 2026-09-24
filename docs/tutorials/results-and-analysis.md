# Inspect, join, and export results

This example reuses the bundled synthetic BTC perpetual panel. It declares
`America/New_York` timestamps on the price bars, a fixed prediction value, and
two funding rates. The strategy buys one contract and later closes it. No
commission or slippage is charged. The resulting equity curve includes trading
P&L and funding cash flows, while fills, trades, and funding remain separate
records. See [Costs and Funding](costs-and-funding.md) for the event math.

## Run and reconcile the exports

The code compares Polars timestamp dtypes before joining results to inputs.
The joins use `(timestamp, asset)` for fills and funding, and `timestamp` for
equity and predictions. There is no user-side zone cast. A temporary artifact
is written with `to_parquet()` and read with `BacktestResult.from_parquet()`;
frame equality then checks every exported surface used in this example.

<!-- ml4t-doc-test: tutorial-results-export -->
```python
from datetime import datetime
from importlib.metadata import version
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import polars as pl
from polars.testing import assert_frame_equal
from ml4t.backtest import AssetClass, BacktestConfig, BacktestResult, ContractSpec, DataFeed, Engine
from ml4t.backtest.example_data import ExampleRoundTrip, load_example_prices


zone = "America/New_York"
prices = load_example_prices("crypto_perp").filter(pl.col("asset") == "BTC-PERP").with_columns(
    pl.col("timestamp").dt.replace_time_zone(zone)
)
signals = prices.select("timestamp", "asset").with_columns(pl.lit(0.4).alias("prediction"))
funding = pl.DataFrame({
    "timestamp": [datetime(2024, 1, day, tzinfo=ZoneInfo(zone)) for day in (4, 7)],
    "asset": ["BTC-PERP", "BTC-PERP"],
    "rate": [0.001, -0.0005],
})
result = Engine(
    DataFeed(prices_df=prices, signals_df=signals), ExampleRoundTrip("BTC-PERP", 1),
    BacktestConfig(initial_cash=100_000, allow_leverage=True, timezone=zone),
    contract_specs={"BTC-PERP": ContractSpec("BTC-PERP", AssetClass.FUTURE, margin=4000)},
    funding_df=funding,
).run()
equity = result.to_equity_dataframe()
fills = result.to_fills_dataframe()
trades = result.to_trades_dataframe()
state = result.to_portfolio_state_dataframe()
payments = result.to_funding_dataframe()
predictions = result.to_predictions_dataframe()
print("ml4t-backtest " + version("ml4t-backtest"))
frames = {
    "prices": prices,
    "predictions": predictions,
    "equity": equity,
    "fills": fills,
    "state": state,
    "funding": payments,
}
for name, frame in frames.items():
    assert frame.schema["timestamp"] == prices.schema["timestamp"]
assert trades.schema["entry_time"] == prices.schema["timestamp"]
assert trades.schema["exit_time"] == prices.schema["timestamp"]
print(f"timestamp zone: {prices.schema['timestamp'].time_zone}")
fill_match = fills.join(prices.select("timestamp", "asset", "close"),
                        on=["timestamp", "asset"], how="inner")
equity_match = equity.join(predictions.select("timestamp").unique(),
                           on="timestamp", how="inner")
funding_match = payments.join(prices.select("timestamp", "asset"),
                              on=["timestamp", "asset"], how="inner")
print(f"joined: fills {fill_match.height}/{fills.height}, "
      f"equity {equity_match.height}/{equity.height}, "
      f"funding {funding_match.height}/{payments.height}")
trading_pnl = trades.select(pl.col("pnl").sum()).item()
funding_cash = payments.select(pl.col("cash_delta").sum()).item()
net_change = result.metrics["final_value"] - 100_000
assert abs(trading_pnl + funding_cash - net_change) < 0.01
print(f"trading={trading_pnl:.2f} funding={funding_cash:.2f} "
      f"net={net_change:.2f}")
print(f"returns={equity.height} portfolio states={state.height}")
with TemporaryDirectory() as directory:
    result.to_parquet(directory)
    restored = BacktestResult.from_parquet(directory)
    for method in (
        "to_fills_dataframe", "to_trades_dataframe", "to_equity_dataframe",
        "to_portfolio_state_dataframe", "to_funding_dataframe",
        "to_predictions_dataframe",
    ):
        original = getattr(result, method)()
        loaded = getattr(restored, method)()
        assert_frame_equal(original, loaded)
    print("artifact round trip: fills, trades, equity, state, funding, predictions equal")
```

<!-- ml4t-doc-output: tutorial-results-export -->
```text
ml4t-backtest {package_version}
timestamp zone: America/New_York
joined: fills 2/2, equity 9/9, funding 2/2
trading=700.00 funding=-20.45 net=679.55
returns=9 portfolio states=9
artifact round trip: fills, trades, equity, state, funding, predictions equal
```

The `return` column in the equity frame is the bar-to-bar equity return; its
first row is zero because there is no preceding bar. Portfolio state records
end-of-bar cash, equity, and exposure. A fill is an execution event, while a
trade is a realized exit leg. Funding payments are cash flows outside the fill
stream. Thus the $700 realized trading P&L plus -$20.45 funding equals the
$679.55 net equity change. Nine prediction and equity timestamps, two fills,
and two funding events join to the original input timestamps exactly.

The default artifact includes a manifest, config and run spec, metrics, and
Parquet components for trades, fills, rejected orders, predictions, equity,
portfolio state, daily P&L, and funding. The manifest is checked on load; see
[Results & Analysis](../user-guide/results.md#parquet-export) for the full
format and recovery rules. The separate [diagnostic handoff](diagnostic-handoff.md)
shows an optional, tested post-backtest analysis step.

## In the book

Chapter 16, [Strategy simulation and reporting](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/16_strategy_simulation/README.md),
uses fill and trade reconciliation in [notebook 04, Single asset ml4t-backtest](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/16_strategy_simulation/04_single_asset_ml4t_backtest.ipynb).
Chapter 17's [portfolio metrics notebook](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/17_portfolio_construction/01_portfolio_metrics.ipynb)
extends the reporting workflow to allocation diagnostics.
