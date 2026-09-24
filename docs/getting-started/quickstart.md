# Run your first completed backtest

Install the package as described in [Installation](installation.md), then run the
following Python script. It loads nine synthetic daily AAPL bars from the wheel,
submits a buy order, submits a close order four asset bars later, and inspects
the resulting fills, trade, and equity. The bundled data is described in
[Example Data](../tutorials/data.md).

<!-- ml4t-doc-test: quickstart-minimal -->
```python
from importlib.metadata import version

import polars as pl

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.example_data import load_example_prices


class FirstRoundTrip(Strategy):
    def __init__(self):
        self.asset_bars = 0

    def on_data(self, timestamp, data, context, broker):
        if "AAPL" not in data:
            return
        self.asset_bars += 1
        if self.asset_bars == 1:
            broker.submit_order("AAPL", 10)
        elif self.asset_bars == 5:
            broker.close_position("AAPL")


prices = load_example_prices("equity").filter(pl.col("asset") == "AAPL")
feed = DataFeed(prices_df=prices)
engine = Engine(feed, FirstRoundTrip(), BacktestConfig(initial_cash=100_000))
result = engine.run()

print("ml4t-backtest " + version("ml4t-backtest"))
print(f"bars: {len(feed)}")
for fill in result.fills:
    print(f"{fill.timestamp:%Y-%m-%d} {fill.side.value} {fill.quantity:g} @ ${fill.price:.2f}")
print(f"closed trades: {len(result.to_trades_dataframe())}")
print(f"trade P&L: ${result.trades[0].pnl:.2f}")
equity = result.to_equity_dataframe()
print(f"equity points: {equity.height}")
print(f"final equity: ${result.metrics['final_value']:.2f}")
```

<!-- ml4t-doc-output: quickstart-minimal -->
```text
ml4t-backtest {package_version}
bars: 9
2024-01-03 buy 10 @ $188.37
2024-01-09 sell 10 @ $191.37
closed trades: 1
trade P&L: $30.00
equity points: 9
final equity: $100030.00
```

The feed provides `timestamp`, `asset`, and OHLCV columns. `DataFeed` delivers
the AAPL bar to `on_data` at each timestamp. The first callback queues a buy;
the fifth queues a close. The default `NEXT_BAR` execution mode fills both
market orders at the next AAPL bar's open. The orders were submitted on January
2 and January 8, and the fills occurred on January 3 and January 9. The run
starts with $100,000, charges no commission or slippage, and ends with one
closed trade worth $30. The nine equity points include cash and the marked
position after each bar. These synthetic prices illustrate accounting, not a
claim about AAPL's historical return.

`result.to_fills_dataframe()` gives one row per execution. `result.to_trades_dataframe()`
gives the completed position round trip, including entry, exit, and P&L.
`result.to_equity_dataframe()` gives the dated account value. Inspect these
three frames before interpreting a summary metric; [Results & Analysis](../user-guide/results.md)
covers export and additional measures.

An `Engine` instance runs once. Create a new instance to test another strategy
or setting. Continue with [Order Types](../user-guide/orders.md) and
[Execution Semantics](../user-guide/execution-semantics.md) to see how order
choice and timing change fills.

## In the book

Chapter 16, Section 16.3, [Vectorized and event-driven backtesting](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/16_strategy_simulation/README.md),
and [notebook 04, Single Asset Backtest with ml4t-backtest](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/16_strategy_simulation/04_single_asset_ml4t_backtest.ipynb)
extend this first round trip to a stateful RSI rule, explicit costs, and a matched
comparison with a vectorized backtest.
