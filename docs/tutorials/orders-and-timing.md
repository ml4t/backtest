# Change the order, change the fill

Start with the [completed AAPL backtest](../getting-started/quickstart.md).
Here, the strategy makes the same fixed decision on the first bar in every run:
submit one buy order for one share. Only the order type or execution setting
changes. The prices come from the bundled synthetic equity panel.

<!-- ml4t-doc-test: tutorial-orders-timing -->
```python
from importlib.metadata import version

import polars as pl
from ml4t.backtest import BacktestConfig, DataFeed, Engine, ExecutionMode, OrderType, Strategy
from ml4t.backtest.config import ExecutionPrice
from ml4t.backtest.example_data import load_example_prices


class SingleOrder(Strategy):
    def __init__(self, order_type, *, limit_price=None, stop_price=None):
        self.order_type = order_type
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.submitted = False

    def on_data(self, timestamp, data, context, broker):
        if self.submitted:
            return
        broker.submit_order(
            "AAPL", 1, order_type=self.order_type,
            limit_price=self.limit_price, stop_price=self.stop_price,
        )
        self.submitted = True


prices = load_example_prices("equity").filter(pl.col("asset") == "AAPL")
scenarios = [
    ("next market", ExecutionMode.NEXT_BAR, OrderType.MARKET, None, None),
    ("same market", ExecutionMode.SAME_BAR, OrderType.MARKET, None, None),
    ("next limit", ExecutionMode.NEXT_BAR, OrderType.LIMIT, 188.50, None),
    ("next stop", ExecutionMode.NEXT_BAR, OrderType.STOP, None, 191.00),
    ("unfilled limit", ExecutionMode.NEXT_BAR, OrderType.LIMIT, 170.00, None),
]
print("ml4t-backtest " + version("ml4t-backtest"))
for name, timing, kind, limit, stop in scenarios:
    strategy = SingleOrder(kind, limit_price=limit, stop_price=stop)
    engine = Engine(DataFeed(prices_df=prices), strategy, BacktestConfig(
        execution_mode=timing,
        execution_price=ExecutionPrice.CLOSE if timing is ExecutionMode.SAME_BAR else ExecutionPrice.OPEN,
    ))
    result = engine.run()
    fills = result.to_fills_dataframe()
    if fills.height:
        fill = result.fills[0]
        print(f"{name}: {fill.timestamp:%Y-%m-%d} @ ${fill.price:.2f}")
    else:
        print(f"{name}: no fill; pending={len(engine.broker.get_pending_orders())}")
```

<!-- ml4t-doc-output: tutorial-orders-timing -->
```text
ml4t-backtest {package_version}
next market: 2024-01-03 @ $188.37
same market: 2024-01-02 @ $188.00
next limit: 2024-01-03 @ $188.50
next stop: 2024-01-03 @ $191.00
unfilled limit: no fill; pending=1
```

The next-bar market order was submitted on January 2 and filled at January 3's
open. The same-bar run uses `ExecutionPrice.CLOSE`, so its fill is at January
2's close. The strategy does not inspect the bar price before submitting; the
order is fixed in advance. If a decision depends on the completed bar's close,
use next-bar execution. A close-conditioned signal cannot trade at an earlier
open or assume it joined the already completed closing auction.

The buy limit at $188.50 and buy stop at $191.00 both reached their trigger
on the next bar and filled at their declared levels. The $170 limit never
traded and remains pending at the end of the input. `result.to_fills_dataframe()`
contains only executed orders; `engine.broker.get_pending_orders()` shows the
unfilled instruction. The [Order Types](../user-guide/orders.md) guide lists
all supported order fields, and [Execution Semantics](../user-guide/execution-semantics.md)
describes processing order and price selection.

## In the book

Chapter 16, Section 16.3, [Vectorized and event-driven backtesting](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/16_strategy_simulation/README.md),
and [notebook 04, Single Asset Backtest with ml4t-backtest](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/16_strategy_simulation/04_single_asset_ml4t_backtest.ipynb)
apply order timing to a stateful RSI example with explicit costs.
