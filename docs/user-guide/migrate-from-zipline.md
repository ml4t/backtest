# Migrate a Zipline strategy

Move one strategy at a time. Start with the same prepared bars, decision times,
order sizes, and account assumptions, then compare fills and equity. The
[complete example below](#run-the-mapped-strategy) runs from the installed
`ml4t-backtest` wheel with bundled synthetic bars. It does not require a Zipline
installation or a data service.

## Map the work, not just the function names

| Zipline Reloaded task | ML4T Backtest task | Difference to check |
|---|---|---|
| Ingest a data bundle, then select assets with `symbol()` | Prepare a Polars panel with `timestamp`, `asset`, and `close`; pass it to [`DataFeed`](data-feed.md) | `DataFeed` does not ingest a Zipline bundle or apply its asset metadata and corporate-action adjustments. Prepare and validate those inputs before the run. |
| Define `initialize(context)` and `handle_data(context, data)` | Subclass [`Strategy`](strategies.md), initialize state in `__init__` or `on_start`, and decide in `on_data(timestamp, data, context, broker)` | `on_data` receives a mapping of the assets present at that event. Strategy state belongs on the instance. `on_start` runs before any bar; `on_prepare` is reserved for causally available preopen decisions. |
| Read `data.current(asset, "price")` and `data.history(...)` | Read the current asset's bar in `data[asset]`; precompute rolling features and supply them through `signals_df` | A `DataFeed` callback does not expose Zipline's rolling `data.history` API. Align each feature with the bar on which it becomes available. |
| Call `order(asset, quantity)` or `order_target_percent(asset, weight)` | Call [`broker.submit_order`](../api/index.md#ml4t.backtest.broker.Broker.submit_order) or [`broker.order_target_percent`](../api/index.md#ml4t.backtest.broker.Broker.order_target_percent) inside `on_data` | The asset is a string identifier. Check share precision, pending orders, buying power, and next eligible fill; identical method names do not imply identical execution. |
| Use `schedule_function` with date and time rules | Decide in `on_data`, or use [`RebalanceSchedule`](rebalancing.md#schedule-metadata) with `TargetWeightExecutor` for session-based rebalancing | The scheduling APIs are different. Supply calendar and session metadata explicitly; for assets with different close times, use `session_col` and next-bar execution. |
| Set commission and slippage models | Set explicit [`BacktestConfig`](configuration.md) costs and, where needed, market impact and funding inputs | Default ML4T examples charge no commission or slippage. Reconcile cost amounts separately from fills and share quantities. |
| Call `record(...)` and inspect the performance frame | Keep custom observations on the strategy instance; inspect [`BacktestResult`](results.md) fills, trades, equity, and exported frames | There is no `record` call with Zipline's performance-frame contract. Join your observations to result timestamps explicitly. |
| Run `zipline run` or `run_algorithm(...)` | Construct [`Engine`](../api/index.md#ml4t.backtest.engine.Engine) and call `run()`, or use `run_backtest(...)` | A run uses prepared inputs and one `Engine` instance. Create a new engine for another scenario. |

Zipline's [tutorial](https://zipline.ml4trading.io/beginner-tutorial) documents
its callback, bundle, order, history, and `record` workflow. Its
[API reference](https://zipline.ml4trading.io/api-reference.html) defines
`order_target_percent` and `schedule_function`. The mapping above names
corresponding tasks; it does not promise framework equivalence.

## Run the mapped strategy

This strategy targets 10% AAPL exposure at its first daily bar and zero at its
fifth. The nine AAPL bars are synthetic. With integer shares and $100,000
starting cash, the first target becomes 53 shares. The default `NEXT_BAR`
execution mode fills market orders at the next asset bar's open. No cost,
slippage, corporate action, or live market behavior is modeled in this example.

<!-- ml4t-doc-test: migration-zipline-target -->
```python
from importlib.metadata import version

import polars as pl
from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.example_data import load_example_prices


class TargetStrategy(Strategy):
    def __init__(self):
        self.asset_bars = 0

    def on_data(self, timestamp, data, context, broker):
        if "AAPL" not in data:
            return
        self.asset_bars += 1
        if self.asset_bars == 1:
            broker.order_target_percent("AAPL", 0.10)
        elif self.asset_bars == 5:
            broker.order_target_percent("AAPL", 0.0)


prices = load_example_prices("equity").filter(pl.col("asset") == "AAPL")
result = Engine(
    DataFeed(prices_df=prices), TargetStrategy(), BacktestConfig(initial_cash=100_000)
).run()
print("ml4t-backtest " + version("ml4t-backtest"))
for fill in result.fills:
    print(f"{fill.timestamp:%Y-%m-%d} {fill.side.value} {fill.quantity:g} @ ${fill.price:.2f}")
print(f"closed trades: {len(result.to_trades_dataframe())}")
print(f"final equity: ${result.metrics['final_value']:.2f}")
```

<!-- ml4t-doc-output: migration-zipline-target -->
```text
ml4t-backtest {package_version}
2024-01-03 buy 53 @ $188.37
2024-01-09 sell 53 @ $191.37
closed trades: 1
final equity: $100159.00
```

The January 2 and January 8 decisions fill on January 3 and January 9. The
$159 gain is 53 shares times the $3 change between the two fill prices. Check
[`to_fills_dataframe()`](../api/index.md#ml4t.backtest.result.BacktestResult.to_fills_dataframe)
and [`to_equity_dataframe()`](../api/index.md#ml4t.backtest.result.BacktestResult.to_equity_dataframe)
when migrating a real strategy; a matching final value alone can hide different
orders or interim exposure.

## Check a migrated run

1. Export Zipline's ordered transactions and portfolio values from the original
   run. Preserve its data bundle, calendar, cost models, and package version.
2. Prepare the same adjusted price history and asset set for `DataFeed`. State
   each timestamp's timezone and whether it represents a bar open or close.
3. Start with one asset and one order. Compare submission times, fill times,
   quantities, fill prices, cash, and equity. Add target sizing, costs, and
   multi-asset sessions separately.
4. Use the [`zipline` profile](profiles.md) only for the library's documented
   comparison protocol. It is not a substitute for recording the original
   Zipline run's native settings.

The [engine-divergence notebook](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/16_strategy_simulation/07_engine_divergence_anatomy.ipynb)
calls `ml4t-backtest` with controlled setting changes. The [Book Guide](../book-guide/index.md)
identifies the role of other linked notebooks. For existing framework evidence,
see [Profiles](profiles.md); its supported workload and cost boundaries apply.
