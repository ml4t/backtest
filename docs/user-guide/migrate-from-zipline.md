# Migrate a Zipline strategy

Move one strategy at a time. Keep its input bars, asset identities, decision
times, order sizes, and account assumptions fixed while you compare fills and
equity. The [example below](#run-the-mapped-strategy) shows the Backtest side
of that migration with bundled synthetic bars. It runs from an installed
`ml4t-backtest` wheel without Zipline or a data service. Its output is a
Backtest check, not a comparison with a Zipline run.

Before using your own strategy, record the Zipline version, bundle name and
ingestion timestamp, trading calendar, data frequency, initial cash, commission
and slippage models, and asset IDs. Zipline's [bundle documentation](https://zipline.ml4trading.io/bundles.html)
describes the price, adjustment, and asset metadata stored together. `DataFeed`
accepts prepared rows; it does not read that bundle or reproduce its asset
lookup and adjustment rules.

## Map the work, not just the function names

| Zipline Reloaded task | ML4T Backtest task | Difference to check |
|---|---|---|
| Ingest a data bundle, then select assets with `symbol()` | Prepare a Polars panel with `timestamp`, `asset`, and `close`; pass it to [`DataFeed`](data-feed.md) | Map each Zipline asset ID and its valid dates to a stable feed identifier. Match the source price adjustment basis and account for corporate actions upstream. `DataFeed` does not ingest the bundle or perform that mapping. |
| Define `initialize(context)` and `handle_data(context, data)` | Subclass [`Strategy`](strategies.md), initialize state in `__init__` or `on_start`, and decide in `on_data(timestamp, data, context, broker)` | `on_data` receives a mapping of assets present at that event. State belongs on the strategy instance. `on_start` and `on_prepare` run before the first feed bar; neither receives future bars. |
| Read `data.current(asset, "price")` and `data.history(...)` | Read the current asset's bar in `data[asset]`; precompute rolling features and supply them through `signals_df` | A `DataFeed` callback does not expose Zipline's rolling `data.history` API. Align each feature with the bar on which it becomes available. |
| Call `order(asset, quantity)` or `order_target_percent(asset, weight)` | Call [`broker.submit_order`](../api/index.md#ml4t.backtest.broker.Broker.submit_order) or [`broker.order_target_percent`](../api/index.md#ml4t.backtest.broker.Broker.order_target_percent) inside `on_data` | The asset is a string identifier. Check share precision, pending orders, buying power, and next eligible fill; identical method names do not imply identical execution. |
| Use `schedule_function` with date and time rules | Decide in `on_data`, or use [`RebalanceSchedule`](rebalancing.md#schedule-metadata) with `TargetWeightExecutor` for session-based rebalancing | Match the source decision instant and calendar explicitly. For assets with different close times, `session_col` requires one bar per asset per decision session and next-bar execution. |
| Set commission and slippage models | Set explicit [`BacktestConfig`](configuration.md) costs and, where needed, market impact and funding inputs | Default ML4T examples charge no commission or slippage. Reconcile cost amounts separately from fills and share quantities. |
| Call `record(...)` and inspect the performance frame | Keep custom observations on the strategy instance; inspect [`BacktestResult`](results.md) fills, trades, equity, and exported frames | There is no `record` call with Zipline's performance-frame contract. Join your observations to result timestamps explicitly. |
| Run `zipline run` or `run_algorithm(...)` | Construct [`Engine`](../api/index.md#ml4t.backtest.engine.Engine) and call `run()`, or use `run_backtest(...)` | A run uses prepared inputs and one `Engine` instance. Create a new engine for another scenario. |

Zipline's [tutorial](https://zipline.ml4trading.io/beginner-tutorial) documents
its callback, bundle, order, history, and `record` workflow. Its
[API reference](https://zipline.ml4trading.io/api-reference.html) defines
`order_target_percent` and `schedule_function`. The mapping above names
corresponding tasks; it does not promise framework equivalence.

## Run the mapped strategy

Follow [installation](../getting-started/installation.md), save the Python block
as `migrate_from_zipline.py`, and run `python migrate_from_zipline.py`. It uses
nine synthetic AAPL OHLCV rows from the installed wheel. Their naive 16:00
timestamps label daily bar closes; this example applies no exchange calendar,
timezone conversion, split, or dividend. The cash account starts with $100,000,
uses integer shares, and charges no commission or slippage. `NEXT_BAR` market
orders fill at the next AAPL bar's open. These settings produce 53 shares from
the first 10% target; the fifth bar targets zero.

<!-- ml4t-doc-test: migration-zipline-target -->
```python
from importlib.metadata import version

import polars as pl
from ml4t.backtest import BacktestConfig, CommissionType, DataFeed, Engine, ExecutionMode, Strategy
from ml4t.backtest.config import ShareType, SlippageType
from ml4t.backtest.example_data import load_example_prices


class TargetStrategy(Strategy):
    def __init__(self):
        self.asset_bars = 0
        self.decisions = []

    def on_data(self, timestamp, data, context, broker):
        if "AAPL" not in data:
            return
        self.asset_bars += 1
        if self.asset_bars == 1:
            self.decisions.append((timestamp, 0.10))
            broker.order_target_percent("AAPL", 0.10)
        elif self.asset_bars == 5:
            self.decisions.append((timestamp, 0.0))
            broker.order_target_percent("AAPL", 0.0)


prices = load_example_prices("equity").filter(pl.col("asset") == "AAPL")
strategy = TargetStrategy()
result = Engine(
    DataFeed(prices_df=prices),
    strategy,
    BacktestConfig(
        initial_cash=100_000,
        execution_mode=ExecutionMode.NEXT_BAR,
        share_type=ShareType.INTEGER,
        commission_type=CommissionType.NONE,
        slippage_type=SlippageType.NONE,
    ),
).run()
print("ml4t-backtest " + version("ml4t-backtest"))
print(f"input: {prices.height} synthetic AAPL bars")
for timestamp, target in strategy.decisions:
    print(f"{timestamp:%Y-%m-%d} target {target:.0%}")
for fill in result.fills:
    print(f"{fill.timestamp:%Y-%m-%d} {fill.side.value} {fill.quantity:g} @ ${fill.price:.2f}")
print(f"closed trades: {len(result.to_trades_dataframe())}")
print(f"final equity: ${result.metrics['final_value']:.2f}")
```

<!-- ml4t-doc-output: migration-zipline-target -->
```text
ml4t-backtest {package_version}
input: 9 synthetic AAPL bars
2024-01-02 target 10%
2024-01-08 target 0%
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

1. Export Zipline's ordered transactions and portfolio values. Preserve the
   package version, bundle ingestion timestamp, asset IDs and lifetimes,
   calendar, initial cash, and cost and order settings with that output.
2. Prepare the same asset set and OHLCV history for `DataFeed`. Keep a
   source-ID-to-feed-ID mapping and match the source run's price adjustment
   basis. State each timestamp's timezone and whether it labels a bar open or
   close. Reconcile splits, dividend cash effects, missing bars, and exchange
   holidays before comparing trades.
3. If the source uses `data.history`, compute each rolling input from bars
   available at its decision time and pass it through `signals_df`. If it uses
   `schedule_function`, map its date and time rules to actual feed event times
   and explicit [schedule metadata](rebalancing.md#schedule-metadata).
4. Start with one asset and one order. Compare decision and submission times,
   fill times, quantities, prices, cash, and equity. Add target sizing, costs,
   and multi-asset sessions separately. Record every setting changed between
   comparison runs.
5. Use the [`zipline` profile](profiles.md) only for the library's documented
   comparison protocol. It is not a substitute for recording the original
   Zipline run's native settings.

The [engine-divergence notebook](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/16_strategy_simulation/07_engine_divergence_anatomy.ipynb)
calls `ml4t-backtest` with controlled setting changes. The [Book Guide](../book-guide/index.md)
identifies the role of other linked notebooks. For existing framework evidence,
see [Profiles](profiles.md); its supported workload and cost boundaries apply.
