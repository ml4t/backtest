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
| Ingest a data bundle, then select assets with `symbol()` | Prepare a Polars panel with `timestamp`, `asset`, and `close`; pass it to [`DataFeed`](data-feed.md) | Map each Zipline asset ID and valid dates to stable feed IDs; Data's registry can supply native runtime lookup. Prepare causal adjusted history in Data, retain the declared execution basis, and check the account-action scope below. `DataFeed` does not ingest a bundle. |
| Define `initialize(context)` and `handle_data(context, data)` | Use the [callback adapter](#run-original-callback-signatures), or subclass [`Strategy`](strategies.md) and decide in `on_data(timestamp, data, context, broker)` | `on_data` receives a mapping of assets present at that event. State belongs on the strategy instance. `on_start` and `on_prepare` run before the first feed bar; neither receives future bars. |
| Read `data.current(asset, "price")` and `data.history(...)` | Use [`RuntimeServices`](strategies.md#use-causal-history-and-session-callbacks) for causal Polars reads, or the [Pandas read adapter](#preserve-zipline-read-shapes) | Declare the grid, availability, adjustment basis and stale-read policy. Prepared features can also enter through `signals_df`; their decision and publication clocks remain explicit. |
| Call `order(asset, quantity)` or `order_target_percent(asset, weight)` | Call [`broker.submit_order`](../api/index.md#ml4t.backtest.broker.Broker.submit_order) or [`broker.order_target_percent`](../api/index.md#ml4t.backtest.broker.Broker.order_target_percent) inside `on_data` | The asset is a string identifier. Check share precision, pending orders, buying power, and next eligible fill; identical method names do not imply identical execution. |
| Use `schedule_function` with date and time rules | Register native callbacks through [`RuntimeServices.scheduler`](strategies.md#use-causal-history-and-session-callbacks), or use [`RebalanceSchedule`](rebalancing.md#schedule-metadata) for rebalancing | Match phase, calendar and offset. Pre-open callbacks have an explicit clock; other scheduled callbacks require a real observed bar at their requested instant. |
| Set commission and slippage models | Set explicit [`BacktestConfig`](configuration.md) costs and, where needed, market impact and funding inputs | Default ML4T examples charge no commission or slippage. Reconcile cost amounts separately from fills and share quantities. |
| Call `record(...)` and inspect the performance frame | Call [`broker.record`](results.md#recorded-strategy-values); inspect native account, holdings, fills and recorded-value exports | Recorded values have their own bar-end component and do not overwrite account fields. The callback adapter projects recorded values beside native cash and portfolio value; the full native result remains available on the frame. |
| Run `zipline run` or `run_algorithm(...)` | Use the [bundle-free callback adapter](#run-original-callback-signatures), or construct [`Engine`](../api/index.md#ml4t.backtest.engine.Engine) and call `run()` | A run uses prepared inputs and one `Engine` instance. Create a new engine for another scenario. |

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
        calendar=None,
        initial_cash=100_000,
        execution_mode=ExecutionMode.NEXT_BAR,
        share_type=ShareType.INTEGER,
        commission_type=CommissionType.NONE,
        slippage_type=SlippageType.NONE,
    ),
).run()
assert result.config.resolved_calendar is None
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

If you need portfolio analysis after migration, evaluate the optional
[diagnostic handoff](../tutorials/diagnostic-handoff.md). Its separate checked
example passes a `BacktestResult` to `ml4t-diagnostic==0.1.4` and computes
return and drawdown statistics.

## Preserve Zipline read shapes

The upcoming Backtest release includes `compat.zipline.assets.Asset` and
`compat.zipline.data.BarData` for Pandas read containers inside a native
`Strategy`. Supply the stable account key and any actual legacy SID explicitly;
these containers do not resolve symbols or ingest a bundle. Native dated-symbol
resolution is [documented separately](strategies.md#resolve-dated-instruments-and-enforce-their-lifetimes).
The [callback adapter](#run-original-callback-signatures) also uses these read containers.

The read-container fixtures use the published
[Zipline Reloaded 3.2.0](https://pypi.org/project/zipline-reloaded/3.2.0/)
CPython 3.12 Linux wheel with Pandas 2.3.3, SHA-256
`1d7a3c4f547c02e9a3c3cc1f8ea2394641124c1fdb35ef27cdf30fa488d3899b`.
Real daily and minute `run_algorithm` probes cover scalar/sequence selections,
reversed asset and field order, numeric dtypes and datetime indexes. This is a
read-container comparison, separate from the retained trading and timing reports.

| Requested assets / fields | `current` | `history` |
|---|---|---|
| One / one | Scalar | Series, named by asset |
| One / several | Series, named by symbol | MultiIndex DataFrame |
| Several / one | Series, indexed by assets | DataFrame, asset columns |
| Several / several | DataFrame, asset rows | MultiIndex DataFrame |

Published 3.2.0 returns the MultiIndex for a single asset with multiple fields,
although its prose describes a simple datetime-indexed frame. It also leaves
the second index level unnamed. The adapter preserves the actual output:
`["date", None]` for daily history and `["date_time", None]` for minute history.
Multi-field history sorts its asset index by supplied legacy SID (otherwise native key); single-field
history and current reads retain the requested order. History numeric columns
are Float64. With a declared calendar, daily indexes are naive session dates;
minute indexes are UTC instants. Both use nanosecond datetime units.

<!-- ml4t-doc-test: zipline-read-shapes -->
```python
from datetime import UTC, datetime
import polars as pl
from ml4t.specs import FeedSpec
from ml4t.backtest import BacktestConfig, DataFeed, Engine, RuntimeServices, Strategy
from ml4t.backtest.compat.zipline.assets import Asset
from ml4t.backtest.compat.zipline.data import BarData

asset = Asset("A", symbol="ALPHA", legacy_sid=0)
dates = [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [6, 7]]
spec = FeedSpec(time_contract_version=2, calendar="NYSE", data_frequency="1d",
                timestamp_semantics="bar_close", bar_grid_anchor="session_open",
                bar_label="right", partial_bar_policy="include")
prices = pl.DataFrame({"timestamp": dates, "asset": ["A"] * 2,
                       "close": [100.0, 101.0], "volume": [1000, 2000]})
services = RuntimeServices()

class ReadPrices(Strategy):
    def on_data(self, timestamp, data, context, broker):
        if timestamp == dates[-1]:
            reader = BarData(services)
            assert reader.current(asset, "price") == 101.0
            history = reader.history(asset, ["price", "volume"], 2, "1d")
            assert history.index.names == ["date", None]
            assert history["price"].to_list() == [100.0, 101.0]
            assert str(history["volume"].dtype) == "float64"

result = Engine(DataFeed(prices_df=prices, feed_spec=spec), ReadPrices(),
                BacktestConfig(calendar="NYSE"), runtime=services).run()
assert result.to_fills_dataframe().is_empty()
print("Zipline read containers preserve the native causal history.")
```

History requests require `"1d"` or `"1m"` matching the declared source grid,
with OHLCV/price fields. With a declared calendar, the default uses native expected-bar windows.
Explicit `calendar=None` selects observation windows and retains physical UTC
indexes for daily and minute reads. An explicit expected-bar request still
requires a calendar. Requesting before captured source coverage raises and
requires warmup bars. Cross-frequency
aggregation belongs in Data preparation. `history_view="prepared"` selects the
[Data-prepared view](strategies.md#read-data-prepared-adjusted-history); it does
not adjust executable prices. Asset, action and futures-account migration still
requires the separate contracts below.

Reads follow native causal visibility and explicit stale-view policies. In
particular, a pre-open current read reports missing values rather than exposing
a future daily close. The native Polars APIs remain available when a strategy
needs their window and provenance fields. An `Asset` compares by its stable
native key; use `int(asset)` only when an actual legacy integer SID was supplied.

## Match ledger and corporate-action assumptions

`Engine` does not automatically turn arbitrary `split_ratio` or `cash_dividend`
columns into share adjustments or cash payments. Prepare the input price and
share-unit basis explicitly upstream. Adjusted bars alone do not reproduce an
unadjusted ledger's dividend entitlement, payment date or split-adjusted share
count. Keep an action-sensitive migration on the reference engine until its
required event workflow is supported and compared.

Contract multipliers and fixed margin schedules support contract P&L and
buying-power checks. Backtest's current leveraged ledger still moves full trade
notional through cash and values the position at notional. It does not implement
Zipline's futures variation-cash ledger. With $100,000 initial cash, buying one
contract at 4,000 with a $50 multiplier moves $200,000 from cash into the
position. A later 10-point rise adds $500 to equity; matching that equity does
not establish cash-ledger parity. See [account policies](accounts.md).

The retained CME workload contains continuous root series without a dated
contract chain and roll map, so its Zipline comparison is explicitly unsupported.
That input limitation is separate from cash-settlement semantics. The supported
ETF and US-equity comparisons do not establish universal futures migration.

## Check a migrated run

The retained comparisons pin [Zipline Reloaded 3.1.1](https://pypi.org/project/zipline-reloaded/3.1.1/),
using the source distribution SHA-256
`4a305524616f7aad836f929e5a2ba5afc7db0e238757f47eb49487d9e2457a6f`.
The [framework manifest](https://github.com/ml4t/backtest/blob/dea188c6ecf832a4adeeef4e211489c5dd424b55/validation/framework_targets.toml)
records that identity. The [real-strategy report](https://github.com/ml4t/backtest/blob/dea188c6ecf832a4adeeef4e211489c5dd424b55/validation/REAL_STRATEGY_RESULTS.json)
covers ETF allocation and a US-equity panel under the frozen zero-cost protocol;
the [synthetic report](https://github.com/ml4t/backtest/blob/dea188c6ecf832a4adeeef4e211489c5dd424b55/validation/CORRECTNESS_RESULTS.json)
and [Profiles](profiles.md) state the remaining supported and excluded surfaces.
These comparisons do not establish parity for arbitrary bundles, corporate
actions, history queries or scheduling rules.

1. Export Zipline's ordered transactions and portfolio values. Preserve the
   package version, bundle ingestion timestamp, asset IDs and lifetimes,
   calendar, initial cash, and cost and order settings with that output.
2. Prepare the same asset set and OHLCV history for `DataFeed`. Keep a
   source-ID-to-feed-ID mapping and match the source run's price adjustment
   basis. State each timestamp's timezone and whether it labels a bar open or
   close. Reconcile splits, dividend cash effects, missing bars, and exchange
   holidays before comparing trades.
3. If the source uses `data.history`, use the native causal read services or the
   read adapter below; use Engineer for prepared feature panels. If it uses
   `schedule_function`, map its date and time rules to native session callbacks
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

## Run original callback signatures

In a development build containing the callback adapter, import `run_algorithm`
and API names from `ml4t.backtest.compat.zipline`. This API is not part of the
pending 0.1.13 release candidate. Pass prepared prices and an explicit mapping
from native account keys to display symbols and optional original SIDs. The
adapter executes callbacks through the native Engine; it does not discover a
Zipline bundle or rewrite imports in user source.

This example preserves the `initialize`, `handle_data` and `analyze` signatures.
Change the imports in your strategy to the migration namespace. Default adapter
execution is next-bar close with integer shares, FIFO processing, shorting and
leverage enabled, and unconstrained cash admission as in the reference workflow.
Default costs are per-share 0.001, minimum zero, with quadratic volume impact
0.1 and an asset/bar participation cap of 0.025. These adapter choices leave
native Engine defaults unchanged. Supply actual positive observed volume;
missing or zero volume prevents fills, including with `NoSlippage`.

<!-- ml4t-doc-test: zipline-callback-runner -->
```python
from datetime import UTC, datetime

import polars as pl
from ml4t.backtest import DataFeed
from ml4t.backtest.compat.zipline import run_algorithm
from ml4t.backtest.compat.zipline.api import order, record, set_commission, set_slippage, symbol
from ml4t.backtest.compat.zipline.assets import Asset
from ml4t.backtest.compat.zipline.finance import commission, slippage


def initialize(context):
    context.asset = symbol("ALPHA")
    context.count = 0
    set_commission(commission.PerShare(cost=0.1, min_trade_cost=1.0))
    set_slippage(slippage.VolumeShareSlippage(volume_limit=1.0, price_impact=0.0))


def handle_data(context, data):
    if context.count == 0:
        order(context.asset, 10)
    record(observed_price=data.current(context.asset, "price"))
    context.count += 1


def analyze(context, performance):
    assert context.portfolio.positions[context.asset].amount == 10
    assert performance.portfolio_value.tolist() == [10000.0, 9999.0, 10099.0]


feed = DataFeed(
    prices_df=pl.DataFrame({
        "timestamp": [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [6, 7, 8]],
        "asset": ["A"] * 3,
        "close": [100.0, 110.0, 120.0],
        "volume": [1000] * 3,
    }),
    feed_spec={"time_contract_version": 2, "timestamp_semantics": "event_time", "calendar": None},
)
performance = run_algorithm(
    feed=feed, assets=[Asset("A", symbol="ALPHA", legacy_sid=17)],
    initialize=initialize, handle_data=handle_data, analyze=analyze, capital_base=10000,
)
native = performance.attrs["backtest_result"]
assert [(fill.quantity, fill.price, fill.commission) for fill in native.fills] == [(10, 110, 1)]
assert performance.observed_price.tolist() == [100, 110, 120]
print("native callback run: 10 shares, one fee, all three account observations")
```

<!-- ml4t-doc-output: zipline-callback-runner -->
```text
native callback run: 10 shares, one fee, all three account observations
```

The returned Pandas frame contains native cash, portfolio value, returns and recorded
variables at actual account observations, preserving timestamp precision. Names
that collide with those account columns are rejected. Access native holdings,
fills, returns, configuration and versioned artifacts through
`performance.attrs["backtest_result"]`. The adapter does not reconstruct NAV
from completed trades. Account and position objects read the native broker;
position `cost_basis` projects its explicit commission-adjusted reporting basis.

The preserved OLMAR fixture from published Zipline Reloaded 3.2.0 is exercised
in `tests/test_zipline_callback_runner.py`. With seven explicit assets and 13
shared daily sessions, its weights, quantities, basis, cash, position value,
NAV and 47 fills match the reference. This is a correctness comparison for that
input and its declared costs, not a performance measurement or a claim that
all Zipline workflows are supported.

This initial adapter does not yet implement `schedule_function`,
the original Pipeline construction API, `fetch_csv`, benchmark
return inputs, raw corporate actions or futures variation-cash settlement.
Native scheduler and prepared feature/history routes remain available as
shown above. `set_benchmark(None)` explicitly disables a benchmark; omission
records it as not supplied. An asset benchmark currently needs a separately
prepared workflow. No implicit SPY download or synthetic benchmark return is
created. Pass physical run bounds and warmup through `RuntimeServices`; do not
reinterpret session labels as event instants.

### Quantity, value and target orders

The callback API and context expose `order`, `order_value`, `order_percent`,
`order_target`, `order_target_value` and `order_target_percent`. They use native
scalar sizing and order submission. Value orders divide by the last observed
native price and contract multiplier. Targets subtract held quantity before
rounding their signed delta. Integer amounts within 0.0001 of an integer snap
to it; other amounts truncate toward zero. Targets deliberately ignore pending
orders, so submitting the same target twice can create two orders. This is not
a batch weight rebalance or a guarantee that the target is affordable.

Pass explicit `limit_price`/`stop_price`, or a `MarketOrder`, `LimitOrder`,
`StopOrder` or `StopLimitOrder` from
`ml4t.backtest.compat.zipline.finance.execution`. Native trigger state,
next-bar eligibility, volume limits and final slipped-price limit checks still
apply. Do not pass a style together with separate price arguments.

The frame's `transactions` column groups actual native fills by observation.
It supplies mapped assets, signed amounts, actual price, timestamp, native order
ID and actual fill commission. Zipline's transaction records may leave commission
null; this projection reports the native fill fee. The authoritative fill table
remains in the attached native result. No order or transaction is fabricated to
fill a missing observation.
