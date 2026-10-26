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
[diagnostic handoff](../tutorials/diagnostic-handoff.md). Its checked examples
cover return statistics and observed holdings. The complete Data-to-Diagnostic route is documented in the
[versioned offline workflow](offline-migration.md).

## Preserve Zipline read shapes

Backtest 0.2.0 includes `compat.zipline.assets.Asset` and
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
requires warmup bars. For a separate prepared grid, use the
[native read-source route](#use-a-separate-prepared-history-grid). Cross-frequency
aggregation belongs in Data preparation. `history_view="prepared"` selects the
[Data-prepared view](strategies.md#read-data-prepared-adjusted-history); it does
not adjust executable prices. Asset, action and futures-account migration still
requires the separate contracts below.

Native current reads at a pre-open clock contain no current quote. The
callback adapter can return a known price and a completed-bar projection;
neither reads a future daily close. The native Polars APIs retain the window
and provenance fields. An `Asset` compares by its stable
native key; use `int(asset)` only when an actual legacy integer SID was supplied.

## Match ledger and corporate-action assumptions

Pass explicit raw event facts through `DataFeed(corporate_actions=...,
input_manifest=...)`. The [corporate-action guide](corporate-actions.md) runs a
split and delayed dividend payment, specifies event clocks and input basis, and
documents the tested `zipline_3_2` method. Arbitrary event columns in a price frame
do not activate accounting. Adjusted bars alone do not reproduce raw share counts
or payment entitlements. Match your event terms and method to the tested scope.

The native default remains signed-notional cash accounting. For migrated futures,
declare `ContractSpec(accounting_method="futures_variation", settlement_policy="on_mark")`.
The [futures-accounting guide](futures-accounting.md) runs callback orders through
the native variation ledger. Its captured original 3.2.0 scenarios compare cash,
NAV, basis, positions and fills, including fees and reversals. Scheduled exchange
settlement is a separate explicit policy, with independent dollar oracles.
Margin constrains entry; it does not select cash accounting or payout frequency.

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

In Backtest 0.2.0, import `run_algorithm` and API names from
`ml4t.backtest.compat.zipline`. Pass prepared prices and an explicit mapping
from native account keys to display symbols and optional original SIDs. The
adapter executes callbacks through the native Engine; it does not discover a
Zipline bundle or rewrite imports in user source.

This example preserves the `initialize`, `handle_data` and `analyze` signatures.
Change the imports in your strategy to the migration namespace. Default adapter
execution is next-bar close with integer shares, FIFO processing, shorting and
leverage enabled, and unconstrained cash admission as in the reference workflow.
Default equity costs match published Zipline Reloaded 3.2.0: per-share 0.001,
minimum zero, fixed 5-basis-point slippage and a shared asset/bar volume cap
of 0.1. Select `VolumeShareSlippage` explicitly for quadratic impact. These adapter choices leave
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
        context.order_id = order(context.asset, 10)
    assert context.get_order(context.order_id).filled == (0 if context.count == 0 else 10)
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

This initial adapter does not yet implement
the original Pipeline construction API. Use the [prepared panel route](#deliver-prepared-pipeline-results)
for factor delivery and the [frozen CSV route](#register-an-offline-csv-source)
for auxiliary inputs. [Raw corporate actions](corporate-actions.md) use explicit
event inputs; [futures variation](futures-accounting.md) requires an explicit
per-instrument method and payout policy.
Native scheduler and prepared feature/history routes remain available as
shown above. `set_benchmark(None)` explicitly disables a benchmark; omission
records it as not supplied unless prepared benchmark data is passed. An asset
benchmark uses the [declared return input](#supply-a-declared-benchmark).
No implicit SPY download or synthetic benchmark return is
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
It supplies mapped assets, signed amounts, actual price, timestamp, a scoped migration order
ID, its original `native_order_id` and actual fill commission. Zipline's transaction records may leave commission
null; this projection reports the native fill fee. The authoritative fill table
remains in the attached native result. No order or transaction is fabricated to
fill a missing observation.

### Inspect and cancel orders

`get_order(order_id)` returns a detached order view, or `None` for an unknown
identifier. It preserves original signed `amount`, cumulative signed `filled`,
cumulative `commission`, creation and last-activity clocks, and native limit,
stop and captured stop-trigger state. A partial fill keeps status
`finance.order.ORDER_STATUS.OPEN`; completion and cancellation have their own
statuses. Native remaining quantity is not substituted for original amount.
The view does not synthesize a `limit_reached` flag that the native order has
not recorded.

`get_open_orders()` returns an asset-to-list mapping; passing one mapped Asset
returns its list. These containers are detached. `cancel_order` accepts the
migration string ID or an order view and delegates to native cancellation.
Cancellation is allowed during market callbacks, not after simulation in
`analyze`. Unknown IDs have no effect.

Migration order IDs include a run identifier. An ID retained from an earlier
run cannot refer to a later run's local `ORD-1`. Use `native_order_id` when
joining the projected order or transaction to the attached native result.
Native IDs and their serialization are unchanged. Native strategies can call
`broker.get_fills(order_id)` to obtain detached actual fill records directly.

### Schedule original callbacks on declared sessions

Register `schedule_function` during `initialize` using `date_rules.every_day`,
`week_start`, `week_end`, `month_start` or `month_end`. Week and month offsets
count trading sessions in the full calendar period, including sessions before
the supplied feed starts. `half_days=False` suppresses callbacks on an early
close identified by the calendar; it does not move them to a different day.

Daily runs ignore the intraday `time_rule` and execute selected callbacks after
`handle_data`, in registration order, before the account observation. Minute
runs require explicitly right-labelled minute observations. `market_open()`
means one minute after open; `market_open(minutes=30)` means 30 minutes after
open. `market_close()` means one minute before close. With no time rule, or
with `every_minute()`, a scheduled function runs once per selected session at
its first observed regular minute, following the reference's once-per-day
wrapper. The native `handle_data` callback still runs at every observed event.
Missing endpoints do not produce artificial bars or callback executions.

Pass `before_trading_start(context, data)` to the runner for a native PRE callback
45 minutes before the declared session open. This matches the tested NYSE
clock. Other calendars may differ from a legacy fixed wall-clock or restricted
execution window; compare those clocks explicitly. Raw same-day completed prices
are unavailable then. All order helper variants are disallowed in this callback.
`calendar=None` disables session scheduling and rejects these session callbacks;
it does not restore NYSE implicitly. Failed or reused runs remove registrations
owned by the adapter.

<!-- ml4t-doc-test: zipline-scheduled-callbacks -->
```python
from datetime import UTC, datetime

import polars as pl
from ml4t.backtest import DataFeed
from ml4t.backtest.compat.zipline import run_algorithm
from ml4t.backtest.compat.zipline.api import date_rules, get_datetime, record, schedule_function
from ml4t.backtest.compat.zipline.assets import Asset

trace = []


def scheduled(context, data):
    trace.append(get_datetime().isoformat())
    record(selected_close=data.current(context.asset, "price"))


def initialize(context):
    context.asset = context.symbol("ALPHA")
    schedule_function(scheduled, date_rules.every_day())


feed = DataFeed(
    prices_df=pl.DataFrame({
        "timestamp": [datetime(2025, 7, 2, 20, tzinfo=UTC), datetime(2025, 7, 3, 17, tzinfo=UTC)],
        "asset": ["A", "A"], "close": [100.0, 101.0], "volume": [1000, 1000],
    }),
    feed_spec={"time_contract_version": 2, "calendar": "NYSE", "data_frequency": "1d",
               "timestamp_semantics": "bar_close", "bar_label": "right",
               "bar_grid_anchor": "session_open", "partial_bar_policy": "include"},
)
performance = run_algorithm(feed=feed, assets=[Asset("A", "ALPHA")], initialize=initialize)
assert trace == ["2025-07-02T20:00:00+00:00", "2025-07-03T17:00:00+00:00"]
assert performance.selected_close.tolist() == [100, 101]
print("daily callbacks include the observed early-close session")
```

<!-- ml4t-doc-output: zipline-scheduled-callbacks -->
```text
daily callbacks include the observed early-close session
```

The retained published Zipline Reloaded 3.2.0 daily reference omits the July 3
`every_day` callback after July 2's normal close. Its `OncePerDay` state resets
after 24 elapsed hours, so the earlier close does not reset an already-triggered
rule. Backtest executes once on each selected session, including the early
close. This is an intentional correction, recorded separately from matching
callback traces. The minute reference matches on 600 observed events across a
normal and an early-close session, including default, open-offset, close-offset
and half-day rules. Native performance rows remain per observed minute;
Zipline's minute simulation returns daily performance packets. Use native
account/return exports with the intended observation frequency when comparing
analysis.

### Compile trading constraints without changing their meaning

During `initialize`, use `set_max_order_size`, `set_max_position_size`,
`set_max_order_count` and `set_long_only` through the API or context. Size rules
accept a mapped `asset` or apply to every asset, with `max_shares`,
`max_notional`, or both. They compile to native `OrderConstraints` using held
quantity without pending orders, and quoted price times quantity without a
contract multiplier, matching the reference control definitions. Native
`OrderConstraints` retain their default pending-order and economic-notional
choices. These pre-submission limits do not promise that separately admitted
pending orders will remain within a combined position limit after execution.

`on_error="fail"` raises the native `TradingControlViolation` before creating
an order. `on_error="log"` logs the violation and allows submission, as in the
reference; it does not silently reject the order. Native order parameter,
execution, account and data-availability checks still apply.

Zipline's order quota counts requests that reach its count rule, including
zero-sized requests and requests that fail a later rule. The adapter selects
the explicit native `OrderRequestCount` model with UTC civil dates. Rule order
therefore matters. A request rejected by an earlier rule does not consume the
quota. The audit count can advance without an order, fill or cash change.
Native `OrderConstraints(max_orders_per_day=...)` continues counting submitted
orders instead. Both definitions remain available; the adapter does not change
the existing native default.

<!-- ml4t-doc-test: zipline-trading-constraints -->
```python
from datetime import UTC, datetime

import polars as pl
from ml4t.backtest import DataFeed
from ml4t.backtest.compat.zipline import run_algorithm
from ml4t.backtest.compat.zipline.assets import Asset
from ml4t.backtest.controls import TradingControlViolation

answers = []


def initialize(context):
    context.asset = context.symbol("ALPHA")
    context.set_max_order_count(2)
    context.set_max_order_size(max_shares=5)


def handle_data(context, data):
    for quantity in [0, 6, 1]:
        try:
            result = context.order(context.asset, quantity)
            answers.append("none" if result is None else "submitted")
        except TradingControlViolation:
            answers.append("violation")


feed = DataFeed(
    prices_df=pl.DataFrame({"timestamp": [datetime(2025, 1, 6, 21, tzinfo=UTC)],
                            "asset": ["A"], "close": [100.0], "volume": [1000]}),
    feed_spec={"time_contract_version": 2, "timestamp_semantics": "event_time", "calendar": None},
)
performance = run_algorithm(feed=feed, assets=[Asset("A", "ALPHA")],
                            initialize=initialize, handle_data=handle_data)
assert answers == ["none", "violation", "violation"]
assert performance.attrs["backtest_result"].fills == []
print("request quota consumed without creating an order or changing cash")
```

<!-- ml4t-doc-output: zipline-trading-constraints -->
```text
request quota consumed without creating an order or changing cash
```

The zero request passes and consumes the first quota entry. The six-share
request passes the count rule, consumes the second entry, and fails the later
size rule. The final one-share request then fails the count rule. The retained
published Zipline 3.2.0 fixture exercises this sequence and the nonzero,
log-and-allow and long-only variants with divergent open/close prices.

## Account leverage at session close

`set_max_leverage(value)` and `set_min_leverage(value, grace_period)` register native
`AccountConstraints` at the explicit `session_end` boundary. Declare them during
`initialize`. Maximum leverage uses gross exposure divided by NAV. Minimum leverage
starts after UTC midnight of the first accepted session plus the given `timedelta`.
Both validate after the close callback and fills, including an early close. A
position that breaches the maximum intraday and returns within the limit by the
close passes, as in the published Zipline 3.2.0 reference.

`context.account` exposes the native immutable account snapshot: `timestamp`,
`cash`, `equity`, `gross_exposure`, `net_exposure`, `positions_value` and `leverage`.
Other legacy account fields are not synthesized. A violation raises native
`AccountControlViolation`; it does not liquidate a position or undo completed fills.

Session controls require a declared calendar, physical observation timestamps and
a clock observation at every touched session's actual close. A CSV row with null
prices can supply that clock point. It cannot supply a new execution price: held
positions retain their actual prior mark and its timestamp. An incomplete session
raises an input error rather than silently skipping its control. Calendar-free
observation runs can use native `after_callbacks` or `after_fills` controls instead.

<!-- ml4t-doc-test: zipline-account-constraints -->
```python
from datetime import UTC, datetime

import polars as pl
from ml4t.backtest import DataFeed
from ml4t.backtest.compat.zipline import run_algorithm
from ml4t.backtest.compat.zipline.assets import Asset


def initialize(context):
    context.set_max_leverage(1.0)


feed = DataFeed(
    prices_df=pl.DataFrame({"timestamp": [datetime(2025, 7, 3, 17, tzinfo=UTC)],
                            "asset": ["A"], "close": [100.0], "volume": [1000]}),
    feed_spec={"calendar": "NYSE", "timestamp_semantics": "bar_close",
               "bar_label": "right", "data_frequency": "1m",
               "time_contract_version": 2, "bar_grid_anchor": "session_open",
               "partial_bar_policy": "include"},
)
performance = run_algorithm(feed=feed, assets=[Asset("A", "ALPHA")], initialize=initialize)
assert performance["portfolio_value"].iloc[-1] == 100000.0
print("early-close account control passed")
```

<!-- ml4t-doc-output: zipline-account-constraints -->
```text
early-close account control passed
```

## Reopen a native result as a performance frame

`to_performance_frame(result)` is the same projection used by `run_algorithm`
and its `analyze` callback. It reads an in-memory or reloaded `BacktestResult`;
it does not rerun a strategy. Columns contain observed `portfolio_value`, `cash`,
returns including the first period, actual `transactions` and captured recorded
variables. Rows keep the native observation frequency. Minute account rows are
not converted into Zipline's daily performance packets, and uncaptured positions
or statistics are not reconstructed.

New migration runs retain their explicit asset mapping and scoped order-ID prefix
in artifact configuration, so a reload preserves callback-visible identities.
For ordinary native results, the converter uses native stable keys with unknown
symbols and native order IDs. Supply `assets=[Asset(...)]` to declare display
symbols or legacy SIDs; no ticker is inferred from a key. Older migration artifacts
without this declaration need that explicit mapping to recover display metadata.
The `native_order_id` transaction field always preserves the underlying broker ID.

<!-- ml4t-doc-test: zipline-saved-performance -->
```python
from datetime import UTC, datetime
from tempfile import TemporaryDirectory

import polars as pl
from ml4t.backtest import BacktestConfig, BacktestResult, DataFeed, Engine, Strategy
from ml4t.backtest.compat.zipline import to_performance_frame


class HoldCash(Strategy):
    def on_data(self, timestamp, data, context, broker):
        pass


feed = DataFeed(
    prices_df=pl.DataFrame({"timestamp": [datetime(2025, 7, 3, 17, tzinfo=UTC)],
                            "asset": ["A"], "close": [100.0]}),
    feed_spec={"calendar": None, "timestamp_semantics": "event_time"},
)
native = Engine(feed, HoldCash(), BacktestConfig(calendar=None, initial_cash=1000)).run()
with TemporaryDirectory() as directory:
    native.to_parquet(directory)
    restored = BacktestResult.from_parquet(directory)
    reopened = to_performance_frame(restored)
assert reopened["portfolio_value"].iloc[-1] == 1000.0
print("saved native account projected without rerunning the strategy")
```

<!-- ml4t-doc-output: zipline-saved-performance -->
```text
saved native account projected without rerunning the strategy
```

## Move the strategy itself to the native API

The source distribution includes `examples/zipline_native_strategies.py`, with
`DualMovingAverage` and `OnlineMovingAverageReversion`. Both subclass the existing
`Strategy`, receive `RuntimeServices`, read native long-form current/history
views and submit orders through `Broker`. They use no callback facade or Zipline
installation. Construct the runtime once and pass the same instance to the
strategy and `Engine(..., runtime=runtime)`.

`reference_execution()` returns the explicit configuration and shared-volume
limit used for these examples. It selects next-bar **close** fills, FIFO,
per-order per-share fees with a one-dollar minimum, and cumulative volume-share
impact. Pass both the configuration and `execution_limits` to `Engine`; using
its default execution assumptions changes the experiment. This helper describes
the qualified example settings, not all Zipline strategies or financial models.

The OLMAR journey uses seven equity assets, a five-session window, epsilon 1.05
and 13 complete daily sessions. The dual moving-average journey uses 350 daily
sessions, waits for its full 300-session window and exercises both crossovers.
The same frozen OHLCV inputs run through published Zipline 3.2.0, the imported
callbacks above and these native ports. Checks compare intermediate weights,
cash, NAV, position quantities and cost basis where applicable, recorded moving
averages, and every signed fill and execution price. The native ports retain
Apache-2.0 attribution in the example and `examples/LICENSE.zipline`.

These examples use complete daily equity grids. They do not demonstrate
corporate actions, futures accounting, cross-frequency preparation or missing-bar
imputation. For causal feature preparation use Engineer, for adjustment inputs
use Data, and keep their declared availability and basis when constructing the
feed. The strategy owns decisions; the native broker owns fills and accounts.

## Session run dates and execution environment

`run_algorithm(start=..., end=...)` accepts session dates, represented as `date`,
date strings or timezone-naive midnight timestamps. It selects the first calendar
session on or after `start` through the last on or before `end`. Non-midnight and
timezone-aware timestamps are rejected as session labels. For exact physical
cutoffs, supply `RuntimeServices(start=instant, end=instant)` instead. Calendar-free
runs use physical bounds; they cannot interpret legacy session dates.

For a session window, the first tradable minute is one minute after the declared
calendar open, matching the right-labelled minute reference clock; the end is the
actual close, including early closes. Existing pre-start observations remain
available for causal history, but produce no trading callbacks or account rows.
A borrowed runtime retains its original bounds after completion or any error.
Conflicting explicit native and session bounds raise before the initializer runs.

`get_environment(field="platform")` is available through the context or scoped
API, including during initialization. `"*"` returns an independent dictionary
with `arena`, `data_frequency`, `start`, `end`, `capital_base` and `platform`.
The platform is truthfully `"ml4t.backtest"`. Daily and one-minute frequency
names are `"daily"` and `"minute"`; other prepared native frequencies retain
their declared values. Bounds are UTC physical timestamps. When no session dates
are supplied, this adapter preserves native observation filtering and reports
the native bounds or first/last accepted observations.

<!-- ml4t-doc-test: zipline-session-window -->
```python
from datetime import UTC, datetime

import polars as pl
from ml4t.backtest import DataFeed
from ml4t.backtest.compat.zipline import run_algorithm
from ml4t.backtest.compat.zipline.api import get_environment
from ml4t.backtest.compat.zipline.assets import Asset

environments = []


def initialize(context):
    environments.append(get_environment("*"))


feed = DataFeed(
    prices_df=pl.DataFrame({"timestamp": [datetime(2025, 7, 2, 20, tzinfo=UTC),
                                          datetime(2025, 7, 3, 17, tzinfo=UTC)],
                            "asset": ["A", "A"], "close": [100.0, 101.0]}),
    feed_spec={"calendar": "NYSE", "time_contract_version": 2,
               "timestamp_semantics": "bar_close", "data_frequency": "1d",
               "bar_label": "right", "bar_grid_anchor": "session_open",
               "partial_bar_policy": "include"},
)
performance = run_algorithm(feed=feed, assets=[Asset("A")], initialize=initialize,
                            start="2025-07-03", end="2025-07-05")
assert len(performance) == 1
assert environments[0]["end"].isoformat() == "2025-07-03T17:00:00+00:00"
assert environments[0]["platform"] == "ml4t.backtest"
print("session window ends at the actual early close")
```

<!-- ml4t-doc-output: zipline-session-window -->
```text
session window ends at the actual early close
```

## Resolve dated symbols from a Data snapshot

`PreparedAssetLookup(snapshot)` consumes the `zipline_equities`,
`zipline_symbol_lookup` and available exchange-country facts produced by Data's
bundle exporter. It supplies both immutable `Asset` identifiers and the captured
symbol-ownership policy. Pass either `asset_lookup=...` or an explicit `assets`
sequence to `run_algorithm`.

With a declared calendar, `symbol` defaults to the selected **end session**, as
in published Zipline 3.2.0. Calendar-free observation mode instead uses the UTC
civil date of the last accepted instant and names that policy explicitly. `set_symbol_lookup_date(value)` overrides that date;
timezone-aware overrides are converted to UTC before removing the timezone.
This lookup-date contract differs from the midnight-only run-date contract.
`symbols(...)` returns a list in request order. Case and the first share-class
separator (`.`, `/`, `-`, `_`) follow the reference spelling rules. Optional
`country_code` selects declared exchange-country facts; unavailable or ambiguous
facts raise an error instead of selecting an arbitrary asset.

A lookup selects an identity, never a future market observation. In the qualified
minute fixture, initializing `symbol("ALPHA")` selects the later owner, while
`data.current` still returns a missing price until that asset's observations
arrive. An earlier lookup date selects the old stable ID with its exported
latest display symbol. These are separate facts.

The exported ownership intervals intentionally extend beyond listing lifetimes.
Resolving an old alias therefore does not establish that it is active or
tradable. This object does not install native instrument eligibility or establish
historical publication times. When that guarantee is required, supply a
Data-backed `RuntimeServices(instruments=..., instrument_feed_spec=...)` with
its declared historical availability. See [dated instruments and lifetimes](strategies.md#resolve-dated-instruments-and-enforce-their-lifetimes).
The raw bundle export explicitly reports unavailable historical publication facts.

The following preparation/read route was qualified using
[Data source c3b9db80](https://github.com/ml4t/data/commit/c3b9db80a66397420fb27adeccc762559dd41e07),
Specs 0.2.0 and published Zipline Reloaded 3.2.0. This is a source-candidate Data
integration, not a claim that every published Data version has the exporter.
Zipline is needed only in the one-time bundle export environment; reading the
portable snapshot and running the strategy needs no legacy package.

```python
from ml4t.data.storage import read_snapshot
from ml4t.specs import FeedSpec
from ml4t.backtest import DataFeed
from ml4t.backtest.compat.zipline import PreparedAssetLookup, run_algorithm

# Read a snapshot previously written by ml4t.data.compat.zipline.export_bundle.
snapshot = read_snapshot("prepared-bundle")
clock = snapshot.manifest["content"]["tables"]["minute_bars"]["clock"]
feed_spec = FeedSpec.from_any(clock["feed_spec"]).with_overrides(
    calendar=clock["resolved_calendar"]
)
feed = DataFeed(prices_df=snapshot.tables["minute_bars"], feed_spec=feed_spec)
performance = run_algorithm(
    feed=feed,
    asset_lookup=PreparedAssetLookup(snapshot),
    initialize=initialize,
    handle_data=handle_data,
    start="2025-07-02",
    end="2025-07-03",
)
```

The complete exported fixture and its original callback trace are retained in
`tests/fixtures/zipline_symbol_lookup`. The installed Data-reader test exercises
the recipe above with actual producer output. Static manual asset mappings also
support case/share-class spelling and list containers, but cannot answer dated
or country-specific queries without their source facts.

### Deliver prepared Pipeline results

Keep factor calculations and universe preparation in Engineer. Backtest's
`PreparedPipeline` consumes that panel and its `FeatureSpec`, then exposes
`attach_pipeline` and `pipeline_output` on the callback API and context. It
uses the native auxiliary-data service and pre-open scheduler.

In the original workflow, `SimpleMovingAverage(close, window_length=2)` on
July 2 uses June 30 and July 1 closes. A screen based on latest volume also
uses July 1. Prepare those values upstream with an explicit July 2 decision
instant. The adapter does not calculate factors from execution prices.

The following example starts from a prepared panel. Its two closes, 10 and
20, give mean 15. The panel is published at July 1 close and delivered at
July 2's NYSE premarket clock, 12:45 UTC.

<!-- ml4t-doc-test: prepared-pipeline-delivery -->
```python
from datetime import UTC, datetime
import polars as pl
from ml4t.backtest import DataFeed
from ml4t.backtest.compat.zipline import PreparedPipeline, run_algorithm
from ml4t.backtest.compat.zipline.assets import Asset

decision = datetime(2025, 7, 2, 12, 45, tzinfo=UTC)
panel = pl.DataFrame({
    "timestamp": [datetime(2025, 7, 1).date()], "asset": ["A"],
    "decision_time": [decision],
    "available_at": [datetime(2025, 7, 1, 20, tzinfo=UTC)],
    "eligible": [True], "mean": [15.0], "latest": [20.0],
})
feature_spec = {
    "artifact_id": "two-price-mean", "kind": "features", "version": 1,
    "schema": {"timestamp_col": "timestamp", "entity_col": "asset",
               "feature_columns": ["mean", "latest"],
               "decision_time_col": "decision_time", "available_at_col": "available_at",
               "eligible_col": "eligible"},
    "definition": {"feed_spec": {"timestamp_col": "timestamp", "entity_col": "asset",
                                 "timestamp_semantics": "session_label", "calendar": "NYSE"},
                   "price_basis": "raw"},
}
pipeline = PreparedPipeline(panel, feature_spec=feature_spec)
prices = pl.DataFrame({
    "timestamp": [datetime(2025, 7, 1, 20, tzinfo=UTC),
                  datetime(2025, 7, 2, 20, tzinfo=UTC)],
    "asset": ["A", "A"], "close": [20.0, 30.0], "volume": [1000, 1000],
})
feed = DataFeed(prices_df=prices, feed_spec={
    "calendar": "NYSE", "time_contract_version": 2, "data_frequency": "1d",
    "timestamp_semantics": "bar_close", "bar_label": "right",
})
asset, seen = Asset("A", "ALPHA", 0), []

def initialize(context):
    context.attach_pipeline(pipeline, "alpha", eager=False)

def before_trading_start(context, data):
    frame = context.pipeline_output("alpha")
    assert frame.loc[asset, "mean"] == 15.0
    assert frame.loc[asset, "latest"] == 20.0
    seen.append(context.get_datetime())

performance = run_algorithm(
    feed=feed, assets=[asset], initialize=initialize,
    before_trading_start=before_trading_start, start="2025-07-02", end="2025-07-02",
)
assert seen == [decision]
assert performance.portfolio_value.tolist() == [100000.0]
print("Prepared Pipeline delivery verified.")
```

For actual preparation, use Engineer's `prepare_feature_panel` with its
declared lookbacks, observation cutoff, decision/publication instants and dated
universe. Pass the resulting panel and `FeatureSpec` to the constructor above.
The integration test in `tests/test_zipline_pipeline.py` regenerates
`tests/fixtures/prepared_pipeline` through the public Engineer API. That route
was qualified with [Engineer source 093b72d6](https://github.com/ml4t/engineer/commit/093b72d6801cf2b279f1aa682c6d2cabe45d494d)
and Specs 0.2.0. The Backtest consumer needs neither Engineer nor Zipline
installed.

`columns={"output_name": "feature_name"}` selects and renames declared fields
in insertion order. Each session uses the panel rows at its exact premarket
decision instant, with publication and eligibility checked by the native
runtime. Later publication cannot enter that session's cached output.
`eager=False` delays Pandas conversion until the first read; both modes capture
values at the same premarket instant. Returned frames are independent copies.
Rows use the declared `Asset` index, sorted by legacy SID when supplied.

A decision with no eligible rows returns an empty frame with the declared
columns and dtypes. Supply the optional `decisions` table to declare sessions
whose panel contains no rows at all. An undeclared decision raises an error
instead of reusing the previous session's factors. A genuine missing factor
stays missing.

The original Pipeline expression DSL and `chunks` calculation scheduling move
to upstream preparation. This adapter requires a prepared panel; it rejects a
legacy Pipeline object or non-`None` `chunks`. Translate the calculation and
batch preparation before attaching it. The pinned published 3.2.0 comparison
uses explicit finite asset lifetimes, two output columns, a volume screen,
eager/lazy attachments and July 3's early close. Its premarket and market
frames match the prepared route, including column order and dtypes. This
comparison does not qualify arbitrary custom Pipeline terms.

### Register an offline CSV source

Move CSV loading and preprocessing before the run. `PreparedCSV` accepts the
caller-loaded Pandas frame and freezes it as native `AuxiliaryData`. Pass it
through `run_algorithm(csv_sources=[...])`; the original callback's
`fetch_csv(url)` then selects that exact original source identity. An
unregistered identity raises an error. No callback downloads data.

The original `pre_func` runs before date parsing and symbol mapping. The
original `post_func` receives a UTC DatetimeIndex and a `sid` column containing
the mapped immutable `Asset` objects or fixed research identifiers. Move both
functions, parser options and symbol lookup into `PreparedCSV`. The callback
keeps only `fetch_csv(original_source_id)`.

<!-- ml4t-doc-test: prepared-csv-delivery -->
```python
from datetime import UTC, datetime
import pandas as pd
import polars as pl
from ml4t.backtest import DataFeed
from ml4t.backtest.compat.zipline import PreparedCSV, run_algorithm
from ml4t.backtest.compat.zipline.assets import Asset

source_id = "https://example.org/original-research.csv"
asset = Asset("A", "ALPHA", 7)
# In production, load these bytes through your data-preparation workflow.
raw = pd.DataFrame({"date": ["2025-07-01"], "ticker": ["ALPHA"], "raw_score": [5.0],
                    "published": ["2025-07-01T18:00:00Z"]})

def preprocess(frame):
    return frame.rename(columns={"ticker": "symbol"})

def postprocess(frame):
    return frame.assign(signal=frame["raw_score"] * 2 + 1).drop(columns="raw_score")

prepared = PreparedCSV(
    source_id, raw, available_at_col="published", assets={"ALPHA": asset},
    pre_func=preprocess, post_func=postprocess,
)
prices = pl.DataFrame({"timestamp": [datetime(2025, 7, 1, 20, tzinfo=UTC),
                                     datetime(2025, 7, 2, 20, tzinfo=UTC)],
                       "asset": ["A", "A"], "close": [10.0, 20.0], "volume": [1000, 1000]})
feed = DataFeed(prices_df=prices, feed_spec={
    "calendar": "NYSE", "time_contract_version": 2, "data_frequency": "1d",
    "timestamp_semantics": "bar_close", "bar_label": "right",
})
seen = []

def initialize(context):
    context.fetch_csv(source_id)

def handle_data(context, data):
    seen.append(data.current(asset, "signal"))
    assert data.fetcher_assets == [asset]

performance = run_algorithm(feed=feed, assets=[asset], csv_sources=[prepared],
                            initialize=initialize, handle_data=handle_data)
assert seen == [11.0, 11.0]
assert performance.attrs["backtest_result"].config.metadata["migration_csv"][source_id]["source_id"] == source_id
print("Frozen original CSV source verified.")
```

`assets` may instead be a `PreparedAssetLookup` for dated, country-specific
symbol resolution. `mask=True` excludes unresolved symbols and records them
in preparation metadata; `mask=False` retains unresolved nonempty strings as
research keys. `symbol="VIX"` assigns one fixed research key to every row.
Read those values with `data.current("VIX", "field")`; an order still requires
a declared tradable `Asset`. With several registered sources, `fetcher_assets`
reports the last selected source's known asset keys.

CSV dates identify physical observations in the declared `timezone` (UTC by
default). An explicit `date_format` is honored. Invalid dates are dropped and
counted; duplicate observation/asset keys require upstream resolution.
Publication comes from the separate required `available_at_col`. Obtain that
timestamp from the source's availability facts; the adapter does not infer it
from the observation date. Unknown publication stays unavailable, and a value
published one nanosecond after the decision cannot be read at the decision.

These clock choices differ from the published reference's session-date
normalization and deprecated `date_format` handling. Prepare any required
session rolling upstream. The frozen reference fixture in
`tests/fixtures/zipline_original/csv-reference.json` qualifies the preprocessing,
postprocessing, symbol mapping, daily stale delivery and fetcher-asset route
against published Zipline Reloaded 3.2.0 with SID 7. Native tests additionally
cover SID 0, delayed publication, timezones, missing values and research keys.

Market OHLCV for a declared `Asset` keeps its execution-source meaning. An
auxiliary column named `close` cannot replace its market close. Conflicting
research field/asset pairs across sources raise an error; query
`runtime.auxiliary(original_source_id)` explicitly to distinguish them. The
native result retains the original identity, prepared content hash, parser
choices, transformation names and excluded-row counts. No extra market event
is created by an auxiliary row.

### Supply a declared benchmark

Pass native [`BenchmarkData`](results.md#prepared-benchmark-returns) through
`run_algorithm(benchmark=prepared_benchmark, ...)`. An existing
`set_benchmark(asset)` callback selects that supplied input only when its
stable asset key matches. It does not compute returns from `data.current`.
Omitting `set_benchmark` uses the supplied input; `set_benchmark(None)`
explicitly disables it. Without a prepared input, selecting an asset raises
an error before market callbacks.

The performance frame adds `benchmark_return` for each observed period and
`benchmark_period_return` for the cumulative product. The native result keeps
the original table and provenance, including through artifact reload.
Benchmark data does not change cash, holdings, fills, NAV or strategy returns.

The dividend-bearing reference in
`tests/fixtures/zipline_original/benchmark-dividend-reference.json` verifies
the actual published 3.2.0 asset-benchmark convention and early-close period
keys. `tests/test_benchmark_data.py` compares those values with prepared
native and actual Data output. Prepare returns at the required frequency;
no daily-to-minute repetition or automatic download occurs. If dividends
matter, a price-only proxy cannot stand in for the reference's declared
dividend-adjusted convention.

### Restrict assets by effective time

Import `NoRestrictions`, `StaticRestrictions`, `HistoricalRestrictions`,
`Restriction`, `RESTRICTION_STATES` and `SecurityListRestrictions` from
`ml4t.backtest.compat.zipline.finance.asset_restrictions`. Set them during
`initialize` with `set_asset_restrictions(rules, on_error="fail")` or the
equivalent context method. `set_do_not_order_list` remains an alias for an
explicit asset container or a caller-owned `current_securities(dt)` provider.
No security-list download or legacy runtime is required.

The adapter compiles each declaration into the existing native
`OrderConstraints.restriction` validator. Native strategies can use that
predicate directly, or `blocked_assets` for fixed stable keys. No separate
order book or rejection path is created.

<!-- ml4t-doc-test: dated-asset-restrictions -->
```python
from datetime import UTC, datetime
import pandas as pd
import polars as pl
from ml4t.backtest import DataFeed
from ml4t.backtest.compat.zipline import run_algorithm
from ml4t.backtest.compat.zipline.assets import Asset
from ml4t.backtest.compat.zipline.finance.asset_restrictions import HistoricalRestrictions
from ml4t.backtest.compat.zipline.finance.asset_restrictions import Restriction
from ml4t.backtest.compat.zipline.finance.asset_restrictions import RESTRICTION_STATES
from ml4t.backtest.compat.zipline.finance.asset_restrictions import StaticRestrictions

a, b = Asset("A", "ALPHA", 7), Asset("B", "BETA", 8)
times = [datetime(2025, 7, 1, 20, tzinfo=UTC), datetime(2025, 7, 2, 20, tzinfo=UTC),
         datetime(2025, 7, 3, 17, tzinfo=UTC)]
prices = pl.DataFrame([{"timestamp": point, "asset": asset.key,
                        "close": 100.0 + index, "volume": 1000}
                       for index, point in enumerate(times) for asset in [a, b]])
feed = DataFeed(prices_df=prices, feed_spec={
    "calendar": "NYSE", "time_contract_version": 2, "data_frequency": "1d",
    "timestamp_semantics": "bar_close", "bar_label": "right",
})
flags = []

def initialize(context):
    dated = HistoricalRestrictions([
        Restriction(a, pd.Timestamp(times[1]), RESTRICTION_STATES.FROZEN),
        Restriction(a, pd.Timestamp(times[2]), RESTRICTION_STATES.ALLOWED),
    ])
    context.set_asset_restrictions(StaticRestrictions([b]) | dated)

def handle_data(context, data):
    flags.append(data.can_trade([a, b]).tolist())
    if len(flags) == 1:
        context.order(a, 1)

performance = run_algorithm(feed=feed, assets=[a, b], initialize=initialize,
                            handle_data=handle_data, capital_base=1000)
assert flags == [[True, False], [False, False], [True, False]]
fills = performance.attrs["backtest_result"].to_fills_dataframe()
assert fills["quantity"].to_list() == [1.0]
assert abs(fills["price"][0] - 101.0505) < 1e-12
print("Dated restriction queries and native orders verified.")
```

The first order was submitted before the restriction became effective. It
still fills at the next close; these rules do not cancel existing orders.
Equality with an effective timestamp selects the new state. Naive effective
dates use the query's timezone; migration callbacks query in UTC. A
`SecurityListRestrictions` provider may return declared `Asset` objects or
explicit legacy integer SIDs. Strings are not inferred as symbols or IDs.
The `|` operator combines rules by logical OR and preserves Boolean Series
shapes in requested asset order. Future effective events cannot change earlier
queries.

`on_error="fail"` rejects a restricted order through native validation.
`on_error="log"` records the violation and permits the order, while
`data.can_trade` still reports the restriction. Failed requests continue to
consume the adapter's declared order-request count. The captured published
3.2.0 fixture in `tests/fixtures/zipline_original/restrictions-reference.json`
checks those queries, submissions, fills and cash for both policies.

`data.can_trade` combines known-price eligibility, declared lifetimes and
these restriction predicates. It does not promise an execution fill or check
buying power. Native lifetime eligibility remains with the Data-backed runtime
instrument service; metadata absent from that service is not inferred.

### Configure equity basis-point slippage

`FixedBasisPointsSlippage(basis_points=5.0, volume_limit=0.1)` is the adapter's
default equity model, matching published Zipline Reloaded 3.2.0. It compiles
to native percentage slippage and the shared asset/bar volume limit. For a
buy at close 101, 5 basis points gives 101.0505. Two orders share the same
bar's capacity; at volume 1000, the default cap fills at most 100 shares in
total. The supported `volume_limit` range is `(0, 1]`.

Both positional model arguments and the original `us_equities=` keyword are
accepted by `set_slippage` and `set_commission`. A call with no model preserves
the selected model. Use `VolumeShareSlippage(volume_limit=..., price_impact=...)`
explicitly for the quadratic cumulative-participation method, `FixedSlippage`
for a fixed spread, or `NoSlippage` for zero impact with positive observed
volume. These adapter choices leave native Backtest defaults unchanged.

### Declare futures contracts and per-order costs

For `continuous_future`, `data.current_chain`, and the selected dated contract,
use the registered Data-backed route in
[dated-futures migration](dated-futures.md#retain-continuous-futures-callback-reads).
Its complete offline example covers raw, additive and multiplicative history,
with explicit roll-mark conventions and separate raw-contract execution.

Use `Future` for a dated contract whose root, multiplier and tick size are
known. `PreparedAssetLookup` also builds these objects from Data's exported
`zipline_futures` table. `future_symbol("ESH25")` resolves only declared
contracts; it does not infer a root, select a continuous series or discover a
bundle. Equity `symbol` lookup remains separate. These frozen bundle facts do
not establish when historical instrument metadata was published; use the
native instrument service when causal eligibility is required.

`set_commission(us_futures=PerContract(cost, exchange_fee, min_trade_cost))`
compiles to native `AssetCommission` and `PerOrderCommission`.
`PerFutureTrade(cost)` charges its fixed amount once per order. Scalar and
root-symbol mappings are supported. A partial fill pays only the additional
cumulative fee due. The contract multiplier affects value sizing and P&L,
and does not multiply the per-contract fee.

<!-- ml4t-doc-test: declared-future-costs -->
```python
from datetime import datetime, timezone
import polars as pl
from ml4t.backtest import DataFeed
from ml4t.backtest.compat.zipline import run_algorithm
from ml4t.backtest.compat.zipline.assets import Future
from ml4t.backtest.compat.zipline.finance import commission, slippage

contract = Future("contract:8", "ESH25", 8, root_symbol="ES",
                  multiplier=50., tick_size=.25)
prices = pl.DataFrame({
    "timestamp": [datetime(2025, 3, d, 21, tzinfo=timezone.utc) for d in [3, 4, 5, 6]],
    "asset": [contract.key] * 4, "close": [100.] * 4, "volume": [2.] * 4,
})
def initialize(context):
    context.sent = False
    context.set_commission(us_futures=commission.PerContract({"ES": 2.}, {"ES": 3.}, 8.))
    context.set_slippage(us_futures=slippage.FixedBasisPointsSlippage(0, .5))
def handle_data(context, data):
    if not context.sent:
        context.order(context.future_symbol("ESH25"), 3)
        context.sent = True

performance = run_algorithm(
    feed=DataFeed(prices_df=prices, feed_spec={"calendar": None,
        "time_contract_version": 2, "timestamp_semantics": "event_time"}),
    assets=[contract], initialize=initialize, handle_data=handle_data)
result = performance.attrs["backtest_result"]
assert result.to_fills_dataframe()["quantity"].to_list() == [1., 1., 1.]
assert result.to_fills_dataframe()["commission"].to_list() == [8., 0., 1.]
assert performance["portfolio_value"].iloc[-1] == 99991.
assert performance["cash"].iloc[-1] == 84991.
print("Declared contract units and incremental fees verified.")
```

`run_algorithm(contract_specs=...)` passes explicit native specifications to
the Engine. A supplied specification for a `Future` must agree on asset class,
multiplier and tick size; margin and currency remain explicitly configurable.
Conflicts fail before initialization. Saved native results preserve the
declared `Future` facts when projected back to a performance frame.

The adapter's default futures commission uses the frozen published 3.2.0
cost of 0.85 per contract and its root exchange-fee table. Missing root entries
in cost mappings use 0.85; missing exchange/trade-fee entries use that frozen
table. An unknown root requires an explicit fee. These are reference
reproduction values, not current exchange quotations. Explicit values should
describe the account being simulated.

The published 3.2.0 comparison uses the same four physical closes and actual
exported Data contract facts, with one contract per bar to isolate fees. It
checks mixed equity/futures NAV and fees, scalar/root mappings, fallback
options and saved-result identity. This does not establish rolling or
variation-cash parity. The example explicitly chooses zero basis-point
slippage with a per-bar participation cap, and its cash follows the native signed-notional method
described in [ledger assumptions](#match-ledger-and-corporate-action-assumptions).

### Use futures volatility-volume slippage

The adapter selects `VolatilityVolumeShare(volume_limit=0.05, eta=None)` for
declared `Future` assets. Configure it with `set_slippage(us_futures=...)`;
the positional or `us_equities=` model continues to select equity execution
only. Futures also accept `FixedBasisPointsSlippage`, `FixedSlippage` and
`NoSlippage`. These options compile to native per-asset models and limits.

Volatility-volume execution uses the observed high/low midpoint, with impact
`midpoint * eta * annualized_volatility * sqrt(fill_quantity / ADV) / 10000`.
The statistics use the prior 20 completed daily sessions: mean reported
volume and sample standard deviation of 19 close returns, multiplied by
`sqrt(252)`. The fixed 252 is this legacy cost convention, separate from
the Engine's account-return annualization. A scalar `eta` applies to every
future; a root-symbol mapping falls back to the frozen published 3.2.0 root
table and then its unknown-root value. These are reference parameters, not
current market estimates.

Without sufficient history, or with zero ADV, impact is 7.5 basis points.
A known zero-volume daily row with a missing close carries the preceding
close **within that statistics window** for its return calculation, matching
the reference. Raw quotes remain missing. A missing session or unknown volume
does not become a zero-volume observation. With `calendar=None`, the window
counts 20 strictly prior observations instead of calendar sessions.

Minute execution can use an explicitly declared completed daily source via
`history_sources={"1d": "daily"}` and
`RuntimeServices(history_sources={"daily": provider})`; the existing
[independent history source contract](#use-a-separate-prepared-history-grid)
supplies causal, prepared observations. Without one, a minute feed uses the
insufficient-history method. This cost model does not resample minute data or
read a future completed daily close. Newly available source revisions apply
at their actual fill clock. High and low must be actual published observations;
the native feed's close fallback is not accepted. Opening-auction targets
cannot use this completed-bar model.

<!-- ml4t-doc-test: futures-volatility-volume -->
```python
from datetime import UTC, datetime
import polars as pl
from ml4t.backtest import DataFeed
from ml4t.backtest.compat.zipline import run_algorithm
from ml4t.backtest.compat.zipline.assets import Future
from ml4t.backtest.compat.zipline.finance import commission, slippage

contract = Future("contract:8", "ESH25", 8, root_symbol="ES",
                  multiplier=50., tick_size=.25)
prices = pl.DataFrame({
    "timestamp": [datetime(2025, 3, day, 21, tzinfo=UTC) for day in [3, 4, 5]],
    "asset": [contract.key] * 3, "close": [105.] * 3,
    "high": [110.] * 3, "low": [90.] * 3, "volume": [100.] * 3,
})
def initialize(context):
    context.sent = False
    context.set_commission(us_futures=commission.PerFutureTrade(0))
    context.set_slippage(us_futures=slippage.VolatilityVolumeShare(.05, eta=.5))
def handle(context, data):
    if not context.sent:
        context.order(contract, 8)
        context.sent = True

performance = run_algorithm(
    feed=DataFeed(prices_df=prices, feed_spec={"calendar": "NYSE",
        "time_contract_version": 2, "timestamp_semantics": "bar_close",
        "data_frequency": "daily", "bar_label": "right",
        "bar_grid_anchor": "session_open", "partial_bar_policy": "include"}),
    assets=[contract], initialize=initialize, handle_data=handle)
result = performance.attrs["backtest_result"]
assert [fill.quantity for fill in result.fills] == [5., 3.]
assert [fill.price for fill in result.fills] == [100.075, 100.075]
assert performance["portfolio_value"].iloc[-1] == 101970.
print("Futures midpoint, insufficient-history impact and partial fills verified.")
```

The limit applies per order, while all orders together remain bounded by
reported bar volume. Two four-contract orders at volume 110 and a 5% limit
can each fill four contracts. The retained published 3.2.0 reference can
overfill raw volume at larger participation settings: two 100-contract
orders with a 60% limit fill 66 each against volume 110. Backtest instead
fills 66 and 44, carrying the remainder forward. This volume correction is
an explicit reference difference. The reference fixtures in
`tests/fixtures/zipline_volatility` and `zipline_volatility_missing` also cover
long/short fills, scalar/root/default eta, insufficient and complete history,
and mixed equity/futures NAV. They do not establish variation-cash settlement.

### Cancel pending orders at session end

Declare `set_cancel_policy(EODCancel(warn_on_cancel=True))` during
initialization, or `NeverCancel()` to keep the default pending-order policy.
The adapter selects native session-end expiration. Orders can still fill on
the closing bar. After all callbacks and fills at that bar, remaining
quantities expire, including orders submitted by the close callback.
Warnings report each actual cancellation and its filled/remaining quantities;
`warn_on_cancel=False` suppresses those messages.

<!-- ml4t-doc-test: legacy-session-end-cancel -->
```python
from datetime import datetime, timezone
import polars as pl
from ml4t.backtest import DataFeed
from ml4t.backtest.compat.zipline import run_algorithm
from ml4t.backtest.compat.zipline.assets import Asset
from ml4t.backtest.compat.zipline.finance import commission, slippage
from ml4t.backtest.compat.zipline.finance.cancel_policy import EODCancel

asset = Asset("A", "ALPHA", 7)
times = [datetime(2025, 7, 3, 16, 59, tzinfo=timezone.utc),
         datetime(2025, 7, 3, 17, tzinfo=timezone.utc)]
final = []
def initialize(context):
    context.ids = []
    context.set_cancel_policy(EODCancel(warn_on_cancel=False))
    context.set_commission(commission.PerShare(0))
    context.set_slippage(slippage.FixedBasisPointsSlippage(0, .5))
def handle_data(context, data):
    context.ids.append(context.order(asset, 3 if not context.ids else 4))
def analyze(context, performance):
    final.extend((context.get_order(key).amount, context.get_order(key).filled,
                  int(context.get_order(key).status)) for key in context.ids)

performance = run_algorithm(feed=DataFeed(prices_df=pl.DataFrame({
    "timestamp": times, "asset": ["A", "A"], "close": [100., 100.], "volume": [2., 2.],
}), feed_spec={"calendar": "NYSE", "time_contract_version": 2,
    "timestamp_semantics": "bar_close", "bar_label": "right", "data_frequency": "1m"}),
    assets=[asset], initialize=initialize, handle_data=handle_data, analyze=analyze)
assert final == [(3., 1., 2), (4., 0., 2)]
assert performance.attrs["backtest_result"].to_fills_dataframe()["quantity"].to_list() == [1.]
print("Half-day close fills precede remainder cancellation.")
```

The calendar is required. Each selected session must contain its actual
execution-close observation; a truncated capture cannot claim EOD processing
completed. For the legacy `us_futures` route, expiration uses the explicit
17:00 New York execution close rather than the full calendar's 18:00 close.
The underlying calendar and native default remain unchanged.

The published 3.2.0 minute captures compare unfilled limits, partial orders,
orders submitted at the close, next-session visibility, execution times and
cancellation timestamps across an NYSE half-day and the synthetic futures
clock. Its daily EOD implementation behaves differently: with three unfilled
orders, it cancels only one per session and leaves the last pending order
open. The daily fixture records final statuses `[2, 2, 0]`. Backtest cancels
all three at the first close, giving `[2, 2, 2]`; it does not reproduce that
reference defect.

Native callers can configure a shorter declared order session with
`engine.broker.configure_session_cancellation(bounds=..., warn_on_cancel=...)`
before the first market event. Each civil session date maps to aware physical
open/close instants within that calendar session. Use it with
`cancel_policy="session_end"`; other sessions retain their full calendar
close. It cannot change cancellation settings during a running strategy.

### Separate stale reads from executable quotes

The callback runner selects `BarData(read_policy="zipline")`. In this read
policy, `current(asset, "price")` and price history may carry the most recent
known finite price. `close` remains the value of the requested observation;
it is not forward-filled. `history(..., ffill=False)` disables stale price
values in that history request. Missing nulls and NaNs follow the same price
policy.

<!-- ml4t-doc-test: legacy-stale-price-view -->
```python
from datetime import UTC, datetime
import pandas as pd
import polars as pl
from ml4t.backtest import DataFeed, RuntimeServices
from ml4t.backtest.compat.zipline import run_algorithm
from ml4t.backtest.compat.zipline.assets import Asset

times = [datetime(2025, 7, 1, 20, tzinfo=UTC), datetime(2025, 7, 2, 20, tzinfo=UTC),
         datetime(2025, 7, 3, 17, tzinfo=UTC)]
prices = pl.DataFrame({"timestamp": times, "asset": ["A"] * 3,
                       "close": [100.0, None, 110.0], "volume": [1000, 0, 0]})
feed = DataFeed(prices_df=prices, feed_spec={
    "calendar": "NYSE", "time_contract_version": 2, "data_frequency": "1d",
    "timestamp_semantics": "bar_close", "bar_label": "right",
    "bar_grid_anchor": "session_open", "partial_bar_policy": "include",
})
asset, runtime, seen = Asset("A", "ALPHA", 7), RuntimeServices(), []

def handle_data(context, data):
    value = data.current(asset, "price")
    seen.append((value, data.can_trade(asset), data.is_stale(asset)))
    assert data.current(asset, "last_traded") == pd.Timestamp("2025-07-01")
    if context.get_datetime().day == 2:
        assert value == 100.0 and pd.isna(data.current(asset, "close"))
        assert not runtime.market.can_trade("A")
        context.order(asset, 1)

performance = run_algorithm(feed=feed, assets=[asset], runtime=runtime,
                            handle_data=handle_data)
assert seen == [(100.0, True, False), (100.0, True, True), (110.0, True, True)]
assert performance.attrs["backtest_result"].fills == []
print("Stale reads do not create executable prices or volume.")
```

`is_stale` distinguishes a known earlier trade from an asset that has never
traded. In the Zipline read policy, a live asset with zero current volume and
an earlier actual trade is stale, even if an indicative close was observed
now. `last_traded` requires positive observed volume and a finite published
price. Its daily output uses the naive session label; minute and calendar-free
output uses the physical UTC instant with nanosecond precision.

Supply Data-backed `RuntimeServices(instruments=..., instrument_feed_spec=...)`
to enforce listing lifetimes. The adapter masks OHLC/price outside those
declared lifetimes and presents volume as zero there. It retains historical
last-traded dates. Historical lifetime masks apply at each requested history
point, using facts known at the current cutoff. The actual Data integration in
`tests/test_zipline_stale_reads.py` checks a July 2 listing that expires before
July 3, alongside the published 3.2.0 current/history/staleness fixture. The
fixture declares its metadata known before the run; it does not infer actual
historical publication times from listing dates.

Direct `BarData(runtime)` preserves the existing native raw-read policy. Native
`runtime.market.can_trade` continues to require a current executable quote.
Both modes share the native market service; neither fills missing prices into
the broker's input. A liquid bar whose price is unpublished cannot execute a
pending order merely because the read view can display an older known price.

Before trading starts, price reads can return causal warmup prices. They do
not expose that session's future close. Pass `history_view="prepared"` to the
callback runner to select the existing causal Data history provider, or
`history_unit="observation"` to count observations explicitly. This prepared
view retains the execution grid. For daily reads during minute execution,
the Zipline policy projects native partial-session aggregation as described
below. Other history grids require an explicitly prepared source.

### Read the current partial daily bar

With the default Zipline read policy, `data.history(..., "1d")` on a declared
minute feed includes the current session through the active observation.
It uses native `runtime.market.session_bars(...)`; raw minute observations
and executable quotes remain unchanged. Supply the earlier minute observations
needed for each requested session, including warmup before the run starts.
The native service counts exchange sessions and requires a complete declared
intraday bar grid and calendar. `calendar=None` does not invent session boundaries.

Open is the first available value, high/low are extrema of available values,
and close is the last available value in that session. Only the explicit
`price` read view can carry a prior price across missing observations.
Volume is the sum of known expected-bar volumes; an absent or unknown value
makes the session total unavailable. A captured reader zero remains zero.
These reads do not infer whether a missing quote means no trade.

The native frame retains physical `timestamp` and `window_start` cutoffs,
`session_closed`, `expected_bars` and `observed_bars`. The latter counts input
bar rows with published requested fields, including rows with unknown values;
it does not count carried prices as observations. These columns describe
coverage, not complete-data certification. The adapter projects daily session
labels and the original Pandas container shape. It does not label an in-progress
price as a future completed close.

<!-- ml4t-doc-test: zipline-partial-session -->
```python
from datetime import UTC, datetime
import polars as pl
from ml4t.backtest import DataFeed, RuntimeServices
from ml4t.backtest.compat.zipline import run_algorithm
from ml4t.backtest.compat.zipline.assets import Asset

asset = Asset("A", "ALPHA", 7)
times = [datetime(2024, 7, 3, 13, minute, tzinfo=UTC) for minute in [31, 32, 33]]
prices = pl.DataFrame({"timestamp": times, "asset": ["A"] * 3,
                       "close": [201., 202., 203.], "volume": [1000.] * 3})
feed = DataFeed(prices_df=prices, feed_spec={"time_contract_version": 2,
    "calendar": "NYSE", "data_frequency": "1m", "timestamp_semantics": "bar_close",
    "bar_label": "right", "bar_grid_anchor": "session_open", "partial_bar_policy": "include"})
runtime = RuntimeServices()
seen = []

def handle(context, data):
    history = data.history(asset, ["close", "volume"], 1, "1d")
    native = runtime.market.session_bars([asset.key], ["close", "volume"], bar_count=1)
    assert not native["session_closed"][0]
    assert native["timestamp"][0] == context.get_datetime().to_pydatetime()
    assert native["window_start"][0] == times[0]
    assert native["expected_bars"][0] == native["observed_bars"][0] == len(seen) + 1
    assert native.select("close", "volume").rows() == [tuple(history.iloc[0])]
    seen.append(tuple(history.iloc[0]))

performance = run_algorithm(feed=feed, assets=[asset], runtime=runtime, handle_data=handle)
assert seen == [(201., 1000.), (202., 2000.), (203., 3000.)]
assert performance.attrs["backtest_result"].fills == []
print("Partial daily history uses only observed minutes and creates no fills.")
```

Pass `history_view="prepared"` to aggregate the existing causal Data provider's
minute observations after its selected adjustment policy. Backtest owns the
simulation-clock consolidation; Data owns as-of source preparation, revisions
and adjustment basis. The published 3.2.0 fixture in
`tests/fixtures/zipline_partial_session` checks all OHLCV fields, a sparse second
asset, price carry, the first three minute cutoffs and a half-day close. Its
actual Data export keeps raw missingness; synthetic input availability is
explicitly declared at each bar close. No historical publication is inferred.

### Use a separate prepared history grid

Register named providers with `RuntimeServices(history_sources={...})`. Each
provider receives the actual decision cutoff as an aware physical instant and
returns Data's `prices` and `metadata` interface. The source carries its own
version-2 `FeedSpec`, calendar, frequency, price basis and source availability.
Native strategies read it with `runtime.history_source(name).history(...)`.
The callback adapter maps `"1d"` or `"1m"` to those names with its separate
`history_sources` argument.

The following offline example supplies completed daily observations while the
execution feed advances in minutes. Its frozen rows have explicitly declared
availability at their daily closes. In a Data workflow, the provider can instead
return `prepare_adjusted_history(..., as_of=as_of, feed_spec=daily_spec, ...)`;
aggregation must already have happened before that history preparation call.

<!-- ml4t-doc-test: zipline-independent-history-grid -->
```python
from datetime import datetime
from types import SimpleNamespace
import pandas as pd
import polars as pl
from ml4t.specs import FeedSpec
from ml4t.backtest import DataFeed, RuntimeServices
from ml4t.backtest.compat.zipline import run_algorithm
from ml4t.backtest.compat.zipline.assets import Asset

asset = Asset("A", "ALPHA", 7)
daily_spec = FeedSpec(time_contract_version=2, entity_col="asset", calendar="NYSE",
                      data_frequency="daily", timestamp_semantics="bar_close",
                      bar_label="right", bar_grid_anchor="session_open",
                      partial_bar_policy="include")
closes = [datetime.fromisoformat(s) for s in
          ["2024-07-01T20:00:00+00:00", "2024-07-02T20:00:00+00:00"]]
daily = pl.DataFrame({"timestamp": closes, "available_at": closes,
                      "asset": ["A", "A"], "close": [90., 100.]})

def prepare_daily(*, as_of, observation_before=None):
    visible = daily.filter(pl.col("available_at") <= as_of)
    if observation_before is not None:
        visible = visible.filter(pl.col("timestamp") < observation_before)
    return SimpleNamespace(prices=visible, metadata={
        "perspective": "causal", "as_of": as_of.isoformat(), "price_basis": "raw",
        "feed_spec": daily_spec.to_dict(),
    })

minutes = [datetime.fromisoformat(s) for s in
           ["2024-07-03T13:31:00+00:00", "2024-07-03T13:32:00+00:00"]]
feed = DataFeed(prices_df=pl.DataFrame({"timestamp": minutes, "asset": ["A", "A"],
                                       "close": [201., 202.], "volume": [1000., 1000.]}),
                feed_spec=daily_spec.with_overrides(data_frequency="1m"))
seen = []

def handle_data(context, data):
    history = data.history(asset, "price", 2, "1d")
    assert history.index.tolist() == [pd.Timestamp("2024-07-01"), pd.Timestamp("2024-07-02")]
    assert history.tolist() == [90., 100.]
    seen.append(data.current(asset, "price"))
    if len(seen) == 1:
        context.order(asset, 1)

performance = run_algorithm(feed=feed, assets=[asset], handle_data=handle_data,
                            runtime=RuntimeServices(history_sources={"daily": prepare_daily}),
                            history_sources={"1d": "daily"})
assert seen == [201., 202.]
fills = performance.attrs["backtest_result"].to_fills_dataframe()
assert abs(fills["price"][0] - 202.101) < 1e-12
print("Separate daily reads preserve minute execution prices.")
```

The source is cached only at its exact decision cutoff. Obtain it again after
the clock advances; retained snapshots cannot silently answer for a later
cutoff. Expected-bar windows use the source's grid. Future observations, late
revisions, imputed observations and asset keys outside the execution universe
are rejected. Source metadata is copied when read. The existing same-grid
`history_view="prepared"` checks remain in force.

This example intentionally returns the last two completed sessions. Published
Zipline's daily history during minute execution can include the current
session's in-progress daily bar. The retained published-3.2 probe
(`tests/fixtures/zipline_original/cross-frequency-reference.json`) observes
previous close 100 and current closes 201, 202 and 203, with current-session
volumes 1000, 2000 and 3000 at its first three minutes. Use the
[partial daily route](#read-the-current-partial-daily-bar) for that definition.
An explicit named daily source selects its own completed grid instead.
Declaring minute rows as daily data does not aggregate them. No partial daily
close is inferred from a future completed daily close.

The captured and installed producer/consumer tests use Data revision
`c3b9db80a66397420fb27adeccc762559dd41e07`, version
`0.2.2.dev21+gc3b9db80a`, with wheel SHA-256
`713b032ddd07b5512837e093249fed2e56aad8fabad26a6352e38164674dfe88`.
This is a qualified source build, not a claim about a published Data version.

### Use the legacy synthetic futures execution clock

With a feed declared as `calendar="us_futures"`, the callback adapter uses the
published Zipline 3.2.0 synthetic futures execution window. It starts 12 hours
and 31 minutes after the calendar open and ends one hour before its close.
For the retained March fixtures, that is 6:31 through 17:00 New York time,
630 minute callbacks per session. Scheduled market-open and market-close rules
use those execution endpoints. `before_trading_start` runs at 8:45 New York
time, before the quote update and pending fills at that instant.

The underlying calendar and captured overnight observations remain available
for history. These intervals restrict execution events; they do not relabel
prices or create missing bars. Native `Engine` usage continues to use the
full declared calendar unless the caller supplies execution windows. A
product calendar such as `CME_Equity` remains a separate choice; it is not
silently replaced by the synthetic `us_futures` calendar.

<!-- ml4t-doc-test: legacy-us-futures-clock -->
```python
from datetime import timedelta
import polars as pl
from ml4t.backtest import DataFeed, RuntimeServices
from ml4t.backtest.calendar import get_schedule
from ml4t.backtest.compat.zipline import run_algorithm
from ml4t.backtest.compat.zipline.assets import Asset

row = get_schedule("us_futures", "2025-03-03", "2025-03-03").row(0, named=True)
minutes = pl.datetime_range(row["market_open"] + timedelta(minutes=1), row["market_close"],
                             interval="1m", time_unit="ns", eager=True)
asset = Asset("ES", "ESH25", 8)
feed = DataFeed(prices_df=pl.DataFrame({"timestamp": minutes, "asset": ["ES"] * len(minutes),
                                       "close": [100.] * len(minutes), "volume": [1000.] * len(minutes)}),
                feed_spec={"time_contract_version": 2, "calendar": "us_futures",
                           "data_frequency": "1m", "timestamp_semantics": "bar_close",
                           "bar_label": "right", "bar_grid_anchor": "session_open",
                           "partial_bar_policy": "include"})
seen = []
services = RuntimeServices()

def before_trading_start(context, data):
    seen.append(context.get_datetime().isoformat())
    assert data.current(asset, "price") == 100.
    assert data.current(asset, ["price", "close", "volume"]).tolist() == [100., 100., 1000.]
    assert not data.is_stale(asset)
    assert services.market.current(["ES"], ["close", "volume"]).is_empty()

performance = run_algorithm(feed=feed, assets=[asset], runtime=services,
                            before_trading_start=before_trading_start,
                            handle_data=lambda context, data: None)
assert seen == ["2025-03-03T13:45:00+00:00"]
assert len(performance) == 630
assert str(performance.index[0]) == "2025-03-03 11:31:00+00:00"
assert str(performance.index[-1]) == "2025-03-03 22:00:00+00:00"
print("Legacy futures execution windows preserve the captured overnight history.")
```

The original-engine fixtures cover 2,880 captured minute observations, the
1,260 selected events across two sessions, scheduled endpoint callbacks,
known-price reads and the daylight-saving transition. The callback's known
`price` is a read view; no current quote or fill is synthesized at a clock
callback. This example makes no order, settlement or default futures-slippage
comparison. Those financial definitions require their separate qualifications.

Native callers can supply inclusive physical intervals through
`RuntimeServices(execution_windows=[(start, end), ...])`. Intervals must be
ordered and nonoverlapping. Feed rows outside them remain available for
causal history but do not produce market callbacks or account samples.
The adapter intersects existing caller intervals with its legacy window and
restores the caller's runtime configuration after the run. Session account
controls use the legacy execution close, and require that closing observation.
A failed control does not turn a partial capture into a completed session.

For a physical timer that is independent of observed bars, register an
explicit `RebalanceSchedule` with `runtime.scheduler.register(...,
phase=LifecyclePhase.INTRABAR, at="clock", schedule=...)`. These callbacks
receive empty data and context mappings. They run before an observation at
the same instant. The timer does not update quotes, execute fills or add an
account sample. Session expiration still follows its declared policy. A selected timer
after the last captured market event is rejected. Explicit run bounds exclude
clock registrations outside those bounds.


### Align daily reads with the legacy futures close

Published 3.2.0 daily `us_futures` callbacks also occur at 17:00 New York,
one hour before that synthetic calendar's full close. The adapter selects
the same earlier daily history endpoint using native
`history(unit="expected_bar", session_close_offset=-timedelta(hours=1))`.
Native history's default calendar grid remains unchanged. The option accepts
only nonpositive offsets on right-labelled daily expected-bar reads, with
the selected endpoint after session open.

Supply actual observations available at that earlier endpoint. A full-calendar
18:00 quote cannot be relabelled or made available at 17:00. Data bundle daily
session labels alone do not establish physical publication times; declare that
input policy explicitly. The following controlled example supplies quotes at
the captured legacy clock.

<!-- ml4t-doc-test: legacy-daily-futures-history -->
```python
from datetime import UTC, datetime, timedelta
import polars as pl
from ml4t.backtest import DataFeed, RuntimeServices
from ml4t.backtest.compat.zipline import run_algorithm
from ml4t.backtest.compat.zipline.assets import Future

asset = Future("ES", "ESH25", 8, root_symbol="ES", multiplier=50., tick_size=.25)
times = [datetime(2025, 3, day, 22, tzinfo=UTC) for day in [3, 4, 5]]
values = [100., 101., 102.]
prices = pl.DataFrame({"timestamp": times, "asset": [asset.key] * 3,
    "open": values, "high": values, "low": values, "close": values, "volume": [1000.] * 3})
feed = DataFeed(prices_df=prices, feed_spec={"calendar": "us_futures",
    "time_contract_version": 2, "data_frequency": "daily", "timestamp_semantics": "event_time",
    "bar_label": "right", "bar_grid_anchor": "session_open", "partial_bar_policy": "include"})
runtime = RuntimeServices()
seen = []

def handle(context, data):
    legacy = data.history(asset, "close", 1, "1d")
    native = runtime.market.history([asset.key], ["close"], bar_count=1, unit="expected_bar",
                                    session_close_offset=-timedelta(hours=1))
    assert native["timestamp"][0] == context.get_datetime().to_pydatetime()
    assert native["close"][0] == legacy.iloc[0] == data.current(asset, "price")
    seen.append(native["close"][0])

performance = run_algorithm(feed=feed, assets=[asset], runtime=runtime, handle_data=handle)
assert seen == values
assert performance.attrs["backtest_result"].fills == []
print("Daily futures history uses actual quotes at the legacy execution close.")
```

The captured `tests/fixtures/zipline_daily_futures` original-engine workflows
also check UTC and naive New York input, a DST weekend, pre-event OHLCV reads
and actual Data-prepared history. A later full-calendar quote remains excluded
at the earlier cutoff. This clock qualification does not establish settlement
or variation-cash parity.

### Read completed bars before trading

At a clock before a market observation, the Zipline read policy projects
OHLCV from the preceding point on the declared history grid. It does not fill
missing OHLCV fields. `price` keeps its explicit known-price policy;
`last_traded` retains the actual trade timestamp, and `is_stale` examines the
completed bar's volume without indexing an absent current quote. Native
`market.current` remains strict and can still be empty at the same clock.

The default projection uses expected bars with a calendar and observations
without one. An explicit `history_unit` selects the same supported unit as
history requests. `history_view="prepared"` also applies to this historical
pre-event projection and requires the causal Data provider. It does not
reprice executable quotes. The installed producer test checks a known 2:1
split: prepared prior price/volume are 100/20, raw prior values remain 200/10,
and the active raw quote remains absent until its actual market event.

The published 3.2.0 minute fixture checks current price, OHLCV, last trade,
staleness and tradability at the real futures callback clocks.
Those before-trading values come from 8:44 New York time at an 8:45 callback.
Supply complete clock metadata for expected-bar reads. No future completed
daily close is inferred by this projection.

Prepared providers used before a market event must accept both `as_of` and
`observation_before`. `as_of` remains the actual decision time: source facts
published exactly then are available. `observation_before` excludes price
observations at that time, because the market callback has not observed them
yet. During a market callback, Backtest calls the provider with only `as_of`,
so existing providers used only there remain compatible.

Filter the input price observations before calculating adjusted history.
For example, a caller-owned Data provider can select
`prices.filter(pl.col(spec.timestamp_col) < observation_before)` when the
boundary is present, then call `prepare_adjusted_history` with the unchanged
`as_of`. The boundary is a Backtest provider argument, not an argument to
Data's preparation function. Filtering only the prepared output is too late:
an excluded close could already have influenced older adjusted prices.
Backtest also rejects prepared output containing an excluded observation.

A native cash-dividend factor that requires an ex-date close cannot be
calculated before that close is observed. The installed producer regression
checks that Data rejects this request, then computes the factor once the
market event has occurred. Do not substitute a future close or move `as_of`
back by a nanosecond; either would change the information available to the
strategy. Split factors whose required facts are already available can be
used before the market event.
