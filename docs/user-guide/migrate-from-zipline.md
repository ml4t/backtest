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
