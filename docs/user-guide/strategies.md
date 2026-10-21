# Strategies

Every strategy subclasses `Strategy` and implements `on_data`. The broker is your interface to the market -- use it to submit orders, query positions, and set risk rules.

The `MyStrategy` class sketches the callback interface. Later `broker` calls belong inside a callback; the [quickstart](../getting-started/quickstart.md) supplies a complete, checked run.

## Strategy Interface

```python
from ml4t.backtest import Strategy

class MyStrategy(Strategy):
    def on_prepare(self, broker, config=None):
        """Called once with resolved configuration and no future feed data. Optional."""
        pass

    def on_start(self, broker):
        """Called once before the first bar. Optional.

        Use this to set position rules, initialize state, or log config.
        """
        pass

    def on_data(self, timestamp, data, context, broker):
        """Called on every bar. Required.

        Args:
            timestamp: Current bar's datetime
            data: Dict of {asset: bar_dict}
            context: Dict of context data (if provided via DataFeed)
            broker: Broker for orders and position queries
        """
        pass

    def on_end(self, broker):
        """Called once after the last bar. Optional.

        Use this for final cleanup or logging.
        """
        pass
```

The callback order is `on_start`, `on_prepare`, one `on_data` call for each accepted
market event, and finally `on_end`. `on_prepare` receives the resolved config but no
future feed timestamps. Use `broker.register_target_intent(...)` during `on_prepare`
for a portfolio decision that was made before an opening auction. The
[Execution Semantics](execution-semantics.md#causal-strategy-lifecycle) reference
defines cutoff validation, opening fills, and position-rule activation.

### What `data` Contains

```python
data = {
    "AAPL": {
        "open": 150.0,
        "high": 152.0,
        "low": 149.5,
        "close": 151.0,
        "volume": 1_000_000,
        "signals": {"prediction": 0.85, "momentum": 0.12},
    },
    "MSFT": { ... },
}
```

Signals come from the optional signals DataFrame passed to DataFeed. Access them via `bar.get("signals", {}).get("column_name")`.

## Use causal history and session callbacks

These additional runtime services are development-source features, outside the
preserved 0.1.13 release candidate.

Pass a `RuntimeServices` instance to the existing `Engine`. The strategy's `context`
remains the input feed context. Rows before `start` supply history but cannot trade
or create account observations. The service becomes unavailable after the run.
The same `runtime` argument is accepted by `Engine.from_config` and `run_backtest`.

This example requires actual timestamped closing bars and Specs 0.2.0. It does
not reinterpret midnight session labels as close instants. Runtime services
require a declared `event_time` or `bar_close` timestamp role. Convert session
labels to physical instants before using these services.

<!-- ml4t-doc-test: runtime-causal-session -->
```python
from datetime import UTC, datetime
import polars as pl
from ml4t.specs import FeedSpec, LifecyclePhase, TimestampSemantics
from ml4t.backtest import BacktestConfig, DataFeed, Engine, RuntimeServices, Strategy

dates = [datetime(2025, 1, day, 21, tzinfo=UTC) for day in [2, 3, 6]]
prices = pl.DataFrame({
    "timestamp": dates,
    "asset": ["TEST"] * 3,
    "close": [100.0, 101.0, 102.0],
})
services = RuntimeServices(start=datetime(2025, 1, 6, tzinfo=UTC))
trace = []

def before(timestamp, data, context, broker):
    history = services.market.history(["TEST"], ["close"], bar_count=2)
    assert history["close"].to_list() == [100.0, 101.0]
    assert data == {} and context == {}
    trace.append(("before", timestamp))

def at_close(timestamp, data, context, broker):
    assert services.market.current(["TEST"], ["close"])["close"].to_list() == [102.0]
    trace.append(("close", timestamp))

services.scheduler.register(
    before, phase=LifecyclePhase.PRE_OPEN, at="session_open", minutes_before=45
)
services.scheduler.register(at_close, phase=LifecyclePhase.CLOSE, at="session_close")

class Observe(Strategy):
    def on_data(self, timestamp, data, context, broker):
        trace.append(("data", timestamp))

spec = FeedSpec(time_contract_version=2, calendar="NYSE", timezone="UTC",
                data_frequency="1d", timestamp_semantics=TimestampSemantics.BAR_CLOSE,
                bar_label="right")
result = Engine(DataFeed(prices_df=prices, feed_spec=spec), Observe(),
                BacktestConfig(initial_cash=10000), runtime=services).run()
assert trace == [("before", datetime(2025, 1, 6, 13, 45, tzinfo=UTC)),
                 ("data", dates[-1]), ("close", dates[-1])]
assert result.to_equity_dataframe()["timestamp"].to_list() == [dates[-1]]
print("Runtime callbacks and causal warmup verified.")
```

By default, scheduled clocks also obey `RuntimeServices.start`. For a run that
starts at the first tradable minute, register a pre-open callback with
`include_start_session=True` to include that session's earlier preparation clock.
This option applies only to `PRE_OPEN`; market observations before `start`
remain warmup rows. The Zipline callback adapter selects it for
`before_trading_start`.

History's default `unit="observation"` counts observed price instants, retaining
missing asset rows. `known_at_cols={"field": "publication_timestamp_column"}` masks fields until
available. `forward_fill=True` is an explicit read view with source timestamps;
it never adds executable quotes or volume. Missing raw volume remains unknown
in the view. `last_traded` requires actual positive observed volume;
`last_observed` works with quotes alone. `can_trade` checks the current quote,
not buying power or all order controls.

Raw reads retain non-finite input values. In a price read view,
`finite_prices_only=True` treats non-finite quote prices as missing, including
when selecting a previous value with `forward_fill`. The option applies to
price roles and leaves genuine feature NaNs unchanged. It does not alter
execution data or create volume. The Zipline callback adapter selects this
price-view policy explicitly.

`runtime.market.session_bars(assets, fields, bar_count=...)` consolidates a
declared intraday grid into causal session OHLCV, including the current partial
session. It requires an explicit calendar. The returned `timestamp` is capped
at the active simulation clock, with `window_start`, `session_closed`,
`expected_bars` and `observed_bars` describing coverage. Unknown minute volume
leaves the total unknown. `view="prepared"` uses Data's causal input and
adjustment policy before consolidation. Ordinary `history(unit="session")`
continues to return the individual observations without aggregation. See the
[runnable partial-session example](migrate-from-zipline.md#read-the-current-partial-daily-bar)
for native reads alongside the original callback projection.

For an explicit earlier daily read endpoint, `history(unit="expected_bar",
session_close_offset=-timedelta(hours=1), ...)` selects grid points one hour
before the calendar close. It requires right-labelled daily bars and a
nonpositive offset that remains after session open. Only observations actually
available at those instants can be returned; the option does not shift quotes,
calendar sessions or execution. The default grid is unchanged. See the
[daily futures example](migrate-from-zipline.md#align-daily-reads-with-the-legacy-futures-close).

Register callbacks before execution or in `on_start`/`on_prepare`. Registrations
made during a run are removed afterward; registrations made beforehand persist
when reusing the service for a new Engine. Callbacks use the existing lifecycle
transaction and roll back broker mutations if they fail.

Pre-open callbacks are calendar events with empty data/context. They use only
previously available history and cannot fill orders immediately. Other scheduled
callbacks require an observed price event at the requested instant; the last
input row is never promoted to an official session close. Offsets count elapsed
minutes from the named endpoint. Calendar events can occur between sparse price
observations without adding price bars or account marks. A disabled calendar
rejects session schedules.

Opening-auction callbacks require `event_time` opening observations. Completed
bars cannot supply an opening quote. Where continuous sessions meet at the same
instant, a right-labeled closing bar and its close callback run before the next
session's pre-open callback. That callback can read the bar that just closed.

### Bound stale reads

Use a per-field `forward_fill` mapping to carry prices while leaving volume or
other fields missing. Forward filling observed volume or quote depth is rejected.
`max_age` bounds a carried value by elapsed time from its original observation to
the requested history/current instant. An expired value becomes `None`; its
`<field>_observed_at` column retains the actual old source instant.

<!-- ml4t-doc-test: runtime-stale-views -->
```python
from datetime import UTC, datetime, timedelta
import polars as pl
from ml4t.backtest import BacktestConfig, DataFeed, Engine, RuntimeServices, Strategy

times = [datetime(2025, 1, day, 21, tzinfo=UTC) for day in [6, 7]]
prices = pl.DataFrame({"timestamp": times, "asset": ["A", "B"],
                       "close": [100.0, 50.0], "volume": [10.0, 20.0]})
runtime = RuntimeServices()
seen = []

class Read(Strategy):
    def on_data(self, timestamp, data, context, broker):
        if timestamp == times[1]:
            values = runtime.market.current(["A"], ["price", "volume"],
                forward_fill={"price": True, "volume": False}, max_age=timedelta(days=1))
            assert values["price"].to_list() == [100.0]
            assert values["volume"].to_list() == [None]
            assert values["price_observed_at"].to_list() == [times[0]]
            assert runtime.market.observation_age("A") == timedelta(days=1)
            assert runtime.market.is_stale("A")
            assert not runtime.market.can_trade("A")
            seen.append(values["price"][0])

result = Engine(DataFeed(prices_df=prices, feed_spec={"timestamp_semantics": "event_time"}),
                Read(), BacktestConfig(calendar=None), runtime=runtime).run()
assert seen == [100.0] and result.to_fills_dataframe().is_empty()
print("Stale price views retain provenance and cannot create a tradable quote.")
```

`observation_age` measures time since the last known non-null observation;
`is_stale` compares that age with a nonnegative `max_age` (zero by default).
An asset with no observed history has age `None` and `is_stale=False`.
Use `can_trade` to distinguish that missing or unusable quote from an executable
one. Recency, validity and buying power are separate checks.

Before an opening event, ordinary `current` remains empty. An explicit stale
read can return the prior close with its original observation timestamp. Orders
from that callback wait for an eligible real quote; reading a stale price does
not make it an execution price.

### Select a history window

`unit="expected_bar"` counts the declared calendar grid, including missing bars
as nulls. It requires a version-2 feed with frequency, `bar_grid_anchor`,
`bar_label` and `partial_bar_policy`. The grid restarts at each tradable interval
open, excludes exchange breaks and follows early closes. A visible off-grid
price in the requested window raises an error.

`unit="session"` selects observations from the last `bar_count` exchange sessions,
including the active session up to the decision cutoff. It retains original
instants and does not aggregate OHLCV into daily bars. Both calendar units reject
`calendar=None`.

`unit="elapsed"` instead requires a positive `duration=datetime.timedelta(...)`
and no `bar_count`. It selects `(cutoff-duration, cutoff]`, excluding the current
event during initialization or pre-open access. It also works without a calendar.

<!-- ml4t-doc-test: runtime-history-windows -->
```python
from datetime import UTC, datetime, timedelta
import polars as pl
from ml4t.specs import FeedSpec
from ml4t.backtest import BacktestConfig, DataFeed, Engine, RuntimeServices, Strategy

dates = [datetime(2025, 1, 2, 14, minute, tzinfo=UTC) for minute in [35, 45]]
prices = pl.DataFrame({"timestamp": dates, "asset": ["TEST"] * 2,
                       "close": [100.0, 110.0]})
spec = FeedSpec(time_contract_version=2, calendar="NYSE", data_frequency="5m",
                timestamp_semantics="bar_close", bar_grid_anchor="session_open",
                bar_label="right", partial_bar_policy="include")
services = RuntimeServices()

class ReadWindows(Strategy):
    def on_data(self, timestamp, data, context, broker):
        if timestamp == dates[-1]:
            grid = services.market.history(["TEST"], ["close"], bar_count=2,
                                           unit="expected_bar")
            assert grid["close"].to_list() == [None, 110.0]
            elapsed = services.market.history(["TEST"], ["close"], unit="elapsed",
                                              duration=timedelta(minutes=5))
            assert elapsed["timestamp"].to_list() == [dates[-1]]

result = Engine(DataFeed(prices_df=prices, feed_spec=spec), ReadWindows(),
                BacktestConfig(), runtime=services).run()
assert result.to_equity_dataframe()["timestamp"].to_list() == dates
print("History windows retain the gap without creating an executable bar.")
```

### Read a frozen auxiliary source

`AuxiliaryData` registers a prepared research table under its original source
identifier. Prepare downloads, date parsing, dated symbol mapping and preprocessing
before the run. Reuse Data's preparation APIs when they own those operations.
Backtest stores a copy of the declared columns and their input-content hash; it
performs no network request or implicit symbol substitution.

Supply the existing `FeedSpec` with physical observation instants, one stable
entity column, and an explicit `known_at_cols` entry for each payload field.
Unknown publication times remain unavailable. Duplicate observation/entity keys
raise; resolve revisions or economically distinct events upstream instead of
silently keeping one row. Polars, lazy Polars and named-column/index Pandas inputs
are supported. Saved CSV or Parquet inputs can use the same route after explicit
schema/date parsing. Preserve timezone and nanosecond precision during preparation.

<!-- ml4t-doc-test: runtime-auxiliary-source -->
```python
from datetime import UTC, datetime, timedelta
import polars as pl
from ml4t.specs import FeedSpec
from ml4t.backtest import AuxiliaryData, BacktestConfig, DataFeed, Engine, RuntimeServices, Strategy

source_id = "https://example.org/research/captured.csv"
first = datetime(2025, 1, 6, 21, tzinfo=UTC)
second = first + timedelta(days=1)
source = AuxiliaryData(
    source_id,
    pl.DataFrame({"observed": [first], "asset": ["TEST"],
                  "score": [10.0], "published": [second]}),
    feed_spec=FeedSpec(timestamp_col="observed", entity_col="asset",
                       timestamp_semantics="event_time", calendar=None),
    known_at_cols={"score": "published"},
    preparation={"symbol_mapping": "captured stable IDs", "timezone": "UTC"},
)
services = RuntimeServices(auxiliary=[source])
seen = []

class ReadResearch(Strategy):
    def on_data(self, timestamp, data, context, broker):
        view = services.auxiliary(source_id)
        seen.append(view.current(["TEST"], ["score"], forward_fill=True)["score"][0])

prices = pl.DataFrame({"timestamp": [first, second], "asset": ["TEST"] * 2,
                       "close": [100.0, 101.0]})
result = Engine(DataFeed(prices_df=prices, feed_spec={"timestamp_semantics": "event_time"}),
                ReadResearch(), BacktestConfig(calendar=None), runtime=services).run()
assert seen == [None, 10.0]
assert result.to_fills_dataframe().is_empty()
print("Auxiliary publication does not create an executable quote or market event.")
```

`services.auxiliary(source_id)` requires an exact registered identifier. Its
`current`, `history` and `last_observed` methods expose only declared payload
fields; publication and undeclared columns cannot be queried as values.
`history` supports observation and elapsed windows. Forward filling is explicit
and can use per-field policies and `max_age`, as with feature reads. A request
that combines a payload name with the same generated `field_observed_at`
provenance name raises rather than overwriting either value.

The view's `assets` contains stable keys with at least one published observation
known so far. Its `metadata` returns a copy of the source identifier, preparation
declarations, input-content hash, field mappings and `FeedSpec`. Views expire
when their owning run ends, including empty-source views. Auxiliary values never
become broker prices, traded volume or trading events. For a Zipline `fetch_csv`
port, register the captured source with its original URL; parsing and preprocessing
remain explicit preparation steps.

Auxiliary sources can declare `eligible_cols={"score": "eligible"}` alongside
`known_at_cols`. The eligibility columns must be Boolean. False or unknown
eligibility hides the associated payload at its publication cutoff and prevents
an older value from being carried forward across that exclusion. The source
producer supplies the mask; Backtest computes no membership or feature values.

For sources needed only by one run, use
`with services.auxiliary_scope((source,)):` around `Engine(..., runtime=services).run()`.
The scope must begin before binding the engine. It preserves the caller's input
registry after completion or errors, permits repeated references to the same
frozen source object, and rejects a different source using an existing identifier.

### Resolve dated instruments and enforce their lifetimes

Pass Data's offline `InstrumentRegistry` and the same `FeedSpec` used to construct
it as `RuntimeServices(instruments=registry, instrument_feed_spec=identity_spec)`.
Backtest does not fetch symbols, infer delistings from missing bars, or build a
second asset database. Use stable IDs in prices and orders. Their dtype must match
the registry's declared entity column.

During the run, `services.assets.resolve(symbol, exchange=..., on=...)` delegates
dated symbol resolution to Data. `active(on=..., universe=...)` returns Data's
selected identities and membership; `is_active(asset)` checks current identity
coverage. Every query supplies the physical simulation instant as `as_of`.
Omitting `on` uses that instant, with the registry's declared session-label
conversion. Future revisions cannot become visible early. Saved views stop
working when their owning run ends.

Inactive instruments reject new orders before allocating an order ID. A pre-open
order instead checks identity coverage at its earliest eligible opening instant,
using only facts known at submission. It still cannot fill before that opening
or obtain a future quote. Pending orders are cancelled when their instrument
becomes inactive, with `cancellation_reason="instrument_inactive"` and the actual
event timestamp. `broker.get_orders(pending_only=True)` returns detached snapshots.
A real input quote for an unknown or inactive instrument raises before updating
account marks; fix the input or declared lifetime rather than silently dropping it.

The default `inactive_position_policy="retain_last_mark"` retains an existing
holding at its last eligible observed mark and timestamp. It does not infer a
liquidation, dividend or zero valuation. Choose `"require_flat"` to fail before
account mutation if a position reaches an inactive event; close it explicitly
while it is active. Corporate actions and futures roll execution have separate
accounting contracts.

This recipe uses the same qualified Data development artifact and Specs 0.2.0
identified in the adjusted-history section below. It requires that Data producer;
it is not an API claim about an older published Data package.

<!-- ml4t-doc-optional: runtime-instruments -->
```python
from datetime import UTC, date, datetime
import polars as pl
from ml4t.data.assets import InstrumentRegistry
from ml4t.specs import FeedSpec
from ml4t.backtest import BacktestConfig, DataFeed, Engine, RuntimeServices, Strategy

identity_spec = FeedSpec(time_contract_version=2, entity_col="asset",
                         timestamp_col="valid_from", calendar="NYSE",
                         timezone="America/New_York", timestamp_semantics="session_label")
identities = pl.DataFrame({
    "asset": ["ID1", "ID1", "ID2"], "symbol": ["ALPHA", "BETA", "ALPHA"],
    "exchange": ["NYSE"] * 3, "instrument_type": ["equity"] * 3,
    "valid_from": [date(2025, 1, 1), date(2025, 1, 8), date(2025, 1, 8)],
    "valid_to": pl.Series([date(2025, 1, 8), None, None], dtype=pl.Date),
    "available_at": [datetime(2025, 1, 1, tzinfo=UTC)] * 3,
    "source": ["example"] * 3, "revision_id": ["r1"] * 3,
})
registry = InstrumentRegistry.from_frame(identities, feed_spec=identity_spec)
services = RuntimeServices(instruments=registry, instrument_feed_spec=identity_spec)
seen = []

class ReadSymbols(Strategy):
    def on_data(self, timestamp, data, context, broker):
        seen.append(services.assets.resolve("ALPHA"))

prices = pl.DataFrame({
    "timestamp": [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [6, 8]],
    "asset": ["ID1", "ID2"], "close": [100.0, 50.0],
})
result = Engine(DataFeed(prices_df=prices, feed_spec={"timestamp_semantics": "event_time"}),
                ReadSymbols(), BacktestConfig(calendar=None), runtime=services).run()
assert seen == ["ID1", "ID2"]
```

### Read Data-prepared adjusted history

The optional `history_provider` receives `as_of` at the active decision cutoff.
It returns Data's prepared prices and metadata. Request `view="prepared"` in
`history` or `current`; the default `view="raw"` reads the untransformed execution
input. `market.prepared_metadata` returns a defensive copy of the provider's
cutoff, price basis and preparation declarations.

Backtest rejects retrospective or mismatched cutoffs, unavailable revisions,
changed timestamp roles/calendars and invented observation keys. Preparation
does not add executable quotes or apply account corporate actions. Split and
dividend accounting are separate from a historical read view.

The Data producer was qualified from source
[`fd646c982ed12fca193f6cf713be1ac8803b17df`](https://github.com/ml4t/data/blob/fd646c982ed12fca193f6cf713be1ac8803b17df/src/ml4t/data/adjustments/history.py),
development artifact `0.2.2.dev55+gfd646c982`, with Specs 0.2.0. This is not a
claim that an older published Data package contains this API. With that producer,
the preparation recipe is:

```python
from ml4t.data.adjustments import prepare_adjusted_history
from ml4t.backtest import RuntimeServices

def prepare(*, as_of):
    return prepare_adjusted_history(
        raw_source_prices, captured_actions, as_of=as_of, feed_spec=source_spec,
        basis="split_adjusted", action_coverage="complete",
    )

services = RuntimeServices(history_provider=prepare)
```

Here `raw_source_prices` retains source revision IDs, `available_at` instants,
stable asset keys and physical observation instants. `captured_actions` supplies
available event terms. `source_spec` declares their mappings and calendar.
Data selects revisions and adjustment factors; Backtest selects the runtime
window. A provider may prepare from captured in-memory inputs and must not
perform undeclared network fetches during simulation.

### Enforce prepared feature and context availability

Declare per-field `signal_known_at_cols` and `context_known_at_cols` on the runtime.
Unavailable values become `None` in the actual strategy payload. For prepared
panels, `signal_eligible_cols` additionally requires a true Boolean eligibility
value for each named feature. These mappings do not change prices or compute
features. Declare every field whose publication differs from its observation
clock, including eligibility or universe metadata when reading it directly.

<!-- ml4t-doc-test: runtime-feature-availability -->
```python
from datetime import UTC, datetime
import polars as pl
from ml4t.backtest import BacktestConfig, DataFeed, Engine, RuntimeServices, Strategy

dates = [datetime(2025, 1, day, 21, tzinfo=UTC) for day in [2, 3, 6]]
prices = pl.DataFrame({"timestamp": dates, "asset": ["TEST"] * 3,
                       "close": [100.0] * 3})
signals = pl.DataFrame({"timestamp": dates, "asset": ["TEST"] * 3,
                        "alpha": [10.0, -1.0, 100.0],
                        "available_at": [dates[1], dates[1], dates[2]],
                        "eligible": [True, True, False]})
runtime = RuntimeServices(signal_known_at_cols={"alpha": "available_at"},
                           signal_eligible_cols={"alpha": "eligible"})
seen = []

class ReadFeatures(Strategy):
    def on_data(self, timestamp, data, context, broker):
        value = data["TEST"]["signals"]["alpha"]
        seen.append(value)
        assert runtime.signals.current(["TEST"], ["alpha"])["alpha"].to_list() == [value]
        if value is not None and value > 0:
            broker.submit_order("TEST", 1)

result = Engine(DataFeed(prices_df=prices, signals_df=signals,
                        feed_spec={"timestamp_semantics": "bar_close"}),
                ReadFeatures(), BacktestConfig(calendar=None), runtime=runtime).run()
assert seen == [None, -1.0, None]
assert result.to_fills_dataframe().is_empty()
print("Publication and eligibility boundaries verified in strategy delivery.")
```

`runtime.signals` reads prepared per-asset fields on the same simulation cutoff as
the market service. It exposes `current`, `history` and `last_observed`, with no
executable quote methods. A feature named `close` remains a feature. Its history
supports observation counts and elapsed durations; Engineer owns feature-grid
preparation and the lookback method that produced each feature.

### Consuming an Engineer feature panel

Development source accepts `DataFeed(feature_spec=...)` with an Engineer
`FeatureSpec` or its saved `to_dict()` mapping. Supply the prepared panel as
`signals_df`, or use `signals_path` for Parquet. Backtest projects the declared
decision instant onto the feed clock and retains the original observation key.
The manifest's publication and eligibility declarations apply automatically,
including when no explicit `RuntimeServices` object is supplied. A conflicting
runtime mapping raises before strategy initialization.

The following consumer example uses saved feature values. Engineer computes
those values and selects the eligible universe. Backtest performs no feature
calculation, adjustment or membership inference.

<!-- ml4t-doc-test: prepared-feature-panel -->
```python
from datetime import UTC, date, datetime
import polars as pl
from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.config import ExecutionMode

decision = datetime(2025, 7, 9, 17, tzinfo=UTC)
panel = pl.DataFrame({
    "timestamp": [date(2025, 7, 9)] * 2,
    "asset": ["A", "B"], "average": [2.0, 20.0],
    "decision_time": [decision] * 2,
    "available_at": [datetime(2025, 7, 9, 16, tzinfo=UTC)] * 2,
    "eligible": [True, False],
})
feature_spec = {
    "artifact_id": "three-observation-mean", "kind": "features", "version": 1,
    "schema": {"timestamp_col": "timestamp", "entity_col": "asset",
               "feature_columns": ["average"], "decision_time_col": "decision_time",
               "available_at_col": "available_at", "eligible_col": "eligible"},
    "definition": {
        "feed_spec": {"time_contract_version": 2, "calendar": None,
                      "timestamp_semantics": "session_label", "entity_col": "asset"},
        "window_unit": "observation", "price_basis": "raw", "lookbacks": {"average": 3},
    },
}
prices = pl.DataFrame({"timestamp": [decision] * 2, "asset": ["A", "B"],
                       "close": [100.0, 100.0]})
feed = DataFeed(prices_df=prices, signals_df=panel, feature_spec=feature_spec,
                feed_spec={"time_contract_version": 2, "calendar": None,
                           "timestamp_semantics": "event_time"})

class Select(Strategy):
    def on_data(self, timestamp, data, context, broker):
        assert data["A"]["signals"]["average"] == 2.0
        assert data["B"]["signals"]["average"] is None
        broker.submit_order("A", 1)

result = Engine(feed, Select(), BacktestConfig(calendar=None,
                execution_mode=ExecutionMode.SAME_BAR)).run()
assert result.to_fills_dataframe()["asset"].to_list() == ["A"]
assert feed.signals["feature_observation_time"].to_list() == panel["timestamp"].to_list()
print("Prepared features consumed at the declared decision instant.")
```

Polars lazy panels are collected once before simulation; multi-chunk frames keep
their rows. Pandas panels may use columns or named index levels for declared
keys. Primitive nullable values, aware timestamps and integer IDs do not require
PyArrow conversion. Supply explicit parsing and dtypes when loading CSV before
passing the resulting frame. The adapter never fetches a dataset during a run.

Only declared features and the retained clock/universe metadata enter signal
delivery. `feed.feature_metadata` returns a defensive copy of the original spec,
column projection and excluded-column list. The feature calculation's calendar,
window unit and price basis remain in that spec; they are not rewritten as the
simulation's settings. This handoff does not adjust execution prices.

The producer/consumer tests rebuild captured panels through installed Engineer
`0.1.7.dev58+g093b72d68` from
[revision 093b72d](https://github.com/ml4t/engineer/blob/093b72d6801cf2b279f1aa682c6d2cabe45d494d/src/ml4t/engineer/features/preparation.py),
using published Specs 0.2.0. That Engineer artifact is a source-qualified development
build. This statement does not attribute its preparation API to PyPI Engineer 0.1.7.

Known feature rows at the decision instant can be visible during pre-open, when
`available_at <= decision_time`; opening quote observations remain excluded until
their market phase. Explicit feature forward filling retains source timestamps.
It stops at a known ineligible row, while unavailable future rows cannot erase
valid earlier history. Forward filling changes the read view, not the strategy's
raw signal payload or executable prices.

Engineer-prepared panels retain observation time, decision time, availability and
eligibility separately. For a default single-asset-key panel, an explicit input
projection can preserve the observation key while selecting the decision clock:

```python
signals = panel.rename({"timestamp": "feature_observed_at", "decision_time": "timestamp"})
```

Use the actual artifact's `FeatureSchema` mappings for custom columns. Signal and
price asset key dtypes must agree. Duplicate signal keys, duplicate context clocks
with publication declarations, and missing or mistyped declared publication
columns fail before strategy initialization. Prepared Pipeline and auxiliary
source adapters are additional migration work; this example does not execute a
legacy Pipeline object.

## Broker Methods

### Submitting Orders

```python
# Market order: buy 100 shares
broker.submit_order("AAPL", 100)

# Short sell: negative quantity
broker.submit_order("AAPL", -100)

# Limit order
from ml4t.backtest.types import OrderType
broker.submit_order("AAPL", 100, order_type=OrderType.LIMIT, limit_price=150.0)

# Stop order
broker.submit_order("AAPL", 100, order_type=OrderType.STOP, stop_price=145.0)

# Stop-limit order
broker.submit_order("AAPL", 100, order_type=OrderType.STOP_LIMIT,
                    stop_price=145.0, limit_price=144.0)

# Market-on-close order
broker.submit_order("AAPL", 100, order_type=OrderType.MOC)
```

### Bracket Orders

Submit entry + take-profit + stop-loss in one call:

```python
orders = broker.submit_bracket(
    asset="AAPL",
    quantity=100,
    take_profit=165.0,  # TP target price
    stop_loss=145.0,    # SL price
)
# Returns (entry_order, tp_order, sl_order) or None if rejected
```

### Querying Positions

```python
# Single position
pos = broker.get_position("AAPL")
if pos is not None:
    print(pos.quantity, pos.entry_price, pos.bars_held)

# All positions
for asset, pos in broker.get_positions().items():
    print(f"{asset}: {pos.quantity} shares @ ${pos.entry_price:.2f}")

# Account value and cash
equity = broker.get_account_value()
cash = broker.get_cash()
```

### Closing Positions

```python
# Close a specific position
broker.close_position("AAPL")

# Close at the current bar close
broker.close_position("AAPL", order_type=OrderType.MOC)
```

### Rebalancing

```python
# Target weight allocation
broker.rebalance_to_weights({
    "AAPL": 0.30,   # 30% of portfolio
    "MSFT": 0.30,
    "GOOG": 0.20,
    # Remaining 20% stays in cash
})
```

### Order Management

```python
# Cancel a pending order
broker.cancel_order(order_id)

# Check rejected orders
rejected = broker.get_rejected_orders()
for order in rejected:
    print(f"{order.asset}: {order.rejection_reason}")
```

## Setting Risk Rules

Set position rules in `on_start`. They apply automatically on every bar:

```python
from ml4t.backtest import StopLoss, TakeProfit, TrailingStop, RuleChain

class ProtectedStrategy(Strategy):
    def on_start(self, broker):
        broker.set_position_rules(RuleChain([
            StopLoss(pct=0.05),
            TakeProfit(pct=0.15),
            TrailingStop(pct=0.03),
        ]))
```

See [Risk Management](risk-management.md) for the full rule catalog and composition patterns.

## Strategy Templates

The library includes four ready-to-use templates. Subclass them and override the decision methods.

### SignalFollowingStrategy

For ML predictions, technical indicators, or any pre-computed signal:

```python
from ml4t.backtest.strategies.templates import SignalFollowingStrategy

class MyMLStrategy(SignalFollowingStrategy):
    signal_column = "rf_prediction"  # Column name in signals DataFrame
    position_size = 0.05             # 5% of equity per position

    def should_enter_long(self, signal):
        return signal > 0.7

    def should_exit(self, signal):
        return signal < 0.3
```

### MomentumStrategy

Enter on positive momentum, exit on negative:

```python
from ml4t.backtest.strategies.templates import MomentumStrategy

class MyMomentum(MomentumStrategy):
    lookback = 60            # 60-bar momentum window
    entry_threshold = 0.10   # Enter on 10% return
    exit_threshold = 0.0     # Exit when momentum turns negative
    position_size = 0.10     # 10% of equity
```

### MeanReversionStrategy

Buy oversold, sell at reversion:

```python
from ml4t.backtest.strategies.templates import MeanReversionStrategy

class MyMeanReversion(MeanReversionStrategy):
    lookback = 30
    entry_zscore = -2.5   # Enter when 2.5 std below mean
    exit_zscore = 0.5     # Exit when 0.5 std above mean
    position_size = 0.10
```

### LongShortStrategy

Rank assets and go long top N, short bottom N:

```python
from ml4t.backtest.strategies.templates import LongShortStrategy

class MyLongShort(LongShortStrategy):
    signal_column = "alpha_score"
    long_count = 10
    short_count = 10
    position_size = 0.05
    rebalance_frequency = 21  # Monthly rebalance
```

## Patterns

### Multi-Asset with Position Sizing

```python
class SizedStrategy(Strategy):
    max_positions = 10

    def on_data(self, timestamp, data, context, broker):
        positions = broker.get_positions()
        if len(positions) >= self.max_positions:
            return

        equity = broker.get_account_value()
        per_position = equity / self.max_positions

        for asset, bar in data.items():
            if broker.get_position(asset) is not None:
                continue
            signal = bar.get("signals", {}).get("score", 0)
            if signal > 0.7:
                shares = int(per_position / bar["close"])
                if shares > 0:
                    broker.submit_order(asset, shares)
```

### Per-Asset Risk Rules

```python
class AssetSpecificRules(Strategy):
    def on_start(self, broker):
        # Different rules for different assets
        broker.set_position_rules(
            RuleChain([StopLoss(pct=0.03), TrailingStop(pct=0.02)]),
            asset="AAPL",
        )
        broker.set_position_rules(
            RuleChain([StopLoss(pct=0.10), TakeProfit(pct=0.20)]),
            asset="BTC",
        )
        # Global fallback for any other asset
        broker.set_position_rules(StopLoss(pct=0.05))
```

## Best Practices

1. **Avoid look-ahead bias** -- only use data from the current bar and earlier
2. **Account for costs** -- test with realistic commission and slippage
3. **Size positions conservatively** -- don't risk more than 1-2% per trade
4. **Use NEXT_BAR mode** -- for production strategies, avoid SAME_BAR
5. **Validate with profiles** -- compare results across framework profiles

## In the book

Chapter 16, Section 16.3, [Single asset ml4t-backtest](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/16_strategy_simulation/04_single_asset_ml4t_backtest.ipynb) implements an event-driven strategy and reconciles its trade log. This page describes the reusable `Strategy` interface.

## Next Steps

- [Book Guide](../book-guide/index.md) -- chapter and case-study map for strategy workflows
- [Stateful Strategies](stateful-strategies.md) -- advanced patterns that require event-driven execution
- [Risk Management](risk-management.md) -- full rule catalog and composition
- [Order Types](orders.md) -- limit, stop, bracket orders in detail
- [Data Feed](data-feed.md) -- how data and signals are structured
