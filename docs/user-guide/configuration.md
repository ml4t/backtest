# Configuration

`BacktestConfig` records the execution and account settings used for a run. Named profiles choose settings for the supported comparison scenarios; they do not reproduce every behavior of another framework.

It is also the canonical serializable backtest preset:

- pass a partial config as a Python `dict`, YAML, or JSON-equivalent mapping
- let `BacktestConfig` fill in defaults
- persist the fully resolved snapshot from the executed result

This keeps the input simple while still giving you an exact replayable record of what ran.

Configuration snippets isolate individual settings and assume the named classes and enums have been imported. The [profiles tutorial](../tutorials/profiles-and-parity.md) runs a complete comparison with declared inputs.

## Creating a Config

```python
from ml4t.backtest import BacktestConfig

# Sensible defaults
config = BacktestConfig()

# From a framework preset
config = BacktestConfig.from_preset("backtrader")

# From structured broker assumptions
config = BacktestConfig.from_assumptions(
    broker="ibkr",
    region="us",
    asset_class="stocks",
    plan="fixed",
)

# From layered user config
config = BacktestConfig.from_user_config()

# From a YAML file
config = BacktestConfig.from_yaml("config/my_strategy.yaml")

# Override specific settings
config = BacktestConfig.from_preset("backtrader")
config.commission_rate = 0.002
config.initial_cash = 500_000
```

## Structured Assumptions

Broker and asset-class assumptions are separate from strategy logic.

Use structured inputs when you want broker-specific fees without hiding them in
generic defaults:

```python
config = BacktestConfig.from_assumptions(
    broker="ibkr",
    region="us",
    asset_class="stocks",
    plan="fixed",
    slippage_type=SlippageType.NONE,
)
```

This currently resolves to the `ibkr_us_stocks_fixed` preset.

## User Config Directory

For personal defaults that apply across strategies, use a config directory:

- `~/.config/ml4t/backtest/defaults.yaml`
- `~/.config/ml4t/backtest/assumptions.yaml`

Load it with:

```python
config = BacktestConfig.from_user_config()
```

Recommended layout:

```yaml
# defaults.yaml
cash:
  initial: 250000
calendar:
  timezone: America/New_York
slippage:
  model: none
```

```yaml
# assumptions.yaml
default_assumptions:
  broker: ibkr
  region: us
  asset_class: stocks
  plan: fixed

brokers:
  ibkr:
    us:
      stocks:
        fixed:
          commission:
            minimum: 1.0
```

Layering order:

1. library neutral defaults
2. selected broker/asset-class preset
3. user global defaults from `defaults.yaml`
4. user broker-specific overrides from `assumptions.yaml`
5. explicit per-run keyword overrides

Broker assumptions currently own commission defaults. Slippage remains a
separate assumption layer.

## Parameter Reference

### Account

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_cash` | float | 100,000 | Starting cash balance |
| `allow_short_selling` | bool | False | Enable short positions |
| `allow_leverage` | bool | False | Enable margin borrowing |
| `initial_margin` | float | 0.5 | Reg T initial margin (50%) |
| `long_maintenance_margin` | float | 0.25 | Long position maintenance |
| `short_maintenance_margin` | float | 0.30 | Short position maintenance |
| `fixed_margin_schedule` | dict \| None | None | Per-asset `(initial, maintenance)` fixed dollar margin per contract |
| `margin_pct_schedule` | dict \| None | None | Per-asset `(initial, maintenance)` fractions of notional; cannot overlap the fixed schedule |
| `short_cash_policy` | ShortCashPolicy | CREDIT | How short proceeds affect cash |
| `lock_notional_update_mode` | LockNotionalUpdateMode | POSITION_LEGS | Whether locked short collateral updates by closing/opening legs or by the combined order |

Account type is determined by the flag combination:

| `allow_short` | `allow_leverage` | Account Type |
|:-:|:-:|:--|
| False | False | Cash (long-only) |
| True | False | Crypto-style (short OK, no leverage) |
| True | True | Margin (full Reg T) |

### Execution Timing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `execution_mode` | ExecutionMode | NEXT_BAR | When orders fill (SAME_BAR or NEXT_BAR) |
| `execution_price` | ExecutionPrice | OPEN | Price used for market fills |
| `mark_price` | ExecutionPrice | PRICE | Price used for open-position marking |
| `next_bar_execution_price` | ExecutionPrice or None | None | Explicit deferred execution source; None preserves the native next-open rule |
| `conditional_order_mode` | ConditionalOrderMode | BAR_OHLC | Extrema/level-based conditional orders, or EXECUTION_PRICE for triggers and fills on the selected observation |

To execute an ordinary market order at the next observed bar's close, select
`execution_mode=ExecutionMode.NEXT_BAR` and
`next_bar_execution_price=ExecutionPrice.CLOSE`. The order remains deferred;
changing this source does not move it to its decision bar. The default `None`
retains existing next-open behavior, including existing quote/VWAP handling.
Market-on-close orders retain their own close rule, and canonical opening-auction
child intents continue to use the opening price. The option is persisted under
`execution.next_bar_execution_price` in the resolved JSON/YAML configuration.

Available `ExecutionPrice` values:

| Value | Meaning |
|-------|---------|
| `OPEN` | Use the bar open |
| `CLOSE` | Use the bar close |
| `VWAP` | Use the feed reference price as a VWAP proxy |
| `MID` | Use `(high + low) / 2` |
| `PRICE` | Use `FeedSpec.price_col` / `bar["price"]` |
| `BID` | Use best bid |
| `ASK` | Use best ask |
| `QUOTE_MID` | Use explicit or derived midpoint |
| `QUOTE_SIDE` | Buy at ask, sell at bid; for marking, longs use bid and shorts use ask |

Quote-aware settings change both execution semantics and reporting. When you use
`BID`, `ASK`, `QUOTE_MID`, or `QUOTE_SIDE`, fills and trades preserve the
underlying quote context and portfolio-state snapshots reflect the configured
mark source.

### Stop Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stop_fill_mode` | StopFillMode | STOP_PRICE | Stop order fill price (STOP_PRICE, CLOSE_PRICE, BAR_EXTREME, NEXT_BAR_OPEN) |
| `stop_level_basis` | StopLevelBasis | FILL_PRICE | Base price for stop levels (FILL_PRICE, SIGNAL_PRICE) |
| `trail_hwm_source` | WaterMarkSource | CLOSE | Water mark update price (CLOSE, BAR_EXTREME) |
| `trail_include_entry_bar_extremes` | bool | False | Include the completed entry bar's high or low in the next bar's trailing water mark |
| `initial_hwm_source` | InitialHwmSource | FILL_PRICE | Initial water mark on entry (FILL_PRICE, BAR_CLOSE, BAR_HIGH) |
| `trail_stop_timing` | TrailStopTiming | LAGGED | Timing of water mark vs stop check (LAGGED, INTRABAR, VBT_PRO) |

### Commission

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `commission_type` | CommissionType | NONE | Model: NONE, PERCENTAGE, PER_SHARE, PER_TRADE, TIERED, PER_ORDER_PER_SHARE, PER_ORDER |
| `commission_rate` | float | 0.0 | Rate for percentage model |
| `commission_per_share` | float | 0.0 | Dollar amount per share |
| `commission_per_trade` | float | 0.0 | Flat fee per fill for PER_TRADE; once per order for PER_ORDER |
| `commission_minimum` | float | 0.0 | Per-fill minimum for PER_SHARE; cumulative order minimum for PER_ORDER_PER_SHARE/PER_ORDER |

### Slippage

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `slippage_type` | SlippageType | NONE | Model: NONE, PERCENTAGE, FIXED, SPREAD, VOLUME_BASED, CUMULATIVE_VOLUME_SHARE |
| `slippage_rate` | float | 0.0 | Percentage rate, or coefficient for the selected volume-impact model |
| `slippage_fixed` | float | 0.0 | Fixed dollar amount per share |
| `slippage_spread` | float | 0.0 | Spread in currency units for bar-only spread approximation |
| `slippage_spread_by_asset` | dict[str, float] | `{}` | Optional per-asset spread overrides |
| `slippage_spread_convention` | SpreadConvention | FULL_SPREAD | Interpret spread input as full spread or per-side cost |
| `stop_slippage_rate` | float | 0.0 | Additional slippage for stop exits |

The plain `BacktestConfig()` defaults are intentionally neutral about hidden
friction:

- commission defaults to `NONE`
- slippage defaults to `NONE`
- if you want broker-specific fees or synthetic execution costs, opt in explicitly

### Position Sizing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `share_type` | ShareType | INTEGER | FRACTIONAL or INTEGER |
| `share_rounding` | ShareRounding | NEAREST | NEAREST or TRUNCATE for integer quantities |

### Result evidence

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `retain_intent_history` | bool | False | Include full target, child-order, and reconciliation histories in result artifacts; counts are always retained |
| `retain_lifecycle_history` | bool | False | Include the per-callback lifecycle trace in result artifacts; callback counts are always retained |

### Cash Management

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cash_buffer_pct` | float | 0.0 | Reserve this % of cash (never invest) |
| `reject_on_insufficient_cash` | bool | True | Reject orders exceeding buying power |
| `skip_cash_validation` | bool | False | Bypass gatekeeper (Zipline-style) |

### Settlement

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `settlement_delay` | int | 0 | Bars until sale proceeds are spendable (T+N) |
| `settlement_reduces_buying_power` | bool | True | Unsettled cash reduces buying power |

### Order Handling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fill_ordering` | FillOrdering | EXIT_FIRST | Processing sequence: EXIT_FIRST, FIFO, SEQUENTIAL |
| `entry_order_priority` | EntryOrderPriority | SUBMISSION | Entry sequencing: SUBMISSION, NOTIONAL_DESC, NOTIONAL_ASC |
| `partial_fills_allowed` | bool | False | Allow partial order fills |
| `next_bar_submission_precheck` | bool | False | Pre-check cash at submission time |
| `next_bar_simple_cash_check` | bool | False | Simple cash check for next-bar orders |
| `buying_power_reservation` | bool | False | Reserve cash at submission (LEAN-style) |
| `next_bar_queue_shadow_validation` | bool | False | Revalidate aged next-bar entries against the eligible queue before fills |
| `immediate_fill` | bool | False | Fill same-bar market orders at submit time |

### Rebalancing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rebalance_mode` | RebalanceMode | INCREMENTAL | Serialized profile setting; does not select `TargetWeightExecutor` mode |
| `rebalance_headroom_pct` | float | 1.0 | Scale requested weights in `Broker.rebalance_to_weights()` |
| `missing_price_policy` | MissingPricePolicy | SKIP | Handle missing prices: SKIP or USE_LAST |
| `late_asset_policy` | LateAssetPolicy | ALLOW | Handle late-starting assets: ALLOW or REQUIRE_HISTORY |
| `late_asset_min_bars` | int | 1 | Minimum bars of history before trading |

Set `RebalanceConfig.rebalance_mode` for `TargetWeightExecutor`. The broker's
`rebalance_to_weights()` helper uses account value captured at the start of its call.
Headroom changes requested targets; rounding, costs and price changes still affect
realized exposure. See [Rebalancing](rebalancing.md#rebalance-modes) for the two configuration
objects and executor mode behavior.

### Calendar

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `calendar` | str \| None | omitted | Explicit name or `None`; omission inherits declared feed metadata, then NYSE. |
| `time_contract_version` | int | 2 | Version 1 retains legacy null handling; version 2 distinguishes omission and explicit null. |
| `periods_per_year` | float \| None | None | Optional positive finite return-period scale; observation mode defaults to 252. |
| `timezone` | str | "UTC" | Timezone for naive datetimes |
| `data_frequency` | DataFrequency | DAILY | Data frequency (DAILY, 1m, 5m, 15m, 30m, 1h) |
| `enforce_sessions` | bool | False | Skip bars outside trading sessions |

New Python configs use `time_contract_version=2`. Explicit `calendar=None` disables
calendar handling, even when the feed declares an exchange. Omission inherits a declared
feed calendar, including explicit null in a v2 `FeedSpec`; otherwise it resolves to NYSE.
A named continuous calendar still defines a schedule and is not observation mode.
Inspect `config.resolved_calendar` for the effective selection.
When `config.feed_spec` leaves a calendar or other time field undeclared, the Engine inherits
that field from the `DataFeed` metadata. A declaration in `config.feed_spec` takes precedence.
With `enforce_sessions=True`, `event_time` and `bar_close` timestamps use actual session intervals;
`session_label` timestamps use trading dates. Frequency supplies this distinction only when
timestamp semantics are undeclared.

<!-- ml4t-doc-test: calendar-selection-v2 -->
```python
from ml4t.backtest import BacktestConfig
from ml4t.specs import FeedSpec

metadata = FeedSpec(time_contract_version=2, calendar="CME_Equity")
assert BacktestConfig(feed_spec=metadata).resolved_calendar == "CME_Equity"
assert BacktestConfig(calendar=None, feed_spec=metadata).resolved_calendar is None
assert BacktestConfig().resolved_calendar == "NYSE"
print(BacktestConfig(feed_spec=metadata).resolved_calendar)
print(BacktestConfig(calendar=None, feed_spec=metadata).resolved_calendar)
print(BacktestConfig().resolved_calendar)
```

<!-- ml4t-doc-output: calendar-selection-v2 -->
```text
CME_Equity
None
NYSE
```

In observation mode, annualized metrics use 252 return periods per numerical year by
default, regardless of timestamp spacing or `data_frequency`. This is a numerical
convention, not a claim about exchange sessions. Set a positive finite `periods_per_year`
to choose another scale; the override also survives config serialization. Calendar-free
CAGR uses `(N - 1) / periods_per_year` years for `N` recorded equity values. A legacy payload without
`calendar.time_contract_version` remains version 1, including its old null-as-inheritance
behavior. Use `to_dict()` or `to_yaml()` to retain the v2 distinction between omission
and explicit null; do not relabel an old payload without deciding what its null means.
Direct result calls use the current selector contract: `result.to_daily_returns(calendar=None)`
returns observations even when the result was built from a legacy configuration.

Calendar v2 requires `ml4t-specs>=0.2.0`. The published
[Specs 0.2.0](https://pypi.org/project/ml4t-specs/0.2.0/) package contains the versioned
selector and serialization API. This is next-version Backtest behavior; the pending
0.1.13 release candidate remains unchanged.

### Feed Contract

`BacktestConfig` can also carry a serialized `FeedSpec` in `feed_spec`, represented
under the top-level `feed`
section. This lets you capture how the input data should be interpreted without
introducing a second config object.

Supported keys mirror `FeedSpec`:

- `timestamp_col`
- `entity_col`
- `price_col`
- `open_col`
- `high_col`
- `low_col`
- `close_col`
- `volume_col`
- `bid_col`
- `ask_col`
- `mid_col`
- `bid_size_col`
- `ask_size_col`
- `calendar`
- `timezone`
- `data_frequency`
- `bar_type`
- `timestamp_semantics`
- `session_start_time`
- `time_contract_version`
- `bar_grid_anchor`, `bar_label`, and `partial_bar_policy` (preserved v2 declarations)

`session_start_time` can override an evening session boundary. Morning-start sessions use the
local calendar date, so morning values must match the configured calendar's standard open.

### Metadata

Use the top-level `metadata` section for any user-defined provenance that the
library does not interpret directly, for example:

- strategy id or strategy name
- paths to price or prediction inputs
- experiment ids
- notes

`metadata` round-trips through `to_dict()`, `from_dict()`, `to_yaml()`, and
`from_yaml()` unchanged.

`preset_name` records the preset or config source used to construct the config. It
is provenance metadata and does not change execution by itself.

## YAML Configuration

Save and load configs for reproducibility:

```python
# Save
config = BacktestConfig.from_preset("realistic")
config.to_yaml("config/realistic_v2.yaml")

# Load
config = BacktestConfig.from_yaml("config/realistic_v2.yaml")
```

YAML format uses nested sections:

```yaml
account:
  allow_short_selling: false
  allow_leverage: false
execution:
  execution_price: open
  mark_price: price
  execution_mode: next_bar
stops:
  stop_fill_mode: stop_price
  stop_level_basis: fill_price
commission:
  model: percentage
  rate: 0.001
slippage:
  model: percentage
  rate: 0.001
cash:
  initial: 100000.0
  buffer_pct: 0.0
orders:
  fill_ordering: exit_first
  reject_on_insufficient_cash: true
feed:
  timestamp_col: timestamp
  entity_col: symbol
  price_col: close
metadata:
  strategy_id: topk_monthly_v1
  prices_path: /path/to/prices.parquet
```

You can keep input specs sparse. Any omitted fields fall back to library defaults.
After execution, `result.config.to_dict()` gives you the fully resolved config
snapshot with defaults filled in.

## Resolved Snapshot

`BacktestResult` can export a richer runtime snapshot that includes the resolved
config plus run metadata such as the realized time window:

```python
result = run_backtest(...)

# Replayable config payload
resolved_config = result.config.to_dict()

# Richer runtime spec
runtime_spec = result.to_spec_dict()
```

`runtime_spec["config"]` remains compatible with `BacktestConfig.from_dict()`.

## Validation

Call `validate()` to check for potential issues:

```python
config = BacktestConfig(execution_mode=ExecutionMode.SAME_BAR)
warnings = config.validate()
# ["SAME_BAR execution has look-ahead bias risk..."]
```

## Describe

Get a human-readable summary:

```python
config = BacktestConfig.from_preset("backtrader")
print(config.describe())
```

## Common Recipes

### Realistic US Equities

```python
config = BacktestConfig(
    initial_cash=100_000,
    execution_mode=ExecutionMode.NEXT_BAR,
    execution_price=ExecutionPrice.OPEN,
    mark_price=ExecutionPrice.PRICE,
    commission_type=CommissionType.PERCENTAGE,
    commission_rate=0.002,
    slippage_type=SlippageType.PERCENTAGE,
    slippage_rate=0.002,
    stop_slippage_rate=0.001,
    share_type=ShareType.INTEGER,
    cash_buffer_pct=0.02,
    stop_fill_mode=StopFillMode.NEXT_BAR_OPEN,
)
```

### Crypto (24/7, Fractional)

```python
config = BacktestConfig(
    initial_cash=10_000,
    allow_short_selling=True,
    execution_mode=ExecutionMode.SAME_BAR,
    execution_price=ExecutionPrice.CLOSE,
    mark_price=ExecutionPrice.PRICE,
    share_type=ShareType.FRACTIONAL,
    commission_type=CommissionType.PERCENTAGE,
    commission_rate=0.001,
    slippage_type=SlippageType.PERCENTAGE,
    slippage_rate=0.0005,
    calendar="crypto",
)
```

### Zero-Cost Comparison

```python
config = BacktestConfig(
    commission_type=CommissionType.NONE,
    slippage_type=SlippageType.NONE,
    execution_mode=ExecutionMode.SAME_BAR,
    mark_price=ExecutionPrice.PRICE,
    share_type=ShareType.FRACTIONAL,
    skip_cash_validation=True,
)
```

### Quote-Aware Microstructure

```python
config = BacktestConfig(
    execution_mode=ExecutionMode.NEXT_BAR,
    execution_price=ExecutionPrice.QUOTE_SIDE,
    mark_price=ExecutionPrice.QUOTE_MID,
    commission_type=CommissionType.PERCENTAGE,
    commission_rate=0.0005,
    slippage_type=SlippageType.NONE,
)
```

This configuration:

- crosses the spread at execution via `QUOTE_SIDE`
- marks inventory at midpoint
- keeps commission separate
- avoids layering synthetic slippage on top unless you explicitly want extra impact

### Bar-Only Spread Approximation

```python
from ml4t.backtest.config import SlippageType, SpreadConvention

config = BacktestConfig(
    execution_price=ExecutionPrice.CLOSE,
    slippage_type=SlippageType.SPREAD,
    slippage_spread=0.02,  # $0.02 quoted spread
    slippage_spread_convention=SpreadConvention.FULL_SPREAD,
)
```

This configuration applies half the configured spread per side, so a buy at
`100.00` fills at `100.01` and a sell at `100.00` fills at `99.99`.

If your input value is already the per-side crossing cost, use:

```python
config = BacktestConfig(
    slippage_type=SlippageType.SPREAD,
    slippage_spread=0.01,
    slippage_spread_convention=SpreadConvention.HALF_SPREAD,
    slippage_spread_by_asset={"AAPL": 0.01, "MSFT": 0.015},
)
```

### Quote-Aware Equities

```python
config = BacktestConfig(
    execution_mode=ExecutionMode.SAME_BAR,
    execution_price=ExecutionPrice.QUOTE_SIDE,
    mark_price=ExecutionPrice.QUOTE_SIDE,
    commission_type=CommissionType.NONE,
    slippage_type=SlippageType.NONE,
)
```

Use this with a `DataFeed` whose `FeedSpec` maps `price_col`, `bid_col`, `ask_col`, and optionally quote sizes.

## In the book

Chapter 16, Section 16.3, [Engine divergence anatomy](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/16_strategy_simulation/07_engine_divergence_anatomy.ipynb) changes one execution assumption at a time. Use this reference to identify and record the corresponding `BacktestConfig` fields.

## Next Steps

- [Book Guide](../book-guide/index.md) -- find the matching notebook or case-study configuration pattern
- [Profiles](profiles.md) -- pre-built configs for each framework
- [Execution Semantics](execution-semantics.md) -- deep dive into each parameter's behavior
