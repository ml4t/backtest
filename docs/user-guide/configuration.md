# Configuration

`BacktestConfig` is the single source of truth for all backtest behavior. Every behavioral difference between frameworks is a named parameter -- no subclassing or monkey-patching required.

## Creating a Config

```python
from ml4t.backtest import BacktestConfig

# Sensible defaults
config = BacktestConfig()

# From a framework preset
config = BacktestConfig.from_preset("backtrader")

# From a YAML file
config = BacktestConfig.from_yaml("config/my_strategy.yaml")

# Override specific settings
config = BacktestConfig.from_preset("backtrader")
config.commission_rate = 0.002
config.initial_cash = 500_000
```

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
| `short_cash_policy` | ShortCashPolicy | CREDIT | How short proceeds affect cash |

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
| `execution_price` | ExecutionPrice | OPEN | Price used for market fills (OPEN, CLOSE, VWAP, MID) |

### Stop Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stop_fill_mode` | StopFillMode | STOP_PRICE | Stop order fill price (STOP_PRICE, CLOSE_PRICE, BAR_EXTREME, NEXT_BAR_OPEN) |
| `stop_level_basis` | StopLevelBasis | FILL_PRICE | Base price for stop levels (FILL_PRICE, SIGNAL_PRICE) |
| `trail_hwm_source` | WaterMarkSource | CLOSE | Water mark update price (CLOSE, BAR_EXTREME) |
| `initial_hwm_source` | InitialHwmSource | FILL_PRICE | Initial water mark on entry (FILL_PRICE, BAR_CLOSE, BAR_HIGH) |
| `trail_stop_timing` | TrailStopTiming | LAGGED | Timing of water mark vs stop check (LAGGED, INTRABAR, VBT_PRO) |

### Commission

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `commission_type` | CommissionType | PERCENTAGE | Model: NONE, PERCENTAGE, PER_SHARE, PER_TRADE, TIERED |
| `commission_rate` | float | 0.001 | Rate for percentage model (0.1%) |
| `commission_per_share` | float | 0.0 | Dollar amount per share |
| `commission_per_trade` | float | 0.0 | Dollar amount per trade |
| `commission_minimum` | float | 0.0 | Minimum commission per trade |

### Slippage

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `slippage_type` | SlippageType | PERCENTAGE | Model: NONE, PERCENTAGE, FIXED, VOLUME_BASED |
| `slippage_rate` | float | 0.001 | Rate for percentage model (0.1%) |
| `slippage_fixed` | float | 0.0 | Fixed dollar amount per share |
| `stop_slippage_rate` | float | 0.0 | Additional slippage for stop exits |

### Position Sizing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `share_type` | ShareType | FRACTIONAL | FRACTIONAL or INTEGER |

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
| `immediate_fill` | bool | False | Fill same-bar market orders at submit time |

### Rebalancing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rebalance_mode` | RebalanceMode | INCREMENTAL | SNAPSHOT, INCREMENTAL, or HYBRID |
| `rebalance_headroom_pct` | float | 1.0 | Scale target weights (< 1.0 leaves cash buffer) |
| `missing_price_policy` | MissingPricePolicy | SKIP | Handle missing prices: SKIP or USE_LAST |
| `late_asset_policy` | LateAssetPolicy | ALLOW | Handle late-starting assets: ALLOW or REQUIRE_HISTORY |
| `late_asset_min_bars` | int | 1 | Minimum bars of history before trading |

### Calendar

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `calendar` | str \| None | None | Exchange calendar ("NYSE", "CME_Equity", "LSE", etc.) |
| `timezone` | str | "UTC" | Timezone for naive datetimes |
| `data_frequency` | DataFrequency | DAILY | Data frequency (DAILY, 1m, 5m, 15m, 30m, 1h) |
| `enforce_sessions` | bool | False | Skip bars outside trading sessions |

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
```

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
    share_type=ShareType.FRACTIONAL,
    skip_cash_validation=True,
)
```

## See It in Action

The [Machine Learning for Trading](https://github.com/stefan-jansen/machine-learning-for-trading) book uses BacktestConfig across all case studies:

- **Ch16 case studies** — each case study loads config from `setup.yaml` via `get_backtest_config()`, setting initial_cash, commission_rate, slippage_rate, and execution_mode
- **Ch16 / NB13** (`futures_backtesting`) — ContractSpec with CommissionType.PER_CONTRACT for CME futures
- **Ch19 case studies** — risk management config (stop fill modes, trailing stop timing)

The book pattern: `BacktestConfig()` with 4 overrides (initial_cash, commission_rate, slippage_rate, execution_mode), loaded from YAML. Costs come from `setup.yaml` via a utility function. This covers the vast majority of use cases.

## Next Steps

- [Profiles](profiles.md) -- pre-built configs for each framework
- [Execution Semantics](execution-semantics.md) -- deep dive into each parameter's behavior
