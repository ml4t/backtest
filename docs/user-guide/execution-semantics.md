# Execution Semantics

This is the reference for how orders execute in ml4t-backtest. Every behavioral detail described here corresponds to a named config parameter, so you can tune or override it.

## Execution Timing

### NEXT_BAR (Default)

Orders submitted during `on_data()` are queued and filled at the **next bar's open** price.

```
Bar N:  Strategy sees close=$100, submits buy order
Bar N+1: Order fills at open=$101
```

This is the realistic model -- your strategy decides based on today's information and the trade executes at tomorrow's opening auction.

```python
from ml4t.backtest.config import BacktestConfig
from ml4t.backtest.types import ExecutionMode

config = BacktestConfig(execution_mode=ExecutionMode.NEXT_BAR)  # default
```

### SAME_BAR

Orders fill at the **current bar's close** price, in the same bar they are submitted.

```
Bar N:  Strategy sees close=$100, submits buy order, fills at close=$100
```

This mode is useful for comparing against vectorized frameworks (VectorBT) where signals and fills happen simultaneously. It carries look-ahead risk for production strategies because the strategy can "see" the close before deciding to trade at the close.

```python
config = BacktestConfig(execution_mode=ExecutionMode.SAME_BAR)
```

### Execution Price

The `execution_price` parameter controls which bar price is used for market order fills:

| Value | Fill Price | Typical Use |
|-------|-----------|-------------|
| `OPEN` | Next bar's open | Default, realistic |
| `CLOSE` | Current bar's close | VectorBT comparison |
| `VWAP` | Volume-weighted average | Requires volume data |
| `MID` | (high + low) / 2 | Simple approximation |

## Fill Ordering

When multiple orders are pending on the same bar, the engine must decide the processing sequence. This matters because fills affect cash, which affects whether subsequent orders can be accepted.

### EXIT_FIRST (Default)

All exits process first (freeing cash), then all entries:

```
1. Process all exit orders → cash freed
2. Mark-to-market remaining positions
3. Process all entry orders → gatekeeper checks cash
```

This is the most capital-efficient ordering. Matches VectorBT's `call_seq='auto'`.

### FIFO

Orders process in submission order. Each order's gatekeeper check sees cash from all prior fills:

```
1. First submitted order fills → cash updated
2. Second submitted order fills → cash updated
3. ...
```

Matches Backtrader's submission-order processing.

### SEQUENTIAL

Orders process in submission order (typically alphabetical by asset) without exit/entry separation. Unlike EXIT_FIRST, exits do not pre-free cash for later entries.

Matches LEAN's per-order sequential buying-power model.

```python
from ml4t.backtest.config import FillOrdering

config = BacktestConfig(fill_ordering=FillOrdering.EXIT_FIRST)  # default
config = BacktestConfig(fill_ordering=FillOrdering.FIFO)
config = BacktestConfig(fill_ordering=FillOrdering.SEQUENTIAL)
```

### Entry Order Priority

When using EXIT_FIRST, entries are processed after exits. The `entry_order_priority` controls the sequence of entry orders:

| Value | Behavior |
|-------|----------|
| `SUBMISSION` | Keep strategy submission order (default) |
| `NOTIONAL_DESC` | Larger dollar entries first |
| `NOTIONAL_ASC` | Smaller dollar entries first |

## Stop and Take-Profit Execution

Position rules (StopLoss, TakeProfit, TrailingStop) are evaluated on every bar using OHLC data. The key question is: **at what price does a triggered stop fill?**

### Stop Fill Modes

The `stop_fill_mode` parameter controls stop/take-profit fill prices:

| Mode | Fill Price | Use Case |
|------|-----------|----------|
| `STOP_PRICE` | Exact stop/target level | Default, standard model |
| `CLOSE_PRICE` | Bar's close price | VectorBT with close-only data |
| `BAR_EXTREME` | Bar's low (stop) or high (TP) | Conservative/optimistic model |
| `NEXT_BAR_OPEN` | Next bar's open price | Zipline-style deferred exits |

**Gap handling.** If the bar opens beyond the stop level (a gap through the stop), the fill price is the bar's open, not the stop price. This accurately models gap risk:

```
Stop set at $95.00
Bar opens at $93.00 (gap down)
Fill price = $93.00 (open), not $95.00 (stop)
```

### Stop Level Basis

The `stop_level_basis` controls what price the stop percentage is calculated from:

| Value | Base Price | Use Case |
|-------|-----------|----------|
| `FILL_PRICE` | Actual fill price including slippage | Default, most frameworks |
| `SIGNAL_PRICE` | Close price when signal was generated | Backtrader behavior |

Example: You set `StopLoss(pct=0.05)`. With `FILL_PRICE`, the stop is 5% below where you actually got filled. With `SIGNAL_PRICE`, it's 5% below the close of the bar where you submitted the order.

## Trailing Stop Mechanics

Trailing stops track a "water mark" -- the highest price since entry (for longs) or lowest price since entry (for shorts) -- and exit when price retraces by a percentage from that mark.

### Water Mark Source

`trail_hwm_source` controls which price updates the water mark:

| Value | Update Price | Framework |
|-------|-------------|-----------|
| `CLOSE` | Bar's close | Default, most frameworks |
| `BAR_EXTREME` | Bar's high (longs) / low (shorts) | VectorBT Pro with OHLC |

### Initial Water Mark

`initial_hwm_source` controls the water mark on the entry bar:

| Value | Initial HWM | Framework |
|-------|------------|-----------|
| `FILL_PRICE` | Actual fill price | Default, event-driven frameworks |
| `BAR_CLOSE` | Entry bar's close | |
| `BAR_HIGH` | Entry bar's high | VectorBT Pro with OHLC |

### Trailing Stop Timing

`trail_stop_timing` controls when water marks update relative to the stop check. This is the subtlest parameter and the one that causes the most divergence between frameworks.

**LAGGED** (default): Check stop using the *previous* bar's water mark, then update water mark at end of current bar. This creates a 1-bar lag -- the stop can't trigger based on a new high set in the current bar.

**INTRABAR**: Update water mark before checking. The stop can trigger based on the current bar's extreme. More aggressive than LAGGED.

**VBT_PRO**: Two-pass algorithm matching VectorBT Pro exactly:

1. **Pass 1**: Check stop using the *previous* bar's water mark against the current bar's HIGH (long) or LOW (short)
2. If pass 1 doesn't trigger, update the water mark from the current bar's extreme
3. **Pass 2**: Check stop using the *updated* water mark against **CLOSE only** (not HIGH/LOW)

This precisely reproduces VBT Pro's `can_use_ohlc=False` behavior in the second pass.

```python
from ml4t.backtest.config import TrailStopTiming, WaterMarkSource, InitialHwmSource

# VectorBT Pro compatible trailing stops
config = BacktestConfig(
    trail_stop_timing=TrailStopTiming.VBT_PRO,
    trail_hwm_source=WaterMarkSource.BAR_EXTREME,
    initial_hwm_source=InitialHwmSource.BAR_HIGH,
)
```

## Commission and Slippage

### Commission Models

| Type | Calculation | Config |
|------|------------|--------|
| `NONE` | No commission | `commission_rate=0` |
| `PERCENTAGE` | % of trade value | `commission_rate=0.001` (0.1%) |
| `PER_SHARE` | Fixed $ per share | `commission_per_share=0.005` |
| `PER_TRADE` | Fixed $ per trade | `commission_per_trade=5.0` |
| `TIERED` | Volume-based tiers | Custom model |

```python
from ml4t.backtest.config import CommissionType

# Interactive Brokers style
config = BacktestConfig(
    commission_type=CommissionType.PER_SHARE,
    commission_per_share=0.005,
    commission_minimum=1.0,
)
```

### Slippage Models

| Type | Calculation | Config |
|------|------------|--------|
| `NONE` | No slippage | |
| `PERCENTAGE` | % of price | `slippage_rate=0.001` (0.1%) |
| `FIXED` | Fixed $ per share | `slippage_fixed=0.01` |
| `VOLUME_BASED` | Size vs volume | `slippage_rate=0.1` (10% volume limit) |

Stop orders can have additional slippage via `stop_slippage_rate`:

```python
config = BacktestConfig(
    slippage_rate=0.001,        # 0.1% for market orders
    stop_slippage_rate=0.001,   # Additional 0.1% for stop exits
)
```

## Settlement

The `settlement_delay` parameter simulates T+N settlement:

```python
config = BacktestConfig(
    settlement_delay=2,                    # T+2 (US equities standard)
    settlement_reduces_buying_power=True,  # Unsettled cash not spendable
)
```

With T+2 settlement, cash from selling shares on Monday isn't available for buying until Wednesday.

## Cash Management

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `initial_cash` | 100,000 | Starting cash |
| `cash_buffer_pct` | 0.0 | Reserve this % of cash (never invest it) |
| `reject_on_insufficient_cash` | True | Reject orders that exceed buying power |
| `skip_cash_validation` | False | Bypass gatekeeper entirely (Zipline-style) |
| `buying_power_reservation` | False | Reserve cash at submission time (LEAN-style) |

## Position Sizing

```python
from ml4t.backtest.config import ShareType

# Allow fractional shares (crypto, some brokers)
config = BacktestConfig(share_type=ShareType.FRACTIONAL)

# Round down to whole shares (most equity brokers)
config = BacktestConfig(share_type=ShareType.INTEGER)
```

## See It in Action

The [Machine Learning for Trading](https://github.com/stefan-jansen/machine-learning-for-trading) book demonstrates execution semantics across chapters:

- **Ch16 / NB11** (`engine_divergence_anatomy`) — detailed analysis of how SAME_BAR vs NEXT_BAR and fill ordering affect backtest results
- **Ch18** (`portfolio_construction`) — LinearImpact and SquareRootImpact market impact models with VolumeParticipationLimit
- **Ch16 case studies** — each case study uses setup.yaml to configure commission_rate, slippage_rate, and execution_mode

## Next Steps

- [Configuration](configuration.md) -- complete reference for all 40+ parameters
- [Profiles](profiles.md) -- pre-configured settings for each framework
- [Risk Management](risk-management.md) -- position rules and portfolio limits
