# Framework Behavior Catalog

**Purpose**: Authoritative reference for how ml4t-backtest matches or differs from VectorBT Pro, VectorBT OSS, Backtrader, and Zipline.

**Last Updated**: 2026-01-02

---

## VectorBT Pro

### Order Execution

| Behavior | VectorBT Pro | ml4t Config |
|----------|--------------|-------------|
| **Execution timing** | Same bar (signal bar close) | `ExecutionMode.SAME_BAR` |
| **Fill price** | Bar close by default | Default close price |
| **With full OHLC** | Uses open/high/low for stops | Requires OHLC in DataFeed |

### Trailing Stops

| Behavior | VectorBT Pro | ml4t Config |
|----------|--------------|-------------|
| **HWM source** | Bar HIGH (with OHLC) | `TrailHwmSource.HIGH` |
| **Initial HWM** | Bar CLOSE (not fill price) | `InitialHwmSource.BAR_CLOSE` |
| **Trigger condition** | `bar_low < trail_level` | Same |
| **Fill price** | Trail level exactly | `StopFillMode.STOP_PRICE` |
| **API parameter** | `tsl_stop=0.05` | `TrailingStop(pct=0.05)` |

**CRITICAL**: VectorBT Pro requires full OHLC data for proper trailing stop behavior.

### Stop-Loss / Take-Profit

| Behavior | VectorBT Pro | ml4t Config |
|----------|--------------|-------------|
| **SL trigger** | `bar_low <= stop_price` | Same |
| **TP trigger** | `bar_high >= target_price` | Same |
| **Fill price** | Stop/target level exactly | `StopFillMode.STOP_PRICE` |
| **API parameters** | `sl_stop=0.05, tp_stop=0.10` | `StopLoss(pct=0.05), TakeProfit(pct=0.10)` |

### Cost Models

| Behavior | VectorBT Pro | ml4t Config |
|----------|--------------|-------------|
| **Commission** | `fees=0.001` (0.1% of trade value) | `PercentageCommission(0.001)` |
| **Slippage** | `slippage=0.0005` (0.05% directional) | `PercentageSlippage(0.0005)` |
| **Slippage direction** | Worse for trader (buy higher, sell lower) | Same |

### Position Management

| Behavior | VectorBT Pro | ml4t Config |
|----------|--------------|-------------|
| **Re-entry blocking** | `accumulate=False` | Strategy checks `current_qty == 0` |
| **Position flipping** | Auto-close then reverse | Manual or via strategy |
| **Short positions** | Negative size | Negative quantity |

### Edge Cases

| Behavior | VectorBT Pro | ml4t |
|----------|--------------|------|
| **Gap through stop** | Fills at close | Fills at stop price or open |
| **Same-bar re-entry** | Blocked with `accumulate=False` | Strategy-controlled |
| **Incomplete positions** | Counted as trades | Same |

---

## VectorBT OSS (Open Source)

### Differences from VectorBT Pro

| Aspect | VBT Pro | VBT OSS |
|--------|---------|---------|
| **OHLC support** | Full OHLC for stops | Close-only for stops |
| **Trailing stops** | `tsl_stop` parameter | `tsl_stop` (close-based only) |
| **Commercial license** | Required | MIT license |

### Order Execution

Same as VectorBT Pro for basic long/short execution.

### Cost Models

Identical API to VectorBT Pro:
- `fees=0.001` for commission
- `slippage=0.0005` for slippage

---

## Backtrader

### Order Execution

| Behavior | Backtrader | ml4t Config |
|----------|------------|-------------|
| **Execution timing** | Next bar (COO/COC) | `ExecutionMode.NEXT_BAR` |
| **Fill price** | Open of next bar | With `use_open=True` |
| **Cheat-on-open** | `cerebro.broker.set_coo(True)` | `ExecutionMode.SAME_BAR` |
| **Cheat-on-close** | `cerebro.broker.set_coc(True)` | `ExecutionMode.SAME_BAR` |

### Stop-Loss / Take-Profit

| Behavior | Backtrader | ml4t Config |
|----------|------------|-------------|
| **Stop order type** | `bt.Order.Stop` | Via `StopLoss` rule |
| **Stop trigger** | Next bar check | Same |
| **Gap handling** | Fills at open if gaps through | `StopFillMode.STOP_PRICE` with gap logic |

### Commission Models

| Behavior | Backtrader | ml4t Config |
|----------|------------|-------------|
| **Percentage** | `CommInfoBase(commission=0.001)` | `PercentageCommission(0.001)` |
| **Per-share** | `commission=0.005, commtype=CommInfoBase.COMM_FIXED` | `PerShareCommission(0.005)` |
| **Minimum fee** | Supported | Not yet implemented |

### Position Management

| Behavior | Backtrader | ml4t |
|----------|------------|------|
| **Integer shares** | Enforced by default | Strategy responsibility |
| **Position reversal** | Auto-close then reverse | Manual |
| **Margin** | Configurable | `account_type="margin"` |

---

## Zipline

### Order Execution

| Behavior | Zipline | ml4t Config |
|----------|---------|-------------|
| **Execution timing** | Next bar open | `ExecutionMode.NEXT_BAR` |
| **Fill price** | Open of next bar | With `use_open=True` |
| **Order types** | Market, limit, stop | Market + rules |

### Slippage Models

| Behavior | Zipline | ml4t Config |
|----------|---------|-------------|
| **Volume slippage** | `VolumeShareSlippage` | `PercentageSlippage` (approximation) |
| **Fixed slippage** | `FixedSlippage(spread=0.05)` | `FixedSlippage(0.05)` |

### Commission Models

| Behavior | Zipline | ml4t Config |
|----------|---------|-------------|
| **Per-share** | `PerShare(cost=0.005)` | `PerShareCommission(0.005)` |
| **Per-trade** | `PerTrade(cost=10.0)` | `FixedCommission(10.0)` |

### Data Requirements

**NOTE**: Zipline uses bundle data, which requires a custom bundle registration pattern for validation.

| Aspect | Zipline | ml4t |
|--------|---------|------|
| **Data source** | Custom bundle (synthetic data) | DataFrame input |
| **Symbols** | Bundle-registered | Any string |
| **Calendar** | NYSE calendar required | Optional (pandas_market_calendars) |

**Custom Bundle Pattern**: Zipline validation uses a custom bundle that registers synthetic test data:
```python
from zipline.data.bundles import register
register('test_bundle', lambda environ, asset_db_writer, minute_bar_writer,
         daily_bar_writer, adjustment_writer, calendar, start_session,
         end_session, cache, show_progress, output_dir: None)
```

**Result**: 10/10 scenarios pass with 100% exact match (119,577 trades at 500 assets × 10 years).

---

## Configuration Quick Reference

### VectorBT Pro Match

```python
from ml4t.backtest import (
    Engine, TrailHwmSource, InitialHwmSource, StopFillMode,
    ExecutionMode, PercentageCommission, PercentageSlippage,
)

engine = Engine(
    feed,
    strategy,
    initial_cash=100_000.0,
    execution_mode=ExecutionMode.SAME_BAR,
    trail_hwm_source=TrailHwmSource.HIGH,
    initial_hwm_source=InitialHwmSource.BAR_CLOSE,
    stop_fill_mode=StopFillMode.STOP_PRICE,
    commission_model=PercentageCommission(0.001),
    slippage_model=PercentageSlippage(0.0005),
)
```

### Backtrader Match (Next-Bar)

```python
engine = Engine(
    feed,
    strategy,
    execution_mode=ExecutionMode.NEXT_BAR,
    stop_fill_mode=StopFillMode.STOP_PRICE,
    commission_model=PercentageCommission(0.001),
)
```

### Zipline Match (Best Effort)

```python
engine = Engine(
    feed,
    strategy,
    execution_mode=ExecutionMode.NEXT_BAR,
    stop_fill_mode=StopFillMode.NEXT_BAR_OPEN,
    commission_model=PerShareCommission(0.005),
)
```

---

## Validation Status by Feature

| Feature | VBT Pro | VBT OSS | Backtrader | Zipline |
|---------|---------|---------|------------|---------|
| Long-only | 100% | 100% | 100% | 100% |
| Long/short | 100% | 100% | 100% | 100% |
| Stop-loss | 100% | 100% | 100% | 100% |
| Take-profit | 100% | 100% | 100% | 100% |
| Trailing stop | 100% | Partial | Partial | 100% |
| Bracket order | 100% | Partial | Partial | N/A |
| % commission | 100% | 100% | 100% | 100% |
| Per-share commission | 100% | 100% | 100% | 100% |
| % slippage | 100% | 100% | 100% | 100% |
| Fixed slippage | 100% | 100% | 100% | 100% |

**Note**: Zipline uses custom bundle pattern with NYSE calendar from `exchange_calendars`. See `validation/zipline/` for implementation.

---

## Calendar Handling

| Framework | Calendar | Trading Days | Session Boundaries |
|-----------|----------|--------------|-------------------|
| VectorBT Pro | Any (DataFrame index) | User-provided | N/A (vectorized) |
| VectorBT OSS | Any (DataFrame index) | User-provided | N/A (vectorized) |
| Backtrader | Optional | `freq="B"` or `exchange_calendars` | Configurable |
| Zipline | Required (NYSE default) | NYSE via `exchange_calendars` | Market open/close |
| ml4t.backtest | Optional | `pandas_market_calendars` | `enforce_sessions` config |

### Calendar Alignment Critical for Validation

**Problem**: Using Python `freq="B"` (business days) vs NYSE calendar produces different date sequences.
- NYSE excludes ~9 holidays per year (MLK Day, Presidents Day, Good Friday, etc.)
- Over 10 years, this creates 87+ day drift between `freq="B"` and NYSE

**Solution**: Always use `exchange_calendars` for cross-framework validation:
```python
from exchange_calendars import get_calendar
nyse = get_calendar("XNYS")
trading_days = nyse.sessions_in_range("2013-01-01", "2023-01-01")
```

### Session Boundaries (CME Futures)

For futures trading with non-standard sessions (e.g., CME 5pm CT Sunday - 4pm CT Friday):

```python
from ml4t.backtest.sessions import SessionConfig, compute_session_pnl

session_config = SessionConfig(
    calendar="CME_Equity",
    timezone="America/Chicago",
    session_start_time="17:00",  # 5pm CT
)

# Compute session-aligned daily P&L
daily_pnl = compute_session_pnl(equity_curve, session_config)
```

---

## Edge Cases Cross-Framework Comparison

| Edge Case | VBT Pro | VBT OSS | Backtrader | Zipline | ml4t.backtest |
|-----------|---------|---------|------------|---------|---------------|
| **End-of-sim positions** | Auto-close | Auto-close | Leave open | Leave open | Leave open |
| **Same-bar re-entry** | Blocked (`accumulate=False`) | Blocked | Allowed | Allowed | Strategy-controlled |
| **Gap-through stop** | Fill at close | Fill at close | Fill at open | Fill at open | Configurable |
| **Incomplete trades** | Counted | Counted | Counted | Counted | Counted |
| **Cash constraints** | Skip order | Skip (`lock_cash=True`) | Reject order | Reject order | Gatekeeper rejects |
| **Position flipping** | Auto | Auto | Manual close+reverse | Manual | Strategy-controlled |

### End-of-Simulation Position Handling

- **VBT Pro/OSS**: Automatically closes all open positions on the last bar
- **Backtrader/Zipline/ml4t**: Leaves positions open at end of simulation
- **Impact**: ~50 trade difference in large-scale validation (not a bug, design choice)

### Gap-Through Stop Fill Behavior

When price gaps through a stop level:
- **VBT Pro/OSS**: Fill at bar close (vectorized, no intrabar resolution)
- **Backtrader/Zipline**: Fill at bar open (event-driven, checks at bar start)
- **ml4t.backtest**: Configurable via `StopFillMode`:
  - `STOP_PRICE`: Fill at exact stop level
  - `NEXT_BAR_OPEN`: Fill at next bar's open (matches Zipline behavior)

---

## Bug Fixes Applied (2026-01-02)

1. **Missing HWM update**: `Engine` was not calling `broker._update_water_marks()` in the event loop. Fixed by adding the call after `evaluate_position_rules()`.

2. **VBT Pro API changes**: Parameters changed from `tsl_th`/`sl_th`/`tp_th` to `tsl_stop`/`sl_stop`/`tp_stop`. Also requires full OHLC data for proper stop behavior.

3. **Column name change**: VBT Pro trades DataFrame uses `Status` column instead of `Exit Type`.

4. **Engine missing parameters**: `trail_hwm_source` and `initial_hwm_source` were on Broker but not exposed through Engine. Added to Engine.__init__.
