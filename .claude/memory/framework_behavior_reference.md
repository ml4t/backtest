# Framework Behavior Reference

**Purpose**: Systematic catalog of execution semantics across backtesting frameworks.
**Last Updated**: 2026-01-02
**Frameworks**: VectorBT Pro, VectorBT OSS, Backtrader, Zipline-Reloaded

---

## 1. Order Execution Timing

### Market Orders

| Framework | Default Timing | Fill Price | Config Option |
|-----------|---------------|------------|---------------|
| VBT Pro | Same bar | Close price | (none) |
| VBT OSS | Same bar | Close price | (none) |
| Backtrader | Next bar | Open price | `broker.set_coo(True)` for same-bar |
| Zipline | Next bar | Open price | (none) |
| ml4t | Same bar | Close price | `ExecutionMode.SAME_BAR/NEXT_BAR` |

### Limit Orders

| Framework | Trigger Check | Fill Price |
|-----------|---------------|------------|
| VBT Pro | Close touches limit | Limit price |
| Backtrader | Intrabar (H/L) | Limit price |
| ml4t | Intrabar (H/L) | Limit price |

### Stop Orders

| Framework | Trigger Check | Fill Price |
|-----------|---------------|------------|
| VBT Pro | Intrabar (L for sell, H for buy) | Stop price |
| Backtrader | Intrabar (H/L) | Stop price |
| ml4t | Intrabar (H/L) | Stop price or gap-through |

---

## 2. Stop-Loss / Take-Profit

### Stop Level Calculation

| Framework | Basis Price | Example |
|-----------|-------------|---------|
| VBT Pro (same-bar) | Fill price | `stop = fill * (1 - pct)` |
| VBT OSS (same-bar) | Fill price | `stop = fill * (1 - pct)` |
| Backtrader (next-bar) | Signal close | `stop = signal_close * (1 - pct)` |
| Zipline (next-bar) | Signal close | Strategy handles in `handle_data()` |
| ml4t | Configurable | `StopLevelBasis.FILL_PRICE/SIGNAL_PRICE` |

### Trigger Detection

| Framework | Price Used | When Checked |
|-----------|-----------|--------------|
| VBT Pro | Bar LOW for stop (sells) | Intrabar simulation |
| VBT OSS | Bar LOW for stop (sells) | Intrabar simulation |
| Backtrader | Bar LOW/HIGH | Intrabar check |
| ml4t | Bar LOW/HIGH | `evaluate_position_rules()` |

### Fill Price Options

| Framework | Fill At | Gap-Through Handling |
|-----------|---------|---------------------|
| VBT Pro | Stop level | Opens beyond stop → fill at open |
| Backtrader | Stop level | Opens beyond stop → fill at open |
| Zipline | Next bar open | N/A (deferred fill) |
| ml4t | Configurable | `StopFillMode.STOP_PRICE/NEXT_BAR_OPEN` |

---

## 3. Trailing Stop (CRITICAL FINDINGS)

### High Water Mark (HWM) Source

| Framework | HWM Source | Notes |
|-----------|-----------|-------|
| **VBT Pro** | **HIGH** | Uses bar HIGH for HWM tracking |
| VBT OSS | HIGH | Same as Pro |
| Backtrader | Varies | Depends on implementation |
| ml4t | Configurable | `TrailHwmSource.HIGH/CLOSE` |

**Key Findings (2026-01-02)**:
1. **Initial HWM**: VBT Pro initializes HWM to bar's CLOSE (not fill price with slippage)
2. **Subsequent HWM**: Updates from bar's HIGH on following bars
3. **Trigger**: Uses bar's LOW to check if trail level breached
4. **Fill**: Uses trail level directly (not min/max with close)

Verified at scale: 99.3% exact match on 1,022 trades (100K data points).

### Trigger Detection

| Framework | Trigger Condition | Price Used |
|-----------|------------------|-----------|
| VBT Pro | `bar_low < trail_level` | Bar LOW |
| ml4t | `bar_low < trail_level` | Bar LOW |

### Trail Level Calculation

```
trail_level = HWM * (1 - trail_pct)  # For long positions
trail_level = LWM * (1 + trail_pct)  # For short positions
```

### Fill Price at Trigger

| Framework | Fill Price | With Slippage |
|-----------|-----------|---------------|
| **VBT Pro** | **min(trail_level, bar_close)** | Applied to min value |
| ml4t | Configurable | `StopFillMode.STOP_PRICE/CLOSE_PRICE` |

**CRITICAL FINDING (2026-01-02)**: VBT Pro's `tsl_stop` fills at `min(trail_level, bar_close)` for longs (or `max()` for shorts) when triggered. Example: trail_level=103.87, close=104.93 → fill=103.87*0.9995=103.82 (with slippage). Verified on asset_000/bar 89-92.

### VBT Pro Config for Exact Match

```python
# VBT Pro call
pf = vbt.Portfolio.from_signals(
    open=open_df, high=high_df, low=low_df, close=close_df,
    entries=entries, exits=exits,
    tsl_stop=0.03,  # 3% trailing stop
    init_cash=100_000.0,
    fees=0.001,
    slippage=0.0005,
)

# Matching ml4t config (achieves 99.3% exact match via CONFIGURATION)
from ml4t.backtest import InitialHwmSource, TrailHwmSource, StopFillMode

broker = Broker(
    100_000.0,
    PercentageCommission(0.001),
    PercentageSlippage(0.0005),
    trail_hwm_source=TrailHwmSource.HIGH,  # VBT Pro updates HWM from HIGH
    initial_hwm_source=InitialHwmSource.BAR_CLOSE,  # VBT Pro uses bar close for initial HWM
    stop_fill_mode=StopFillMode.STOP_PRICE,  # VBT Pro fills at trail level
)
broker.set_position_rules(TrailingStop(pct=0.03))

# Configuration options for trailing stop behavior:
# - trail_hwm_source: HIGH (bar high) or CLOSE (bar close) for HWM updates
# - initial_hwm_source: BAR_CLOSE (VBT Pro) or FILL_PRICE (most frameworks)
# - stop_fill_mode: STOP_PRICE (trail level), CLOSE_PRICE, or BAR_EXTREME
```

---

## 4. Commission Models

### VectorBT Pro

```python
# Percentage of trade value
pf = vbt.Portfolio.from_signals(
    ...,
    fees=0.001,  # 0.1% per trade (entry + exit separately)
)
# Commission per side: price * size * 0.001
```

### Backtrader

```python
# Percentage
cerebro.broker.setcommission(commission=0.001)  # 0.1% per trade

# Per-share
cerebro.broker.setcommission(commission=0.01, margin=False)  # $0.01 per share
```

### ml4t Configuration

```python
from ml4t.backtest.models import PercentageCommission, PerShareCommission

# Match VBT Pro
PercentageCommission(0.001)

# Match per-share
PerShareCommission(0.01)
```

---

## 5. Slippage Models

### VectorBT Pro

```python
# Percentage slippage
pf = vbt.Portfolio.from_signals(
    ...,
    slippage=0.0005,  # 0.05% slippage per side
)
# Fill price: base_price * (1 + slip) for buys, base_price * (1 - slip) for sells
```

### Backtrader

```python
# Fixed slippage
cerebro.broker.set_slippage_fixed(fixed=0.01)  # $0.01 per share

# Percentage
cerebro.broker.set_slippage_perc(perc=0.0005)  # 0.05%
```

### Zipline

```python
# Custom slippage model for open-price fills
from zipline.api import set_slippage
set_slippage(FixedSlippage(spread=0.0))  # Or custom model
```

### ml4t Configuration

```python
from ml4t.backtest.models import PercentageSlippage, FixedSlippage

# Match VBT Pro
PercentageSlippage(0.0005)

# Match Backtrader fixed
FixedSlippage(0.01)
```

---

## 6. Same-Bar Re-Entry After Exit

| Framework | Re-entry Allowed? | Notes |
|-----------|------------------|-------|
| VBT Pro | NO | `accumulate=False` blocks re-entry |
| Backtrader | NO | Order processing order prevents |
| ml4t | NO (with fix) | `_stop_exits_this_bar` tracking |

---

## 7. Multi-Asset Handling

### Position Tracking

| Framework | Cash Sharing | Position Limits |
|-----------|-------------|-----------------|
| VBT Pro | `cash_sharing=True/False` | Configurable |
| Backtrader | Single broker cash | By strategy |
| Zipline | Single account | By strategy |
| ml4t | Single broker cash | Portfolio limits |

### Order Rejection

| Framework | Rejection Behavior |
|-----------|-------------------|
| VBT Pro | Silently skip (no error) |
| Backtrader | Order rejected message |
| ml4t | Gatekeeper rejection |

---

## 8. VectorBT OSS vs Pro Differences

| Feature | VBT OSS | VBT Pro |
|---------|---------|---------|
| `lock_cash` default | `False` (DANGER!) | Leverage controlled |
| Short selling limits | None without `lock_cash=True` | Controlled |
| `.vbt` accessor | Conflicts with Pro | Conflicts with OSS |

**CRITICAL**: VBT OSS `lock_cash=False` allows unlimited leverage. Always set `lock_cash=True` for realistic results.

---

## 9. Configuration Mapping Summary

### To Match VBT Pro

```python
engine = Engine(
    feed, strategy,
    execution_mode=ExecutionMode.SAME_BAR,
    stop_fill_mode=StopFillMode.STOP_PRICE,
    stop_level_basis=StopLevelBasis.FILL_PRICE,
)
# For trailing stop:
broker.trail_hwm_source = TrailHwmSource.HIGH  # VBT Pro updates HWM from HIGH
broker.initial_hwm_source = InitialHwmSource.BAR_CLOSE  # VBT Pro uses bar close
broker.stop_fill_mode = StopFillMode.STOP_PRICE  # VBT Pro fills at trail level
```

### To Match Backtrader

```python
engine = Engine(
    feed, strategy,
    execution_mode=ExecutionMode.NEXT_BAR,
    stop_fill_mode=StopFillMode.STOP_PRICE,
    stop_level_basis=StopLevelBasis.SIGNAL_PRICE,
)
```

### To Match Zipline

```python
engine = Engine(
    feed, strategy,
    execution_mode=ExecutionMode.NEXT_BAR,
    stop_fill_mode=StopFillMode.NEXT_BAR_OPEN,
    stop_level_basis=StopLevelBasis.SIGNAL_PRICE,
)
```

---

## 10. Known Issues and Workarounds

### Python 3.12 Traceback Bug

```python
# traceback.format_exc() can crash with StopIteration error
try:
    result.errors.append(traceback.format_exc())
except RuntimeError:
    result.errors.append(f"Exception type: {type(e).__name__}")
```

### Zipline Bundle/Symbol Resolution

Zipline's `run_algorithm()` uses bundle data, not custom DataFrames. Excluded from validation.

### VBT OSS/Pro Accessor Conflict

Cannot coexist in same environment. Use separate venvs.

---

## 11. Test Environment Matrix

| Environment | Frameworks | Purpose |
|-------------|------------|---------|
| `.venv` | ml4t only | Main development |
| `.venv-vectorbt-pro` | VBT Pro | Commercial validation |
| `.venv-backtrader` | Backtrader | BT validation |
| `.venv-zipline` | Zipline | ZL validation |

---

*This document should be updated whenever new framework behavior is discovered.*
