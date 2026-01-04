# Configuration Mapping: ml4t-backtest ↔ External Frameworks

**Purpose**: Direct mapping of configuration options between ml4t-backtest and VectorBT Pro, VectorBT OSS, Backtrader, and Zipline.

**Last Updated**: 2026-01-02

---

## VectorBT Pro → ml4t-backtest

### Execution Configuration

| VBT Pro Parameter | VBT Pro Value | ml4t Parameter | ml4t Value |
|-------------------|---------------|----------------|------------|
| (default) | same-bar close | `execution_mode` | `ExecutionMode.SAME_BAR` |
| `accumulate` | `False` | Strategy logic | Check `current_qty == 0` before entry |
| `size_type` | `"amount"` | Strategy logic | Pass share count to `submit_order()` |

### Trailing Stop Configuration

| VBT Pro Parameter | VBT Pro Value | ml4t Parameter | ml4t Value |
|-------------------|---------------|----------------|------------|
| `tsl_stop` | `0.05` | `TrailingStop(pct=)` | `0.05` |
| (with OHLC) | Updates HWM from bar HIGH | `trail_hwm_source` | `TrailHwmSource.HIGH` |
| (default) | Initial HWM = bar CLOSE | `initial_hwm_source` | `InitialHwmSource.BAR_CLOSE` |
| (default) | Fills at trail level | `stop_fill_mode` | `StopFillMode.STOP_PRICE` |
| (gap-through) | Fills at bar OPEN | Built-in | TrailingStop handles internally |

**CRITICAL**: VBT Pro uses CLOSE for initial HWM on entry bar, then updates from
bar HIGHs starting on the NEXT bar. The entry bar's HIGH is NOT included in
initial HWM calculation.

### Stop-Loss / Take-Profit

| VBT Pro Parameter | VBT Pro Value | ml4t Config | ml4t Value |
|-------------------|---------------|-------------|------------|
| `sl_stop` | `0.05` | `StopLoss(pct=)` | `0.05` |
| `tp_stop` | `0.10` | `TakeProfit(pct=)` | `0.10` |
| (with OHLC) | Intrabar trigger | Pass OHLC to DataFeed | Same behavior |

### Cost Models

| VBT Pro Parameter | VBT Pro Value | ml4t Model | ml4t Value |
|-------------------|---------------|------------|------------|
| `fees` | `0.001` | `PercentageCommission` | `0.001` |
| `slippage` | `0.0005` | `PercentageSlippage` | `0.0005` |

---

## VectorBT OSS → ml4t-backtest

Same as VBT Pro except:
- No OHLC support for trailing stops (close-only)
- Use `TrailHwmSource.CLOSE` instead of `HIGH`

---

## Backtrader → ml4t-backtest

### Execution Configuration

| Backtrader Setting | Backtrader Value | ml4t Parameter | ml4t Value |
|-------------------|------------------|----------------|------------|
| Default execution | Next-bar open | `execution_mode` | `ExecutionMode.NEXT_BAR` |
| `set_coo(True)` | Cheat-on-open | `execution_mode` | `ExecutionMode.SAME_BAR` |
| `set_coc(True)` | Cheat-on-close | `execution_mode` | `ExecutionMode.SAME_BAR` |

### Commission Configuration

| Backtrader Setting | Backtrader Value | ml4t Model | ml4t Value |
|-------------------|------------------|------------|------------|
| `commission` | `0.001` | `PercentageCommission` | `0.001` |
| `COMM_FIXED` | Per-share | `PerShareCommission` | Per-share cost |
| `mult` | Contract multiplier | `ContractSpec.multiplier` | Same |

### Stop Order Configuration

| Backtrader | ml4t |
|------------|------|
| `bt.Order.Stop` | `StopLoss` rule |
| `bt.Order.StopLimit` | Not yet implemented |
| Gap handling | `StopFillMode.STOP_PRICE` with gap logic |

---

## Zipline → ml4t-backtest

### Execution Configuration

| Zipline Setting | Zipline Value | ml4t Parameter | ml4t Value |
|-----------------|---------------|----------------|------------|
| Default execution | Next-bar open | `execution_mode` | `ExecutionMode.NEXT_BAR` |
| Order processing | After market open | Same behavior | Same |

### Slippage Configuration

| Zipline Model | Zipline Value | ml4t Model | ml4t Value |
|---------------|---------------|------------|------------|
| `VolumeShareSlippage` | Volume-based | `PercentageSlippage` | Approximation |
| `FixedSlippage` | `spread=0.05` | `FixedSlippage` | `0.05` |

### Commission Configuration

| Zipline Model | Zipline Value | ml4t Model | ml4t Value |
|---------------|---------------|------------|------------|
| `PerShare` | `cost=0.005` | `PerShareCommission` | `0.005` |
| `PerTrade` | `cost=10.0` | `FixedCommission` | `10.0` |

---

## Full Configuration Examples

### Match VectorBT Pro

```python
from ml4t.backtest import (
    Engine, DataFeed, Strategy,
    TrailHwmSource, InitialHwmSource, StopFillMode, ExecutionMode,
    PercentageCommission, PercentageSlippage, NoCommission, NoSlippage,
)
from ml4t.backtest.risk.position import TrailingStop, StopLoss, TakeProfit

# VBT Pro equivalent:
# pf = vbt.Portfolio.from_signals(
#     open=df["open"], high=df["high"], low=df["low"], close=df["close"],
#     entries=entries, exits=exits,
#     init_cash=100_000, size=100, size_type="amount",
#     fees=0.001, slippage=0.0005,
#     tsl_stop=0.05,
#     accumulate=False,
# )

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

# In strategy.on_start():
broker.set_position_rules(TrailingStop(pct=0.05))
```

### Match Backtrader (Default)

```python
# Backtrader equivalent:
# cerebro = bt.Cerebro()
# cerebro.broker.setcommission(commission=0.001)
# cerebro.addstrategy(MyStrategy)

engine = Engine(
    feed,
    strategy,
    execution_mode=ExecutionMode.NEXT_BAR,
    commission_model=PercentageCommission(0.001),
    slippage_model=NoSlippage(),
)
```

### Match Backtrader (Cheat-on-Close)

```python
# Backtrader equivalent:
# cerebro.broker.set_coc(True)

engine = Engine(
    feed,
    strategy,
    execution_mode=ExecutionMode.SAME_BAR,
    commission_model=PercentageCommission(0.001),
)
```

---

## Enum Value Reference

### ExecutionMode

| Enum | Value | Description |
|------|-------|-------------|
| `SAME_BAR` | 0 | Execute on signal bar (VBT Pro default) |
| `NEXT_BAR` | 1 | Execute on next bar open (Backtrader/Zipline default) |

### TrailHwmSource

| Enum | Value | Description |
|------|-------|-------------|
| `CLOSE` | 0 | Update HWM from close price (VBT OSS) |
| `HIGH` | 1 | Update HWM from bar high (VBT Pro with OHLC) |

### InitialHwmSource

| Enum | Value | Description |
|------|-------|-------------|
| `FILL_PRICE` | 0 | Initial HWM = fill price including slippage |
| `BAR_CLOSE` | 1 | Initial HWM = bar close (VBT Pro behavior) |

### StopFillMode

| Enum | Value | Description |
|------|-------|-------------|
| `STOP_PRICE` | 0 | Fill at exact stop level (default) |
| `CLOSE_PRICE` | 1 | Fill at bar close |
| `BAR_EXTREME` | 2 | Fill at bar low (long) or high (short) |
| `NEXT_BAR_OPEN` | 3 | Defer to next bar open (Zipline) |

---

## Quick Reference Table

| Behavior | VBT Pro | VBT OSS | Backtrader | Zipline |
|----------|---------|---------|------------|---------|
| Same-bar exec | Default | Default | `set_coc(True)` | N/A |
| Next-bar exec | N/A | N/A | Default | Default |
| Trail from HIGH | Default (OHLC) | N/A | N/A | N/A |
| Trail from CLOSE | close-only | Default | N/A | N/A |
| % commission | `fees=0.001` | `fees=0.001` | `commission=0.001` | N/A |
| Per-share comm | N/A | N/A | `COMM_FIXED` | `PerShare` |
| % slippage | `slippage=0.0005` | `slippage=0.0005` | N/A | N/A |
