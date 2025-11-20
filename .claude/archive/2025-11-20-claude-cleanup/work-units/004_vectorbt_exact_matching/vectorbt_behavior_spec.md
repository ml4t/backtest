# VectorBT Behavior Specification

**Work Unit**: 20251026_002_vectorbt_exact_matching
**Phase**: Investigation Phase (Tasks 1-11)
**Status**: 54% complete (6/11 tasks)
**Last Updated**: 2025-10-27

---

## Purpose

This document captures the verified behavior of VectorBT Pro's backtesting engine based on source code analysis (TASK-002) and empirical testing (TASK-005, TASK-006, TASK-007).

**Critical Note**: This documents VectorBT's implementation as a **baseline**. The future ml4t.backtest implementation must be **configurable** - these behaviors represent reasonable defaults, not the only options. Users should be able to configure:
- Exit price models (stop level vs actual OHLC vs market impact)
- Execution timing (intra-bar vs open/close only)
- Slippage models (fixed vs volume-based vs liquidity-based)

---

## Exit Behavior Summary

### Common Pattern: All Exits Use Stop Levels

**Critical Discovery**: TP, SL, and TSL all exhibit **symmetric behavior**:

| Exit Type | Trigger Condition | Exit Price | Reference |
|-----------|------------------|------------|-----------|
| **TP** | high >= tp_level | tp_level * (1 - slippage) | TASK-005 |
| **SL** | low <= sl_level | sl_level * (1 - slippage) | TASK-006 |
| **TSL** | low <= tsl_level | tsl_level * (1 - slippage) | TASK-007 |

**Key Insights**:
1. All three exit at their **stop level**, NOT at extreme prices (high/low)
2. All three apply **exit slippage** the same way: `stop_level * (1 - slippage)`
3. All three trigger **intra-bar** (check high/low, not just close)
4. This creates a **realistic and consistent** fill model across all exit types

---

## 1. Take Profit (TP) - TASK-005

### Trigger Condition
```
Triggers when: high >= tp_level (intra-bar)
```

### TP Level Calculation
```python
# TP calculated from BASE price (close without entry slippage)
base_price = close_at_entry  # NOT entry_price
tp_level = base_price * (1 + tp_stop)
```

### Exit Price
```python
exit_price = tp_level * (1 - slippage)  # NOT at high
```

### Example
```
Close at entry:     $43,885.00
Entry slippage:     +0.02%
Entry price:        $43,893.78

TP stop:            +2.5%
TP level:           $44,982.12  (= $43,885 * 1.025) ← From CLOSE
Exit slippage:      -0.02%
TP exit price:      $44,973.13  (= $44,982.12 * 0.9998)

High at exit bar:   $45,140.00 (TP triggered, but exit at tp_level)
```

### Confidence
95% - Verified with 3 test scenarios, all match within $0.01

---

## 2. Stop Loss (SL) - TASK-006

### Trigger Condition
```
Triggers when: low <= sl_level (intra-bar)
```

### SL Level Calculation
```python
# SL calculated from BASE price (same as TP)
base_price = close_at_entry  # NOT entry_price
sl_level = base_price * (1 - sl_stop)
```

### Exit Price
```python
exit_price = sl_level * (1 - slippage)  # NOT at low
```

### Example
```
Close at entry:     $45,825.00
Entry slippage:     +0.02%
Entry price:        $45,834.17

SL stop:            -2.5%
SL level:           $44,679.38  (= $45,825 * 0.975) ← From CLOSE
Exit slippage:      -0.02%
SL exit price:      $44,670.44  (= $44,679.38 * 0.9998)

Low at exit bar:    $44,605.00 (SL triggered, but exit at sl_level)
Favorable difference: $65.44 (1.46% better than exiting at low)
```

### Key Difference from Initial Understanding
**TASK-004 incorrectly stated**: "SL/TSL use actual low"
**TASK-006 corrected**: SL exits at sl_level (with slippage), NOT at low
- This is symmetric with TP behavior
- More favorable than exiting at low
- Realistic stop-limit order simulation

### Confidence
95% - Verified with 3 test scenarios, all match within $0.01

---

## 3. Trailing Stop Loss (TSL) - TASK-007

### Critical Discovery: TSL Based on Peak Price

**Most Important Finding**: TSL is calculated from **peak price**, not entry price

### Peak Tracking (4-Stage Process Per Bar)
```python
# Stage 1: Update peak with open
if bar.open > peak_price:
    peak_price = bar.open

# Stage 2: Check stop against open+H/L
tsl_level = peak_price * (1 - tsl_stop)
if bar.low <= tsl_level:
    trigger_exit()

# Stage 3: Update peak with high (long) or low (short)
if bar.high > peak_price:  # For long
    peak_price = bar.high

# Stage 4: Re-check stop against close
# (if not triggered in Stage 2)
```

### Trigger Condition
```
Triggers when: low <= tsl_level (intra-bar)
Where: tsl_level = peak_price * (1 - tsl_stop)  ← Peak, not entry!
```

### TSL Level Calculation
```python
# TSL follows the PEAK, not entry
peak_price = max(high) since entry  # Track highest high
tsl_level = peak_price * (1 - tsl_stop)
```

### Exit Price
```python
exit_price = tsl_level * (1 - slippage)  # Like SL, NOT at low
```

### Example
```
Entry price:        $43,893.78
Entry close:        $43,885.00

Price rises:
  Peak price:       $44,665.00  ← Highest high reached

TSL calculation:
  TSL stop:         1%
  TSL level:        $44,218.35  (= $44,665 * 0.99) ← From PEAK
  TSL with slip:    $44,209.51  (= $44,218.35 * 0.9998)

Exit:
  Actual exit:      $44,209.51 ✅ EXACT MATCH
  Low at exit:      $44,210.00
  Favorable diff:   $-0.49
```

### TSL with Threshold (Optional)
```python
# Optional parameter: tsl_th (threshold)
# TSL only activates after peak rises threshold% above entry

if tsl_th:
    threshold_price = entry_price * (1 + tsl_th)
    if peak_price < threshold_price:
        # TSL not active yet
        return None
```

**Example**:
```
Entry:          $44,183.83
Threshold:      2% → $45,067.51
Peak reached:   $45,600.00 ✅ Activates
TSL level:      $45,144.00 (= $45,600 * 0.99)
Exit:           $45,134.97
```

### Confidence
95% - Verified with 3 test scenarios, all match perfectly (< $0.01)

---

## 4. Exit Priority Order

From TASK-002 source analysis, confirmed in testing:

**Priority** (highest to lowest):
1. **SL** - Stop Loss (highest priority)
2. **TSL** - Trailing Stop Loss
3. **TP** - Take Profit
4. **TD** - Time-based duration stop
5. **DT** - Time-based datetime stop

**Rules**:
- **Open-triggered** exits have absolute priority over intra-bar
- Among open-triggered: SL > TSL > TP > TD > DT
- Among intra-bar: SL > TSL > TP > TD > DT
- **Only one exit per bar** - first match wins, others ignored
- **Not price-based**: Priority is deterministic, not based on which price hit first chronologically

---

## 5. Slippage Application

### Entry Slippage
```python
entry_price = close * (1 + slippage)  # Pay MORE to enter
```

### Exit Slippage
```python
exit_price = stop_level * (1 - slippage)  # Receive LESS on exit
```

**Key Points**:
- Entry and exit slippage are **independent**
- Both applied in the direction that hurts the trader
- Slippage is **directional**, not absolute
- Applied **before** fee calculation

---

## 6. Base Price Concept

**Critical Pattern**: All stops (TP, SL, TSL) calculate from "base price"

```python
# Base price = close at entry WITHOUT entry slippage
base_price = close_at_entry

# Entry execution includes slippage
entry_price = base_price * (1 + entry_slippage)

# But stops are relative to base price, not entry price
tp_level = base_price * (1 + tp_stop)
sl_level = base_price * (1 - sl_stop)

# TSL is special: tracks peak, not entry
# But initial reference can be thought of as base_price
```

**Implication**: Stop percentages are relative to "true" entry price (pre-slippage), making them slightly more conservative (TP a bit lower gain, SL a bit bigger loss than nominal percentages).

---

## 7. Fee Calculation

**Status**: Not yet documented (TASK-010)

**Known** from source analysis:
- Fees applied to both entry and exit
- Formula appears to be: `(order_value * fees) + fixed_fees`
- Order value calculated with slippage-adjusted price
- Order of operations: Slippage → Size → Fees → Cash

**To verify**: Exact fee calculation order and rounding

---

## 8. Position Sizing

**Status**: Not yet documented (TASK-009)

**Known** from source analysis:
- `size=np.inf` means "use all available cash"
- Formula: `max_size = (cash - fixed_fees) / (1 + fees) / adj_price`
- Where `adj_price` includes slippage

**To verify**: Exact position sizing calculation with fees and slippage

---

## 9. Intra-Bar Execution Model

**VectorBT's Approach**:
- Uses OHLC data to detect intra-bar price movements
- Assumes exits can occur at any point during the bar when condition met
- Exit prices are stop levels (not extreme prices), simulating limit/stop orders

**Future Configurability** (ml4t.backtest):
- **Default**: VectorBT-style (intra-bar with stop level fills)
- **Conservative**: Only check at open or close
- **Aggressive**: Assume fills at actual high/low
- **Realistic**: Market impact models based on liquidity and size

---

## 10. Symmetric Exit Behavior

**Key Architectural Insight**: All three exit types (TP, SL, TSL) follow the same pattern:

1. **Trigger**: Check intra-bar against condition (high for TP, low for SL/TSL)
2. **Fill**: Execute at stop level, not extreme price
3. **Slippage**: Apply same slippage formula to stop level
4. **Priority**: Deterministic order (SL > TSL > TP)

This creates a **consistent fill model** that is:
- **Realistic**: Simulates stop/limit order behavior
- **Conservative**: Doesn't assume best-case fills
- **Fair**: Doesn't assume worst-case fills either
- **Predictable**: Same logic across all exit types

---

## Implementation Roadmap for ml4t.backtest

### Phase 2 Remaining Tasks

**Investigation** (6/11 complete):
- ✅ TASK-001: VectorBT setup
- ✅ TASK-002: Source code analysis
- ✅ TASK-005: TP triggering
- ✅ TASK-006: SL triggering
- ✅ TASK-007: TSL tracking
- ⏳ TASK-008: Exit priority testing
- ⏳ TASK-009: Position sizing
- ⏳ TASK-010: Fee/slippage order
- ⏳ TASK-011: Intra-bar tests

**Implementation** (0/19 complete):
- TASK-012: Extract VectorBT trade log
- TASK-013: Build comparison framework
- TASK-014-030: Implement and validate exact matching

### Configurability Requirements (Future)

Based on findings, ml4t.backtest should make these aspects configurable:

1. **Exit Price Model**:
   - Option A: Stop level (VectorBT default) ✅
   - Option B: Actual OHLC price (high/low)
   - Option C: Market impact model (advanced)

2. **Slippage Model**:
   - Option A: Fixed percentage (VectorBT default) ✅
   - Option B: Volume-weighted
   - Option C: Liquidity-based
   - Option D: ML-predicted

3. **Execution Timing**:
   - Option A: Intra-bar (VectorBT default) ✅
   - Option B: Open only
   - Option C: Close only
   - Option D: Next bar open

4. **Priority Rules**:
   - Option A: Deterministic (VectorBT default) ✅
   - Option B: Price-based (first hit wins)
   - Option C: User-defined order

---

## References

- **TASK-002**: Source code analysis (`vectorbt_source_analysis.md`)
- **TASK-004**: Exit price calculation (`vectorbt_exit_price_analysis.md`)
- **TASK-005**: TP triggering (`TASK-005_COMPLETION.md`)
- **TASK-006**: SL triggering (`TASK-006_COMPLETION.md`)
- **TASK-007**: TSL tracking (`TASK-007_COMPLETION.md`)

---

**Last Updated**: 2025-10-27
**Next Update**: After completing TASK-008 through TASK-011 (exit priority, sizing, fees)
