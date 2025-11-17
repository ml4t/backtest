# Framework Execution Models - Source Code Analysis

**Created**: 2025-11-16
**Purpose**: Document exact execution behavior from framework source code
**Status**: LIVING DOCUMENT - Updated as we discover implementation details

---

## Overview

This document contains **evidence-based** descriptions of how each benchmark framework executes backtests, derived from reading actual source code in `resources/`.

**Methodology**: Direct source code examination, not documentation or assumptions.

---

## VectorBT (Open Source + Pro)

### Source Code Location
- **OSS**: `resources/vectorbt/vectorbt/portfolio/`
- **Pro**: `resources/vectorbt.pro-main/vectorbtpro/portfolio/`

### Portfolio.from_signals() Execution Model

**Primary file**: `portfolio/base.py`
**Numba execution**: `portfolio/nb/from_signals.py`

#### Default Fill Behavior

**From source code analysis**:
```python
# portfolio/base.py (approximate line 3245)
def from_signals(
    cls,
    close,          # Required: close prices
    entries=None,   # Entry signals (boolean)
    exits=None,     # Exit signals (boolean)
    price=None,     # Execution price (defaults to close!)
    ...
)
```

**KEY FINDING** 🔍:
- **Default fill price**: `close` (same-bar close)
- **Look-ahead risk**: HIGH (you know the bar close before "executing")
- **Alignment strategy**: Set `price=open.shift(-1)` for next-bar execution

#### Signal Processing Logic

**TODO**: Read `portfolio/nb/from_signals.py` to understand:
- How `accumulate` parameter affects same-bar re-entry
- Order in which commission and slippage are applied
- Position sizing calculation with fees

**Expected location**: Lines 100-300 in `from_signals.py` (Numba-compiled function)

#### Commission & Slippage Application

**TODO**: Verify from source code:
```python
# Hypothesis: Commission on (price + slippage)
fill_price = price * (1 + slippage)
commission = fill_price * quantity * fee_rate
```

**File to check**: `portfolio/nb/from_signals.py` or `portfolio/base.py`

---

## Backtrader

### Source Code Location
- **Main**: `resources/backtrader-master/backtrader/`

### Broker Execution Model

**Primary file**: `brokers/bbroker.py` (BackBroker class)

#### Cheat-On-Open (COO) and Cheat-On-Close (COC)

**From source code** (`brokers/bbroker.py` lines 175-189):
```python
params = (
    ('coo', False),  # Cheat-On-Open
    ('coc', False),  # Cheat-On-Close
    ...
)
```

**COO Behavior** (lines 175-181 comments):
> "Setting to True enables matching a Market order to the opening price by using a timer with cheat=True, because timer executes before broker evaluation"

**Interpretation**:
- **COO=False** (default): Orders placed in `next()` fill at **next bar's open**
- **COO=True**: Orders placed in `next_open()` fill at **current bar's open**
- **COC=True**: Market orders fill at **current bar's close**

**Look-ahead risk**:
- COO=False: ✅ Safe (next-bar execution)
- COO=True: ⚠️ Moderate (can see close, fill at open)
- COC=True: 🚨 HIGH (know bar is complete, fill at close)

#### Default Fill Timing

**TODO**: Read `_execute()` method in `brokers/bbroker.py` to confirm:
```python
# Expected behavior (needs verification)
def _execute(self, order, bar):
    if order.type == Order.Market:
        if self.p.coc:
            order.execute(bar.close, ...)
        else:
            order.execute(next_bar.open, ...)  # DEFAULT
```

**Lines to check**: 450-550 in `bbroker.py`

#### Slippage Application

**From parameters** (`brokers/bbroker.py` lines 175-189):
```python
params = (
    ('slip_perc', 0.01),      # 1% slippage (0.01 = 1%)
    ('slip_fixed', 0.0),      # Fixed price slippage
    ('slip_open', False),     # Apply slippage to opening fills
    ('slip_match', True),     # Cap slippage at high/low
    ('slip_limit', True),     # Allow limit order slippage capping
    ('slip_out', False),      # Slippage outside high-low range
)
```

**TODO**: Verify slippage calculation order:
- Is slippage before or after commission?
- How does `slip_match` cap work exactly?

**Lines to check**: 500-600 in `bbroker.py`

---

## Zipline-Reloaded

### Source Code Location
- **Main**: `resources/zipline-reloaded-main/src/zipline/`

### Order Execution Model

**Primary file**: `finance/execution.py`

**TODO**: Read `execution.py` to understand:
- When orders placed in `handle_data()` actually fill
- How `FixedSlippage` affects fill price
- Commission calculation order

**Expected behavior** (needs verification):
- Orders placed in `handle_data(context, data)` for bar N
- Fill at bar N+1 open price
- Cannot achieve same-bar execution (fundamental design)

#### Commission Models

**Source file**: `finance/commission.py`

**From earlier research** (needs line number verification):
```python
class PerShare(CommissionModel):
    def __init__(self, cost=0.001, min_trade_cost=0.0):
        # Default: $0.001/share (0.1 cents), no minimum
        ...

class PerTrade(CommissionModel):
    def __init__(self, cost=0.0):
        # Flat fee per trade
        # Note: "Full commission charged to first fill"
        ...

class PerDollar(CommissionModel):
    def __init__(self, cost=0.0015):
        # Default: $0.0015/dollar (0.15 cents)
        ...
```

**TODO**: Verify exact commission calculation:
```python
# Expected logic
commission = {
    'PerShare': quantity * cost,
    'PerTrade': cost,  # One-time
    'PerDollar': fill_price * quantity * cost
}
```

**Lines to check**: 50-150 in `commission.py`

#### Slippage Models

**Source file**: `finance/slippage.py`

**From earlier research** (needs line number verification):
```python
class FixedSlippage(SlippageModel):
    def __init__(self, spread=0.0):
        # Buy fills: close + (spread / 2)
        # Sell fills: close - (spread / 2)
        ...

class VolumeShareSlippage(SlippageModel):
    def __init__(self, volume_limit=0.025, price_impact=0.1):
        # volume_limit: max % of bar volume (default 2.5%)
        # price_impact: price impact coefficient
        ...
```

**TODO**: Read actual implementation to verify:
- How spread is split between buy/sell
- Order of slippage and commission application
- Volume constraint enforcement

**Lines to check**: 80-200 in `slippage.py`

---

## ml4t.backtest (Reference Implementation)

### Source Code Location
- **Main**: `src/ml4t/backtest/`

### Fill Simulator

**Primary file**: `execution/fill_simulator.py`

**Current understanding** (based on previous work):
- Orders placed on bar N
- Fill at bar N+1 open price (next-bar execution)
- OHLC validation (fill price must be within bar range)

**TODO**: Document exact implementation:
```python
# Expected logic (needs verification)
class FillSimulator:
    def simulate_fill(self, order, bar):
        if order.type == OrderType.MARKET:
            fill_price = bar.open  # Next bar open
            fill_price = self._apply_slippage(fill_price, order)
            commission = self._calculate_commission(fill_price, order)
            return Fill(price=fill_price, commission=commission, ...)
```

**Lines to check**: 100-200 in `fill_simulator.py`

### Commission & Slippage Order

**Files to read**:
- `execution/commission.py`
- `execution/slippage.py`
- `execution/fill_simulator.py`

**TODO**: Verify order of operations:
1. Determine base fill price (open, close, etc.)
2. Apply slippage → adjusted_price
3. Calculate commission on adjusted_price
4. Return final fill

**Expected sequence**:
```python
base_price = bar.open
slipped_price = base_price * (1 + slippage_pct)
commission = slipped_price * quantity * commission_rate
total_cost = (slipped_price * quantity) + commission
```

---

## Execution Timing Comparison Matrix

| Framework | Default Fill Timing | Configurable? | Look-Ahead Risk |
|-----------|---------------------|---------------|-----------------|
| **VectorBT** | Same bar close | ✅ Via `price` param | 🚨 HIGH (close known) |
| **Backtrader** | Next bar open | ✅ COO/COC flags | ✅ LOW (default safe) |
| **Zipline** | Next bar open | ❌ No (built-in) | ✅ LOW (safe) |
| **ml4t.backtest** | Next bar open | ✅ TBD | ✅ LOW (safe) |

**For validation alignment**:
- VectorBT: Set `price=open.shift(-1)` to match others
- Backtrader: Keep `coo=False`, `coc=False` (defaults)
- Zipline: No configuration needed (always next-bar)
- ml4t.backtest: Verify default behavior

---

## Commission & Slippage Order of Operations

### Hypothesis (Needs Verification)

**VectorBT**:
```python
# Order: Slippage first, then commission
fill_price = price * (1 + slippage)
commission = fill_price * quantity * fee_rate
cost = (fill_price * quantity) + commission
```

**Backtrader**:
```python
# Order: Commission on base price, slippage separate
commission = price * quantity * commission_rate
slipped_price = price * (1 + slip_perc)
cost = (slipped_price * quantity) + commission
```

**Zipline**:
```python
# Order: TBD (need to read source)
# Hypothesis: Similar to VectorBT
```

**ml4t.backtest**:
```python
# Order: TBD (need to read source)
# Expected: Slippage first, commission on slipped price
```

**TODO**: Read source code to confirm each framework's exact sequence.

---

## Investigation Tasks

### High Priority (Phase 1)

- [ ] Read VectorBT `from_signals.py` (Numba) - fill price logic
- [ ] Read Backtrader `bbroker.py` - `_execute()` method (lines 450-600)
- [ ] Read Zipline `execution.py` - order placement → fill timing
- [ ] Read ml4t.backtest `fill_simulator.py` - complete fill logic

### Medium Priority (Phase 2)

- [ ] Verify commission calculation order in each framework
- [ ] Verify slippage application order in each framework
- [ ] Document same-bar re-entry handling
- [ ] Document position sizing with fees

### Low Priority (Phase 3)

- [ ] Limit order fill logic comparison
- [ ] Stop order trigger logic comparison
- [ ] Partial fills handling

---

## Source Code Reading Log

### 2025-11-16: Initial Framework Survey

**Backtrader**: Read `brokers/bbroker.py` lines 175-189
- Found COO/COC parameters
- Documented default settings
- **Next**: Read `_execute()` method for fill logic

**VectorBT**: Examined `portfolio/base.py` signature
- Identified `price` parameter defaults to `close`
- **Next**: Read Numba implementation in `nb/from_signals.py`

**Zipline**: Located key files
- `finance/execution.py` - order placement
- `finance/commission.py` - commission models
- `finance/slippage.py` - slippage models
- **Next**: Read execution timing logic

**ml4t.backtest**: Located key files
- `execution/fill_simulator.py`
- **Next**: Document exact fill sequence

---

## Updates Required

As we read source code, update this document with:
1. Exact file paths and line numbers
2. Code snippets proving behavior
3. Confirmed vs hypothetical statements
4. Cross-references between frameworks

**This is a living document - update after every source code investigation.**
