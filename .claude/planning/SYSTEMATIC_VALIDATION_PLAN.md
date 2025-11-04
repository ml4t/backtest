# Systematic Backtesting Engine Validation Plan

**Goal**: Validate qengine against 3 reference implementations by testing isolated functionality incrementally

**Engines to Compare**:
1. VectorBT Pro (reference #1)
2. Zipline (reference #2)
3. Backtrader (reference #3)
4. qengine (implementation under test)

**Validation Principle**: Start simple, add one variable at a time, document differences

---

## Phase 1: Baseline - Entry Signals Only

**Objective**: Verify all engines execute the same trades with identical entry signals

### Test 1.1: Single Asset, Market Orders, No Costs

**Configuration**:
- Asset: BTC futures (minute data, 1000 bars)
- Signals: Fixed entry signals (e.g., every 50 bars)
- Order Type: Market orders (entry only, hold until next entry)
- Fees: 0.0
- Slippage: 0.0
- Initial Cash: $100,000

**Test Data Options**:
- Option A: Real data (first 1000 bars of BTC futures)
- Option B: Synthetic data (controlled OHLCV with known properties)

**Success Criteria**:
- [ ] All 4 engines generate same number of trades
- [ ] All entry timestamps match exactly
- [ ] All entry prices match exactly (should be close price)
- [ ] Position sizes identical
- [ ] Final portfolio value identical

**Validation Script**: `tests/validation/test_1_1_baseline_entries.py`

**Expected Output**:
```
Engine         Trades  Entry_1_Price  Entry_1_Time           Final_Value
VectorBT       20      $35,000.00     2021-01-04 00:50:00   $105,234.56
Zipline        20      $35,000.00     2021-01-04 00:50:00   $105,234.56
Backtrader     20      $35,000.00     2021-01-04 00:50:00   $105,234.56
qengine        20      $35,000.00     2021-01-04 00:50:00   $105,234.56
Status: ✅ PASS - All engines identical
```

### Test 1.2: Entry + Exit Signals, No Costs

**Configuration**:
- Same as 1.1, but add exit signals
- Exit signals: Fixed (e.g., 10 bars after each entry)

**Success Criteria**:
- [ ] All 4 engines generate same entry/exit pairs
- [ ] Exit timestamps match
- [ ] Exit prices match (should be close price)
- [ ] PnL per trade matches
- [ ] Final value matches

**Validation Script**: `tests/validation/test_1_2_baseline_exits.py`

---

## Phase 2: Costs - Isolated Testing

**Objective**: Verify cost models (fees and slippage) work identically

### Test 2.1: Fees Only (No Slippage)

**Configuration**:
- Same signals as 1.2
- Fees: 0.02% (0.0002)
- Slippage: 0.0

**Success Criteria**:
- [ ] All engines apply 0.02% fee to entry fills
- [ ] All engines apply 0.02% fee to exit fills
- [ ] Total fees identical across engines
- [ ] Final value matches

**Validation Script**: `tests/validation/test_2_1_fees_only.py`

**Expected Differences**:
- Document if any engine applies fees differently (e.g., to notional vs. cash)

### Test 2.2: Slippage Only (No Fees)

**Configuration**:
- Same signals as 1.2
- Fees: 0.0
- Slippage: 0.02% (0.0002)

**Success Criteria**:
- [ ] All engines apply slippage to fill prices
- [ ] Entry fill = close × (1 + slippage) for longs
- [ ] Entry fill = close × (1 - slippage) for shorts
- [ ] Total slippage cost identical
- [ ] Final value matches

**Validation Script**: `tests/validation/test_2_2_slippage_only.py`

**Key Question**: Does qengine's SlippageModel actually get applied?

### Test 2.3: Fees + Slippage Combined

**Configuration**:
- Same signals as 1.2
- Fees: 0.02%
- Slippage: 0.02%

**Success Criteria**:
- [ ] Combined costs match across engines
- [ ] Final value matches

**Validation Script**: `tests/validation/test_2_3_fees_and_slippage.py`

---

## Phase 3: Order Types

**Objective**: Verify different order types execute identically

### Test 3.1: Limit Orders (Entry)

**Configuration**:
- Signals: Same as baseline
- Order Type: Limit orders (e.g., limit = close × 0.99)
- Fees/Slippage: 0.0

**Success Criteria**:
- [ ] Limit orders only fill when price reaches limit
- [ ] Fill prices match limit price
- [ ] Unfilled orders handled identically
- [ ] Trade count matches

**Validation Script**: `tests/validation/test_3_1_limit_orders.py`

### Test 3.2: Stop Orders (Exit)

**Configuration**:
- Entry: Market orders
- Exit: Stop loss orders (e.g., stop = entry × 0.95)
- Fees/Slippage: 0.0

**Success Criteria**:
- [ ] Stop orders trigger at correct price
- [ ] Fill prices match across engines
- [ ] Trade count matches

**Validation Script**: `tests/validation/test_3_2_stop_orders.py`

### Test 3.3: Bracket Orders (TP + SL)

**Configuration**:
- Entry: Market orders
- Exit: Bracket with TP=+4%, SL=-2%
- Fees/Slippage: 0.0

**Success Criteria**:
- [ ] Bracket legs created correctly
- [ ] First triggered exit cancels other leg (OCO)
- [ ] Fill prices match
- [ ] Trade count matches

**Validation Script**: `tests/validation/test_3_3_bracket_orders.py`

### Test 3.4: Trailing Stop Loss

**Configuration**:
- Entry: Market orders
- Exit: TSL = 2% from peak
- Fees/Slippage: 0.0

**Success Criteria**:
- [ ] Peak tracking identical
- [ ] TSL trigger logic matches
- [ ] Fill prices match
- [ ] Trade count matches

**Validation Script**: `tests/validation/test_3_4_trailing_stop.py`

**Key Validation**: VectorBT 4-stage TSL logic (update peak with open, check, update with high, check again)

---

## Phase 4: Position Management

**Objective**: Verify position tracking and re-entry logic

### Test 4.1: Same-Bar Re-entry

**Configuration**:
- Signals: Exit and re-entry on same bar
- Order Type: Market orders
- Fees/Slippage: 0.0

**Success Criteria**:
- [ ] All engines allow same-bar re-entry
- [ ] Position state correct after re-entry
- [ ] Trade count matches

**Validation Script**: `tests/validation/test_4_1_same_bar_reentry.py`

**Key Issue**: This may explain the 117 missing trades!

### Test 4.2: Position Sizing

**Configuration**:
- Signals: Same as baseline
- Size: Fixed quantity (e.g., 1.0 BTC)
- Fees/Slippage: 0.0

**Success Criteria**:
- [ ] All engines use same position size
- [ ] Cash constraints handled identically
- [ ] Partial fills (if any) match

**Validation Script**: `tests/validation/test_4_2_position_sizing.py`

### Test 4.3: Accumulation (Adding to Position)

**Configuration**:
- Signals: Multiple entries without exits
- Order Type: Market orders
- Fees/Slippage: 0.0

**Success Criteria**:
- [ ] Position size accumulates correctly
- [ ] Average entry price calculated identically
- [ ] Trade count matches

**Validation Script**: `tests/validation/test_4_3_accumulation.py`

---

## Phase 5: Multi-Asset

**Objective**: Verify portfolio-level behavior with multiple assets

### Test 5.1: Two Assets, Independent Signals

**Configuration**:
- Assets: BTC + ETH
- Signals: Independent entry/exit for each
- Order Type: Market orders
- Cash Sharing: False
- Fees/Slippage: 0.0

**Success Criteria**:
- [ ] Each asset trades independently
- [ ] Cash allocated correctly
- [ ] Final value matches

**Validation Script**: `tests/validation/test_5_1_multi_asset_independent.py`

### Test 5.2: Cash Sharing

**Configuration**:
- Assets: BTC + ETH
- Signals: Overlapping signals
- Cash Sharing: True
- Fees/Slippage: 0.0

**Success Criteria**:
- [ ] Cash shared across assets
- [ ] Order execution priority matches
- [ ] Final value matches

**Validation Script**: `tests/validation/test_5_2_cash_sharing.py`

---

## Phase 6: Advanced Features

**Objective**: Test edge cases and advanced scenarios

### Test 6.1: Intrabar Execution (High/Low)

**Configuration**:
- Order Type: Limit orders
- Validation: Use intrabar high/low for fills
- Fees/Slippage: 0.0

**Success Criteria**:
- [ ] Limit orders fill using high/low
- [ ] Fill detection matches across engines

**Validation Script**: `tests/validation/test_6_1_intrabar_execution.py`

### Test 6.2: Conflicting Signals (Long + Short)

**Configuration**:
- Signals: Both long and short signal on same bar
- Conflict Resolution: Configurable (cancel, take long, take short)

**Success Criteria**:
- [ ] Conflict handling matches across engines
- [ ] Trade count matches

**Validation Script**: `tests/validation/test_6_2_conflicting_signals.py`

---

## Implementation Plan

### Directory Structure

```
tests/validation/
├── README.md                          # Overview and how to run
├── common/
│   ├── data_generator.py              # Synthetic OHLCV data
│   ├── signal_generator.py            # Fixed signals for all engines
│   ├── engine_wrappers.py             # Unified API for 4 engines
│   └── comparison.py                  # Trade comparison logic
├── test_1_1_baseline_entries.py
├── test_1_2_baseline_exits.py
├── test_2_1_fees_only.py
├── test_2_2_slippage_only.py
├── test_2_3_fees_and_slippage.py
├── test_3_1_limit_orders.py
├── test_3_2_stop_orders.py
├── test_3_3_bracket_orders.py
├── test_3_4_trailing_stop.py
├── test_4_1_same_bar_reentry.py
├── test_4_2_position_sizing.py
├── test_4_3_accumulation.py
├── test_5_1_multi_asset_independent.py
├── test_5_2_cash_sharing.py
├── test_6_1_intrabar_execution.py
└── test_6_2_conflicting_signals.py
```

### Common Infrastructure

**1. Unified Engine Wrapper** (`engine_wrappers.py`):
```python
class EngineWrapper(ABC):
    @abstractmethod
    def run_backtest(self, ohlcv, signals, config):
        """Run backtest and return standardized results."""
    
    @abstractmethod
    def get_trades(self):
        """Return trades in standard format."""

class VectorBTWrapper(EngineWrapper): ...
class ZiplineWrapper(EngineWrapper): ...
class BacktraderWrapper(EngineWrapper): ...
class QEngineWrapper(EngineWrapper): ...
```

**2. Signal Generator** (`signal_generator.py`):
```python
def generate_fixed_entries(n_bars, entry_every=50):
    """Generate entry signals every N bars."""
    
def generate_entry_exit_pairs(n_bars, hold_bars=10):
    """Generate entry/exit signal pairs."""
```

**3. Comparison Tool** (`comparison.py`):
```python
def compare_trades(vbt_trades, zipline_trades, bt_trades, qengine_trades):
    """Compare trades across all engines and report differences."""
    
def assert_identical(results_dict, tolerance=0.01):
    """Assert all engines produced identical results."""
```

### Validation Script Template

```python
"""Test 1.1: Baseline - Entry Signals Only"""
import pytest
from tests.validation.common import (
    VectorBTWrapper, ZiplineWrapper, BacktraderWrapper, QEngineWrapper,
    generate_fixed_entries, generate_ohlcv, compare_trades
)

def test_baseline_entries():
    # 1. Generate test data
    ohlcv = generate_ohlcv(n_bars=1000, symbol="BTC")
    signals = generate_fixed_entries(n_bars=1000, entry_every=50)
    
    # 2. Configuration (same for all engines)
    config = {
        'initial_cash': 100000,
        'fees': 0.0,
        'slippage': 0.0,
        'order_type': 'market',
    }
    
    # 3. Run all engines
    vbt = VectorBTWrapper()
    vbt_result = vbt.run_backtest(ohlcv, signals, config)
    
    zipline = ZiplineWrapper()
    zipline_result = zipline.run_backtest(ohlcv, signals, config)
    
    bt = BacktraderWrapper()
    bt_result = bt.run_backtest(ohlcv, signals, config)
    
    qengine = QEngineWrapper()
    qengine_result = qengine.run_backtest(ohlcv, signals, config)
    
    # 4. Compare results
    results = {
        'VectorBT': vbt_result,
        'Zipline': zipline_result,
        'Backtrader': bt_result,
        'qengine': qengine_result,
    }
    
    # 5. Assert identical
    comparison = compare_trades(results)
    print(comparison.to_string())
    
    assert comparison['trade_count_match'] == True
    assert comparison['prices_match'] == True
    assert comparison['final_value_match'] == True
```

---

## Execution Checklist

### Phase 1: Baseline
- [ ] Test 1.1: Entry signals only
- [ ] Test 1.2: Entry + exit signals
- [ ] Document: Any baseline differences

### Phase 2: Costs
- [ ] Test 2.1: Fees only
- [ ] Test 2.2: Slippage only
- [ ] Test 2.3: Fees + slippage
- [ ] Document: How each engine applies costs

### Phase 3: Order Types
- [ ] Test 3.1: Limit orders
- [ ] Test 3.2: Stop orders
- [ ] Test 3.3: Bracket orders
- [ ] Test 3.4: Trailing stop
- [ ] Document: Order type differences

### Phase 4: Position Management
- [ ] Test 4.1: Same-bar re-entry
- [ ] Test 4.2: Position sizing
- [ ] Test 4.3: Accumulation
- [ ] Document: Position tracking differences

### Phase 5: Multi-Asset
- [ ] Test 5.1: Independent assets
- [ ] Test 5.2: Cash sharing
- [ ] Document: Portfolio-level differences

### Phase 6: Advanced
- [ ] Test 6.1: Intrabar execution
- [ ] Test 6.2: Conflicting signals
- [ ] Document: Edge case handling

---

## Deliverables

For each test:
1. **Test Script**: Executable pytest
2. **Results Table**: Comparison across 4 engines
3. **Difference Report**: Document any discrepancies with explanation
4. **Pass/Fail Status**: Clear success criteria

**Final Deliverable**: Validation matrix showing which scenarios pass/fail for each engine

---

## Next Steps

1. **Create infrastructure** (engine wrappers, data generators)
2. **Run Test 1.1** (simplest baseline)
3. **Fix any qengine issues** found in Test 1.1
4. **Proceed incrementally** through each test
5. **Document all differences** in a master comparison table

**Success Metric**: All 17 tests pass for qengine (identical to reference engines)

