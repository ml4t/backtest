# ml4t.backtest Validation Roadmap

**Purpose**: Systematic validation of ml4t.backtest execution against VectorBT Pro
**Approach**: Test one variable at a time, increasing complexity incrementally
**Status**: Phase 1 (Baseline) in progress
**Last Updated**: 2025-11-04

---

## Testing Philosophy

**Guiding Principles** (from user requirements):
1. **Baseline first**: Same signals, no fees, no slippage
2. **One variable at a time**: Add fees, then slippage, then order types
3. **Systematic progression**: Simple → Complex
4. **Exact matching**: Synthetic data for reproducibility
5. **Document differences**: Explain any discrepancies

**Test Structure**:
- **Phases 1-3**: Core execution (baseline → fees → slippage)
- **Phases 4-5**: Advanced features (order types → multi-asset)
- **Phase 6**: Stress testing and edge cases

---

## Phase 1: Baseline Execution (No Fees, No Slippage)

**Goal**: Verify ml4t.backtest execution matches VectorBT with zero transaction costs

### Test 1.1: Entry Signals Only ✅ PASSING
**File**: `test_1_1_baseline_entries.py`
**Status**: ✅ **PASSING** (as of 2025-11-04)

**Configuration**:
- Entry signals: 20 entries (every 50 bars)
- Exit signals: None (position remains open)
- Fees: 0%
- Slippage: 0%
- Initial cash: $100,000

**Expected Behavior**:
- ml4t.backtest: Reports 1 trade (open position with exit = final bar)
- VectorBT: Reports 1 trade (open position)
- Both should have identical final value and PnL

**Current Result**: ✅ Both engines match ($111,140.65 final value)

**Known Issue**: ml4t.backtest entry_time shows "None" (timestamp recording cosmetic issue)

---

### Test 1.2: Entry/Exit Pairs ✅ PASSING
**File**: `test_1_2_entry_exit_pairs.py`
**Status**: ✅ **PASSING** (as of 2025-11-04)

**Configuration**:
- Entry signals: 20 entries
- Exit signals: 20 exits (10-bar hold period after each entry)
- Fees: 0%
- Slippage: 0%
- Initial cash: $100,000

**Expected Behavior**:
- Both engines should produce identical:
  - Number of trades (19-20 round trips)
  - Final portfolio value
  - Total PnL
  - Entry/exit prices (within 1 cent)
  - Entry/exit times (exact match)

**Current Result**: ✅ Exact match ($132,825.62 final value)

**Critical Fix**: Use `NoCommission()` and `NoSlippage()` instead of `None` to prevent FillSimulator applying default fees (0.1% commission, 0.01% slippage)

---

### Test 1.3: Multiple Round Trips ⏳ PENDING
**File**: `test_1_3_multiple_round_trips.py`
**Status**: ⏳ **NOT YET IMPLEMENTED**

**Configuration**:
- Entry signals: 40 entries
- Exit signals: 40 exits (5-bar hold period)
- Fees: 0%
- Slippage: 0%
- Initial cash: $100,000

**Expected Behavior**:
- Test rapid re-entry (exit bar N, enter bar N+1)
- Verify position tracking handles accumulation
- Validate trade pairing with FIFO matching

**Success Criteria**:
- ✅ All 40 round trips matched
- ✅ Final position = 0
- ✅ Final values within $1
- ✅ No orphaned entries or exits

**Estimated Time**: 30 minutes (copy Test 1.2, adjust signals)

---

## Phase 2: Add Transaction Fees

**Goal**: Validate commission calculation across different fee structures

### Test 2.1: Percentage Commission ⏳ PENDING
**File**: `test_2_1_percentage_commission.py`
**Status**: ⏳ **NOT YET IMPLEMENTED**

**Configuration**:
- Same as Test 1.2 (20 entry/exit pairs)
- Commission: 0.1% (10 basis points)
- Slippage: 0%

**Expected Behavior**:
- Gross PnL should match Test 1.2
- Net PnL reduced by 0.2% per round trip (entry + exit)
- Final value = Test 1.2 value - (0.2% * notional per trade)

**Success Criteria**:
- ✅ Commission amounts match VectorBT
- ✅ Net PnL differs from gross PnL by exactly 0.2%
- ✅ Final values within $5 (rounding tolerance)

**Estimated Time**: 30 minutes

---

### Test 2.2: Fixed + Percentage Fees ⏳ PENDING
**File**: `test_2_2_combined_fees.py`
**Status**: ⏳ **NOT YET IMPLEMENTED**

**Configuration**:
- Same as Test 2.1
- Commission: 0.1% + $2 fixed per trade
- Slippage: 0%

**Expected Behavior**:
- Total commission = (notional * 0.001) + $2
- Verify both components applied correctly

**Success Criteria**:
- ✅ Fixed and percentage components both recorded
- ✅ Total commission matches VectorBT
- ✅ Correct accounting for partial fills

**Estimated Time**: 45 minutes (more complex fee model)

---

### Test 2.3: Asset-Specific Fees ⏳ PENDING
**File**: `test_2_3_asset_specific_fees.py`
**Status**: ⏳ **NOT YET IMPLEMENTED**

**Configuration**:
- Multi-asset test (BTC: 0.1%, ETH: 0.05%)
- Different commission rates per asset
- Slippage: 0%

**Expected Behavior**:
- Commission varies by asset_id
- Verify AssetSpec fee structure used correctly

**Success Criteria**:
- ✅ BTC trades: 0.1% commission
- ✅ ETH trades: 0.05% commission
- ✅ Total commission matches expected

**Estimated Time**: 1 hour (requires multi-asset setup)

---

## Phase 3: Add Slippage

**Goal**: Validate slippage calculation and fill price adjustments

### Test 3.1: Fixed Slippage ⏳ PENDING
**File**: `test_3_1_fixed_slippage.py`
**Status**: ⏳ **NOT YET IMPLEMENTED**

**Configuration**:
- Same as Test 1.2
- Commission: 0%
- Slippage: $10 fixed per trade

**Expected Behavior**:
- BUY fills: price + $10
- SELL fills: price - $10
- Total slippage = $20 per round trip

**Success Criteria**:
- ✅ Fill prices adjusted by $10
- ✅ Slippage amounts recorded correctly
- ✅ Final PnL reduced by $20 * num_trades

**Estimated Time**: 30 minutes

---

### Test 3.2: Percentage Slippage ⏳ PENDING
**File**: `test_3_2_percentage_slippage.py`
**Status**: ⏳ **NOT YET IMPLEMENTED**

**Configuration**:
- Same as Test 1.2
- Commission: 0%
- Slippage: 0.05% (5 basis points)

**Expected Behavior**:
- BUY fills: price * 1.0005
- SELL fills: price * 0.9995
- Total slippage ≈ 0.1% per round trip

**Success Criteria**:
- ✅ Fill prices adjusted by percentage
- ✅ Slippage proportional to price
- ✅ Higher priced assets have higher $ slippage

**Estimated Time**: 30 minutes

---

### Test 3.3: Combined Fees + Slippage ⏳ PENDING
**File**: `test_3_3_combined_costs.py`
**Status**: ⏳ **NOT YET IMPLEMENTED**

**Configuration**:
- Same as Test 1.2
- Commission: 0.1%
- Slippage: 0.05%

**Expected Behavior**:
- Total cost = commission + slippage
- Commission calculated on adjusted (slipped) price
- Total cost ≈ 0.3% per round trip

**Success Criteria**:
- ✅ Both commission and slippage applied
- ✅ Order of operations correct (slippage → commission)
- ✅ Final PnL matches VectorBT

**Estimated Time**: 45 minutes

---

## Phase 4: Order Types

**Goal**: Validate limit, stop, and stop-limit order execution

### Test 4.1: Limit Orders ⏳ PENDING
**File**: `test_4_1_limit_orders.py`
**Status**: ⏳ **NOT YET IMPLEMENTED**

**Configuration**:
- Order type: LIMIT
- Limit prices: 2% below market (BUY), 2% above market (SELL)
- Fees/slippage: 0%

**Expected Behavior**:
- Orders only fill when market reaches limit price
- Fill price = limit price (or better)
- Some orders may not fill (if price doesn't reach limit)

**Success Criteria**:
- ✅ No fills above BUY limit or below SELL limit
- ✅ Unfilled orders tracked correctly
- ✅ Fill timing matches VectorBT

**Estimated Time**: 1 hour (more complex execution logic)

---

### Test 4.2: Stop Orders ⏳ PENDING
**File**: `test_4_2_stop_orders.py`
**Status**: ⏳ **NOT YET IMPLEMENTED**

**Configuration**:
- Order type: STOP (stop-loss)
- Stop prices: 5% below entry (long), 5% above entry (short)
- Fees/slippage: 0%

**Expected Behavior**:
- Orders trigger when market reaches stop price
- Fill at next available price (may be worse than stop)

**Success Criteria**:
- ✅ Stops trigger correctly
- ✅ Slippage past stop price acceptable
- ✅ Position closed when stop hit

**Estimated Time**: 1 hour

---

### Test 4.3: Stop-Limit Orders ⏳ PENDING
**File**: `test_4_3_stop_limit_orders.py`
**Status**: ⏳ **NOT YET IMPLEMENTED**

**Configuration**:
- Order type: STOP_LIMIT
- Stop + limit price bands
- Fees/slippage: 0%

**Expected Behavior**:
- Order activates at stop price
- Fills only within limit price range

**Success Criteria**:
- ✅ Two-stage execution (stop trigger → limit fill)
- ✅ Order may not fill if price gaps past limit
- ✅ Correct handling of fast-moving markets

**Estimated Time**: 1.5 hours (most complex order type)

---

## Phase 5: Advanced Features

**Goal**: Multi-asset portfolios, position sizing, leverage

### Test 5.1: Multi-Asset Portfolio ⏳ PENDING
**File**: `test_5_1_multi_asset.py`
**Status**: ⏳ **NOT YET IMPLEMENTED**

**Configuration**:
- Assets: BTC, ETH, SOL (3 assets)
- Concurrent positions
- Fees/slippage: 0.1% each

**Expected Behavior**:
- Independent position tracking per asset
- Cash allocated across positions
- No cross-contamination

**Success Criteria**:
- ✅ 3 separate position streams
- ✅ Portfolio value = sum(positions) + cash
- ✅ Asset isolation verified

**Estimated Time**: 1.5 hours

---

### Test 5.2: Position Sizing ⏳ PENDING
**File**: `test_5_2_position_sizing.py`
**Status**: ⏳ **NOT YET IMPLEMENTED**

**Configuration**:
- Position sizer: PercentageOfEquity (10% per trade)
- Dynamic sizing as equity changes
- Fees/slippage: 0.1%

**Expected Behavior**:
- First trade: 10% of $100K = $10K notional
- Position size adjusts with equity
- Winners increase position size, losers decrease

**Success Criteria**:
- ✅ Position size varies per trade
- ✅ Respects equity limits
- ✅ Matches VectorBT sizing logic

**Estimated Time**: 1 hour

---

### Test 5.3: Margin Trading ⏳ PENDING
**File**: `test_5_3_margin.py`
**Status**: ⏳ **NOT YET IMPLEMENTED**

**Configuration**:
- Leverage: 2x
- Margin requirements tracked
- Margin calls enforced

**Expected Behavior**:
- Can open positions up to 2x equity
- Margin call if equity falls below threshold
- Forced liquidation if required

**Success Criteria**:
- ✅ 2x buying power
- ✅ Margin calls triggered correctly
- ✅ Liquidation logic matches VectorBT

**Estimated Time**: 2 hours (complex margin accounting)

---

## Phase 6: Stress Testing

**Goal**: Edge cases, high-frequency, large positions

### Test 6.1: High-Frequency Trading ⏳ PENDING
**File**: `test_6_1_high_frequency.py`
**Status**: ⏳ **NOT YET IMPLEMENTED**

**Configuration**:
- 1000+ trades
- Rapid entry/exit (1-2 bar holding periods)
- Fees: 0.1%

**Expected Behavior**:
- Performance: <5 seconds for 1000 trades
- Memory: No leaks
- Accuracy: All trades matched

**Success Criteria**:
- ✅ 1000+ trades complete successfully
- ✅ Execution time <5s
- ✅ Final value within $10 of VectorBT

**Estimated Time**: 1 hour

---

### Test 6.2: Edge Cases ⏳ PENDING
**File**: `test_6_2_edge_cases.py`
**Status**: ⏳ **NOT YET IMPLEMENTED**

**Configuration**:
- Price gaps (10%+ moves)
- Trading halts (missing bars)
- Corporate actions (splits, dividends)

**Expected Behavior**:
- Graceful handling of gaps
- Correct split adjustments
- Dividend accounting

**Success Criteria**:
- ✅ No crashes on edge cases
- ✅ Split adjustments correct
- ✅ Dividend payments recorded

**Estimated Time**: 2 hours (requires special test data)

---

## Test Execution Summary

### Current Status

| Phase | Tests | Passing | Pending | Estimated Time |
|-------|-------|---------|---------|----------------|
| **1. Baseline** | 3 | 2 ✅ | 1 ⏳ | 0.5h remaining |
| **2. Fees** | 3 | 0 | 3 ⏳ | 2.25h |
| **3. Slippage** | 3 | 0 | 3 ⏳ | 1.75h |
| **4. Order Types** | 3 | 0 | 3 ⏳ | 3.5h |
| **5. Advanced** | 3 | 0 | 3 ⏳ | 4.5h |
| **6. Stress** | 2 | 0 | 2 ⏳ | 3h |
| **TOTAL** | **17** | **2** | **15** | **~15.5h** |

### Completion Tracking

**Tests Passing**: 2/17 (12%)
**Current Phase**: Phase 1 - Baseline (67% complete)
**Next Test**: Test 1.3 - Multiple Round Trips

---

## Development Guidelines

### Test Implementation Template

```python
def test_X_Y_description():
    """
    Test X.Y: Description

    Goal: What this test validates
    Expected: What should happen
    """
    # 1. Generate synthetic data
    ohlcv = generate_synthetic_ohlcv(num_bars=1000, seed=42)

    # 2. Generate signals
    entries = generate_entry_signals(...)
    exits = generate_exit_signals(...)

    # 3. Configure engines
    config = BacktestConfig(
        initial_cash=100_000,
        fees=0.001,  # 0.1%
        slippage=0.0005,  # 0.05%
    )

    # 4. Run backtests
    ml4t.backtest_result = ml4t.backtestWrapper().run_backtest(ohlcv, entries, exits, config)
    vbt_result = VectorBTWrapper().run_backtest(ohlcv, entries, exits, config)

    # 5. Compare results
    success, report = compare_results(ml4t.backtest_result, vbt_result, tolerance=...)

    # 6. Print report
    print(report)

    # 7. Assert
    assert success, "Engines produced different results"
```

### Success Criteria

For each test to pass:
1. **Trade Count**: ml4t.backtest matches VectorBT (± 1 trade tolerance)
2. **Final Value**: Within $10 (0.01% of $100K)
3. **PnL**: Within $10
4. **Prices**: Within 1 cent per trade
5. **Timestamps**: Exact match (for completed trades)

### Tolerance Guidelines

| Metric | Baseline | With Fees | With Slippage |
|--------|----------|-----------|---------------|
| Final Value | ± $1 | ± $5 | ± $10 |
| Trade Count | Exact | Exact | ± 1 |
| Entry Price | ± $0.01 | ± $0.01 | ± 2% |
| Exit Price | ± $0.01 | ± $0.01 | ± 2% |
| Commission | N/A | Exact | Exact |
| Slippage | N/A | N/A | ± $0.10 |

---

## Known Issues & Decisions

### Issue 1: FillSimulator Default Fees
**Problem**: FillSimulator applies default 0.1% commission and 0.01% slippage when `None` passed
**Solution**: Use explicit `NoCommission()` and `NoSlippage()` for zero-cost tests
**Affected Tests**: All baseline tests (1.1-1.3)
**Status**: ✅ Fixed in Test 1.2

### Issue 2: Open Position Reporting
**Problem**: TradeTracker only reports completed trades, not open positions
**Solution**: Added `get_open_positions_as_trades()` method to convert open positions to pseudo-trades
**Affected Tests**: Test 1.1 (entries only)
**Status**: ✅ Fixed

### Issue 3: Timestamp Recording (Cosmetic)
**Problem**: ml4t.backtest open position trades show entry_time=None
**Root Cause**: FillEvent.timestamp not properly passed in some scenarios
**Impact**: Low (tests pass, only affects display)
**Status**: ⏳ Deferred (cosmetic issue)

---

## Next Actions

1. **Immediate** (this session):
   - ✅ Fix TradeTracker open position reporting
   - ✅ Verify Tests 1.1 and 1.2 pass
   - ✅ Create this roadmap document

2. **Next Session**:
   - Implement Test 1.3 (multiple round trips)
   - Implement Test 2.1 (percentage commission)
   - Begin systematic execution of roadmap

3. **Long Term**:
   - Complete all 17 tests
   - Document any VectorBT discrepancies
   - Create summary validation report

---

## References

- **Handoff 235838.md**: Test 1.2 fix and TradeTracker analysis
- **Handoff 220125.md**: Original systematic validation plan concept
- **STATUS.md**: Scenario tests 001-005 (different validation approach)
- **User Directive**: "Make testing systematic. Test without fees and slippage first. Use same entry signals for all engines. Test one variable at a time."

---

**Document Version**: 1.0
**Created**: 2025-11-04
**Last Updated**: 2025-11-04
**Next Review**: After Test 1.3 implementation
