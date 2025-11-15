# Requirements: Systematic Baseline Validation

## Source
- **Type**: Document
- **Reference**: `tests/validation/VALIDATION_ROADMAP.md`
- **Date**: 2025-11-04T23:00:00Z
- **Work Unit**: 006

## Overview

Implement a systematic 17-test validation suite comparing ml4t.backtest execution against VectorBT Pro. Tests progress incrementally from baseline (no fees/slippage) through fees, slippage, order types, advanced features, and stress testing.

**Current Status**: 2/17 tests passing (Tests 1.1 and 1.2)
**Remaining**: 15 tests to implement
**Estimated Time**: ~15.5 hours

## Testing Philosophy

1. **Baseline First**: Validate core execution with zero transaction costs
2. **One Variable at a Time**: Add complexity incrementally (fees → slippage → order types)
3. **Systematic Progression**: Simple → Complex
4. **Exact Matching**: Use synthetic data for reproducible, exact comparisons
5. **Document Differences**: Explain any discrepancies between engines

## Test Structure (6 Phases, 17 Tests)

### Phase 1: Baseline Execution (No Fees, No Slippage)
**Goal**: Verify ml4t.backtest matches VectorBT with zero transaction costs

- ✅ **Test 1.1**: Entry signals only (PASSING)
- ✅ **Test 1.2**: Entry/exit pairs (PASSING)
- ⏳ **Test 1.3**: Multiple round trips (40 trades, 5-bar hold) - **NEXT**

### Phase 2: Transaction Fees
**Goal**: Validate commission calculation across different fee structures

- ⏳ **Test 2.1**: Percentage commission (0.1%)
- ⏳ **Test 2.2**: Fixed + percentage fees (0.1% + $2)
- ⏳ **Test 2.3**: Asset-specific fees (BTC: 0.1%, ETH: 0.05%)

### Phase 3: Slippage
**Goal**: Validate slippage calculation and fill price adjustments

- ⏳ **Test 3.1**: Fixed slippage ($10 per trade)
- ⏳ **Test 3.2**: Percentage slippage (0.05%)
- ⏳ **Test 3.3**: Combined fees + slippage (0.1% + 0.05%)

### Phase 4: Order Types
**Goal**: Validate limit, stop, and stop-limit execution

- ⏳ **Test 4.1**: Limit orders (2% from market)
- ⏳ **Test 4.2**: Stop orders (5% stops)
- ⏳ **Test 4.3**: Stop-limit orders (combined stop + limit)

### Phase 5: Advanced Features
**Goal**: Multi-asset portfolios, position sizing, leverage

- ⏳ **Test 5.1**: Multi-asset portfolio (BTC, ETH, SOL)
- ⏳ **Test 5.2**: Position sizing (10% of equity per trade)
- ⏳ **Test 5.3**: Margin trading (2x leverage)

### Phase 6: Stress Testing
**Goal**: Edge cases, high-frequency, large positions

- ⏳ **Test 6.1**: High-frequency (1000+ trades)
- ⏳ **Test 6.2**: Edge cases (gaps, halts, corporate actions)

## Success Criteria

For each test to pass:
1. **Trade Count**: ml4t.backtest matches VectorBT (± 1 trade tolerance)
2. **Final Value**: Within $10 (0.01% of $100K)
3. **PnL**: Within $10
4. **Prices**: Within 1 cent per trade
5. **Timestamps**: Exact match (for completed trades)

## Tolerance Guidelines

| Metric | Baseline | With Fees | With Slippage |
|--------|----------|-----------|---------------|
| Final Value | ± $1 | ± $5 | ± $10 |
| Trade Count | Exact | Exact | ± 1 |
| Entry Price | ± $0.01 | ± $0.01 | ± 2% |
| Exit Price | ± $0.01 | ± $0.01 | ± 2% |
| Commission | N/A | Exact | Exact |
| Slippage | N/A | N/A | ± $0.10 |

## Test Implementation Template

```python
def test_X_Y_description():
    """
    Test X.Y: Description

    Goal: What this test validates
    Expected: What should happen
    """
    # 1. Generate synthetic data
    ohlcv = generate_ohlcv(num_bars=1000, seed=42)

    # 2. Generate signals
    entries, exits = generate_entry_exit_pairs(...)

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
    success = print_validation_report({'ml4t.backtest': ml4t.backtest_result, 'VectorBT': vbt_result})

    # 6. Assert
    assert success, "Engines produced different results"
```

## Known Issues (Already Fixed)

### Issue 1: FillSimulator Default Fees
- **Problem**: FillSimulator applies default 0.1% commission and 0.01% slippage when `None` passed
- **Solution**: Use explicit `NoCommission()` and `NoSlippage()` for zero-cost tests
- **Status**: ✅ Fixed in Test 1.2

### Issue 2: Open Position Reporting
- **Problem**: TradeTracker only reports completed trades, not open positions
- **Solution**: Added `get_open_positions_as_trades()` method
- **Status**: ✅ Fixed - Test 1.1 now passes

### Issue 3: Timestamp Recording (Cosmetic)
- **Problem**: ml4t.backtest open position trades show entry_time=None
- **Impact**: Low (tests pass, only affects display)
- **Status**: ⏳ Deferred (cosmetic issue)

## Dependencies

### External
- VectorBT Pro (for comparison)
- pytest (test framework)
- pandas, numpy (data manipulation)

### Internal
- `tests/validation/common/` - Test infrastructure
  - `data_generator.py` - Synthetic OHLCV generation
  - `signal_generator.py` - Signal generation utilities
  - `engine_wrappers.py` - ml4t.backtest and VectorBT wrappers
  - `comparison.py` - Result comparison tools

## Out of Scope

- Multi-platform validation (Backtrader, Zipline) - covered by work unit 005
- Real market data testing - covered by scenario tests
- Performance benchmarking - covered by Test 6.1
- Production deployment - covered by separate work unit

## Risks and Assumptions

### Risks
1. **VectorBT API Changes**: VectorBT Pro API may change between versions
2. **Order Type Complexity**: Tests 4.1-4.3 may require significant debugging
3. **Margin Accounting**: Test 5.3 may expose complex edge cases

### Assumptions
1. VectorBT Pro is installed and functional
2. Synthetic data accurately represents real market behavior
3. Zero-cost baseline tests provide meaningful validation
4. Test infrastructure (wrappers, comparison tools) is robust

### Mitigation
1. Pin VectorBT version in requirements
2. Allocate extra time for order type tests (1-1.5 hours each)
3. Reference VectorBT documentation for margin logic
4. Validate test infrastructure before proceeding

## Deliverables

1. **15 Test Files**:
   - `test_1_3_multiple_round_trips.py`
   - `test_2_1_percentage_commission.py`
   - `test_2_2_combined_fees.py`
   - `test_2_3_asset_specific_fees.py`
   - `test_3_1_fixed_slippage.py`
   - `test_3_2_percentage_slippage.py`
   - `test_3_3_combined_costs.py`
   - `test_4_1_limit_orders.py`
   - `test_4_2_stop_orders.py`
   - `test_4_3_stop_limit_orders.py`
   - `test_5_1_multi_asset.py`
   - `test_5_2_position_sizing.py`
   - `test_5_3_margin.py`
   - `test_6_1_high_frequency.py`
   - `test_6_2_edge_cases.py`

2. **Updated Documentation**:
   - `VALIDATION_ROADMAP.md` (status updates)
   - Individual test docstrings
   - Summary report of all 17 tests

3. **Test Results**:
   - All 17 tests passing
   - Documented discrepancies (if any)
   - Performance metrics for Test 6.1

## References

- **Primary**: `tests/validation/VALIDATION_ROADMAP.md`
- **Handoff**: `.claude/transitions/2025-11-04/235838.md`
- **Test Infrastructure**: `tests/validation/test_1_2_entry_exit_pairs.py` (template)
- **User Directive**: "Make testing systematic. Test without fees and slippage first. Use same entry signals for all engines. Test one variable at a time."
