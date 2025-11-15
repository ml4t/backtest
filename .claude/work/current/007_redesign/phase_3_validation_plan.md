# Phase 3: Post-Redesign Validation (Hybrid Approach)

**Created**: 2025-11-13T03:00:00-05:00
**Status**: Planning Complete → Ready for Implementation
**Estimated Time**: 5-6 hours
**Dependencies**: Phase 1 & 2 Complete ✅

## Overview

Now that the architectural redesign is complete (Phases 1 & 2), we need to validate that the new architecture works correctly. This phase focuses on:

1. **Immediate Validation**: Fix broken comparison tests to prove redesign works
2. **Systematic Validation**: Implement next 3-4 validation tests (fees/slippage)
3. **Documentation**: Clear handoff for remaining work

## Current Situation

### ✅ What's Working
- **Unit Tests**: 490/490 non-skipped tests passing (100%)
- **Baseline Validation**: 2/17 tests passing (Tests 1.1 & 1.2)
- **Core Architecture**: Clock-driven loop + Portfolio facade complete
- **Coverage**: 80% overall, 97-100% on new architecture

### ❌ What's Broken
- **Comparison Tests**: 4 collection errors (missing modules)
- **Validation Suite**: 15/17 tests not yet implemented
- **Integration Tests**: Import errors from redesign

### 📊 Test Status Breakdown
```
Unit Tests:         490/490 passing (100%) ✅
Validation Tests:     2/17  passing (11.8%) ⚠️
Comparison Tests:     0/4   passing (BROKEN) ❌
Integration Tests:    0/2   passing (BROKEN) ❌
```

## Phase 3 Tasks

### TASK-3.1: Fix Comparison/Integration Tests (Quick Win)
**Goal**: Make comparison tests runnable to validate redesign against VectorBT
**Estimated Time**: 2 hours
**Priority**: Critical

#### Subtasks:
1. **Create stub `spy_order_flow_adapter.py`** (30 min)
   - Location: `src/ml4t.backtest/strategy/spy_order_flow_adapter.py`
   - Reference: `crypto_basis_adapter.py` as template
   - Minimal implementation for tests to import

2. **Fix `portfolio.simple` imports** (30 min)
   - Update `tests/comparison/` to use new Portfolio API
   - Update `tests/integration/` to use new Portfolio API
   - Replace `SimplePortfolio` → `Portfolio`

3. **Verify comparison tests run** (30 min)
   - Run `tests/comparison/test_backtester_validation.py`
   - Run `tests/comparison/test_baseline_evaluation.py`
   - Fix any remaining import errors

4. **Document comparison results** (30 min)
   - Record ml4t.backtest vs VectorBT discrepancies
   - Document any failing comparisons
   - Create baseline for future validation

#### Acceptance Criteria:
- ✅ No collection errors in `tests/comparison/`
- ✅ No collection errors in `tests/integration/`
- ✅ Tests can be imported and run (pass/fail doesn't matter yet)
- ✅ Documented any discrepancies found

#### Deliverables:
- `src/ml4t.backtest/strategy/spy_order_flow_adapter.py` (stub)
- Updated comparison test files
- Comparison results report

---

### TASK-3.2: Implement Test 1.3 (Multiple Round Trips)
**Goal**: Validate ml4t.backtest handles multiple entry/exit cycles correctly
**Estimated Time**: 1 hour
**Priority**: High

#### Implementation:
```python
def test_1_3_multiple_round_trips():
    """
    Test 1.3: Multiple Round Trips (40 trades, 5-bar hold)

    Goal: Verify ml4t.backtest correctly handles multiple entry/exit cycles
    Expected: 40 trades, exact PnL match with VectorBT
    """
    # 1. Generate 1000-bar OHLCV data
    ohlcv = generate_ohlcv(num_bars=1000, seed=42)

    # 2. Generate 40 entry/exit pairs (5-bar hold time)
    entries, exits = generate_entry_exit_pairs(
        num_trades=40,
        hold_bars=5,
        spacing_bars=20  # 20 bars between trades
    )

    # 3. Configure engines (zero fees/slippage)
    config = BacktestConfig(
        initial_cash=100_000,
        fees=NoCommission(),
        slippage=NoSlippage()
    )

    # 4. Run backtests
    ml4t.backtest_result = ml4t.backtestWrapper().run_backtest(ohlcv, entries, exits, config)
    vbt_result = VectorBTWrapper().run_backtest(ohlcv, entries, exits, config)

    # 5. Compare results
    report = print_validation_report({
        'ml4t.backtest': ml4t.backtest_result,
        'VectorBT': vbt_result
    })

    # 6. Assertions
    assert abs(ml4t.backtest_result.num_trades - vbt_result.num_trades) <= 1
    assert abs(ml4t.backtest_result.final_value - vbt_result.final_value) <= 10.0
```

#### Acceptance Criteria:
- ✅ Test file created: `tests/validation/test_1_3_multiple_round_trips.py`
- ✅ Test passes (ml4t.backtest matches VectorBT)
- ✅ Trade count: 40 ± 1
- ✅ Final value: Within $10

---

### TASK-3.3: Implement Test 2.1 (Percentage Commission)
**Goal**: Validate commission calculation with percentage fees
**Estimated Time**: 1 hour
**Priority**: High

#### Implementation:
```python
def test_2_1_percentage_commission():
    """
    Test 2.1: Percentage Commission (0.1%)

    Goal: Validate commission is calculated correctly
    Expected: Entry/exit fees = 0.1% each (0.2% round-trip)
    """
    # Similar structure to Test 1.3, but with:
    config = BacktestConfig(
        initial_cash=100_000,
        fees=PercentageCommission(rate=0.001),  # 0.1%
        slippage=NoSlippage()
    )

    # Assertions include commission checks:
    assert ml4t.backtest_result.total_commission > 0
    assert abs(ml4t.backtest_result.total_commission - vbt_result.total_commission) <= 1.0
```

#### Acceptance Criteria:
- ✅ Test file created: `tests/validation/test_2_1_percentage_commission.py`
- ✅ Test passes (commission matches VectorBT)
- ✅ Total commission > 0
- ✅ Commission match: Within $1

---

### TASK-3.4: Implement Test 2.2 (Combined Fees)
**Goal**: Validate combined percentage + fixed fees
**Estimated Time**: 1 hour
**Priority**: High

#### Implementation:
```python
def test_2_2_combined_fees():
    """
    Test 2.2: Combined Fees (0.1% + $2 per trade)

    Goal: Validate both percentage and fixed fees
    Expected: Each trade pays 0.1% + $2
    """
    config = BacktestConfig(
        initial_cash=100_000,
        fees=CombinedCommission(
            percentage=0.001,  # 0.1%
            fixed=2.0          # $2 per trade
        ),
        slippage=NoSlippage()
    )
```

#### Acceptance Criteria:
- ✅ Test file created: `tests/validation/test_2_2_combined_fees.py`
- ✅ Test passes (fees match VectorBT)
- ✅ Commission calculation correct

---

### TASK-3.5: Implement Test 3.1 (Fixed Slippage)
**Goal**: Validate slippage reduces fill prices correctly
**Estimated Time**: 1 hour
**Priority**: High

#### Implementation:
```python
def test_3_1_fixed_slippage():
    """
    Test 3.1: Fixed Slippage ($10 per trade)

    Goal: Validate slippage reduces entry prices (buys pay more)
    Expected: Each entry costs $10 more, each exit receives $10 less
    """
    config = BacktestConfig(
        initial_cash=100_000,
        fees=NoCommission(),
        slippage=FixedSlippage(amount=10.0)
    )

    # Verify fill prices differ from market by $10
    for trade in ml4t.backtest_result.trades:
        assert abs(trade.slippage) >= 9.0  # Allow 10% tolerance
```

#### Acceptance Criteria:
- ✅ Test file created: `tests/validation/test_3_1_fixed_slippage.py`
- ✅ Test passes (slippage matches VectorBT)
- ✅ Slippage amount correct per trade

---

### TASK-3.6: Documentation & Handoff
**Goal**: Document progress and remaining work
**Estimated Time**: 30 minutes
**Priority**: Medium

#### Deliverables:
1. **Progress Report**: Update `006_systematic_baseline_validation/state.json`
2. **Validation Status**: Update `VALIDATION_ROADMAP.md`
3. **Next Steps**: Document remaining 12 tests
4. **Known Issues**: Document any VectorBT discrepancies

#### Contents:
```markdown
## Phase 3 Results (Session 2025-11-13)

### Completed:
- ✅ Comparison tests fixed (4 collection errors resolved)
- ✅ Test 1.3: Multiple round trips (40 trades)
- ✅ Test 2.1: Percentage commission (0.1%)
- ✅ Test 2.2: Combined fees (0.1% + $2)
- ✅ Test 3.1: Fixed slippage ($10)

### Status:
- Validation Tests: 6/17 passing (35.3%) ← up from 11.8%
- Comparison Tests: 4/4 runnable ← fixed from BROKEN

### Remaining Work (10 hours):
- Test 2.3: Asset-specific fees (1 hour)
- Test 3.2: Percentage slippage (1 hour)
- Test 3.3: Combined costs (1 hour)
- Test 4.1-4.3: Order types (3 hours)
- Test 5.1-5.3: Advanced features (3 hours)
- Test 6.1-6.2: Stress testing (2 hours)
```

---

## Success Criteria

### For TASK-3.1 (Comparison Tests):
- ✅ No import/collection errors
- ✅ Tests run (even if they fail)
- ✅ Documented discrepancies

### For TASK-3.2 through 3.5 (Validation Tests):
- ✅ Each test passes independently
- ✅ Trade counts match VectorBT (± 1)
- ✅ Final values match (± $10)
- ✅ Fees/slippage calculated correctly

### For TASK-3.6 (Documentation):
- ✅ Progress tracked in state.json
- ✅ VALIDATION_ROADMAP.md updated
- ✅ Clear handoff for next session

---

## Timeline

**Session 1 (Current)**: ~6 hours
- TASK-3.1: Comparison tests (2 hours)
- TASK-3.2: Test 1.3 (1 hour)
- TASK-3.3: Test 2.1 (1 hour)
- TASK-3.4: Test 2.2 (1 hour)
- TASK-3.5: Test 3.1 (1 hour)
- TASK-3.6: Documentation (30 min)

**Session 2 (Future)**: ~4 hours
- Tests 2.3, 3.2, 3.3 (3 hours)
- Documentation (1 hour)

**Session 3 (Future)**: ~6 hours
- Tests 4.1-4.3 (order types)

**Session 4 (Future)**: ~5 hours
- Tests 5.1-5.3 (advanced features)
- Tests 6.1-6.2 (stress testing)

---

## Dependencies

### External:
- VectorBT Pro (for comparison tests)
- pytest
- pandas, numpy

### Internal:
- `tests/validation/common/` infrastructure
- `src/ml4t.backtest/` (post-redesign)
- New Portfolio API

---

## Risks & Mitigation

### Risk 1: VectorBT Not Installed
**Impact**: Comparison tests can't run
**Mitigation**: Document expected behavior, skip tests with clear TODO

### Risk 2: VectorBT API Differences
**Impact**: Tests may fail even if ml4t.backtest is correct
**Mitigation**: Document discrepancies, consider acceptable tolerances

### Risk 3: Slippage/Commission Models Don't Match
**Impact**: Tests fail due to model differences
**Mitigation**: Adjust tolerance levels, document assumptions

---

## References

- Work Unit 006: `006_systematic_baseline_validation/requirements.md`
- Work Unit 007: Current redesign state
- Validation Roadmap: `tests/validation/VALIDATION_ROADMAP.md`
- Portfolio Redesign: `007_redesign/enhanced_facade_design.md`
