# Exploration Summary: Systematic Baseline Validation

**Work Unit**: 006
**Date**: 2025-11-04
**Status**: Ready for implementation

---

## Codebase Analysis

### Test Infrastructure (Robust and Ready)

**Location**: `tests/validation/common/`

1. **Data Generation** (`data_generator.py`):
   - `generate_ohlcv()`: Creates synthetic OHLCV data with realistic properties
   - Guarantees OHLC constraints (High >= max(O,C), Low <= min(O,C))
   - Ensures continuity (Close[i] = Open[i+1])
   - Fixed seed for reproducibility
   - Validates generated data

2. **Signal Generation** (`signal_generator.py`):
   - `generate_fixed_entries()`: Regular entry signals
   - `generate_entry_exit_pairs()`: Paired entry/exit with fixed hold period
   - `generate_random_signals()`: Random signals with configurable probability
   - `validate_signals()`: Ensures signal consistency

3. **Engine Wrappers** (`engine_wrappers.py`):
   - `ml4t.backtestWrapper`: Wraps ml4t.backtest with standardized interface
   - `VectorBTWrapper`: Wraps VectorBT Pro with matching interface
   - `BacktestConfig`: Configuration dataclass
   - `BacktestResult`: Standardized result format
   - **Key fix applied**: Uses `NoCommission()` and `NoSlippage()` for zero-cost tests

4. **Comparison Tools** (`comparison.py`):
   - `compare_trades()`: Side-by-side engine comparison
   - `assert_identical()`: Validates exact matching with tolerances
   - `compare_first_trades()`: Detailed trade-by-trade comparison
   - `print_validation_report()`: Comprehensive validation output

### Current Test Status

**Passing Tests** (2/17):
- ✅ **Test 1.1** (`test_1_1_baseline_entries.py`): Entry signals only
  - Both engines: 1 trade (open position)
  - Final value: $111,140.65 (exact match)
  - Status: PASSING

- ✅ **Test 1.2** (`test_1_2_entry_exit_pairs.py`): Entry/exit pairs
  - Both engines: 20 trades
  - Final value: $132,825.62 (exact match)
  - Status: PASSING

**Pending Tests** (15/17):
- All 15 remaining tests are well-specified in VALIDATION_ROADMAP.md
- Clear acceptance criteria for each test
- Estimated time per test ranges from 0.5 to 2 hours

### Key Technical Insights

1. **TradeTracker Enhancement**:
   - Added `get_open_positions_as_trades()` method (line 304-362)
   - Converts open positions to pseudo-trades for comparison
   - Matches VectorBT behavior of reporting open positions

2. **FillSimulator Behavior**:
   - Applies default 0.1% commission and 0.01% slippage when `None` passed
   - **Solution**: Must use explicit `NoCommission()` and `NoSlippage()`
   - Affects all baseline tests (Phase 1)

3. **Execution Delay**:
   - `execution_delay=False` for VectorBT comparison (same-bar fills)
   - `execution_delay=True` for realistic next-bar execution
   - Baseline tests use `False` for exact matching

---

## Implementation Approach

### Phase-by-Phase Strategy

**Phase 1: Baseline (0.5h remaining)**
- Test 1.3: Copy test_1_2, increase to 40 trades with 5-bar hold
- Validates rapid re-entry and FIFO trade pairing

**Phase 2: Fees (2.25h)**
- Test 2.1: Add percentage commission (simplest)
- Test 2.2: Add fixed + percentage fees (more complex)
- Test 2.3: Multi-asset with asset-specific fees (requires multi-asset setup)

**Phase 3: Slippage (1.75h)**
- Test 3.1: Fixed slippage ($10 per trade)
- Test 3.2: Percentage slippage (0.05%)
- Test 3.3: Combined fees + slippage (integration test)

**Phase 4: Order Types (3.5h)**
- Test 4.1: Limit orders (more complex execution logic)
- Test 4.2: Stop orders (trigger logic)
- Test 4.3: Stop-limit orders (two-stage execution)
- **Note**: These may require more time if execution logic differs

**Phase 5: Advanced (4.5h)**
- Test 5.1: Multi-asset portfolio (3 concurrent positions)
- Test 5.2: Position sizing (dynamic sizing logic)
- Test 5.3: Margin trading (complex margin accounting)
- **Note**: Most complex tests, may expose edge cases

**Phase 6: Stress (3h)**
- Test 6.1: High-frequency (1000+ trades, performance test)
- Test 6.2: Edge cases (gaps, halts, corporate actions)
- **Note**: Requires special test data generation

### Test Template (Standardized)

All tests follow this structure (from test_1_2):

```python
def test_X_Y_description():
    # 1. Generate synthetic data
    ohlcv = generate_ohlcv(n_bars=1000, seed=42)

    # 2. Generate signals
    entries, exits = generate_entry_exit_pairs(...)

    # 3. Configure engines
    config = BacktestConfig(...)

    # 4. Run backtests
    ml4t.backtest_result = ml4t.backtestWrapper().run_backtest(ohlcv, entries, exits, config)
    vbt_result = VectorBTWrapper().run_backtest(ohlcv, entries, exits, config)

    # 5. Compare and assert
    success = print_validation_report(...)
    assert success
```

**Implementation Pattern**:
1. Copy test_1_2 as template
2. Modify configuration (fees, slippage, order type)
3. Adjust signals if needed (for order type tests)
4. Update docstring with test-specific details
5. Run test and verify passing
6. Update VALIDATION_ROADMAP.md status

### Key Technologies

- **Python 3.9+**: Core language
- **pytest**: Test framework
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **VectorBT Pro**: Reference engine
- **ml4t.backtest**: Engine under test

### Development Workflow

1. **Sequential Execution**: Implement tests in order (1.3 → 2.1 → 2.2 → ...)
2. **Incremental Validation**: Each test builds on previous tests
3. **Document Failures**: If a test fails, document why before moving forward
4. **Update Roadmap**: Mark tests as passing in VALIDATION_ROADMAP.md
5. **Commit Frequently**: Commit after each passing test

---

## Codebase Integration Points

### Files to Modify

1. **New Test Files** (15 files):
   - `tests/validation/test_1_3_multiple_round_trips.py`
   - `tests/validation/test_2_1_percentage_commission.py`
   - ... (remaining 13 test files)

2. **Updated Documentation**:
   - `tests/validation/VALIDATION_ROADMAP.md` (status updates)

3. **Potential Infrastructure Updates**:
   - May need new signal generators for order type tests
   - May need multi-asset data generator for Phase 2-3
   - May need special data generator for Test 6.2 (gaps, halts)

### Files to NOT Modify

- `tests/validation/common/*.py` - Infrastructure is stable
- `src/ml4t.backtest/` - No core engine changes expected
- `tests/validation/test_1_1*.py` and `test_1_2*.py` - Already passing

---

## Risk Assessment

### Low Risk (Phases 1-3)
- Tests 1.3, 2.1-2.3, 3.1-3.3
- Simple configuration changes
- Well-understood execution model
- Estimated time accurate

### Medium Risk (Phase 4)
- Tests 4.1-4.3 (Order types)
- More complex execution logic
- May require debugging VectorBT vs ml4t.backtest order type differences
- May exceed estimated time

### High Risk (Phases 5-6)
- Tests 5.1-5.3, 6.1-6.2
- Complex features (margin, multi-asset, edge cases)
- May expose ml4t.backtest limitations or bugs
- May require infrastructure enhancements
- Time estimates uncertain

### Mitigation Strategies

1. **Start Simple**: Complete Phases 1-3 before tackling complex tests
2. **Early Warning**: If a test fails unexpectedly, pause and investigate
3. **Document Differences**: If ml4t.backtest and VectorBT differ, document why
4. **Flexible Timeline**: Phase 5-6 may take longer than estimated
5. **Skip if Blocked**: If a test is blocked, skip and revisit later

---

## Next Steps

### Immediate Actions
1. ✅ Create work unit 006
2. ✅ Document requirements and exploration
3. ✅ Create comprehensive task breakdown (state.json)
4. ➡️ Run `/workflow:plan` to generate implementation plan (OPTIONAL - tasks already clear)
5. ➡️ Run `/workflow:next` to implement Test 1.3

### Short-Term (Next 3-5 Tests)
1. Complete Phase 1 (Test 1.3)
2. Complete Phase 2 (Tests 2.1-2.3)
3. Begin Phase 3 (Tests 3.1-3.3)

### Long-Term (All 15 Tests)
1. Complete all 6 phases
2. Document any VectorBT discrepancies
3. Create summary validation report
4. Update VALIDATION_ROADMAP.md as complete
5. Run `/workflow:ship` to finalize work unit

---

## Success Metrics

**Completion Criteria**:
- ✅ All 17 tests passing (currently 2/17)
- ✅ VALIDATION_ROADMAP.md updated with status
- ✅ All discrepancies documented
- ✅ Performance metrics for Test 6.1 recorded
- ✅ Summary report created

**Quality Metrics**:
- All tests follow standardized template
- Test docstrings are comprehensive
- Acceptance criteria clearly validated
- Code is clean and maintainable
- No skipped or xfailed tests

**Time Metrics**:
- Target: 15.5 hours for 15 tests
- Actual: Track per-test time
- Efficiency: Compare actual vs estimated

---

## Exploration Complete

**Assessment**: ✅ Ready for Implementation

The test infrastructure is robust, the roadmap is comprehensive, and the implementation approach is clear. All 15 pending tests have well-defined specifications, acceptance criteria, and time estimates.

**Recommended Next Action**: Run `/workflow:next` to implement Test 1.3 (Multiple Round Trips)
