# Phase 2: VectorBT Exact Matching - Implementation Plan

**Work Unit**: `20251026_002_vectorbt_exact_matching`
**Created**: 2025-10-26
**Objective**: Reverse engineer VectorBT Pro and achieve 100% exact matching with qengine

---

## Project Overview

### Goal
Achieve complete understanding of VectorBT Pro's backtesting calculations and replicate them exactly in qengine. No approximations - **100% exact matching required**.

### Scope
- ✅ **In Scope**: TP/SL/TSL logic, fee/slippage calculations, position sizing, entry/exit prices
- ❌ **Out of Scope**: Signal generation (using Phase 1 indicators), ML strategies, optimization

### Success Criteria
- 100% trade count match (352 trades on Q1 2024 test period)
- 100% entry/exit price match (within tick precision)
- 100% fee/slippage calculation match
- 100% PnL match (within rounding precision)
- Complete VectorBT behavior documentation

### Timeline Estimate
- **Total**: ~40-60 hours (1-2 weeks)
- **Phase 1** (Investigation): 16-20 hours
- **Phase 2** (Test Setup): 6-8 hours
- **Phase 3** (Implementation): 12-16 hours
- **Phase 4** (Validation): 10-14 hours
- **Phase 5** (Documentation): 6-8 hours

---

## Technical Architecture

### VectorBT Pro (Reference)
```
Portfolio.from_signals(
    entries=signals,
    tp_stop=0.025,    # 2.5% take profit
    tsl_stop=0.01,    # 1% trailing stop loss
    fees=0.0002,      # 0.02% fees
    slippage=0.0002,  # 0.02% slippage
    size=np.inf       # Full capital allocation
)
```

**Black Box Behavior to Reverse Engineer**:
- When are exits checked? (every bar, intra-bar, close only?)
- What prices are used? (open, close, high, low?)
- How is TSL tracked? (highest close, highest high?)
- Exit priority? (TP vs TSL same bar?)
- Fee timing? (entry, exit, both?)
- Position sizing? (how does size=np.inf work?)

### qengine (ML4T - Target)
```
Event-Driven Architecture:
├── Strategy (signal generation + exit logic)
├── DataFeed (market events)
├── Broker (order execution)
├── Portfolio (position tracking)
└── Risk Manager (optional)
```

**Must Implement**:
- Manual TP/SL/TSL checking in `on_market_event()`
- Identical price calculations
- Identical fee/slippage application
- Identical position sizing
- Point-in-time correctness

---

## Task Breakdown

### Phase 1: VectorBT Reverse Engineering (Investigation)

#### TASK-001: Set up VectorBT Pro investigation environment
**Type**: foundation
**Priority**: high
**Estimated**: 2 hours
**Dependencies**: []

**Description**: Create investigation environment for VectorBT Pro analysis

**Acceptance Criteria**:
- VectorBT Pro installed and importable
- Simple test backtest runs successfully
- Can inspect Portfolio object attributes
- Can access trade log and order records
- Jupyter notebook set up for exploration

**Files**:
- `projects/crypto_futures/notebooks/vectorbt_investigation.ipynb`

---

#### TASK-002: Analyze VectorBT source code for exit logic
**Type**: investigation
**Priority**: high
**Estimated**: 3 hours
**Dependencies**: [TASK-001]

**Description**: Read VectorBT Pro source code to understand TP/SL/TSL implementation

**Acceptance Criteria**:
- Located relevant source code files
- Documented TP/SL/TSL classes and methods
- Identified key parameters and logic flow
- Documented any proprietary or complex algorithms
- Created source code reference notes

**Files**:
- `projects/crypto_futures/docs/vectorbt_source_analysis.md`

---

#### TASK-003: Document entry price calculation mechanism
**Type**: investigation
**Priority**: high
**Estimated**: 2 hours
**Dependencies**: [TASK-001, TASK-002]

**Description**: Determine exactly how VectorBT calculates entry prices

**Investigation Questions**:
- Is entry at open of next bar after signal?
- Is entry at close of signal bar?
- How is slippage applied to entry?
- How are fees applied to entry?

**Acceptance Criteria**:
- Entry price formula documented
- Slippage application documented
- Fee application documented
- Verified with test cases (3+ examples)

**Files**:
- `projects/crypto_futures/docs/vectorbt_entry_logic.md`

---

#### TASK-004: Document exit price calculation mechanism
**Type**: investigation
**Priority**: high
**Estimated**: 2 hours
**Dependencies**: [TASK-001, TASK-002]

**Description**: Determine exactly how VectorBT calculates exit prices

**Investigation Questions**:
- Is exit at close of exit bar?
- Can exits trigger intra-bar (using high/low)?
- How is slippage applied to exit?
- How are fees applied to exit?

**Acceptance Criteria**:
- Exit price formula documented
- Intra-bar behavior documented (if applicable)
- Slippage application documented
- Fee application documented
- Verified with test cases (3+ examples)

**Files**:
- `projects/crypto_futures/docs/vectorbt_exit_logic.md`

---

#### TASK-005: Document TP (take profit) triggering logic
**Type**: investigation
**Priority**: high
**Estimated**: 2 hours
**Dependencies**: [TASK-001, TASK-002, TASK-004]

**Description**: Document exactly when and how TP exits trigger

**Investigation Questions**:
- TP based on close price or high?
- TP = entry_price * (1 + tp_pct)?
- Checked every bar or only when conditions met?
- Triggered immediately or next bar?

**Acceptance Criteria**:
- TP trigger condition formula documented
- TP price calculation documented
- Timing of TP check documented
- Verified with test cases showing TP exits

**Files**:
- `projects/crypto_futures/docs/vectorbt_tp_logic.md`

---

#### TASK-006: Document SL (stop loss) triggering logic
**Type**: investigation
**Priority**: high
**Estimated**: 1.5 hours
**Dependencies**: [TASK-001, TASK-002, TASK-004]

**Description**: Document exactly when and how SL exits trigger

**Investigation Questions**:
- SL based on close price or low?
- SL = entry_price * (1 - sl_pct)?
- Checked every bar?
- Fixed SL or can be modified?

**Acceptance Criteria**:
- SL trigger condition formula documented
- SL price calculation documented
- Timing of SL check documented
- Verified with test cases showing SL exits

**Files**:
- `projects/crypto_futures/docs/vectorbt_sl_logic.md`

---

#### TASK-007: Document TSL (trailing stop loss) tracking and triggering
**Type**: investigation
**Priority**: high
**Estimated**: 3 hours
**Dependencies**: [TASK-001, TASK-002, TASK-004]

**Description**: Document exactly how TSL is tracked and triggered

**Investigation Questions**:
- Trailing from highest close or highest high?
- TSL = highest * (1 - tsl_pct)?
- Updated every bar?
- TSL can only tighten, not loosen?
- Initial TSL position?

**Acceptance Criteria**:
- TSL tracking mechanism documented
- TSL trigger condition documented
- TSL update frequency documented
- Initial TSL calculation documented
- Verified with test cases showing TSL exits

**Files**:
- `projects/crypto_futures/docs/vectorbt_tsl_logic.md`

---

#### TASK-008: Document exit priority rules (same-bar conflicts)
**Type**: investigation
**Priority**: medium
**Estimated**: 1.5 hours
**Dependencies**: [TASK-005, TASK-006, TASK-007]

**Description**: Determine what happens when multiple exit conditions trigger same bar

**Investigation Questions**:
- TP and TSL both trigger - which executes?
- TP and SL both trigger - which executes?
- Is there a priority order?
- Or is it based on price/timing?

**Acceptance Criteria**:
- Exit priority rules documented
- Test cases for simultaneous triggers
- Behavior verified empirically

**Files**:
- `projects/crypto_futures/docs/vectorbt_exit_priority.md`

---

#### TASK-009: Document fee calculation and application
**Type**: investigation
**Priority**: high
**Estimated**: 1.5 hours
**Dependencies**: [TASK-003, TASK-004]

**Description**: Document exactly how fees are calculated and applied

**Investigation Questions**:
- Fees on entry, exit, or both?
- Percentage of trade value?
- Applied to gross or net?
- Affect position size?

**Acceptance Criteria**:
- Fee formula documented
- Fee application points documented
- Impact on PnL calculation documented
- Verified with test trades

**Files**:
- `projects/crypto_futures/docs/vectorbt_fees.md`

---

#### TASK-010: Document slippage calculation and application
**Type**: investigation
**Priority**: high
**Estimated**: 1.5 hours
**Dependencies**: [TASK-003, TASK-004]

**Description**: Document exactly how slippage is calculated and applied

**Investigation Questions**:
- Slippage on entry, exit, or both?
- Percentage of price?
- Directional (always worse)?
- Applied before or after fees?

**Acceptance Criteria**:
- Slippage formula documented
- Slippage application points documented
- Direction (buy vs sell) documented
- Verified with test trades

**Files**:
- `projects/crypto_futures/docs/vectorbt_slippage.md`

---

#### TASK-011: Document position sizing with size=np.inf
**Type**: investigation
**Priority**: high
**Estimated**: 2 hours
**Dependencies**: [TASK-003, TASK-009, TASK-010]

**Description**: Understand how VectorBT handles size=np.inf (full capital allocation)

**Investigation Questions**:
- How much of available cash is used?
- Are fees/slippage deducted from position size?
- How is quantity calculated?
- Rounding behavior?
- Concurrent positions?

**Acceptance Criteria**:
- Position sizing formula documented
- Cash allocation rules documented
- Fee/slippage impact on sizing documented
- Quantity calculation documented
- Verified with test trades

**Files**:
- `projects/crypto_futures/docs/vectorbt_position_sizing.md`

---

### Phase 2: Test Infrastructure (Setup)

#### TASK-012: Extract complete VectorBT trade log with all details
**Type**: foundation
**Priority**: high
**Estimated**: 2 hours
**Dependencies**: [TASK-001]

**Description**: Extract all trade details from VectorBT Portfolio object

**Acceptance Criteria**:
- Trade log includes: entry time, entry price, exit time, exit price
- Includes: entry fees, exit fees, entry slippage, exit slippage
- Includes: exit reason (TP, SL, TSL, signal)
- Includes: position size, PnL, duration
- Saved as structured data (CSV/Parquet)
- All 352 trades from test period extracted

**Files**:
- `projects/crypto_futures/scripts/extract_vectorbt_trades.py`
- `projects/crypto_futures/data/vectorbt_trades_q1_2024.parquet`

---

#### TASK-013: Create trade-by-trade comparison framework
**Type**: foundation
**Priority**: high
**Estimated**: 3 hours
**Dependencies**: [TASK-012]

**Description**: Build framework to compare VectorBT and qengine trades

**Acceptance Criteria**:
- Loads VectorBT trade log
- Loads qengine trade log
- Matches trades by index/timestamp
- Compares all fields (entry/exit prices, fees, PnL, etc.)
- Reports differences with clear formatting
- Identifies missing/extra trades
- Generates summary statistics

**Files**:
- `projects/crypto_futures/scripts/compare_trades.py`
- `projects/crypto_futures/tests/test_trade_matching.py`

---

#### TASK-014: Identify and document edge cases
**Type**: investigation
**Priority**: medium
**Estimated**: 2 hours
**Dependencies**: [TASK-012]

**Description**: Identify edge cases in VectorBT trade log for testing

**Edge Cases to Find**:
- Trades lasting 1 bar (immediate TP/TSL)
- Trades lasting 100+ bars (long holds)
- Same-bar entry and exit
- TP and TSL both close to trigger
- Multiple entries in short period
- Extreme price movements

**Acceptance Criteria**:
- 10+ edge cases identified
- Edge case characteristics documented
- Test data extracted for each case
- Expected behavior documented

**Files**:
- `projects/crypto_futures/docs/edge_cases.md`

---

#### TASK-015: Create verification test suite structure
**Type**: testing
**Priority**: medium
**Estimated**: 1.5 hours
**Dependencies**: [TASK-013, TASK-014]

**Description**: Set up pytest structure for verification tests

**Acceptance Criteria**:
- pytest configuration in place
- Test fixtures for VectorBT trades
- Test fixtures for qengine setup
- Parameterized tests for edge cases
- Clear test organization (unit vs integration)

**Files**:
- `projects/crypto_futures/tests/test_vectorbt_matching.py`
- `projects/crypto_futures/tests/conftest.py`

---

### Phase 3: qengine Implementation (Build)

#### TASK-016: Implement TP exit logic in qengine
**Type**: feature
**Priority**: high
**Estimated**: 2 hours
**Dependencies**: [TASK-005]

**Description**: Implement TP exit logic exactly matching VectorBT

**Acceptance Criteria**:
- TP trigger condition matches VectorBT formula
- TP price calculation matches VectorBT
- TP timing matches VectorBT
- Code documented with VectorBT reference
- Unit tests pass

**Files**:
- `backtest/src/qengine/strategies/vectorbt_strategy.py`
- `backtest/tests/test_tp_logic.py`

---

#### TASK-017: Implement SL exit logic in qengine
**Type**: feature
**Priority**: high
**Estimated**: 1.5 hours
**Dependencies**: [TASK-006]

**Description**: Implement SL exit logic exactly matching VectorBT

**Acceptance Criteria**:
- SL trigger condition matches VectorBT formula
- SL price calculation matches VectorBT
- SL timing matches VectorBT
- Code documented
- Unit tests pass

**Files**:
- `backtest/src/qengine/strategies/vectorbt_strategy.py`
- `backtest/tests/test_sl_logic.py`

---

#### TASK-018: Implement TSL tracking and exit logic in qengine
**Type**: feature
**Priority**: high
**Estimated**: 3 hours
**Dependencies**: [TASK-007]

**Description**: Implement TSL tracking and exit logic exactly matching VectorBT

**Acceptance Criteria**:
- TSL tracking mechanism matches VectorBT
- TSL trigger condition matches VectorBT
- TSL update frequency matches VectorBT
- Initial TSL calculation matches VectorBT
- Code documented
- Unit tests pass

**Files**:
- `backtest/src/qengine/strategies/vectorbt_strategy.py`
- `backtest/tests/test_tsl_logic.py`

---

#### TASK-019: Implement exit priority handling in qengine
**Type**: feature
**Priority**: medium
**Estimated**: 1.5 hours
**Dependencies**: [TASK-008, TASK-016, TASK-017, TASK-018]

**Description**: Handle same-bar exit conflicts exactly like VectorBT

**Acceptance Criteria**:
- Exit priority rules match VectorBT
- Simultaneous trigger handling verified
- Code documented
- Unit tests for conflict scenarios

**Files**:
- `backtest/src/qengine/strategies/vectorbt_strategy.py`
- `backtest/tests/test_exit_priority.py`

---

#### TASK-020: Implement identical fee calculation in qengine
**Type**: feature
**Priority**: high
**Estimated**: 1.5 hours
**Dependencies**: [TASK-009]

**Description**: Calculate fees exactly matching VectorBT

**Acceptance Criteria**:
- Fee formula matches VectorBT
- Fee application points match VectorBT
- PnL impact matches VectorBT
- Code documented
- Unit tests pass

**Files**:
- `backtest/src/qengine/strategies/vectorbt_strategy.py`
- `backtest/tests/test_fees.py`

---

#### TASK-021: Implement identical slippage calculation in qengine
**Type**: feature
**Priority**: high
**Estimated**: 1.5 hours
**Dependencies**: [TASK-010]

**Description**: Calculate slippage exactly matching VectorBT

**Acceptance Criteria**:
- Slippage formula matches VectorBT
- Slippage application points match VectorBT
- Direction handling matches VectorBT
- Code documented
- Unit tests pass

**Files**:
- `backtest/src/qengine/strategies/vectorbt_strategy.py`
- `backtest/tests/test_slippage.py`

---

#### TASK-022: Implement identical position sizing in qengine
**Type**: feature
**Priority**: high
**Estimated**: 2 hours
**Dependencies**: [TASK-011, TASK-020, TASK-021]

**Description**: Calculate position size exactly matching VectorBT's size=np.inf

**Acceptance Criteria**:
- Position sizing formula matches VectorBT
- Cash allocation matches VectorBT
- Fee/slippage impact on sizing matches
- Quantity calculation matches
- Code documented
- Unit tests pass

**Files**:
- `backtest/src/qengine/strategies/vectorbt_strategy.py`
- `backtest/tests/test_position_sizing.py`

---

#### TASK-023: Implement identical entry price logic in qengine
**Type**: feature
**Priority**: high
**Estimated**: 1.5 hours
**Dependencies**: [TASK-003]

**Description**: Calculate entry prices exactly matching VectorBT

**Acceptance Criteria**:
- Entry price formula matches VectorBT
- Entry timing matches VectorBT
- Slippage application matches
- Fee application matches
- Code documented
- Unit tests pass

**Files**:
- `backtest/src/qengine/strategies/vectorbt_strategy.py`
- `backtest/tests/test_entry_logic.py`

---

### Phase 4: Validation and Debugging (Verify)

#### TASK-024: Run initial qengine backtest with same signals
**Type**: testing
**Priority**: high
**Estimated**: 2 hours
**Dependencies**: [TASK-16 through TASK-23]

**Description**: Run qengine backtest with identical signals from Phase 1

**Acceptance Criteria**:
- qengine backtest completes without errors
- Trade log generated
- Basic metrics calculated (trade count, PnL, etc.)
- Results saved for comparison

**Files**:
- `projects/crypto_futures/scripts/run_qengine_backtest.py`
- `projects/crypto_futures/data/qengine_trades_q1_2024.parquet`

---

#### TASK-025: Compare aggregate metrics (trade count, total PnL)
**Type**: testing
**Priority**: high
**Estimated**: 1 hour
**Dependencies**: [TASK-024, TASK-013]

**Description**: Compare high-level metrics between VectorBT and qengine

**Acceptance Criteria**:
- Trade count comparison (target: exact match)
- Total PnL comparison (target: <0.01% difference)
- Win rate comparison
- Average duration comparison
- Report generated with differences

**Files**:
- `projects/crypto_futures/reports/aggregate_comparison.md`

---

#### TASK-026: Implement trade-by-trade comparison script
**Type**: testing
**Priority**: high
**Estimated**: 2 hours
**Dependencies**: [TASK-024, TASK-013]

**Description**: Run detailed trade-by-trade comparison

**Acceptance Criteria**:
- Every trade compared individually
- Differences highlighted with metrics
- Clear reporting of mismatches
- Summary statistics (% matching, avg deviation)
- Exportable comparison report

**Files**:
- `projects/crypto_futures/scripts/detailed_trade_comparison.py`
- `projects/crypto_futures/reports/trade_by_trade_comparison.csv`

---

#### TASK-027: Debug first 10 trade discrepancies
**Type**: debugging
**Priority**: high
**Estimated**: 3 hours
**Dependencies**: [TASK-026]

**Description**: Investigate and fix first 10 trade differences

**Acceptance Criteria**:
- Root cause identified for each discrepancy
- qengine code updated if needed
- VectorBT behavior clarified if needed
- First 10 trades match exactly
- Fixes documented

**Files**:
- `projects/crypto_futures/docs/debugging_notes.md`

---

#### TASK-028: Debug remaining trade discrepancies
**Type**: debugging
**Priority**: high
**Estimated**: 4 hours
**Dependencies**: [TASK-027]

**Description**: Iterate until all trades match

**Acceptance Criteria**:
- All 352 trades analyzed
- All discrepancies resolved
- 100% matching achieved
- Common error patterns documented

**Files**:
- `projects/crypto_futures/docs/debugging_notes.md` (updated)

---

#### TASK-029: Verify 100% matching on test period
**Type**: testing
**Priority**: high
**Estimated**: 1 hour
**Dependencies**: [TASK-028]

**Description**: Final verification of exact matching

**Acceptance Criteria**:
- Trade count: 352 = 352 ✅
- Entry prices: 100% match ✅
- Exit prices: 100% match ✅
- Fees: 100% match ✅
- PnL: 100% match (within rounding) ✅
- Verification report generated

**Files**:
- `projects/crypto_futures/reports/final_verification.md`

---

#### TASK-030: Test edge cases and boundary conditions
**Type**: testing
**Priority**: medium
**Estimated**: 2 hours
**Dependencies**: [TASK-029, TASK-015]

**Description**: Run edge case tests from TASK-014

**Acceptance Criteria**:
- All edge cases tested
- Behavior matches VectorBT for each case
- Edge case test suite passes
- Any surprises documented

**Files**:
- `projects/crypto_futures/tests/test_edge_cases.py`
- `projects/crypto_futures/reports/edge_case_results.md`

---

### Phase 5: Documentation (Finalize)

#### TASK-031: Write VECTORBT_BEHAVIOR_SPEC.md
**Type**: documentation
**Priority**: high
**Estimated**: 3 hours
**Dependencies**: [TASK-002 through TASK-011, TASK-029]

**Description**: Comprehensive VectorBT behavior specification

**Contents**:
- Entry price calculation (complete formula)
- Exit price calculation (complete formula)
- TP logic (trigger conditions, calculations)
- SL logic (trigger conditions, calculations)
- TSL logic (tracking, trigger conditions)
- Exit priority rules
- Fee calculation and application
- Slippage calculation and application
- Position sizing with size=np.inf
- Examples for each mechanism
- Edge case behaviors

**Acceptance Criteria**:
- All calculations documented with formulas
- Examples provided for each behavior
- Edge cases covered
- Verified accurate by re-reading

**Files**:
- `projects/crypto_futures/docs/VECTORBT_BEHAVIOR_SPEC.md`

---

#### TASK-032: Write QENGINE_IMPLEMENTATION_GUIDE.md
**Type**: documentation
**Priority**: high
**Estimated**: 2 hours
**Dependencies**: [TASK-016 through TASK-023, TASK-029]

**Description**: Guide for implementing VectorBT behavior in qengine

**Contents**:
- qengine architecture overview
- Strategy implementation details
- Code organization
- How to replicate each VectorBT behavior
- Common pitfalls and solutions
- Code examples

**Acceptance Criteria**:
- Clear step-by-step implementation guide
- Code examples for each component
- Pitfalls documented with solutions
- Usable by someone not familiar with project

**Files**:
- `projects/crypto_futures/docs/QENGINE_IMPLEMENTATION_GUIDE.md`

---

#### TASK-033: Generate TRADE_COMPARISON_REPORT.md
**Type**: documentation
**Priority**: high
**Estimated**: 1.5 hours
**Dependencies**: [TASK-029, TASK-030]

**Description**: Final comparison report with results

**Contents**:
- Test period and configuration
- VectorBT baseline results
- qengine results
- Trade-by-trade comparison summary
- Aggregate metrics comparison
- Edge case test results
- Verification proof (100% matching)

**Acceptance Criteria**:
- Complete results documented
- Clear proof of 100% matching
- Edge cases covered
- Exportable/shareable format

**Files**:
- `projects/crypto_futures/docs/TRADE_COMPARISON_REPORT.md`

---

#### TASK-034: Create verification test suite
**Type**: testing
**Priority**: medium
**Estimated**: 2 hours
**Dependencies**: [TASK-015, TASK-029]

**Description**: Package verification tests for reuse

**Acceptance Criteria**:
- pytest suite runs end-to-end
- Tests VectorBT vs qengine matching
- Can be run on new data periods
- Clear pass/fail reporting
- Documented how to use

**Files**:
- `projects/crypto_futures/tests/test_vectorbt_qengine_matching.py`
- `projects/crypto_futures/tests/README.md`

---

#### TASK-035: Final validation and sign-off
**Type**: testing
**Priority**: high
**Estimated**: 1 hour
**Dependencies**: [TASK-031, TASK-032, TASK-033, TASK-034]

**Description**: Final review and sign-off

**Acceptance Criteria**:
- All 35 tasks completed
- All documentation reviewed
- All tests passing
- 100% matching verified
- Ready for production use

**Deliverables**:
- VECTORBT_BEHAVIOR_SPEC.md ✅
- QENGINE_IMPLEMENTATION_GUIDE.md ✅
- TRADE_COMPARISON_REPORT.md ✅
- Verification test suite ✅
- qengine VectorBT-matching implementation ✅

---

## Critical Path

**Minimum viable sequence for 100% matching**:

1. TASK-001 (setup) →
2. TASK-002 (source analysis) →
3. TASK-003 through TASK-011 (reverse engineering) →
4. TASK-012 (extract VectorBT trades) →
5. TASK-013 (comparison framework) →
6. TASK-016 through TASK-023 (qengine implementation) →
7. TASK-024 (run qengine) →
8. TASK-026 (detailed comparison) →
9. TASK-027, TASK-028 (debugging) →
10. TASK-029 (final verification) →
11. TASK-031, TASK-032, TASK-033 (documentation)

**Parallel opportunities**:
- TASK-014, TASK-015 can run during Phase 1
- Documentation tasks can start once implementation verified

---

## Risk Analysis

### High Risk Areas

1. **VectorBT is proprietary** - Source code may be obfuscated or limited
   - **Mitigation**: Empirical testing + documentation review + trial-and-error

2. **Intra-bar behavior** - May be difficult to determine exact intra-bar logic
   - **Mitigation**: Test with extreme price movements, document assumptions

3. **Floating point precision** - Rounding differences may cause small mismatches
   - **Mitigation**: Define acceptable tolerance (e.g., 1e-8), document rounding

4. **Time complexity** - Debugging 352 trades may take longer than estimated
   - **Mitigation**: Focus on patterns, automate comparison, timebox debugging

### Medium Risk Areas

1. **qengine architecture differences** - Event-driven vs vectorized may cause issues
   - **Mitigation**: Careful implementation, extensive testing

2. **Edge cases** - Unusual market conditions may reveal unexpected behavior
   - **Mitigation**: Comprehensive edge case testing, document surprises

---

## Agent Recommendations

### TASK-002: VectorBT source code analysis
**Recommended**: `/agent architect` - For understanding complex architectural decisions in VectorBT

### TASK-016-023: qengine implementation
**Recommended**: `/agent code-reviewer` - For security and correctness review of exit logic

### TASK-034: Verification test suite
**Recommended**: `/agent test-engineer` - For comprehensive test strategy and coverage

---

## Success Metrics

### Phase Completion Criteria

**Phase 1 (Investigation)**: ✅ when all VectorBT behaviors documented
**Phase 2 (Test Setup)**: ✅ when comparison framework working
**Phase 3 (Implementation)**: ✅ when qengine code matches VectorBT logic
**Phase 4 (Validation)**: ✅ when 100% matching achieved
**Phase 5 (Documentation)**: ✅ when all deliverables complete

### Project Completion Criteria

✅ 100% trade count match (352 = 352)
✅ 100% entry price match
✅ 100% exit price match
✅ 100% fee calculation match
✅ 100% PnL match (within rounding)
✅ Complete VectorBT behavior specification
✅ qengine implementation guide
✅ Trade comparison report
✅ Verification test suite

**Zero tolerance for "approximate" - must be exact.**

---

## Next Steps

After planning approval:
1. Review plan for completeness
2. Adjust task estimates if needed
3. Begin with TASK-001
4. Use `/next` command to track progress

---

**Total Tasks**: 35
**Estimated Duration**: 50-60 hours (1.5-2 weeks)
**Complexity**: High (reverse engineering + exact matching)
**Risk**: Medium (VectorBT black box, trial-and-error required)
**Priority**: CRITICAL (foundational validation)
