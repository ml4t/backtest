# Implementation Plan: Test Fixes and Coverage Expansion

**Created**: 2025-10-04
**Objective**: Fix 41 failing tests and expand coverage to 85%+
**Total Effort**: 6-10 hours
**Priority**: HIGH (blocking deployment)

---

## Project Overview

### Current State
- **Tests**: 282 passing, 41 failing (87% pass rate)
- **Coverage**: 80% overall (3,751 statements)
- **Blocker**: API changes from FillSimulator refactoring

### Success Criteria
- ✅ All 323 tests passing (100% pass rate)
- ✅ 85%+ overall coverage
- ✅ No regressions in existing functionality
- ✅ Quality checks pass (ruff, mypy)

### Scope
**In Scope**:
- Fix all 41 failing tests
- Expand margin module coverage (29% → 70%+)
- Expand clock module coverage (36% → 70%+)
- Improve portfolio coverage (77% → 90%+)

**Out of Scope**:
- New feature development
- Performance optimization
- Documentation updates (unless required)

---

## Task Breakdown

### Phase 1: Critical Test Fixes (HIGH Priority)

#### TASK-001: Fix Commission API Parameter Changes
**Type**: Bug Fix
**Priority**: HIGH
**Estimated Time**: 30 minutes
**Dependencies**: None

**Description**:
Update test files to use new commission model API. The FillSimulator refactoring changed `FlatCommission(fee=X)` to `FlatCommission(commission=X)`.

**Files to Update**:
1. `tests/unit/test_cash_constraints.py` - 8 occurrences
2. `tests/unit/test_slippage.py` - 3 occurrences
3. `tests/unit/test_commission.py` - 2 occurrences

**Implementation**:
```bash
# Regex replacement across affected files
sed -i 's/FlatCommission(fee=/FlatCommission(commission=/g' \
  tests/unit/test_cash_constraints.py \
  tests/unit/test_slippage.py \
  tests/unit/test_commission.py
```

**Acceptance Criteria**:
- [ ] All 13 affected tests pass
- [ ] No new test failures introduced
- [ ] Commission calculations remain correct

**Testing**:
```bash
uv run pytest tests/unit/test_cash_constraints.py tests/unit/test_slippage.py tests/unit/test_commission.py -v
```

---

#### TASK-002: Investigate and Fix Broker Integration Tests
**Type**: Bug Fix
**Priority**: HIGH
**Estimated Time**: 2 hours
**Dependencies**: TASK-001

**Description**:
Fix broker integration tests broken by FillSimulator refactoring. These tests likely need updates to broker initialization or event handling.

**Files to Fix**:
1. `tests/unit/test_market_impact_integration.py` - 6 tests
2. `tests/unit/test_liquidity.py` - 6 tests
3. `tests/unit/test_lookahead_prevention.py` - 5 tests

**Investigation Steps**:
1. Run each test file to identify specific errors
2. Check broker initialization parameters
3. Verify FillSimulator integration
4. Check event handling and timing
5. Update test fixtures as needed

**Common Issues to Check**:
- Broker constructor parameters changed
- FillSimulator integration points
- Event timing and execution delay
- Portfolio integration changes

**Acceptance Criteria**:
- [ ] All 17 broker integration tests pass
- [ ] Market impact calculations correct
- [ ] Liquidity constraints enforced
- [ ] Lookahead prevention working

**Testing**:
```bash
uv run pytest tests/unit/test_market_impact_integration.py -v
uv run pytest tests/unit/test_liquidity.py -v
uv run pytest tests/unit/test_lookahead_prevention.py -v
```

---

#### TASK-003: Fix Clock Multi-Feed Tests
**Type**: Bug Fix
**Priority**: HIGH
**Estimated Time**: 1.5 hours
**Dependencies**: None

**Description**:
Fix 7 failing tests in clock multi-feed functionality. These tests verify proper synchronization of multiple data feeds.

**File**: `tests/unit/test_clock_multi_feed.py`

**Investigation**:
1. Identify specific test failures
2. Check clock API changes
3. Verify feed registration logic
4. Update test setup if needed

**Acceptance Criteria**:
- [ ] All 7 clock multi-feed tests pass
- [ ] Feed synchronization correct
- [ ] Event ordering preserved
- [ ] Time advancement working

**Testing**:
```bash
uv run pytest tests/unit/test_clock_multi_feed.py -v
```

---

#### TASK-004: Fix Engine Integration Tests
**Type**: Bug Fix
**Priority**: HIGH
**Estimated Time**: 1 hour
**Dependencies**: TASK-002

**Description**:
Fix 2 failing engine integration tests. Likely related to broker/portfolio integration changes.

**File**: `tests/unit/test_engine.py`

**Investigation**:
1. Run tests to identify errors
2. Check engine initialization
3. Verify component wiring
4. Update integration points

**Acceptance Criteria**:
- [ ] Both engine tests pass
- [ ] Backtest flow working
- [ ] Event processing correct
- [ ] Results generation working

**Testing**:
```bash
uv run pytest tests/unit/test_engine.py -v
```

---

#### TASK-005: Fix P&L Calculation Tests
**Type**: Bug Fix
**Priority**: HIGH
**Estimated Time**: 1 hour
**Dependencies**: TASK-002

**Description**:
Fix 2 failing P&L calculation tests. These verify profit and loss calculations for FX trades.

**File**: `tests/unit/test_pnl_calculations.py`

**Investigation**:
1. Identify P&L calculation errors
2. Check portfolio accounting
3. Verify position tracking
4. Update assertions if needed

**Acceptance Criteria**:
- [ ] Both P&L tests pass
- [ ] FX profit calculations correct
- [ ] Loss calculations accurate
- [ ] Accounting logic sound

**Testing**:
```bash
uv run pytest tests/unit/test_pnl_calculations.py -v
```

---

### Phase 2: Coverage Expansion (MEDIUM Priority)

#### TASK-006: Create Margin Module Tests
**Type**: Test Development
**Priority**: MEDIUM
**Estimated Time**: 2.5 hours
**Dependencies**: TASK-001 through TASK-005

**Description**:
Create comprehensive test suite for margin module to increase coverage from 29% to 70%+.

**New File**: `tests/unit/test_margin.py`

**Test Coverage Required**:
1. **Initial Margin Calculations**
   - Test margin requirements for different asset types
   - Verify leverage calculations
   - Check minimum margin enforcement

2. **Maintenance Margin**
   - Test maintenance margin thresholds
   - Verify margin call triggers
   - Check account status updates

3. **Position Management**
   - Test position liquidation logic
   - Verify partial liquidation
   - Check priority ordering

4. **Edge Cases**
   - Zero margin scenarios
   - Extreme leverage cases
   - Multiple position interactions

**Implementation Approach**:
```python
# Test structure
class TestMarginAccount:
    def test_initial_margin_calculation(self):
        # Test initial margin requirements

    def test_maintenance_margin_check(self):
        # Test maintenance margin enforcement

    def test_margin_call_trigger(self):
        # Test margin call logic

    def test_position_liquidation(self):
        # Test liquidation mechanics
```

**Acceptance Criteria**:
- [ ] Margin module coverage ≥ 70%
- [ ] All margin calculations tested
- [ ] Edge cases covered
- [ ] Integration with portfolio verified

**Testing**:
```bash
uv run pytest tests/unit/test_margin.py --cov=qengine.portfolio.margin --cov-report=term-missing
```

---

#### TASK-007: Expand Clock Module Coverage
**Type**: Test Development
**Priority**: MEDIUM
**Estimated Time**: 1 hour
**Dependencies**: TASK-003

**Description**:
Expand clock module test coverage from 36% to 70%+ by adding missing test cases after fixing existing tests.

**File**: `tests/unit/test_clock_multi_feed.py` (expand)

**Additional Coverage Needed**:
1. **Time Advancement**
   - Test various time increments
   - Verify boundary conditions
   - Check time zone handling

2. **Feed Registration**
   - Test dynamic feed addition
   - Verify feed removal
   - Check feed priority

3. **Event Sequencing**
   - Test event ordering
   - Verify timestamp handling
   - Check simultaneous events

**Acceptance Criteria**:
- [ ] Clock module coverage ≥ 70%
- [ ] All time logic tested
- [ ] Feed management covered
- [ ] Event sequencing verified

**Testing**:
```bash
uv run pytest tests/unit/test_clock*.py --cov=qengine.core.clock --cov-report=term-missing
```

---

#### TASK-008: Expand SimplePortfolio Coverage
**Type**: Test Development
**Priority**: LOW
**Estimated Time**: 1 hour
**Dependencies**: TASK-001 through TASK-005

**Description**:
Expand SimplePortfolio test coverage from 77% to 90%+ by adding missing test cases.

**File**: `tests/unit/test_portfolio_get_position.py` (expand)

**Additional Coverage Needed**:
1. **Metrics Calculation**
   - Test return calculations
   - Verify Sharpe ratio
   - Check drawdown metrics

2. **Trade History**
   - Test trade recording
   - Verify history retrieval
   - Check trade statistics

3. **Portfolio Finalization**
   - Test finalization logic
   - Verify state cleanup
   - Check final metrics

**Acceptance Criteria**:
- [ ] SimplePortfolio coverage ≥ 90%
- [ ] All methods tested
- [ ] Metrics calculations verified
- [ ] Trade tracking confirmed

**Testing**:
```bash
uv run pytest tests/unit/test_portfolio*.py --cov=qengine.portfolio.simple --cov-report=term-missing
```

---

### Phase 3: Verification and Quality (REQUIRED)

#### TASK-009: Full Test Suite Verification
**Type**: Validation
**Priority**: HIGH
**Estimated Time**: 30 minutes
**Dependencies**: All previous tasks

**Description**:
Run complete test suite to verify all fixes and ensure no regressions.

**Validation Steps**:
1. Run full unit test suite
2. Verify coverage metrics
3. Check quality standards
4. Generate coverage report

**Commands**:
```bash
# Full test suite
uv run pytest tests/unit/ -v

# Coverage report
uv run pytest tests/unit/ --cov=qengine --cov-report=term-missing --cov-report=html

# Quality checks
uv run ruff check src/ tests/
uv run mypy src/
```

**Acceptance Criteria**:
- [ ] All 323 tests passing (100%)
- [ ] Overall coverage ≥ 85%
- [ ] No ruff violations
- [ ] No mypy errors
- [ ] Coverage report generated

**Success Metrics**:
- Tests: 323/323 passing
- Coverage: ≥85%
- Quality: All checks pass

---

## Task Dependencies

```
TASK-001 (Commission API) → TASK-002 (Broker Integration)
                          → TASK-004 (Engine Tests)
                          → TASK-005 (P&L Tests)
                          → TASK-006 (Margin Tests)
                          → TASK-008 (Portfolio Tests)

TASK-003 (Clock Tests) → TASK-007 (Clock Coverage)

All → TASK-009 (Verification)
```

**Critical Path**: TASK-001 → TASK-002 → TASK-009 (4 hours minimum)

**Parallel Opportunities**:
- TASK-003 (Clock) can run parallel to TASK-001/002
- TASK-006, 007, 008 can run in any order after Phase 1

---

## Risk Assessment

### High Risk Items
1. **Broker Integration Complexity** (TASK-002)
   - Mitigation: Investigate systematically, fix one test at a time
   - Fallback: Focus on commission API fixes for quick win

2. **Unknown Test Failures** (TASK-003, 004, 005)
   - Mitigation: Debug individually, document findings
   - Fallback: Skip coverage expansion if time-limited

### Medium Risk Items
1. **Coverage Expansion** (TASK-006, 007, 008)
   - Mitigation: Prioritize critical paths, skip nice-to-haves
   - Fallback: Accept 80% coverage if tests pass

### Low Risk Items
1. **Commission API Fixes** (TASK-001)
   - Simple regex replacement
   - Low chance of issues

---

## Execution Strategy

### Quick Win Path (2-3 hours)
**Goal**: All tests passing, 80% coverage maintained

```
1. TASK-001: Fix commission API (30 min)
2. TASK-002: Fix broker tests (2 hours)
3. TASK-009: Verify (30 min)
```

**Outcome**: 323/323 tests passing, production ready

### Comprehensive Path (6-10 hours)
**Goal**: All tests passing, 85%+ coverage

```
Phase 1: Fix all tests (4.5 hours)
  - TASK-001 through TASK-005

Phase 2: Expand coverage (4.5 hours)
  - TASK-006 through TASK-008

Phase 3: Verify (30 min)
  - TASK-009
```

**Outcome**: 323/323 tests, 85%+ coverage, complete

---

## Implementation Notes

### For TASK-001 (Commission API)
- Use sed for quick replacement
- Verify with grep that all changes found
- Run tests immediately after

### For TASK-002 (Broker Integration)
- Run tests individually to isolate issues
- Check broker constructor signatures
- Verify FillSimulator integration points
- Update fixtures incrementally

### For TASK-006 (Margin Tests)
- Study existing margin module code first
- Use mocks for complex dependencies
- Focus on critical margin calculations
- Test edge cases thoroughly

---

## Success Indicators

### Phase 1 Complete When:
- ✅ All 41 failing tests now pass
- ✅ No new test failures introduced
- ✅ Commission API consistently updated
- ✅ Broker integration working

### Phase 2 Complete When:
- ✅ Margin coverage ≥ 70%
- ✅ Clock coverage ≥ 70%
- ✅ Portfolio coverage ≥ 90%
- ✅ Overall coverage ≥ 85%

### Phase 3 Complete When:
- ✅ 323/323 tests passing
- ✅ Coverage report confirms ≥85%
- ✅ All quality checks pass
- ✅ No regressions detected

---

## Next Steps

### Immediate Action
```bash
# Start with TASK-001
cd /home/stefan/ml4t/backtest

# Fix commission API
sed -i 's/FlatCommission(fee=/FlatCommission(commission=/g' \
  tests/unit/test_cash_constraints.py \
  tests/unit/test_slippage.py \
  tests/unit/test_commission.py

# Verify
uv run pytest tests/unit/test_cash_constraints.py -v
```

### Use `/next` Command
After creating state.json, use:
```bash
/next  # Execute next available task
```

---

## Files to Create/Modify

### Files to Modify (Phase 1)
1. `tests/unit/test_cash_constraints.py`
2. `tests/unit/test_slippage.py`
3. `tests/unit/test_commission.py`
4. `tests/unit/test_market_impact_integration.py`
5. `tests/unit/test_liquidity.py`
6. `tests/unit/test_lookahead_prevention.py`
7. `tests/unit/test_clock_multi_feed.py`
8. `tests/unit/test_engine.py`
9. `tests/unit/test_pnl_calculations.py`

### Files to Create (Phase 2)
1. `tests/unit/test_margin.py` - NEW

### Files to Expand (Phase 2)
1. `tests/unit/test_clock_multi_feed.py`
2. `tests/unit/test_portfolio_get_position.py`

---

**Plan Created**: 2025-10-04
**Ready for Execution**: Use `/next` to start TASK-001
