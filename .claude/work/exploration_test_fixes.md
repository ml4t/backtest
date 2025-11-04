# Test Fixes and Coverage Expansion - Exploration

**Date**: 2025-10-04
**Status**: Analysis Complete
**Work Type**: Bug fixes + Coverage expansion

---

## Executive Summary

**Current State**:
- 41 failing tests (out of 323 total)
- 80% overall coverage
- Low coverage areas: Margin (29%), Clock (36%), some execution modules

**Root Cause**: FillSimulator refactoring changed commission model API from `fee=` to `commission=`

**Impact**: Integration tests using old API are broken

---

## Test Failure Analysis

### Failing Test Categories

| Test File | Failures | Root Cause |
|-----------|----------|------------|
| `test_cash_constraints.py` | 8 | Commission API change (`fee` → `commission`) |
| `test_clock_multi_feed.py` | 7 | Unknown (needs investigation) |
| `test_market_impact_integration.py` | 6 | Likely broker API changes |
| `test_liquidity.py` | 6 | Likely broker API changes |
| `test_lookahead_prevention.py` | 5 | Likely broker API changes |
| `test_slippage.py` | 3 | Commission API change |
| `test_commission.py` | 2 | Commission API change |
| `test_pnl_calculations.py` | 2 | Unknown (needs investigation) |
| `test_engine.py` | 2 | Unknown (needs investigation) |

### Root Cause #1: Commission Model API Change

**Old API** (tests use this):
```python
FlatCommission(fee=10.0)
PercentageCommission(rate=0.001)
```

**New API** (correct):
```python
FlatCommission(commission=10.0)  # parameter renamed
PercentageCommission(rate=0.001)  # unchanged
```

**Affected Tests**:
- `test_cash_constraints.py` - All 8 tests
- `test_slippage.py` - 3 tests
- `test_commission.py` - 2 tests

**Fix**: Simple parameter rename in test setup

### Root Cause #2: Broker Integration Changes

The FillSimulator extraction likely changed how broker is initialized or used in tests. Need to investigate:
- Execution delay parameter changes
- Portfolio integration changes
- Event handling changes

**Affected Tests**:
- `test_market_impact_integration.py` - 6 tests
- `test_liquidity.py` - 6 tests
- `test_lookahead_prevention.py` - 5 tests

### Root Cause #3: Unknown Issues

Need detailed investigation:
- `test_clock_multi_feed.py` - 7 tests (clock functionality)
- `test_pnl_calculations.py` - 2 tests (P&L calculations)
- `test_engine.py` - 2 tests (engine integration)

---

## Coverage Analysis

### Current Coverage: 80% (3,751 statements, 763 missed)

#### High Coverage Components ✅
- **FillSimulator: 93%** (our recent work)
- **Portfolio: 96%** (our recent work)
- **Commission: 98%**
- **Slippage: 88%**
- **Market Impact: 96%**
- **Liquidity: 98%**
- **Engine: 92%**
- **Event: 88%**
- **Types: 100%**

#### Low Coverage Components ⚠️
- **Margin: 29%** (129 stmts, 91 missed) - Needs margin account tests
- **Clock: 36%** (145 stmts, 93 missed) - Needs multi-feed tests
- **SimplePortfolio: 77%** (could improve to 85%+)
- **Assets: 70%** (could improve)
- **Order Router: 67%** (could improve)

---

## Implementation Plan

### Phase 1: Fix Broken Tests (HIGH Priority)

#### Step 1.1: Fix Commission API Changes
**Files to Update**:
- `tests/unit/test_cash_constraints.py` (8 fixes)
- `tests/unit/test_slippage.py` (3 fixes)
- `tests/unit/test_commission.py` (2 fixes)

**Change Pattern**:
```python
# OLD
FlatCommission(fee=10.0)

# NEW
FlatCommission(commission=10.0)
```

**Estimated Time**: 15-30 minutes (simple regex replacement)

#### Step 1.2: Fix Broker Integration Tests
**Files to Update**:
- `tests/unit/test_market_impact_integration.py` (6 tests)
- `tests/unit/test_liquidity.py` (6 tests)
- `tests/unit/test_lookahead_prevention.py` (5 tests)

**Investigation Needed**:
1. Check broker initialization parameters
2. Verify FillSimulator integration
3. Check event handling changes
4. Update test fixtures

**Estimated Time**: 1-2 hours (requires investigation + fixes)

#### Step 1.3: Fix Remaining Tests
**Files to Update**:
- `tests/unit/test_clock_multi_feed.py` (7 tests) - Clock functionality
- `tests/unit/test_pnl_calculations.py` (2 tests) - P&L calculations
- `tests/unit/test_engine.py` (2 tests) - Engine integration

**Estimated Time**: 1-2 hours (investigation + fixes)

### Phase 2: Expand Coverage (MEDIUM Priority)

#### Step 2.1: Margin Module Coverage (29% → 70%+)
**Target**: Add margin account tests
**Files**: `src/qengine/portfolio/margin.py`
**Test File**: Create `tests/unit/test_margin.py`

**Test Coverage Needed**:
- Initial margin calculations
- Maintenance margin enforcement
- Margin calls
- Position liquidation
- Leverage limits

**Estimated Time**: 2-3 hours

#### Step 2.2: Clock Module Coverage (36% → 70%+)
**Target**: Add multi-feed clock tests
**Files**: `src/qengine/core/clock.py`
**Test File**: Fix `tests/unit/test_clock_multi_feed.py` (currently failing)

**Test Coverage Needed**:
- Multiple data feed synchronization
- Event sequencing
- Time advancement
- Feed registration

**Estimated Time**: 1-2 hours (includes fixing existing tests)

#### Step 2.3: SimplePortfolio Coverage (77% → 90%+)
**Target**: Improve portfolio test coverage
**Files**: `src/qengine/portfolio/simple.py`
**Test File**: Expand `tests/unit/test_portfolio_get_position.py`

**Additional Tests Needed**:
- Metrics calculation
- Returns calculation
- Trade history
- Finalization

**Estimated Time**: 1-2 hours

### Phase 3: Verification (Required)

#### Step 3.1: Full Test Suite
```bash
uv run pytest tests/unit/ -v
```
**Expected**: All 323 tests pass

#### Step 3.2: Coverage Report
```bash
uv run pytest tests/unit/ --cov=qengine --cov-report=term-missing
```
**Target**: 85%+ overall coverage

#### Step 3.3: Quality Checks
```bash
uv run ruff check src/ tests/
uv run mypy src/
```
**Expected**: No errors

---

## Effort Estimates

### Total Time: 6-10 hours

| Phase | Task | Time | Priority |
|-------|------|------|----------|
| 1.1 | Fix commission API | 0.5h | HIGH |
| 1.2 | Fix broker integration tests | 2h | HIGH |
| 1.3 | Fix remaining tests | 2h | HIGH |
| 2.1 | Margin coverage | 3h | MEDIUM |
| 2.2 | Clock coverage | 2h | MEDIUM |
| 2.3 | Portfolio coverage | 1h | LOW |
| 3 | Verification | 0.5h | REQUIRED |

### Quick Wins (2-3 hours)
If time is limited, focus on Phase 1 (fix broken tests):
- Commission API fixes (30 min)
- Broker integration fixes (2 hours)
- Run full test suite (30 min)

This gets us to **all tests passing** without coverage expansion.

---

## Risk Assessment

### Low Risk ✅
- Commission API fixes (straightforward parameter rename)
- Margin tests (additive, won't break existing)
- Portfolio coverage expansion (additive)

### Medium Risk ⚠️
- Broker integration tests (may reveal deeper issues)
- Clock multi-feed tests (complex timing logic)
- Engine tests (integration complexity)

### Mitigation
- Fix tests incrementally, one category at a time
- Run tests after each fix to catch regressions
- Keep changes minimal and focused

---

## Success Criteria

### Must Have ✅
- [ ] All 323 tests passing
- [ ] No regressions in existing functionality
- [ ] Commission API consistently updated

### Should Have 🎯
- [ ] 85%+ overall coverage
- [ ] Margin module coverage >70%
- [ ] Clock module coverage >70%

### Nice to Have ⭐
- [ ] 90%+ overall coverage
- [ ] All modules >80% coverage
- [ ] Performance benchmarks pass

---

## Next Steps

### Immediate (Session Start)
1. **Fix commission API** - Quick regex replacement across 3 test files
2. **Run tests** - Verify fixes work
3. **Move to broker integration** - Investigate and fix

### Sequential Approach
```bash
# Step 1: Fix commission API
sed -i 's/FlatCommission(fee=/FlatCommission(commission=/g' tests/unit/test_*.py

# Step 2: Run tests to see progress
uv run pytest tests/unit/test_cash_constraints.py tests/unit/test_commission.py tests/unit/test_slippage.py -v

# Step 3: Investigate broker integration
uv run pytest tests/unit/test_market_impact_integration.py -xvs

# Step 4: Fix and iterate
```

---

## Files Affected

### Test Files to Fix (Phase 1)
1. `tests/unit/test_cash_constraints.py` - Commission API
2. `tests/unit/test_slippage.py` - Commission API
3. `tests/unit/test_commission.py` - Commission API
4. `tests/unit/test_market_impact_integration.py` - Broker API
5. `tests/unit/test_liquidity.py` - Broker API
6. `tests/unit/test_lookahead_prevention.py` - Broker API
7. `tests/unit/test_clock_multi_feed.py` - Clock functionality
8. `tests/unit/test_pnl_calculations.py` - P&L logic
9. `tests/unit/test_engine.py` - Engine integration

### Test Files to Create (Phase 2)
1. `tests/unit/test_margin.py` - NEW (margin account coverage)
2. `tests/unit/test_clock_multi_feed.py` - EXPAND (fix + coverage)
3. `tests/unit/test_portfolio_get_position.py` - EXPAND (more coverage)

---

## Context for Implementation

### Key Insights
1. **API stability matters** - Refactoring changed test contracts
2. **Test maintenance is critical** - 41 broken tests from one refactor
3. **Coverage is good (80%)** - Just needs targeted expansion
4. **Quick wins available** - Commission fixes are trivial

### Recommendations
1. **Start with quick wins** - Fix commission API first
2. **Investigate systematically** - One test category at a time
3. **Add coverage strategically** - Focus on margin and clock
4. **Maintain quality** - Run checks after each change

---

*Exploration Complete*
*Ready for implementation via `/plan` or direct fixes*
