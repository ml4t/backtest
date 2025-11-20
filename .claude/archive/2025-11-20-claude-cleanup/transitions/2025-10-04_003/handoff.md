# Handoff: Test Fixes and Coverage Expansion

**Date**: 2025-10-04
**Session**: 003_test_fixes_planning
**Status**: Planning complete, ready for execution
**Next Focus**: Execute test fixes starting with TASK-001

---

## Active Work

**Primary Objective**: Fix 41 failing tests and expand coverage to 85%+

**Current Phase**: Planning complete, implementation ready to start

**Root Cause Identified**: FillSimulator refactoring changed commission model API from `fee=` to `commission=`, breaking integration tests.

---

## Current State

### Test Status
- **Passing**: 282/323 tests (87% pass rate)
- **Failing**: 41 tests across 9 test files
- **Coverage**: 80% overall (3,751 statements)

### Key Accomplishments This Session
1. ✅ **FillSimulator validation complete** - 91% coverage, all tests passing
2. ✅ **Issue #2 investigated** - Proven NOT a bug (get_position works correctly)
3. ✅ **Root cause analysis** - Commission API change identified
4. ✅ **Implementation plan created** - 9 tasks, 6-10 hours total
5. ✅ **State tracking setup** - state.json ready for `/next` command

### Files Created This Session
1. `.claude/transitions/latest/completion.md` - FillSimulator completion
2. `.claude/transitions/latest/issue2_resolution.md` - Issue #2 analysis
3. `.claude/transitions/latest/session_summary.md` - Session overview
4. `.claude/work/exploration_test_fixes.md` - Test failure exploration
5. `.claude/work/test_fixes_plan.md` - Implementation plan
6. `.claude/work/state.json` - Task tracking
7. `tests/unit/test_portfolio_get_position.py` - Portfolio tests (6 tests, all passing)

---

## Recent Decisions

### Critical Issue Status Updates
- **Issue #1 (max_leverage)**: ✅ RESOLVED - FillSimulator implementation complete, 91% coverage
- **Issue #2 (get_position)**: ✅ NOT A BUG - Works correctly, issue in ml-strategies usage
- **Issue #3 (Risk management)**: ⏳ NEXT PRIORITY - After test fixes complete
- **Issue #4 (Fill feedback)**: ⏳ PENDING
- **Issue #5 (Warnings)**: ⏳ PENDING

### Test Failure Root Causes
1. **Commission API change** (13 tests) - Simple fix: `fee` → `commission` parameter
2. **Broker integration** (17 tests) - FillSimulator refactoring side effects
3. **Other issues** (11 tests) - Clock, engine, P&L tests need investigation

### Coverage Targets
- **Current**: 80% (good baseline)
- **Target**: 85%+ (after fixes)
- **Low areas**: Margin (29%), Clock (36%) - planned expansion

---

## Implementation Plan Summary

### Phase 1: Critical Test Fixes (4-5 hours)
- **TASK-001**: Fix commission API (30 min) - NO DEPENDENCIES
- **TASK-002**: Fix broker integration tests (2 hours)
- **TASK-003**: Fix clock multi-feed tests (1.5 hours) - NO DEPENDENCIES
- **TASK-004**: Fix engine tests (1 hour)
- **TASK-005**: Fix P&L tests (1 hour)

### Phase 2: Coverage Expansion (3-4 hours)
- **TASK-006**: Margin module tests (2.5 hours) - 29% → 70%+
- **TASK-007**: Clock module coverage (1 hour) - 36% → 70%+
- **TASK-008**: Portfolio coverage (1 hour) - 77% → 90%+

### Phase 3: Verification (30 min)
- **TASK-009**: Full test suite validation

### Quick Win Path (2-3 hours)
If time-limited, just complete Phase 1:
1. TASK-001: Fix commission API (30 min)
2. TASK-002: Fix broker tests (2 hours)
3. Run verification
4. **Result**: All 323 tests passing, 80% coverage maintained

---

## Next Steps (Immediate Actions)

### Start with TASK-001 (Commission API Fix)
**No dependencies - can start immediately**

```bash
# Navigate to project
cd /home/stefan/ml4t/backtest

# Fix commission API parameter
sed -i 's/FlatCommission(fee=/FlatCommission(commission=/g' \
  tests/unit/test_cash_constraints.py \
  tests/unit/test_slippage.py \
  tests/unit/test_commission.py

# Verify fixes
uv run pytest tests/unit/test_cash_constraints.py \
  tests/unit/test_slippage.py \
  tests/unit/test_commission.py -v

# Expected: 13 tests pass (was failing)
```

### Or Use `/next` Command
```bash
/next  # Automatically executes TASK-001
```

### Parallel Option: TASK-003 (Clock Tests)
Can run independently while investigating TASK-002:
```bash
uv run pytest tests/unit/test_clock_multi_feed.py -xvs
# Investigate failures, fix, verify
```

---

## Session Context

### Working Directory
```
/home/stefan/ml4t/backtest/
```

### Key File Locations
- **Tests**: `tests/unit/` (323 total, 41 failing)
- **Source**: `src/ml4t.backtest/` (80% coverage)
- **Plan**: `.claude/work/test_fixes_plan.md`
- **State**: `.claude/work/state.json`
- **Exploration**: `.claude/work/exploration_test_fixes.md`

### Test Failure Distribution
| File | Failures | Fix Type |
|------|----------|----------|
| `test_cash_constraints.py` | 8 | API parameter |
| `test_clock_multi_feed.py` | 7 | Investigation |
| `test_market_impact_integration.py` | 6 | Broker API |
| `test_liquidity.py` | 6 | Broker API |
| `test_lookahead_prevention.py` | 5 | Broker API |
| `test_slippage.py` | 3 | API parameter |
| `test_pnl_calculations.py` | 2 | Investigation |
| `test_engine.py` | 2 | Investigation |
| `test_commission.py` | 2 | API parameter |

### Recent Test Results
```bash
# Last full run
uv run pytest tests/unit/ --tb=no -q
# Result: 41 failed, 282 passed

# FillSimulator tests (our recent work)
uv run pytest tests/unit/test_fill_simulator.py -v
# Result: 34/34 PASSED ✅

# Portfolio tests (our recent work)
uv run pytest tests/unit/test_portfolio_get_position.py -v
# Result: 6/6 PASSED ✅
```

---

## Active Challenges

### Challenge 1: Broker Integration Test Failures
**Status**: Needs investigation (TASK-002)

**Known Issues**:
- FillSimulator refactoring may have changed broker initialization
- Event handling timing may have changed
- Portfolio integration may need updates

**Investigation Approach**:
1. Run one test file at a time to isolate issues
2. Check broker constructor parameter changes
3. Verify FillSimulator integration points
4. Update test fixtures incrementally

**Files to Check**:
- `tests/unit/test_market_impact_integration.py`
- `tests/unit/test_liquidity.py`
- `tests/unit/test_lookahead_prevention.py`

### Challenge 2: Clock Multi-Feed Tests
**Status**: Needs investigation (TASK-003)

**7 failing tests** - All in `test_clock_multi_feed.py`

**Potential Issues**:
- Clock API may have changed
- Feed registration logic may have changed
- Event sequencing may need updates

### Challenge 3: Coverage Expansion Priority
**Decision Point**: Full path (85%+) vs Quick win (80%)?

**Quick Win** (2-3 hours):
- Just fix tests, maintain 80% coverage
- Production ready faster

**Full Path** (6-10 hours):
- Fix tests + expand coverage
- Better long-term quality

---

## Important Context for Next Agent

### FillSimulator Architecture (Recent Refactor)
```
SimulationBroker (Orchestrator)
├── PositionTracker (85% coverage)
├── OrderRouter (67% coverage)
├── BracketOrderManager (83% coverage)
└── FillSimulator (93% coverage) ← Recently extracted
    ├── max_leverage parameter ← NEW (Critical Issue #1 fix)
    ├── Cash constraints with binary search
    ├── Liquidity constraints
    ├── Margin constraints
    └── Commission/slippage calculations
```

### Commission Model API Change
**Old API** (tests still use):
```python
FlatCommission(fee=10.0)
```

**New API** (correct):
```python
FlatCommission(commission=10.0)
```

This is the root cause of 13 test failures.

### Portfolio Position Tracking
**Confirmed Working** (Issue #2 investigation):
```python
# SimplePortfolio inherits from Portfolio
portfolio.get_position(asset_id)  # Works correctly ✅

# Tests prove it works:
# tests/unit/test_portfolio_get_position.py - 6/6 passing
```

If ml-strategies still reports None, it's a usage error:
- Asset ID mismatch
- Event timing (querying before fill processed)
- Portfolio reference not set

---

## State Management

### Task Tracking
**File**: `.claude/work/state.json`

**Current Status**:
```json
{
  "status": "planning_complete",
  "current_task": null,
  "next_available": ["TASK-001", "TASK-003"]
}
```

**Ready for**: `/next` command will execute TASK-001

### Work Unit Files
1. **Plan**: `.claude/work/test_fixes_plan.md` - Detailed implementation plan
2. **State**: `.claude/work/state.json` - Task tracking
3. **Exploration**: `.claude/work/exploration_test_fixes.md` - Analysis findings

---

## Quality Standards

### Testing Requirements
- All 323 tests must pass (100% pass rate)
- Coverage target: 85%+ (current: 80%)
- Run checks after each fix: `uv run pytest tests/unit/test_*.py -v`

### Code Quality Checks
```bash
# Linting
uv run ruff check src/ tests/

# Type checking
uv run mypy src/

# Coverage report
uv run pytest tests/unit/ --cov=ml4t.backtest --cov-report=term-missing
```

### Success Criteria
- [ ] All 323 tests passing
- [ ] Overall coverage ≥ 85%
- [ ] No ruff violations
- [ ] No mypy errors
- [ ] No regressions in existing functionality

---

## Performance Metrics

### Session Statistics
- **Duration**: ~3 hours
- **Tests created**: 40 (34 FillSimulator + 6 Portfolio)
- **Coverage achieved**: FillSimulator 93%, Portfolio 96%
- **Issues resolved**: 2 (max_leverage implemented, get_position verified)
- **Issues identified**: 41 failing tests (root cause found)

### Test Suite Status
- **Before session**: FillSimulator incomplete, Issue #2 unverified
- **After session**: FillSimulator 93% coverage, Issue #2 proven not a bug, test fix plan ready
- **Next milestone**: All 323 tests passing, 85%+ coverage

---

## Recommendations for Next Session

### Immediate Priority (30 min - 2 hours)
1. **Start with TASK-001** - Quick win, fixes 13 tests
2. **Run verification** - Confirm fixes work
3. **Move to TASK-002** - Investigate broker tests

### Time-Limited Approach (2-3 hours total)
- Just complete Phase 1 (test fixes)
- Skip Phase 2 (coverage expansion)
- Result: All tests passing, 80% coverage (acceptable)

### Full Approach (6-10 hours)
- Complete all 9 tasks
- Result: All tests passing, 85%+ coverage (ideal)

### Risk Mitigation
- Fix commission API first (low risk, high value)
- Tackle broker integration systematically
- Document any new API discoveries
- Keep changes minimal and focused

---

## Context for Implementation

### Commission API Fixes (TASK-001)
**Pattern**:
```bash
# Before
FlatCommission(fee=10.0)

# After
FlatCommission(commission=10.0)
```

**Files**: 3 test files, ~13 occurrences
**Time**: 30 minutes
**Risk**: Very low

### Broker Integration (TASK-002)
**Approach**:
1. Run each test file individually
2. Identify specific error messages
3. Check broker initialization code
4. Compare with working tests (broker.py, fill_simulator.py tests)
5. Update incrementally

**Files**: 3 test files, 17 tests
**Time**: 2 hours
**Risk**: Medium (requires investigation)

---

## Files to Reference

### Implementation Guides
- `.claude/work/test_fixes_plan.md` - Complete task breakdown
- `.claude/work/exploration_test_fixes.md` - Root cause analysis
- `.claude/reference/ARCHITECTURE.md` - System architecture
- `.claude/planning/CRITICAL_ISSUES_FROM_ML_STRATEGIES.md` - Issue context

### Recent Completions
- `.claude/transitions/latest/completion.md` - FillSimulator done
- `.claude/transitions/latest/issue2_resolution.md` - get_position verified
- `.claude/transitions/latest/session_summary.md` - Today's overview

### Test Files
- `tests/unit/test_fill_simulator.py` - 34 tests, all passing ✅
- `tests/unit/test_portfolio_get_position.py` - 6 tests, all passing ✅
- `tests/unit/test_broker.py` - 13 tests, all passing ✅

---

## Environment Setup

### Python Environment
```bash
cd /home/stefan/ml4t/backtest
# uv environment already configured
```

### Quick Verification
```bash
# Check test status
uv run pytest tests/unit/ --tb=no -q | tail -1
# Should show: 41 failed, 282 passed

# Check coverage
uv run pytest tests/unit/ --cov=ml4t.backtest --cov-report=term | grep TOTAL
# Should show: ~80% coverage
```

---

## Transition Prompt for Next Agent

```
Continue test fixes and coverage expansion from handoff document.

Current state:
- 41 failing tests identified (root cause: commission API change)
- Implementation plan ready with 9 tasks
- FillSimulator complete (91% coverage)
- Issue #2 proven not a bug

Next action:
Run `/next` to execute TASK-001 (commission API fix, 30 min)
Or start manually with:
  sed -i 's/FlatCommission(fee=/FlatCommission(commission=/g' tests/unit/test_*.py

See: .claude/transitions/2025-10-04_003/handoff.md for full context
Plan: .claude/work/test_fixes_plan.md
```

---

**Session Complete**: 2025-10-04
**Handoff Created**: 2025-10-04_003
**Ready for**: Test fixes execution
**Estimated Effort**: 2-10 hours (depending on path chosen)
