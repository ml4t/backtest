# Implementation Plan: Portfolio Rebalancing

## Overview

**Objective**: Add `TargetWeightExecutor` utility to convert portfolio target weights to orders, enabling integration with external optimizers like riskfolio-lib.

**Total Estimated Effort**: ~350 lines across 4 files

## Tasks

### TASK-001: Create rebalancer.py module
**Type**: Feature
**Priority**: High
**Dependencies**: None
**Estimated**: 150 lines

**Description**: Create `src/ml4t/backtest/execution/rebalancer.py` with:
- `RebalanceConfig` dataclass with all config options
- `TargetWeightExecutor` class with execute(), preview() methods
- `_get_current_weights()` and `_get_effective_weights()` helpers
- `WeightProvider` protocol (optional type hint)

**Acceptance Criteria**:
- [ ] `RebalanceConfig` has all fields from v2 proposal
- [ ] `execute()` handles pending order awareness
- [ ] `execute()` supports cancel-before-rebalance mode
- [ ] `execute()` handles fractional vs whole shares
- [ ] `preview()` returns trade details without executing
- [ ] Closes positions not in target weights

### TASK-002: Update execution module exports
**Type**: Integration
**Priority**: High
**Dependencies**: TASK-001
**Estimated**: 5 lines

**Description**: Update `src/ml4t/backtest/execution/__init__.py` to export new classes.

**Acceptance Criteria**:
- [ ] `RebalanceConfig` exported
- [ ] `TargetWeightExecutor` exported
- [ ] Added to `__all__` list

### TASK-003: Update main package exports
**Type**: Integration
**Priority**: High
**Dependencies**: TASK-002
**Estimated**: 10 lines

**Description**: Update `src/ml4t/backtest/__init__.py` to export at package level.

**Acceptance Criteria**:
- [ ] `from ml4t.backtest import TargetWeightExecutor, RebalanceConfig` works
- [ ] Added to `__all__` list

### TASK-004: Create comprehensive tests
**Type**: Testing
**Priority**: High
**Dependencies**: TASK-001
**Estimated**: 200 lines

**Description**: Create `tests/execution/test_rebalancer.py` with tests for:
- Basic rebalancing (buy to target, sell to target)
- Pending order awareness
- Cancel-before-rebalance mode
- Fractional vs whole share handling
- Cash targeting (weights < 1.0)
- Position closing for assets not in target
- Edge cases (empty portfolio, zero equity, etc.)
- Preview functionality

**Acceptance Criteria**:
- [ ] All test scenarios pass
- [ ] Coverage > 90% for rebalancer.py
- [ ] Edge cases handled

## Execution Order

```
TASK-001 (rebalancer.py)
    │
    ├── TASK-002 (execution exports)
    │       │
    │       └── TASK-003 (package exports)
    │
    └── TASK-004 (tests) [can run parallel with 002/003]
```

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Broker API changes | Low | High | Use existing public methods only |
| Pending order edge cases | Medium | Medium | Comprehensive test coverage |
| Float precision | Low | Low | Use tolerance in comparisons |

## Success Criteria

1. All 4 tasks complete
2. All tests pass
3. Import works: `from ml4t.backtest import TargetWeightExecutor, RebalanceConfig`
4. Manual test with sample rebalancing works
