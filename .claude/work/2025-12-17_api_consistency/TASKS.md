# Backtest Library API Consistency - Task List

**Created**: 2025-12-17
**Source**: API audit from parent software/ directory
**Overall Assessment**: 8/10 - Clean architecture with minor inconsistencies

---

## Context

The backtest library has solid API consistency overall. The issues found are naming inconsistencies and missing implementations, not architectural problems.

---

## Medium Priority

### Task 1: Fix commission model mapping
**Problem**: `BacktestConfig.commission_model` can be `PER_TRADE` but no `PerTradeCommission` model exists
**File**: `src/ml4t/backtest/config.py`, `src/ml4t/backtest/models.py`
**Fix Options**:
- Option A: Implement `PerTradeCommission` model
- Option B: Remove `PER_TRADE` from `CommModelEnum`
- Option C: Add validation to reject `PER_TRADE` with clear error
**Recommendation**: Option B or C (simplest)

### Task 2: Unify margin parameter naming
**Problem**: Broker uses `initial_margin`, Config uses `margin_requirement`
**Files**: `src/ml4t/backtest/config.py`, `src/ml4t/backtest/engine.py`
**Fix**: Rename config parameter to `initial_margin` for consistency with Broker interface
**Breaking**: Yes, but pre-release

### Task 3: Fix slippage parameter naming
**Problem**:
- `FixedSlippage` model uses `amount` parameter
- `BacktestConfig` uses `slippage_fixed`
**File**: `src/ml4t/backtest/config.py`
**Fix**: Rename `slippage_fixed` → `slippage_amount` for consistency
**Breaking**: Yes, but pre-release

---

## Low Priority

### Task 4: Create typed BacktestResult
**Problem**: `Engine.run()` returns `dict` with no type hints
**Current**: Users don't know what keys exist or their types
**Fix**: Create `BacktestResult` dataclass with all metrics
**Benefit**: Better IDE support, documentation, type safety
**Note**: Can deprecate dict access gradually

### Task 5: Eliminate duplicate metrics in results
**Problem**: Same metric accessible multiple ways:
- `results['sharpe']` (dict key)
- `results['equity'].sharpe()` (method call)
**Fix**: Remove dict duplication, use only object methods
**Depends on**: Task 4

### Task 6: Document canonical order creation pattern
**Problem**: Four ways to create orders (direct, submit_order, submit_bracket, close_position)
**Fix**: Add documentation recommending `broker.submit_order()` as primary
**No code changes needed**

---

## Verification Checklist

| Check | Status |
|-------|--------|
| No `PER_TRADE` commission model ambiguity | [ ] |
| Margin parameter names consistent | [ ] |
| Slippage parameter names consistent | [ ] |
| Results have typed structure | [ ] |
| Tests pass | [ ] |

---

## Notes

- Overall architecture is solid (8/10)
- Strong design patterns: Factory, Policy, Protocol
- Strategy interface is minimal and clean
- Types (Order, Position, Fill, Trade) are well-designed
- Accounting module is excellent

---

*Generated from backtest API audit - 2025-12-17*
