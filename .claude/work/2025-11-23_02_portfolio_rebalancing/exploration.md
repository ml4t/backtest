# Exploration Summary: Portfolio Rebalancing

## Codebase Analysis

### Existing Infrastructure

**Execution Module** (`src/ml4t/backtest/execution/`):
- Already exists with impact models and execution limits
- Current exports: `ExecutionLimits`, `VolumeParticipationLimit`, `MarketImpactModel`, etc.
- Good home for `TargetWeightExecutor` and `RebalanceConfig`

**Broker** (`src/ml4t/backtest/broker.py`):
- `cancel_order(order_id: str) -> bool` exists at line 341 ✅
- `pending_orders` list is accessible ✅
- `get_account_value()` method exists ✅
- `positions` dict accessible ✅
- `submit_order()` and `close_position()` ready to use ✅

**Types** (`src/ml4t/backtest/types.py`):
- `OrderSide` enum exists
- `Order` dataclass ready
- All needed types available

### Integration Points

1. **New file**: `src/ml4t/backtest/execution/rebalancer.py`
2. **Update**: `src/ml4t/backtest/execution/__init__.py` (add exports)
3. **Update**: `src/ml4t/backtest/__init__.py` (add to package exports)
4. **Tests**: `tests/execution/test_rebalancer.py`

## Implementation Approach

### Phase 1: Core Implementation
Create `rebalancer.py` with:
- `RebalanceConfig` dataclass
- `TargetWeightExecutor` class
- `WeightProvider` protocol (optional, for type hints)

### Phase 2: Integration
Update module exports to make available via:
```python
from ml4t.backtest import TargetWeightExecutor, RebalanceConfig
from ml4t.backtest.execution import TargetWeightExecutor, RebalanceConfig
```

### Phase 3: Testing
Create comprehensive tests for:
- Basic rebalancing (buy/sell to targets)
- Pending order awareness
- Cancel-before-rebalance mode
- Fractional vs whole shares
- Cash targeting (weights < 1.0)
- Edge cases (empty portfolio, single asset, etc.)

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Location | execution module | Fits with existing limits/impact |
| Pending orders | cancel-first default | Safest, prevents double-allocation |
| Fractional shares | Off by default | Most brokers don't support |
| Class vs function | Class | Need config state, preview method |

## Files to Create/Modify

### Create
- `src/ml4t/backtest/execution/rebalancer.py` (~150 lines)
- `tests/execution/test_rebalancer.py` (~200 lines)

### Modify
- `src/ml4t/backtest/execution/__init__.py` (add 2 exports)
- `src/ml4t/backtest/__init__.py` (add 2 exports + __all__)

## Next Steps

Ready to implement. Estimated effort: ~350 lines across 4 files.

**Recommendation**: Run `/plan` or proceed directly to implementation.
