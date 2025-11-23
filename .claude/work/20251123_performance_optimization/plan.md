# Implementation Plan: ml4t.backtest Performance Optimization

## Overview

**Objective**: Optimize event-driven backtester from 90s to <12s (3-4x improvement)
**Baseline**: 500 assets × 10yr daily = 1.26M events
**Target**: Runtime <12s, Memory ≤200MB, 100% API compatibility
**Estimated Effort**: 15 hours across 11 tasks

## Profiling Summary (Validated)

| Bottleneck | Time | Calls | Fix |
|------------|------|-------|-----|
| dict.get | 17.3s | 24M | Array indexing |
| Polars iter_rows | 11.0s | 2.78M | Pre-materialize |
| engine.run loop | 10.6s | - | Array views |
| broker._execute_fill | 10.1s | 252K | SOA storage |
| Enum __eq__ | 7.3s | 5.66M | Int flags |

## Task Dependency Graph

```
TASK-001 (asset ID mapping)
    ├── TASK-002 (NumPy arrays)
    │       └── TASK-003 (engine refactor) ──┐
    └── TASK-004 (broker arrays) ────────────┤
            └── TASK-005 (get_position) ─────┤
                                             │
TASK-008 (int flags) ────────────────────────┤
TASK-009 (__slots__) ────────────────────────┤
                                             │
            ┌────────────────────────────────┘
            ▼
    TASK-006 (vectorize stops)
            └── TASK-007 (vectorize takes)
                    └── TASK-010 (validation)
                            └── TASK-011 (docs)
```

## Phase 1: Foundation (4 hours)

### TASK-001: Add asset ID mapping to DataFeed (1h)
- Create `asset_to_idx: dict[str, int]` at init
- Create `assets: list[str]` for reverse lookup
- No behavior change, just preparation

### TASK-002: Pre-materialize OHLCV as NumPy arrays (2h)
- Pivot Polars → NumPy `(n_bars, n_assets)` arrays
- Add `opens`, `highs`, `lows`, `closes`, `volumes`
- Add `timestamps` array
- Keep existing iteration working

### TASK-003: Refactor Engine to use array slicing (3h) ⚠️ HIGH RISK
- Replace dict creation with array views in main loop
- Pass `t_idx` to broker alongside timestamp
- Broker receives array slices, not dicts
- **Most invasive change - comprehensive testing required**

## Phase 2: Broker Internals (3 hours)

### TASK-004: Add parallel arrays for position state (2h)
- Add `pos_quantities: np.ndarray (n_assets,)`
- Add `pos_entry_prices: np.ndarray (n_assets,)`
- Dual-write: update both on every fill
- Transition period: keep Position dict working

### TASK-005: Migrate get_position() to arrays (1h)
- Reconstruct Position object on-demand from arrays
- Remove Position dict dependency
- User API unchanged: `broker.get_position("AAPL")` still works

## Phase 3: Vectorized Risk Checks (3 hours)

### TASK-006: Vectorize stop-loss evaluation (2h)
```python
# Before: O(n_positions) iteration
for pos in positions:
    if current_low < pos.entry_price * 0.95:
        exit(pos)

# After: O(1) vectorized + O(triggered) iteration
hit_stops = (lows < entry_prices * 0.95) & (quantities != 0)
for idx in np.where(hit_stops)[0]:
    exit(idx)
```

### TASK-007: Vectorize take-profit evaluation (1h)
- Same pattern as stop-loss
- Unified approach for all position rules

## Phase 4: Micro-optimizations (1.5 hours)

### TASK-008: Replace Enum comparisons with int flags (1h)
```python
# Hot path uses int
ORDER_SIDE_BUY = 1
ORDER_SIDE_SELL = -1

# Public API still accepts Enum
def submit_order(self, side: OrderSide | int, ...):
    side_int = side.value if isinstance(side, OrderSide) else side
```

### TASK-009: Add __slots__ to hot-path dataclasses (0.5h)
- Order, Fill, PositionAction classes
- Reduces memory footprint and attribute access time

## Phase 5: Validation & Polish (1.5 hours)

### TASK-010: Run full validation suite (1h)
- All 206 unit tests
- All 4 framework validations:
  - VectorBT Pro: EXACT MATCH
  - VectorBT OSS: EXACT MATCH
  - Backtrader: EXACT MATCH
  - Zipline: EXACT MATCH
- Benchmark: <12s for daily_baseline

### TASK-011: Update documentation (0.5h)
- Document SOA pattern in code
- Update PROJECT_MAP.md
- Record architecture decisions

## Critical Path

```
001 → 002 → 003 → 004 → 005 → 006 → 007 → 010 → 011
 1h    2h    3h    2h    1h    2h    1h    1h   0.5h = 13.5h
```

## Parallel Opportunities

Tasks that can run in parallel:
- TASK-008, TASK-009 (independent, can do anytime)
- TASK-002 || TASK-004 (after TASK-001)

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Engine refactor breaks tests | Medium | High | Run tests after each file change |
| Float precision drift | Low | Medium | Use float64 throughout |
| Index off-by-one bugs | Medium | High | Add boundary tests |
| API breakage | Medium | High | Keep adapter layer |

## Success Criteria

- [ ] Runtime < 12s (from 90s baseline)
- [ ] Memory ≤ 200MB
- [ ] All 206 unit tests pass
- [ ] All 4 framework validations EXACT MATCH
- [ ] Public API unchanged

## Next Steps

Run `/workflow:next` to begin TASK-001 (asset ID mapping).
