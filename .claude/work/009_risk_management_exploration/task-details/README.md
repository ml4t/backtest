# Task Details Index

This directory contains detailed specifications for all 63 tasks in the risk management implementation.

## Quick Reference

**Total Tasks**: 63
**Total Effort**: 260 hours (7 weeks)
**Phases**: 6

## Task Organization

### Phase 1: Core Infrastructure (16 tasks, 40 hours)

**Foundation** (TASK-001 to TASK-005):
- TASK-001: RiskContext dataclass (3h) - Immutable state snapshot with 25+ fields
- TASK-002: RiskDecision dataclass (2h) - Rule output with exit signals
- TASK-003: RiskRule abstract base (2h) - Abstract protocol for rules
- TASK-004: PositionTradeState tracking (2h) - Position entry/MFE/MAE tracking
- TASK-005: RiskManager skeleton (4h) - Orchestrator with 4 core methods

**Engine Integration** (TASK-006 to TASK-009):
- TASK-006: Hook C - check_position_exits (3h) - Position exit checking before strategy
- TASK-007: Hook B - validate_order (3h) - Order validation after strategy
- TASK-008: Hook D - record_fill (2h) - Fill state recording after broker
- TASK-009: Feature flag and backward compat (4h) - Gradual rollout support

**Basic Rules** (TASK-010 to TASK-013):
- TASK-010: TimeBasedExit rule (2h) - Exit after max bars
- TASK-011: FunctionRule wrapper (2h) - Callable adapter
- TASK-012: PriceBasedStopLoss rule (2h) - Simple price-based SL
- TASK-013: PriceBasedTakeProfit rule (2h) - Simple price-based TP

**Testing** (TASK-014 to TASK-016):
- TASK-014: Unit tests for RiskContext/RiskDecision (4h) - 35+ tests, 80% coverage
- TASK-015: Unit tests for RiskManager (4h) - Core method testing
- TASK-016: Integration tests for engine hooks (4h) - End-to-end validation

---

### Phase 2: Position Monitoring (9 tasks, 40 hours)

**Core Implementation** (TASK-017 to TASK-021):
- TASK-017: check_position_exits() implementation (6h) - Complete exit checking logic
- TASK-018: Bar counter and age tracking (3h) - entry_bars_ago tracking
- TASK-019: MFE tracking (2h) - Max favorable excursion
- TASK-020: MAE tracking (2h) - Max adverse excursion
- TASK-021: SessionEndExit rule (4h) - Session-aware exits

**Testing** (TASK-022 to TASK-025):
- TASK-022: Scenario test - Time-based exit (3h) - End-to-end validation
- TASK-023: Scenario test - MFE/MAE tracking (3h) - Verify tracking accuracy
- TASK-024: Scenario test - Session exit (3h) - Session boundary testing
- TASK-025: Performance benchmarks (4h) - Overhead < 5% for 100 positions

---

### Phase 3: Order Validation (9 tasks, 40 hours)

**Core Implementation** (TASK-026 to TASK-031):
- TASK-026: validate_order() implementation (6h) - Complete validation logic
- TASK-027: MaxDailyLossRule (4h) - Daily loss limit constraint
- TASK-028: MaxDrawdownRule (4h) - Drawdown limit constraint
- TASK-029: MaxLeverageRule (4h) - Leverage limit constraint
- TASK-030: MaxPositionSizeRule (4h) - Position size limits
- TASK-031: Rule priority and conflict resolution (4h) - Priority system

**Testing** (TASK-032 to TASK-034):
- TASK-032: Integration test - Order validation (4h) - End-to-end flow
- TASK-033: Scenario test - Portfolio constraints (4h) - Constraint validation
- TASK-034: Scenario test - Rule conflicts (3h) - Priority resolution

---

### Phase 4: Slippage Enhancement (6 tasks, 20 hours) **CAN RUN PARALLEL**

**Refactoring** (TASK-035 to TASK-036):
- TASK-035: Refactor FillSimulator (6h) - **HIGH RISK** - Accept MarketEvent
- TASK-036: Backward compatibility tests (4h) - **CRITICAL** - 100+ tests

**New Models** (TASK-037 to TASK-040):
- TASK-037: SpreadAwareSlippage (4h) - Bid-ask spread modeling
- TASK-038: Enhanced VolumeAwareSlippage (4h) - Participation rate models
- TASK-039: OrderTypeDependentSlippage (4h) - Different slippage by order type
- TASK-040: Integration tests (4h) - Validate new models

---

### Phase 5: Advanced Rules (13 tasks, 80 hours)

**Volatility-Scaled** (TASK-041 to TASK-042):
- TASK-041: VolatilityScaledStopLoss (4h) - ATR-based stop loss
- TASK-042: VolatilityScaledTakeProfit (4h) - ATR-based take profit

**Regime-Dependent** (TASK-043):
- TASK-043: RegimeDependentRule (6h) - Different rules by market regime

**Dynamic Logic** (TASK-044 to TASK-046):
- TASK-044: DynamicTrailingStop (4h) - Trailing stop that tightens
- TASK-045: BreakEvenRule (3h) - Move SL to break-even after threshold
- TASK-046: PartialProfitTaking (6h) - Scale out at profit levels

**Composition** (TASK-047 to TASK-048):
- TASK-047: CompositeRule (6h) - Stack multiple rules (AND/OR/PRIORITY)
- TASK-048: Rule conflict detection (4h) - Diagnostics and warnings

**Testing** (TASK-049 to TASK-053):
- TASK-049: Scenario test - Volatility-scaled (4h) - Validate ATR adaptation
- TASK-050: Scenario test - Regime-dependent (4h) - Regime transitions
- TASK-051: Scenario test - Dynamic trailing (4h) - Trailing stop behavior
- TASK-052: Scenario test - Partial exits (4h) - Scale-out validation
- TASK-053: Performance benchmarks (4h) - Overhead < 10% for 10-rule combo

---

### Phase 6: Configuration & Documentation (10 tasks, 40 hours)

**Configuration** (TASK-054 to TASK-056):
- TASK-054: YAML config loader (6h) - Declarative rule specification
- TASK-055: JSON config loader (3h) - Alternative to YAML
- TASK-056: Config validation schema (4h) - Pydantic validation

**Examples** (TASK-057 to TASK-060):
- TASK-057: Notebook - Volatility-scaled stops (4h) - Complete example
- TASK-058: Notebook - Regime-dependent rules (4h) - VIX-based regimes
- TASK-059: Notebook - Time-based exits (4h) - Holding period strategies
- TASK-060: Notebook - Advanced slippage (4h) - Spread/volume models

**Documentation** (TASK-061 to TASK-063):
- TASK-061: Migration guide (4h) - **HIGH PRIORITY** - Bracket order migration
- TASK-062: API documentation (6h) - Complete API reference
- TASK-063: User tutorials (6h) - Quickstart + advanced tutorials

---

## Task Status Legend

- **pending**: Not started
- **in_progress**: Currently working on
- **completed**: Finished and validated

## Priority Levels

- **critical**: Must complete, blocks other work
- **high**: Important, core feature
- **medium**: Valuable, enhances functionality
- **low**: Nice-to-have, can defer

## Risk Levels

- **high**: TASK-035 (FillSimulator refactor), TASK-036 (backward compat)
- **medium**: TASK-006-008 (engine hooks), TASK-031 (conflict resolution)
- **low**: Most other tasks

## Parallel Execution Opportunities

**Week 1 (Phase 1)**:
- TASK-001-004 can run in parallel (data structures)
- TASK-010-013 can run in parallel (basic rules)

**Week 2-3 (Phases 2-3)**:
- Phase 2 and Phase 4 can run in parallel (independent)
- TASK-027-030 can run in parallel (constraint rules)

**Week 4-5 (Phase 5)**:
- TASK-041-042 can run in parallel (volatility-scaled)
- TASK-044-046 can run in parallel (dynamic rules)

**Week 6-7 (Phase 6)**:
- TASK-054-056 can run in parallel (config loaders)
- TASK-057-060 can run in parallel (notebooks)

---

## Critical Path

**Longest sequential dependency chain** (~37 hours):
1. TASK-001-005: Data structures and RiskManager (11h)
2. TASK-006-008: Engine integration (8h)
3. TASK-017: check_position_exits() (6h)
4. TASK-041: VolatilityScaledStopLoss (4h)
5. TASK-049: Scenario test (4h)
6. TASK-061: Migration guide (4h)

---

## Using This Index

1. **Find task by ID**: Use the phase sections above
2. **Check dependencies**: See `state.json` for full dependency graph
3. **View acceptance criteria**: Each task has 3-7 acceptance criteria in `state.json`
4. **Estimate effort**: Hour estimates provided for each task
5. **Track progress**: Use `/next` command to execute tasks in order

---

## Next Steps

1. Review this index to understand task organization
2. Check `state.json` for complete task specifications
3. Read `implementation-plan.md` for comprehensive context
4. Run `/next` to start TASK-001 (RiskContext dataclass)

---

**Total Estimated Code**: ~1,700 lines
**Total Estimated Tests**: ~400 tests
**Target Coverage**: 80%+
**Timeline**: 7 weeks with single developer
**Parallelization**: Can reduce to ~5 weeks with 2 developers
