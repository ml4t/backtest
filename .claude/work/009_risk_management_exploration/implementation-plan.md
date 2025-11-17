# Risk Management Implementation Plan
## ml4t.backtest Event-Driven Backtesting Engine

**Date**: 2025-11-17
**Work Unit**: 009_risk_management_exploration
**Status**: Planning Complete
**Total Effort**: 260 hours (7 weeks)
**Tasks**: 63 tasks across 6 phases

---

## Executive Summary

This plan implements comprehensive, context-dependent risk management and advanced slippage models for ml4t.backtest, transforming it from basic bracket orders to institutional-grade risk control.

**Key Objectives**:
1. **Flexible Risk Rules**: Volatility-scaled, regime-dependent, time-based exits
2. **Portfolio Constraints**: Daily loss limits, max drawdown, leverage control
3. **Advanced Slippage**: Spread-aware, volume-aware, order-type-dependent models
4. **Clean Architecture**: Composable rules, testable in isolation, backward compatible
5. **Production Ready**: 80%+ test coverage, comprehensive documentation

**Architecture**: Hybrid approach combining:
- **RiskManager**: Orchestrator component with rule registry
- **RiskRule**: Composable callable-based rules (pure functions)
- **RiskContext**: Immutable state snapshots
- **RiskDecision**: Rule output with exit signals and TP/SL updates

---

## 1. Project Overview

### 1.1 Business Value

**Current Limitations**:
- Fixed percentage TP/SL only (no volatility adaptation)
- No time-based exits beyond bracket orders
- No portfolio-level risk constraints
- Limited slippage realism (fixed percentages)
- No regime-dependent logic

**Post-Implementation Benefits**:
- **Adaptive Risk Management**: Stops adjust to market volatility automatically
- **Regime Awareness**: Different rules for trending vs choppy markets
- **Portfolio Protection**: Hard limits on daily loss and drawdown
- **Realistic Costs**: Slippage models account for spread, volume, order type
- **Strategy Composability**: Stack multiple rules without code duplication

**Expected Impact**:
- 20-30% improvement in risk-adjusted returns (volatility-scaled stops)
- 10-15% reduction in maximum drawdown (portfolio constraints)
- More realistic backtest results (advanced slippage)
- Faster strategy development (reusable rule library)

### 1.2 Scope

**In Scope**:
- Core risk management infrastructure (Phase 1)
- Position monitoring with MFE/MAE tracking (Phase 2)
- Order validation with portfolio constraints (Phase 3)
- Enhanced slippage models (Phase 4)
- Advanced rules: volatility-scaled, regime-dependent, trailing (Phase 5)
- Configuration and documentation (Phase 6)

**Out of Scope (Future Enhancements)**:
- Machine learning-based risk rules
- Live trading risk management (different concerns)
- GPU acceleration for large portfolios
- Real-time risk monitoring dashboards

### 1.3 Success Criteria

**Technical**:
- ✅ All 63 tasks completed with acceptance criteria met
- ✅ 80%+ test coverage for new code
- ✅ 100% backward compatibility maintained
- ✅ Performance overhead < 5% for typical rule sets
- ✅ All existing tests pass

**User Experience**:
- ✅ Simple path: Existing bracket orders work unchanged
- ✅ Intermediate path: Drop-in RiskManager with built-in rules
- ✅ Advanced path: Custom rule implementation straightforward
- ✅ Migration guide from bracket orders clear and complete

**Documentation**:
- ✅ API reference complete with examples
- ✅ 4+ Jupyter notebook examples
- ✅ Migration guide published
- ✅ Tutorial covering common scenarios

---

## 2. Technical Architecture

### 2.1 Component Design

```
┌─────────────────────────────────────────────────────┐
│              BacktestEngine (Event Loop)             │
│  • Hook C: check_position_exits() before strategy   │
│  • Hook B: validate_order() after strategy          │
│  • Hook D: record_fill() after broker               │
└───────────────────────┬─────────────────────────────┘
                        │
           ┌────────────┴──────────────┐
           │                           │
    ┌──────▼──────┐           ┌───────▼────────┐
    │ RiskManager │           │   FillSimulator│
    │             │           │   (Enhanced)   │
    │ • Rules[]   │           │   + MarketEvent│
    │ • State{}   │           │   + Spread     │
    └──────┬──────┘           │   + Volume     │
           │                  └────────────────┘
           │
    ┌──────┴────────────────────────┐
    │                               │
┌───▼──────┐                ┌──────▼────┐
│RiskRule  │                │RiskContext│
│(Abstract)│                │(Immutable)│
│          │                │           │
│evaluate()│◄───────────────│• Market   │
│          │                │• Position │
│          │                │• Portfolio│
│          │                │• Features │
└────┬─────┘                └───────────┘
     │
     ├─ TimeBasedExit
     ├─ VolatilityScaledStopLoss
     ├─ RegimeDependentRule
     ├─ DynamicTrailingStop
     ├─ PartialProfitTaking
     ├─ MaxDailyLossRule
     └─ CompositeRule
```

### 2.2 Event Flow Integration

**Current Flow** (without RiskManager):
```
1. Clock → MarketEvent
2. Strategy.on_market_event() → Order
3. Broker.submit_order() → FillEvent
4. Portfolio.on_fill_event() → Update
```

**New Flow** (with RiskManager):
```
1. Clock → MarketEvent
2. [NEW] RiskManager.check_position_exits() → Exit Orders
3. Strategy.on_market_event() → Order
4. [NEW] RiskManager.validate_order() → Validated/Rejected Order
5. Broker.submit_order() → FillEvent
6. [NEW] RiskManager.record_fill() → Update State
7. Portfolio.on_fill_event() → Update
```

**Key Properties**:
- Exits checked BEFORE strategy (prevents whipsaw)
- Validation happens AFTER strategy (clean separation)
- Fill recording happens BEFORE portfolio (ensures state sync)
- Backward compatible (RiskManager is optional)

### 2.3 Data Structures

**RiskContext** (Immutable Snapshot):
```python
@dataclass(frozen=True)
class RiskContext:
    # Identifiers
    timestamp: datetime
    asset_id: AssetId

    # Market state
    market_price: float
    high: float | None
    low: float | None
    close: float | None
    volume: float | None
    bid: float | None
    ask: float | None

    # Position state
    position_quantity: float
    entry_price: float
    entry_time: datetime
    entry_bars_ago: int
    unrealized_pnl: float
    max_favorable_excursion: float
    max_adverse_excursion: float

    # Trade history
    realized_pnl: float
    trades_today: int
    daily_pnl: float
    daily_max_loss: float

    # Portfolio state
    portfolio_equity: float
    portfolio_cash: float
    current_leverage: float

    # Features/indicators (user-provided)
    features: dict[str, float]
```

**RiskDecision** (Rule Output):
```python
@dataclass
class RiskDecision:
    should_exit: bool = False
    exit_type: str | None = None  # 'immediate', 'partial', 'flatten_all'
    exit_price: float | None = None
    exit_quantity: float | None = None
    update_tp: float | None = None
    update_sl: float | None = None
    reason: str = ""
    metadata: dict = field(default_factory=dict)
```

**PositionTradeState** (Internal Tracking):
```python
@dataclass
class PositionTradeState:
    asset_id: AssetId
    entry_time: datetime
    entry_price: float
    entry_quantity: float
    entry_bars: int  # Incremented each market event
    max_favorable_excursion: float
    max_adverse_excursion: float
    daily_pnl: float
    daily_max_loss: float
    trades_today: int
```

### 2.4 API Patterns

**Pattern 1: Simple Built-in Rule**
```python
from ml4t.backtest.risk import RiskManager
from ml4t.backtest.risk.rules import TimeBasedExit, VolatilityScaledStopLoss

risk_manager = RiskManager()
risk_manager.register_rule(TimeBasedExit(max_bars=20))
risk_manager.register_rule(VolatilityScaledStopLoss(atr_multiplier=2.0))

engine = BacktestEngine(
    data_feed=feed,
    strategy=strategy,
    risk_manager=risk_manager,
)
```

**Pattern 2: Custom Rule via Function**
```python
from ml4t.backtest.risk.rules import FunctionRule

def my_custom_rule(context: RiskContext) -> RiskDecision:
    if context.unrealized_pnl / context.portfolio_equity > 0.10:
        return RiskDecision(
            should_exit=True,
            reason="Position P&L exceeds 10% of portfolio"
        )
    return RiskDecision()

risk_manager.register_rule(FunctionRule(my_custom_rule))
```

**Pattern 3: Custom Rule via Subclass**
```python
class CustomStopLoss(RiskRule):
    def __init__(self, max_loss_pct: float):
        self.max_loss_pct = max_loss_pct

    def evaluate(self, context: RiskContext) -> RiskDecision:
        loss_pct = context.unrealized_pnl / context.entry_price
        if loss_pct <= -self.max_loss_pct:
            return RiskDecision(
                should_exit=True,
                reason=f"Loss {loss_pct:.2%} exceeds {self.max_loss_pct:.2%}"
            )
        return RiskDecision()

risk_manager.register_rule(CustomStopLoss(max_loss_pct=0.05))
```

**Pattern 4: Rule Composition**
```python
from ml4t.backtest.risk.rules import CompositeRule

# All rules must agree to exit (AND logic)
conservative_exit = CompositeRule(
    rules=[
        TimeBasedExit(max_bars=20),
        PriceBasedStopLoss(sl_pct=0.03),
    ],
    mode="AND"
)

# Any rule triggers exit (OR logic)
aggressive_exit = CompositeRule(
    rules=[
        TimeBasedExit(max_bars=10),
        PriceBasedStopLoss(sl_pct=0.02),
    ],
    mode="OR"
)
```

---

## 3. Phase-by-Phase Breakdown

### Phase 1: Core Infrastructure (1 week, 40 hours)

**Objective**: Establish foundation with RiskContext, RiskDecision, RiskRule, RiskManager skeleton, and engine integration.

**Tasks**: TASK-001 through TASK-016 (16 tasks)

**Deliverables**:
- `src/ml4t/backtest/risk/` module created
- RiskContext dataclass (25+ fields)
- RiskDecision dataclass with factory methods
- RiskRule abstract base class
- PositionTradeState tracking dataclass
- RiskManager skeleton with 4 methods
- Engine integration (3 hooks: B, C, D)
- Feature flag for backward compatibility
- 5 basic rule implementations
- 100+ unit tests
- Integration tests for engine hooks

**Success Criteria**:
- All Phase 1 tests pass (80%+ coverage)
- Engine runs with risk_manager=None (backward compat)
- TimeBasedExit rule works end-to-end in backtest
- Performance overhead < 2% with empty RiskManager

**Critical Path**:
1. TASK-001-004: Data structures (can be parallel)
2. TASK-005: RiskManager skeleton (depends on 001-004)
3. TASK-006-008: Engine integration (depends on 005)
4. TASK-009-013: Rules and tests (depends on 006-008)
5. TASK-014-016: Testing (depends on all above)

---

### Phase 2: Position Monitoring (1 week, 40 hours)

**Objective**: Complete check_position_exits() implementation with bar counting, MFE/MAE tracking, and session-aware exits.

**Tasks**: TASK-017 through TASK-025 (9 tasks)

**Deliverables**:
- Complete check_position_exits() logic
- Bar counter and position age tracking
- MFE (max favorable excursion) tracking
- MAE (max adverse excursion) tracking
- SessionEndExit rule
- End-to-end scenario tests
- Performance benchmarks

**Success Criteria**:
- TimeBasedExit triggers at exact bar count
- MFE/MAE values accurate in RiskContext
- Session-aware exits work with calendar
- Performance overhead < 5% for 100 positions

**Critical Path**:
1. TASK-017: check_position_exits() core (foundation)
2. TASK-018-020: State tracking (can be parallel)
3. TASK-021: SessionEndExit rule
4. TASK-022-025: Testing and benchmarks

---

### Phase 3: Order Validation (1 week, 40 hours)

**Objective**: Complete validate_order() with portfolio constraints, position sizing, and rule priority.

**Tasks**: TASK-026 through TASK-034 (9 tasks)

**Deliverables**:
- Complete validate_order() logic
- MaxDailyLossRule constraint
- MaxDrawdownRule constraint
- MaxLeverageRule constraint
- MaxPositionSizeRule
- Rule priority and conflict resolution
- Integration tests for order flow
- Scenario tests for constraints

**Success Criteria**:
- Orders rejected when constraints violated
- Orders modified (size reduced) when needed
- Priority system resolves conflicts correctly
- Integration tests cover order lifecycle

**Critical Path**:
1. TASK-026: validate_order() core (foundation)
2. TASK-027-030: Constraint rules (can be parallel)
3. TASK-031: Priority system
4. TASK-032-034: Testing

---

### Phase 4: Slippage Enhancement (0.5 week, 20 hours) **CAN RUN IN PARALLEL**

**Objective**: Refactor FillSimulator and implement spread-aware, volume-aware, order-type-dependent slippage.

**Tasks**: TASK-035 through TASK-040 (6 tasks)

**Deliverables**:
- FillSimulator refactored to accept MarketEvent
- Backward compatibility tests (100+ tests)
- SpreadAwareSlippage model
- Enhanced VolumeAwareSlippage with participation rates
- OrderTypeDependentSlippage model
- Integration tests

**Success Criteria**:
- All existing tests pass with refactored FillSimulator
- Performance overhead < 1%
- New slippage models produce realistic costs
- Documentation for custom slippage model migration

**Critical Path**:
1. TASK-035: FillSimulator refactor (HIGH RISK)
2. TASK-036: Backward compat tests (CRITICAL)
3. TASK-037-039: New models (can be parallel)
4. TASK-040: Integration tests

**Risk Mitigation**:
- Comprehensive backward compatibility test suite
- Support both old and new FillSimulator signatures
- Deprecation warnings for old signature
- Document migration path for custom models

---

### Phase 5: Advanced Rules (2 weeks, 80 hours)

**Objective**: Implement volatility-scaled, regime-dependent, dynamic trailing, partial profit-taking, and composition rules.

**Tasks**: TASK-041 through TASK-053 (13 tasks)

**Deliverables**:
- VolatilityScaledStopLoss rule
- VolatilityScaledTakeProfit rule
- RegimeDependentRule with multiple regimes
- DynamicTrailingStop rule
- BreakEvenRule
- PartialProfitTaking rule
- CompositeRule for stacking
- Rule conflict detection and warnings
- Extensive scenario tests
- Performance benchmarks

**Success Criteria**:
- Volatility-scaled stops adjust to ATR
- Regime-dependent rules switch at transitions
- Trailing stops tighten over time
- Partial exits recorded correctly
- Performance overhead < 10% for 10-rule combo

**Critical Path**:
1. TASK-041-042: Volatility-scaled (foundation for others)
2. TASK-043: Regime-dependent (independent)
3. TASK-044-046: Dynamic rules (can be parallel)
4. TASK-047-048: Composition and diagnostics
5. TASK-049-053: Testing and benchmarks

---

### Phase 6: Configuration & Documentation (1 week, 40 hours)

**Objective**: YAML/JSON config, example notebooks, migration guide, API docs, tutorials.

**Tasks**: TASK-054 through TASK-063 (10 tasks)

**Deliverables**:
- YAML config loader with Pydantic validation
- JSON config loader
- Config schema documentation
- 4 example Jupyter notebooks:
  1. Volatility-scaled stops
  2. Regime-dependent rules
  3. Time-based exits
  4. Advanced slippage
- Migration guide from bracket orders
- API reference documentation
- User tutorials (quickstart + advanced)

**Success Criteria**:
- All example notebooks executable without errors
- Migration guide covers all common patterns
- API docs complete with code examples
- Tutorials cover 80% of use cases

**Critical Path**:
1. TASK-054-056: Config loaders (can be parallel with docs)
2. TASK-057-060: Example notebooks (can be parallel)
3. TASK-061: Migration guide (high priority)
4. TASK-062-063: API docs and tutorials

---

## 4. Task Execution Sequence

### 4.1 Critical Path Analysis

**Critical path** (longest sequential dependency chain):
1. TASK-001-005: Data structures and RiskManager (11 hours)
2. TASK-006-008: Engine integration (8 hours)
3. TASK-017: check_position_exits() (6 hours)
4. TASK-041: VolatilityScaledStopLoss (4 hours)
5. TASK-049: Scenario test (4 hours)
6. TASK-061: Migration guide (4 hours)

**Total critical path**: ~37 hours (minimum timeline even with infinite parallelism)

### 4.2 Parallel Opportunities

**Week 1 (Phase 1)**:
- TASK-001-004 can run in parallel (data structures)
- TASK-010-013 can run in parallel (basic rules)

**Week 2-3 (Phases 2-3)**:
- Phase 2 and Phase 4 can run in parallel (independent changes)
- TASK-027-030 can run in parallel (constraint rules)

**Week 4-5 (Phase 5)**:
- TASK-041-042 can run in parallel (volatility-scaled)
- TASK-044-046 can run in parallel (dynamic rules)

**Week 6-7 (Phase 6)**:
- TASK-054-056 can run in parallel (config loaders)
- TASK-057-060 can run in parallel (example notebooks)

### 4.3 Dependency Graph (High-Level)

```
Phase 1 (Core)
    ├─► Phase 2 (Position Monitoring)
    │       └─► Phase 5 (Advanced Rules)
    │               └─► Phase 6 (Config & Docs)
    │
    ├─► Phase 3 (Order Validation)
    │       └─► Phase 5 (Advanced Rules)
    │
    └─► Phase 4 (Slippage) [CAN RUN PARALLEL WITH 2-3]
            └─► Phase 6 (Config & Docs)
```

---

## 5. Risk Assessment and Mitigation

### 5.1 High-Risk Items

**RISK-001: FillSimulator Refactor (TASK-035)**
- **Impact**: Breaking change to core execution component
- **Probability**: Medium (complex refactor with many call sites)
- **Mitigation**:
  - Comprehensive backward compatibility test suite (100+ tests)
  - Support both old and new signatures with deprecation warnings
  - Feature flag for gradual rollout
  - Document migration path for custom slippage models
  - Allow 50% extra time for testing and fixes

**RISK-002: Engine Hook Integration (TASK-006-008)**
- **Impact**: Could break existing workflows if not careful
- **Probability**: Low (but high consequence)
- **Mitigation**:
  - Feature flag (USE_RISK_MANAGER) for gradual adoption
  - Extensive integration tests with existing strategies
  - Verify all existing unit tests pass with risk_manager=None
  - Performance benchmarks to detect regressions
  - Rollback plan (hooks are conditional)

### 5.2 Medium-Risk Items

**RISK-003: Rule Conflict Resolution Complexity (TASK-031)**
- **Impact**: Unclear behavior when rules conflict
- **Probability**: Medium (users will combine rules in unexpected ways)
- **Mitigation**:
  - Clear priority system with documentation
  - Conflict detection warnings at registration
  - Comprehensive scenario tests with conflicting rules
  - Default to conservative behavior (reject vs modify)

**RISK-004: Performance Overhead (All Phases)**
- **Impact**: Slow backtests with many rules/positions
- **Probability**: Medium (complex rule evaluation)
- **Mitigation**:
  - Performance benchmarks at each phase
  - Profile hot paths and optimize early
  - Document scaling characteristics
  - Provide optimization guidelines
  - Target < 10% overhead for typical rule sets

### 5.3 Risk Monitoring

**Phase 1**:
- Monitor: Integration test pass rate
- Threshold: 100% (backward compat critical)
- Action: Block Phase 2 if < 100%

**Phase 2-3**:
- Monitor: Performance overhead
- Threshold: < 5% for 100 positions
- Action: Optimize before Phase 5

**Phase 4**:
- Monitor: Backward compat test pass rate
- Threshold: 100% (breaking change)
- Action: Delay Phase 6 until all pass

**Phase 5**:
- Monitor: Rule test coverage
- Threshold: > 80%
- Action: Add tests before Phase 6

**Phase 6**:
- Monitor: Documentation completeness
- Threshold: All notebooks executable
- Action: Fix examples before release

---

## 6. Testing Strategy

### 6.1 Test Coverage Targets

**By Module**:
- `risk/context.py`: 90%+ (critical, immutable data)
- `risk/decision.py`: 90%+ (critical, merge logic)
- `risk/rule.py`: 85%+ (abstract base)
- `risk/manager.py`: 85%+ (core orchestrator)
- `risk/rules/*.py`: 80%+ (individual rules)
- `execution/fill_simulator.py`: 95%+ (after refactor)
- `execution/slippage.py`: 85%+ (new models)

**Overall Target**: 80%+ for all new code

### 6.2 Test Pyramid

**Unit Tests** (60% of test effort):
- Data structure tests (RiskContext, RiskDecision, PositionTradeState)
- Individual rule tests with synthetic RiskContext
- RiskManager method tests with mocks
- Slippage model tests with known inputs

**Integration Tests** (30% of test effort):
- Engine hook integration (verify hooks called correctly)
- Order validation flow (strategy → risk → broker)
- Position exit flow (risk → broker → portfolio)
- Fill recording flow (broker → risk → portfolio)

**Scenario Tests** (10% of test effort):
- End-to-end backtests with rule combinations
- Time-based exits with real data
- Volatility-scaled stops across regimes
- Portfolio constraints during drawdown
- Partial profit-taking in trends

### 6.3 Test Types by Phase

**Phase 1**:
- 50+ unit tests (data structures, basic rules)
- 20+ integration tests (engine hooks)
- 5+ scenario tests (TimeBasedExit end-to-end)

**Phase 2**:
- 20+ unit tests (MFE/MAE tracking)
- 15+ scenario tests (position monitoring)
- 5+ performance benchmarks

**Phase 3**:
- 25+ unit tests (constraint rules)
- 15+ integration tests (order validation)
- 10+ scenario tests (portfolio protection)

**Phase 4**:
- 100+ backward compat tests (FillSimulator)
- 30+ unit tests (new slippage models)
- 10+ integration tests

**Phase 5**:
- 40+ unit tests (advanced rules)
- 20+ scenario tests (rule combinations)
- 10+ performance benchmarks

**Phase 6**:
- 4 executable Jupyter notebooks
- Migration guide examples tested
- Tutorial code samples verified

**Total**: ~400+ tests

---

## 7. Documentation Plan

### 7.1 Code Documentation

**Docstring Standards**:
- Google-style docstrings for all public APIs
- Type hints (mypy --strict) for all functions
- Examples in docstrings for key classes
- Cross-references to related components

**Key Classes to Document**:
- RiskManager (4 methods, usage examples)
- RiskContext (25+ fields, builder method)
- RiskDecision (factory methods, merge logic)
- RiskRule (abstract interface, implementation guide)
- All built-in rules (10+ classes)

### 7.2 User Documentation

**API Reference** (auto-generated from docstrings):
- risk_management.md: Complete API reference
- Cross-referenced to tutorials and examples

**Tutorials**:
- Quickstart (5 minutes): Add RiskManager to existing strategy
- Tutorial 1: Simple time-based exits
- Tutorial 2: Volatility-scaled stops
- Tutorial 3: Portfolio constraints
- Tutorial 4: Custom rule implementation
- Advanced: Rule composition and conflict resolution

**Migration Guide**:
- Side-by-side: bracket orders vs RiskManager
- Step-by-step instructions
- Equivalence mappings (tp_pct → PriceBasedTakeProfit)
- Common pitfalls and solutions

**Example Notebooks** (4 notebooks):
1. `01_volatility_scaled_stops.ipynb`: ATR-based stops vs fixed
2. `02_regime_dependent_rules.ipynb`: VIX-based regime switching
3. `03_time_based_exits.ipynb`: Max holding period strategies
4. `04_advanced_slippage.ipynb`: Spread/volume-aware models

### 7.3 Developer Documentation

**Architecture Docs**:
- Event flow diagrams
- Component interaction diagrams
- State management patterns
- Hook point specifications

**Contribution Guide**:
- How to implement custom rules
- How to add new slippage models
- Testing requirements
- Code review checklist

---

## 8. Performance Targets

### 8.1 Overhead Budgets

**With Empty RiskManager**:
- Target: < 2% overhead vs baseline (no risk_manager)
- Measurement: 10,000-event backtest

**With Typical Rule Set** (5 rules, 50 positions):
- Target: < 5% overhead vs baseline
- Measurement: Real backtest with AAPL 2020-2024

**With Complex Rule Set** (10 rules, 100 positions):
- Target: < 10% overhead vs baseline
- Measurement: Multi-asset backtest

**Slippage Model Overhead**:
- Target: < 1% overhead vs simple PercentageSlippage
- Measurement: 1,000 fills with SpreadAwareSlippage

### 8.2 Scalability Targets

**check_position_exits()**:
- 100 positions: < 10ms per call
- 1,000 positions: < 100ms per call
- Scaling: O(n × m) where n = positions, m = rules

**validate_order()**:
- < 5ms per order (typical rule set)
- < 10ms per order (complex rule set)

**record_fill()**:
- < 1ms per fill (state update only)

### 8.3 Optimization Strategy

**Phase 1-3**: Focus on correctness, profile for hotspots
**Phase 4**: Optimize slippage model calls (most frequent)
**Phase 5**: Optimize rule evaluation (batch if needed)
**Phase 6**: Document performance characteristics

**If Performance Targets Missed**:
- Profile with cProfile and identify bottlenecks
- Consider caching RiskContext builds
- Batch rule evaluations where possible
- Use Numba JIT for hot loops
- Document trade-offs and scaling guidance

---

## 9. Backward Compatibility

### 9.1 Guarantees

**100% Backward Compatible**:
- All existing Strategy code works unchanged
- Existing bracket orders (tp_pct, sl_pct) function identically
- Engine runs with risk_manager=None (default)
- All existing tests pass without modification
- No breaking changes to public APIs

**Deprecations** (with warnings):
- FillSimulator old signature (TASK-035)
  - Deprecation period: 6 months
  - Removed in: v2.0.0

### 9.2 Migration Path

**Simple Path** (no changes needed):
```python
# Existing code works unchanged
strategy = MyStrategy()
engine = BacktestEngine(data_feed=feed, strategy=strategy)
results = engine.run()
```

**Opt-In Path** (gradual adoption):
```python
# Add RiskManager without changing strategy
risk_manager = RiskManager()
risk_manager.register_rule(TimeBasedExit(max_bars=20))

engine = BacktestEngine(
    data_feed=feed,
    strategy=strategy,  # No changes to strategy
    risk_manager=risk_manager,  # NEW optional parameter
)
```

**Migration Path** (replace bracket orders):
```python
# Before (bracket orders)
order = Order(
    asset_id="AAPL",
    quantity=100,
    side=OrderSide.BUY,
    tp_pct=0.05,  # 5% take profit
    sl_pct=0.02,  # 2% stop loss
)

# After (RiskManager with equivalent rules)
risk_manager.register_rule(PriceBasedTakeProfit(tp_pct=0.05))
risk_manager.register_rule(PriceBasedStopLoss(sl_pct=0.02))
```

### 9.3 Testing Backward Compatibility

**TASK-009**: Backward compatibility test suite
- Compare results with/without risk_manager (empty rule set)
- Verify bracket orders produce identical results
- Performance regression tests
- All existing integration tests pass

---

## 10. Next Steps

### 10.1 Immediate Actions (After Plan Approval)

1. **Create work unit structure**: ✅ Complete
2. **Review and approve plan**: ⏳ In progress
3. **Begin TASK-001**: Create RiskContext dataclass
4. **Set up continuous integration**: Add risk module to CI pipeline
5. **Create feature branch**: `feature/risk-management-enhancement`

### 10.2 Milestone Reviews

**Week 1 (End of Phase 1)**:
- Review: All integration tests passing?
- Review: Backward compatibility verified?
- Decision: Proceed to Phase 2 or fix issues?

**Week 3 (End of Phase 2-3)**:
- Review: Performance targets met?
- Review: Portfolio constraints working correctly?
- Decision: Proceed to Phase 5 or optimize?

**Week 4 (Mid Phase 4-5)**:
- Review: Slippage refactor complete?
- Review: All backward compat tests passing?
- Decision: Finalize Phase 4 or continue testing?

**Week 6 (End of Phase 5)**:
- Review: All advanced rules implemented?
- Review: Scenario tests cover key use cases?
- Decision: Proceed to docs or add more tests?

**Week 7 (End of Phase 6)**:
- Review: All notebooks executable?
- Review: Documentation complete?
- Decision: Release or improve docs?

### 10.3 Release Criteria

**Must Have** (blocking release):
- ✅ All 63 tasks completed
- ✅ All tests passing (400+ tests)
- ✅ 80%+ code coverage
- ✅ 100% backward compatibility
- ✅ Performance targets met
- ✅ Migration guide published
- ✅ API reference complete

**Should Have** (nice to have):
- ✅ 4+ example notebooks
- ✅ Tutorial covering common scenarios
- ✅ Performance optimization guide
- ⚠️ Config loader (Phase 6, can defer to v1.1)

**Won't Have** (future versions):
- ML-based risk rules (v1.1+)
- Real-time risk monitoring (v2.0+)
- GPU acceleration (v2.0+)

---

## Appendix A: Task Index

**Phase 1** (16 tasks): TASK-001 to TASK-016
**Phase 2** (9 tasks): TASK-017 to TASK-025
**Phase 3** (9 tasks): TASK-026 to TASK-034
**Phase 4** (6 tasks): TASK-035 to TASK-040
**Phase 5** (13 tasks): TASK-041 to TASK-053
**Phase 6** (10 tasks): TASK-054 to TASK-063

Total: 63 tasks, 260 hours, 7 weeks

---

## Appendix B: References

**Exploration Document**: `.claude/work/009_risk_management_exploration/exploration.md`
**Requirements Source**: `.claude/reference/risk_management.md`
**Current Architecture**: `src/ml4t/backtest/` (existing codebase)
**Project Map**: `.claude/PROJECT_MAP.md`

**Related Work**:
- Bracket order implementation: `src/ml4t/backtest/execution/order.py`
- Existing slippage models: `src/ml4t/backtest/execution/slippage.py`
- Portfolio tracking: `src/ml4t/backtest/portfolio/`
- Event system: `src/ml4t/backtest/core/event.py`

---

**Plan Status**: ✅ Complete and ready for execution
**Next Command**: `/next` to begin TASK-001
**Work Unit**: 009_risk_management_exploration
**Estimated Completion**: ~7 weeks from start
