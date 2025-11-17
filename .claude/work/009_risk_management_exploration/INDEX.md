# Risk Management Exploration - Document Index

## Overview
Complete exploration of requirements for implementing context-dependent risk management in ml4t.backtest event-driven backtesting engine.

**Total Content**: 1,608 lines (exploration.md) + 170 lines (summary.md) + this index

---

## Documents in This Work Unit

### 1. exploration.md (51 KB, 1,608 lines)
**Complete comprehensive analysis** with all findings and design details.

**Sections**:
1. **Executive Summary** - High-level findings and recommendations
2. **Current Architecture Map** (Section 1) - Components, event flow, capabilities
3. **Integration Points Analysis** (Section 2) - Where to hook into event loop
4. **Design Space Analysis** (Section 3) - Evaluated 5 approaches, selected hybrid
5. **Gap Analysis** (Section 4) - 11 gaps identified with priorities
6. **API Sketch** (Section 5) - Core interfaces and 5 example scenarios
7. **Next Steps** (Section 6) - Implementation roadmap and dependencies
8. **Detailed Design Specs** (Section 7) - Complete interface definitions
9. **Complexity Estimates** (Section 8) - Lines of code and effort
10. **Trade-offs** (Section 9) - Design decisions with rationale
11. **Validation Scenarios** (Section 10) - Test cases for each feature
12. **Risk Assessment** (Section 11) - Risks and mitigation strategies

### 2. summary.md (4.6 KB, 170 lines)
**Executive summary** for quick reference.

**Contents**:
- What was done (6 major analysis activities)
- Key findings (architecture, why this design, integration points)
- Phased rollout table
- What's next (Phase 1 implementation steps)
- Quick reference tables

### 3. INDEX.md (this file)
Navigation guide and document structure.

---

## Quick Navigation

### By Topic

**Architecture & Design**:
- Executive Summary (exploration.md, top)
- Section 1: Current Architecture Map
- Section 3: Design Space Analysis (5 approaches)
- Section 7: Detailed Design Specifications
- Section 8: Complexity Estimates

**Implementation & Integration**:
- Section 2: Integration Points Analysis (3 hook points)
- Section 6: Next Steps & Roadmap
- Section 8: Complexity Estimates
- Section 9: Trade-offs and Design Decisions

**Requirements & Gaps**:
- Section 4: Gap Analysis (11 gaps)
- Section 5: API Sketch & Example Scenarios
- Section 10: Validation Scenarios

**Risk & Decisions**:
- Section 9: Trade-offs
- Section 11: Risk Assessment

### By Use Case

**"I need to understand the current state"**:
→ Read: Executive Summary + Section 1 (Current Architecture)

**"I need to implement Phase 1"**:
→ Read: Section 7 (Detailed Design Specs) + Section 6 (Implementation Roadmap)

**"I need to convince someone this approach works"**:
→ Read: Section 3 (Design Space) + Section 9 (Trade-offs) + Section 11 (Risk Assessment)

**"I need concrete API examples"**:
→ Read: Section 5 (API Sketch) + Section 10 (Validation Scenarios)

**"I need to estimate effort"**:
→ Read: Section 6 (Roadmap - phasing) + Section 8 (Complexity Estimates)

---

## Key Concepts Reference

### RiskManager (Orchestrator Component)
**Purpose**: Central point for risk rule management
**Methods**:
- `register_rule(rule)` - Add a rule to evaluate
- `check_position_exits(market_event, broker, portfolio)` - Monitor exits
- `validate_order(order, context)` - Validate strategy orders
- `record_fill(fill_event, market_event)` - Track position state

**File Location** (to be created): `src/ml4t/backtest/execution/risk_manager.py`

### RiskRule (Abstract Rule Interface)
**Purpose**: Implement specific risk logic
**Method**: `evaluate(context) -> RiskDecision`
**Examples Provided**:
- TimeBasedExit (exit after N bars)
- VolatilityScaledStopLoss (SL = entry ± k×ATR)
- RegimeDependentRule (different params per regime)
- MaxDailyLossRule (halt trading if daily loss > threshold)
- DynamicTrailingStop (trail tightens over time)

**File Location** (to be created): `src/ml4t/backtest/execution/risk_rules/`

### RiskContext (Immutable State Snapshot)
**Purpose**: Provide rules with read-only state
**Contains**: Market state, position state, portfolio state, features
**Immutability**: Rules cannot modify system state

**File Location** (to be created): `src/ml4t/backtest/execution/risk_context.py`

### RiskDecision (Rule Output)
**Purpose**: Communicate rule decision
**Contains**: should_exit, exit_type, update_tp/sl, reason
**Usage**: Broker/engine act on decisions

**File Location** (to be created): `src/ml4t/backtest/execution/risk_decision.py`

---

## Implementation Roadmap Summary

### Phase 1: Core Infrastructure (Week 1)
- RiskContext (immutable dataclass)
- RiskRule (abstract) + RiskDecision
- RiskManager (skeleton)
- 3 engine hook points
- 5-10 basic rules
- Unit tests

### Phase 2: Position Monitoring (Week 2)
- RiskManager.check_position_exits()
- Bar counting
- Session awareness
- TimeBasedExit rules

### Phase 3: Order Validation (Week 3)
- RiskManager.validate_order()
- Portfolio constraints
- Position sizing rules

### Phase 4: Slippage Enhancement (Week 4)
- Refactor FillSimulator
- Spread-aware models
- Volume-aware models

### Phase 5: Advanced Rules (Weeks 5-6)
- Volatility-scaled rules
- Regime-dependent rules
- Dynamic trailing
- Partial profit-taking

### Phase 6: Configuration & Docs (Week 7)
- Config DSL (YAML/JSON)
- Examples
- Migration guide
- Documentation

---

## Critical Design Decisions

### 1. Hybrid RiskManager + RiskRule Pattern
**Decision**: Separate component (RiskManager) + composable rules (RiskRule)
**Why**: Clean separation, extensibility, testability
**Alternative Rejected**: Mixins pattern (poor reusability)

### 2. Immutable RiskContext Snapshots
**Decision**: Rules operate on frozen dataclasses
**Why**: Prevents side effects, enables composition
**Alternative Rejected**: Direct broker/portfolio access (tight coupling)

### 3. Three Hook Points in Event Loop
**Decision**: Position exits → Strategy signal → Order validation
**Why**: Exits checked before entries, minimal coupling
**Alternative Rejected**: Post-fill only (misses time-based exits)

### 4. Optional Component
**Decision**: RiskManager is optional, not mandatory
**Why**: 100% backward compatibility, no migration pressure
**Alternative Rejected**: Mandatory component (breaks existing code)

### 5. Rule Evaluation Per Market Event
**Decision**: Rules evaluated every market event
**Why**: Enables time-based and trailing exits
**Alternative Rejected**: Only on fill events (misses continuous monitoring)

---

## Key Metrics

### Estimated Implementation Effort
- **Total**: ~7 weeks for complete feature set
- **Core Infrastructure (Phase 1)**: 1-2 weeks
- **Per Phase**: 1-2 weeks
- **Lines of Code**: ~1,700 (core + basic rules)

### Risk Assessment
- **High Risk**: Integration breaks bracket orders (Low Likelihood)
- **Medium Risk**: Rules lag behind execution, state drift (Medium Likelihood)
- **Low Risk**: User confusion, performance (Low Likelihood)

### Architecture Metrics
- **Hook Points**: 3 (before exits, before validation, after fill)
- **Rule Examples**: 5 provided (time, volatility, regime, daily loss, trailing)
- **Backward Compatibility**: 100% (existing code unchanged)

---

## What's NOT Included (Deferred)

### Phase 2+ Features (Not in this exploration)
- Configuration DSL (YAML/JSON) - Phase 6
- Advanced slippage models - Phase 4
- Partial profit-taking rules - Phase 5
- Performance optimization - Post Phase 1

### Out of Scope (For future consideration)
- Machine learning rule inference
- Real-time rule adaptation
- Multi-asset rule composition
- Web UI for rule definition
- Database persistence of rules

---

## Dependencies & Context

### Required Reading
- ml4t.backtest PROJECT_MAP.md (project overview)
- ml4t.backtest CLAUDE.md (development guidelines)
- Risk management requirements document (.claude/reference/risk_management.md)

### Codebase Locations (References in Analysis)
- `src/ml4t/backtest/engine.py` - Event loop
- `src/ml4t/backtest/execution/broker.py` - Order execution
- `src/ml4t/backtest/portfolio/portfolio.py` - Position tracking
- `src/ml4t/backtest/core/event.py` - Event types
- `src/ml4t/backtest/execution/slippage.py` - Current models
- `src/ml4t/backtest/execution/commission.py` - Current models

### Related Work
- ML Signal Integration (Phase 1, already completed)
- Context Cache architecture (already implemented)

---

## How to Use This Exploration

### For Planning
1. Read summary.md for overview (5 min)
2. Read exploration.md Section 6 for roadmap (10 min)
3. Read Section 8 for complexity estimates (5 min)

### For Design Review
1. Read Section 3 for approach evaluation (15 min)
2. Read Section 7 for detailed specs (20 min)
3. Read Section 9 for trade-offs (10 min)

### For Implementation
1. Read Section 7 for complete API (20 min)
2. Read Section 5 for usage examples (15 min)
3. Read Section 10 for test scenarios (15 min)

### For Risk Assessment
1. Read Section 11 (10 min)
2. Review Section 9 trade-offs (10 min)
3. Check git history for related features

---

## Appendix: Section Contents Summary

| Section | Topic | Lines | Purpose |
|---------|-------|-------|---------|
| Exec Sum | Overview & findings | 25 | Quick understanding |
| 1 | Architecture map | 150 | Current state |
| 2 | Integration points | 200 | Hook locations |
| 3 | Design space | 250 | Approach evaluation |
| 4 | Gap analysis | 180 | Missing pieces |
| 5 | API sketch | 300 | Code examples |
| 6 | Roadmap | 150 | Implementation plan |
| 7 | Detailed specs | 250 | Complete interfaces |
| 8 | Complexity | 100 | Effort estimates |
| 9 | Trade-offs | 80 | Design decisions |
| 10 | Scenarios | 200 | Validation tests |
| 11 | Risk assessment | 80 | Risk/mitigation |
| Conclusion | Summary | 20 | Final thoughts |

---

## Questions This Exploration Answers

1. **What's the current architecture?** → Section 1
2. **Where should risk rules hook into the engine?** → Section 2
3. **What design approach is best?** → Section 3
4. **What features are missing?** → Section 4
5. **How do I write a risk rule?** → Section 5
6. **What's the implementation plan?** → Section 6
7. **What are the exact API signatures?** → Section 7
8. **How much work is this?** → Section 8
9. **Why this design vs alternatives?** → Section 9
10. **How do I validate it works?** → Section 10
11. **What could go wrong?** → Section 11

---

**Generated**: 2025-11-17
**Work Unit**: 009_risk_management_exploration
**Status**: COMPLETED ✓
