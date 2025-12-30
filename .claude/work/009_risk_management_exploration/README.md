# Risk Management Architecture Exploration

## Overview

This work unit contains a **comprehensive exploration and design proposal** for implementing context-dependent risk management in the ml4t.backtest event-driven backtesting engine.

**Output**: 2,055 lines of detailed analysis, design, API specifications, and implementation roadmap across 3 documents.

---

## Start Here

### Quick Start (5 minutes)
1. Read this README (you're here!)
2. Open `summary.md` for executive overview
3. Check the **Key Findings** section below

### For Implementation (30 minutes)
1. Read `summary.md` (5 min)
2. Read `exploration.md` Section 6 (Roadmap) (10 min)
3. Read `exploration.md` Section 7 (Detailed Specs) (15 min)

### For Design Review (45 minutes)
1. Read `exploration.md` Executive Summary (5 min)
2. Read Section 3 (Design Space Analysis) (15 min)
3. Read Section 7 (Detailed Specifications) (15 min)
4. Read Section 9 (Trade-offs) (10 min)

---

## What's Inside

### Documents

**exploration.md** (1,608 lines)
- Complete analysis with all findings
- 11 detailed sections covering every aspect
- Code examples, API signatures, test scenarios
- Implementation roadmap with phasing

**summary.md** (129 lines)
- Executive summary
- Key findings and recommendations
- Quick reference tables
- What's next (Phase 1 actions)

**INDEX.md** (318 lines)
- Navigation guide
- Cross-references by topic
- Quick lookup tables
- Document structure

---

## Key Findings

### Recommended Architecture

**Hybrid Approach**: RiskManager component + composable RiskRule implementations

```
RiskManager (orchestrator)
    ├─ RiskRule (abstract)
    │   ├─ VolatilityScaledStopLoss
    │   ├─ TimeBasedExit
    │   ├─ RegimeDependentRule
    │   ├─ MaxDailyLossRule
    │   └─ DynamicTrailingStop (+ custom user rules)
    ├─ RiskContext (immutable state snapshot)
    └─ RiskDecision (rule output)
```

### Why This Design

1. **Clean Separation** - Rules are pure functions of RiskContext
2. **Composable** - Stack multiple rules for complex logic
3. **Testable** - Evaluate rules in isolation
4. **Backward Compatible** - Optional feature, no breaking changes
5. **Extensible** - Users implement custom RiskRule subclasses

### Critical Integration Points

Three hook points in the event loop:

1. **Position Exit Checking** (HOOK C)
   - Every market event, check all open positions
   - Trigger exits for violated rules
   - Before strategy signal generation

2. **Order Validation** (HOOK B) ⭐ PRIMARY
   - Before broker.submit_order()
   - Validate/modify strategy orders
   - Apply risk constraints

3. **Fill Recording** (HOOK D)
   - After fill event generated
   - Update rule state (MFE/MAE, bar count)
   - Enable time-based rule tracking

### What's Missing

11 major gaps identified:

**HIGH Priority (4)**:
- No RiskContext abstraction
- No volatility-scaled TP/SL beyond bracket orders
- No regime-dependent rules
- No time-based exit infrastructure

**MEDIUM Priority (5)**:
- No spread-aware slippage models
- No volume-aware participation rates
- No order-type-dependent costs
- No portfolio-level constraints
- No rule composition framework

**LOW Priority (2)**:
- No configuration DSL (deferred to Phase 6)
- Limited advanced slippage models

---

## Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1)
- RiskContext (immutable dataclass)
- RiskRule abstract base + RiskDecision
- RiskManager skeleton (3 main methods)
- 3 engine hook points
- 5-10 basic rules
- 100+ unit tests
- **Effort**: 1-2 weeks

### Phase 2-6: Progressive Enhancement
- Position monitoring, order validation, slippage models, advanced rules, docs
- **Total**: 7 weeks for complete feature set
- **Effort**: 1-2 weeks per phase
- **Lines of Code**: ~1,700

### Key Dependencies
```
Phase 1 (Core)
    ↓
Phase 2 (Position Monitoring)
    ↓
Phase 3 (Order Validation)
    ↓
Phases 4-5 (Parallel: Slippage + Rules)
    ↓
Phase 6 (Documentation)
```

---

## API Preview

### User Creates Risk Rules

```python
from ml4t.backtest.execution.risk import (
    RiskManager,
    VolatilityScaledStopLoss,
    TimeBasedExit,
)

class MyStrategy(Strategy):
    def on_start(self):
        self.risk_manager = RiskManager()
        self.risk_manager.register_rule(
            VolatilityScaledStopLoss(atr_multiplier=2.0)
        )
        self.risk_manager.register_rule(
            TimeBasedExit(max_bars=20)
        )

    def on_market_event(self, event):
        # Strategy just generates entries
        # Exits handled by RiskManager
        if self.should_trade(event):
            order = Order(asset_id=event.asset_id, ...)
            self.broker.submit_order(order)
```

### Rules Are Pure Functions

```python
class TimeBasedExit(RiskRule):
    def __init__(self, max_bars: int):
        self.max_bars = max_bars

    def evaluate(self, context: RiskContext) -> RiskDecision:
        if context.entry_bars_ago >= self.max_bars:
            return RiskDecision(
                should_exit=True,
                reason=f"Max bars ({self.max_bars}) exceeded"
            )
        return RiskDecision()
```

---

## Backward Compatibility

**100% backward compatible**:
1. RiskManager is optional (not injected by default)
2. Bracket orders continue to work as-is
3. All new features in new namespaces
4. Engine accepts optional risk_manager parameter
5. Strategy continues to work without risk_manager

---

## What's Not Included

### Deferred to Later Phases
- Configuration DSL (YAML/JSON) → Phase 6
- Advanced slippage models → Phase 4
- Partial profit-taking rules → Phase 5
- Multi-asset optimization → Post-Phase 1

### Out of Scope
- ML rule inference
- Real-time rule adaptation
- Web UI for rule definition
- Database persistence

---

## Document Navigation

### By Task

**"I need to understand the current state"**
→ Read: Section 1 (Current Architecture Map)

**"I need to implement Phase 1"**
→ Read: Section 7 (Detailed Design Specs)

**"I need to convince someone"**
→ Read: Section 3 (Design Space) + Section 9 (Trade-offs)

**"I need concrete examples"**
→ Read: Section 5 (API Sketch) + Section 10 (Validation Scenarios)

**"I need to estimate effort"**
→ Read: Section 6 (Roadmap) + Section 8 (Complexity Estimates)

### By Document

**summary.md** - Quick overview (5 min read)
**exploration.md** - Complete analysis (30 min read)
**INDEX.md** - Navigation and reference (lookup only)

---

## Key Sections in exploration.md

| Section | Topic | Key Takeaway |
|---------|-------|--------------|
| 1 | Current Architecture | Event-driven design, existing capabilities |
| 2 | Integration Points | 3 optimal hook points identified |
| 3 | Design Space | 5 approaches evaluated, hybrid selected |
| 4 | Gap Analysis | 11 gaps with priorities and complexity |
| 5 | API Sketch | 4 usage patterns, 5 rule examples |
| 6 | Roadmap | 7-week phased plan with dependencies |
| 7 | Detailed Specs | Complete interfaces and data structures |
| 8 | Complexity | ~1,700 LOC, 7 weeks total effort |
| 9 | Trade-offs | Design decisions with rationale |
| 10 | Scenarios | 5 test cases validating features |
| 11 | Risk Assessment | Risks and mitigation strategies |

---

## Next Action

**Recommended**: Begin Phase 1 implementation

Phase 1 deliverables:
1. RiskContext class (immutable dataclass)
2. RiskRule abstract base + RiskDecision
3. RiskManager skeleton (3 main methods)
4. 3 engine hook point integrations
5. 5-10 basic rule implementations
6. 100+ unit tests

**Estimated effort**: 1-2 weeks

**Dependencies**: None (standalone component)

---

## Design Principles

This exploration followed these core principles:

1. **Simplicity First** - Simple path for basic strategies (bracket orders)
2. **Extensibility** - Advanced path for complex rules via composable RiskRule
3. **Separation of Concerns** - Rules decoupled from engine via RiskContext
4. **Testability** - Rules are pure functions, easily unit-tested
5. **Backward Compatibility** - No breaking changes, optional feature
6. **Event-Driven Alignment** - Leverages existing Clock/event architecture

---

## Questions?

See **INDEX.md** for cross-references or **exploration.md** table of contents.

---

**Generated**: 2025-11-17
**Work Unit**: 009_risk_management_exploration
**Status**: COMPLETED ✓
**Next**: Phase 1 Implementation
