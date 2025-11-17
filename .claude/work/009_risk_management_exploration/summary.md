# Work Unit 009: Risk Management Exploration - Summary

**Status**: COMPLETED
**Date**: 2025-11-17
**Duration**: Exploration phase
**Output**: Comprehensive exploration report (1,608 lines)

## What Was Done

Conducted a very thorough exploration of requirements for implementing context-dependent risk management in ml4t.backtest, resulting in:

1. **Current Architecture Analysis**
   - Mapped all key components (engine, broker, portfolio, strategy)
   - Identified event flow and integration points
   - Catalogued existing risk management capabilities

2. **Gap Analysis**
   - 11 major gaps identified (vs 6 requirement areas)
   - Priority ranking: HIGH (4), MEDIUM (5), LOW (2)
   - Complexity assessment for each gap

3. **Design Space Exploration**
   - Evaluated 5 architectural approaches
   - Selected hybrid: RiskManager component + callable rules
   - Justified trade-offs and design decisions

4. **Integration Strategy**
   - Identified 3 optimal hook points in event loop
   - Pre-submission order validation (HOOK B) ⭐ PRIMARY
   - Continuous position exit checking (HOOK C) ⭐ PRIMARY
   - Fill state recording (HOOK D) SECONDARY

5. **API Design**
   - Core interfaces: RiskContext, RiskRule, RiskManager
   - 5 example rule implementations
   - 4 user-facing API patterns (simple → advanced)

6. **Implementation Roadmap**
   - 7-week phased delivery plan
   - 6 phases with clear dependencies
   - ~1,700 lines estimated code
   - 100% backward compatible

## Key Findings

### Architecture Recommendation
**Hybrid approach**: RiskManager component + composable RiskRule implementations

```
RiskManager (orchestrator)
    ├─ RiskRule (abstract)
    │   ├─ VolatilityScaledStopLoss
    │   ├─ TimeBasedExit
    │   ├─ RegimeDependentRule
    │   ├─ MaxDailyLossRule
    │   └─ DynamicTrailingStop
    ├─ RiskContext (immutable snapshot)
    └─ RiskDecision (output)
```

### Why This Design
1. **Clean separation**: Rules are pure functions of RiskContext
2. **Composable**: Stack multiple rules for complex logic
3. **Testable**: Rules evaluate in isolation with synthetic contexts
4. **Backward compatible**: Optional component, bracket orders unchanged
5. **Extensible**: Users implement custom RiskRule subclasses

### Critical Integration Points
1. **Before order submission**: Validate/modify orders (HOOK B)
2. **Every market event**: Monitor open positions (HOOK C)
3. **After fill**: Update rule state (HOOK D)

These minimize coupling while enabling comprehensive risk management.

### Phased Rollout
| Phase | Focus | Duration | Complexity |
|-------|-------|----------|-----------|
| 1 | Core infrastructure (RiskContext, RiskRule, RiskManager) | 1 week | Low |
| 2 | Position exit checking (time-based, continuous) | 1 week | Medium |
| 3 | Order validation (portfolio constraints) | 1 week | Medium |
| 4 | Slippage enhancement (spread/volume aware) | 1 week | Medium |
| 5 | Advanced rules (volatility-scaled, regime-dependent) | 2 weeks | Medium |
| 6 | Configuration & documentation | 1 week | Low |

## What's Next

**Phase 1 Implementation (Recommended)**:
1. Create RiskContext (immutable dataclass)
2. Create RiskRule (abstract base) + RiskDecision
3. Create RiskManager (skeleton with 3 core methods)
4. Add 3 hook points to BacktestEngine.run()
5. Implement 5-10 basic rules
6. 100+ unit tests

**Estimated effort**: 1-2 weeks of development

## File Location
- **Exploration Report**: `/home/stefan/ml4t/software/backtest/.claude/work/009_risk_management_exploration/exploration.md` (1,608 lines)
- **Summary**: This file

## Appendix: Quick Reference

### RiskContext (What rules see)
- Market state: OHLCV, bid/ask
- Position state: quantity, entry price/time, unrealized PnL
- Extreme prices: max favorable/adverse excursion
- Trade history: daily P&L, max loss, trades today
- Portfolio: equity, cash, leverage
- Features: user-provided (ATR, regime, volatility, etc.)

### RiskRule Interface
```python
class RiskRule(ABC):
    def evaluate(self, context: RiskContext) -> RiskDecision:
        # Return: should_exit, exit_type, update_tp/sl, reason
```

### Hook Points in Event Loop
1. **check_position_exits()** → exits for violated rules
2. **Strategy generates signal** → unchanged
3. **validate_order()** → modify/validate order
4. **Broker executes** → unchanged
5. **record_fill()** → update rule state

### API Progression
- **Simple**: Bracket orders with fixed %TP/SL (existing)
- **Intermediate**: Risk rules with volatility scaling (new)
- **Advanced**: Stacked rules + custom logic (new)
- **Future**: Configuration-driven rules (deferred)
