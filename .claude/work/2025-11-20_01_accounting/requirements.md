# Requirements: Robust Accounting Logic

## Source
- Type: User requirement
- Reference: "Build robust accounting logic"
- Clarification: "The engine needs to handle both cash equities and margin accounts"
- Date: 2025-11-20

## Overview

The backtest engine currently lacks proper accounting constraints, allowing positions to execute with unlimited debt (cash going to -$652k). The engine must be refactored to support **both cash accounts and margin accounts** with proper constraint enforcement.

## Critical Bug Identified

**Current Issue**: Engine executes all orders without checking account constraints
- VectorBT (correct behavior): -$3,891 PnL (respects cash limits)
- Current engine (broken): -$652,580 PnL (executes into massive debt)
- Difference: 99.4%

**Root Cause**: Simple cash tracking without validation
```python
# Line 587 in engine.py
self.cash += cash_change  # Can go negative! No constraint check!
```

## Functional Requirements

### 1. Cash Account Support (Long-Only, No Leverage)

**Constraints:**
- Cash balance must remain >= 0 (hard floor)
- No short selling (all positions must be >= 0)
- No leverage (buying power = available cash)
- Order cost must not exceed available cash

**Behavior:**
- Orders rejected if cost > cash
- Exit-first sequencing to free cash for re-entry
- Position tracking with cost basis
- Commission/slippage properly deducted

**Use Case:** Most retail stock trading, simple backtesting scenarios

### 2. Margin Account Support (Long/Short, Leverage Allowed)

**Constraints:**
- Cash can be negative (margin debt allowed)
- Short selling allowed (positions can be negative)
- Leverage controlled by margin requirements
- New positions constrained by buying power (not just cash)

**Calculations:**
```
Net Liquidating Value (NLV) = Cash + Σ(position_qty × current_price)
Maintenance Margin (MM) = Σ(|position_qty × price| × maintenance_rate)
Buying Power (BP) = (NLV - MM) / initial_margin_rate
```

**Behavior:**
- Orders rejected if margin requirement > buying power
- Short positions tracked with negative quantities
- Position reversals (long→short) handled atomically
- Exit-first sequencing to release margin

**Use Case:** Institutional trading, hedge fund strategies, short-selling strategies

### 3. Common Requirements (Both Account Types)

**Position Management:**
- Track positions with weighted average cost basis
- Support partial position closes
- Calculate unrealized P&L correctly
- Handle position increases (update cost basis)

**Order Processing:**
- Exit orders process before entry orders (capital efficiency)
- Mark-to-market after exits, before entries
- Order validation before execution (not after)
- Rejected orders logged with reason

**Trade Recording:**
- Record complete fill details
- Track commission and slippage per trade
- Calculate realized P&L on position close
- Generate Trade objects for analysis

## Non-Functional Requirements

### Performance
- Maintain current speed (32x faster than Backtrader)
- No significant overhead for cash accounts
- Marginal cost for margin account calculations

### Correctness
- Match VectorBT behavior for cash-constrained scenarios (±0.1%)
- Match Backtrader/VectorBT for margin scenarios
- Pass "Bankruptcy Test" (equity cannot go below zero)
- Pass "Flipping Test" (long↔short commission tracking)

### Maintainability
- Clear separation: accounting logic independent of engine
- No parallel systems or feature flags
- Single Position class (no duplication)
- Policy pattern for account type flexibility

### Extensibility
- Easy to add new account types (portfolio margin, futures margin)
- Commission models remain pluggable
- Slippage models remain pluggable

## Acceptance Criteria

### Cash Account Mode
- [ ] Orders rejected when cost > available cash
- [ ] Cash never goes negative
- [ ] Short orders rejected (not allowed)
- [ ] Exit-first sequencing frees cash for re-entry
- [ ] VectorBT validation passes (fix 99.4% difference)
- [ ] All 17 existing unit tests pass (with updates)

### Margin Account Mode
- [ ] Orders rejected when margin requirement > buying power
- [ ] Short positions allowed and tracked correctly
- [ ] Position reversals handled atomically
- [ ] Bankruptcy Test passes (equity floor at 0)
- [ ] Flipping Test passes (commission tracking)
- [ ] Margin calculations match industry standards

### Both Modes
- [ ] Position cost basis tracking works correctly
- [ ] Commission/slippage deducted properly
- [ ] Trade recording on position close
- [ ] Multi-asset position tracking
- [ ] Mark-to-market updates work correctly

## Out of Scope

### Not in Phase 1
- Portfolio margin (different margin treatment for related positions)
- Options margin (complex rules for multi-leg option strategies)
- Futures margin (per-contract margin requirements)
- Real-time margin calls (intraday margin monitoring)
- Partial fills when insufficient capital (reject vs scale-down)

### Future Enhancements
- Volume-based liquidity constraints (10% of bar volume)
- Zero-volume bar handling (order remains pending)
- T+1 settlement rules (when exit proceeds become available)
- Asset-specific margin rates (stocks vs futures vs crypto)

## Dependencies

### Internal
- Current engine.py (744 lines) - will be refactored
- Order, Position, Fill, Trade classes - will be unified/enhanced
- Commission and Slippage models - will integrate with accounting
- 17 existing unit tests - will be updated

### External
- External review from Gemini (recommendations received)
- VectorBT Pro validation framework (for testing)
- Backtrader validation framework (for testing)

## Risks and Assumptions

### Risks
1. **Test Breakage**: Updating 17 tests to expect rejections
2. **Integration Complexity**: Weaving gatekeeper into execution flow
3. **Edge Cases**: Position reversals, partial closes, commission timing
4. **Performance**: Ensuring margin calculations don't slow execution

### Assumptions
1. **No users yet**: Breaking changes acceptable (confirmed by user)
2. **Cash accounts default**: Most users will use cash accounts initially
3. **Standard margin rates**: RegT (50% initial, 25% maintenance) is default
4. **Daily bars**: Intraday margin calls not required initially

### Mitigation
- Phased implementation (cash accounts first, then margin)
- Comprehensive test suite before integration
- Spreadsheet validation for 5-trade scenarios (manual ground truth)
- Performance benchmarks maintained throughout

## Implementation Strategy

### Phase 1: Core Infrastructure (2-3 hours)
- Create `accounting/` package
- Implement unified Position class
- Create AccountPolicy interface
- Implement CashAccountPolicy
- Write unit tests

### Phase 2: Cash Account Integration (2-3 hours)
- Update Broker to use AccountState
- Implement Gatekeeper validation
- Add exit-first sequencing
- Update existing 17 tests
- Validate against VectorBT

### Phase 3: Margin Account Support (3-4 hours)
- Implement MarginAccountPolicy
- Add short selling support
- Handle position reversals
- Write margin-specific tests

### Phase 4: Documentation (2 hours)
- Update README with examples
- Document margin calculations
- Add architecture decision records
- Clean up and finalize

**Total Estimate**: 10-12 hours (2-3 days)

## Success Metrics

1. **VectorBT Validation**: Match within 0.1% for cash-constrained scenarios
2. **Test Coverage**: All 17 existing tests pass + 10 new tests
3. **Performance**: Maintain 30x+ speed advantage over Backtrader
4. **Code Quality**: Clean separation of concerns, no technical debt
5. **Bankruptcy Test**: Equity cannot go below zero (margin accounts)

## References

- External review: `.claude/code_review/251120-01/gemini_01.md`
- Handoff document: `.claude/transitions/2025-11-20/135656.md`
- Design questions: `DESIGN_QUESTIONS.md` (root)
- Current validation test: `tests/validation/test_validation.py`
