# Backtesting Engine: External Review Package

**Status**: Seeking architectural review before full implementation
**Version**: Minimal prototype (744 lines)
**Performance**: 32x faster than Backtrader, 0.39% PnL match
**Critical Gap**: No cash constraint enforcement (discovered during validation)

---

## Quick Start for Reviewers

### 1. Review Scope

We're seeking expert feedback on **architectural design** before investing in full implementation:

- ✅ We have a fast, minimal working engine
- ✅ Validates against Backtrader (0.39% PnL diff)
- ❌ Lacks realistic accounting for cash, leverage, and liquidity
- ❓ **Should we build this ourselves or adopt an existing framework?**

### 2. Key Documents

| Document | Purpose | Priority |
|----------|---------|----------|
| **[DESIGN_QUESTIONS.md](DESIGN_QUESTIONS.md)** | Design questions needing expert input | **START HERE** |
| `src/ml4t/backtest/engine.py` | Current implementation (744 lines) | High |
| `tests/validation/test_engine_validation.py` | Framework comparison validation | Medium |
| `tests/test_core.py` | Core functionality tests (17/17 passing) | Low |

### 3. Critical Finding

Validation study revealed engine executes orders without cash constraint checking:

```python
# VectorBT (correct):  -$3,891 PnL (respects cash limits)
# engine (broken):  -$652,580 PnL (unlimited debt)
# Difference: 99.4% (16,700% when normalized)
```

This discovery prompted this review request.

---

## Project Context

### What We're Building

A **high-performance backtesting engine** for quantitative trading strategies with:
- Institutional-grade execution modeling
- Realistic accounting for longs, shorts, leverage
- Support for 100s of assets × years of data
- 10-100x faster than existing solutions

### Current State

**Working**:
- Event-driven architecture
- Multi-asset support (validated with 250 assets)
- Pluggable commission/slippage models
- Same-bar and next-bar execution modes
- 76% test coverage, all tests passing

**Missing**:
- Cash constraint enforcement
- Leverage limits
- Realistic execution sequencing (exits before entries)
- Market impact / liquidity constraints
- Position sizing constraints

### Validation Results

| Framework | Match | Speed | Status |
|-----------|-------|-------|--------|
| **Backtrader** | 0.39% ✓ | 32x slower | PASS |
| **VectorBT** | 99.4% ✗ | 5.8x slower | FAIL (cash limits) |
| **Zipline** | Skipped | - | Data incompatibility |

---

## Repository Structure

```
backtest/
├── src/ml4t/backtest/
│   └── engine.py              # Main implementation (744 lines)
│
├── tests/
│   ├── test_core.py              # Unit tests (17/17 passing)
│   └── validation/
│       └── test_engine_validation.py  # Framework comparison
│
├── DESIGN_QUESTIONS.md           # <-- Design review questions
├── README_REVIEW.md              # <-- This file
└── archive/                       # Previous 19,876-line implementation
```

---

## Design Questions Summary

The [DESIGN_QUESTIONS.md](DESIGN_QUESTIONS.md) document contains detailed questions organized into six parts:

### Part 1: Accounting Model (10 questions)
- How to model longs, shorts, leverage, and cash flow?
- When do exit proceeds become available for new entries?
- Should we use gross exposure or net leverage?

### Part 2: Execution Sequencing (8 questions)
- What's the exact event timeline for daily trading?
- Should exits fill before entries (to free cash)?
- How to handle position reversals (long → short)?
- Should we support partial fills?

### Part 3: Liquidity & Market Impact (7 questions)
- How to execute on zero-volume bars (overnight futures)?
- Do we need market impact modeling?
- What's the difference between slippage and impact?
- Should we constrain order size to % of volume?

### Part 4: Architecture (6 questions)
- Build vs buy: extend our engine or adopt Backtrader/VectorBT?
- Realistic vs fast: what's the right tradeoff?
- API design: event-driven vs declarative?

### Part 5: Implementation Details (6 questions)
- When to update cash (immediate vs end-of-bar)?
- When to check leverage (pre-fill vs post-fill)?
- Order of commission/slippage application?

### Part 6: Testing Strategy (4 questions)
- How to validate accounting without real-world ground truth?
- Which framework to match (VectorBT, Backtrader, both)?
- What edge cases reveal accounting bugs?

---

## Specific Review Requests

### 1. Accounting Model Review

**Question**: Is this accounting identity correct?

```python
# Proposed model
Account_Value = Cash + Σ(position_qty × current_price)

# Where:
# - Long positions have positive qty
# - Short positions have negative qty
# - Cash includes proceeds from short sales
```

**Follow-up**: How do we enforce leverage limits with this model?

### 2. Execution Sequence Review

**Question**: Is this execution timeline correct for daily bars?

```
Day N 16:00: Market closes
Day N 16:00-23:59: Strategy generates orders
Day N+1 09:30: Market opens
  1. Execute all exit orders → update cash
  2. Execute all entry orders (using updated cash)
Day N+1 16:00: Market closes
```

**Follow-up**: What about same-bar execution (backtesting simplification)?

### 3. Build vs Buy Decision

**Question**: Should we continue building or adopt an existing framework?

**Options**:
- **Build**: Add accounting to our fast engine (estimated 2-4 weeks)
- **Fork Backtrader**: Keep accounting, rewrite for speed (4-6 weeks)
- **Adopt VectorBT**: Use as-is (immediate) but limited flexibility
- **Hybrid**: Use VectorBT for simple cases, our engine for complex

**Your recommendation**: _______________

### 4. Scope Definition

**Question**: What's the minimum viable scope for "realistic enough"?

**Must have**:
- [ ] Cash constraint enforcement
- [ ] Leverage limits
- [ ] Exit-before-entry sequencing
- [ ] ??? (What else?)

**Nice to have**:
- [ ] Partial fills
- [ ] Market impact
- [ ] Liquidity constraints
- [ ] Settlement delays (T+1/T+2)

### 5. Testing Strategy

**Question**: How do we validate the accounting model?

**Options**:
- Match VectorBT (strictest cash limits)
- Match Backtrader (some leverage allowed)
- Build from first principles (but how to validate?)
- Match real broker fills (need live trading data)

---

## Code Walkthrough

### Current Implementation Highlights

**Broker class** (simplified):
```python
class Broker:
    def __init__(self, initial_cash):
        self.cash = initial_cash
        self.positions = {}  # asset → Position

    def submit_order(self, asset, quantity, side):
        # BUG: No cash check here
        order = Order(asset=asset, quantity=quantity, side=side)
        self.pending_orders.append(order)

    def _execute_fill(self, order, fill_price):
        # BUG: Fills without checking cash
        commission = self.commission_model.calculate(...)
        total_cost = fill_price * quantity + commission
        self.cash -= total_cost  # Can go negative!
        self._update_position(order)
```

**Fix needed**:
```python
def _execute_fill(self, order, fill_price):
    commission = self.commission_model.calculate(...)
    total_cost = fill_price * quantity + commission

    # NEW: Check cash constraint
    if order.side == OrderSide.BUY:
        if total_cost > self.cash:
            return RejectedFill(reason="Insufficient cash")

    self.cash -= total_cost
    self._update_position(order)
```

**But this raises questions**:
1. Should we check BEFORE fill or reject after?
2. What about leverage mode (short proceeds fund longs)?
3. Should we allow partial fills?
4. When do exit proceeds become available?

---

## Performance Characteristics

From validation study (250 assets × 252 days × ~11,000 trades):

```
Backtrader: 18.7s
engine:   0.6s (32x faster)
VectorBT:    3.4s (5.8x faster than Backtrader, 5.7x slower than engine)
```

**Key insight**: We can be very fast, but speed is worthless if accounting is wrong.

---

## Questions for You

### As a Trading System Expert:
1. Is the proposed accounting model realistic for institutional trading?
2. How do production systems handle the exit-before-entry sequencing?
3. What leverage limits are typical for automated futures trading?

### As a Backtest Framework Developer:
1. What were the hardest accounting edge cases you encountered?
2. How did you validate your accounting model was correct?
3. What would you do differently if starting over?

### As a Quantitative Researcher:
1. What level of execution realism is needed for academic research?
2. Do you match a reference framework or model from first principles?
3. How do you calibrate market impact without proprietary data?

---

## How to Provide Feedback

### Preferred Format

**Option 1: Written Review**
- Review [DESIGN_QUESTIONS.md](DESIGN_QUESTIONS.md)
- Answer the specific questions in Parts 1-6
- Provide architectural recommendations
- Suggest reference implementations or papers

**Option 2: Annotated Code Review**
- Review `src/ml4t/backtest/engine.py`
- Add inline comments on what's wrong/missing
- Point to examples of correct implementations

**Option 3: Discussion**
- We can schedule a call to walk through the design
- Discuss specific technical challenges
- Get real-time feedback on proposals

### Specific Asks

1. **Accounting Model**: Is the proposed model correct? What's missing?
2. **Execution Sequence**: Is the timeline realistic? What edge cases are we missing?
3. **Build vs Buy**: Should we continue or adopt an existing framework?
4. **Scope**: What's the minimum viable implementation?
5. **Testing**: How do we validate correctness without real broker fills?

---

## Timeline & Next Steps

**Current state**: Design review phase (2-3 weeks)
**After review**: Implementation decision
  - If build: 4-6 weeks to add accounting + testing
  - If adopt: Immediate integration with VectorBT/Backtrader
**Goal**: Production-ready engine for algorithmic trading research

---

## Contact

**Repository**: `/home/stefan/ml4t/software/backtest/`
**Email**: [Your email]
**Availability**: [Your availability for discussion]

**Thank you for taking the time to review this design!**

Your expertise will help us build a better backtesting engine for the quantitative trading community.
