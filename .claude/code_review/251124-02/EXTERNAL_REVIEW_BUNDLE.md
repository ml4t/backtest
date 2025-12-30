# ml4t.backtest - External Code Review Bundle

**Date**: 2025-11-24
**Package**: ml4t.backtest v0.2.0
**Review Type**: Correctness, Feature Completeness, Design Quality

---

## Contents of This Bundle

1. **Review Request** - Detailed questions and context (see REVIEW_REQUEST.md)
2. **README** - Complete feature documentation (see ../../../README.md)
3. **Source Code** - All 33 source files (see backtest_src.xml)

---

## Quick Start for Reviewer

### Step 1: Read the Review Request

File: `REVIEW_REQUEST.md`

This document contains:
- Specific questions about our implementation
- Comparison with BackTrader/Zipline/VectorBT
- Known limitations
- Edge cases to consider

### Step 2: Review the README

File: `../../../README.md` (1,135 lines)

This documents all claimed features:
- 8 order types
- 2 account types (cash vs margin)
- Portfolio rebalancing
- Execution realism (volume limits, market impact)
- Analytics & metrics
- Framework compatibility

### Step 3: Review the Source Code

File: `backtest_src.xml` (33 files, ~2,800 lines)

Key files to review:
- `engine.py` - Main event loop
- `broker.py` - Order execution logic
- `accounting/policy.py` - Cash vs margin policies
- `accounting/gatekeeper.py` - Order validation
- `execution/rebalancer.py` - Portfolio rebalancing

---

## Critical Questions (High Priority)

### 1. Cash vs Margin Account Implementation

**File**: `accounting/policy.py`

**Question**: Is our buying power formula correct?

```python
# Our formula
buying_power = (net_liquidation_value - maintenance_margin) / initial_margin
```

**Is this correct for US equities (Reg T)?**

### 2. Position Flip Logic

**File**: `broker.py` (position tracking)

**Scenario**: Long 100 shares, submit sell order for -200 shares

**Our behavior**: Create single fill for -200, result in short 100 position

**Question**: Is this correct? Should it be two trades (close long, open short)?

### 3. Exit-First Processing

**File**: `broker.py:_process_orders()`

**Our behavior**: Process exit orders before entry orders to free capital

**Question**: Is this correct for same-bar rebalancing? Should equity update happen between exit and entry?

### 4. Stop Order Fills

**File**: `broker.py:_check_stop_trigger()`

**Question**: Should stops fill at stop price or next available price? How to handle gaps?

### 5. Partial Fills

**File**: `execution/limits.py`

**Question**: Should partially filled orders be marked PARTIALLY_FILLED or stay PENDING?

---

## Feature Completeness (vs BackTrader/Zipline/VectorBT)

### What We Have

✅ Event-driven architecture
✅ Multi-asset support
✅ Cash + margin accounts
✅ 8 order types (market, limit, stop, trailing, bracket)
✅ Commission models (per-share, percentage, tiered)
✅ Slippage models (fixed, percentage, volume-based)
✅ Volume participation limits
✅ Market impact modeling
✅ Portfolio rebalancing (TargetWeightExecutor)
✅ Analytics (trade analysis, metrics)
✅ Framework compatibility presets
✅ Validated against VectorBT/Backtrader/Zipline (EXACT matches)

### What We're Missing

❌ Indicators (design choice: use separate library)
❌ Optimizers (design choice: use optuna/hyperopt)
❌ Portfolio constraints (sector limits, turnover limits)
❌ Options support (Black-Scholes, Greeks)
❌ Futures support (contract specs, roll logic)
❌ Transaction cost analysis (IS, VWAP benchmarking)
❌ Risk metrics during backtest
❌ Corporate actions (splits, dividends)

**Question**: Which of these missing features are critical vs nice-to-have?

---

## Known Edge Cases

| Edge Case | Current Behavior | Correct? |
|-----------|------------------|----------|
| Gap day (open > stop price) | Fill at open | ✅ Probably correct |
| Order size > bar volume | Partial fill + queue | ✅ Realistic |
| Zero volume bar | Skip fill, keep pending | ❓ Should we error? |
| Multiple stops same bar | Fill all that trigger | ❓ Which fills first? |
| Position flip with stops | Stop becomes invalid | ❓ Cancel or adjust? |
| Rebalance insufficient capital | Skip orders | ✅ Correct |
| Short squeeze | Fill fails | ❓ How to model? |
| Delisted stock | Position stays | ❌ Should go to zero |
| Stock split | No adjustment | ❌ Need corporate actions |
| Market halt | Not modeled | ❌ How to handle? |

**Question**: Which of these MUST be handled vs can be documented limitations?

---

## Validation Results

We validated against 3 frameworks with EXACT numeric matches:

| Framework | Scenarios | Match Type |
|-----------|-----------|------------|
| VectorBT Pro | 4/4 | EXACT (within 1e-10) |
| VectorBT OSS | 4/4 | EXACT (within 1e-10) |
| Backtrader | 4/4 | EXACT (within 1e-10) |
| Zipline | 4/4 | Within tolerance (strategy-level stops) |

**Validation scenarios**:
1. Long-only basic trades
2. Long-short with position flipping
3. Stop-loss execution
4. Take-profit execution

**Validation code**: See `validation/` directory

**Question**: Are there other critical scenarios we should validate?

---

## API Design Questions

### 1. Order Return Type

Current:
```python
order = broker.submit_order("AAPL", 100)
# Returns Order object with order_id immediately available
```

Alternative:
```python
order_id = broker.submit_order("AAPL", 100)
# Just return ID, query order later
```

**Question**: Which is better for live trading transition?

### 2. Position Access

Current:
```python
position = broker.get_position("AAPL")  # Method
positions = broker.positions              # Property
```

**Question**: Should both be methods or both properties?

### 3. ExecutionResult Semantics

Current:
```python
result = executor.execute(targets, data, broker)
if result.success:
    # All orders submitted (some may be skipped due to constraints)
```

**Question**: Should "success" mean "all executed" or "no errors"?

---

## Performance Benchmarks

**Current**: ~100k events/sec (event-driven iteration on MacBook Pro M1)

**Comparison**:
- VectorBT: ~1M+ events/sec (vectorized, no event loop)
- Backtrader: ~50k events/sec (event-driven, more overhead)
- Zipline: ~20k events/sec (heavy machinery)

**Question**: Is 100k events/sec acceptable for daily bars over 20 years (5,000 bars)?

---

## Files in This Bundle

### 1. REVIEW_REQUEST.md (this file)

Comprehensive review request with:
- Specific technical questions
- Feature comparison tables
- Edge case analysis
- API design concerns

### 2. README.md (../../../README.md)

Complete feature documentation:
- 1,135 lines
- 16 sections
- Code examples for every feature
- API reference

### 3. backtest_src.xml

Complete source code:
- 33 Python files
- ~2,800 lines of production code
- XML format for easy parsing

---

## How to Review

### Priority 1: Critical Logic (1-2 hours)

Review these files for correctness:

1. `accounting/policy.py` - Cash vs margin logic
2. `broker.py:_process_orders()` - Order execution
3. `broker.py:_try_fill()` - Fill logic
4. `accounting/gatekeeper.py` - Order validation

### Priority 2: Feature Completeness (30 min)

Answer:
- What features from BackTrader/Zipline/VectorBT are we missing?
- Which missing features are dealbreakers for users?
- Are there obvious bugs waiting to happen?

### Priority 3: API Design (30 min)

Answer:
- Will we regret these API choices?
- Are there better patterns we should use?
- Is the API intuitive for quantitative traders?

---

## Expected Deliverable

A review document covering:

1. **Correctness Issues**: Bugs, logical flaws, incorrect formulas
2. **Feature Gaps**: Critical missing features from BackTrader/Zipline/VectorBT
3. **Account Policy Review**: Is cash vs margin implementation correct?
4. **Edge Cases**: Which edge cases will break the engine?
5. **API Concerns**: Design choices we'll regret?
6. **Recommendations**: Prioritized list of improvements

---

**Location**: `/home/stefan/ml4t/software/backtest/.claude/code_review/251124-02/`

**Files**:
- `REVIEW_REQUEST.md` - This document
- `backtest_src.xml` - Complete source code (33 files)
- `../../../README.md` - Feature documentation

**Thank you for your thorough review!**
