# ml4t.backtest Code Review Report

**Reviewer**: Claude (External)
**Date**: 2025-11-24
**Package Version**: 0.2.0

---

## Executive Summary

The ml4t.backtest engine is a well-designed, minimal backtesting framework with solid fundamentals. The code is clean,
well-documented, and demonstrates thoughtful architecture. However, I've identified several issues ranging from critical
bugs to design improvements that should be addressed before production use.

**Overall Assessment**: Good foundation, needs targeted fixes before v1.0

| Category     | Rating          | Notes                                   |
|--------------|-----------------|-----------------------------------------|
| Architecture | ✅ Strong        | Clean separation, protocol-based design |
| Correctness  | ⚠️ Issues       | Several bugs in edge cases              |
| Completeness | ✅ Good          | Covers 80%+ of typical use cases        |
| API Design   | ⚠️ Minor issues | Some inconsistencies                    |
| Performance  | ✅ Adequate      | 100k events/sec is fine for daily bars  |

---

## 1. Critical Issues (Must Fix)

### 1.1 Commission Calculation on Position Flip

**File**: `broker.py:_execute_fill()` (lines ~340-360)

**Bug**: When flipping a position (long 100 → short 100 via -200 order), commission is calculated on the full 200 shares
but only applied to the closing trade's PnL.

```python
# Current (buggy):
elif (old_qty > 0) != (new_qty > 0):
# Position flipped
pnl = (fill_price - pos.entry_price) * old_qty - commission  # Commission on 200, applied to close only
self.trades.append(Trade(..., commission=commission))  # Full commission here
self.positions[order.asset] = Position(...)  # New position has no commission tracked
```

**Impact**: Overstates closing trade costs, understates opening position costs. Affects trade analytics.

**Fix**:

```python
elif (old_qty > 0) != (new_qty > 0):
# Split commission between close and open
close_qty = abs(old_qty)
open_qty = abs(new_qty)
close_commission = self.commission_model.calculate(order.asset, close_qty, fill_price)
open_commission = self.commission_model.calculate(order.asset, open_qty, fill_price)

pnl = (fill_price - pos.entry_price) * old_qty - close_commission
self.trades.append(Trade(..., commission=close_commission))
# Track open_commission somewhere (position metadata or separate)
```

### 1.2 Stop Order Gap Fill Logic

**File**: `broker.py:_check_fill()` (lines ~250-270)

**Bug**: Gap-through fill price logic is incorrect for sell stops.

```python
# Current:
elif order.side == OrderSide.SELL and low <= order.stop_price:
return min(order.stop_price, high)  # This returns stop_price most of the time
```

**Scenario**: Stock gaps down through stop

- Yesterday close: $150
- Stop price: $145
- Today: Open=$140, High=$142, Low=$138
- **Current behavior**: Returns min(145, 142) = $142 ❌
- **Correct behavior**: Should fill at $140 (open) since price gapped through

**Fix**:

```python
elif order.side == OrderSide.SELL and low <= order.stop_price:
# Check if gapped through (opened below stop)
bar_open = self._current_opens.get(order.asset, price)
if bar_open <= order.stop_price:
    return bar_open  # Gap through - fill at open
else:
    return order.stop_price  # Normal trigger - fill at stop
```

### 1.3 Margin Account: Asymmetric Long/Short Requirements

**File**: `accounting/policy.py:MarginAccountPolicy`

**Issue**: The buying power formula treats longs and shorts symmetrically, but Reg T has different requirements:

| Position Type | Initial Margin             | Maintenance Margin |
|---------------|----------------------------|--------------------|
| Long          | 50%                        | 25%                |
| Short         | 150% (50% + 100% proceeds) | 30-40%             |

**Current formula**:

```python
buying_power = (nlv - maintenance_margin_requirement) / self.initial_margin
```

This doesn't account for:

1. Short positions requiring collateral beyond the sale proceeds
2. Different maintenance margin rates for shorts
3. Short sale proceeds crediting to cash

**Recommendation**: Either:

- Document this as a simplification (acceptable for most backtests)
- Implement separate `long_initial_margin`, `short_initial_margin`, etc.

For now, add a prominent note in docstring:

```python
"""
Note: This is a simplified margin model that treats long and short positions
symmetrically. Real Reg T has higher requirements for shorts (150% initial,
30%+ maintenance). For accurate short selling simulation, consider using
custom validation or the upcoming FuturesAccountPolicy.
"""
```

---

## 2. Significant Issues (Should Fix)

### 2.1 Bracket Orders Not Cancelled on Position Flip

**File**: `broker.py:_execute_fill()`

**Issue**: When a position flips (long → short), existing bracket orders (stop-loss, take-profit) from the original
position remain active and become dangerous.

**Scenario**:

1. Long 100 AAPL with stop-loss at $145, take-profit at $160
2. Strategy flips to short 100 via sell -200
3. Old stop-loss ($145 sell) is now WRONG for short position
4. If price drops to $145, the sell stop triggers → now short 200!

**Current handling**: Only cancels sibling brackets when a bracket order fills, not on position flip.

**Fix**: In `_execute_fill()`, when detecting position flip, cancel all pending orders for that asset:

```python
elif (old_qty > 0) != (new_qty > 0):
# Position flipped - cancel all pending orders for this asset
for o in self.pending_orders[:]:
    if o.asset == order.asset and o.order_id != order.order_id:
        o.status = OrderStatus.CANCELLED
        self.pending_orders.remove(o)
# ... rest of flip logic
```

### 2.2 Multiple Stops Same Bar: Non-Deterministic Order

**File**: `broker.py:_process_orders()`

**Issue**: When multiple stop orders trigger on the same bar, fill order depends on list iteration order, which is
undefined.

**Scenario**: Long 100 AAPL with:

- Stop-loss at $145
- Trailing stop currently at $148
- Bar: Open=$150, Low=$143 (both trigger)

Which fills first matters because:

- First fill closes position
- Second fill creates new (unintended) short position

**Fix**: Add explicit priority to stop processing:

```python
# Sort stops by price (for sells: higher price first, for buys: lower price first)
exit_orders.sort(key=lambda o: (
    o.stop_price if o.stop_price else float('inf'),
    o.order_id  # Stable sort tie-breaker
), reverse=(o.side == OrderSide.SELL))
```

Or, more conservatively: Cancel duplicate stops once one triggers.

### 2.3 Gatekeeper Doesn't Check Order Already in Pending

**File**: `accounting/gatekeeper.py`

**Issue**: The gatekeeper validates buying power at fill time, but doesn't check if there are already pending orders
that would consume that buying power.

**Scenario**:

1. Cash = $10,000, buying power = $10,000
2. Submit order: Buy 100 AAPL @ $100 = $10,000 (validated ✅)
3. Submit order: Buy 100 GOOG @ $100 = $10,000 (validated ✅ - same buying power!)
4. First order fills → cash = $0
5. Second order rejects at fill time

This is **correct behavior** but may surprise users. Consider:

- Adding a `validate_at_submission` flag to catch this earlier
- Documenting this clearly in README

---

## 3. Correctness Analysis by Question

### Q1: Cash vs Margin Buying Power Formula

**Formula**: `BP = (NLV - MM) / IM`

**Assessment**: Correct conceptually for a simplified model. Real-world nuances:

| Aspect            | Your Model | Reality | Impact                    |
|-------------------|------------|---------|---------------------------|
| Long initial      | 50%        | 50%     | ✅ Correct                 |
| Short initial     | 50%        | 150%    | ⚠️ Overstates short BP    |
| Long maintenance  | 25%        | 25%     | ✅ Correct                 |
| Short maintenance | 25%        | 30%+    | ⚠️ Understates short risk |
| Settlement        | Immediate  | T+2     | ⚠️ Overstates BP          |

**Verdict**: Acceptable for most backtests. Document limitations.

### Q2: Position Flip - Single Trade vs Two Trades?

**Your approach**: Single fill, but record as one closing trade + new position

**Assessment**: This is **correct** and matches real brokerage behavior. A flip order is a single order that results in
one fill. The accounting correctly:

- Closes the old position (records Trade)
- Opens new opposite position (creates Position)

**BackTrader/Zipline**: Same behavior
**VectorBT**: Handles this vectorized, but semantically equivalent

### Q3: Exit-First Processing

**Your implementation**: ✅ Correct

The sequence:

1. Process exits → frees capital
2. `mark_to_market()` → updates equity
3. Process entries → can use freed capital

This is the right approach for same-bar rebalancing. The equity update between phases is crucial and correctly
implemented.

### Q4: Stop Order Fills at Stop Price or Next Available?

**Your implementation**: Fills at stop price (with bug in gap handling noted above)

**Comparison**:
| Framework | Stop Fill Price |
|-----------|-----------------|
| BackTrader | Stop price (or worse if gapped) |
| Zipline | Next bar open (strategy-level stops) |
| VectorBT | Configurable (close, stop price, etc.) |
| ml4t | Stop price (intended), gaps handled |

**Assessment**: Your choice (stop price) is the most common and realistic. Fix the gap handling bug.

### Q5: Partial Fills Status

**Your implementation**: Orders stay PENDING, track remaining in `_partial_orders`

**Assessment**: This is a valid design choice. Alternatives:

| Approach                  | Pros                              | Cons                           |
|---------------------------|-----------------------------------|--------------------------------|
| Stay PENDING (yours)      | Simple, orders naturally continue | Status doesn't reflect reality |
| PARTIALLY_FILLED status   | More realistic                    | Requires more status handling  |
| Clone order for remainder | Clean separation                  | More objects, harder to track  |

**Recommendation**: Current approach is fine for backtesting. Add `filled_quantity` tracking to Order for inspection:

```python
# Already done! Order has filled_quantity field
```

---

## 4. Feature Gap Analysis

### Critical Missing Features

| Feature      | Impact                           | Recommendation                          |
|--------------|----------------------------------|-----------------------------------------|
| Stock splits | Position quantities become wrong | **Must add** before production          |
| Dividends    | Cash not credited                | Important for realistic equity curves   |
| Delisting    | Zombie positions                 | Handle as position closed at last price |

### Important but Not Critical

| Feature            | Impact                         | Recommendation            |
|--------------------|--------------------------------|---------------------------|
| T+2 settlement     | BP overstated                  | Document limitation       |
| Margin interest    | Short costs understated        | Add optional borrow rate  |
| Pattern day trader | Unrealistic for small accounts | Add optional check        |
| Wash sale tracking | Tax implications               | Out of scope for backtest |

### Nice to Have (Defer to v0.3+)

- Options Greeks
- Futures roll logic
- Real-time risk metrics
- Transaction cost analysis

---

## 5. Edge Cases Assessment

| Edge Case                      | Current         | Correct?             | Fix Priority |
|--------------------------------|-----------------|----------------------|--------------|
| Gap day (open > stop)          | Fill at stop    | **No** (see 1.2)     | **High**     |
| Order > bar volume             | Partial + queue | ✅ Yes                | -            |
| Zero volume bar                | Skip fill       | ✅ Yes                | -            |
| Multiple stops same bar        | All fire        | ⚠️ Non-deterministic | Medium       |
| Position flip with stops       | Stops stay      | **No** (see 2.1)     | **High**     |
| Rebalance insufficient capital | Skip orders     | ✅ Yes                | -            |
| Delisted stock                 | Position stays  | **No**               | Medium       |
| Stock split                    | No adjustment   | **No**               | **High**     |
| Market halt                    | Not modeled     | ⚠️ Document          | Low          |

---

## 6. API Design Review

### 6.1 Order Return Type

**Current**: Returns Order object
**Alternative**: Return just order_id

**Recommendation**: Keep current. Order object is immediately useful and matches typical brokerage APIs. The object
reference allows:

```python
order = broker.submit_order("AAPL", 100)
if order.status == OrderStatus.REJECTED:
    logger.warning(f"Order rejected: {order}")
```

### 6.2 Position Access Inconsistency

**Current**:

```python
position = broker.get_position("AAPL")  # Method
positions = broker.positions  # Property
```

**Issue**: Inconsistent API style

**Recommendation**: Make both properties OR both methods. Prefer properties for simple lookups:

```python
@property
def positions(self) -> dict[str, Position]:
    return self._positions


def position(self, asset: str) -> Position | None:  # Keep method for None handling
    return self._positions.get(asset)
```

Or use `__getitem__` for dict-like access:

```python
broker.positions["AAPL"]  # Raises KeyError if not found
broker.positions.get("AAPL")  # Returns None
```

### 6.3 ExecutionResult.success Semantics

**Current**: `success` means "no errors occurred"
**Issue**: Ambiguous when orders are skipped due to constraints

**Recommendation**: Be explicit:

```python
@dataclass
class ExecutionResult:
    all_executed: bool  # All requested orders submitted
    errors: list[str]  # Hard errors (exceptions, etc.)
    skipped: list[tuple[str, str]]  # (asset, reason) for skipped
    orders: list[Order]  # Successfully submitted orders

    @property
    def success(self) -> bool:
        """True if no hard errors occurred (skips are OK)."""
        return len(self.errors) == 0

    @property
    def complete(self) -> bool:
        """True if all orders executed with no skips."""
        return self.all_executed and len(self.skipped) == 0
```

---

## 7. Performance Assessment

**Your benchmark**: ~100k events/sec
**Comparison**:

- VectorBT: 1M+ (vectorized)
- BackTrader: ~50k (event-driven)
- Zipline: ~20k (full pipeline)

**Assessment**: 100k events/sec is **excellent** for an event-driven engine.

For 20 years of daily data (5,000 bars × 100 assets = 500k events):

- Your engine: ~5 seconds
- BackTrader: ~10 seconds
- Zipline: ~25 seconds

**Known bottlenecks** (from your list):

1. **Polars iteration**: Converting to dict per bar - minor impact
2. **Order validation**: O(1) per order - fine
3. **Position deep copy**: Not actually doing this - you update in place ✅

**Recommendations**:

- Current performance is fine for daily/hourly bars
- For minute bars (390 × 252 × 20 = 2M bars), consider lazy position copy
- For HFT simulation, would need architectural changes (not in scope)

---

## 8. Validation Recommendations

### Additional Scenarios to Test

| Scenario                         | Why Important                      |
|----------------------------------|------------------------------------|
| Gap-through stop                 | Tests fix for bug 1.2              |
| Position flip with brackets      | Tests fix for bug 2.1              |
| Multiple assets, same stop price | Tests deterministic ordering       |
| Partial fill over 3+ bars        | Tests cumulative tracking          |
| Margin call simulation           | Tests MM < equity scenario         |
| Negative cash recovery           | Tests buying power when underwater |

### Suggested Test Cases

```python
def test_gap_through_sell_stop():
    """Stop should fill at open when price gaps through."""
    # Setup: Long 100 @ $150, stop at $145
    # Bar: Open=$140, High=$142, Low=$138
    # Expected: Fill at $140, not $145


def test_position_flip_cancels_brackets():
    """Bracket orders should be cancelled when position flips."""
    # Setup: Long 100 with SL=$145, TP=$160
    # Action: Sell -200 (flip to short)
    # Expected: SL and TP cancelled


def test_commission_split_on_flip():
    """Commission should be split between close and open on flip."""
    # Setup: Long 100 @ $100
    # Action: Sell -200 @ $110
    # Expected: Close commission on 100, open commission on 100
```

---

## 9. Prioritized Recommendations

### P0 - Must Fix Before Beta

1. **Stop gap-through fill price** (Bug 1.2)
2. **Cancel brackets on position flip** (Bug 2.1)
3. **Commission split on flip** (Bug 1.1)

### P1 - Should Fix Before v1.0

4. **Document margin model limitations**
5. **Deterministic stop order processing**
6. **Add stock split handling**
7. **Add delisting handling**

### P2 - Consider for v1.x

8. **Separate long/short margin rates**
9. **T+2 settlement option**
10. **Dividend handling**
11. **API consistency cleanup**

---

## 10. Summary

The ml4t.backtest engine is well-architected with clean code and good separation of concerns. The validation against
VectorBT/BackTrader/Zipline is impressive and demonstrates attention to correctness.

**Strengths**:

- Clean, minimal codebase (~2,800 lines)
- Protocol-based design enables extensibility
- Exit-first processing is correct
- Framework presets are valuable
- Good documentation

**Areas for Improvement**:

- Three critical bugs need immediate attention
- Margin model simplifications should be documented
- Edge case handling needs hardening
- API consistency could be improved

**Bottom Line**: With the P0 fixes applied, this is a solid backtesting engine suitable for daily/hourly equity
strategies. The framework compatibility and validation approach sets it apart from many alternatives.

---

*Review completed. Please reach out with questions on any specific findings.*