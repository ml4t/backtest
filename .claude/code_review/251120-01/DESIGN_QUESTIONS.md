# Backtesting Engine: Design Questions for External Review

**Status**: Pre-implementation architectural review
**Purpose**: Obtain expert feedback on design before full implementation
**Context**: We have a minimal working engine (744 lines) that validates against Backtrader (0.39% PnL difference, 32x faster) but lacks proper accounting for leverage, cash constraints, and realistic execution.

---

## Executive Summary

We've built a fast, minimal backtesting engine that passes validation against Backtrader but has fundamental gaps in modeling realistic trading mechanics. Before continuing development, we need expert guidance on:

1. **Accounting Model**: How to properly model longs, shorts, leverage, and cash constraints
2. **Execution Sequencing**: The exact event flow from signal → decision → order → fill → account update
3. **Liquidity & Market Impact**: How to handle thin markets, zero-volume bars, and realistic fills
4. **Architecture**: Whether to build this ourselves or adopt an existing framework

---

## Current State

### What Works
- **Performance**: 32x faster than Backtrader, comparable to VectorBT
- **Validation**: 0.39% PnL difference vs Backtrader on 250 assets × 1 year × 11,230 trades
- **Simplicity**: Single-file implementation (744 lines)
- **Test Coverage**: 76% coverage, 17/17 tests passing

### Critical Gap Discovered
The validation study revealed a fundamental flaw: **no cash constraint enforcement**. When trying to open 25 long + 25 short positions simultaneously (50 × $20k = $1M), the engine:
- ✓ VectorBT: Correctly limits orders to available cash → -$3.9k total P&L
- ✗ engine: Executes all orders regardless of cash → -$652k total P&L (massive debt)

This exposed the need for a complete rethinking of the accounting and execution model.

---

## Part 1: Accounting Model Design Questions

### 1.1 Fundamental Account Structure

**Question**: What is the correct accounting identity for a trading account with long/short positions?

**Current thinking**:
```
Account Value = Cash + Long Market Value - Short Market Value - Short Obligations
```

Where:
- `Cash` = Available cash (deposits - withdrawals + realized P&L)
- `Long Market Value` = Σ(long_qty × current_price)
- `Short Market Value` = Σ(|short_qty| × current_price)
- `Short Obligations` = Amount owed to cover short positions

**Issues**:
1. Is "Short Market Value" double-counting? Should it be:
   ```
   Account Value = Cash + Σ(qty × price)  # where qty can be negative
   ```
2. How do we track the cash received from short sales vs the obligation to return shares?
3. When shorting generates cash, does that cash become immediately available for longs?

### 1.2 Leverage Definition

**Question**: How should we define and enforce leverage limits?

**Option A: Gross Exposure**
```python
gross_exposure = |long_value| + |short_value|
leverage = gross_exposure / account_value
max_leverage = 2.0  # 200% gross exposure
```

**Option B: Net Leverage**
```python
net_exposure = long_value - short_value
leverage = net_exposure / cash
```

**Option C: Buying Power**
```python
# Margin account style
buying_power = cash + (cash × margin_multiplier)
available = buying_power - maintenance_margin_required
```

**Our preference**: Option A (gross exposure) because it's simpler and more conservative. But we need validation that this matches real-world broker constraints.

### 1.3 Cash Flow from Exits

**Question**: When we exit a position, how do we calculate available cash for new entries?

**Scenario**:
1. Long 100 shares @ $100 = $10,000 position
2. Price rises to $110
3. Sell 100 shares @ $110
4. After 0.1% commission ($11) and 0.05% slippage ($5.50)
5. Net proceeds = $10,983.50

**Questions**:
- Is this $10,983.50 immediately available for new positions?
- Or is there a settlement delay (T+1, T+2)?
- For same-bar execution, can we use exit proceeds to fund entries on the same bar?
- For next-bar execution, exit proceeds from close are available at next open?

### 1.4 Short Sale Mechanics

**Question**: What are the exact cash flows for short selling?

**Example short sale**:
1. Short 100 shares @ $100
2. Cash received = $10,000 - commission - slippage = $9,983.50
3. Obligation created = 100 shares @ $100
4. Is the $9,983.50 available for long positions?
5. What happens to this cash when price moves?

**Follow-up**:
- If we use short proceeds for longs, are we leveraged 2:1?
- Do we need to track "short proceeds" separately from "available cash"?
- How do margin requirements affect this (e.g., Reg T 50% initial margin)?

---

## Part 2: Execution Sequencing Design Questions

### 2.1 Event Timeline (Daily Frequency)

**Question**: What is the exact sequence of events for daily trading?

**Current thinking**:
```
Day N:
  16:00 - Market closes at price P_close
  16:00-09:30 - Decision window
    - Strategy receives bar data (OHLCV for day N)
    - Strategy generates signals
    - Strategy submits orders

Day N+1:
  09:30 - Market opens at price P_open(N+1)
  09:30 - Orders execute
    - Exit orders fill first
    - Cash becomes available
    - Entry orders fill with updated cash
  09:30-16:00 - Market trades
  16:00 - Market closes
```

**Questions**:
1. Is this the correct mental model?
2. Should ALL exits fill before ANY entries (to maximize available cash)?
3. What if we have simultaneous exit+entry for the same asset (position reversal)?
4. Does order of signal generation matter (ASSET_001 before ASSET_250)?

### 2.2 Order Processing Sequence

**Question**: Within a single timestamp, what is the correct order of operations?

**Option A: Exit-First (No Leverage)**
```python
1. Process all exit orders (sells) → update cash
2. Calculate available cash after exits
3. Process all entry orders (buys) up to available cash
4. Reject orders that exceed cash
```

**Option B: Simultaneous (With Leverage)**
```python
1. Process all orders simultaneously
2. Calculate resulting leverage
3. Reject orders that exceed leverage limit
```

**Option C: FIFO**
```python
1. Process orders in submission order
2. Update cash after each order
3. Reject order if insufficient cash/leverage
```

**Our preference**: Option A for no-leverage mode, Option B for leverage mode. But this requires two different execution paths.

### 2.3 Position Reversal Edge Case

**Question**: How do we handle going from long → short or short → long on the same bar?

**Scenario**:
- Currently long 100 shares of AAPL
- New signal says short 100 shares
- Net change = sell 200 shares

**Questions**:
1. Should this be one order (sell 200) or two (close long 100, open short 100)?
2. Does the answer change if we're enforcing cash constraints?
3. How do we calculate P&L: one trade or two?

### 2.4 Partial Fills

**Question**: Should we support partial fills when cash/leverage is insufficient?

**Example**:
- Want to buy 100 shares @ $100 = $10,000
- Only have $5,000 cash
- Options:
  A. Partial fill: buy 50 shares
  B. Reject entire order
  C. Scale all orders proportionally to fit within cash

**VectorBT behavior**: Appears to do proportional scaling (Option C)
**Backtrader behavior**: Needs investigation
**Our preference**: ??? (Need guidance)

---

## Part 3: Liquidity & Market Impact Design Questions

### 3.1 Zero-Volume Bars

**Question**: How do we execute orders when historical data shows zero volume?

**Real-world context**:
- Futures markets trade 24 hours with electronic liquidity
- Historical bars may show zero volume during overnight hours
- But real traders CAN execute overnight with market impact

**Options**:
A. **Reject orders**: Can't trade when volume = 0
B. **Use last valid price**: Execute at previous close
C. **Model synthetic liquidity**: Assume minimum volume threshold
D. **Use bid-ask spread**: Model from OHLC range

**Current issue**: Our validation uses random data with non-zero volume, but real futures data will have zero-volume bars overnight.

### 3.2 Market Impact Model

**Question**: Do we need market impact for a realistic backtest?

**Arguments FOR**:
- Large orders (even modest size) in thin overnight markets have significant impact
- Ignoring impact overstates performance for strategies that trade frequently
- Critical for institutional-size positions

**Arguments AGAINST**:
- Adds complexity
- Hard to calibrate without real transaction data
- Most backtests ignore it

**Previous decision**: We removed market impact module to simplify. But validation revealed we need it for realistic modeling.

**Specific question**: For a $20k order in an overnight futures market, what's a reasonable impact estimate?

### 3.3 Slippage vs Impact

**Question**: What's the difference between slippage and market impact, and how do we model each?

**Current thinking**:
- **Slippage**: Percentage-based cost applied to all orders (e.g., 0.05%)
- **Market Impact**: Price movement from order size relative to volume/liquidity

**Questions**:
1. Are these independent (additive) or related?
2. Should impact feed back into fills (e.g., price moves 0.1%, subsequent orders pay more)?
3. For multi-asset strategies, do we model cross-asset impact (large sell affects correlated assets)?

### 3.4 Liquidity Constraints

**Question**: Should we limit order size based on volume?

**Common rule**: Don't exceed 10% of bar volume

**Issues**:
1. What if we're trading futures that trade 24/7 but daily bars aggregate all volume?
2. Should constraint be on notional value or share count?
3. What happens if order exceeds limit: partial fill or reject?

---

## Part 4: Architecture Design Questions

### 4.1 Build vs Buy

**Question**: Should we build this accounting/execution engine or adopt an existing framework?

**Frameworks we've validated**:

| Framework | Speed | PnL Match | Pros | Cons |
|-----------|-------|-----------|------|------|
| **Backtrader** | 1x | 0.39% ✓ | Mature, realistic | Slow (32x), complex API |
| **VectorBT** | 1.7x | N/A* | Very fast, vectorized | No leverage modeling* |
| **engine** | 32x | 0.39% ✓ | Fastest, simple | Missing accounting, liquidity |

*VectorBT comparison failed because engine lacks cash constraints

**Options**:
A. **Extend engine**: Add proper accounting, execution, liquidity
B. **Fork Backtrader**: Keep the accounting, rewrite for performance
C. **Adopt VectorBT**: Use as-is or contribute enhancements
D. **Hybrid**: Use VectorBT for simple strategies, engine for complex

### 4.2 Realistic vs Fast

**Question**: What's the right tradeoff between realism and speed?

**Use cases**:
1. **Rapid prototyping**: Test 100s of strategy variants quickly → need speed
2. **Pre-production validation**: Realistic assessment before live trading → need realism
3. **Research**: Publish-quality results → need realism + transparency

**Current state**: engine optimizes for speed but sacrifices realism. This validation study proves we can't skip realism.

### 4.3 API Design Philosophy

**Question**: What should the user-facing API look like?

**Option A: Event-Driven (Current)**
```python
class MyStrategy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        # Explicit order submission
        if signal > 0:
            broker.submit_order(asset, quantity, OrderSide.BUY)
```

**Option B: Declarative (VectorBT-style)**
```python
# User provides signals, engine handles execution
pf = Portfolio.from_signals(
    close=prices,
    entries=long_signals,
    size=0.02,
    size_type='percent',
)
```

**Option C: Hybrid**
```python
class MyStrategy(Strategy):
    def generate_signals(self, timestamp, data):
        return {"AAPL": 1.0, "MSFT": -1.0}  # long/short/flat

    # Engine handles order generation, fills, cash management
```

**Our preference**: Option A for flexibility, but many users prefer Option B's simplicity.

---

## Part 5: Specific Implementation Questions

### 5.1 Cash Update Timing

**Code-level question**: When do we update cash after a fill?

**Option A: Immediate**
```python
def _execute_fill(self, order, fill_price):
    commission = self.commission_model.calculate(...)
    total_cost = fill_price * quantity + commission
    self.cash -= total_cost  # Immediate update
    self._update_position(...)
```

**Option B: End-of-Bar**
```python
def _execute_fill(self, order, fill_price):
    self.pending_cash_updates.append(-total_cost)
    # Cash updated at end of bar
```

**Question**: Which is correct for same-bar vs next-bar execution?

### 5.2 Leverage Check Timing

**Question**: Do we check leverage before or after executing fills?

**Option A: Pre-Fill**
```python
def submit_order(self, order):
    if self._would_violate_leverage(order):
        return RejectedOrder(reason="Leverage limit")
    self.pending_orders.append(order)
```

**Option B: Post-Fill (Pessimistic)**
```python
def _process_orders(self):
    for order in pending:
        self._execute_fill(order)
        if self._is_over_leveraged():
            self._reverse_fill(order)  # Undo
```

### 5.3 Commission/Slippage Application

**Question**: In what order do we apply commission and slippage?

**Current code**:
```python
fill_price = base_price + slippage  # Slippage moves price
commission = commission_model.calculate(quantity, fill_price)  # Commission on filled price
total_cost = fill_price * quantity + commission
```

**Is this correct?** Or should commission be calculated on base price, then slippage on notional?

---

## Part 6: External Review Checklist

We're seeking expert feedback on:

### Design Validation
- [ ] Is the proposed accounting model correct and complete?
- [ ] Is the execution sequencing realistic and implementable?
- [ ] Are we missing any critical edge cases?
- [ ] Is the leverage model appropriate for institutional trading?

### Implementation Guidance
- [ ] Should we build this or adopt an existing framework?
- [ ] What's the minimum viable scope for "realistic enough"?
- [ ] Which features can be deferred to v2 (e.g., partial fills, T+2 settlement)?
- [ ] Are there regulatory/compliance concerns we're unaware of?

### Architecture Review
- [ ] Is the current 744-line codebase a good foundation or should we start over?
- [ ] What's the right abstraction boundary between "engine" and "accounting"?
- [ ] Should we separate simulation from backtesting (to support live trading)?
- [ ] How do production trading systems handle these same issues?

### Testing Strategy
- [ ] How do we validate realistic accounting (no real-world ground truth)?
- [ ] Should we match VectorBT, Backtrader, both, or real broker fills?
- [ ] What test scenarios would reveal accounting bugs?
- [ ] Do we need Monte Carlo testing of edge cases (bankruptcy, margin call)?

---

## Appendix A: Current Code Structure

The current implementation (744 lines) in `src/ml4t/backtest/engine.py`:

```python
# Core classes (174 lines)
@dataclass
class Order:
    """Represents a trading order"""

@dataclass
class Position:
    """Represents a position in an asset"""

@dataclass
class Fill:
    """Represents an executed fill"""

# Accounting (96 lines)
class Broker:
    def __init__(self, initial_cash, commission_model, slippage_model):
        self.cash = initial_cash
        self.positions = {}
        # NO CASH CONSTRAINT CHECKING

    def submit_order(self, asset, quantity, side):
        # Accepts all orders

    def _execute_fill(self, order, fill_price):
        # Fills without checking cash
        # BUG: Can go negative cash

# Execution (118 lines)
class Engine:
    def run(self):
        for timestamp, data in feed:
            # Process orders (no cash checks)
            broker._process_orders()
            # Strategy generates new orders
            strategy.on_data(timestamp, data, broker)
            # Process new orders (no cash checks)
            broker._process_orders()
```

**Key insight**: Minimal design made it fast but unrealistic. Adding proper accounting will add complexity but is necessary.

---

## Appendix B: Validation Results

Full validation study available in `tests/validation/test_engine_validation.py`.

**Test scenario**: 250 assets, 252 days, 25 long + 25 short positions daily

| Framework | PnL | Trades | Runtime | vs engine |
|-----------|-----|--------|---------|--------------|
| **Backtrader** | -$647,604 | 11,230 | 18.7s | 0.39% diff, 32x slower |
| **engine (next-bar)** | -$650,115 | 11,230 | 0.59s | - |
| **VectorBT OSS** | -$3,891 | 11,321 | 3.44s | 99.4% diff (cash limits) |
| **engine (same-bar)** | -$652,580 | 11,271 | 0.59s | - (no cash limits) |

**Key finding**: Backtrader enforces some cash limits (0.39% diff), VectorBT enforces strict limits (99.4% diff), engine enforces none (unlimited debt).

---

## Appendix C: Questions for Specific Reviewers

### For Trading System Architects:
1. How do production trading systems handle the exit-before-entry sequencing?
2. What's a realistic leverage limit for algorithmic futures trading?
3. Do you model settlement delays (T+1/T+2) or assume instant settlement?

### For Quantitative Researchers:
1. What level of execution realism is needed for published research?
2. Do you match a reference framework (like Backtrader) or model from first principles?
3. How do you calibrate market impact without proprietary transaction data?

### For Framework Developers (Backtrader, VectorBT, etc.):
1. How did you approach the accounting model design?
2. What were the hardest edge cases to get right?
3. Would you do anything differently if starting over?

---

## Contact & Review Process

**Repository**: `/home/stefan/ml4t/software/backtest/`
**Current implementation**: `src/ml4t/backtest/engine.py` (744 lines)
**Validation tests**: `tests/validation/test_engine_validation.py`
**Documentation**: This file + inline docstrings

**Preferred review format**:
1. Written feedback on design questions (Parts 1-4)
2. Architectural recommendations (Part 4)
3. Suggested reading / reference implementations
4. Offer to discuss specific technical details

**Timeline**: Seeking feedback within 2 weeks to inform development roadmap.

---

**Thank you for your expert review!**
