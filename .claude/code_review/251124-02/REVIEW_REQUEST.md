# ml4t.backtest - External Code Review Request

**Date**: 2025-11-24
**Package**: ml4t.backtest v0.2.0
**Review Type**: Correctness, Feature Completeness, Design Quality
**Reviewer**: External (Gemini/Claude/GPT-4)

---

## Executive Summary

ml4t.backtest is a minimal event-driven backtesting engine for quantitative trading strategies. It aims to provide institutional-grade execution fidelity in ~2,800 lines of production code.

**Core Value Proposition**:
- Event-driven architecture matching live trading
- Point-in-time correctness (no look-ahead bias)
- Multi-asset support with realistic execution
- Framework compatibility (VectorBT, Backtrader, Zipline)
- Cash and margin account policies
- Validated against established frameworks (EXACT numeric matches)

---

## Review Objectives

We need expert review on:

1. **Correctness**: Is the implementation sound? Are there logical flaws?
2. **Feature Completeness**: What do BackTrader, Zipline, and VectorBT users expect that we're missing?
3. **Account Policies**: Is our cash vs margin distinction correct and complete?
4. **Edge Cases**: What scenarios might break or produce incorrect results?
5. **API Design**: Are there design choices we'll regret later?
6. **Performance**: Are there obvious performance bottlenecks?

---

## What We Provide

1. **README.md** - Comprehensive feature documentation (1,135 lines)
2. **backtest_src.xml** - Complete source code (33 files, ~2,800 lines)
3. **This review request** - Specific questions and context

---

## Core Architecture

### Event Loop

```python
for timestamp, asset_data, context in datafeed:
    # 1. Update broker time and prices
    broker._update_time(timestamp, prices, opens, volumes, signals)

    # 2. Process pending orders (exits first)
    if execution_mode == NEXT_BAR:
        broker._process_orders(use_open=True)
        strategy.on_data(timestamp, asset_data, context, broker)
    else:  # SAME_BAR
        broker._process_orders()  # Process from previous bar
        strategy.on_data(timestamp, asset_data, context, broker)
        broker._process_orders()  # Process orders submitted this bar
```

### Account System

**Two account policies** handle constraints:

| Feature | CashAccountPolicy | MarginAccountPolicy |
|---------|-------------------|---------------------|
| Shorts | Prohibited | Allowed |
| Leverage | None | 2x (Reg T default) |
| Buying Power | Cash available | (NLV - MM) / IM |
| Position Flip | N/A | Long → Short in single order |

### Position Tracking

**Exit-first processing**:
- Exit orders processed before entry orders
- Frees capital immediately
- Prevents "locked capital" in rebalancing

---

## Claimed Features

### 1. Order Types (Core)

- [x] Market orders
- [x] Limit orders
- [x] Stop orders (stop-loss, stop-limit)
- [x] Trailing stops
- [x] Bracket orders (entry + TP + SL)

**Implementation**: `broker.py` (~480 lines)

### 2. Execution Modes

- [x] Same-bar execution (orders fill at current bar)
- [x] Next-bar execution (orders fill at next bar's open)

**Implementation**: `engine.py` ExecutionMode enum

### 3. Account Policies

- [x] Cash account (no leverage, no shorts)
- [x] Margin account (leverage, shorts, position flip)
- [x] Buying power calculation
- [x] Maintenance margin tracking
- [x] Gatekeeper (order validation)

**Implementation**: `accounting/` module (6 files)

### 4. Commission & Slippage

- [x] Per-share commission
- [x] Percentage commission
- [x] Tiered commission
- [x] Combined commission
- [x] Fixed slippage
- [x] Percentage slippage
- [x] Volume-based slippage

**Implementation**: `models.py` (~250 lines)

### 5. Execution Realism

- [x] Volume participation limits (max % of bar volume)
- [x] Market impact modeling (linear, square-root)
- [x] Partial fills (fill what's possible, queue remainder)
- [x] OHLC-bounded fills (price must be in bar range)

**Implementation**: `execution/` module (5 files)

### 6. Portfolio Rebalancing

- [x] TargetWeightExecutor (rebalance to target weights)
- [x] RebalanceConfig (min trade value, tolerance)
- [x] ExecutionResult (success, orders, skipped)

**Implementation**: `execution/rebalancer.py` (~200 lines)

### 7. Analytics

- [x] EquityCurve tracking
- [x] TradeAnalyzer (win rate, profit factor, avg win/loss)
- [x] Performance metrics (Sharpe, Sortino, Calmar, max DD)

**Implementation**: `analytics/` module (3 files)

### 8. Market Calendar

- [x] pandas_market_calendars integration
- [x] Trading day filtering
- [x] Holiday detection
- [x] Intraday minute generation

**Implementation**: `calendar.py` (~300 lines)

### 9. Multi-Asset Support

- [x] Simultaneous trading across multiple assets
- [x] Unified position tracking
- [x] Per-asset signals
- [x] Market-wide context data

**Implementation**: Core engine design

### 10. Framework Compatibility

- [x] VectorBT preset (same-bar, fractional shares)
- [x] Backtrader preset (next-bar, integer shares)
- [x] Zipline preset (next-bar, per-share comm, volume slippage)

**Validation**: EXACT numeric matches on 4 scenarios each

---

## Key Questions for Review

### 1. Cash vs Margin Account Implementation

**Our Implementation**:

```python
class CashAccountPolicy:
    def validate_order(self, order, state, price):
        # Block shorts
        if quantity < 0 and current_position.quantity >= 0:
            raise InsufficientFundsError("Cash account cannot short")

        # Block if insufficient cash
        required = order_value + commission
        if state.cash < required:
            raise InsufficientFundsError()

class MarginAccountPolicy:
    def validate_order(self, order, state, price):
        # Allow shorts
        # Check buying power instead of cash
        required = order_value + commission
        buying_power = self._calculate_buying_power(state)
        if required > buying_power:
            raise InsufficientBuyingPowerError()

    def _calculate_buying_power(self, state):
        nlv = state.cash + sum(position.market_value)
        mm = sum(abs(position.market_value) * self.maintenance_margin)
        return (nlv - mm) / self.initial_margin
```

**Questions**:
1. Is the buying power formula correct for US equities (Reg T)?
2. Should maintenance margin be calculated differently for long vs short positions?
3. Are we handling position reversals correctly (long → short)?
4. What about interest charges on shorts? (We don't model this yet)
5. Should we track "short stock proceeds" separately from cash?

### 2. Position Flip Logic

**Scenario**: Long 100 shares, submit order for -200 shares

**Our Behavior**:
- Creates fill for -200 shares
- Updates position to -100 shares (short)
- Tracks new cost basis at flip price

**Questions**:
1. Is this correct? Should it be two separate trades (close long, open short)?
2. How do BackTrader/Zipline/VectorBT handle this?
3. Should commission be calculated on 100 (close) + 100 (new short)?

### 3. Exit-First Processing

**Our Implementation**:
```python
def _process_orders(self):
    # 1. Separate exits from entries
    exits = [o for o in self._pending_orders.values()
             if self._is_exit_order(o)]
    entries = [o for o in self._pending_orders.values()
               if not self._is_exit_order(o)]

    # 2. Process exits first
    for order in exits:
        self._try_fill(order)

    # 3. Update equity (freed capital now available)
    self._update_account_state()

    # 4. Process entries
    for order in entries:
        self._try_fill(order)
```

**Questions**:
1. Is this correct for same-bar rebalancing?
2. Should equity update happen between exit and entry processing?
3. What if an exit order partially fills? Should we still process entries?

### 4. Partial Fills

**Implementation**: VolumeParticipationLimit

```python
def apply_limit(self, order, bar_volume):
    max_fill = bar_volume * self.max_participation
    if order.quantity <= max_fill:
        return order.quantity, 0  # Fill entire order
    else:
        return max_fill, order.quantity - max_fill  # Partial, queue remainder
```

**Questions**:
1. Should partially filled orders stay in pending_orders or be marked PARTIALLY_FILLED?
2. How do we track "cumulative filled quantity" for orders that fill over multiple bars?
3. Should price improve/worsen for later fills?

### 5. Stop Order Fills

**Implementation**:
```python
def _check_stop_trigger(self, order, bar):
    if order.stop_level_basis == StopLevelBasis.CLOSE:
        # Use previous close as reference
        if bar.close >= order.stop_price:  # Trigger
            return self._fill_at_stop_price(order, bar)
    elif order.stop_level_basis == StopLevelBasis.HIGH_LOW:
        # Use intrabar high/low
        if bar.high >= order.stop_price or bar.low <= order.stop_price:
            return self._fill_at_stop_price(order, bar)
```

**Questions**:
1. Should stop fills happen at stop price or next available price?
2. Are we handling gap days correctly (open > stop price)?
3. Should trailing stops adjust during the bar or only at bar end?

### 6. Same-Bar Re-Entry

**Scenario**: Strategy closes position, then immediately re-enters same bar

```python
def on_data(self, timestamp, data, context, broker):
    position = broker.get_position("AAPL")
    if position and position.unrealized_pnl < -1000:
        broker.close_position("AAPL")  # Exit order

    # Later same bar...
    if some_condition:
        broker.submit_order("AAPL", 100)  # Re-entry
```

**Questions**:
1. Does exit order fill before re-entry is attempted?
2. Is the freed capital available for re-entry in same bar?
3. Is this realistic for daily bars? (Probably not)

### 7. Market Impact on Stop Orders

**Current Behavior**:
- Market impact applies to market orders
- Limit/stop orders fill at limit/stop price (no impact)

**Questions**:
1. Should stop orders that convert to market orders have impact?
2. Should large stop orders move the fill price?

---

## Feature Comparison vs Other Frameworks

### BackTrader Features

| Feature | BackTrader | ml4t.backtest | Notes |
|---------|------------|---------------|-------|
| Indicators | ✅ Built-in | ❌ User computes | Design choice: indicators in separate library |
| Analyzers | ✅ Built-in | ✅ TradeAnalyzer | We have basic trade analysis |
| Optimizers | ✅ Built-in | ❌ Not included | Users can use optuna/hyperopt |
| Data resamplers | ✅ Built-in | ❌ Not included | Users can use polars |
| Live trading | ✅ Built-in | ✅ Separate package | ml4t-live in development |
| Notifications | ✅ Email/SMS | ❌ Not included | |
| Cerebro API | ✅ | ❌ We use Engine | Different design philosophy |

**Question**: What BackTrader features do quantitative traders actually use daily?

### Zipline Features

| Feature | Zipline | ml4t.backtest | Notes |
|---------|---------|---------------|-------|
| Pipeline API | ✅ | ❌ Not included | Design choice: signals computed outside |
| Bundles | ✅ | ❌ Not included | Users manage data |
| Factors | ✅ Built-in | ❌ User computes | Separate library |
| Risk models | ✅ Built-in | ❌ Not included | |
| Slippage | ✅ Volume-based | ✅ Equivalent | VolumeShareSlippage |
| Commission | ✅ Per-share | ✅ Equivalent | PerShareCommission + tiers |
| Benchmark tracking | ✅ | ❌ Not included | Users can track SPY manually |

**Question**: Is the Pipeline API the main reason people use Zipline?

### VectorBT Features

| Feature | VectorBT | ml4t.backtest | Notes |
|---------|----------|---------------|-------|
| Vectorized execution | ✅ | ❌ Event-driven | Design choice |
| Portfolio optimization | ✅ Built-in | ✅ TargetWeightExecutor | Different approach |
| Indicators | ✅ TA-Lib | ❌ User computes | Separate library |
| Interactive plots | ✅ Plotly | ❌ Not included | Users can use any plotting lib |
| Flexible data | ✅ | ✅ Polars | Equivalent |
| Multi-asset | ✅ | ✅ | Equivalent |
| Fractional shares | ✅ | ✅ | Equivalent |
| Same-bar fills | ✅ | ✅ | Equivalent (validated) |

**Question**: Is event-driven execution a dealbreaker for VectorBT users?

---

## Known Limitations

### 1. No Options Support

- No Black-Scholes pricing
- No Greeks calculation
- No options strategies (spreads, straddles, etc.)

**Question**: Is this a critical gap for systematic traders?

### 2. No Futures Support (Yet)

- No contract specifications
- No roll logic
- No margin calculations (initial vs maintenance per contract)
- No cash-settled vs physical delivery

**Question**: What % of quantitative traders need futures? Can we punt to v0.3?

### 3. No Portfolio Constraints

- No sector exposure limits
- No correlation limits
- No turnover limits
- No max concentration

**Question**: Are these "nice to have" or "must have" for institutional use?

### 4. No Transaction Cost Analysis

- No implementation shortfall tracking
- No VWAP/TWAP benchmarking
- No market impact attribution

**Question**: Do retail/small fund traders need this?

### 5. No Risk Metrics During Backtest

- Metrics calculated only at end
- No intra-backtest risk limits (e.g., "stop trading if Sharpe < 0.5")
- No dynamic position sizing based on volatility

**Question**: Should these be strategy-level logic or engine-level?

---

## Specific Code Review Requests

### 1. Broker._try_fill() Logic

**File**: `broker.py:200-250`

**Request**: Review fill logic for correctness:
- Is OHLC boundary checking correct?
- Are stop orders handled properly?
- Is commission/slippage applied in right order?

### 2. AccountState Management

**File**: `accounting/account.py`

**Request**: Review position tracking:
- Is cost basis calculated correctly for averaging?
- Are fills properly updating positions?
- Is bars_held incremented correctly?

### 3. Gatekeeper Validation

**File**: `accounting/gatekeeper.py`

**Request**: Review order validation logic:
- Are all constraint violations caught?
- Is buying power check correct for margin accounts?
- Should we reject or queue orders that exceed limits?

### 4. TargetWeightExecutor

**File**: `execution/rebalancer.py`

**Request**: Review rebalancing logic:
- Are target weights converted to orders correctly?
- Is rounding handled properly?
- What if an asset has no price data?

### 5. ExecutionLimits + MarketImpact

**Files**: `execution/limits.py`, `execution/impact.py`

**Request**: Review interaction:
- If order is partially filled due to volume limit, is impact calculated on partial or full size?
- Should impact be compounded across multiple bars?

---

## Testing & Validation

### Our Test Coverage

```
tests/
├── test_core.py              # 154 tests (engine, broker, datafeed)
└── accounting/               # 50+ tests (policies, gatekeeper, account state)
```

### Framework Validation Results

| Framework | Scenario | Match Status |
|-----------|----------|--------------|
| VectorBT Pro | Long-only | ✅ EXACT |
| VectorBT Pro | Long-short | ✅ EXACT |
| VectorBT Pro | Stop-loss | ✅ EXACT |
| VectorBT Pro | Take-profit | ✅ EXACT |
| Backtrader | Long-only | ✅ EXACT |
| Backtrader | Long-short | ✅ EXACT |
| Backtrader | Stop-loss | ✅ EXACT |
| Backtrader | Take-profit | ✅ EXACT |
| Zipline | Long-only | ✅ Within tolerance |
| Zipline | Long-short | ✅ Within tolerance |
| Zipline | Stop-loss | ✅ Within tolerance |
| Zipline | Take-profit | ✅ Within tolerance |

**Validation Code**: `validation/` directory (per-framework scripts)

**Question**: Are there other validation scenarios we should test?

---

## Edge Cases to Consider

1. **Gap Days**: Stock opens way above/below stop price
2. **Low Volume**: Order size > bar volume
3. **Zero Volume Bars**: How to handle fills?
4. **Position Flip with Stops**: Long with stop-loss, order flips to short
5. **Multiple Stops Same Bar**: Multiple stop orders trigger simultaneously
6. **Rebalance with Insufficient Capital**: Target weights require more cash than available
7. **Short Squeeze**: Cannot close short position (no shares available to buy)
8. **Delisted Stock**: Position goes to zero
9. **Corporate Actions**: Stock split, dividend
10. **Market Halt**: How to handle?

**Question**: Which of these MUST be handled vs can be documented limitations?

---

## API Design Concerns

### 1. Order Submission Return Value

Current API:
```python
order = broker.submit_order("AAPL", 100)
# order.order_id immediately available
# order.status = PENDING
```

**Question**: Should we return Order or just order_id? Order object may not be fully populated yet.

### 2. Position Property vs Method

Current API:
```python
position = broker.get_position("AAPL")  # Method
positions = broker.positions              # Property
```

**Question**: Should both be methods? Both properties?

### 3. ExecutionResult.success Semantics

Current behavior:
```python
result = executor.execute(targets, data, broker)
if result.success:
    # All orders submitted
else:
    # Some orders skipped due to constraints
```

**Question**: Should "success" mean "all orders executed" or "no errors occurred"?

### 4. DataFeed Iteration

Current API:
```python
for timestamp, data, context in feed:
    # data: Dict[asset, Dict[field, value]]
    # context: Dict[field, value]
```

**Question**: Should we return a dataclass instead of tuple?

---

## Performance Concerns

### Current Performance

- ~100k events/sec (event-driven iteration)
- ~2,800 lines of source code
- 154 tests pass in ~0.5 seconds

### Known Bottlenecks

1. **Polars iteration**: Converting to dict on each bar
2. **Order validation**: Runs on every order submission
3. **Position tracking**: Deep copy on each update

**Question**: Are these real bottlenecks or premature optimization concerns?

---

## Request Summary

**Please analyze**:

1. ✅ **Correctness**: Are there bugs, logical flaws, incorrect formulas?
2. ✅ **Feature Gaps**: What do BackTrader/Zipline/VectorBT users expect that we lack?
3. ✅ **Account Policies**: Is cash vs margin implementation correct?
4. ✅ **Edge Cases**: Which edge cases will break the engine?
5. ✅ **API Design**: Will we regret these API choices?
6. ✅ **Missing Validations**: What scenarios should we test?

**Specific Focus Areas**:
- Cash vs margin account logic (most critical)
- Position flip handling
- Exit-first processing
- Partial fill handling
- Stop order execution

---

## Files Provided

1. **README.md** (this project directory)
2. **backtest_src.xml** (complete source code)
3. **This review request**

---

**Thank you for your thorough review!**
