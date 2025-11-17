# Risk Management Exploration Report
## ml4t.backtest Event-Driven Backtesting Engine

**Date**: 2025-11-17
**Work Unit**: 009_risk_management_exploration
**Status**: Completed
**Thoroughness Level**: Very Thorough

---

## Executive Summary

This exploration analyzes requirements for incorporating richer, context-dependent risk management into ml4t.backtest. The analysis identifies a **hybrid architectural approach** combining a new RiskManager component with composable, callable-based rules that integrate cleanly with the existing event-driven engine.

**Key Findings**:
1. **Current Strengths**: Event-driven architecture provides clean integration points; pluggable models for commission/slippage already exist; bracket orders with percentage-based TP/SL are functional
2. **Critical Gaps**: No unified risk rules interface; no volatility-scaled/regime-dependent rules; no time-based exits beyond bracket orders; limited spread/volume-aware slippage models
3. **Recommended Design**: Lightweight RiskManager component with composable RiskRule implementations evaluated against immutable RiskContext snapshots
4. **Integration Strategy**: Three hook points in event loop (pre-signal, pre-order submission, post-fill) minimize coupling while enabling comprehensive risk management
5. **Implementation Path**: 7-week phased rollout maintaining full backward compatibility

---

## 1. Current Architecture Map

### 1.1 Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      BacktestEngine                              │
│  • Coordinates all components                                    │
│  • Manages Clock (event synchronization)                         │
│  • Optional: ContextCache for market-wide data                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
           ┌─────────────┴──────────────┬──────────────┬─────────────┐
           │                            │              │              │
    ┌──────▼──────┐           ┌────────▼────┐  ┌──────▼────┐  ┌─────▼──────┐
    │  DataFeed   │           │  Strategy   │  │  Broker   │  │ Portfolio  │
    │  (OHLCV)    │           │ (generates  │  │(executes) │  │(tracks P&L)│
    │  + signals  │           │   orders)   │  │           │  │            │
    └──────┬──────┘           └────────┬────┘  └─────┬─────┘  └──────┬─────┘
           │                           │              │              │
           │ Market                    │ Order       │ Fill         │
           │ Events                    │ Events      │ Events       │
           │ (OHLCV,                   │             │              │
           │  signals)                 │             │              │
           │                           │             │              │
           └───────────────────────────┼─────────────┼──────────────┘
                                       │             │
                                       ▼             ▼
                                  ┌────────────────────┐
                                  │    Clock (Event)   │
                                  │  Priority Queue    │
                                  │  Subscriptions     │
                                  └────────────────────┘
```

### 1.2 Current Event Flow

```
1. Clock yields next event (Market, Order, Fill, Corporate Action)
   ↓
2. Engine dispatches to subscribers (pub-sub via Clock)
   ├─ Strategy.on_event() → generates orders
   ├─ Broker.on_market_event() → processes pending orders
   ├─ Broker.on_order_event() → routes orders
   ├─ Portfolio.on_market_event() → updates position prices
   ├─ Portfolio.on_fill_event() → updates positions and cash
   └─ Reporter.on_event() → logs events
   ↓
3. FillSimulator applies transaction costs:
   ├─ Determines fill eligibility (OHLC bounds, liquidity)
   ├─ Applies market impact to price
   ├─ Calculates slippage (model-dependent)
   ├─ Applies commission (model-dependent)
   ├─ Updates order state
   └─ Generates FillEvent
   ↓
4. Filled orders create FillEvent
   ↓
5. All subscribers process FillEvent
```

### 1.3 Key Classes and Responsibilities

| Component | File | Responsibility | State Tracking |
|-----------|------|-----------------|-----------------|
| BacktestEngine | engine.py | Orchestrates simulation, injects components | Event stats |
| Clock | core/clock.py | Time management, event queue, subscriptions | Current time, event queue |
| Strategy | strategy/base.py | Generates signals/orders from market data | Internal state (user-defined) |
| SimulationBroker | execution/broker.py | Order execution, position tracking | Positions, cash, orders |
| FillSimulator | execution/fill_simulator.py | Realistic fill modeling | Fill count (for trade IDs) |
| Portfolio | portfolio/portfolio.py | Position tracking, P&L, analytics | Positions, cash, unrealized/realized P&L |
| TradeTracker | execution/trade_tracker.py | Entry/exit tracking, trade records | Completed trades, open positions |
| MarketEvent | core/event.py | Market data with optional signals | OHLCV, bid/ask, signals dict |
| Order | execution/order.py | Order specification and state | Order state, fill info |
| Position | portfolio/state.py | Single-asset position | Quantity, cost basis, unrealized P&L |

### 1.4 Current Risk Management Capabilities

**Existing Features** (PARTIAL support):
1. **Bracket Orders** (execution/order.py, execution/broker.py):
   - Percentage-based TP: `tp_pct` (e.g., 0.05 = 5%)
   - Percentage-based SL: `sl_pct` (e.g., 0.02 = 2%)
   - Trailing stop SL: `tsl_pct` (e.g., 0.01 = 1%)
   - Activation threshold: `tsl_threshold_pct`
   - Implementation: BracketOrderManager handles child order creation and monitoring

2. **Position Tracking** (portfolio/core.py, portfolio/state.py):
   - Current quantity, cost basis, last price
   - Unrealized P&L calculation
   - Realized P&L on exits

3. **Transaction Costs** (execution/broker.py, fill_simulator.py):
   - Commission models: NoCommission, FlatCommission, PercentageCommission, PerShareCommission, TieredCommission, MakerTakerCommission, AssetClassCommission, InteractiveBrokersCommission
   - Slippage models: NoSlippage, FixedSlippage, PercentageSlippage, LinearImpactSlippage, SquareRootImpactSlippage, VolumeShareSlippage, AssetClassSlippage
   - Market impact: MarketImpactModel (basic interface)

4. **Context Support** (core/context.py, engine.py):
   - ContextCache for market-wide data (VIX, SPY, etc.)
   - Passed to strategy via context parameter in on_market_event()

**Missing Features**:
- Volatility-scaled TP/SL (beyond fixed percentages)
- Regime-dependent rules
- Dynamic trailing logic beyond bracket orders
- Time-based exits (max holding period, session-aware)
- Portfolio-level constraints (max daily loss, max drawdown)
- Rule composition/stacking
- Spread-aware slippage models
- Volume-aware slippage with participation rates
- Order-type-dependent slippage
- Unified risk rules interface

---

## 2. Integration Points Analysis

### 2.1 Where Risk Rules Should Hook Into Event Loop

#### Hook Point A: Pre-Signal Risk Constraints
**Timing**: Market event arrives, before strategy generates signal
**Purpose**: Portfolio-level constraints that prevent strategy from trading
**Example**: "Don't trade if daily loss exceeds threshold"

```python
# Pseudocode
market_event = clock.get_next_event()

# NEW: Check portfolio constraints
risk_flags = risk_manager.check_portfolio_constraints(market_event)

# Strategy consults flags before generating signal
if not risk_flags['daily_loss_exceeded']:
    strategy.on_market_event(market_event)
```

**Pros**: Prevents wasteful signal generation, clear separation
**Cons**: Requires strategy cooperation, may miss intermediate opportunities
**Status**: Nice-to-have, low priority

---

#### Hook Point B: Order Validation (PRE-SUBMISSION) ⭐ RECOMMENDED
**Timing**: Strategy generates order, before broker.submit_order()
**Purpose**: Validate order against risk rules, update TP/SL, check constraints
**Example**: "Adjust stop loss based on current volatility"

```python
# Pseudocode
order = strategy.on_market_event(market_event)

# NEW: Risk manager validates and potentially modifies order
if order:
    validated_order = risk_manager.validate_order(order, context)
    if validated_order:
        broker.submit_order(validated_order)
```

**Pros**: Clean integration point, minimal engine changes, all rules in one place
**Cons**: Tight coupling to order submission timing
**Status**: HIGH priority, core feature

**Implementation Detail**:
```python
class RiskManager:
    def validate_order(
        self,
        order: Order,
        context: RiskContext,  # Current market/portfolio state
    ) -> Order | None:
        """
        Validate and potentially modify order before submission.
        
        Returns:
            - Modified order if valid
            - None if order rejected
        """
```

---

#### Hook Point C: Position Exit Checking (CONTINUOUS) ⭐ RECOMMENDED
**Timing**: Every market event, check all open positions
**Purpose**: Trigger exits for time-based, volatility-based, drawdown rules
**Example**: "Exit after 20 bars", "Trailing stop hit"

```python
# Pseudocode
for market_event in clock:
    # NEW: Check if any open positions trigger exit rules
    exit_orders = risk_manager.check_position_exits(
        market_event,
        broker,
        portfolio
    )
    
    for exit_order in exit_orders:
        broker.submit_order(exit_order)
    
    # Strategy continues with signal generation
    strategy.on_market_event(market_event)
```

**Pros**: Continuous monitoring, independent of strategy logic, enables time-based exits
**Cons**: Requires position tracking within risk manager
**Status**: HIGH priority, core feature

**Implementation Detail**:
```python
class RiskManager:
    def check_position_exits(
        self,
        market_event: MarketEvent,
        broker: Broker,
        portfolio: Portfolio,
    ) -> list[Order]:
        """
        Check all open positions against exit rules.
        
        Returns:
            List of exit orders for positions that violate rules
        """
```

---

#### Hook Point D: Position Fill Recording (ON FILL)
**Timing**: After fill event is created, before portfolio update
**Purpose**: Update rule state (entry prices, times, MFE/MAE)
**Example**: "Record entry time for max holding period calculation"

```python
# Pseudocode
fill_event = broker.try_fill_order(...)

# NEW: Risk manager records position state for rule evaluation
risk_manager.record_fill(fill_event, market_event)

# Portfolio and other subscribers process fill
broker.on_fill_event(fill_event)
```

**Pros**: Minimal overhead, maintains rule state at positions, enables MFE/MAE tracking
**Cons**: Requires care to avoid losing track of closed positions
**Status**: MEDIUM priority, enables advanced features

---

### 2.2 Recommended Hook Integration Strategy

Integrate all three hooks (B, C, D) in event loop:

```python
# In BacktestEngine.run() - main loop pseudocode
for market_event in self.clock:
    # Hook C: Check position exits FIRST
    exit_orders = self.risk_manager.check_position_exits(
        market_event, 
        self.broker, 
        self.portfolio
    )
    for exit_order in exit_orders:
        self.broker.submit_order(exit_order)
    
    # Strategy generates signal
    if self.strategy:
        self.strategy.on_market_event(market_event, context)
    
    # Hook B: Validate strategy orders
    for pending_order in self.broker.get_pending_orders():
        validated = self.risk_manager.validate_order(
            pending_order, 
            self._build_context(market_event)
        )
        if not validated:
            self.broker.cancel_order(pending_order.order_id)
    
    # Broker executes orders and generates fills
    fills = self.broker.on_market_event(market_event)
    
    # Hook D: Record fills for rule state tracking
    for fill in fills:
        self.risk_manager.record_fill(fill, market_event)
    
    # Propagate events to portfolio and reporter
    for fill in fills:
        self.portfolio.on_fill_event(fill)
    
    # Update position prices
    self.portfolio.on_market_event(market_event)
```

**Benefits**:
- Exits checked before new entries (prevents whipsaw)
- Rules applied after strategy but before execution (clean separation)
- Fill state recorded immediately after execution
- Minimal coupling to existing components

---

### 2.3 State Access Pattern for Risk Rules

Risk rules need read-only access to:
- Market state (OHLCV, bid/ask, volume)
- Position state (quantity, entry price, entry time, cost basis)
- Portfolio state (cash, total equity, realized/unrealized P&L)
- Feature data (volatility, regime, etc.)

**Design**: Immutable RiskContext snapshot passed to rules

```python
@dataclass(frozen=True)
class RiskContext:
    # Identifiers
    timestamp: datetime
    asset_id: AssetId
    
    # Market state (read-only snapshot)
    market_price: float
    high: float | None
    low: float | None
    close: float | None
    volume: float | None
    bid: float | None
    ask: float | None
    
    # Position state (aggregated from broker.position_tracker)
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

**Advantages**:
- Rules are pure functions of context
- Context is immutable → thread-safe
- No hidden dependencies on broker/portfolio
- Easy to test rules in isolation
- Rules can't modify system state (only return decisions)

---

## 3. Design Space Analysis

### 3.1 Architectural Approaches Evaluated

#### Approach 1: Separate RiskManager Component ⭐ RECOMMENDED

**Design**: New RiskManager class that strategies instantiate or engine injects

**Responsibilities**:
- Maintains list of registered RiskRule instances
- Called at strategic points in event loop
- Builds RiskContext from current state
- Evaluates rules and returns decisions
- Generates exit orders for triggered rules

**Integration**:
```python
class RiskManager:
    def __init__(self):
        self._rules: list[RiskRule] = []
        self._position_state: dict[AssetId, PositionTradeState] = {}
    
    def register_rule(self, rule: RiskRule) -> None:
        """Register a risk rule."""
    
    def check_position_exits(self, market_event, broker, portfolio) -> list[Order]:
        """Check if any rules trigger exit orders."""
    
    def validate_order(self, order, context) -> Order | None:
        """Validate order against rules."""
    
    def record_fill(self, fill_event, market_event) -> None:
        """Update rule state after fill."""
```

**Pros**:
- Clean separation of concerns
- Reusable across strategies
- Testable in isolation
- Composable (multiple rules)
- Optional feature (backward compatible)

**Cons**:
- Adds complexity to engine
- Requires careful state synchronization

**Recommendation**: ADOPT - Best balance of clarity and extensibility

---

#### Approach 2: Risk Rules as Order Decorators

**Design**: Orders carry risk configuration; broker evaluates on each market event

**Example**:
```python
order = Order(
    asset_id="AAPL",
    quantity=100,
    side=OrderSide.BUY,
    risk_rules=[
        VolatilityScaledStopLoss(atr_multiplier=2.0),
        TimeBasedExit(max_bars=20),
    ]
)
```

**Pros**:
- Minimal architecture change
- Leverages existing bracket order concept
- Rules travel with orders

**Cons**:
- Limited to per-order scope
- Hard to access portfolio-level state
- Rules scattered across order objects

**Recommendation**: REJECT - Too limited for portfolio-level constraints

---

#### Approach 3: Strategy Mixin Pattern

**Design**: Mixins provide helper methods for rule evaluation

**Example**:
```python
class MyStrategy(
    Strategy,
    VolatilityScaledRulesMixin,
    RegimeDependentRulesMixin,
    TimeBoundExitMixin,
):
    pass
```

**Pros**:
- Simple for strategy author
- Direct access to strategy state
- Backward compatible

**Cons**:
- Rules scattered across strategy class
- Hard to reuse rules across strategies
- Difficult to test rules in isolation
- Coupling between strategy and rules

**Recommendation**: REJECT - Poor separation of concerns

---

#### Approach 4: Configuration-Driven Rules (YAML/JSON)

**Design**: Risk rules defined in configuration files

**Example**:
```yaml
risk_rules:
  - type: volatility_scaled_stop_loss
    atr_multiplier: 2.0
  - type: time_based_exit
    max_bars: 20
```

**Pros**:
- Non-programmers can configure rules
- Easy to track and audit configs
- Simple versioning

**Cons**:
- Limited expressiveness for complex logic
- Requires interpretation layer
- Hard to support custom rules

**Recommendation**: DEFER - Phase 2 enhancement after RiskManager core

---

#### Approach 5: Lightweight Rule Registry

**Design**: Strategies register rule callables; evaluated at strategic points

**Example**:
```python
def my_rule(context: RiskContext) -> RiskDecision:
    if context.entry_bars_ago > 20:
        return RiskDecision(should_exit=True, reason="max bars exceeded")
    return RiskDecision()

risk_manager.register_rule(FunctionRule(my_rule))
```

**Pros**:
- Lightweight, minimal overhead
- Composable
- Testable
- Flexible

**Cons**:
- Requires discipline
- Less structure/guidance for users

**Recommendation**: ADOPT (as part of Approach 1) - Core mechanism for rule implementation

---

### 3.2 Recommended Hybrid Design

**Combine Approach 1 (RiskManager) + Approach 5 (Callable Rules)**

```
┌───────────────────────────────────────────┐
│         RiskManager (Component)           │
│  • Orchestrates rule evaluation           │
│  • Maintains position state tracking      │
│  • Generates exit orders                  │
│  • Validates strategy orders              │
└───────────────────────────────────────────┘
           │
           ├──► RiskRule (Abstract)
           │    ├─ VolatilityScaledStopLoss
           │    ├─ TimeBasedExit
           │    ├─ DynamicTrailingStop
           │    ├─ RegimeDependentRule
           │    ├─ FunctionRule (callable)
           │    └─ CompositeRule (stack rules)
           │
           ├──► RiskContext (Immutable snapshot)
           │    ├─ Market state
           │    ├─ Position state
           │    ├─ Portfolio state
           │    └─ Features
           │
           └──► RiskDecision (Output)
                ├─ should_exit
                ├─ exit_type
                ├─ update_tp/sl
                └─ metadata
```

**Benefits**:
- Clean architecture (RiskManager is orchestrator)
- Lightweight rules (can be simple callables)
- Extensible (users can implement RiskRule)
- Testable (rules as pure functions of context)
- Composable (stack multiple rules)
- Optional (backward compatible)

---

## 4. Gap Analysis

### 4.1 Missing Infrastructure by Requirement Area

| Requirement | Current State | Gap | Priority | Complexity |
|------------|--------------|-----|----------|------------|
| **Access to strategy/market state** | Position tracking exists | Need RiskContext abstraction | HIGH | Low |
| **Volatility-scaled TP/SL** | Fixed percentages only | Need context-aware scaling | HIGH | Medium |
| **Regime-dependent rules** | No mechanism | Need feature passing infrastructure | MEDIUM | Medium |
| **Dynamic trailing stops** | Basic bracket support | Need continuous monitoring | MEDIUM | Medium |
| **Time-based exits** | No infrastructure | Need bar counting, session awareness | MEDIUM | Medium |
| **Portfolio constraints** | No infrastructure | Need daily loss, max drawdown tracking | MEDIUM | Low |
| **Rule composition** | No mechanism | Need rule stacking framework | LOW | Low |
| **Spread-aware slippage** | Not implemented | Need bid/ask in slippage models | MEDIUM | Medium |
| **Volume-aware slippage** | Partial (VolumeShareSlippage exists) | Need participation rate models | LOW | Low |
| **Order-type-dependent TC** | Not implemented | Need order type awareness in models | MEDIUM | Low |
| **Configuration DSL** | Not implemented | Need config interpreter | LOW | High |

### 4.2 Implementation Roadmap by Priority

**Tier 1 (Phase 1: Core Infrastructure) - 1 week**
- [ ] RiskContext data structure (immutable snapshot)
- [ ] RiskRule abstract base class + RiskDecision
- [ ] RiskManager skeleton (register_rule, check_position_exits, validate_order, record_fill)
- [ ] Engine integration points (3 hooks)
- [ ] 5-10 basic rule implementations
- [ ] Unit tests + integration tests

**Tier 2 (Phase 2: Position Monitoring) - 1 week**
- [ ] RiskManager.check_position_exits() implementation
- [ ] Bar counter for position age tracking
- [ ] Session awareness (calendar support)
- [ ] TimeBasedExit rule implementations
- [ ] MFE/MAE tracking infrastructure
- [ ] End-to-end scenario tests

**Tier 3 (Phase 3: Order Validation) - 1 week**
- [ ] RiskManager.validate_order() implementation
- [ ] Portfolio constraint checking (daily loss, max drawdown, leverage)
- [ ] Position sizing rules
- [ ] Rule priority/conflict resolution
- [ ] Comprehensive tests

**Tier 4 (Phase 4: Slippage Enhancement) - 1 week**
- [ ] Refactor FillSimulator to accept MarketEvent (instead of separate OHLC)
- [ ] SpreadAwareSlippage model
- [ ] VolumeAwareSlippage model (enhance existing)
- [ ] OrderTypeDependentSlippage model
- [ ] Backward compatibility tests

**Tier 5 (Phase 5: Advanced Rules) - 2 weeks**
- [ ] VolatilityScaledStopLoss rule
- [ ] RegimeDependentRule rule
- [ ] DynamicTrailingStop rule
- [ ] PartialProfitTaking rule
- [ ] CompositeRule (rule composition)
- [ ] Extensive scenario tests

**Tier 6 (Phase 6: Configuration & Docs) - 1 week**
- [ ] Configuration DSL interpreter (YAML/JSON)
- [ ] Comprehensive examples for each rule type
- [ ] Migration guide for existing strategies
- [ ] Performance benchmarks
- [ ] API documentation

**Total: 7 weeks**

### 4.3 Key Missing Data Structures

```python
# NEW: Risk Context (RiskContext.py)
@dataclass(frozen=True)
class RiskContext:
    # 20+ fields as specified in Section 2.3
    pass

# NEW: Risk Decision (RiskDecision.py)
@dataclass
class RiskDecision:
    should_exit: bool = False
    exit_type: str | None = None
    exit_price: float | None = None
    update_tp: float | None = None
    update_sl: float | None = None
    reason: str = ""
    metadata: dict = field(default_factory=dict)

# NEW: Abstract Rule (RiskRule.py)
class RiskRule(ABC):
    @abstractmethod
    def evaluate(self, context: RiskContext) -> RiskDecision:
        pass

# NEW: Position Trade State (for rule tracking)
@dataclass
class PositionTradeState:
    asset_id: AssetId
    entry_time: datetime
    entry_price: float
    entry_quantity: float
    entry_bars: int  # Updated on each market event
    max_favorable_excursion: float
    max_adverse_excursion: float
    daily_pnl: float
    daily_max_loss: float
    trades_today: int
```

---

## 5. API Sketch and Example Scenarios

### 5.1 Core API Signatures

```python
# ============ RiskManager (main user interface) ============

class RiskManager:
    """Manages risk rules and position monitoring."""
    
    def register_rule(self, rule: RiskRule) -> None:
        """Register a risk rule to be evaluated."""
    
    def check_position_exits(
        self,
        market_event: MarketEvent,
        broker: Broker,
        portfolio: Portfolio,
    ) -> list[Order]:
        """Check all open positions against rules.
        
        Returns:
            List of exit orders for positions violating rules.
        """
    
    def validate_order(
        self,
        order: Order,
        context: RiskContext,
    ) -> Order | None:
        """Validate and potentially modify order.
        
        Returns:
            Modified order if valid, None if rejected.
        """
    
    def record_fill(
        self,
        fill_event: FillEvent,
        market_event: MarketEvent,
    ) -> None:
        """Update rule state after fill."""


# ============ RiskRule (user implements) ============

class RiskRule(ABC):
    """Abstract base for risk rules."""
    
    @abstractmethod
    def evaluate(self, context: RiskContext) -> RiskDecision:
        """Evaluate rule against context.
        
        Args:
            context: Immutable snapshot of current state
        
        Returns:
            RiskDecision with actions (exit, TP/SL updates, etc.)
        """


# ============ Example Implementations ============

class TimeBasedExit(RiskRule):
    """Exit position after max holding period."""
    
    def __init__(self, max_bars: int):
        self.max_bars = max_bars
    
    def evaluate(self, context: RiskContext) -> RiskDecision:
        if context.entry_bars_ago >= self.max_bars:
            return RiskDecision(
                should_exit=True,
                exit_type="immediate",
                reason=f"Max bars ({self.max_bars}) exceeded"
            )
        return RiskDecision()


class VolatilityScaledStopLoss(RiskRule):
    """Stop loss scaled to current volatility."""
    
    def __init__(self, atr_multiplier: float = 2.0):
        self.atr_multiplier = atr_multiplier
    
    def evaluate(self, context: RiskContext) -> RiskDecision:
        atr = context.features.get('atr', 0)
        if atr == 0:
            return RiskDecision()
        
        # SL is N ATRs below entry
        sl_price = context.entry_price - self.atr_multiplier * atr
        
        return RiskDecision(
            update_sl=sl_price,
            reason=f"Volatility-scaled SL: {self.atr_multiplier}x ATR"
        )


class RegimeDependentRule(RiskRule):
    """Different rules based on market regime."""
    
    def __init__(self):
        self.high_vol_params = {'sl_multiplier': 1.5, 'tp_multiplier': 0.75}
        self.low_vol_params = {'sl_multiplier': 3.0, 'tp_multiplier': 1.5}
    
    def evaluate(self, context: RiskContext) -> RiskDecision:
        regime = context.features.get('regime', 'neutral')
        atr = context.features.get('atr', 0)
        
        if regime == 'high_vol':
            params = self.high_vol_params
        else:
            params = self.low_vol_params
        
        sl = context.entry_price - params['sl_multiplier'] * atr
        tp = context.entry_price + params['tp_multiplier'] * atr
        
        return RiskDecision(
            update_sl=sl,
            update_tp=tp,
            reason=f"Regime-dependent rule ({regime})"
        )


class MaxDailyLossRule(RiskRule):
    """Prevent trading if daily loss exceeds threshold."""
    
    def __init__(self, max_loss_pct: float = 0.02):
        self.max_loss_pct = max_loss_pct
    
    def evaluate(self, context: RiskContext) -> RiskDecision:
        max_loss_amount = context.portfolio_equity * self.max_loss_pct
        
        if context.daily_max_loss >= max_loss_amount:
            return RiskDecision(
                should_exit=True,
                exit_type="flatten_all",
                reason=f"Daily max loss ({self.max_loss_pct:.1%}) exceeded"
            )
        return RiskDecision()


class DynamicTrailingStop(RiskRule):
    """Trailing stop that tightens over time."""
    
    def __init__(self, initial_trail: float, tighten_per_bar: float = 0.001):
        self.initial_trail = initial_trail
        self.tighten_per_bar = tighten_per_bar
    
    def evaluate(self, context: RiskContext) -> RiskDecision:
        # Trail amount decreases as position ages
        bars_held = context.entry_bars_ago
        trail = self.initial_trail - (bars_held * self.tighten_per_bar)
        trail = max(trail, 0.01)  # Minimum 1 bp
        
        mfe = context.max_favorable_excursion
        sl = mfe - (mfe * trail)
        
        return RiskDecision(
            update_sl=sl,
            reason=f"Dynamic trailing stop (trail={trail:.4f})"
        )
```

### 5.2 Simple Path (Backward Compatible)

```python
from ml4t.backtest import BacktestEngine, Strategy
from ml4t.backtest.execution.order import Order, OrderSide, OrderType

class SimpleStrategy(Strategy):
    """Basic strategy using bracket orders (no RiskManager needed)."""
    
    def on_market_event(self, event):
        if self.should_trade(event):
            # Existing bracket order API still works
            order = Order(
                asset_id=event.asset_id,
                quantity=100,
                side=OrderSide.BUY,
                order_type=OrderType.BRACKET,
                tp_pct=0.05,  # 5% take profit
                sl_pct=0.02,  # 2% stop loss
            )
            self.broker.submit_order(order)

# Run backtest WITHOUT RiskManager
engine = BacktestEngine(
    data_feed=feed,
    strategy=SimpleStrategy(),
    initial_capital=100000
)
results = engine.run()
```

### 5.3 Intermediate Path (Volatility-Scaled)

```python
from ml4t.backtest import BacktestEngine, Strategy
from ml4t.backtest.execution.risk import RiskManager, VolatilityScaledStopLoss
from ml4t.backtest.core.event import MarketEvent

class VolatilityStrategy(Strategy):
    """Strategy using volatility-scaled stops via RiskManager."""
    
    def on_start(self, **kwargs):
        # Create and configure risk manager
        self.risk_manager = RiskManager()
        self.risk_manager.register_rule(
            VolatilityScaledStopLoss(atr_multiplier=2.0)
        )
    
    def on_market_event(self, event: MarketEvent):
        # Strategy just generates entry signals
        # Exit rules are handled by RiskManager
        if self.should_trade(event):
            order = Order(
                asset_id=event.asset_id,
                quantity=100,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
            )
            self.broker.submit_order(order)
    
    def should_trade(self, event):
        # Simplified logic
        return event.close > event.open

# Run backtest WITH RiskManager
engine = BacktestEngine(
    data_feed=feed,
    strategy=VolatilityStrategy(),
    risk_manager=RiskManager(),  # NEW: Inject risk manager
    initial_capital=100000
)
results = engine.run()
```

### 5.4 Advanced Path (Custom Rules)

```python
from ml4t.backtest import BacktestEngine, Strategy
from ml4t.backtest.execution.risk import (
    RiskManager,
    RiskRule,
    RiskContext,
    RiskDecision,
    VolatilityScaledStopLoss,
    TimeBasedExit,
    RegimeDependentRule,
    MaxDailyLossRule,
)

class CustomExitRule(RiskRule):
    """Custom rule: exit if price touches upper band."""
    
    def evaluate(self, context: RiskContext) -> RiskDecision:
        upper_band = context.features.get('upper_band')
        if upper_band and context.market_price >= upper_band:
            return RiskDecision(
                should_exit=True,
                reason="Price touched upper band"
            )
        return RiskDecision()

class AdvancedStrategy(Strategy):
    """Strategy with stacked rules and custom logic."""
    
    def on_start(self, **kwargs):
        self.risk_manager = RiskManager()
        
        # Stack multiple rules
        self.risk_manager.register_rule(
            VolatilityScaledStopLoss(atr_multiplier=2.0)
        )
        self.risk_manager.register_rule(
            TimeBasedExit(max_bars=20)
        )
        self.risk_manager.register_rule(
            RegimeDependentRule()
        )
        self.risk_manager.register_rule(
            MaxDailyLossRule(max_loss_pct=0.05)
        )
        self.risk_manager.register_rule(
            CustomExitRule()  # User-defined
        )
    
    def on_market_event(self, event):
        # Entry logic unaware of exit rules
        if self._ml_signal_strong(event):
            # Feeds events with features (ATR, regime, bands)
            order = Order(...)
            self.broker.submit_order(order)
    
    def _ml_signal_strong(self, event):
        # User's ML logic
        confidence = event.signals.get('ml_confidence', 0)
        return confidence > 0.7

# Engine orchestrates everything
engine = BacktestEngine(
    data_feed=data_feed_with_features,  # Must include ATR, regime, bands
    strategy=AdvancedStrategy(),
    risk_manager=RiskManager(),
    initial_capital=100000
)
results = engine.run()
```

### 5.5 Configuration Path (Future, Phase 2)

```yaml
# risk_config.yaml
risk_rules:
  - type: volatility_scaled_stop_loss
    name: vol_stops
    atr_multiplier: 2.0
    asset_filter: [AAPL, MSFT]
  
  - type: time_based_exit
    name: max_holding
    max_bars: 20
  
  - type: regime_dependent
    name: regime_rules
    high_vol_config:
      sl_multiplier: 1.5
      tp_multiplier: 0.75
    low_vol_config:
      sl_multiplier: 3.0
      tp_multiplier: 1.5
  
  - type: max_daily_loss
    name: daily_limit
    max_loss_pct: 0.05
    action: flatten

portfolio_constraints:
  max_leverage: 2.0
  max_position_pct: 0.05
  max_daily_trades: 10
```

```python
from ml4t.backtest.execution.risk import RiskConfigLoader

# Load from config
config_loader = RiskConfigLoader()
risk_manager = config_loader.load_config('risk_config.yaml')

engine = BacktestEngine(
    data_feed=feed,
    strategy=strategy,
    risk_manager=risk_manager,
    initial_capital=100000
)
```

---

## 6. Next Steps and Recommendations

### 6.1 Implementation Roadmap (7 weeks)

**Phase 1: Core Infrastructure (Week 1)** - START HERE
- [ ] Create RiskContext dataclass (immutable snapshot)
- [ ] Create RiskRule abstract base + RiskDecision
- [ ] Create RiskManager skeleton
- [ ] Add 3 engine hook points
- [ ] Implement 5-10 basic rules
- [ ] 100+ unit tests

**Phase 2: Position Monitoring (Week 2)**
- [ ] Complete RiskManager.check_position_exits()
- [ ] Bar counter for position age
- [ ] Session awareness
- [ ] TimeBasedExit implementations
- [ ] Integration tests with examples

**Phase 3: Order Validation (Week 3)**
- [ ] Complete RiskManager.validate_order()
- [ ] Portfolio constraints
- [ ] Position sizing rules
- [ ] Rule priority/conflict resolution

**Phase 4: Slippage Enhancement (Week 4)**
- [ ] Refactor FillSimulator to accept MarketEvent
- [ ] Spread-aware slippage
- [ ] Volume-aware models
- [ ] Order-type-dependent models

**Phase 5: Advanced Rules (Weeks 5-6)**
- [ ] Volatility-scaled TP/SL
- [ ] Regime-dependent rules
- [ ] Dynamic trailing stops
- [ ] Partial profit-taking
- [ ] Rule composition

**Phase 6: Configuration & Docs (Week 7)**
- [ ] Config DSL interpreter
- [ ] Comprehensive examples
- [ ] Migration guide
- [ ] Performance benchmarks
- [ ] API documentation

### 6.2 Key Dependencies Between Components

```
Phase 1 (Core)
    ↓
Phase 2 (Position Monitoring)
    ↓
Phase 3 (Order Validation)
    ↓
Phase 4 & 5 (Slippage + Rules) - Can run in parallel
    ↓
Phase 6 (Documentation)
```

### 6.3 Testing Strategy

**Unit Tests** (per rule):
- Test rules in isolation with synthetic RiskContext objects
- No engine/broker/portfolio needed
- Fast, deterministic

**Integration Tests** (per hook point):
- Test with real engine, broker, portfolio
- Verify hook points work correctly
- End-to-end position exit/entry flows

**Scenario Tests** (requirements validation):
- Replicate paper trading scenarios
- Validate against known-good results
- Benchmark against existing implementations

### 6.4 Backward Compatibility

**Zero breaking changes**:
1. RiskManager is optional (not injected by default)
2. Bracket orders continue to work as-is
3. All new features in new namespaces
4. Engine accepts optional risk_manager parameter
5. Strategy continues to work without risk_manager

**Migration path** (for advanced users):
1. Keep existing bracket orders
2. Optionally add RiskManager for additional rules
3. Gradually move exit logic from strategy to rules
4. Eventually: rules become primary exit mechanism

### 6.5 Performance Considerations

**Design for efficiency**:
- RiskContext: immutable dataclass (zero-copy passing)
- Rules: pure functions of context (stateless)
- Position state tracking: O(1) lookups
- Bar counting: integer increment per market event
- Exit order creation: lazy (only when rules trigger)

**Expected overhead** (per market event):
- RiskContext creation: ~0.1ms
- Rule evaluation: ~0.01ms per rule (10-20 rules typical)
- Exit order generation: ~0.1ms if triggered

**Total**: <1ms for 20 rules on modern hardware

---

## 7. Detailed Design Specifications

### 7.1 RiskContext Structure (Complete)

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from ml4t.backtest.core.types import AssetId

@dataclass(frozen=True)
class RiskContext:
    """Immutable snapshot of strategy and market state.
    
    Passed to RiskRule.evaluate() for decision-making.
    All rules operate on this read-only context.
    """
    
    # ===== Identifiers =====
    timestamp: datetime
    asset_id: AssetId
    
    # ===== Market State (Current Bar) =====
    market_price: float  # Latest price
    high: Optional[float]  # Bar high
    low: Optional[float]  # Bar low
    close: Optional[float]  # Bar close
    volume: Optional[float]  # Bar volume
    bid: Optional[float]  # Best bid
    ask: Optional[float]  # Best ask
    
    # ===== Position State (Current) =====
    position_quantity: float  # Signed quantity (+ long, - short)
    entry_price: float  # Average entry price
    entry_time: datetime  # When position opened
    entry_bars_ago: int  # Bars since entry (0 = current bar)
    unrealized_pnl: float  # Current P&L
    
    # ===== Price Extremes Since Entry =====
    max_favorable_excursion: float  # Best price since entry
    max_adverse_excursion: float  # Worst price since entry
    
    # ===== Trade History (Day-Level) =====
    realized_pnl: float  # P&L from closed positions today
    trades_today: int  # Number of closed trades today
    daily_pnl: float  # Total P&L (realized + unrealized) today
    daily_max_loss: float  # Largest single-trade loss today
    
    # ===== Portfolio State =====
    portfolio_equity: float  # Total portfolio value
    portfolio_cash: float  # Available cash
    current_leverage: float  # Current leverage (gross notional / equity)
    
    # ===== External Features (User-Provided) =====
    features: dict[str, float]  # {atr, volatility, regime, upper_band, ...}
```

### 7.2 RiskRule and RiskDecision

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

class RiskRule(ABC):
    """Abstract base class for risk rules.
    
    Rules are pure functions that evaluate context
    and return decisions (exit, TP/SL updates, etc.).
    
    Rules CANNOT modify system state.
    Rules CANNOT access broker/portfolio directly.
    Rules CANNOT generate random behavior (must be deterministic).
    """
    
    @abstractmethod
    def evaluate(self, context: RiskContext) -> "RiskDecision":
        """Evaluate this rule against current context.
        
        Args:
            context: Immutable snapshot of current state
        
        Returns:
            RiskDecision with recommended actions
        
        Raises:
            ValueError: If context is invalid
        """

@dataclass
class RiskDecision:
    """Output of rule evaluation.
    
    Contains recommended actions:
    - Exit position
    - Update take-profit price
    - Update stop-loss price
    - Custom metadata
    """
    
    # ===== Exit Decision =====
    should_exit: bool = False
    exit_type: Optional[str] = None  # "immediate", "limit", "stop", "flatten_all"
    exit_price: Optional[float] = None  # If exit_type="limit"
    
    # ===== Price Adjustments =====
    update_tp: Optional[float] = None  # New take-profit price
    update_sl: Optional[float] = None  # New stop-loss price
    
    # ===== Metadata =====
    reason: str = ""  # Human-readable explanation
    metadata: dict = field(default_factory=dict)  # Custom data
    
    # ===== Priority (for composition) =====
    priority: int = 0  # Higher = evaluated first
```

### 7.3 RiskManager Core Methods

```python
class RiskManager:
    """Orchestrates risk rule evaluation and position monitoring.
    
    Responsibilities:
    1. Register/manage risk rules
    2. Monitor open positions against rules
    3. Validate strategy orders
    4. Track position state for rule evaluation
    """
    
    def register_rule(self, rule: RiskRule, priority: int = 0) -> None:
        """Register a risk rule.
        
        Args:
            rule: RiskRule instance to evaluate
            priority: Evaluation priority (higher first)
        """
    
    def check_position_exits(
        self,
        market_event: MarketEvent,
        broker: Broker,
        portfolio: Portfolio,
    ) -> list[Order]:
        """Check all open positions against registered rules.
        
        Called every market event to monitor position exits.
        
        Args:
            market_event: Current market event with OHLCV data
            broker: Broker for position access
            portfolio: Portfolio for state access
        
        Returns:
            List of exit orders for positions violating rules
        """
    
    def validate_order(
        self,
        order: Order,
        context: RiskContext,
    ) -> Optional[Order]:
        """Validate and potentially modify order before submission.
        
        Called before broker.submit_order() to apply risk constraints.
        
        Args:
            order: Order to validate
            context: Current market/portfolio context
        
        Returns:
            - Modified order if valid
            - None if order should be rejected
        """
    
    def record_fill(
        self,
        fill_event: FillEvent,
        market_event: MarketEvent,
    ) -> None:
        """Update position state after fill.
        
        Called when position is filled to maintain rule state.
        
        Args:
            fill_event: Fill event from broker
            market_event: Associated market event
        """
    
    # Private methods (implementation details)
    def _build_context(
        self,
        market_event: MarketEvent,
        broker: Broker,
        portfolio: Portfolio,
        position_state: "PositionTradeState",
    ) -> RiskContext:
        """Build immutable context snapshot."""
    
    def _create_exit_order(
        self,
        asset_id: AssetId,
        quantity: float,
        decision: RiskDecision,
        market_event: MarketEvent,
    ) -> Order:
        """Create exit order from rule decision."""
```

### 7.4 Position State Tracking (Internal to RiskManager)

```python
@dataclass
class PositionTradeState:
    """Internal state for tracking open positions (RiskManager use only)."""
    
    asset_id: AssetId
    entry_time: datetime
    entry_price: float
    entry_quantity: float
    entry_bars: int = 0
    
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    
    daily_pnl: float = 0.0
    daily_max_loss: float = 0.0
    trades_today: int = 0
    
    def update_on_market_event(self, context: RiskContext) -> None:
        """Update MFE/MAE and bar count."""
        self.entry_bars += 1
        
        price = context.market_price
        if (entry_direction := 1 if context.position_quantity > 0 else -1) > 0:
            # Long position
            self.max_favorable_excursion = max(
                self.max_favorable_excursion,
                price - self.entry_price
            )
            self.max_adverse_excursion = min(
                self.max_adverse_excursion,
                price - self.entry_price
            )
        else:
            # Short position
            self.max_favorable_excursion = max(
                self.max_favorable_excursion,
                self.entry_price - price
            )
            self.max_adverse_excursion = min(
                self.max_adverse_excursion,
                self.entry_price - price
            )
```

---

## 8. Implementation Complexity Estimates

| Component | Estimated Lines | Complexity | Notes |
|-----------|-----------------|------------|-------|
| RiskContext | 50 | Low | Dataclass, immutable |
| RiskDecision | 30 | Low | Dataclass |
| RiskRule (abstract) | 15 | Low | Abstract base |
| RiskManager | 300 | Medium | Core logic, integration |
| PositionTradeState | 50 | Low | State tracking |
| TimeBasedExit | 20 | Low | Simple logic |
| VolatilityScaledStopLoss | 25 | Low | Math only |
| RegimeDependentRule | 40 | Low | Conditional logic |
| MaxDailyLossRule | 30 | Low | State checking |
| DynamicTrailingStop | 35 | Low | Math over time |
| Engine integration | 50 | Medium | 3 hook points |
| Unit tests | 500 | Medium | Comprehensive coverage |
| Integration tests | 300 | Medium | End-to-end scenarios |
| Documentation | 200 | Low | API docs + examples |
| **TOTAL** | **~1,700** | **Medium** | **1-2 weeks implementation** |

---

## 9. Trade-offs and Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Rule evaluation timing | Every market event | Continuous monitoring required for time-based/trailing exits |
| State immutability | RiskContext frozen | Prevents side effects, enables safe composition |
| Rule interface | Abstract class (not callable only) | More structured, easier for users to discover API |
| Position state storage | Internal to RiskManager | Separates concerns, rules stay stateless |
| Hook point count | 3 hooks (exits, validate, record) | Balances coverage vs complexity |
| Backward compatibility | Bracket orders unchanged | Eases adoption, no migration pressure |
| Feature passing | Via features dict | Flexible, extensible, no coupling |
| Configuration support | Phase 2 (deferred) | Core API first, config layer later |
| Portfolio constraints | Supported in Phase 3 | Builds on Phase 1-2 foundation |
| Performance | Lazy exit order creation | Only create orders when rules trigger |

---

## 10. Example Validation Scenarios

### Scenario 1: Simple Time-Based Exit
**Requirement**: Exit position after 20 bars

```python
rule = TimeBasedExit(max_bars=20)

# Market events 1-19: No exit
context_bar_10 = RiskContext(entry_bars_ago=10, ...)
assert rule.evaluate(context_bar_10).should_exit == False

# Market event 20: Exit triggered
context_bar_20 = RiskContext(entry_bars_ago=20, ...)
decision = rule.evaluate(context_bar_20)
assert decision.should_exit == True
assert decision.reason == "Max bars (20) exceeded"
```

### Scenario 2: Volatility-Scaled Stop Loss
**Requirement**: Stop loss moves with volatility (ATR)

```python
rule = VolatilityScaledStopLoss(atr_multiplier=2.0)

context = RiskContext(
    entry_price=100.0,
    features={'atr': 2.0},
    ...
)

decision = rule.evaluate(context)
assert decision.update_sl == 100.0 - (2.0 * 2.0)  # 96.0
```

### Scenario 3: Regime-Dependent Rules
**Requirement**: Different SL width based on regime

```python
rule = RegimeDependentRule()

# High volatility: tight stops
context_high_vol = RiskContext(
    entry_price=100.0,
    features={'atr': 2.0, 'regime': 'high_vol'},
    ...
)
decision = rule.evaluate(context_high_vol)
assert decision.update_sl == 100.0 - (1.5 * 2.0)  # 97.0

# Low volatility: wide stops
context_low_vol = RiskContext(
    entry_price=100.0,
    features={'atr': 2.0, 'regime': 'low_vol'},
    ...
)
decision = rule.evaluate(context_low_vol)
assert decision.update_sl == 100.0 - (3.0 * 2.0)  # 94.0
```

### Scenario 4: Max Daily Loss
**Requirement**: Stop trading after daily loss threshold

```python
rule = MaxDailyLossRule(max_loss_pct=0.05)

context = RiskContext(
    portfolio_equity=100000.0,
    daily_max_loss=5500.0,  # 5.5% loss
    ...
)

decision = rule.evaluate(context)
assert decision.should_exit == True
```

### Scenario 5: Dynamic Trailing Stop
**Requirement**: Trail tightens as position ages

```python
rule = DynamicTrailingStop(initial_trail=0.05, tighten_per_bar=0.001)

# Early in position (5 bars)
context_5bars = RiskContext(
    entry_price=100.0,
    entry_bars_ago=5,
    max_favorable_excursion=5.0,  # Up $5
    features={},
    ...
)
decision = rule.evaluate(context_5bars)
trail = 0.05 - (5 * 0.001)  # 0.045
sl = (100.0 + 5.0) - ((100.0 + 5.0) * 0.045)  # ~99.77
assert decision.update_sl == pytest.approx(sl)

# Later in position (15 bars)
context_15bars = RiskContext(
    entry_price=100.0,
    entry_bars_ago=15,
    max_favorable_excursion=5.0,
    features={},
    ...
)
decision = rule.evaluate(context_15bars)
trail = 0.05 - (15 * 0.001)  # 0.035
sl = (100.0 + 5.0) - ((100.0 + 5.0) * 0.035)  # ~99.84
assert decision.update_sl == pytest.approx(sl)
```

---

## 11. Risk Assessment

### Potential Risks and Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|-----------|
| Integration breaks existing bracket orders | High | Low | Comprehensive backward compat tests |
| Rules lag behind price movements | Medium | Medium | Ensure hooks run before order submission |
| Position state drift vs broker state | Medium | Medium | Sync position state from broker on each event |
| Rules interfere with each other | Medium | Low | Clear priority system, conflict detection |
| Performance degradation with many rules | Medium | Low | Lazy evaluation, benchmark early |
| User confusion with config vs API | Low | Medium | Clear documentation, migration guide |

---

## Conclusion

The recommended approach combines:
1. **Separate RiskManager component** for clean architecture
2. **Composable RiskRule implementations** for extensibility  
3. **Immutable RiskContext snapshots** for safety and testability
4. **Three strategic hook points** in the event loop
5. **Full backward compatibility** with existing code
6. **Phased 7-week rollout** balancing delivery and quality

This design enables users to progress from simple bracket orders to sophisticated, context-dependent risk management without breaking existing strategies or adding unwarranted complexity.

**Next Action**: Begin Phase 1 implementation with RiskContext, RiskRule, and RiskManager core infrastructure.
