# Risk Management API Reference

**Package:** `ml4t.backtest.risk`
**Version:** 1.0.0
**Last Updated:** November 2025

## Table of Contents

1. [Module Overview](#module-overview)
2. [RiskContext](#riskcontext)
3. [RiskDecision](#riskdecision)
4. [ExitType](#exittype)
5. [RiskRule](#riskrule)
6. [RiskRuleProtocol](#riskruleprotocol)
7. [CompositeRule](#compositerule)
8. [RiskManager](#riskmanager)
9. [PositionTradeState](#positiontradestate)
10. [PositionLevels](#positionlevels)
11. [Built-In Rules](#built-in-rules)
12. [Type Aliases](#type-aliases)

---

## Module Overview

The `ml4t.backtest.risk` module provides a composable risk management framework for backtesting.

### Import Structure

```python
# Main classes
from ml4t.backtest.risk import (
    RiskContext,      # State snapshot for risk evaluation
    RiskDecision,     # Risk rule output
    ExitType,         # Exit signal types
    RiskRule,         # Base class for rules
    RiskManager,      # Rule orchestrator
)

# Built-in rules
from ml4t.backtest.risk import (
    TimeBasedExit,           # Exit after N bars
    PriceBasedStopLoss,      # Price-based stop loss
    PriceBasedTakeProfit,    # Price-based take profit
)

# Advanced
from ml4t.backtest.risk.rule import (
    RiskRuleProtocol,  # Protocol for callable rules
    CompositeRule,     # Combine multiple rules
)

from ml4t.backtest.risk.manager import (
    PositionTradeState,  # Position tracking state
    PositionLevels,      # Stop/target levels
)
```

### Module Files

- `context.py` - RiskContext dataclass
- `decision.py` - RiskDecision and ExitType
- `rule.py` - RiskRule interface and CompositeRule
- `manager.py` - RiskManager orchestrator
- `rules/time_based.py` - TimeBasedExit
- `rules/price_based.py` - PriceBasedStopLoss, PriceBasedTakeProfit

---

## RiskContext

**File:** `ml4t.backtest.risk.context`

An immutable snapshot of all state needed for risk rule evaluation at a specific point in time.

### Class Definition

```python
@dataclass(frozen=True)
class RiskContext:
    """Immutable snapshot of position, market, and portfolio state."""

    # Event metadata
    timestamp: datetime
    asset_id: AssetId

    # Market prices (OHLCV)
    open: Optional[Price]
    high: Optional[Price]
    low: Optional[Price]
    close: Price
    volume: Optional[float]

    # Quote prices
    bid_price: Optional[Price]
    ask_price: Optional[Price]

    # Position state
    position_quantity: Quantity
    entry_price: Price
    entry_time: Optional[datetime]
    bars_held: int

    # Portfolio state
    equity: float
    cash: float
    leverage: float

    # Features
    features: dict[str, float]          # Per-asset features
    market_features: dict[str, float]   # Market-wide features
```

### Constructor

```python
RiskContext(
    timestamp: datetime,
    asset_id: AssetId,
    open: Optional[Price] = None,
    high: Optional[Price] = None,
    low: Optional[Price] = None,
    close: Price,
    volume: Optional[float] = None,
    bid_price: Optional[Price] = None,
    ask_price: Optional[Price] = None,
    position_quantity: Quantity = 0.0,
    entry_price: Price = 0.0,
    entry_time: Optional[datetime] = None,
    bars_held: int = 0,
    equity: float = 0.0,
    cash: float = 0.0,
    leverage: float = 0.0,
    features: dict[str, float] = None,
    market_features: dict[str, float] = None
)
```

**Note:** Typically created via `RiskContext.from_state()` factory method, not direct construction.

### Class Methods

#### from_state

```python
@classmethod
def from_state(
    cls,
    market_event: MarketEvent,
    position: Optional[Position] = None,
    portfolio: Optional[Portfolio] = None,
    feature_provider: Optional[FeatureProvider] = None,
    entry_time: Optional[datetime] = None,
    bars_held: int = 0,
) -> "RiskContext"
```

Build RiskContext from market event, position, and portfolio state.

**Args:**
- `market_event` (MarketEvent): MarketEvent with prices, signals, and context
- `position` (Optional[Position]): Optional Position object (None if no position)
- `portfolio` (Optional[Portfolio]): Optional Portfolio object (None to use defaults)
- `feature_provider` (Optional[FeatureProvider]): Optional FeatureProvider to fetch additional features
- `entry_time` (Optional[datetime]): Optional entry timestamp (None if no position or unknown)
- `bars_held` (int): Number of bars held (0 if no position)

**Returns:**
- `RiskContext`: Immutable context snapshot

**Example:**
```python
context = RiskContext.from_state(
    market_event=event,
    position=broker.get_position(asset_id),
    portfolio=broker.portfolio
)
```

### Properties

#### unrealized_pnl

```python
@cached_property
def unrealized_pnl(self) -> float
```

Unrealized P&L for current position in currency units.

**Returns:**
- `float`: Unrealized P&L (0.0 if no position)

**Formula:**
```python
unrealized_pnl = position_quantity * (close - entry_price)
```

#### unrealized_pnl_pct

```python
@cached_property
def unrealized_pnl_pct(self) -> float
```

Unrealized P&L as percentage of entry value.

**Returns:**
- `float`: Percentage return (0.0 if no position or zero entry price)

**Formula:**
```python
unrealized_pnl_pct = (close - entry_price) / entry_price
```

#### max_favorable_excursion

```python
@cached_property
def max_favorable_excursion(self) -> float
```

Maximum favorable excursion (MFE) - highest unrealized profit since entry.

**Returns:**
- `float`: MFE in currency units (0.0 if no position)

**Logic:**
- Uses tracked MFE if available (from `PositionTradeState` via `RiskManager`)
- Otherwise computes intra-bar MFE from current bar's high/low
- **Long positions:** `(high - entry_price) * quantity`
- **Short positions:** `(entry_price - low) * abs(quantity)`

**Note:** Tracked MFE is more accurate as it spans multiple bars.

#### max_adverse_excursion

```python
@cached_property
def max_adverse_excursion(self) -> float
```

Maximum adverse excursion (MAE) - lowest unrealized profit since entry.

**Returns:**
- `float`: MAE in currency units (0.0 if no position)

**Logic:**
- Uses tracked MAE if available (from `PositionTradeState` via `RiskManager`)
- Otherwise computes intra-bar MAE from current bar's high/low
- **Long positions:** `(low - entry_price) * quantity`
- **Short positions:** `(entry_price - high) * abs(quantity)`

**Note:** Tracked MAE is more accurate as it spans multiple bars.

#### max_favorable_excursion_pct

```python
@cached_property
def max_favorable_excursion_pct(self) -> float
```

MFE as percentage of entry value.

**Returns:**
- `float`: MFE percentage (0.0 if no position or zero entry price)

**Formula:**
```python
mfe_pct = max_favorable_excursion / (abs(position_quantity) * entry_price)
```

#### max_adverse_excursion_pct

```python
@cached_property
def max_adverse_excursion_pct(self) -> float
```

MAE as percentage of entry value.

**Returns:**
- `float`: MAE percentage (0.0 if no position or zero entry price)

**Formula:**
```python
mae_pct = max_adverse_excursion / (abs(position_quantity) * entry_price)
```

#### current_price

```python
@property
def current_price(self) -> Price
```

Current market price (alias for `close`).

**Returns:**
- `Price`: Current market price (same as `close`)

**Usage:** Semantic alias for readability in rule code.

### Attributes

All attributes are public and immutable (frozen dataclass).

**Event Metadata:**
- `timestamp` (datetime): Event timestamp
- `asset_id` (AssetId): Asset identifier

**Market Prices:**
- `open` (Optional[Price]): Open price (None if not bar data)
- `high` (Optional[Price]): High price (None if not bar data)
- `low` (Optional[Price]): Low price (None if not bar data)
- `close` (Price): Close/last price
- `volume` (Optional[float]): Volume (None if not bar data)

**Quote Prices:**
- `bid_price` (Optional[Price]): Best bid price (None if not quote data)
- `ask_price` (Optional[Price]): Best ask price (None if not quote data)

**Position State:**
- `position_quantity` (Quantity): Current position quantity (0 if no position)
- `entry_price` (Price): Average entry price (0.0 if no position)
- `entry_time` (Optional[datetime]): First entry timestamp (None if no position)
- `bars_held` (int): Number of bars since entry (0 if no position)

**Portfolio State:**
- `equity` (float): Total portfolio equity (cash + positions)
- `cash` (float): Available cash
- `leverage` (float): Current leverage ratio

**Features:**
- `features` (dict[str, float]): Per-asset numerical features (from `MarketEvent.signals`)
- `market_features` (dict[str, float]): Market-wide features (from `MarketEvent.context`)

### Usage Examples

```python
# Check position direction
if context.position_quantity > 0:
    print("Long position")
elif context.position_quantity < 0:
    print("Short position")
else:
    print("No position")

# Access market data
current_price = context.close
intraday_range = context.high - context.low if context.high and context.low else 0

# Access unrealized P&L
pnl = context.unrealized_pnl
pnl_pct = context.unrealized_pnl_pct

# Access per-asset features
atr = context.features.get('atr_20', 0.0)
rsi = context.features.get('rsi_14', 50.0)
ml_score = context.features.get('ml_score', 0.0)

# Access market-wide features
vix = context.market_features.get('vix', 15.0)
spy_return = context.market_features.get('spy_return', 0.0)

# Access excursions (lazy - computed only when accessed)
mfe_pct = context.max_favorable_excursion_pct
mae_pct = context.max_adverse_excursion_pct
```

---

## RiskDecision

**File:** `ml4t.backtest.risk.decision`

Immutable decision output from a risk rule evaluation.

### Class Definition

```python
@dataclass(frozen=True)
class RiskDecision:
    """Immutable decision output from a risk rule evaluation."""

    should_exit: bool = False
    exit_type: Optional[ExitType] = None
    exit_price: Optional[Price] = None
    update_stop_loss: Optional[Price] = None
    update_take_profit: Optional[Price] = None
    reason: str = ""
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    asset_id: Optional[AssetId] = None
```

### Constructor

```python
RiskDecision(
    should_exit: bool = False,
    exit_type: Optional[ExitType] = None,
    exit_price: Optional[Price] = None,
    update_stop_loss: Optional[Price] = None,
    update_take_profit: Optional[Price] = None,
    reason: str = "",
    priority: int = 0,
    metadata: dict[str, Any] = None,
    asset_id: Optional[AssetId] = None
)
```

**Validation:**
- If `should_exit=True`, `exit_type` must be specified
- If `exit_type` is not None, `should_exit` must be True

### Factory Methods

#### no_action

```python
@classmethod
def no_action(
    cls,
    reason: str = "No action required",
    metadata: Optional[dict[str, Any]] = None,
    asset_id: Optional[AssetId] = None,
) -> "RiskDecision"
```

Create a no-action decision.

**Args:**
- `reason` (str): Explanation for no action (default: "No action required")
- `metadata` (Optional[dict]): Additional context
- `asset_id` (Optional[AssetId]): Asset this decision applies to

**Returns:**
- `RiskDecision`: Decision with all action flags set to False/None

**Example:**
```python
decision = RiskDecision.no_action(
    reason="Position within risk limits",
    metadata={"mae_pct": 0.02, "unrealized_pnl": 150.0}
)
```

#### exit_now

```python
@classmethod
def exit_now(
    cls,
    exit_type: ExitType,
    reason: str,
    exit_price: Optional[Price] = None,
    priority: int = 10,
    metadata: Optional[dict[str, Any]] = None,
    asset_id: Optional[AssetId] = None,
) -> "RiskDecision"
```

Create an immediate exit decision.

**Args:**
- `exit_type` (ExitType): Type of exit (STOP_LOSS, TAKE_PROFIT, etc.)
- `reason` (str): Explanation for exit
- `exit_price` (Optional[Price]): Specific exit price (None = use market order)
- `priority` (int): Priority for conflict resolution (default: 10)
- `metadata` (Optional[dict]): Additional context
- `asset_id` (Optional[AssetId]): Asset this decision applies to

**Returns:**
- `RiskDecision`: Decision with `should_exit=True`

**Example:**
```python
decision = RiskDecision.exit_now(
    exit_type=ExitType.STOP_LOSS,
    reason="Price fell below stop-loss at $95.00",
    exit_price=Decimal("95.00"),
    metadata={"breach_pct": 0.05}
)
```

#### update_stops

```python
@classmethod
def update_stops(
    cls,
    update_stop_loss: Optional[Price] = None,
    update_take_profit: Optional[Price] = None,
    reason: str = "",
    priority: int = 5,
    metadata: Optional[dict[str, Any]] = None,
    asset_id: Optional[AssetId] = None,
) -> "RiskDecision"
```

Create a decision to update stop-loss or take-profit levels.

**Args:**
- `update_stop_loss` (Optional[Price]): New stop-loss price (None = no change)
- `update_take_profit` (Optional[Price]): New take-profit price (None = no change)
- `reason` (str): Explanation for update
- `priority` (int): Priority for conflict resolution (default: 5)
- `metadata` (Optional[dict]): Additional context
- `asset_id` (Optional[AssetId]): Asset this decision applies to

**Returns:**
- `RiskDecision`: Decision with updated stop levels

**Raises:**
- `ValueError`: If both `update_stop_loss` and `update_take_profit` are None

**Example:**
```python
decision = RiskDecision.update_stops(
    update_stop_loss=Decimal("98.50"),
    reason="Trailing stop raised to lock in profit",
    metadata={"previous_sl": Decimal("96.00")}
)
```

### Class Methods

#### merge

```python
@classmethod
def merge(
    cls,
    decisions: list["RiskDecision"],
    default_priority_order: Optional[list[ExitType]] = None,
) -> "RiskDecision"
```

Merge multiple risk decisions into a single decision.

**Merging Logic:**
1. Exit decisions take precedence over stop updates and no-action
2. Among exit decisions, highest priority wins
3. If priorities are equal, use `default_priority_order` (if provided)
4. Stop-loss/take-profit updates use highest priority values
5. Metadata is merged (later decisions override earlier ones)

**Args:**
- `decisions` (list[RiskDecision]): List of RiskDecisions to merge
- `default_priority_order` (Optional[list[ExitType]]): Optional ordering of exit types for tie-breaking

**Returns:**
- `RiskDecision`: Single merged decision

**Raises:**
- `ValueError`: If decisions list is empty

**Example:**
```python
decision1 = RiskDecision.update_stops(update_stop_loss=Decimal("98.50"))
decision2 = RiskDecision.exit_now(exit_type=ExitType.STOP_LOSS, priority=10)
decision3 = RiskDecision.no_action()

merged = RiskDecision.merge([decision1, decision2, decision3])
# Result: EXIT takes precedence
assert merged.should_exit == True
assert merged.exit_type == ExitType.STOP_LOSS
```

### Instance Methods

#### is_action_required

```python
def is_action_required(self) -> bool
```

Check if this decision requires any action.

**Returns:**
- `bool`: True if decision requires exit or stop update, False otherwise

**Example:**
```python
if decision.is_action_required():
    print(f"Action needed: {decision.reason}")
```

#### __str__

```python
def __str__(self) -> str
```

Human-readable representation of the decision.

**Returns:**
- `str`: Formatted string representation

**Example:**
```python
print(str(decision))
# Output: "RiskDecision(EXIT: stop_loss, reason='Stop hit', priority=10)"
```

### Attributes

All attributes are public and immutable (frozen dataclass).

- `should_exit` (bool): Whether to exit the position immediately
- `exit_type` (Optional[ExitType]): Type of exit signal (if `should_exit=True`)
- `exit_price` (Optional[Price]): Suggested exit price (None = use market order)
- `update_stop_loss` (Optional[Price]): New stop-loss price (None = no change)
- `update_take_profit` (Optional[Price]): New take-profit price (None = no change)
- `reason` (str): Human-readable explanation of the decision
- `priority` (int): Priority for conflict resolution (higher = more important)
- `metadata` (dict[str, Any]): Additional context for logging/debugging
- `asset_id` (Optional[AssetId]): Asset this decision applies to (None = universal)

---

## ExitType

**File:** `ml4t.backtest.risk.decision`

Enumeration of exit signal types.

### Class Definition

```python
class ExitType(Enum):
    """Type of exit signal."""

    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    TIME_EXIT = "time_exit"
    RISK_EXIT = "risk_exit"
    OTHER = "other"
```

### Values

- `STOP_LOSS`: Protective stop triggered
- `TAKE_PROFIT`: Target price reached
- `TRAILING_STOP`: Trailing stop triggered
- `TIME_EXIT`: Maximum holding period reached
- `RISK_EXIT`: Risk event (VIX spike, regime change, etc.)
- `OTHER`: Custom exit reason

### Usage

```python
from ml4t.backtest.risk.decision import ExitType

decision = RiskDecision.exit_now(
    exit_type=ExitType.STOP_LOSS,
    reason="Stop loss breach"
)

if decision.exit_type == ExitType.STOP_LOSS:
    print("Stop loss triggered")
```

---

## RiskRule

**File:** `ml4t.backtest.risk.rule`

Abstract base class for risk management rules.

### Class Definition

```python
class RiskRule(ABC):
    """Abstract base class for risk management rules."""

    @abstractmethod
    def evaluate(self, context: RiskContext) -> RiskDecision:
        """Evaluate risk context and generate decision."""
        pass

    def validate_order(
        self, order: Order, context: RiskContext
    ) -> Optional[Order]:
        """Optional pre-execution order validation and modification."""
        return order  # Default: no validation

    @property
    def priority(self) -> int:
        """Priority for conflict resolution."""
        return 0  # Default: lowest priority

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"{self.__class__.__name__}(priority={self.priority})"
```

### Abstract Methods

#### evaluate

```python
@abstractmethod
def evaluate(self, context: RiskContext) -> RiskDecision
```

Evaluate risk context and generate decision.

**Args:**
- `context` (RiskContext): Snapshot of risk-relevant state

**Returns:**
- `RiskDecision`: Recommended action

**Implementation Requirements:**
- Must be implemented by all subclasses
- Should be stateless (no mutable internal state)
- Should be fast (called on every market event)
- Should return `RiskDecision.no_action()` if no action needed

**Example:**
```python
class MyStopLoss(RiskRule):
    def evaluate(self, context: RiskContext) -> RiskDecision:
        if context.unrealized_pnl_pct < -0.05:
            return RiskDecision.exit_now(
                exit_type=ExitType.STOP_LOSS,
                reason="5% stop loss breach"
            )
        return RiskDecision.no_action()
```

### Optional Methods

#### validate_order

```python
def validate_order(
    self, order: Order, context: RiskContext
) -> Optional[Order]
```

Optional pre-execution order validation and modification.

**Args:**
- `order` (Order): Order about to be submitted
- `context` (RiskContext): Current risk context for decision making

**Returns:**
- `Optional[Order]`: Modified order, original order (if valid), or None (to reject)

**Default Implementation:** Pass-through (no validation)

**Use Cases:**
- Reject orders during adverse conditions
- Modify order size based on risk constraints
- Add stop-loss/take-profit to orders

**Example:**
```python
class VIXFilter(RiskRule):
    def validate_order(self, order, context):
        vix = context.market_features.get('vix', 15.0)
        if vix > 30:
            return None  # Reject order
        return order  # Accept
```

### Properties

#### priority

```python
@property
def priority(self) -> int
```

Priority for conflict resolution when merging decisions.

**Returns:**
- `int`: Integer priority (default: 0)

**Priority Levels:**
- 0: Informational (default)
- 5: Stop updates (trailing stops, profit targets)
- 10: Critical exits (stop-loss breaches, risk limits)
- 15: Emergency exits (circuit breakers, system issues)

**Example:**
```python
class MyStopLoss(RiskRule):
    @property
    def priority(self) -> int:
        return 10  # High priority
```

### Usage

**Subclass Example:**
```python
class TrailingStop(RiskRule):
    """Trailing stop rule."""

    def __init__(self, trail_pct: float = 0.02):
        self.trail_pct = trail_pct

    def evaluate(self, context: RiskContext) -> RiskDecision:
        if context.position_quantity == 0:
            return RiskDecision.no_action()

        if context.max_favorable_excursion_pct > self.trail_pct:
            new_stop = context.close * (1 - self.trail_pct)
            return RiskDecision.update_stops(
                update_stop_loss=Decimal(str(new_stop)),
                reason=f"Trailing stop: {self.trail_pct:.2%}"
            )

        return RiskDecision.no_action()

    @property
    def priority(self) -> int:
        return 5
```

---

## RiskRuleProtocol

**File:** `ml4t.backtest.risk.rule`

Protocol for callable risk rules (allows functions without subclassing).

### Protocol Definition

```python
@runtime_checkable
class RiskRuleProtocol(Protocol):
    """Protocol for callable risk rules."""

    def __call__(self, context: RiskContext) -> RiskDecision:
        """Evaluate risk context and return decision."""
        ...
```

### Usage

```python
# Define a simple callable rule
def simple_stop_loss(context: RiskContext) -> RiskDecision:
    """5% stop loss as a simple function."""
    if context.unrealized_pnl_pct < -0.05:
        return RiskDecision.exit_now(
            exit_type=ExitType.STOP_LOSS,
            reason=f"Stop-loss breach: {context.unrealized_pnl_pct:.2%}"
        )
    return RiskDecision.no_action()

# Use directly with RiskManager (no class needed!)
risk_manager.add_rule(simple_stop_loss)
```

**Note:** Callable rules cannot implement `validate_order()` or `priority`. Use `RiskRule` subclass if you need those features.

---

## CompositeRule

**File:** `ml4t.backtest.risk.rule`

Composite rule that combines multiple sub-rules.

### Class Definition

```python
class CompositeRule(RiskRule):
    """Composite rule that combines multiple sub-rules."""

    def __init__(self, rules: list[RiskRule | RiskRuleProtocol]):
        """Initialize composite rule."""
        self.rules = rules

    def evaluate(self, context: RiskContext) -> RiskDecision:
        """Evaluate all sub-rules and merge their decisions."""
        ...

    def validate_order(
        self, order: Order, context: RiskContext
    ) -> Optional[Order]:
        """Run order validation through all sub-rules."""
        ...

    @property
    def priority(self) -> int:
        """Use maximum priority of all sub-rules."""
        ...
```

### Constructor

```python
CompositeRule(rules: list[RiskRule | RiskRuleProtocol])
```

**Args:**
- `rules` (list): List of sub-rules to evaluate

### Methods

#### evaluate

```python
def evaluate(self, context: RiskContext) -> RiskDecision
```

Evaluate all sub-rules and merge their decisions.

**Process:**
1. Call `evaluate()` on each sub-rule
2. Collect all decisions
3. Merge using `RiskDecision.merge()`

**Returns:**
- `RiskDecision`: Merged decision from all sub-rules

#### validate_order

```python
def validate_order(self, order: Order, context: RiskContext) -> Optional[Order]
```

Run order validation through all sub-rules.

**Process:**
- If any rule rejects the order (returns None), the order is rejected
- Otherwise, modifications are applied sequentially

**Returns:**
- `Optional[Order]`: Validated/modified order or None if rejected

### Properties

#### priority

```python
@property
def priority(self) -> int
```

Use maximum priority of all sub-rules.

**Returns:**
- `int`: Maximum priority among sub-rules (0 if no rules)

### Usage

```python
from ml4t.backtest.risk import CompositeRule

# Create a composite protective rule
protective = CompositeRule([
    PriceBasedStopLoss(stop_loss_price=Decimal("92.00")),
    PriceBasedTakeProfit(take_profit_price=Decimal("115.00")),
    TimeBasedExit(max_bars=20)
])

# Add all 3 rules at once
risk_manager.add_rule(protective)
```

---

## RiskManager

**File:** `ml4t.backtest.risk.manager`

Orchestrates risk rule evaluation and position monitoring.

### Class Definition

```python
class RiskManager:
    """Orchestrates risk rule evaluation and position monitoring."""

    def __init__(self, feature_provider: Optional[FeatureProvider] = None):
        """Initialize RiskManager."""
        self.feature_provider = feature_provider
        self._rules: list[Union[RiskRule, RiskRuleProtocol]] = []
        self._position_state: dict[AssetId, PositionTradeState] = {}
        self._position_levels: dict[AssetId, PositionLevels] = {}
        self._context_cache: dict[tuple[AssetId, datetime], RiskContext] = {}
```

### Constructor

```python
RiskManager(feature_provider: Optional[FeatureProvider] = None)
```

**Args:**
- `feature_provider` (Optional[FeatureProvider]): Optional provider for additional features beyond what's in `MarketEvent.signals` and `MarketEvent.context` dicts

**Example:**
```python
risk_manager = RiskManager()

# With feature provider
from ml4t.backtest.data.feature_provider import FeatureProvider
provider = MyFeatureProvider()
risk_manager = RiskManager(feature_provider=provider)
```

### Public Methods

#### add_rule

```python
def add_rule(self, rule: Union[RiskRule, RiskRuleProtocol]) -> None
```

Register a risk rule.

**Args:**
- `rule`: Risk rule to add (RiskRule subclass or callable matching Protocol)

**Raises:**
- `TypeError`: If rule doesn't implement RiskRule or RiskRuleProtocol

**Example:**
```python
risk_manager.add_rule(TimeBasedExit(max_bars=60))
risk_manager.add_rule(simple_stop_loss)  # Callable
```

#### remove_rule

```python
def remove_rule(self, rule: Union[RiskRule, RiskRuleProtocol]) -> None
```

Unregister a risk rule.

**Args:**
- `rule`: Rule to remove

**Raises:**
- `ValueError`: If rule not found

**Example:**
```python
stop_loss_rule = PriceBasedStopLoss(stop_loss_price=Decimal("95.00"))
risk_manager.add_rule(stop_loss_rule)
# Later...
risk_manager.remove_rule(stop_loss_rule)
```

#### check_position_exits

```python
def check_position_exits(
    self,
    market_event: MarketEvent,
    broker: Broker,
    portfolio: Portfolio,
) -> list[Order]
```

Check all open positions and generate exit orders if rules triggered.

**Hook:** C (BEFORE strategy)

**Process:**
1. Iterate through all open positions from broker
2. Build RiskContext for each (with caching)
3. Update position state (bars_held, MFE, MAE)
4. Evaluate all rules
5. Merge decisions
6. Generate exit order if should_exit=True

**Args:**
- `market_event` (MarketEvent): Current market event with prices and features
- `broker` (Broker): Broker for position lookup
- `portfolio` (Portfolio): Portfolio for equity and cash

**Returns:**
- `list[Order]`: Exit orders to submit (Market orders to close positions)

**Example:**
```python
# In BacktestEngine event loop (automatic)
exit_orders = risk_manager.check_position_exits(event, broker, portfolio)
for order in exit_orders:
    broker.submit_order(order)
```

#### validate_order

```python
def validate_order(
    self,
    order: Order,
    market_event: MarketEvent,
    broker: Broker,
    portfolio: Portfolio,
) -> Optional[Order]
```

Validate order against risk rules before execution.

**Hook:** B (AFTER strategy, BEFORE broker)

**Process:**
1. Build RiskContext for the order's asset
2. Call validate_order() on each rule that implements it
3. Return order if all rules pass, None if any rule rejects

**Args:**
- `order` (Order): Order to validate (from strategy)
- `market_event` (MarketEvent): Current market event
- `broker` (Broker): Broker for position lookup
- `portfolio` (Portfolio): Portfolio for equity and cash

**Returns:**
- `Optional[Order]`: Order if validated, None if rejected by rules

**Example:**
```python
# In BacktestEngine event loop (automatic)
strategy_order = strategy.on_market_data(event)
if strategy_order:
    validated = risk_manager.validate_order(strategy_order, event, broker, portfolio)
    if validated:
        broker.submit_order(validated)
```

#### record_fill

```python
def record_fill(self, fill_event: FillEvent, market_event: MarketEvent) -> None
```

Record fill event and update position state tracking.

**Hook:** D (AFTER fills)

**Process:**
1. If opening new position: create PositionTradeState
2. If closing position: remove PositionTradeState and PositionLevels
3. Update position quantities

**Args:**
- `fill_event` (FillEvent): Fill event from broker
- `market_event` (MarketEvent): Current market event

**Returns:** None (side-effect only)

**Example:**
```python
# In BacktestEngine event loop (automatic)
fills = broker.process_fills(event)
for fill in fills:
    risk_manager.record_fill(fill, event)
```

#### evaluate_all_rules

```python
def evaluate_all_rules(self, context: RiskContext) -> RiskDecision
```

Evaluate all registered rules and merge decisions.

**Process:**
1. Call evaluate() on each rule
2. Collect all decisions
3. Merge using RiskDecision.merge() with priority resolution

**Args:**
- `context` (RiskContext): RiskContext for current position/market state

**Returns:**
- `RiskDecision`: Merged RiskDecision from all rules

**Example:**
```python
context = RiskContext.from_state(event, position, portfolio)
decision = manager.evaluate_all_rules(context)
if decision.should_exit:
    print(f"Exit signal: {decision.reason}")
```

#### clear_cache

```python
def clear_cache(self, before_timestamp: Optional[datetime] = None) -> None
```

Clear context cache to free memory.

**Args:**
- `before_timestamp` (Optional[datetime]): If specified, only clear entries before this time. If None, clear entire cache.

**Example:**
```python
# Clear cache at end of day
if event.timestamp.hour == 16:  # Market close
    manager.clear_cache()
```

### Private Attributes

- `_rules` (list): List of registered risk rules
- `_position_state` (dict): Per-asset position tracking state
- `_position_levels` (dict): Per-asset stop-loss and take-profit levels
- `_context_cache` (dict): Context cache for performance optimization

### Performance

**Context Caching:**
- Caches `RiskContext` objects by `(asset_id, timestamp)`
- Reduces construction from O(n×m) to O(n) where n=positions, m=rules
- ~10x speedup with 10 rules
- Cache invalidated automatically on timestamp changes

**Overhead:**
- <3% overhead vs no risk manager (500 days, 3 rules)
- Scales linearly with rule count

---

## PositionTradeState

**File:** `ml4t.backtest.risk.manager`

Position tracking state for bar counting and MFE/MAE calculation.

### Class Definition

```python
@dataclass
class PositionTradeState:
    """Position tracking state."""

    asset_id: AssetId
    entry_time: datetime
    entry_price: Decimal
    entry_quantity: float
    bars_held: int = 0
    max_favorable_excursion: Decimal = Decimal("0.0")
    max_adverse_excursion: Decimal = Decimal("0.0")
```

### Methods

#### update_on_market_event

```python
def update_on_market_event(self, market_price: Decimal) -> None
```

Update bar counter and MFE/MAE on each market event.

**Args:**
- `market_price` (Decimal): Current market price (close price from MarketEvent)

**Logic:**
- Increments `bars_held`
- Updates `max_favorable_excursion` (best price movement in your favor)
- Updates `max_adverse_excursion` (worst price movement against you)
- **Long positions:** MFE = max upward move, MAE = max downward move
- **Short positions:** MFE = max downward move, MAE = max upward move

**Note:** This is an internal method called by `RiskManager.check_position_exits()`.

---

## PositionLevels

**File:** `ml4t.backtest.risk.manager`

Stop-loss and take-profit price levels for a position.

### Class Definition

```python
@dataclass
class PositionLevels:
    """Stop-loss and take-profit price levels."""

    asset_id: AssetId
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
```

**Note:** This is an internal class used by `RiskManager` to track position levels. Updated when rules return `RiskDecision` with `update_stop_loss` or `update_take_profit`.

---

## Built-In Rules

### TimeBasedExit

**File:** `ml4t.backtest.risk.rules.time_based`

Exit position after holding for a maximum number of bars.

#### Class Definition

```python
class TimeBasedExit(RiskRule):
    """Exit position after holding for maximum number of bars."""

    def __init__(self, max_bars: int):
        """Initialize TimeBasedExit rule."""
        if max_bars < 1:
            raise ValueError(f"max_bars must be >= 1, got {max_bars}")
        self.max_bars = max_bars

    def evaluate(self, context: RiskContext) -> RiskDecision:
        """Evaluate whether position should exit based on holding period."""
        ...

    @property
    def priority(self) -> int:
        return 5  # Medium priority
```

#### Parameters

- `max_bars` (int): Maximum number of bars to hold position (must be >= 1)

#### Priority

5 (medium - after critical stops but before take profits)

#### Example

```python
from ml4t.backtest.risk import TimeBasedExit

# Exit after 60 bars (60 days for daily data)
risk_manager.add_rule(TimeBasedExit(max_bars=60))
```

### PriceBasedStopLoss

**File:** `ml4t.backtest.risk.rules.price_based`

Exit position when price hits stop-loss level.

#### Class Definition

```python
class PriceBasedStopLoss(RiskRule):
    """Exit position when price hits stop-loss level."""

    def __init__(self, stop_loss_price: Optional[Price] = None):
        """Initialize PriceBasedStopLoss rule."""
        self.stop_loss_price = stop_loss_price

    def evaluate(self, context: RiskContext) -> RiskDecision:
        """Evaluate whether position should exit based on stop loss."""
        ...

    @property
    def priority(self) -> int:
        return 10  # High priority
```

#### Parameters

- `stop_loss_price` (Optional[Price]): Fixed stop price, or None to use position levels

#### Priority

10 (high - stop losses are critical)

#### Logic

- **Long positions:** Exit if `current_price <= stop_loss_price`
- **Short positions:** Exit if `current_price >= stop_loss_price`

#### Example

```python
from decimal import Decimal
from ml4t.backtest.risk import PriceBasedStopLoss

# Fixed stop at $95
risk_manager.add_rule(PriceBasedStopLoss(stop_loss_price=Decimal("95.00")))

# Dynamic stop from position levels
risk_manager.add_rule(PriceBasedStopLoss())
```

### PriceBasedTakeProfit

**File:** `ml4t.backtest.risk.rules.price_based`

Exit position when price hits take-profit level.

#### Class Definition

```python
class PriceBasedTakeProfit(RiskRule):
    """Exit position when price hits take-profit level."""

    def __init__(self, take_profit_price: Optional[Price] = None):
        """Initialize PriceBasedTakeProfit rule."""
        self.take_profit_price = take_profit_price

    def evaluate(self, context: RiskContext) -> RiskDecision:
        """Evaluate whether position should exit based on take profit."""
        ...

    @property
    def priority(self) -> int:
        return 8  # Medium-high priority
```

#### Parameters

- `take_profit_price` (Optional[Price]): Fixed target price, or None to use position levels

#### Priority

8 (medium-high - after stop losses but before time exits)

#### Logic

- **Long positions:** Exit if `current_price >= take_profit_price`
- **Short positions:** Exit if `current_price <= take_profit_price`

#### Example

```python
from decimal import Decimal
from ml4t.backtest.risk import PriceBasedTakeProfit

# Fixed take profit at $110
risk_manager.add_rule(PriceBasedTakeProfit(take_profit_price=Decimal("110.00")))

# Dynamic take profit from position levels
risk_manager.add_rule(PriceBasedTakeProfit())
```

---

## Type Aliases

```python
# Type alias for anything that can be used as a risk rule
RiskRuleLike = RiskRule | RiskRuleProtocol
```

---

## Module Exports

```python
__all__ = [
    # Core classes
    "RiskContext",
    "RiskDecision",
    "ExitType",
    "RiskRule",
    "RiskRuleProtocol",
    "CompositeRule",
    "RiskManager",

    # Built-in rules
    "TimeBasedExit",
    "PriceBasedStopLoss",
    "PriceBasedTakeProfit",

    # Internal classes (advanced usage)
    "PositionTradeState",
    "PositionLevels",
]
```

---

**Document Version:** 1.0.0
**Last Updated:** November 2025
**Author:** ml4t.backtest development team
