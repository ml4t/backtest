# Exploration Summary: Robust Accounting Logic

## Executive Summary

The backtest engine needs proper accounting constraints to support **both cash accounts** (simple, no leverage) and **margin accounts** (shorts allowed, leverage enabled). The current implementation allows unlimited debt (-$652k), failing validation by 99.4% vs VectorBT.

**Key Architectural Decision**: Use Policy pattern for account types rather than forcing margin accounting on all accounts.

```
AccountState + AccountPolicy
├── CashAccountPolicy: cash >= 0, no shorts, buying_power = cash
└── MarginAccountPolicy: NLV/BP/MM calculations, shorts allowed
```

## Codebase Analysis

### Current Implementation (812 lines)

**Structure:**
```
src/ml4t/backtest/
└── engine.py (744 lines)
    ├── Types: OrderType, OrderSide, OrderStatus, ExecutionMode
    ├── Models: Order, Position, Fill, Trade
    ├── Protocols: CommissionModel, SlippageModel
    ├── Implementations: NoCommission, PercentageCommission, etc.
    ├── DataFeed: Polars-based multi-asset feed
    ├── Broker: Order processing and position tracking
    └── Engine: Event loop orchestration
```

**Tests:**
```
tests/
├── test_core.py (17 tests, 100% passing)
└── validation/
    ├── test_validation.py (VectorBT comparison, fails 99.4%)
    └── test_vectorbt_pro.py (additional validation)
```

### Critical Findings

#### 1. **Missing Cash Constraint (Line 587)**
```python
cash_change = -signed_qty * fill_price - commission
self.cash += cash_change  # ❌ No validation! Can go negative!
```

**Impact**: Executes ALL orders regardless of available capital

#### 2. **Oversimplified Position Class**
```python
@dataclass
class Position:
    asset: str
    quantity: float
    entry_price: float      # ❌ Not weighted average
    entry_time: datetime
    bars_held: int
```

**Missing:**
- Weighted average cost basis (needed for adding to positions)
- Current price (for mark-to-market)
- Market value calculation
- Unrealized P&L

#### 3. **No Order Validation**
```python
def _execute_fill(self, order: Order, base_price: float):
    # Calculate slippage and commission
    # ...
    # ❌ No check before execution
    # Update position immediately
```

**Missing:**
- Pre-execution validation
- Order rejection logic
- Reason tracking for rejections

#### 4. **No Exit-First Sequencing**
```python
def process_pending_orders(self):
    for order in self.pending_orders:
        self._check_and_fill(order)  # ❌ Random order
```

**Impact**: Exits don't free capital before entries process

### What Works Well (Keep These)

✅ **DataFeed Design**: Polars-based, multi-asset, clean interface
✅ **Order Types**: MARKET, LIMIT, STOP, TRAILING_STOP (comprehensive)
✅ **Order Lifecycle**: PENDING → FILLED/REJECTED/CANCELLED (clear states)
✅ **Commission/Slippage**: Protocol-based, pluggable, multiple implementations
✅ **Fill Tracking**: Complete fill details with commission/slippage
✅ **Trade Recording**: Captures entry/exit, P&L, bars held
✅ **Execution Modes**: SAME_BAR vs NEXT_BAR (realistic timing)

## Implementation Approach

### Architecture: Policy Pattern for Account Types

**Core Insight**: Don't force margin accounting on everyone. Let users choose explicitly.

```python
# Cash account (simple)
broker = Broker(
    initial_cash=100000,
    account_type="cash",  # ← Explicit choice
)

# Margin account (complex)
broker = Broker(
    initial_cash=100000,
    account_type="margin",  # ← Different policy
    initial_margin=0.5,      # 2x leverage
    maintenance_margin=0.25,
)
```

### Module Structure

```
src/ml4t/backtest/
├── engine.py                 # Keep existing (with modifications)
└── accounting/               # NEW package
    ├── __init__.py           # Exports: Position, AccountState, Policies, Gatekeeper
    ├── models.py             # Position class (unified, enhanced)
    ├── policy.py             # AccountPolicy + implementations
    ├── account.py            # AccountState (owns positions, cash, delegates to policy)
    └── gatekeeper.py         # Order validation orchestrator
```

### Key Classes

#### 1. **Position (Unified)**
```python
@dataclass
class Position:
    asset: str
    quantity: float           # Positive=Long, Negative=Short
    avg_entry_price: float    # Weighted average cost basis
    current_price: float      # Mark-to-market price
    entry_time: datetime
    bars_held: int = 0

    @property
    def market_value(self) -> float:
        """Current value (positive for longs, negative for shorts)"""
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L"""
        return (self.current_price - self.avg_entry_price) * self.quantity
```

**Benefits:**
- Works for both long and short positions
- Tracks cost basis correctly
- Mark-to-market aware
- Replaces engine's simple Position class

#### 2. **AccountPolicy (Interface)**
```python
class AccountPolicy(ABC):
    @abstractmethod
    def calculate_buying_power(
        self,
        cash: float,
        positions: dict[str, Position]
    ) -> float:
        """How much capital can be deployed?"""
        pass

    @abstractmethod
    def allows_short_selling(self) -> bool:
        """Can positions go negative?"""
        pass

    @abstractmethod
    def validate_new_position(
        self,
        asset: str,
        quantity: float,
        price: float,
        current_positions: dict[str, Position],
        cash: float
    ) -> tuple[bool, str]:
        """Can we afford this position?"""
        pass
```

#### 3. **CashAccountPolicy (Implementation)**
```python
class CashAccountPolicy(AccountPolicy):
    def calculate_buying_power(self, cash, positions):
        return max(0.0, cash)  # Simple: buying power = cash

    def allows_short_selling(self):
        return False  # No shorts in cash accounts

    def validate_new_position(self, asset, quantity, price, positions, cash):
        if quantity < 0:  # Trying to short
            return False, "Cash accounts do not allow short selling"

        cost = quantity * price
        if cost > cash:
            return False, f"Insufficient cash: need ${cost:,.2f}, have ${cash:,.2f}"

        return True, "Approved"
```

#### 4. **MarginAccountPolicy (Implementation)**
```python
class MarginAccountPolicy(AccountPolicy):
    def __init__(self, initial_margin=0.5, maintenance_margin=0.25):
        self.initial_margin = initial_margin
        self.maintenance_margin = maintenance_margin

    def calculate_buying_power(self, cash, positions):
        # Calculate NLV
        nlv = cash + sum(p.market_value for p in positions.values())

        # Calculate maintenance margin used
        mm_used = sum(
            abs(p.market_value) * self.maintenance_margin
            for p in positions.values()
        )

        # Buying power = (NLV - MM_used) / initial_margin_rate
        excess_liquidity = nlv - mm_used
        return max(0.0, excess_liquidity / self.initial_margin)

    def allows_short_selling(self):
        return True  # Margin accounts allow shorts

    def validate_new_position(self, asset, quantity, price, positions, cash):
        # Calculate margin requirement for new position
        notional = abs(quantity * price)
        margin_required = notional * self.initial_margin

        # Check buying power
        bp = self.calculate_buying_power(cash, positions)
        if notional > bp:
            return False, f"Insufficient buying power: need ${notional:,.2f}, have ${bp:,.2f}"

        return True, "Approved"
```

#### 5. **AccountState (Ledger)**
```python
class AccountState:
    def __init__(self, initial_cash: float, policy: AccountPolicy):
        self.cash = initial_cash
        self.positions: dict[str, Position] = {}
        self.policy = policy

    @property
    def total_equity(self) -> float:
        """Net Liquidating Value"""
        return self.cash + sum(p.market_value for p in self.positions.values())

    @property
    def buying_power(self) -> float:
        """Available capital for new positions"""
        return self.policy.calculate_buying_power(self.cash, self.positions)

    def mark_to_market(self, prices: dict[str, float]):
        """Update position prices"""
        for asset, price in prices.items():
            if asset in self.positions:
                self.positions[asset].current_price = price

    def apply_fill(self, asset: str, signed_qty: float, price: float, commission: float):
        """Update positions and cash after fill"""
        # Update cash
        trade_cost = signed_qty * price
        self.cash -= (trade_cost + commission)

        # Update position
        if asset not in self.positions:
            if signed_qty != 0:
                self.positions[asset] = Position(
                    asset, signed_qty, price, price, datetime.now(), 0
                )
        else:
            pos = self.positions[asset]
            new_qty = pos.quantity + signed_qty

            if new_qty == 0:
                del self.positions[asset]  # Position closed
            else:
                # Update weighted average if increasing position
                if (pos.quantity > 0 and signed_qty > 0) or \
                   (pos.quantity < 0 and signed_qty < 0):
                    total_cost = (pos.quantity * pos.avg_entry_price) + (signed_qty * price)
                    pos.avg_entry_price = total_cost / new_qty

                pos.quantity = new_qty
                pos.current_price = price
```

#### 6. **Gatekeeper (Validator)**
```python
class Gatekeeper:
    def __init__(self, account: AccountState, commission_model):
        self.account = account
        self.commission_model = commission_model

    def validate_order(self, order, price: float) -> tuple[bool, str]:
        """Check if order can be executed"""
        # Calculate signed quantity
        signed_qty = order.quantity if order.side == OrderSide.BUY else -order.quantity

        # Check if reducing position (always allowed)
        pos = self.account.positions.get(order.asset)
        if pos and self._is_reducing(pos, signed_qty):
            return True, "Reducing risk"

        # Check if account allows this direction
        if signed_qty < 0 and not self.account.policy.allows_short_selling():
            return False, "Account does not allow short selling"

        # Estimate commission
        commission = self.commission_model.calculate(order.asset, order.quantity, price)

        # Validate via policy
        return self.account.policy.validate_new_position(
            order.asset,
            abs(signed_qty),
            price,
            self.account.positions,
            self.account.cash
        )

    def _is_reducing(self, pos: Position, signed_qty: float) -> bool:
        """Check if trade reduces position size"""
        return (pos.quantity > 0 and signed_qty < 0) or \
               (pos.quantity < 0 and signed_qty > 0)
```

### Integration with Broker

**Modified Broker.__init__():**
```python
def __init__(
    self,
    initial_cash: float = 100000.0,
    account_type: str = "cash",  # NEW: explicit account type
    initial_margin: float = 0.5,      # For margin accounts
    maintenance_margin: float = 0.25,  # For margin accounts
    commission_model: CommissionModel | None = None,
    slippage_model: SlippageModel | None = None,
    execution_mode: ExecutionMode = ExecutionMode.SAME_BAR,
):
    from .accounting import AccountState, CashAccountPolicy, MarginAccountPolicy, Gatekeeper

    # Create appropriate policy
    if account_type == "cash":
        policy = CashAccountPolicy()
    elif account_type == "margin":
        policy = MarginAccountPolicy(initial_margin, maintenance_margin)
    else:
        raise ValueError(f"Unknown account type: {account_type}")

    # Create account with policy
    self.account = AccountState(initial_cash, policy)
    self.commission_model = commission_model or NoCommission()
    self.gatekeeper = Gatekeeper(self.account, self.commission_model)

    # Rest of initialization...
```

**Modified _execute_fill():**
```python
def _execute_fill(self, order: Order, base_price: float):
    """Execute fill with validation"""
    # NEW: Validate order first
    is_valid, reason = self.gatekeeper.validate_order(order, base_price)
    if not is_valid:
        order.status = OrderStatus.REJECTED
        return  # Don't execute

    # Calculate slippage and commission (existing)
    volume = self._current_volumes.get(order.asset)
    slippage = self.slippage_model.calculate(...)
    fill_price = base_price + slippage if order.side == OrderSide.BUY else base_price - slippage
    commission = self.commission_model.calculate(...)

    # Record fill (existing)
    fill = Fill(...)
    self.fills.append(fill)

    # Update order status (existing)
    order.status = OrderStatus.FILLED
    order.filled_at = self._current_time
    order.filled_price = fill_price
    order.filled_quantity = order.quantity

    # NEW: Update account state
    signed_qty = order.quantity if order.side == OrderSide.BUY else -order.quantity
    self.account.apply_fill(order.asset, signed_qty, fill_price, commission)

    # NEW: Keep positions reference in sync
    self.positions = self.account.positions

    # Record completed trades (existing logic)...
```

**NEW: Exit-first sequencing in process_pending_orders():**
```python
def process_pending_orders(self):
    """Process orders with exit-first sequencing"""
    # Separate exits from entries
    exits = []
    entries = []

    for order in self.pending_orders:
        if self._is_exit_order(order):
            exits.append(order)
        else:
            entries.append(order)

    # Process exits first (releases capital)
    for order in exits:
        self._check_and_fill(order)

    # Mark-to-market after exits
    self.account.mark_to_market(self._current_prices)

    # Process entries with updated buying power
    for order in entries:
        self._check_and_fill(order)

def _is_exit_order(self, order: Order) -> bool:
    """Check if order reduces position"""
    pos = self.account.positions.get(order.asset)
    if not pos:
        return False

    signed_qty = order.quantity if order.side == OrderSide.BUY else -order.quantity
    return (pos.quantity > 0 and signed_qty < 0) or \
           (pos.quantity < 0 and signed_qty > 0)
```

## Testing Strategy

### Unit Tests (Accounting Package)

**test_position.py:**
- Cost basis updates correctly when adding to position
- Market value calculated correctly (long and short)
- Unrealized P&L calculated correctly

**test_cash_account_policy.py:**
- Buying power equals cash
- Short selling rejected
- Orders rejected when cost > cash
- Orders approved when cost <= cash

**test_margin_account_policy.py:**
- Buying power calculation (NLV, MM, BP formula)
- Short selling allowed
- Margin requirements calculated correctly
- Orders rejected when insufficient buying power

**test_gatekeeper.py:**
- Reducing trades always approved
- New positions validated via policy
- Commission included in cost calculation
- Position reversals handled correctly

### Integration Tests (Broker + Accounting)

**test_cash_account_integration.py:**
- Exit-first sequencing frees cash
- Order rejection flows correctly
- Trade recording on position close
- Multi-asset position tracking

**test_margin_account_integration.py:**
- Short positions tracked correctly
- Position reversals (long→short) work
- Margin calculations update after fills
- Bankruptcy scenario (equity floor at 0)

### Validation Tests (Framework Comparison)

**test_vectorbt_validation.py:**
- Cash account mode matches VectorBT within 0.1%
- 50-asset scenario with cash constraints
- Should fix current 99.4% difference

**test_bankruptcy.py:**
- Martingale strategy (double-down on losses)
- Engine stops trading when equity hits zero
- No negative equity allowed

**test_flipping.py:**
- Long↔Short every bar
- Cash decreases exactly by commission + spread
- Position tracking remains accurate

## Next Steps

### Immediate Actions

1. **Create Work Unit**: `.claude/work/2025-11-20_01_accounting/`
2. **Generate Implementation Plan**: Detailed task breakdown with dependencies
3. **Setup Development**: Create accounting package structure

### Recommended Workflow

```bash
# After this exploration completes
/plan  # Generate detailed task breakdown
/next  # Start implementing Phase 1 (accounting infrastructure)
```

### Implementation Phases

**Phase 1** (2-3 hours): Accounting infrastructure
- Create package, Position class, Policy interface
- Implement CashAccountPolicy
- Unit tests

**Phase 2** (2-3 hours): Cash account integration
- Update Broker initialization
- Integrate Gatekeeper validation
- Add exit-first sequencing
- Update 17 existing tests
- VectorBT validation

**Phase 3** (3-4 hours): Margin account support
- Implement MarginAccountPolicy
- Short selling support
- Position reversal handling
- Margin-specific tests

**Phase 4** (2 hours): Documentation and cleanup
- README updates with examples
- Margin calculation documentation
- Architecture decision records

## Key Decisions

### 1. Policy Pattern vs Forced Margin Model
✅ **Chosen**: Policy pattern with explicit account_type
- Simpler for cash accounts (no confusing margin=1.0 hacks)
- Clear user intent
- Extensible for future account types

### 2. Single vs Dual Position Class
✅ **Chosen**: Single unified Position class
- Avoid duplication between engine and accounting
- Cost basis tracking benefits both account types
- Cleaner architecture

### 3. Validation Timing
✅ **Chosen**: Pre-execution validation in Gatekeeper
- Prevents invalid state (cash never goes negative)
- Clear rejection reasons
- Matches real broker behavior

### 4. Exit-First Sequencing
✅ **Chosen**: Always process exits before entries
- Capital efficiency (freed capital available for re-entry)
- Matches institutional best practices
- Benefits both account types

### 5. Default Account Type
✅ **Chosen**: "cash" as default
- Safer default (simpler constraints)
- Most common use case
- Explicit opt-in for margin accounts

## Risk Mitigation

### Test Breakage
- **Risk**: 17 existing tests expect no rejections
- **Mitigation**: Update tests incrementally, default to cash accounts

### Integration Complexity
- **Risk**: Weaving gatekeeper into execution flow
- **Mitigation**: Phased approach (cash first, then margin)

### Performance Regression
- **Risk**: Margin calculations slow execution
- **Mitigation**: Benchmark before/after, profile hot paths

### Edge Cases
- **Risk**: Position reversals, partial closes, commission timing
- **Mitigation**: Comprehensive test suite, spreadsheet validation

## Success Criteria

- [x] Requirements documented
- [x] Architecture designed
- [x] Integration strategy clear
- [ ] Implementation plan generated
- [ ] Phase 1 complete (accounting package)
- [ ] Phase 2 complete (cash account integration)
- [ ] Phase 3 complete (margin account support)
- [ ] All tests passing
- [ ] VectorBT validation within 0.1%
- [ ] Documentation complete

---

**Status**: Exploration complete, ready for planning
**Next Command**: `/plan` to generate detailed task breakdown
**Estimated Effort**: 10-12 hours (2-3 days)
