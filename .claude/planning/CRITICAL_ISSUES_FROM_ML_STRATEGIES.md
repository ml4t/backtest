# Critical Issues Discovered from ml-strategies Integration

**Date**: 2025-10-03
**Source**: ml-strategies backtest validation (qengine vs vectorbt comparison)
**Status**: 🔴 CRITICAL - Production-blocking issues

---

## Overview

During validation of qengine against vectorbt on a real ML trading strategy, we discovered critical issues that make qengine unsafe for production use. While **signal generation is perfectly aligned**, the **position sizing and risk management** allows unlimited leverage, causing 57x worse returns than vectorbt on identical signals.

**Test Summary**:
- Both engines: Same features, same signals (20% LONG, 20% SHORT, 60% FLAT)
- qengine: -1,207% return (-$1.2M on $100K capital)
- vectorbt: -21% return (-$21K on $100K capital)
- Difference: **57x worse** due to unlimited leverage

---

## Issue #1: Position Sizing Without Capital Constraints

**Priority**: 🔴 CRITICAL
**Component**: `execution/broker.py` or `portfolio/`
**Impact**: Allows unlimited leverage, catastrophic losses

### Problem

The execution layer accepts orders without checking if there's sufficient capital. Strategies can submit orders for any quantity regardless of available capital, leading to extreme leverage as capital depletes.

### Evidence

Test case: ML strategy trading BTC futures with $100K initial capital.

**qengine behavior**:
```python
# Strategy submits: OrderEvent(side=BUY, quantity=1.0)
# Broker executes: 1.0 BTC @ $115,000 = $115,000 notional
# Capital: $100,000
# Effective leverage: 1.15x ✓ OK

# After losses, capital drops to $50,000
# Strategy submits: OrderEvent(side=BUY, quantity=1.0)
# Broker executes: 1.0 BTC @ $115,000 = $115,000 notional
# Capital: $50,000
# Effective leverage: 2.3x ⚠️ RISK

# After more losses, capital drops to $25,000
# Strategy submits: OrderEvent(side=BUY, quantity=1.0)
# Broker executes: 1.0 BTC @ $115,000 = $115,000 notional
# Capital: $25,000
# Effective leverage: 4.6x 🔴 CRITICAL
```

**vectorbt behavior**:
```python
# With size=1.0, size_type='amount' and insufficient capital
# Position size automatically reduces to fractional contracts
# Observed range: 0.69 - 1.0 contracts
# Prevents leverage from exceeding configured limits
```

### Current Code Gap

Looking at `execution/simulation_broker.py` (assumed location), orders are likely processed as:

```python
def process_order(self, order: OrderEvent):
    # Current: No capital check
    fill_price = self._get_fill_price(order)
    quantity = order.quantity  # Used as-is

    # Execute without checking if capital is sufficient
    self._execute_fill(order, fill_price, quantity)
```

### Expected Behavior

```python
def process_order(self, order: OrderEvent):
    fill_price = self._get_fill_price(order)
    notional = order.quantity * fill_price
    available_capital = self.portfolio.get_available_capital()

    # Check capital constraints
    max_leverage = self.config.max_leverage  # e.g., 2.0
    max_notional = available_capital * max_leverage

    if notional > max_notional:
        # Option 1: Reject order
        self._reject_order(order, reason="Insufficient capital")
        return

        # Option 2: Reduce quantity (like vectorbt)
        adjusted_quantity = max_notional / fill_price
        self.logger.warning(
            f"Order quantity reduced from {order.quantity:.2f} to "
            f"{adjusted_quantity:.2f} due to capital constraints"
        )
        quantity = adjusted_quantity
    else:
        quantity = order.quantity

    self._execute_fill(order, fill_price, quantity)
```

### Proposed Fix

**Location**: `src/qengine/execution/simulation_broker.py`

**Changes needed**:
1. Add `max_leverage` parameter to broker config (default: 1.0, no leverage)
2. Add `capital_check_mode` parameter: `"reject"` or `"reduce"` or `"none"`
3. Implement `_check_capital_constraints()` method
4. Call capital check before order execution
5. Add tests for capital constraint enforcement

**Configuration**:
```python
# In BacktestEngine config
broker_config = SimulationBrokerConfig(
    max_leverage=2.0,  # Maximum 2x leverage
    capital_check_mode="reduce",  # Reduce quantity if insufficient capital
    warn_on_reduction=True,  # Log warnings when reducing
)
```

**Test cases needed**:
1. Order with sufficient capital → executes fully
2. Order exceeding max leverage → reduced quantity
3. Order with zero capital → rejected
4. Multiple orders depleting capital → progressive reduction

### Related Files

- `src/qengine/execution/simulation_broker.py` - Order execution
- `src/qengine/portfolio/portfolio.py` - Capital tracking
- `src/qengine/core/config.py` - Configuration

---

## Issue #2: Portfolio.get_position() Returns None

**Priority**: 🟡 HIGH
**Component**: `portfolio/portfolio.py`
**Impact**: Forces strategies to implement internal position tracking

### Problem

When strategies call `portfolio.get_position(asset_id)`, it returns `None` even when positions exist. This forces every strategy to maintain internal position state, duplicating logic and creating inconsistency.

### Evidence

From `ml_signal_strategy_simple.py`:

```python
def _generate_signal(self, timestamp: datetime, price: float, signal: int):
    # Attempted to use portfolio position
    if hasattr(self, "portfolio") and self.portfolio is not None:
        try:
            pos = self.portfolio.get_position(self.asset_id)
            # ❌ pos is None even when position exists
            if pos is None:
                self._current_position = 0.0
```

**Workaround required**:
```python
# Strategy must track position internally
self._current_position = 0.0

def _submit_order(self, side: OrderSide, quantity: float, ...):
    # ... submit order ...

    # Manual position update
    if side == OrderSide.BUY:
        self._current_position += quantity
    else:
        self._current_position -= quantity
```

This workaround:
- Duplicates portfolio logic in every strategy
- Can drift out of sync with actual portfolio
- Doesn't account for partial fills
- Doesn't account for rejected orders

### Expected Behavior

```python
def _generate_signal(self, timestamp: datetime, price: float, signal: int):
    # Should work
    position = self.portfolio.get_position(self.asset_id)
    current_qty = position.quantity if position else 0.0

    # Use actual portfolio position for trading logic
    if signal == 1 and current_qty <= 0:
        self._submit_order(OrderSide.BUY, 1.0, price)
```

### Root Cause Analysis Needed

Possible causes:
1. **Asset ID mismatch**: Portfolio uses different ID format than strategy
2. **Event timing**: Position updates happen after strategy queries
3. **Portfolio not connected**: Strategy doesn't have portfolio reference
4. **Bug in get_position()**: Logic error in position lookup

### Proposed Investigation

**Step 1**: Add debug logging to portfolio
```python
# In portfolio.py
def get_position(self, asset_id: AssetId) -> Position | None:
    self.logger.debug(f"get_position called for {asset_id}")
    self.logger.debug(f"Current positions: {list(self._positions.keys())}")
    pos = self._positions.get(asset_id)
    self.logger.debug(f"Returning position: {pos}")
    return pos
```

**Step 2**: Test in isolation
```python
# Unit test
def test_position_tracking():
    portfolio = Portfolio(initial_capital=100000)

    # Execute buy
    portfolio.process_fill(Fill(
        asset_id="BTC",
        side=OrderSide.BUY,
        quantity=1.0,
        price=50000.0,
    ))

    # Should return position
    pos = portfolio.get_position("BTC")
    assert pos is not None, "Position should exist after fill"
    assert pos.quantity == 1.0
```

**Step 3**: Verify in BacktestEngine
```python
# In BacktestEngine, verify portfolio is passed to strategy
strategy.on_start(portfolio=self.portfolio, event_bus=self.event_bus)
```

### Proposed Fix

Once root cause is identified, fix should ensure:
1. Strategies can reliably query current positions
2. Position updates happen before strategy's next event
3. Asset ID format is consistent across components
4. Unit tests validate position tracking

---

## Issue #3: No Portfolio-Level Risk Management

**Priority**: 🟡 HIGH
**Component**: `portfolio/portfolio.py` or new `portfolio/risk.py`
**Impact**: No protection against excessive risk-taking

### Problem

Even with capital-aware position sizing (Issue #1), there's no portfolio-level risk management:
- No maximum position size limits (e.g., max 20% per position)
- No volatility-based position sizing
- No portfolio stop-loss
- No exposure limits by asset class

### Example Scenario

```python
# Strategy can build up enormous single position
strategy.submit_order(BUY, quantity=1.0)  # $115K notional
strategy.submit_order(BUY, quantity=1.0)  # $115K notional
strategy.submit_order(BUY, quantity=1.0)  # $115K notional
# Total: $345K notional on $100K capital = 3.45x leverage
# All in one asset!
```

### Proposed Enhancement

Add `RiskManager` component:

```python
class RiskManager:
    """Portfolio-level risk management."""

    def __init__(
        self,
        max_position_pct: float = 0.2,  # Max 20% per position
        max_total_leverage: float = 2.0,  # Max 2x total leverage
        portfolio_stop_loss_pct: float = 0.15,  # Stop trading at -15%
        max_correlated_exposure: float = 0.5,  # Max 50% in correlated assets
    ):
        self.max_position_pct = max_position_pct
        self.max_total_leverage = max_total_leverage
        self.portfolio_stop_loss_pct = portfolio_stop_loss_pct
        self.max_correlated_exposure = max_correlated_exposure

    def check_order(
        self,
        order: OrderEvent,
        portfolio: Portfolio
    ) -> tuple[bool, str | None]:
        """Check if order violates risk limits.

        Returns:
            (allowed, reason) - True if order is allowed
        """
        # Check portfolio stop-loss
        if self._portfolio_stop_loss_hit(portfolio):
            return False, "Portfolio stop-loss hit"

        # Check position size limit
        if not self._check_position_size(order, portfolio):
            return False, "Position size limit exceeded"

        # Check total leverage
        if not self._check_total_leverage(order, portfolio):
            return False, "Total leverage limit exceeded"

        return True, None

    def _portfolio_stop_loss_hit(self, portfolio: Portfolio) -> bool:
        total_return = (portfolio.value - portfolio.initial_capital) / portfolio.initial_capital
        return total_return < -self.portfolio_stop_loss_pct

    def _check_position_size(self, order: OrderEvent, portfolio: Portfolio) -> bool:
        # Calculate resulting position size as % of portfolio
        current_pos = portfolio.get_position(order.asset_id)
        current_qty = current_pos.quantity if current_pos else 0.0

        new_qty = current_qty + order.quantity if order.side == OrderSide.BUY else current_qty - order.quantity

        # Estimate notional value
        price = self._estimate_price(order.asset_id)
        position_value = abs(new_qty * price)
        position_pct = position_value / portfolio.value

        return position_pct <= self.max_position_pct
```

**Integration**:
```python
# In BacktestEngine
engine = BacktestEngine(
    data_feed=feed,
    strategy=strategy,
    initial_capital=100000,
    risk_manager=RiskManager(
        max_position_pct=0.2,
        max_total_leverage=2.0,
        portfolio_stop_loss_pct=0.15,
    )
)
```

### Implementation Priority

This is lower priority than Issues #1 and #2, but important for production use:
1. First: Fix capital constraints (Issue #1)
2. Second: Fix position querying (Issue #2)
3. Third: Add portfolio risk management (Issue #3)

---

## Issue #4: Position Size Not Returned in Fill Events

**Priority**: 🟢 MEDIUM
**Component**: `execution/simulation_broker.py`
**Impact**: Strategies don't know actual executed quantity

### Problem

When capital constraints reduce order quantity (Issue #1 fix), strategies don't receive feedback about the actual executed quantity. They only know what they requested, not what was filled.

### Expected Behavior

```python
# Strategy submits order
order = OrderEvent(side=BUY, quantity=1.0, ...)
self.event_bus.publish(order)

# Broker reduces quantity due to capital
# actual_quantity = 0.7

# Strategy receives FillEvent
def on_fill(self, fill: FillEvent):
    # fill.quantity should be 0.7 (actual), not 1.0 (requested)
    self._current_position += fill.quantity  # Update with actual
```

### Proposed Fix

Ensure `FillEvent` contains actual executed quantity:
```python
@dataclass
class FillEvent:
    order_id: str
    asset_id: AssetId
    side: OrderSide
    quantity: float  # ACTUAL quantity executed
    price: float
    commission: float
    timestamp: datetime

    # Optional: for debugging
    requested_quantity: float | None = None
    reduction_reason: str | None = None
```

Strategies should listen for FillEvents to track actual positions.

---

## Issue #5: No Validation Warning When Leverage Exceeds Limits

**Priority**: 🟢 LOW
**Component**: `engine.py` or `reporting/`
**Impact**: Users don't know when backtest is unrealistic

### Problem

Even after fixing Issues #1-3, users might configure unrealistic parameters:
- `max_leverage=10.0` (unrealistic for most brokers)
- No warnings about excessive trading
- No summary of leverage usage

### Proposed Enhancement

Add validation warnings at end of backtest:

```python
# In BacktestEngine.run()
results = self._generate_results()

# Add leverage analysis
max_leverage_used = max(results["leverage_history"])
avg_leverage = np.mean(results["leverage_history"])

if max_leverage_used > 3.0:
    warnings.warn(
        f"Maximum leverage used: {max_leverage_used:.1f}x. "
        f"This may be unrealistic for most brokers."
    )

if avg_leverage > 2.0:
    warnings.warn(
        f"Average leverage: {avg_leverage:.1f}x. "
        f"High leverage increases risk of margin calls."
    )
```

---

## Testing Requirements

All fixes must include:

### Unit Tests
```python
# test_capital_constraints.py
def test_order_reduced_when_insufficient_capital():
    """Order quantity reduced when capital is insufficient."""

def test_order_rejected_when_zero_capital():
    """Order rejected when portfolio has zero capital."""

def test_leverage_limit_enforced():
    """Orders respect max_leverage configuration."""
```

### Integration Tests
```python
# test_backtest_integration.py
def test_strategy_with_capital_constraints():
    """Full backtest with capital constraints enabled."""

def test_vectorbt_alignment():
    """Results align with vectorbt reference implementation."""
```

### Validation Tests
```python
# test_validation.py
def test_qengine_vs_vectorbt_identical_signals():
    """Given same signals, qengine and vectorbt produce similar results."""
    # Tolerance: Within 10% PnL difference
```

---

## Implementation Priority

1. 🔴 **CRITICAL**: Issue #1 - Capital constraints (production blocker)
2. 🟡 **HIGH**: Issue #2 - Position querying (affects strategy dev)
3. 🟡 **HIGH**: Issue #3 - Risk management (production feature)
4. 🟢 **MEDIUM**: Issue #4 - Fill feedback (nice to have)
5. 🟢 **LOW**: Issue #5 - Validation warnings (polish)

---

## Validation Criteria

Before marking these issues as resolved:

### Functional Tests
- ✅ Orders respect capital constraints
- ✅ Position queries return correct values
- ✅ Risk limits are enforced
- ✅ Fill events contain actual quantities

### Validation Tests
- ✅ qengine vs vectorbt alignment test passes
- ✅ PnL difference < 10% on same signals
- ✅ No unlimited leverage scenarios

### Documentation
- ✅ Configuration guide for risk parameters
- ✅ Examples showing capital-aware strategies
- ✅ Migration guide from unconstrained to constrained mode

---

## References

- **Source repository**: `/home/stefan/clients/wyden/long-short/development/ml-strategies`
- **Validation script**: `scripts/validate_qengine_final.py`
- **Alignment docs**: `.claude/memory/qengine_vectorbt_alignment.md`
- **Summary**: `.claude/memory/qengine_alignment_summary.md`

---

## Contact

For questions about these issues, refer to the ml-strategies validation session logs (2025-10-03).

---

*Created: 2025-10-03*
*Priority: CRITICAL - Production blocking*
*Estimated effort: 2-3 days for Issues #1-3*
