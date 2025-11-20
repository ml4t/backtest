"""Portfolio-level constraint rules for risk management.

This module implements portfolio-level risk constraints that limit trading activity
based on portfolio-wide metrics:

- MaxDailyLossRule: Halt trading when daily loss exceeds threshold
- MaxDrawdownRule: Halt trading when drawdown from high-water mark exceeds limit
- MaxLeverageRule: Reject/reduce orders to maintain leverage within bounds

These rules implement validate_order() (not evaluate()) since they prevent new trades
rather than managing existing positions.

Examples:
    >>> # Daily loss limit - stop trading after losing 2% of equity in one day
    >>> daily_loss = MaxDailyLossRule(max_loss_pct=0.02)
    >>> manager.add_rule(daily_loss)
    >>>
    >>> # Drawdown limit - stop trading when down 10% from high-water mark
    >>> drawdown = MaxDrawdownRule(max_dd_pct=0.10)
    >>> manager.add_rule(drawdown)
    >>>
    >>> # Leverage limit - prevent leverage from exceeding 2.0x
    >>> leverage = MaxLeverageRule(max_leverage=2.0)
    >>> manager.add_rule(leverage)
    >>>
    >>> # Combine all portfolio constraints
    >>> manager.add_rule(MaxDailyLossRule(0.02))
    >>> manager.add_rule(MaxDrawdownRule(0.10))
    >>> manager.add_rule(MaxLeverageRule(2.0))
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional

from ml4t.backtest.core.types import AssetId, Price
from ml4t.backtest.execution.order import Order
from ml4t.backtest.risk.context import RiskContext
from ml4t.backtest.risk.decision import RiskDecision
from ml4t.backtest.risk.rule import RiskRule


@dataclass
class MaxDailyLossRule(RiskRule):
    """Portfolio constraint that halts trading when daily loss exceeds threshold.

    This rule tracks daily profit/loss and prevents new orders from being submitted
    when the loss for the current trading day exceeds a specified percentage of
    starting equity.

    The rule:
    1. Tracks session start equity (reset at start of each trading day)
    2. Calculates current daily P&L = current_equity - session_start_equity
    3. Rejects all new orders when daily_pnl < -max_loss_pct * session_start_equity

    Trading resumes automatically on the next trading day (session reset).

    Attributes:
        max_loss_pct: Maximum allowed daily loss as fraction (e.g., 0.02 = 2%)
        session_start_equity: Equity at session start (reset daily)
        current_date: Current trading date for session detection
        priority: Rule priority (default: 15 - highest, prevents catastrophic losses)

    Examples:
        >>> # 2% daily loss limit
        >>> rule = MaxDailyLossRule(max_loss_pct=0.02)
        >>> manager.add_rule(rule)
        >>>
        >>> # Portfolio starts at $100k
        >>> # After losing $2k (-2%), rule rejects all new orders
        >>> # Next day, session resets and trading resumes
        >>>
        >>> # Check if rule would reject an order
        >>> order = MarketOrder.buy("AAPL", 100)
        >>> validated = rule.validate_order(order, context)
        >>> if validated is None:
        ...     print("Order rejected due to daily loss limit")

    Notes:
        - Session reset occurs when context.timestamp.date() changes
        - Existing positions are not closed (use evaluate() for position exits)
        - Only prevents NEW orders, doesn't modify existing stop-losses
        - High priority (15) ensures it runs before other validation rules
    """

    max_loss_pct: float
    _session_start_equity: Optional[float] = field(default=None, init=False, repr=False)
    _current_date: Optional[date] = field(default=None, init=False, repr=False)
    _priority: int = field(default=15, init=False, repr=False)

    def __post_init__(self):
        """Validate parameters."""
        if self.max_loss_pct <= 0 or self.max_loss_pct >= 1.0:
            raise ValueError(
                f"max_loss_pct must be between 0 and 1, got {self.max_loss_pct}"
            )

    def _reset_session_if_needed(self, context: RiskContext) -> None:
        """Reset session tracking at start of new trading day.

        Args:
            context: Current risk context
        """
        current_date = context.timestamp.date()

        # First call or new trading day
        if self._current_date is None or current_date != self._current_date:
            self._current_date = current_date
            self._session_start_equity = context.equity

    def evaluate(self, context: RiskContext) -> RiskDecision:
        """Daily loss rule doesn't manage existing positions.

        Use validate_order() to prevent new trades.

        Args:
            context: Current risk context

        Returns:
            RiskDecision.no_action() - never exits positions
        """
        # Update session tracking
        self._reset_session_if_needed(context)
        return RiskDecision.no_action()

    def validate_order(
        self, order: Order, context: RiskContext
    ) -> Optional[Order]:
        """Reject orders when daily loss exceeds threshold.

        Args:
            order: Order to validate
            context: Current risk context for portfolio state

        Returns:
            Original order if within daily loss limit, None if rejected
        """
        # Reset session if new trading day
        self._reset_session_if_needed(context)

        # If session start equity not set (shouldn't happen), allow order
        if self._session_start_equity is None or self._session_start_equity == 0:
            return order

        # Calculate daily P&L
        daily_pnl = context.equity - self._session_start_equity
        daily_pnl_pct = daily_pnl / self._session_start_equity

        # Reject if loss meets or exceeds threshold
        if daily_pnl_pct <= -self.max_loss_pct:
            return None  # Reject order

        return order  # Allow order

    @property
    def priority(self) -> int:
        """High priority (15) to prevent catastrophic losses."""
        return self._priority

    def __repr__(self) -> str:
        """String representation."""
        return f"MaxDailyLossRule(max_loss_pct={self.max_loss_pct:.1%}, priority={self.priority})"


@dataclass
class MaxDrawdownRule(RiskRule):
    """Portfolio constraint that halts trading when drawdown exceeds limit.

    This rule tracks the portfolio high-water mark (peak equity) and prevents new
    orders when the current drawdown from that peak exceeds a specified percentage.

    The rule:
    1. Tracks high-water mark (highest equity achieved)
    2. Calculates drawdown = (current_equity - high_water_mark) / high_water_mark
    3. Rejects all new orders when drawdown < -max_dd_pct

    Trading resumes when equity recovers above the drawdown threshold.

    Attributes:
        max_dd_pct: Maximum allowed drawdown as fraction (e.g., 0.10 = 10%)
        high_water_mark: Highest portfolio equity achieved (tracks peak)
        priority: Rule priority (default: 15 - highest, prevents catastrophic losses)

    Examples:
        >>> # 10% maximum drawdown limit
        >>> rule = MaxDrawdownRule(max_dd_pct=0.10)
        >>> manager.add_rule(rule)
        >>>
        >>> # Portfolio peaks at $120k (high-water mark)
        >>> # Current equity: $108k → drawdown = -10% → rule rejects new orders
        >>> # Equity recovers to $110k → drawdown = -8.3% → trading resumes
        >>>
        >>> # Check drawdown status
        >>> order = MarketOrder.buy("AAPL", 100)
        >>> validated = rule.validate_order(order, context)
        >>> if validated is None:
        ...     current_dd = (context.equity - rule.high_water_mark) / rule.high_water_mark
        ...     print(f"Order rejected - drawdown: {current_dd:.1%}")

    Notes:
        - High-water mark only increases, never decreases
        - Unlike daily loss (which resets), drawdown is cumulative
        - Trading resumes automatically when drawdown improves
        - High priority (15) ensures it runs before other validation rules
    """

    max_dd_pct: float
    _high_water_mark: float = field(default=0.0, init=False, repr=False)
    _priority: int = field(default=15, init=False, repr=False)

    def __post_init__(self):
        """Validate parameters."""
        if self.max_dd_pct <= 0 or self.max_dd_pct >= 1.0:
            raise ValueError(
                f"max_dd_pct must be between 0 and 1, got {self.max_dd_pct}"
            )

    def _update_high_water_mark(self, equity: float) -> None:
        """Update high-water mark if current equity is higher.

        Args:
            equity: Current portfolio equity
        """
        if equity > self._high_water_mark:
            self._high_water_mark = equity

    @property
    def high_water_mark(self) -> float:
        """Current high-water mark (peak equity)."""
        return self._high_water_mark

    def evaluate(self, context: RiskContext) -> RiskDecision:
        """Drawdown rule doesn't manage existing positions.

        Use validate_order() to prevent new trades.

        Args:
            context: Current risk context

        Returns:
            RiskDecision.no_action() - never exits positions
        """
        # Update high-water mark
        self._update_high_water_mark(context.equity)
        return RiskDecision.no_action()

    def validate_order(
        self, order: Order, context: RiskContext
    ) -> Optional[Order]:
        """Reject orders when drawdown exceeds threshold.

        Args:
            order: Order to validate
            context: Current risk context for portfolio state

        Returns:
            Original order if within drawdown limit, None if rejected
        """
        # Update high-water mark
        self._update_high_water_mark(context.equity)

        # If high-water mark not set (first trade), allow order
        if self._high_water_mark == 0:
            return order

        # Calculate current drawdown
        drawdown = (context.equity - self._high_water_mark) / self._high_water_mark

        # Reject if drawdown meets or exceeds threshold
        if drawdown <= -self.max_dd_pct:
            return None  # Reject order

        return order  # Allow order

    @property
    def priority(self) -> int:
        """High priority (15) to prevent catastrophic losses."""
        return self._priority

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MaxDrawdownRule(max_dd_pct={self.max_dd_pct:.1%}, "
            f"hwm=${self._high_water_mark:.2f}, priority={self.priority})"
        )


@dataclass
class MaxLeverageRule(RiskRule):
    """Portfolio constraint that maintains leverage within specified bounds.

    This rule prevents leverage from exceeding a maximum threshold by rejecting
    or reducing order sizes. Leverage is calculated as:

        leverage = total_position_value / equity

    The rule:
    1. Calculates current leverage from open positions
    2. Simulates proposed order to calculate new leverage
    3. Rejects order if new_leverage > max_leverage
    4. Optionally reduces order size to maintain max_leverage (if allow_partial=True)

    Attributes:
        max_leverage: Maximum allowed leverage ratio (e.g., 2.0 = 200%)
        allow_partial: If True, reduce order size to fit within leverage limit
                       If False, reject entire order when limit exceeded
        priority: Rule priority (default: 10 - high, prevents excessive risk)

    Examples:
        >>> # Strict leverage limit - reject orders that exceed 2x
        >>> rule = MaxLeverageRule(max_leverage=2.0, allow_partial=False)
        >>> manager.add_rule(rule)
        >>>
        >>> # Flexible leverage limit - reduce order size to fit
        >>> rule = MaxLeverageRule(max_leverage=2.0, allow_partial=True)
        >>> order = MarketOrder.buy("AAPL", 1000)  # Too large
        >>> validated = rule.validate_order(order, context)  # Returns reduced order
        >>>
        >>> # Portfolio: $100k equity, $150k position value → 1.5x leverage
        >>> # Order: Buy $80k → would be 2.3x leverage → rejected
        >>>
        >>> # With allow_partial=True:
        >>> # Order reduced to $50k → new leverage = 2.0x → allowed

    Notes:
        - Leverage calculated from absolute position values (long + short)
        - allow_partial=True enables position sizing within constraints
        - allow_partial=False is safer (prevents unexpected small fills)
        - Priority 10 (high but below 15) runs after catastrophic loss rules
    """

    max_leverage: float
    allow_partial: bool = False
    _priority: int = field(default=10, init=False, repr=False)

    def __post_init__(self):
        """Validate parameters."""
        if self.max_leverage <= 0:
            raise ValueError(
                f"max_leverage must be positive, got {self.max_leverage}"
            )

    def _calculate_position_value(self, context: RiskContext) -> float:
        """Calculate total position value (sum of absolute values).

        Args:
            context: Current risk context

        Returns:
            Total position value (always positive)
        """
        # Get portfolio positions from context
        # Context only has current asset's position, need to get total from portfolio
        # This is approximated using leverage from context
        # leverage = total_position_value / equity
        # So: total_position_value = leverage * equity
        return abs(context.leverage * context.equity)

    def _calculate_order_value(self, order: Order, context: RiskContext) -> float:
        """Calculate value of proposed order.

        Args:
            order: Order to value
            context: Current risk context for pricing

        Returns:
            Order value (quantity * price, always positive)
        """
        # Use current market price as estimate for order fill price
        price = context.close if context.close is not None else 0.0
        return abs(float(order.quantity) * price)

    def evaluate(self, context: RiskContext) -> RiskDecision:
        """Leverage rule doesn't manage existing positions.

        Use validate_order() to prevent excessive leverage.

        Args:
            context: Current risk context

        Returns:
            RiskDecision.no_action() - never exits positions
        """
        return RiskDecision.no_action()

    def validate_order(
        self, order: Order, context: RiskContext
    ) -> Optional[Order]:
        """Reject or reduce orders that would exceed leverage limit.

        Args:
            order: Order to validate
            context: Current risk context for portfolio state

        Returns:
            Original order, reduced order (if allow_partial), or None (rejected)
        """
        # Avoid division by zero
        if context.equity <= 0:
            return None  # Reject if equity is zero or negative

        # Calculate current leverage
        current_position_value = self._calculate_position_value(context)
        current_leverage = current_position_value / context.equity

        # Calculate order value
        order_value = self._calculate_order_value(order, context)

        # Calculate proposed leverage after order fills
        # Note: This is approximate since we don't know exact fill price
        proposed_position_value = current_position_value + order_value
        proposed_leverage = proposed_position_value / context.equity

        # If within leverage limit, allow order
        if proposed_leverage <= self.max_leverage:
            return order

        # Leverage limit exceeded
        if not self.allow_partial:
            return None  # Reject entire order

        # Calculate maximum allowed order value
        max_allowed_position_value = self.max_leverage * context.equity
        max_order_value = max(0.0, max_allowed_position_value - current_position_value)

        if max_order_value <= 0:
            return None  # Already at/above leverage limit

        # Calculate reduced order quantity
        price = context.close if context.close is not None else 1.0
        reduced_quantity = int(max_order_value / price)

        if reduced_quantity <= 0:
            return None  # Can't even fill 1 share

        # Create reduced order (preserve order type and other attributes)
        # This is a simplified version - would need to properly clone order
        order.quantity = reduced_quantity
        return order

    @property
    def priority(self) -> int:
        """High priority (10) to prevent excessive leverage."""
        return self._priority

    def __repr__(self) -> str:
        """String representation."""
        partial_str = ", allow_partial=True" if self.allow_partial else ""
        return f"MaxLeverageRule(max_leverage={self.max_leverage:.1f}x{partial_str}, priority={self.priority})"
