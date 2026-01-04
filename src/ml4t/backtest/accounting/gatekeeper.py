"""Order validation gatekeeper.

This module provides the Gatekeeper class that validates orders before execution,
ensuring they meet account policy constraints and preventing invalid trades.
"""

from ..models import CommissionModel
from ..types import Order, OrderSide
from .account import AccountState


class Gatekeeper:
    """Pre-execution order validation to enforce account constraints.

    The Gatekeeper is the critical component that prevents invalid orders from
    executing. It checks account policy constraints (cash limits, short selling,
    margin requirements) BEFORE orders are filled.

    Key Responsibilities:
    - Validate orders against account policy constraints
    - Distinguish reducing (exit) orders from opening (entry) orders
    - Include commission in cost calculations
    - Prevent unlimited debt bug (line 587 in engine.py)

    Example:
        >>> from ml4t.backtest.accounting import AccountState, CashAccountPolicy, Gatekeeper
        >>> from ml4t.backtest.engine import Order, OrderSide, OrderType, PercentageCommission
        >>> policy = CashAccountPolicy()
        >>> account = AccountState(initial_cash=100000.0, policy=policy)
        >>> commission_model = PercentageCommission(rate=0.001)
        >>> gatekeeper = Gatekeeper(account, commission_model)
        >>>
        >>> # Validate a buy order
        >>> order = Order(asset="AAPL", side=OrderSide.BUY, quantity=100)
        >>> valid, reason = gatekeeper.validate_order(order, price=150.0)
        >>> print(valid)  # True (have enough cash)
    """

    def __init__(self, account: AccountState, commission_model: CommissionModel):
        """Initialize gatekeeper with account and commission model.

        Args:
            account: AccountState instance to validate against
            commission_model: CommissionModel for calculating transaction costs
        """
        self.account = account
        self.commission_model = commission_model

    def validate_order(self, order: Order, price: float) -> tuple[bool, str]:
        """Validate order before execution.

        This is the main validation entry point called by the Broker before
        executing any order. It performs the following checks:

        1. Detect position reversals (long→short or short→long)
        2. Check if order is reducing existing position
        3. Reducing orders always approved (closing positions frees capital)
        4. Opening orders and reversals validated via account policy
        5. Commission included in cost calculation

        Args:
            order: Order to validate
            price: Expected fill price

        Returns:
            (is_valid, rejection_reason) tuple:
                - is_valid: True if order can proceed, False if rejected
                - rejection_reason: Human-readable explanation (empty if valid)

        Examples:
            Reducing order (always approved):
                >>> # Current: Long 100 shares, Order: Sell 50
                >>> valid, reason = gatekeeper.validate_order(sell_order, 150.0)
                >>> assert valid == True
                >>> assert reason == ""

            Position reversal (cash account):
                >>> # Current: Long 100, Order: Sell 150 (reverse to short 50)
                >>> valid, reason = gatekeeper.validate_order(sell_order, 150.0)
                >>> assert valid == False
                >>> assert "Position reversal not allowed" in reason

            Position reversal (margin account):
                >>> # Current: Long 100, Order: Sell 150 (reverse to short 50)
                >>> valid, reason = gatekeeper.validate_order(sell_order, 150.0)
                >>> # valid depends on buying power for new short 50 position

            Opening order (validated):
                >>> # No position, Order: Buy 100 shares @ $150
                >>> valid, reason = gatekeeper.validate_order(buy_order, 150.0)
                >>> # valid depends on cash: need $15,000 + commission

            Rejected order:
                >>> # Cash account trying to short
                >>> short_order = Order(asset="AAPL", side=OrderSide.SELL, quantity=100)
                >>> valid, reason = gatekeeper.validate_order(short_order, 150.0)
                >>> assert valid == False
                >>> assert "Short selling not allowed" in reason
        """
        # Get current position quantity (0 if no position)
        current_qty = self.account.get_position_quantity(order.asset)

        # Determine order direction (positive=buy, negative=sell)
        order_qty_delta = self._calculate_quantity_delta(order.side, order.quantity)

        # Check for position reversal (long→short or short→long)
        # Delegate to policy's handle_reversal() method
        if self._is_reversal(current_qty, order_qty_delta):
            commission = self.commission_model.calculate(order.asset, order.quantity, price)
            return self.account.policy.handle_reversal(
                asset=order.asset,
                current_quantity=current_qty,
                order_quantity_delta=order_qty_delta,
                price=price,
                current_positions=self.account.positions,
                cash=self.account.cash,
                commission=commission,
            )

        # Check if this is a reducing order (closing/reducing existing position)
        if self._is_reducing_order(current_qty, order_qty_delta):
            # Reducing orders always allowed (frees up capital)
            return True, ""

        # This is an opening order (new position or adding to existing)
        # Calculate commission to include in cost
        commission = self.commission_model.calculate(order.asset, order.quantity, price)

        # Validate based on whether we have an existing position
        if current_qty == 0.0:
            # New position - use validate_new_position
            new_qty = order_qty_delta  # Full order quantity
            return self.account.policy.validate_new_position(
                asset=order.asset,
                quantity=new_qty,
                price=price,
                current_positions=self.account.positions,
                cash=self.account.cash - commission,  # Account for commission cost
            )
        else:
            # Adding to existing position - use validate_position_change
            return self.account.policy.validate_position_change(
                asset=order.asset,
                current_quantity=current_qty,
                quantity_delta=order_qty_delta,
                price=price,
                current_positions=self.account.positions,
                cash=self.account.cash - commission,  # Account for commission cost
            )

    def _is_reversal(self, current_qty: float, order_qty_delta: float) -> bool:
        """Check if order reverses position (long → short or short → long).

        A reversal occurs when an order causes the position to change sign,
        creating a new opposite position. This is distinct from simply closing
        a position (where the result would be zero or same sign).

        Args:
            current_qty: Current position quantity (positive=long, negative=short)
            order_qty_delta: Order quantity delta (positive=buy, negative=sell)

        Returns:
            True if order reverses position, False otherwise

        Examples:
            Reversals (returns True):
                >>> gatekeeper._is_reversal(100, -150)  # Long 100, sell 150 → short 50
                True
                >>> gatekeeper._is_reversal(-100, 150)  # Short 100, buy 150 → long 50
                True

            Non-reversals (returns False):
                >>> gatekeeper._is_reversal(0, 100)     # No position, buy 100
                False
                >>> gatekeeper._is_reversal(100, -100)  # Long 100, sell 100 → flat
                False
                >>> gatekeeper._is_reversal(100, -50)   # Long 100, sell 50 → long 50
                False
                >>> gatekeeper._is_reversal(100, 50)    # Long 100, buy 50 → long 150
                False

        Note:
            Position reversals are only allowed in margin accounts. Cash accounts
            do not support short selling and therefore cannot have reversals.
        """
        if current_qty == 0.0:
            # No existing position - cannot reverse
            return False

        new_qty = current_qty + order_qty_delta

        # Check if signs differ between current and new position
        # (both must be non-zero for a true reversal)
        return (current_qty > 0 and new_qty < 0) or (current_qty < 0 and new_qty > 0)

    def _calculate_quantity_delta(self, side: OrderSide, quantity: float) -> float:
        """Convert order side and quantity to signed delta.

        Args:
            side: BUY or SELL
            quantity: Order quantity (always positive)

        Returns:
            Signed quantity delta (positive=buy, negative=sell)

        Examples:
            >>> gatekeeper._calculate_quantity_delta(OrderSide.BUY, 100)
            100.0
            >>> gatekeeper._calculate_quantity_delta(OrderSide.SELL, 100)
            -100.0
        """
        return quantity if side == OrderSide.BUY else -quantity

    def _is_reducing_order(self, current_qty: float, order_qty_delta: float) -> bool:
        """Check if order reduces existing position.

        A reducing order is one that moves the position closer to flat (zero).

        Args:
            current_qty: Current position quantity (positive=long, negative=short)
            order_qty_delta: Order quantity delta (positive=buy, negative=sell)

        Returns:
            True if order reduces position, False if opens/adds

        Examples:
            Reducing orders:
                >>> gatekeeper._is_reducing_order(100, -50)  # Long 100, sell 50
                True
                >>> gatekeeper._is_reducing_order(-100, 50)  # Short 100, buy 50
                True
                >>> gatekeeper._is_reducing_order(100, -100)  # Closing
                True

            Opening/Adding orders:
                >>> gatekeeper._is_reducing_order(0, 100)    # No position, buy
                False
                >>> gatekeeper._is_reducing_order(100, 50)   # Long 100, buy more
                False
                >>> gatekeeper._is_reducing_order(100, -150) # Reversal (long->short)
                False  # Not just reducing, this reverses position!

        Note:
            Position reversals (e.g., long 100 -> short 50) are NOT reducing orders
            because they require opening a new short position. These must be validated.
        """
        if current_qty == 0.0:
            # No position - this is opening, not reducing
            return False

        # Check if order and position have opposite signs
        if current_qty > 0 and order_qty_delta < 0:
            # Long position, sell order - reducing if not reversing
            new_qty = current_qty + order_qty_delta
            return new_qty >= 0  # True if still long or flat, False if reverses to short
        elif current_qty < 0 and order_qty_delta > 0:
            # Short position, buy order - reducing if not reversing
            new_qty = current_qty + order_qty_delta
            return new_qty <= 0  # True if still short or flat, False if reverses to long
        else:
            # Same sign - this is adding to position, not reducing
            return False
