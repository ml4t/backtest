"""Order management for QEngine."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from qengine.core.types import (
    AssetId,
    OrderId,
    OrderSide,
    OrderStatus,
    OrderType,
    Price,
    Quantity,
    TimeInForce,
)
from qengine.core.precision import PrecisionManager


class OrderState(Enum):
    """Order lifecycle states."""

    PENDING = "pending"  # Created but not yet submitted
    SUBMITTED = "submitted"  # Sent to broker
    ACKNOWLEDGED = "acknowledged"  # Broker confirmed receipt
    PARTIALLY_FILLED = "partially_filled"  # Some quantity filled
    FILLED = "filled"  # Completely filled
    CANCELLED = "cancelled"  # Cancelled by user
    REJECTED = "rejected"  # Rejected by broker
    EXPIRED = "expired"  # Expired due to time constraint


@dataclass
class Order:
    """Represents a trading order."""

    # Core identifiers
    order_id: OrderId = field(default_factory=lambda: str(uuid.uuid4()))
    asset_id: AssetId = ""

    # Order specifications
    order_type: OrderType = OrderType.MARKET
    side: OrderSide = OrderSide.BUY
    quantity: Quantity = 0.0

    # Price constraints
    limit_price: Price | None = None
    stop_price: Price | None = None

    # Advanced order type parameters
    trail_amount: Price | None = None  # For trailing stops (absolute)
    trail_percent: float | None = None  # For trailing stops (percentage)
    trailing_stop_price: Price | None = None  # Current trailing stop level

    # Bracket order parameters (absolute prices)
    profit_target: Price | None = None  # Take profit level (absolute price)
    stop_loss: Price | None = None  # Stop loss level (absolute price)

    # Bracket order parameters (percentage-based, VectorBT compatible)
    tp_pct: float | None = None  # Take profit as percentage (e.g., 0.025 = 2.5%)
    sl_pct: float | None = None  # Stop loss as percentage (e.g., 0.02 = 2%)
    tsl_pct: float | None = None  # Trailing stop as percentage (e.g., 0.01 = 1%)

    # Time constraints
    time_in_force: TimeInForce = TimeInForce.DAY
    expire_time: datetime | None = None

    # State tracking
    state: OrderState = OrderState.PENDING
    status: OrderStatus = OrderStatus.CREATED

    # Timestamps
    created_time: datetime = field(default_factory=datetime.now)
    submitted_time: datetime | None = None
    acknowledged_time: datetime | None = None
    filled_time: datetime | None = None
    cancelled_time: datetime | None = None

    # Fill information
    filled_quantity: Quantity = 0.0
    average_fill_price: Price | None = None
    fill_count: int = 0

    # Costs
    commission: float = 0.0
    slippage: float = 0.0

    # Relationships
    parent_order_id: OrderId | None = None
    child_order_ids: list[OrderId] = field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Precision management
    precision_manager: Optional[PrecisionManager] = None

    def __post_init__(self):
        """Validate order on creation."""
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("Limit orders must have a limit price")
        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError("Stop orders must have a stop price")
        if self.order_type == OrderType.STOP_LIMIT:
            if self.stop_price is None or self.limit_price is None:
                raise ValueError("Stop-limit orders must have both stop and limit prices")
        if self.order_type == OrderType.TRAILING_STOP:
            if self.trail_amount is None and self.trail_percent is None:
                raise ValueError("Trailing stop orders must have trail_amount or trail_percent")
        if self.order_type == OrderType.BRACKET:
            # Allow either absolute prices OR percentage-based parameters
            has_absolute = (self.profit_target is not None) or (self.stop_loss is not None)
            has_percentage = (self.tp_pct is not None) or (self.sl_pct is not None) or (self.tsl_pct is not None)

            if not (has_absolute or has_percentage):
                raise ValueError(
                    "Bracket orders must have exit parameters: "
                    "profit_target/stop_loss (absolute) OR tp_pct/sl_pct/tsl_pct (percentage)"
                )
        if self.quantity <= 0:
            raise ValueError("Order quantity must be positive")

    @property
    def is_buy(self) -> bool:
        """Check if this is a buy order."""
        return self.side == OrderSide.BUY

    @property
    def is_sell(self) -> bool:
        """Check if this is a sell order."""
        return self.side == OrderSide.SELL

    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.state == OrderState.FILLED

    @property
    def is_partially_filled(self) -> bool:
        """Check if order is partially filled."""
        return self.state == OrderState.PARTIALLY_FILLED

    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.state in [
            OrderState.PENDING,
            OrderState.SUBMITTED,
            OrderState.ACKNOWLEDGED,
            OrderState.PARTIALLY_FILLED,
        ]

    @property
    def is_terminal(self) -> bool:
        """Check if order is in a terminal state."""
        return self.state in [
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
        ]

    @property
    def remaining_quantity(self) -> Quantity:
        """Get remaining quantity to fill."""
        return self.quantity - self.filled_quantity

    @property
    def fill_ratio(self) -> float:
        """Get the ratio of filled quantity to total."""
        if self.quantity == 0:
            return 0.0
        return self.filled_quantity / self.quantity

    def can_fill(
        self,
        price: Price | None = None,
        high: Price | None = None,
        low: Price | None = None,
    ) -> bool:
        """
        Check if order can be filled at given price or OHLC range.

        For intrabar execution (matching VectorBT Pro):
        - Limit orders (TP): Check if high (for longs) or low (for shorts) reached limit
        - Stop orders (SL/TSL): Check if low (for longs) or high (for shorts) reached stop
        - Market orders: Always fill

        Args:
            price: Current market price (close) - for backward compatibility
            high: Bar's high price (for intrabar limit order checks)
            low: Bar's low price (for intrabar stop order checks)

        Returns:
            True if order can be filled

        Note:
            If high/low are provided, uses intrabar execution logic.
            If only price is provided, falls back to end-of-bar logic.
        """
        if not self.is_active:
            return False

        # Market orders always fill
        if self.order_type == OrderType.MARKET:
            return True

        # Determine check price based on order type and available data
        # For intrabar execution, use high/low to detect if limit/stop was touched
        use_intrabar = high is not None and low is not None

        if self.order_type == OrderType.LIMIT:
            if self.limit_price is None:
                return False

            if use_intrabar:
                # Intrabar check: Did price touch the limit during the bar?
                # For LIMIT orders, we want to check if the bar touched the favorable side
                if self.is_buy:
                    # BUY LIMIT: Want to buy at or below limit (e.g., entry or short TP)
                    # Check if LOW reached limit (price went down to our buy limit)
                    return low <= self.limit_price
                else:
                    # SELL LIMIT: Want to sell at or above limit (e.g., long TP)
                    # Check if HIGH reached limit (price went up to our sell limit)
                    return high >= self.limit_price
            else:
                # End-of-bar check: Use closing price
                check_price = price if price is not None else 0.0
                if self.is_buy:
                    return check_price <= self.limit_price
                return check_price >= self.limit_price

        if self.order_type == OrderType.STOP:
            if self.stop_price is None:
                return False

            if use_intrabar:
                # Intrabar check: Did price touch the stop during the bar?
                if self.is_buy:
                    # Short cover stop: Check if HIGH reached stop
                    return high >= self.stop_price
                else:
                    # Long SL: Check if LOW reached stop (price went down to SL)
                    return low <= self.stop_price
            else:
                # End-of-bar check
                check_price = price if price is not None else 0.0
                if self.is_buy:
                    return check_price >= self.stop_price
                return check_price <= self.stop_price

        if self.order_type == OrderType.STOP_LIMIT:
            # For simplicity, assume stop has been triggered if we get here
            # The broker will handle the trigger logic
            if self.limit_price is None:
                return False

            if use_intrabar:
                if self.is_buy:
                    return high >= self.limit_price
                return low <= self.limit_price
            else:
                check_price = price if price is not None else 0.0
                if self.is_buy:
                    return check_price <= self.limit_price
                return check_price >= self.limit_price

        if self.order_type == OrderType.TRAILING_STOP:
            if self.trailing_stop_price is None:
                return False

            if use_intrabar:
                # Intrabar check for trailing stops
                if self.is_buy:
                    # Short cover: Check if HIGH reached trailing stop
                    return high >= self.trailing_stop_price
                else:
                    # Long TSL: Check if LOW reached trailing stop
                    return low <= self.trailing_stop_price
            else:
                # End-of-bar check
                check_price = price if price is not None else 0.0
                if self.is_buy:
                    return check_price >= self.trailing_stop_price
                return check_price <= self.trailing_stop_price

        if self.order_type == OrderType.BRACKET:
            # Bracket orders fill based on their entry criteria (limit_price if set)
            if self.limit_price is not None:
                # Act like a limit order for entry
                if use_intrabar:
                    if self.is_buy:
                        return high >= self.limit_price
                    return low <= self.limit_price
                else:
                    check_price = price if price is not None else 0.0
                    if self.is_buy:
                        return check_price <= self.limit_price
                    return check_price >= self.limit_price
            # Act like a market order for entry
            return True

        # OCO and other special orders
        return False

    def update_fill(
        self,
        fill_quantity: Quantity,
        fill_price: Price,
        commission: float = 0.0,
        timestamp: datetime | None = None,
    ) -> None:
        """
        Update order with fill information.

        Args:
            fill_quantity: Quantity filled
            fill_price: Price of fill
            commission: Commission charged
            timestamp: Time of fill
        """
        if fill_quantity <= 0:
            raise ValueError("Fill quantity must be positive")

        if fill_quantity > self.remaining_quantity:
            raise ValueError(
                f"Fill quantity {fill_quantity} exceeds remaining {self.remaining_quantity}",
            )

        # Update fill tracking
        if self.average_fill_price is None:
            self.average_fill_price = fill_price
        else:
            # Calculate weighted average
            total_value = (
                self.filled_quantity * self.average_fill_price + fill_quantity * fill_price
            )
            new_filled_quantity = self.filled_quantity + fill_quantity
            self.average_fill_price = total_value / new_filled_quantity
            # Round average fill price to avoid float drift (Location 1/3)
            if self.precision_manager:
                self.average_fill_price = self.precision_manager.round_cash(self.average_fill_price)

        self.filled_quantity += fill_quantity
        # Round filled quantity to avoid float drift (Location 2/3)
        if self.precision_manager:
            self.filled_quantity = self.precision_manager.round_quantity(self.filled_quantity)

        self.fill_count += 1

        self.commission += commission
        # Round commission to avoid float drift (Location 3/3)
        if self.precision_manager:
            self.commission = self.precision_manager.round_cash(self.commission)

        # Update state
        if self.filled_quantity >= self.quantity:
            self.state = OrderState.FILLED
            self.status = OrderStatus.FILLED
            self.filled_time = timestamp
        else:
            self.state = OrderState.PARTIALLY_FILLED
            self.status = OrderStatus.PARTIALLY_FILLED

    def cancel(self, timestamp: datetime | None = None) -> None:
        """Cancel the order."""
        if self.is_terminal:
            raise ValueError(f"Cannot cancel order in state {self.state}")

        self.state = OrderState.CANCELLED
        self.status = OrderStatus.CANCELED
        self.cancelled_time = timestamp

    def reject(self, reason: str = "", timestamp: datetime | None = None) -> None:
        """Reject the order."""
        self.state = OrderState.REJECTED
        self.status = OrderStatus.REJECTED
        self.metadata["rejection_reason"] = reason
        self.cancelled_time = timestamp

    def update_trailing_stop(self, current_price: Price) -> bool:
        """
        Update trailing stop price based on current market price.

        Args:
            current_price: Current market price

        Returns:
            True if trailing stop was updated, False otherwise
        """
        if self.order_type != OrderType.TRAILING_STOP:
            return False

        # Initialize trailing stop price if not set
        if self.trailing_stop_price is None:
            if self.trail_amount is not None:
                if self.is_buy:
                    self.trailing_stop_price = current_price + self.trail_amount
                else:
                    self.trailing_stop_price = current_price - self.trail_amount
            elif self.trail_percent is not None:
                trail_amount = current_price * (self.trail_percent / 100.0)
                if self.is_buy:
                    self.trailing_stop_price = current_price + trail_amount
                else:
                    self.trailing_stop_price = current_price - trail_amount
            return True

        # Update trailing stop if price moves favorably
        updated = False

        if self.trail_amount is not None:
            # Absolute trailing amount
            if self.is_buy:
                # For buy stops, trail up when price falls
                new_stop = current_price + self.trail_amount
                if new_stop < self.trailing_stop_price:
                    self.trailing_stop_price = new_stop
                    updated = True
            else:
                # For sell stops, trail down when price rises
                new_stop = current_price - self.trail_amount
                if new_stop > self.trailing_stop_price:
                    self.trailing_stop_price = new_stop
                    updated = True

        elif self.trail_percent is not None:
            # Percentage trailing amount
            trail_amount = current_price * (self.trail_percent / 100.0)
            if self.is_buy:
                new_stop = current_price + trail_amount
                if new_stop < self.trailing_stop_price:
                    self.trailing_stop_price = new_stop
                    updated = True
            else:
                new_stop = current_price - trail_amount
                if new_stop > self.trailing_stop_price:
                    self.trailing_stop_price = new_stop
                    updated = True

        return updated

    def __repr__(self) -> str:
        return (
            f"Order(id={self.order_id[:8]}, {self.side.value} {self.quantity} "
            f"{self.asset_id} @ {self.order_type.value}, state={self.state.value})"
        )
