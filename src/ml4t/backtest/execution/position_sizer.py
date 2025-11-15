"""Position sizing models for order quantity calculation."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import math

if TYPE_CHECKING:
    from ml4t.backtest.core.types import Price, Quantity, Cash
    from ml4t.backtest.execution.order import Order
    from ml4t.backtest.execution.commission import CommissionModel
    from ml4t.backtest.execution.slippage import SlippageModel


class PositionSizer(ABC):
    """Abstract base class for position sizing models."""

    @abstractmethod
    def calculate_quantity(
        self,
        price: "Price",
        available_cash: "Cash",
        commission_model: "CommissionModel",
        slippage_model: "SlippageModel",
        order: "Order",
    ) -> "Quantity":
        """Calculate position size for an order.

        Args:
            price: Current market price (base price before slippage)
            available_cash: Available cash for the trade
            commission_model: Commission model to estimate fees
            slippage_model: Slippage model to estimate price adjustment
            order: The order being sized (contains side, asset_id, etc.)

        Returns:
            Quantity to trade (always positive, sign handled by order side)
        """

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}()"


class FixedQuantitySizer(PositionSizer):
    """Fixed quantity position sizer.

    Always returns the same quantity regardless of cash or price.
    """

    def __init__(self, quantity: "Quantity"):
        """Initialize fixed quantity sizer.

        Args:
            quantity: Fixed quantity to trade
        """
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        self.quantity = quantity

    def calculate_quantity(
        self,
        price: "Price",
        available_cash: "Cash",
        commission_model: "CommissionModel",
        slippage_model: "SlippageModel",
        order: "Order",
    ) -> "Quantity":
        """Return fixed quantity."""
        return self.quantity

    def __repr__(self) -> str:
        """String representation."""
        return f"FixedQuantitySizer(quantity={self.quantity})"


class PercentageOfEquitySizer(PositionSizer):
    """Size positions as percentage of total equity.

    Useful for Kelly criterion or risk-based position sizing.
    """

    def __init__(self, percentage: float = 0.1):
        """Initialize percentage of equity sizer.

        Args:
            percentage: Fraction of equity to allocate (0.1 = 10%)
        """
        if not 0 < percentage <= 1.0:
            raise ValueError("Percentage must be between 0 and 1")
        self.percentage = percentage

    def calculate_quantity(
        self,
        price: "Price",
        available_cash: "Cash",
        commission_model: "CommissionModel",
        slippage_model: "SlippageModel",
        order: "Order",
    ) -> "Quantity":
        """Calculate quantity as percentage of equity.

        Note: This uses available_cash as proxy for equity.
        For accurate equity, pass (cash + position_value) as available_cash.
        """
        # Apply slippage to get adjusted price
        adj_price = slippage_model.calculate_fill_price(
            order=order,
            market_price=price,
        )

        # Calculate target value
        target_value = available_cash * self.percentage

        # Estimate fees - check for both 'fee_rate' and 'rate' attributes
        fee_rate = 0.0
        if hasattr(commission_model, 'fee_rate'):
            fee_rate = commission_model.fee_rate
        elif hasattr(commission_model, 'rate'):
            fee_rate = commission_model.rate

        # Adjust for fees
        target_value_after_fees = target_value / (1.0 + fee_rate)

        # Calculate quantity
        quantity = target_value_after_fees / adj_price

        return max(0.0, quantity)

    def __repr__(self) -> str:
        """String representation."""
        return f"PercentageOfEquitySizer(percentage={self.percentage})"
