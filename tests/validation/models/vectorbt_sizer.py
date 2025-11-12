"""VectorBT Pro size=np.inf position sizer for validation testing.

This module exists ONLY for validation purposes - to ensure qengine produces
identical position sizing to VectorBT Pro during comparative testing.

DO NOT import this into production code (src/qengine). This is a test fixture.
"""

import math
from typing import TYPE_CHECKING

# Import from production for base class
from qengine.execution.position_sizer import PositionSizer

if TYPE_CHECKING:
    from qengine.core.types import Price, Quantity, Cash
    from qengine.execution.order import Order
    from qengine.execution.commission import CommissionModel
    from qengine.execution.slippage import SlippageModel


class VectorBTInfiniteSizer(PositionSizer):
    """VectorBT Pro size=np.inf position sizer.

    Calculates maximum position size that can be purchased with available cash,
    accounting for fees, slippage, and granularity constraints.

    This replicates VectorBT Pro's size=np.inf behavior exactly:
    1. Apply slippage to base price
    2. Reserve cash for fixed fees
    3. Calculate fee-adjusted cash
    4. Calculate maximum size
    5. Round down to granularity
    6. Recalculate actual fees

    Formula:
        adj_price = base_price * (1 + slippage)
        max_req_cash = (cash_limit - fixed_fees) / (1 + fee_rate)
        max_size = max_req_cash / adj_price
        final_size = floor(max_size / granularity) * granularity

    Reference: TASK-011 completion document
    """

    def __init__(
        self,
        granularity: float = 0.001,
        allow_partial: bool = True,
    ):
        """Initialize VectorBT infinite sizer.

        Args:
            granularity: Minimum quantity increment (e.g., 0.001 BTC)
            allow_partial: If False, reject orders that can't fill full granularity
        """
        if granularity <= 0:
            raise ValueError("Granularity must be positive")
        self.granularity = granularity
        self.allow_partial = allow_partial

    def calculate_quantity(
        self,
        price: "Price",
        available_cash: "Cash",
        commission_model: "CommissionModel",
        slippage_model: "SlippageModel",
        order: "Order",
    ) -> "Quantity":
        """Calculate maximum quantity using VectorBT formula.

        Args:
            price: Base market price (before slippage)
            available_cash: Cash available for trade
            commission_model: Commission model for fee estimation
            slippage_model: Slippage model for price adjustment
            order: Order being sized

        Returns:
            Maximum quantity that can be purchased, rounded to granularity

        Raises:
            ValueError: If insufficient cash to cover fees or minimum quantity
        """
        # Step 1: Apply slippage to get adjusted price
        adj_price = slippage_model.calculate_fill_price(
            order=order,
            market_price=price,
        )

        if adj_price <= 0:
            raise ValueError(f"Invalid adjusted price: {adj_price}")

        # Step 2: Estimate fixed fees (if commission model supports it)
        # For two-component models like VectorBTCommission, we need to get fixed_fee
        fixed_fees = 0.0
        fee_rate = 0.0

        # Try to extract fixed fee and rate from commission model
        if hasattr(commission_model, 'fixed_fee'):
            fixed_fees = commission_model.fixed_fee
        if hasattr(commission_model, 'fee_rate'):
            fee_rate = commission_model.fee_rate
        elif hasattr(commission_model, 'rate'):
            fee_rate = commission_model.rate
        elif hasattr(commission_model, 'commission'):
            # Flat commission models - treat as fixed fee
            fixed_fees = commission_model.commission
        else:
            # Fallback: estimate using a small quantity
            test_commission = commission_model.calculate(
                order=order,
                fill_quantity=0.001,
                fill_price=adj_price,
            )
            # If commission is non-zero, treat as percentage
            if test_commission > 0:
                test_value = 0.001 * adj_price
                fee_rate = test_commission / test_value

        # Step 3: Calculate fee-adjusted cash
        max_req_cash = (available_cash - fixed_fees) / (1.0 + fee_rate)

        if max_req_cash <= 0:
            raise ValueError(
                f"Insufficient cash to cover fees: available={available_cash}, "
                f"fixed_fees={fixed_fees}, fee_rate={fee_rate}"
            )

        # Step 4: Calculate maximum size
        max_size = max_req_cash / adj_price

        if max_size <= 0:
            raise ValueError(f"Invalid max_size: {max_size}")

        # Step 5: Apply granularity (round down)
        final_size = math.floor(max_size / self.granularity) * self.granularity

        # Check if we have at least one granularity unit
        if final_size < self.granularity:
            if not self.allow_partial:
                raise ValueError(
                    f"Insufficient cash for minimum quantity: "
                    f"max_size={max_size}, granularity={self.granularity}"
                )
            # Allow zero quantity (will be rejected by broker)
            return 0.0

        # Step 6: Verify total cost doesn't exceed available cash
        # (This is a sanity check - formula should guarantee it)
        order_value = final_size * adj_price
        actual_commission = commission_model.calculate(
            order=order,
            fill_quantity=final_size,
            fill_price=adj_price,
        )
        total_cost = order_value + actual_commission

        if total_cost > available_cash:
            # This should never happen if formula is correct, but be defensive
            # Reduce size by one granularity unit
            final_size = final_size - self.granularity
            if final_size < self.granularity:
                raise ValueError(
                    f"Cannot reduce size further: cost={total_cost}, "
                    f"cash={available_cash}"
                )

        return final_size

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"VectorBTInfiniteSizer(granularity={self.granularity}, "
            f"allow_partial={self.allow_partial})"
        )
