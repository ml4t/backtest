"""VectorBT Pro compatible commission model for validation testing.

This module exists ONLY for validation purposes - to ensure qengine produces
identical commission calculations to VectorBT Pro during comparative testing.

DO NOT import this into production code (src/qengine). This is a test fixture.
"""

from typing import TYPE_CHECKING

# Import from production for base class
from qengine.execution.commission import CommissionModel

if TYPE_CHECKING:
    from qengine.core.types import Price, Quantity
    from qengine.execution.order import Order


class VectorBTCommission(CommissionModel):
    """VectorBT Pro compatible two-component commission model.

    Implements VectorBT's fee calculation exactly as documented in TASK-009:
    - Percentage fees: Applied to slippage-adjusted order value
    - Fixed fees: Added per transaction (entry or exit)
    - Total fees = (order_value * fees) + fixed_fees

    This model calculates fees on the FILLED price (after slippage), which
    matches VectorBT behavior where slippage is applied BEFORE fee calculation.

    Reference: TASK-009 - VectorBT Fee Calculation and Application
    """

    def __init__(self, fee_rate: float = 0.0002, fixed_fee: float = 0.0):
        """Initialize VectorBT-compatible commission model.

        Args:
            fee_rate: Percentage fee rate as decimal (0.0002 = 0.02% = 2bps)
            fixed_fee: Fixed fee per transaction in currency terms (default 0.0)

        Example:
            # VectorBT default: 0.02% fees, no fixed fee
            commission = VectorBTCommission(fee_rate=0.0002, fixed_fee=0.0)

            # With fixed fee: 0.02% + $5 per trade
            commission = VectorBTCommission(fee_rate=0.0002, fixed_fee=5.0)
        """
        if fee_rate < 0:
            raise ValueError("Fee rate cannot be negative")
        if fee_rate > 0.1:  # 10% cap as sanity check
            raise ValueError("Fee rate too high (>10%)")
        if fixed_fee < 0:
            raise ValueError("Fixed fee cannot be negative")

        self.fee_rate = fee_rate
        self.fixed_fee = fixed_fee

    def calculate(
        self,
        order: "Order",
        fill_quantity: "Quantity",
        fill_price: "Price",
    ) -> float:
        """Calculate VectorBT two-component commission.

        Formula (from TASK-009):
            total_fees = (order_value * fees) + fixed_fees

        Where:
            order_value = fill_quantity * fill_price
            fill_price = slippage-adjusted price (already applied by FillSimulator)

        Args:
            order: The order being filled
            fill_quantity: Quantity of the fill
            fill_price: Price at which order was filled (post-slippage)

        Returns:
            Total commission in currency terms

        Note:
            FillSimulator applies slippage BEFORE calling this method, so
            fill_price is already the slippage-adjusted price. This matches
            VectorBT's order of operations: Slippage → Order Value → Fees
        """
        # Calculate order value (slippage already applied to fill_price)
        order_value = fill_quantity * fill_price

        # Calculate percentage fees on order value
        percentage_fees = order_value * self.fee_rate

        # Add fixed fees
        total_fees = percentage_fees + self.fixed_fee

        return self._round_commission(total_fees)

    def __repr__(self) -> str:
        """String representation."""
        return f"VectorBTCommission(fee_rate={self.fee_rate}, fixed_fee={self.fixed_fee})"
