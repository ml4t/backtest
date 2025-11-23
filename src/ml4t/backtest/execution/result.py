"""Execution result for partial fills."""

from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result of execution limit/impact calculation.

    Attributes:
        fillable_quantity: Quantity that can be filled this bar
        remaining_quantity: Quantity that must wait for next bar
        adjusted_price: Price after market impact adjustment
        impact_cost: Cost of market impact (price difference * quantity)
        participation_rate: Actual participation rate (fillable / volume)
    """

    fillable_quantity: float
    remaining_quantity: float
    adjusted_price: float
    impact_cost: float = 0.0
    participation_rate: float = 0.0

    @property
    def is_partial(self) -> bool:
        """True if order was partially filled."""
        return self.remaining_quantity > 0

    @property
    def is_full(self) -> bool:
        """True if entire order was filled."""
        return self.remaining_quantity == 0
