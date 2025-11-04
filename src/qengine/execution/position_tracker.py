"""Position and cash tracking for broker simulation.

This module provides position and cash management functionality extracted from
SimulationBroker to follow the Single Responsibility Principle.
"""

import logging
from collections import defaultdict
from typing import Any

from qengine.core.types import AssetId, OrderSide, Quantity

logger = logging.getLogger(__name__)


class PositionTracker:
    """Tracks positions and cash for a simulated broker.

    Responsibilities:
    - Maintain position quantities per asset
    - Track available cash
    - Update positions on fills
    - Query current positions and cash

    This class does NOT:
    - Execute orders (FillSimulator)
    - Manage order queues (OrderRouter)
    - Calculate P&L (Portfolio)
    """

    def __init__(self, initial_cash: float) -> None:
        """Initialize position tracker.

        Args:
            initial_cash: Starting cash balance
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self._positions: dict[AssetId, Quantity] = defaultdict(float)

        logger.debug(f"PositionTracker initialized with ${initial_cash:,.2f}")

    def update_position(
        self,
        asset_id: AssetId,
        quantity: Quantity,
        side: OrderSide,
        fill_price: float,
        commission: float,
        slippage: float,
    ) -> None:
        """Update position and cash after a fill.

        Args:
            asset_id: Asset identifier
            quantity: Fill quantity (always positive)
            side: Order side (BUY or SELL)
            fill_price: Execution price (already includes slippage)
            commission: Commission paid
            slippage: Slippage cost (for statistics only, already in fill_price)

        Raises:
            ValueError: If quantity is negative

        Note:
            The slippage parameter is for tracking/statistics only. The actual
            slippage cost is already included in the fill_price.
        """
        if quantity < 0:
            raise ValueError(f"Fill quantity must be positive, got {quantity}")

        # Update position
        if side == OrderSide.BUY:
            self._positions[asset_id] += quantity
            # Deduct cash: cost + commission (slippage already in fill_price)
            cash_impact = quantity * fill_price + commission
            self.cash -= cash_impact
            logger.debug(
                f"BUY {quantity} {asset_id} @ ${fill_price:.2f}, "
                f"cash impact: -${cash_impact:.2f} (incl. slippage=${slippage:.2f})"
            )
        else:  # SELL
            self._positions[asset_id] -= quantity
            # Add cash: proceeds - commission (slippage already in fill_price)
            cash_impact = quantity * fill_price - commission
            self.cash += cash_impact
            logger.debug(
                f"SELL {quantity} {asset_id} @ ${fill_price:.2f}, "
                f"cash impact: +${cash_impact:.2f} (incl. slippage=${slippage:.2f})"
            )

        # Clean up zero positions
        if abs(self._positions[asset_id]) < 1e-9:
            del self._positions[asset_id]

    def get_position(self, asset_id: AssetId) -> Quantity:
        """Get current position quantity for an asset.

        Args:
            asset_id: Asset identifier

        Returns:
            Position quantity (positive for long, negative for short, 0 for flat)
        """
        return self._positions.get(asset_id, 0.0)

    def get_all_positions(self) -> dict[AssetId, Quantity]:
        """Get all non-zero positions.

        Returns:
            Dictionary mapping asset_id to quantity
        """
        return dict(self._positions)

    def get_cash(self) -> float:
        """Get current cash balance.

        Returns:
            Available cash
        """
        return self.cash

    def has_sufficient_cash(self, required_cash: float) -> bool:
        """Check if sufficient cash is available.

        Args:
            required_cash: Amount needed

        Returns:
            True if cash >= required_cash
        """
        return self.cash >= required_cash

    def reset(self) -> None:
        """Reset to initial state."""
        self.cash = self.initial_cash
        self._positions.clear()
        logger.debug("PositionTracker reset")

    def get_statistics(self) -> dict[str, Any]:
        """Get position statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "cash": self.cash,
            "initial_cash": self.initial_cash,
            "num_positions": len(self._positions),
            "positions": dict(self._positions),
        }
