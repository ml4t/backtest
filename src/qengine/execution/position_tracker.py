"""Position and cash tracking for broker simulation.

This module provides position and cash management functionality extracted from
SimulationBroker to follow the Single Responsibility Principle.
"""

import logging
from collections import defaultdict
from typing import Any, TYPE_CHECKING

from qengine.core.types import AssetId, OrderSide, Quantity

if TYPE_CHECKING:
    from qengine.core.precision import PrecisionManager

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

    def __init__(
        self,
        initial_cash: float,
        precision_manager: "PrecisionManager | None" = None,
    ) -> None:
        """Initialize position tracker.

        Args:
            initial_cash: Starting cash balance
            precision_manager: Optional PrecisionManager for rounding (if None, no rounding applied)
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self._positions: dict[AssetId, Quantity] = defaultdict(float)
        self.precision_manager = precision_manager

        logger.debug(f"PositionTracker initialized with ${initial_cash:,.2f}")

    def update_position(
        self,
        asset_id: AssetId,
        quantity: Quantity,
        side: OrderSide,
        fill_price: float,
        commission: float,
        slippage: float,
        asset_precision_manager: "PrecisionManager | None" = None,
    ) -> None:
        """Update position and cash after a fill.

        Args:
            asset_id: Asset identifier
            quantity: Fill quantity (always positive, should already be rounded by FillSimulator)
            side: Order side (BUY or SELL)
            fill_price: Execution price (already includes slippage)
            commission: Commission paid
            slippage: Slippage cost (for statistics only, already in fill_price)
            asset_precision_manager: Optional asset-specific PrecisionManager for position rounding

        Raises:
            ValueError: If quantity is negative

        Note:
            The slippage parameter is for tracking/statistics only. The actual
            slippage cost is already included in the fill_price.

            Quantity should already be rounded by FillSimulator, but we round again here
            to ensure consistency if position arithmetic introduces float drift.
        """
        if quantity < 0:
            raise ValueError(f"Fill quantity must be positive, got {quantity}")

        # Choose precision manager for position rounding (asset-specific) vs cash rounding (USD)
        position_pm = asset_precision_manager if asset_precision_manager else self.precision_manager
        cash_pm = self.precision_manager  # Always use cash precision manager for USD operations

        # Update position
        if side == OrderSide.BUY:
            # INTEGRATION POINT 1: Round position quantity after addition
            new_position = self._positions[asset_id] + quantity
            if position_pm:
                new_position = position_pm.round_quantity(new_position)
            self._positions[asset_id] = new_position

            # INTEGRATION POINT 2: Round cash impact (using cash precision)
            cash_impact = quantity * fill_price + commission
            if cash_pm:
                cash_impact = cash_pm.round_cash(cash_impact)

            # INTEGRATION POINT 3: Round cash balance after update
            self.cash -= cash_impact
            if cash_pm:
                self.cash = cash_pm.round_cash(self.cash)

            logger.debug(
                f"BUY {quantity} {asset_id} @ ${fill_price:.2f}, "
                f"cash impact: -${cash_impact:.2f} (incl. slippage=${slippage:.2f})"
            )
        else:  # SELL
            # INTEGRATION POINT 4: Round position quantity after subtraction
            new_position = self._positions[asset_id] - quantity
            if position_pm:
                new_position = position_pm.round_quantity(new_position)
            self._positions[asset_id] = new_position

            # INTEGRATION POINT 5: Round cash impact (using cash precision)
            cash_impact = quantity * fill_price - commission
            if cash_pm:
                cash_impact = cash_pm.round_cash(cash_impact)

            # INTEGRATION POINT 6: Round cash balance after update
            self.cash += cash_impact
            if cash_pm:
                self.cash = cash_pm.round_cash(self.cash)

            logger.debug(
                f"SELL {quantity} {asset_id} @ ${fill_price:.2f}, "
                f"cash impact: +${cash_impact:.2f} (incl. slippage=${slippage:.2f})"
            )

        # INTEGRATION POINT 7 & 8: Clean up zero positions using precision-aware check
        if position_pm:
            if position_pm.is_position_zero(self._positions[asset_id]):
                del self._positions[asset_id]
        else:
            # Fallback to original logic if no precision manager
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
