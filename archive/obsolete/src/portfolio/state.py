"""Portfolio state data structures."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from ml4t.backtest.core.precision import PrecisionManager
from ml4t.backtest.core.types import AssetId, Cash, Quantity


@dataclass
class Position:
    """Represents a position in a single asset - current holdings only."""

    asset_id: AssetId
    quantity: Quantity = 0.0
    cost_basis: float = 0.0
    last_price: float = 0.0
    unrealized_pnl: float = 0.0
    precision_manager: Optional[PrecisionManager] = None

    @property
    def market_value(self) -> float:
        """Current market value of the position."""
        return self.quantity * self.last_price

    def update_price(self, price: float) -> None:
        """Update position with new market price."""
        self.last_price = price
        if self.quantity != 0:
            avg_cost = self.cost_basis / self.quantity
            self.unrealized_pnl = self.quantity * (price - avg_cost)
            # Round unrealized P&L to avoid float drift
            if self.precision_manager:
                self.unrealized_pnl = self.precision_manager.round_cash(self.unrealized_pnl)

    def add_shares(self, quantity: Quantity, price: float) -> None:
        """Add shares to position."""
        new_quantity = self.quantity + quantity

        # Round new quantity to avoid float drift
        if self.precision_manager:
            new_quantity = self.precision_manager.round_quantity(new_quantity)

        if new_quantity == 0 or (
            self.precision_manager and self.precision_manager.is_position_zero(new_quantity)
        ):
            # Closing position
            self.unrealized_pnl = 0.0
            self.cost_basis = 0.0
            self.quantity = 0.0
        else:
            # Update cost basis
            new_cost = quantity * price
            self.cost_basis += new_cost
            # Round cost basis to avoid float drift
            if self.precision_manager:
                self.cost_basis = self.precision_manager.round_cash(self.cost_basis)
            self.quantity = new_quantity

        self.update_price(price)

    def remove_shares(self, quantity: Quantity, price: float) -> float:
        """Remove shares from position, returns realized P&L for this transaction."""
        # Use precision-aware tolerance if available
        if self.precision_manager:
            if quantity > self.quantity and not self.precision_manager.is_position_zero(
                quantity - self.quantity
            ):
                raise ValueError(f"Cannot remove {quantity} shares, only have {self.quantity}")
        else:
            # Fall back to fixed tolerance (1e-6 handles floating point precision issues)
            TOLERANCE = 1e-6
            if abs(quantity) > abs(self.quantity) + TOLERANCE:
                raise ValueError(f"Cannot remove {quantity} shares, only have {self.quantity}")

        # Calculate realized P&L for the shares being removed
        avg_cost = self.cost_basis / self.quantity if self.quantity != 0 else 0
        realized = quantity * (price - avg_cost)
        # Round realized P&L to avoid float drift
        if self.precision_manager:
            realized = self.precision_manager.round_cash(realized)

        # Update cost basis and quantity
        self.cost_basis -= quantity * avg_cost
        # Round cost basis to avoid float drift
        if self.precision_manager:
            self.cost_basis = self.precision_manager.round_cash(self.cost_basis)

        self.quantity -= quantity
        # Round quantity to avoid float drift
        if self.precision_manager:
            self.quantity = self.precision_manager.round_quantity(self.quantity)

        self.update_price(price)
        return realized


@dataclass
class PortfolioState:
    """Complete portfolio state at a point in time."""

    timestamp: datetime
    cash: Cash
    positions: dict[AssetId, Position] = field(default_factory=dict)
    pending_orders: list[Any] = field(default_factory=list)
    filled_orders: list[Any] = field(default_factory=list)

    # Performance metrics
    total_value: float = 0.0
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0

    # Risk metrics
    leverage: float = 1.0
    max_position_value: float = 0.0
    concentration: float = 0.0

    @property
    def equity(self) -> float:
        """Total equity (cash + positions)."""
        position_value = sum(p.market_value for p in self.positions.values())
        return float(self.cash) + position_value

    @property
    def total_pnl(self) -> float:
        """Total P&L across all positions."""
        return self.total_realized_pnl + self.total_unrealized_pnl

    def update_metrics(self) -> None:
        """Update portfolio metrics."""
        # Update position values
        position_values = [p.market_value for p in self.positions.values()]

        if position_values:
            self.max_position_value = max(abs(v) for v in position_values)
            total_position_value = sum(abs(v) for v in position_values)

            # Update concentration (largest position as % of portfolio)
            if self.equity > 0:
                self.concentration = self.max_position_value / self.equity
                self.leverage = total_position_value / self.equity
            else:
                self.concentration = 0.0
                self.leverage = 0.0
        else:
            self.max_position_value = 0.0
            self.concentration = 0.0
            self.leverage = 0.0

        # Update P&L (realized P&L is tracked at Portfolio level, not per Position)
        self.total_unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        # total_realized_pnl is set from Portfolio.total_realized_pnl (passed in get_current_state)
        self.total_value = self.equity


__all__ = ["Position", "PortfolioState"]
