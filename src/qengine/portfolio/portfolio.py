"""Portfolio state management for QEngine."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Union

from qengine.core.types import AssetId, Cash, Quantity


@dataclass
class Position:
    """Represents a position in a single asset - current holdings only."""

    asset_id: AssetId
    quantity: Quantity = 0.0
    cost_basis: float = 0.0
    last_price: float = 0.0
    unrealized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        """Current market value of the position."""
        return self.quantity * self.last_price

    def update_price(self, price: float) -> None:
        """Update position with new market price."""
        self.last_price = price
        if self.quantity != 0:
            self.unrealized_pnl = self.quantity * (price - self.cost_basis / self.quantity)

    def add_shares(self, quantity: Quantity, price: float) -> None:
        """Add shares to position."""
        if self.quantity + quantity == 0:
            # Closing position
            self.unrealized_pnl = 0.0
            self.cost_basis = 0.0
        else:
            # Update cost basis
            new_cost = quantity * price
            self.cost_basis += new_cost

        self.quantity += quantity
        self.update_price(price)

    def remove_shares(self, quantity: Quantity, price: float) -> float:
        """Remove shares from position, returns realized P&L for this transaction."""
        # Use tolerance for floating-point comparison (e.g., 1.0 vs 0.9999999999999998)
        TOLERANCE = 1e-9
        if abs(quantity) > abs(self.quantity) + TOLERANCE:
            raise ValueError(f"Cannot remove {quantity} shares, only have {self.quantity}")

        # Calculate realized P&L for the shares being removed
        avg_cost = self.cost_basis / self.quantity if self.quantity != 0 else 0
        realized = quantity * (price - avg_cost)

        # Update cost basis
        self.cost_basis -= quantity * avg_cost
        self.quantity -= quantity

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


class Portfolio:
    """Portfolio management with state tracking."""

    def __init__(self, initial_cash: Cash = 100000.0):
        """Initialize portfolio with starting cash."""
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.positions: dict[AssetId, Position] = {}

        # Track cumulative costs and P&L
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_realized_pnl = 0.0
        self.asset_realized_pnl: dict[AssetId, float] = {}  # Per-asset P&L tracking

        # History tracking
        self.state_history: list[PortfolioState] = []

    def get_position(self, asset_id: AssetId) -> Union[Position, None]:
        """Get position for an asset."""
        return self.positions.get(asset_id)

    def update_position(
        self,
        asset_id: AssetId,
        quantity_change: Quantity,
        price: float,
        commission: float = 0.0,
        slippage: float = 0.0,
    ) -> None:
        """
        Update a position with a trade.

        Args:
            asset_id: Asset identifier
            quantity_change: Change in quantity (positive for buy, negative for sell)
            price: Execution price
            commission: Commission paid
            slippage: Slippage cost
        """
        # Get or create position
        if asset_id not in self.positions:
            self.positions[asset_id] = Position(asset_id=asset_id)

        position = self.positions[asset_id]

        # Update position
        if quantity_change > 0:
            position.add_shares(quantity_change, price)
            self.cash = float(self.cash) - (quantity_change * price + commission)
        else:
            realized_pnl = position.remove_shares(-quantity_change, price)
            self.cash = float(self.cash) + ((-quantity_change) * price - commission)

            # Track realized P&L (portfolio-level and per-asset)
            self.total_realized_pnl += realized_pnl
            if asset_id not in self.asset_realized_pnl:
                self.asset_realized_pnl[asset_id] = 0.0
            self.asset_realized_pnl[asset_id] += realized_pnl

        # Track costs
        self.total_commission += commission
        self.total_slippage += slippage

        # Remove empty positions (clean deletion rule)
        if position.quantity == 0:
            del self.positions[asset_id]

    def update_prices(self, prices: dict[AssetId, float]) -> None:
        """Update all positions with new market prices."""
        for asset_id, price in prices.items():
            if asset_id in self.positions:
                self.positions[asset_id].update_price(price)

    def get_current_state(self, timestamp: datetime) -> PortfolioState:
        """Get current portfolio state."""
        state = PortfolioState(
            timestamp=timestamp,
            cash=self.cash,
            positions=self.positions.copy(),
            total_commission=self.total_commission,
            total_slippage=self.total_slippage,
            total_realized_pnl=self.total_realized_pnl,
        )
        state.update_metrics()
        return state

    def save_state(self, timestamp: datetime) -> None:
        """Save current state to history."""
        self.state_history.append(self.get_current_state(timestamp))

    @property
    def equity(self) -> float:
        """Total equity (cash + positions)."""
        position_value = sum(p.market_value for p in self.positions.values())
        return float(self.cash) + position_value

    @property
    def returns(self) -> float:
        """Simple returns from initial capital."""
        if self.initial_cash == 0:
            return 0.0
        return (self.equity - float(self.initial_cash)) / float(self.initial_cash)

    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P&L."""
        return sum(p.unrealized_pnl for p in self.positions.values())

    def get_position_summary(self) -> dict[str, Any]:
        """Get summary of all positions."""
        return {
            "cash": self.cash,
            "equity": self.equity,
            "positions": len(self.positions),
            "realized_pnl": self.total_realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.total_realized_pnl + self.unrealized_pnl,
            "returns": self.returns,
            "commission": self.total_commission,
            "slippage": self.total_slippage,
        }
