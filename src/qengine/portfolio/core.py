"""Core position tracking - domain logic only (no analytics)."""

from typing import Any, Optional

from qengine.core.precision import PrecisionManager
from qengine.core.types import AssetId, Cash, Quantity
from qengine.portfolio.state import Position


class PositionTracker:
    """
    Core position and cash tracking - pure domain logic.

    This class handles:
    - Position management (buy/sell)
    - Cash tracking
    - Commission and slippage accounting
    - Realized P&L calculation

    This class does NOT handle:
    - Performance metrics (Sharpe, drawdown, etc.) → PerformanceAnalyzer
    - Trade history / persistence → TradeJournal
    - State history → Portfolio facade
    """

    def __init__(
        self,
        initial_cash: Cash = 100000.0,
        precision_manager: Optional[PrecisionManager] = None,
    ):
        """
        Initialize position tracker.

        Args:
            initial_cash: Starting cash balance
            precision_manager: PrecisionManager for cash rounding (USD precision)
        """
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.positions: dict[AssetId, Position] = {}
        self.precision_manager = precision_manager

        # Cumulative costs and P&L
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_realized_pnl = 0.0
        self.asset_realized_pnl: dict[AssetId, float] = {}  # Per-asset P&L tracking

    def get_position(self, asset_id: AssetId) -> Position | None:
        """Get position for an asset."""
        return self.positions.get(asset_id)

    def update_position(
        self,
        asset_id: AssetId,
        quantity_change: Quantity,
        price: float,
        commission: float = 0.0,
        slippage: float = 0.0,
        asset_precision_manager: Optional[PrecisionManager] = None,
    ) -> None:
        """
        Update a position with a trade.

        Args:
            asset_id: Asset identifier
            quantity_change: Change in quantity (positive for buy, negative for sell)
            price: Execution price
            commission: Commission paid
            slippage: Slippage cost
            asset_precision_manager: PrecisionManager for asset-specific quantity precision
        """
        # Get or create position with precision manager
        if asset_id not in self.positions:
            self.positions[asset_id] = Position(
                asset_id=asset_id,
                precision_manager=asset_precision_manager or self.precision_manager,
            )

        position = self.positions[asset_id]

        # Update position
        if quantity_change > 0:
            # Buy
            position.add_shares(quantity_change, price)
            cash_change = quantity_change * price + commission
            self.cash = float(self.cash) - cash_change
            # Round cash after BUY
            if self.precision_manager:
                self.cash = self.precision_manager.round_cash(self.cash)
        else:
            # Sell
            realized_pnl = position.remove_shares(-quantity_change, price)
            cash_change = (-quantity_change) * price - commission
            self.cash = float(self.cash) + cash_change
            # Round cash after SELL
            if self.precision_manager:
                self.cash = self.precision_manager.round_cash(self.cash)

            # Track realized P&L (portfolio-level and per-asset)
            self.total_realized_pnl += realized_pnl
            # Round total realized P&L
            if self.precision_manager:
                self.total_realized_pnl = self.precision_manager.round_cash(self.total_realized_pnl)

            if asset_id not in self.asset_realized_pnl:
                self.asset_realized_pnl[asset_id] = 0.0
            self.asset_realized_pnl[asset_id] += realized_pnl
            # Round asset realized P&L
            if self.precision_manager:
                self.asset_realized_pnl[asset_id] = self.precision_manager.round_cash(
                    self.asset_realized_pnl[asset_id]
                )

        # Track costs
        self.total_commission += commission
        self.total_slippage += slippage
        # Round cumulative costs
        if self.precision_manager:
            self.total_commission = self.precision_manager.round_cash(self.total_commission)
            self.total_slippage = self.precision_manager.round_cash(self.total_slippage)

        # Remove empty positions (clean deletion rule with precision-aware check)
        is_empty = position.quantity == 0
        if asset_precision_manager:
            is_empty = is_empty or asset_precision_manager.is_position_zero(position.quantity)
        if is_empty:
            del self.positions[asset_id]

    def update_prices(self, prices: dict[AssetId, float]) -> None:
        """Update all positions with new market prices."""
        for asset_id, price in prices.items():
            if asset_id in self.positions:
                self.positions[asset_id].update_price(price)

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

    def get_summary(self) -> dict[str, Any]:
        """Get summary of current position tracking state."""
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

    def reset(self) -> None:
        """Reset to initial state."""
        self.cash = self.initial_cash
        self.positions.clear()
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_realized_pnl = 0.0
        self.asset_realized_pnl.clear()


__all__ = ["PositionTracker"]
