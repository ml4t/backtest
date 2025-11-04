"""Simple portfolio implementation for basic backtesting."""

import logging
from datetime import datetime
from typing import Any, Optional

import polars as pl

from qengine.core.event import FillEvent, MarketEvent
from qengine.core.types import AssetId, Cash
from qengine.portfolio.portfolio import Portfolio

logger = logging.getLogger(__name__)


class SimplePortfolio(Portfolio):
    """Simple portfolio implementation with basic tracking.

    This portfolio provides:
    - Position tracking
    - P&L calculation
    - Basic performance metrics
    - Event handling integration
    """

    def __init__(self, initial_capital: Cash = 100000.0, currency: str = "USD"):
        """Initialize simple portfolio.

        Args:
            initial_capital: Starting cash balance
            currency: Base currency
        """
        super().__init__(initial_cash=initial_capital)
        self.currency = currency
        self.trades: list[dict[str, Any]] = []  # Track all trades
        self.current_prices: dict[AssetId, float] = {}  # Latest market prices

    def initialize(self) -> None:
        """Initialize portfolio for new backtest."""
        logger.debug(f"Initializing portfolio with ${self.initial_cash:,.2f} {self.currency}")

    def on_fill_event(self, event: FillEvent) -> None:
        """Handle fill event from broker.

        Args:
            event: Fill event with execution details
        """
        # Update position
        self.update_position(
            asset_id=event.asset_id,
            quantity_change=event.fill_quantity
            if event.side.value in ["buy", "BUY"]
            else -event.fill_quantity,
            price=float(event.fill_price),
            commission=event.commission,
            slippage=event.slippage,
        )

        # Record trade
        self.trades.append(
            {
                "timestamp": event.timestamp,
                "asset_id": event.asset_id,
                "side": event.side.value,
                "quantity": event.fill_quantity,
                "price": float(event.fill_price),
                "commission": event.commission,
                "slippage": event.slippage,
                "pnl": 0.0,  # Will be calculated later
            },
        )

        logger.info(
            f"Fill: {event.side.value.upper()} {event.fill_quantity} {event.asset_id} "
            f"@ ${float(event.fill_price):.2f} (commission: ${event.commission:.2f})",
        )

    def update_market_value(self, event: MarketEvent) -> None:
        """Update portfolio with latest market prices.

        Args:
            event: Market event with price data
        """
        # Update current price for the asset
        if hasattr(event, "close") and event.close is not None:
            self.current_prices[event.asset_id] = float(event.close)
        elif hasattr(event, "price") and event.price is not None:
            self.current_prices[event.asset_id] = float(event.price)

        # Update all positions with latest prices
        self.update_prices(self.current_prices)

    def get_total_value(self) -> float:
        """Get total portfolio value (cash + positions).

        Returns:
            Total portfolio value
        """
        return self.equity

    def get_positions(self) -> pl.DataFrame:
        """Get DataFrame of current positions.

        Returns:
            DataFrame with position details
        """
        if not self.positions:
            return pl.DataFrame()

        positions_data = []
        for asset_id, position in self.positions.items():
            # Get realized P&L from portfolio (tracked at portfolio level, not position level)
            realized_pnl = self.asset_realized_pnl.get(asset_id, 0.0)
            positions_data.append(
                {
                    "asset_id": asset_id,
                    "quantity": position.quantity,
                    "cost_basis": position.cost_basis,
                    "market_value": position.market_value,
                    "unrealized_pnl": position.unrealized_pnl,
                    "realized_pnl": realized_pnl,
                    "last_price": position.last_price,
                },
            )

        return pl.DataFrame(positions_data)

    def get_trades(self) -> pl.DataFrame:
        """Get DataFrame of all trades.

        Returns:
            DataFrame with trade history
        """
        if not self.trades:
            return pl.DataFrame()
        return pl.DataFrame(self.trades)

    def get_returns(self) -> pl.Series:
        """Get returns series.

        Returns:
            Series of portfolio returns
        """
        if not self.state_history:
            return pl.Series([])

        returns = []
        prev_value = self.initial_cash

        for state in self.state_history:
            current_value = state.equity
            ret = (current_value - prev_value) / prev_value if prev_value != 0 else 0
            returns.append(ret)
            prev_value = current_value

        return pl.Series(returns)

    def calculate_metrics(self) -> dict[str, Any]:
        """Calculate performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        returns = self.get_returns()

        metrics = {
            "total_return": self.returns * 100,  # Percentage
            "total_trades": len(self.trades),
            "winning_trades": sum(1 for t in self.trades if t.get("pnl", 0) > 0),
            "losing_trades": sum(1 for t in self.trades if t.get("pnl", 0) < 0),
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
            "final_equity": self.equity,
            "cash_remaining": self.cash,
        }

        # Calculate returns-based metrics if we have data
        if len(returns) > 0:
            import numpy as np

            returns_array = returns.to_numpy()

            # Remove any NaN values
            returns_array = returns_array[~np.isnan(returns_array)]

            if len(returns_array) > 0:
                metrics["avg_return"] = float(np.mean(returns_array))
                metrics["std_return"] = float(np.std(returns_array))

                # Sharpe ratio (assuming 0 risk-free rate)
                if metrics["std_return"] > 0:
                    metrics["sharpe_ratio"] = (
                        np.sqrt(252) * metrics["avg_return"] / metrics["std_return"]
                    )
                else:
                    metrics["sharpe_ratio"] = 0.0

                # Maximum drawdown
                cumulative = np.cumprod(1 + returns_array)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                metrics["max_drawdown"] = np.min(drawdown) * 100  # Percentage

                # Win rate
                if metrics["total_trades"] > 0:
                    metrics["win_rate"] = (
                        metrics["winning_trades"] / metrics["total_trades"]
                    ) * 100
                else:
                    metrics["win_rate"] = 0.0

        return metrics

    def finalize(self, timestamp: Optional[datetime] = None) -> None:
        """Finalize portfolio at end of backtest.

        Args:
            timestamp: Current simulation time (if None, uses datetime.now() for compatibility)
        """
        # Save final state
        self.save_state(timestamp or datetime.now())

        # Calculate P&L for all trades
        for i, trade in enumerate(self.trades):
            if i > 0 and trade["side"] in ["sell", "SELL"]:
                # Find corresponding buy trade and calculate P&L
                # This is simplified - real implementation would match specific lots
                prev_trades = [t for t in self.trades[:i] if t["asset_id"] == trade["asset_id"]]
                if prev_trades:
                    avg_buy_price = sum(
                        t["price"] * t["quantity"]
                        for t in prev_trades
                        if t["side"] in ["buy", "BUY"]
                    ) / sum(t["quantity"] for t in prev_trades if t["side"] in ["buy", "BUY"])
                    trade["pnl"] = (trade["price"] - avg_buy_price) * trade["quantity"] - trade[
                        "commission"
                    ]

        logger.info(f"Portfolio finalized. Final equity: ${self.equity:,.2f}")

    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.cash = self.initial_cash
        self.positions.clear()
        self.trades.clear()
        self.current_prices.clear()
        self.state_history.clear()
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_realized_pnl = 0.0


__all__ = ["SimplePortfolio"]
