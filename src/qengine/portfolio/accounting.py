"""Portfolio accounting and P&L tracking for QEngine."""

from datetime import datetime
from typing import Any, Optional

import polars as pl

from qengine.core.event import FillEvent
from qengine.core.types import AssetId, Cash
from qengine.portfolio.portfolio import Portfolio


class PortfolioAccounting:
    """
    Handles portfolio accounting, P&L calculation, and performance tracking.

    This class integrates with the broker to track trades and calculate
    real-time portfolio metrics including realized/unrealized P&L,
    returns, and risk metrics.
    """

    def __init__(self, initial_cash: Cash = 100000.0, track_history: bool = True):
        """
        Initialize portfolio accounting.

        Args:
            initial_cash: Starting cash balance
            track_history: Whether to keep detailed history
        """
        self.portfolio = Portfolio(initial_cash)
        self.track_history = track_history

        # Track all fills
        self.fills: list[FillEvent] = []

        # Performance tracking
        self.high_water_mark = float(initial_cash)
        self.max_drawdown = 0.0
        self.daily_returns: list[float] = []
        self.timestamps: list[datetime] = []
        self.equity_curve: list[float] = []

        # Risk tracking
        self.max_leverage = 0.0
        self.max_concentration = 0.0

        # Note: equity_curve and timestamps will be populated as events are processed

    def process_fill(self, fill_event: FillEvent) -> None:
        """
        Process a fill event and update portfolio.

        Args:
            fill_event: Fill event from broker
        """
        # Record fill
        self.fills.append(fill_event)

        # Determine quantity change (positive for buy, negative for sell)
        quantity_change = fill_event.fill_quantity
        if fill_event.side.value == "sell":
            quantity_change = -quantity_change

        # Update portfolio position
        self.portfolio.update_position(
            asset_id=fill_event.asset_id,
            quantity_change=quantity_change,
            price=float(fill_event.fill_price),
            commission=fill_event.commission,
            slippage=fill_event.slippage,
        )

        # Update performance metrics
        self._update_metrics(fill_event.timestamp)

    def update_prices(self, prices: dict[AssetId, float], timestamp: datetime) -> None:
        """
        Update portfolio with new market prices.

        Args:
            prices: Dictionary of asset prices
            timestamp: Current timestamp
        """
        self.portfolio.update_prices(prices)
        self._update_metrics(timestamp)

    def _update_metrics(self, timestamp: datetime) -> None:
        """Update performance and risk metrics."""
        current_equity = self.portfolio.equity

        # Update high water mark and drawdown
        if current_equity > self.high_water_mark:
            self.high_water_mark = current_equity

        if self.high_water_mark > 0:
            current_drawdown = (self.high_water_mark - float(current_equity)) / float(
                self.high_water_mark
            )
            self.max_drawdown = max(self.max_drawdown, current_drawdown)

        # Track equity curve
        if self.track_history:
            self.timestamps.append(timestamp)
            self.equity_curve.append(current_equity)

            # Calculate daily return if we have previous data
            if len(self.equity_curve) > 1:
                prev_equity = self.equity_curve[-2]
                if prev_equity > 0:
                    daily_return = (current_equity - prev_equity) / prev_equity
                    self.daily_returns.append(daily_return)

        # Update risk metrics from current state
        state = self.portfolio.get_current_state(timestamp)
        self.max_leverage = max(self.max_leverage, state.leverage)
        self.max_concentration = max(self.max_concentration, state.concentration)

        # Save state if tracking history
        if self.track_history:
            self.portfolio.save_state(timestamp)

    def calculate_win_rate(self) -> float:
        """Calculate win rate from trades (percentage of winning trades)."""
        if not self.fills:
            return 0.0

        # Group trades by asset to calculate P&L
        winning_trades = 0
        total_trades = 0

        # Track positions to determine win/loss
        position_pnl: dict[str, list[dict[str, float]]] = {}

        for fill in self.fills:
            asset_id = fill.asset_id

            if fill.side.value == "buy":
                if asset_id not in position_pnl:
                    position_pnl[asset_id] = []
                position_pnl[asset_id].append(
                    {"quantity": fill.fill_quantity, "price": float(fill.fill_price)},
                )
            elif fill.side.value == "sell":
                if position_pnl.get(asset_id):
                    # Calculate P&L for this trade
                    buy_info = position_pnl[asset_id].pop(0)
                    pnl = (float(fill.fill_price) - buy_info["price"]) * min(
                        fill.fill_quantity,
                        buy_info["quantity"],
                    )

                    total_trades += 1
                    if pnl > 0:
                        winning_trades += 1

        return winning_trades / total_trades if total_trades > 0 else 0.0

    def calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        gross_profit = 0.0
        gross_loss = 0.0

        # Track positions to determine profits and losses
        position_costs: dict[str, list[dict[str, float]]] = {}

        for fill in self.fills:
            asset_id = fill.asset_id

            if fill.side.value == "buy":
                if asset_id not in position_costs:
                    position_costs[asset_id] = []
                position_costs[asset_id].append(
                    {"quantity": fill.fill_quantity, "price": float(fill.fill_price)},
                )
            elif fill.side.value == "sell":
                if position_costs.get(asset_id):
                    # Calculate P&L for this trade
                    buy_info = position_costs[asset_id].pop(0)
                    pnl = (float(fill.fill_price) - buy_info["price"]) * min(
                        fill.fill_quantity,
                        buy_info["quantity"],
                    )

                    if pnl > 0:
                        gross_profit += pnl
                    else:
                        gross_loss += abs(pnl)

        return (
            gross_profit / gross_loss
            if gross_loss > 0
            else float("inf")
            if gross_profit > 0
            else 0.0
        )

    def calculate_avg_commission(self) -> float:
        """Calculate average commission per trade."""
        if not self.fills:
            return 0.0
        return self.portfolio.total_commission / len(self.fills)

    def calculate_avg_slippage(self) -> float:
        """Calculate average slippage per trade."""
        if not self.fills:
            return 0.0
        return self.portfolio.total_slippage / len(self.fills)

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics = {
            "total_return": self.portfolio.returns,
            "total_pnl": self.portfolio.total_realized_pnl + self.portfolio.unrealized_pnl,
            "realized_pnl": self.portfolio.total_realized_pnl,
            "unrealized_pnl": self.portfolio.unrealized_pnl,
            "max_drawdown": self.max_drawdown,
            "current_equity": self.portfolio.equity,
            "current_cash": self.portfolio.cash,
            "total_commission": self.portfolio.total_commission,
            "total_slippage": self.portfolio.total_slippage,
            "num_trades": len(self.fills),
            "max_leverage": self.max_leverage,
            "max_concentration": self.max_concentration,
        }

        # Add Sharpe ratio if we have enough data
        if len(self.daily_returns) > 1:
            import numpy as np

            returns = np.array(self.daily_returns)
            if returns.std() > 0:
                # Annualized Sharpe (assuming 252 trading days)
                metrics["sharpe_ratio"] = (returns.mean() / returns.std()) * np.sqrt(252)
            else:
                metrics["sharpe_ratio"] = 0.0

        # Add trading performance metrics
        metrics["win_rate"] = self.calculate_win_rate()
        metrics["profit_factor"] = self.calculate_profit_factor()
        metrics["avg_commission_per_trade"] = self.calculate_avg_commission()
        metrics["avg_slippage_per_trade"] = self.calculate_avg_slippage()

        return metrics

    def get_trades_df(self) -> Optional[pl.DataFrame]:
        """Get all trades as a Polars DataFrame."""
        if not self.fills:
            return None

        trades_data = []
        for fill in self.fills:
            trades_data.append(
                {
                    "timestamp": fill.timestamp,
                    "order_id": fill.order_id,
                    "trade_id": fill.trade_id,
                    "asset_id": fill.asset_id,
                    "side": fill.side.value,
                    "quantity": fill.fill_quantity,
                    "price": fill.fill_price,
                    "commission": fill.commission,
                    "slippage": fill.slippage,
                    "total_cost": fill.total_cost,
                },
            )

        return pl.DataFrame(trades_data)

    def get_equity_curve_df(self) -> Optional[pl.DataFrame]:
        """Get equity curve as a Polars DataFrame."""
        if not self.timestamps:
            return None

        return pl.DataFrame(
            {
                "timestamp": self.timestamps,
                "equity": self.equity_curve,
                "returns": [0.0, *self.daily_returns],  # Pad with 0 for first day
            },
        )

    def get_positions_df(self) -> Optional[pl.DataFrame]:
        """Get current positions as a Polars DataFrame."""
        if not self.portfolio.positions:
            return None

        positions_data = []
        for position in self.portfolio.positions.values():
            # Get realized P&L from portfolio (tracked at portfolio level, not position level)
            realized_pnl = self.portfolio.asset_realized_pnl.get(position.asset_id, 0.0)
            total_pnl = position.unrealized_pnl + realized_pnl

            positions_data.append(
                {
                    "asset_id": position.asset_id,
                    "quantity": position.quantity,
                    "cost_basis": position.cost_basis,
                    "last_price": position.last_price,
                    "market_value": position.market_value,
                    "unrealized_pnl": position.unrealized_pnl,
                    "realized_pnl": realized_pnl,
                    "total_pnl": total_pnl,
                },
            )

        return pl.DataFrame(positions_data)

    def get_summary(self) -> dict[str, Any]:
        """Get portfolio summary."""
        summary = self.portfolio.get_position_summary()
        summary.update(self.get_performance_metrics())
        return summary

    def reset(self) -> None:
        """Reset portfolio to initial state."""
        initial_cash = self.portfolio.initial_cash
        self.portfolio = Portfolio(initial_cash)
        self.fills.clear()
        self.high_water_mark = float(initial_cash)
        self.max_drawdown = 0.0
        self.daily_returns.clear()
        self.timestamps.clear()
        self.equity_curve.clear()
        self.max_leverage = 0.0
        self.max_concentration = 0.0

        if self.track_history:
            # Note: Initial timestamp will be set when first event is processed
            self.equity_curve.append(initial_cash)
