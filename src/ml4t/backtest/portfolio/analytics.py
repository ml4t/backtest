"""Portfolio analytics and performance tracking for ml4t.backtest.

This module provides:
- PerformanceAnalyzer: Performance metrics and risk analytics
- TradeJournal: Trade history and persistence
"""

from datetime import datetime
from typing import Any, Optional

import polars as pl

from ml4t.backtest.core.event import FillEvent
from ml4t.backtest.core.types import AssetId


class PerformanceAnalyzer:
    """Performance metrics and risk analytics.

    Tracks real-time metrics and calculates performance statistics
    based on a PositionTracker instance.

    This class is designed to be:
    - Lightweight: Can be disabled for HFT scenarios
    - Extensible: Can be subclassed to add custom metrics
    - Independent: Can be unit tested with mock trackers
    """

    def __init__(self, tracker: Any):
        """Initialize analyzer with position tracker.

        Args:
            tracker: PositionTracker to analyze
        """
        self.tracker = tracker

        # Real-time metric tracking
        self.high_water_mark = tracker.initial_cash
        self.max_drawdown = 0.0
        self.daily_returns: list[float] = []
        self.timestamps: list[datetime] = []
        self.equity_curve: list[float] = []

        # Risk metrics
        self.max_leverage = 0.0
        self.max_concentration = 0.0

    def update(self, timestamp: datetime) -> None:
        """Update metrics after position/price change.

        Called by Portfolio facade after every fill or market event.

        Args:
            timestamp: Current timestamp
        """
        current_equity = self.tracker.equity

        # Update high water mark and drawdown
        if current_equity > self.high_water_mark:
            self.high_water_mark = current_equity

        if self.high_water_mark > 0:
            current_drawdown = (self.high_water_mark - current_equity) / self.high_water_mark
            self.max_drawdown = max(self.max_drawdown, current_drawdown)

        # Track equity curve and timestamps
        self.timestamps.append(timestamp)
        self.equity_curve.append(current_equity)

        # Calculate return if we have previous data
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2]
            if prev_equity > 0:
                daily_return = (current_equity - prev_equity) / prev_equity
                self.daily_returns.append(daily_return)

        # Update risk metrics
        position_values = [p.market_value for p in self.tracker.positions.values()]
        if position_values and current_equity > 0:
            max_position_value = max(abs(v) for v in position_values)
            total_position_value = sum(abs(v) for v in position_values)
            concentration = max_position_value / current_equity
            leverage = total_position_value / current_equity
            self.max_concentration = max(self.max_concentration, concentration)
            self.max_leverage = max(self.max_leverage, leverage)

    def calculate_sharpe_ratio(self) -> float | None:
        """Calculate Sharpe ratio (annualized).

        Returns:
            Annualized Sharpe ratio, or None if insufficient data
        """
        if len(self.daily_returns) < 2:
            return None

        import numpy as np

        returns = np.array(self.daily_returns)
        if returns.std() > 0:
            return (returns.mean() / returns.std()) * np.sqrt(252)
        return 0.0

    def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics.

        Returns:
            Dictionary of performance metrics including:
            - total_return, total_pnl, realized_pnl, unrealized_pnl
            - max_drawdown, current_equity, current_cash
            - total_commission, total_slippage
            - max_leverage, max_concentration
            - sharpe_ratio (if sufficient data)
        """
        metrics = {
            "total_return": self.tracker.returns,
            "total_pnl": self.tracker.total_realized_pnl + self.tracker.unrealized_pnl,
            "realized_pnl": self.tracker.total_realized_pnl,
            "unrealized_pnl": self.tracker.unrealized_pnl,
            "max_drawdown": self.max_drawdown,
            "current_equity": self.tracker.equity,
            "current_cash": self.tracker.cash,
            "total_commission": self.tracker.total_commission,
            "total_slippage": self.tracker.total_slippage,
            "max_leverage": self.max_leverage,
            "max_concentration": self.max_concentration,
        }

        # Add Sharpe ratio if available
        sharpe = self.calculate_sharpe_ratio()
        if sharpe is not None:
            metrics["sharpe_ratio"] = sharpe

        return metrics

    def get_equity_curve(self) -> pl.DataFrame:
        """Get equity curve as DataFrame.

        Returns:
            DataFrame with columns: timestamp, equity, returns
        """
        if not self.timestamps:
            return pl.DataFrame()

        return pl.DataFrame({
            "timestamp": self.timestamps,
            "equity": self.equity_curve,
            "returns": [0.0, *self.daily_returns],  # Pad with 0 for first timestamp
        })

    def reset(self) -> None:
        """Reset analyzer state."""
        self.high_water_mark = self.tracker.initial_cash
        self.max_drawdown = 0.0
        self.daily_returns.clear()
        self.timestamps.clear()
        self.equity_curve.clear()
        self.max_leverage = 0.0
        self.max_concentration = 0.0


class TradeJournal:
    """Trade tracking and history management.

    Records fill events and provides trade analysis capabilities
    including win rate and profit factor calculations.
    """

    def __init__(self):
        """Initialize trade journal."""
        self.fills: list[FillEvent] = []

    def record_fill(self, fill_event: FillEvent) -> None:
        """Record a fill event.

        Args:
            fill_event: Fill event from broker
        """
        self.fills.append(fill_event)

    def get_trades(self) -> pl.DataFrame:
        """Get all trades as DataFrame.

        Returns:
            DataFrame with trade details including timestamp, order_id,
            trade_id, asset_id, side, quantity, price, commission, slippage
        """
        if not self.fills:
            return pl.DataFrame()

        trades_data = []
        for fill in self.fills:
            trades_data.append({
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
            })

        return pl.DataFrame(trades_data)

    def calculate_win_rate(self) -> float:
        """Calculate win rate via lot matching.

        Matches buy and sell fills using FIFO to determine winning vs.
        losing trades.

        Returns:
            Win rate as a fraction (0.0 to 1.0)
        """
        if not self.fills:
            return 0.0

        # Lot matching algorithm (same as PortfolioAccounting)
        winning_trades = 0
        total_trades = 0
        position_lots: dict[str, list[dict[str, float]]] = {}

        for fill in self.fills:
            asset_id = fill.asset_id

            if fill.side.value == "buy":
                if asset_id not in position_lots:
                    position_lots[asset_id] = []
                position_lots[asset_id].append({
                    "quantity": fill.fill_quantity,
                    "price": float(fill.fill_price),
                })
            elif fill.side.value == "sell":
                if position_lots.get(asset_id):
                    buy_lot = position_lots[asset_id].pop(0)
                    pnl = (float(fill.fill_price) - buy_lot["price"]) * min(
                        fill.fill_quantity,
                        buy_lot["quantity"],
                    )
                    total_trades += 1
                    if pnl > 0:
                        winning_trades += 1

        return winning_trades / total_trades if total_trades > 0 else 0.0

    def calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss).

        Returns:
            Profit factor (>1 is profitable, <1 is unprofitable).
            Returns float('inf') if gross_loss is 0 but gross_profit > 0.
            Returns 0.0 if both are 0.
        """
        gross_profit = 0.0
        gross_loss = 0.0

        # Track positions to determine profits and losses
        position_costs: dict[str, list[dict[str, float]]] = {}

        for fill in self.fills:
            asset_id = fill.asset_id

            if fill.side.value == "buy":
                if asset_id not in position_costs:
                    position_costs[asset_id] = []
                position_costs[asset_id].append({
                    "quantity": fill.fill_quantity,
                    "price": float(fill.fill_price),
                })
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
        """Calculate average commission per trade.

        Returns:
            Average commission, or 0.0 if no trades
        """
        if not self.fills:
            return 0.0
        total_commission = sum(fill.commission for fill in self.fills)
        return total_commission / len(self.fills)

    def calculate_avg_slippage(self) -> float:
        """Calculate average slippage per trade.

        Returns:
            Average slippage, or 0.0 if no trades
        """
        if not self.fills:
            return 0.0
        total_slippage = sum(fill.slippage for fill in self.fills)
        return total_slippage / len(self.fills)

    def reset(self) -> None:
        """Reset journal."""
        self.fills.clear()
