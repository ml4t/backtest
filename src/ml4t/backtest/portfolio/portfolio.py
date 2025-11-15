"""Portfolio facade combining position tracking, analytics, and trade history.

This module provides the main Portfolio class - a facade that orchestrates:
- PositionTracker (core position/cash management)
- PerformanceAnalyzer (metrics and analytics, optional)
- TradeJournal (trade history and persistence)

The facade pattern provides:
- Simple API for beginners (portfolio.on_fill_event())
- Performance opt-out for HFT (track_analytics=False)
- Easy extension for researchers (custom analyzer/journal classes)
"""

import logging
from datetime import datetime
from typing import Any, Optional, Type

from ml4t.backtest.core.event import FillEvent, MarketEvent
from ml4t.backtest.core.precision import PrecisionManager
from ml4t.backtest.core.types import AssetId, Cash, Quantity
from ml4t.backtest.portfolio.core import PositionTracker
from ml4t.backtest.portfolio.state import Position, PortfolioState

logger = logging.getLogger(__name__)


class Portfolio:
    """
    Unified portfolio management facade - simple API with modular internals.

    This is the main entry point for portfolio management in ml4t.backtest.
    It combines three independent components:

    1. PositionTracker - core position and cash tracking
    2. PerformanceAnalyzer - performance metrics (optional, can disable for HFT)
    3. TradeJournal - trade history and persistence

    Example usage:

        # Simple API (for beginners)
        portfolio = Portfolio(initial_cash=100000)
        portfolio.on_fill_event(fill)
        metrics = portfolio.get_performance_metrics()

        # Performance opt-out (for HFT)
        portfolio = Portfolio(track_analytics=False)  # Zero overhead

        # Easy extension (for researchers)
        class MyAnalyzer(PerformanceAnalyzer):
            def calculate_sortino_ratio(self): pass

        portfolio = Portfolio(analyzer_class=MyAnalyzer)
    """

    def __init__(
        self,
        initial_cash: Cash = 100000.0,
        currency: str = "USD",
        track_analytics: bool = True,
        precision_manager: Optional[PrecisionManager] = None,
        analyzer_class: Optional[Type] = None,
        journal_class: Optional[Type] = None,
    ):
        """Initialize portfolio with modular components.

        Args:
            initial_cash: Starting cash balance
            currency: Base currency (default: USD)
            track_analytics: Whether to enable analytics (disable for HFT, default: True)
            precision_manager: PrecisionManager for cash rounding
            analyzer_class: Custom analyzer class (default: PerformanceAnalyzer)
            journal_class: Custom journal class (default: TradeJournal)
        """
        # Core tracking (always present)
        self._tracker = PositionTracker(initial_cash, precision_manager)

        # Optional analytics (can disable for performance)
        if track_analytics:
            from ml4t.backtest.portfolio.analytics import PerformanceAnalyzer

            AnalyzerClass = analyzer_class or PerformanceAnalyzer
            self._analyzer: Optional[Any] = AnalyzerClass(self._tracker)
        else:
            self._analyzer = None

        # Trade journal
        from ml4t.backtest.portfolio.analytics import TradeJournal

        JournalClass = journal_class or TradeJournal
        self._journal = JournalClass()

        # Portfolio-level attributes
        self.currency = currency
        self.initial_cash = initial_cash
        self.current_prices: dict[AssetId, float] = {}
        self.precision_manager = precision_manager

        # State history (for backward compatibility)
        self.state_history: list[PortfolioState] = []

    # ===== Event Handlers =====
    def on_market_event(self, event: "MarketEvent") -> None:
        """Handle market event to update position prices.

        This ensures unrealized PnL reflects current market prices.

        Args:
            event: MarketEvent with current price
        """
        if event.close is not None:
            # Update position prices with current market price
            self._tracker.update_prices({event.asset_id: float(event.close)})

    def on_fill_event(self, event: FillEvent) -> None:
        """Handle fill event from broker.

        This is the main entry point for updating portfolio state.
        Delegates to all three components:
        1. TradeJournal - records the fill
        2. PositionTracker - updates position and cash
        3. PerformanceAnalyzer - updates metrics (if enabled)

        Args:
            event: FillEvent from broker
        """
        # Record in journal
        self._journal.record_fill(event)

        # Update position (convert Decimal to float for tracker)
        quantity_change = (
            float(event.fill_quantity) if event.side.value in ["buy", "BUY"] else -float(event.fill_quantity)
        )
        self._tracker.update_position(
            asset_id=event.asset_id,
            quantity_change=quantity_change,
            price=float(event.fill_price),
            commission=event.commission,
            slippage=event.slippage,
        )

        # Update analytics (if enabled)
        if self._analyzer:
            self._analyzer.update(event.timestamp)

        logger.info(
            f"Fill: {event.side.value.upper()} {event.fill_quantity} {event.asset_id} "
            f"@ ${float(event.fill_price):.2f}"
        )

    # ===== Delegate to PositionTracker =====
    @property
    def cash(self) -> float:
        """Current cash balance."""
        return self._tracker.cash

    @cash.setter
    def cash(self, value: float) -> None:
        """Set cash balance (backward compatibility for tests).

        Args:
            value: New cash value
        """
        self._tracker.cash = value

    @property
    def equity(self) -> float:
        """Total equity (cash + position market values)."""
        return self._tracker.equity

    @property
    def returns(self) -> float:
        """Simple returns from initial capital."""
        return self._tracker.returns

    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P&L across all positions."""
        return self._tracker.unrealized_pnl

    @property
    def total_realized_pnl(self) -> float:
        """Total realized P&L."""
        return self._tracker.total_realized_pnl

    @property
    def total_commission(self) -> float:
        """Total commission paid."""
        return self._tracker.total_commission

    @property
    def total_slippage(self) -> float:
        """Total slippage cost."""
        return self._tracker.total_slippage

    @property
    def positions(self) -> dict[AssetId, Position]:
        """Current positions (direct access for advanced users)."""
        return self._tracker.positions

    def get_position(self, asset_id: AssetId) -> Position | None:
        """Get position for a specific asset.

        Args:
            asset_id: Asset identifier

        Returns:
            Position object or None if no position
        """
        return self._tracker.get_position(asset_id)

    def get_all_positions(self) -> dict[AssetId, Quantity]:
        """Get all current positions as dict of quantities.

        Returns:
            Dictionary mapping asset_id to quantity
        """
        return {asset_id: pos.quantity for asset_id, pos in self._tracker.positions.items()}

    def update_prices(self, prices: dict[AssetId, float]) -> None:
        """Update all positions with new market prices.

        Args:
            prices: Dictionary mapping asset_id to price
        """
        self._tracker.update_prices(prices)
        self.current_prices.update(prices)

    def update_position(
        self,
        asset_id: AssetId,
        quantity_change: float,
        price: float,
        commission: float = 0.0,
        slippage: float = 0.0,
    ) -> None:
        """Update position (backward compatibility).

        This method is kept for backward compatibility with PortfolioAccounting.
        New code should use on_fill_event() instead.

        Args:
            asset_id: Asset identifier
            quantity_change: Change in quantity (positive for buy, negative for sell)
            price: Execution price
            commission: Commission paid
            slippage: Slippage cost
        """
        self._tracker.update_position(
            asset_id=asset_id,
            quantity_change=quantity_change,
            price=price,
            commission=commission,
            slippage=slippage,
        )

    # ===== Delegate to PerformanceAnalyzer =====
    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics.

        Returns:
            Dictionary with metrics like Sharpe ratio, max drawdown, etc.

        Raises:
            ValueError: If analytics is disabled (track_analytics=False)
        """
        if not self._analyzer:
            raise ValueError("Analytics disabled. Set track_analytics=True to enable.")
        return self._analyzer.get_metrics()

    def calculate_sharpe_ratio(self) -> float | None:
        """Calculate Sharpe ratio.

        Returns:
            Sharpe ratio or None if analytics disabled or insufficient data
        """
        if not self._analyzer:
            return None
        return self._analyzer.calculate_sharpe_ratio()

    # ===== Delegate to TradeJournal =====
    def get_trades(self) -> Any:  # Returns pl.DataFrame but avoid import for type hint
        """Get trade history as Polars DataFrame.

        Returns:
            Polars DataFrame with columns: timestamp, asset_id, side, quantity, price, etc.
        """
        return self._journal.get_trades()

    # ===== For Advanced Users =====
    @property
    def tracker(self) -> PositionTracker:
        """Access position tracker directly (advanced users).

        Returns:
            PositionTracker instance
        """
        return self._tracker

    @property
    def analyzer(self) -> Optional[Any]:
        """Access performance analyzer directly (advanced users).

        Returns:
            PerformanceAnalyzer instance or None if analytics disabled
        """
        return self._analyzer

    @property
    def journal(self) -> Any:
        """Access trade journal directly (advanced users).

        Returns:
            TradeJournal instance
        """
        return self._journal

    # ===== Backward Compatibility =====
    def get_current_state(self, timestamp: datetime) -> PortfolioState:
        """Get current portfolio state (backward compatibility).

        Args:
            timestamp: Current timestamp

        Returns:
            PortfolioState snapshot
        """
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
        """Save current state to history (backward compatibility).

        Args:
            timestamp: Current timestamp
        """
        self.state_history.append(self.get_current_state(timestamp))

    def get_position_summary(self) -> dict[str, Any]:
        """Get summary of all positions (backward compatibility).

        Returns:
            Dictionary with cash, equity, positions count, P&L, etc.
        """
        return self._tracker.get_summary()

    def reset(self) -> None:
        """Reset portfolio to initial state (backward compatibility).

        This method is used by Broker during reset operations.
        """
        self._tracker.reset()
        if self._analyzer:
            # Reset analyzer state
            self._analyzer.high_water_mark = self._tracker.initial_cash
            self._analyzer.max_drawdown = 0.0
            self._analyzer.daily_returns.clear()
            self._analyzer.timestamps.clear()
            self._analyzer.equity_curve.clear()
            self._analyzer.max_leverage = 0.0
            self._analyzer.max_concentration = 0.0
        self._journal.reset()
        self.state_history.clear()


__all__ = ["Portfolio"]
