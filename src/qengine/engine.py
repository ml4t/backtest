"""Main backtest engine that orchestrates the simulation."""

import logging
from datetime import datetime
from typing import Any

import polars as pl

from qengine.core.clock import Clock
from qengine.core.event import EventType
from qengine.data.feed import DataFeed
from qengine.execution.broker import Broker
from qengine.portfolio.portfolio import Portfolio
from qengine.reporting.reporter import Reporter
from qengine.strategy.base import Strategy

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Main backtesting engine that coordinates all components.

    The engine follows an event-driven architecture where:
    1. Data feeds generate market events
    2. Strategies consume events and generate signals/orders
    3. Broker executes orders and generates fills
    4. Portfolio tracks positions and P&L
    5. Reporter captures results

    Example:
        >>> from ml4t_backtest import BacktestEngine
        >>> from qengine.data import ParquetDataFeed
        >>> from qengine.strategy import BuyAndHoldStrategy
        >>>
        >>> engine = BacktestEngine(
        ...     data_feed=ParquetDataFeed("data.parquet"),
        ...     strategy=BuyAndHoldStrategy(),
        ...     initial_capital=100000
        ... )
        >>> results = engine.run()
    """

    def __init__(
        self,
        data_feed: DataFeed,
        strategy: Strategy,
        broker: Broker | None = None,
        portfolio: Portfolio | None = None,
        reporter: Reporter | None = None,
        initial_capital: float = 100000.0,
        currency: str = "USD",
        use_priority_queue: bool = True,
        corporate_actions: list | None = None,
    ):
        """Initialize the backtest engine.

        Args:
            data_feed: Source of market data events
            strategy: Trading strategy to execute
            broker: Order execution broker (default: SimulationBroker)
            portfolio: Portfolio tracker (default: SimplePortfolio)
            reporter: Results reporter (default: InMemoryReporter)
            initial_capital: Starting capital
            currency: Base currency for the portfolio
            use_priority_queue: Use priority queue for event ordering
            corporate_actions: List of corporate actions to process during backtest
        """
        self.data_feed = data_feed
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.currency = currency

        # Event distribution now handled by Clock (subscribe/publish/dispatch)

        # Create clock for time management
        self.clock = Clock()

        # Initialize broker if not provided
        if broker is None:
            from qengine.execution.broker import SimulationBroker

            self.broker = SimulationBroker()
        else:
            self.broker = broker

        # Initialize portfolio if not provided
        if portfolio is None:
            from qengine.portfolio.simple import SimplePortfolio

            self.portfolio = SimplePortfolio(initial_capital=initial_capital, currency=currency)
        else:
            self.portfolio = portfolio

        # Initialize reporter if not provided
        if reporter is None:
            from qengine.reporting.reporter import InMemoryReporter

            self.reporter = InMemoryReporter()
        else:
            self.reporter = reporter

        # Initialize corporate action processor
        from qengine.execution.corporate_actions import CorporateActionProcessor

        self.corporate_action_processor = CorporateActionProcessor()
        if corporate_actions:
            for action in corporate_actions:
                self.corporate_action_processor.add_action(action)

        # Wire up event handlers
        self._setup_event_handlers()

        # Statistics
        self.events_processed = 0
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None

    def _setup_event_handlers(self) -> None:
        """Connect components via Clock event subscriptions."""
        # Strategy subscribes to market and fill events
        # Note: Using on_event allows strategies to route events internally
        self.clock.subscribe(EventType.MARKET, self.strategy.on_event)
        self.clock.subscribe(EventType.FILL, self.strategy.on_event)

        # Broker subscribes to order events AND market events (for fill checking)
        self.clock.subscribe(EventType.ORDER, self.broker.on_order_event)
        self.clock.subscribe(EventType.MARKET, self.broker.on_market_event)

        # Portfolio subscribes to fill events
        self.clock.subscribe(EventType.FILL, self.portfolio.on_fill_event)

        # Reporter subscribes to all events for logging
        for event_type in EventType:
            self.clock.subscribe(event_type, self.reporter.on_event)

    def run(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        max_events: int | None = None,
    ) -> dict[str, Any]:
        """Run the backtest simulation.

        Args:
            start_date: Start date for backtest (None = use data start)
            end_date: End date for backtest (None = use data end)
            max_events: Maximum events to process (for debugging)

        Returns:
            Dictionary containing backtest results including:
            - trades: DataFrame of executed trades
            - positions: DataFrame of position history
            - returns: Series of strategy returns
            - metrics: Performance metrics dict
            - events_processed: Number of events processed
        """
        logger.info("Starting backtest engine")
        self.start_time = datetime.now()  # Wall clock time for performance measurement

        # Initialize components
        self.strategy.on_start(self.portfolio, self.clock)
        self.broker.initialize(self.portfolio, self.clock)
        self.portfolio.initialize()
        self.reporter.on_start()

        # Add data feed to clock for event-driven processing
        self.clock.add_data_feed(self.data_feed)

        # Initialize clock with data feed's time range
        if hasattr(self.data_feed, "get_time_range"):
            data_start, data_end = self.data_feed.get_time_range()
            self.clock.advance_to(start_date or data_start)

        # Main event loop
        self.events_processed = 0

        while not self.data_feed.is_exhausted:
            # Check max events limit
            if max_events and self.events_processed >= max_events:
                logger.info(f"Reached max events limit: {max_events}")
                break

            # Get next market event from data feed
            market_event = self.data_feed.get_next_event()
            if market_event is None:
                break

            # Update clock
            self.clock.advance_to(market_event.timestamp)

            # Publish market event FIRST (this allows pending orders to be filled)
            self.clock.publish(market_event)

            # Dispatch event to all subscribers
            self.clock.dispatch_event(market_event)

            # TODO: TASK-1.4 will rewrite this loop to be Clock-driven
            # For now, we still iterate over data_feed but use Clock for distribution
            self.events_processed += 1

            # Process corporate actions AFTER fills (at start of each trading day)
            # Check if we've moved to a new date
            if hasattr(self, "_last_processed_date"):
                current_date = market_event.timestamp.date()
                if current_date != self._last_processed_date:
                    # Process any pending corporate actions
                    # Convert Position objects to quantities for corporate action processor
                    current_positions = {
                        asset_id: position.quantity
                        for asset_id, position in self.portfolio.positions.items()
                    }
                    open_orders = self.broker.get_open_orders()  # Get all open orders
                    cash = self.portfolio.cash

                    # Process corporate actions
                    updated_positions, updated_orders, updated_cash, notifications = (
                        self.corporate_action_processor.process_actions(
                            current_date, current_positions, open_orders, cash
                        )
                    )

                    # Update portfolio and broker with adjustments
                    if notifications:
                        for notification in notifications:
                            logger.info(f"Corporate Action: {notification}")

                    # Apply position adjustments to Portfolio objects
                    for asset_id, new_quantity in updated_positions.items():
                        if asset_id in self.portfolio.positions:
                            # Update existing position quantity
                            self.portfolio.positions[asset_id].quantity = new_quantity
                        elif new_quantity != 0:
                            # Create new position if needed
                            from qengine.portfolio.portfolio import Position

                            self.portfolio.positions[asset_id] = Position(
                                asset_id=asset_id, quantity=new_quantity
                            )

                    # Remove positions with zero quantity
                    assets_to_remove = [
                        asset_id
                        for asset_id, position in self.portfolio.positions.items()
                        if position.quantity == 0
                    ]
                    for asset_id in assets_to_remove:
                        del self.portfolio.positions[asset_id]

                    # Update cash
                    self.portfolio.cash = updated_cash
                    # Note: Orders are updated in place, so broker already has updated orders

                    self._last_processed_date = current_date
            else:
                self._last_processed_date = market_event.timestamp.date()

            # Update portfolio valuations
            self.portfolio.update_market_value(market_event)

            # Log progress periodically
            if self.events_processed % 10000 == 0:
                logger.info(f"Processed {self.events_processed:,} events")

        # Finalize
        self.strategy.on_end()
        self.broker.finalize()
        self.portfolio.finalize()
        self.reporter.on_end()

        self.end_time = datetime.now()  # Wall clock time for performance measurement
        duration = (self.end_time - self.start_time).total_seconds()

        logger.info(
            f"Backtest complete: {self.events_processed:,} events in {duration:.2f}s "
            f"({self.events_processed / duration:.0f} events/sec)",
        )

        # Compile results
        results = self._compile_results()
        return results

    def _compile_results(self) -> dict[str, Any]:
        """Compile backtest results from all components.

        Returns:
            Dictionary with comprehensive backtest results
        """
        # Get data from components
        trades = self.broker.get_trades()
        positions = self.portfolio.get_positions()
        returns = self.portfolio.get_returns()
        metrics = self.portfolio.calculate_metrics()

        # Add engine statistics
        duration = (self.end_time - self.start_time).total_seconds() if self.end_time else 0

        results = {
            "trades": trades,
            "positions": positions,
            "returns": returns,
            "metrics": metrics,
            "events_processed": self.events_processed,
            "duration_seconds": duration,
            "events_per_second": self.events_processed / duration if duration > 0 else 0,
            "initial_capital": self.initial_capital,
            "final_value": self.portfolio.get_total_value(),
            "total_return": (self.portfolio.get_total_value() / self.initial_capital - 1) * 100,
        }

        # Add reporter data if available
        if hasattr(self.reporter, "get_report"):
            results["report"] = self.reporter.get_report()

        return results

    def reset(self) -> None:
        """Reset the engine for another run."""
        logger.info("Resetting backtest engine")

        # Clear clock event queue and reset subscribers
        self.clock.reset()

        # Reset components
        self.data_feed.reset()
        self.strategy.reset()
        self.broker.reset()
        self.portfolio.reset()
        self.reporter.reset()

        # Reset statistics
        self.events_processed = 0
        self.start_time = None
        self.end_time = None

        # Re-setup event handlers
        self._setup_event_handlers()


class BacktestResults:
    """Container for backtest results with analysis methods."""

    def __init__(self, results: dict[str, Any]):
        """Initialize with results dictionary from BacktestEngine.

        Args:
            results: Results dictionary from engine.run()
        """
        self.results = results
        self.trades = results.get("trades", pl.DataFrame())
        self.positions = results.get("positions", pl.DataFrame())
        self.returns = results.get("returns", pl.Series())
        self.metrics = results.get("metrics", {})

    @property
    def total_return(self) -> float:
        """Total return percentage."""
        return self.results.get("total_return", 0.0)

    @property
    def sharpe_ratio(self) -> float:
        """Sharpe ratio of returns."""
        return self.metrics.get("sharpe_ratio", 0.0)

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown percentage."""
        return self.metrics.get("max_drawdown", 0.0)

    @property
    def win_rate(self) -> float:
        """Percentage of winning trades."""
        if self.trades.is_empty():
            return 0.0
        winning = self.trades.filter(pl.col("pnl") > 0)
        return len(winning) / len(self.trades) * 100

    def summary(self) -> str:
        """Generate a text summary of results.

        Returns:
            Formatted summary string
        """
        return f"""
Backtest Results Summary
========================
Total Return: {self.total_return:.2f}%
Sharpe Ratio: {self.sharpe_ratio:.2f}
Max Drawdown: {self.max_drawdown:.2f}%
Win Rate: {self.win_rate:.2f}%
Total Trades: {len(self.trades):,}
Events Processed: {self.results.get("events_processed", 0):,}
Duration: {self.results.get("duration_seconds", 0):.2f}s
        """.strip()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary of results
        """
        return self.results

    def save(self, path: str) -> None:
        """Save results to file.

        Args:
            path: Output file path (supports .parquet, .json, .html)
        """
        if path.endswith(".parquet"):
            # Save DataFrames to parquet
            self.trades.write_parquet(path.replace(".parquet", "_trades.parquet"))
            self.positions.write_parquet(path.replace(".parquet", "_positions.parquet"))
        elif path.endswith(".json"):
            # Save as JSON
            import json

            with open(path, "w") as f:
                # Convert non-serializable objects
                data = {
                    k: v if not isinstance(v, (pl.DataFrame, pl.Series)) else None
                    for k, v in self.results.items()
                }
                json.dump(data, f, indent=2, default=str)
        elif path.endswith(".html"):
            # Generate HTML report
            from qengine.reporting.html import generate_html_report

            html = generate_html_report(self)
            with open(path, "w") as f:
                f.write(html)
        else:
            raise ValueError(f"Unsupported file format: {path}")


__all__ = [
    "BacktestEngine",
    "BacktestResults",
]
