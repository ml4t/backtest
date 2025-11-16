"""Main backtest engine that orchestrates the simulation."""

import logging
from datetime import datetime
from typing import Any

import polars as pl

from ml4t.backtest.core.clock import Clock
from ml4t.backtest.core.context import ContextCache
from ml4t.backtest.core.event import EventType, MarketEvent
from ml4t.backtest.data.feed import DataFeed
from ml4t.backtest.execution.broker import Broker
from ml4t.backtest.portfolio.portfolio import Portfolio
from ml4t.backtest.reporting.reporter import Reporter
from ml4t.backtest.strategy.base import Strategy

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
        >>> from ml4t.backtest.data import ParquetDataFeed
        >>> from ml4t.backtest.strategy import BuyAndHoldStrategy
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
        context_data: dict[datetime, dict[str, Any]] | None = None,
    ):
        """Initialize the backtest engine.

        Args:
            data_feed: Source of market data events
            strategy: Trading strategy to execute
            broker: Order execution broker (default: SimulationBroker)
            portfolio: Portfolio tracker (default: Portfolio)
            reporter: Results reporter (default: InMemoryReporter)
            initial_capital: Starting capital
            currency: Base currency for the portfolio
            use_priority_queue: Use priority queue for event ordering
            corporate_actions: List of corporate actions to process during backtest
            context_data: Market-wide context data indexed by timestamp
                         Example: {datetime(2024,1,15): {'VIX': 18.5, 'SPY': 485.0}}
                         Strategies receive this as context parameter in on_market_event
        """
        self.data_feed = data_feed
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.currency = currency

        # Initialize context cache
        self.context_cache = ContextCache()
        self.context_data = context_data or {}

        # Event distribution now handled by Clock (subscribe/publish/dispatch)

        # Create clock for time management
        self.clock = Clock()

        # Initialize broker if not provided
        if broker is None:
            from ml4t.backtest.execution.broker import SimulationBroker

            self.broker = SimulationBroker()
        else:
            self.broker = broker

        # Initialize portfolio if not provided
        if portfolio is None:
            from ml4t.backtest.portfolio.portfolio import Portfolio

            self.portfolio = Portfolio(initial_cash=initial_capital, currency=currency)
        else:
            self.portfolio = portfolio

        # Initialize reporter if not provided
        if reporter is None:
            from ml4t.backtest.reporting.reporter import InMemoryReporter

            self.reporter = InMemoryReporter()
        else:
            self.reporter = reporter

        # Initialize corporate action processor
        from ml4t.backtest.execution.corporate_actions import CorporateActionProcessor

        self.corporate_action_processor = CorporateActionProcessor()
        if corporate_actions:
            for action in corporate_actions:
                self.corporate_action_processor.add_action(action)

        # Inject broker into strategy for helper methods
        self.strategy.broker = self.broker

        # Wire up event handlers
        self._setup_event_handlers()

        # Statistics
        self.events_processed = 0
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None

    def _setup_event_handlers(self) -> None:
        """Connect components via Clock event subscriptions."""
        # Strategy subscribes to fill events
        # Note: MARKET events are handled specially by Engine to pass context
        self.clock.subscribe(EventType.FILL, self.strategy.on_event)

        # Broker subscribes to order events AND market events (for fill checking)
        self.clock.subscribe(EventType.ORDER, self.broker.on_order_event)
        self.clock.subscribe(EventType.MARKET, self.broker.on_market_event)

        # Portfolio subscribes to fill and market events
        # Market events update position prices for accurate unrealized PnL
        self.clock.subscribe(EventType.MARKET, self.portfolio.on_market_event)
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
        # Portfolio initialization happens in __init__, no separate initialize() needed
        self.reporter.on_start()

        # Add data feed to clock for event-driven processing
        self.clock.add_data_feed(self.data_feed)

        # Initialize clock with data feed's time range
        if hasattr(self.data_feed, "get_time_range"):
            data_start, data_end = self.data_feed.get_time_range()
            self.clock.advance_to(start_date or data_start)

        # Main event loop - Clock-driven (Phase 1 redesign complete)
        # The Clock pulls events from all registered feeds (market data, signals, corporate actions)
        # and returns them in strict chronological order, ensuring point-in-time correctness
        self.events_processed = 0
        PROGRESS_LOG_INTERVAL = 10000

        while True:
            # Check max events limit
            if max_events and self.events_processed >= max_events:
                logger.info(f"Reached max events limit: {max_events}")
                break

            # Get next event from Clock's priority queue (across ALL feeds)
            event = self.clock.get_next_event()
            if event is None:
                # All feeds exhausted
                break

            # Dispatch event to all registered subscribers
            # Special handling for MARKET events: pass context to strategy
            # Other events go through normal Clock dispatch
            if event.event_type == EventType.MARKET:
                # Get or create context for this timestamp
                context_dict = self.context_data.get(event.timestamp, {})
                context = self.context_cache.get_or_create(event.timestamp, context_dict)

                # Call strategy with context (convert Context to dict for simpler API)
                self.strategy.on_market_event(event, context.data)

                # Also call on_event for backward compatibility with strategies
                # that don't override on_market_event
                self.strategy.on_event(event)

                # Still dispatch to other subscribers (broker, portfolio, reporter)
                self.clock.dispatch_event(event)
            else:
                # Normal dispatch for non-MARKET events
                # Subscribers handle events based on event_type:
                # - Strategy: receives FILL events
                # - Broker: receives ORDER and MARKET events
                # - Portfolio: receives FILL and MARKET events
                # - Reporter: receives ALL events
                self.clock.dispatch_event(event)

            # Process corporate actions AFTER market events are dispatched
            # This ensures positions are filled/updated before corporate actions are applied
            # (e.g., pending orders moved to open and filled, positions exist)
            if event.event_type == EventType.MARKET:
                # Extract date from event timestamp
                current_date = event.timestamp.date()

                # Check for pending corporate actions on this date
                pending_actions = self.corporate_action_processor.get_pending_actions(current_date)

                if pending_actions:
                    # Get current state from broker (after fills)
                    positions = self.broker.get_positions()  # Returns dict[AssetId, Quantity]
                    orders = self.broker.get_open_orders()
                    cash = self.broker.get_cash()

                    # Process all actions for this date
                    updated_positions, updated_orders, updated_cash, notifications = \
                        self.corporate_action_processor.process_actions(
                            current_date,
                            positions,
                            orders,
                            cash,
                        )

                    # Apply position adjustments to broker
                    for asset_id, new_quantity in updated_positions.items():
                        if asset_id in positions:
                            # Existing position - adjust quantity
                            self.broker.adjust_position_quantity(asset_id, new_quantity)
                        elif new_quantity > 0:
                            # New position created (e.g., spin-off, merger)
                            # This shouldn't happen for simple splits/dividends
                            logger.warning(
                                f"Corporate action created new position for {asset_id} "
                                f"with quantity {new_quantity}. Manual position creation not yet supported."
                            )

                    # Apply cash adjustment (dividends, merger proceeds, etc.)
                    cash_change = updated_cash - cash
                    if cash_change != 0:
                        self.broker.adjust_cash(cash_change)

                    # Orders are already adjusted in-place by the processor
                    # No need to apply them back

                    # Log notifications
                    for notification in notifications:
                        logger.info(f"Corporate action: {notification}")
                        if self.reporter and hasattr(self.reporter, 'add_note'):
                            self.reporter.add_note(event.timestamp, notification)

            self.events_processed += 1

            # Log progress periodically
            if self.events_processed % PROGRESS_LOG_INTERVAL == 0:
                logger.info(
                    f"Processed {self.events_processed:,} events at {event.timestamp}"
                )

        # Finalize
        self.strategy.on_end()
        self.broker.finalize()
        # Portfolio has no finalize() method - state is already complete
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
        positions = self.portfolio.get_all_positions()  # New API: dict[AssetId, Quantity]
        returns = self.portfolio.returns  # New API: property instead of method
        metrics = self.portfolio.get_performance_metrics()  # New API name

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
            "final_value": self.portfolio.equity,  # New API: property
            "total_return": (self.portfolio.equity / self.initial_capital - 1) * 100,
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
            from ml4t.backtest.reporting.html import generate_html_report

            html = generate_html_report(self)
            with open(path, "w") as f:
                f.write(html)
        else:
            raise ValueError(f"Unsupported file format: {path}")


__all__ = [
    "BacktestEngine",
    "BacktestResults",
]
