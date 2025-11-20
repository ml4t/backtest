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
from ml4t.backtest.risk.manager import RiskManager
from ml4t.backtest.strategy.base import Strategy

logger = logging.getLogger(__name__)


class NoOpReporter:
    """Minimal reporter that does nothing (used when reporting module not installed)."""

    def on_event(self, event) -> None:
        pass

    def get_results(self) -> dict:
        return {}


# Try to import full Reporter, fall back to no-op
try:
    from ml4t.backtest.reporting.reporter import Reporter, InMemoryReporter
except ImportError:
    Reporter = NoOpReporter  # type: ignore
    InMemoryReporter = NoOpReporter  # type: ignore


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
        risk_manager: RiskManager | None = None,
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
            risk_manager: Optional risk management system (default: None)
                          When provided, integrates position exit checking, order validation,
                          and fill recording for risk rule enforcement
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
        self.risk_manager = risk_manager
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

            self.broker = SimulationBroker(initial_cash=initial_capital)
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
            self.reporter = InMemoryReporter()
        else:
            self.reporter = reporter

        # Initialize corporate action processor
        from ml4t.backtest.execution.corporate_actions import CorporateActionProcessor

        self.corporate_action_processor = CorporateActionProcessor()
        if corporate_actions:
            for action in corporate_actions:
                self.corporate_action_processor.add_action(action)

        # Hook B: Wrap broker's submit_order if risk_manager is enabled
        # This intercepts all strategy order submissions for validation
        if self.risk_manager:
            self._wrap_broker_for_risk_validation()

        # Inject broker into strategy for helper methods
        self.strategy.broker = self.broker

        # Wire up event handlers
        self._setup_event_handlers()

        # Check strategy execution mode for batch processing
        self._strategy_mode = self.strategy.execution_mode
        logger.info(f"Strategy execution mode: {self._strategy_mode}")

        # Batch mode state
        self._event_batch: list[MarketEvent] = []
        self._current_batch_timestamp: datetime | None = None

        # Statistics
        self.events_processed = 0
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None

    def _wrap_broker_for_risk_validation(self) -> None:
        """Wrap broker's submit_order method to inject risk validation (Hook B).

        This creates a wrapper around the broker's submit_order that:
        1. Calls risk_manager.validate_order() before submission
        2. Only submits if validation passes (returns non-None order)
        3. Allows risk rules to reject or modify orders

        The wrapper is transparent - strategies call broker.submit_order() normally.
        """
        # Store original submit_order method
        original_submit_order = self.broker.submit_order

        def submit_order_with_validation(order, timestamp=None):
            """Wrapped submit_order that validates through risk_manager."""
            # Get current market event for context (if available)
            # Note: This is best-effort - risk_manager.validate_order will build context
            current_event = getattr(self, '_current_market_event', None)

            if current_event and self.risk_manager:
                # Hook B: Validate order through risk manager
                validated_order = self.risk_manager.validate_order(
                    order=order,
                    market_event=current_event,
                    broker=self.broker,
                    portfolio=self.portfolio,
                )

                # If risk manager rejects (returns None), don't submit
                if validated_order is None:
                    logger.info(f"Order rejected by risk manager: {order.asset_id} {order.quantity}")
                    return None  # Return None to indicate rejection

                # Use validated (possibly modified) order
                order = validated_order

            # Submit the validated order
            return original_submit_order(order, timestamp)

        # Replace broker's submit_order with wrapped version
        self.broker.submit_order = submit_order_with_validation

    def _setup_event_handlers(self) -> None:
        """Connect components via Clock event subscriptions."""
        # Strategy subscribes to fill events
        # Note: MARKET events are handled specially by Engine to pass context
        self.clock.subscribe(EventType.FILL, self.strategy.on_event)

        # Broker subscribes to order events AND market events (for fill checking)
        self.clock.subscribe(EventType.ORDER, self.broker.on_order_event)
        self.clock.subscribe(EventType.MARKET, self.broker.on_market_event)

        # Portfolio subscribes to market events for position price updates
        # Market events update position prices for accurate unrealized PnL
        self.clock.subscribe(EventType.MARKET, self.portfolio.on_market_event)
        # NOTE: Portfolio does NOT subscribe to FILL events. The broker updates
        # _internal_portfolio directly during order processing, and after initialize()
        # _internal_portfolio IS the portfolio. Subscribing here would cause double-application.

        # Hook D: Risk manager subscribes to fill events for position state tracking
        if self.risk_manager:
            def record_fill_with_context(fill_event):
                """Wrapper to provide market_event context to risk_manager.record_fill."""
                current_event = getattr(self, '_current_market_event', None)
                if current_event:
                    self.risk_manager.record_fill(fill_event, current_event)

            self.clock.subscribe(EventType.FILL, record_fill_with_context)

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

        # Choose processing path based on strategy execution mode
        if self._strategy_mode == "batch":
            # Batch mode: Use timestamp-grouped processing for portfolio strategies
            self._run_batch_mode(start_date, end_date, max_events)
        else:
            # Atomic mode: Use Clock-driven event-by-event processing
            self._run_atomic_mode(start_date, end_date, max_events)

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

    def _run_atomic_mode(
        self,
        start_date: datetime | None,
        end_date: datetime | None,
        max_events: int | None,
    ) -> None:
        """Run backtest in atomic mode (event-by-event processing).

        This is the traditional event-driven approach where each market event
        is processed individually. Used by simple single-asset strategies.
        """
        logger.info("Running in ATOMIC mode (event-by-event processing)")

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
                break

            # Dispatch event to all registered subscribers
            # Special handling for MARKET events: pass context to strategy
            # Other events go through normal Clock dispatch
            if event.event_type == EventType.MARKET:
                # Get or create context for this timestamp
                context_dict = self.context_data.get(event.timestamp, {})
                context = self.context_cache.get_or_create(event.timestamp, context_dict)

                # Store current market event for Hook B (order validation)
                self._current_market_event = event

                # Hook C: Check risk management exit conditions BEFORE strategy
                # This allows risk rules to exit positions independently of strategy logic
                if self.risk_manager:
                    exit_orders = self.risk_manager.check_position_exits(
                        market_event=event,
                        broker=self.broker,
                        portfolio=self.portfolio,
                    )
                    # Submit risk-driven exit orders immediately
                    for order in exit_orders:
                        self.broker.submit_order(order)

                # Atomic mode: process immediately
                self.strategy.on_market_event(event, context.data)

                # Also call on_event for backward compatibility with strategies
                # that don't override on_market_event
                self.strategy.on_event(event)

                # Dispatch to other subscribers (broker, portfolio, reporter)
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

    def _run_batch_mode(
        self,
        start_date: datetime | None,
        end_date: datetime | None,
        max_events: int | None,
    ) -> None:
        """Run backtest in batch mode (time-slice processing).

        This is the optimized approach for portfolio strategies where all assets
        need to be processed simultaneously. Events are grouped by timestamp and
        processed in three phases:
          1. Fill old orders (from previous timestamp)
          2. Mark-to-market (update portfolio values)
          3. Strategy decisions (new orders)

        This approach:
          - Reduces iterations from 126,000 (500 × 252) to 252 (timestamps only)
          - Enables simultaneous cross-asset portfolio rebalancing
          - Eliminates T+2 execution delay bugs
        """
        logger.info("Running in BATCH mode (time-slice processing)")

        # Check if data feed supports batch streaming
        if not hasattr(self.data_feed, 'stream_by_timestamp'):
            raise ValueError(
                f"Data feed {type(self.data_feed).__name__} does not support "
                "stream_by_timestamp(). Use MultiSymbolDataFeed for batch mode."
            )

        self.events_processed = 0
        PROGRESS_LOG_INTERVAL = 50  # Log every 50 timestamps (not 10k events)

        # Iterate by timestamp using the feed's batch streaming
        for timestamp, events in self.data_feed.stream_by_timestamp():
            # Check max events limit
            if max_events and self.events_processed >= max_events:
                logger.info(f"Reached max events limit: {max_events}")
                break

            # Filter by date range if specified
            if start_date and timestamp < start_date:
                continue
            if end_date and timestamp > end_date:
                break

            # Process this time-slice
            self._process_time_slice(timestamp, events)

            self.events_processed += len(events)

            # Log progress periodically
            if (self.events_processed // len(events)) % PROGRESS_LOG_INTERVAL == 0:
                logger.info(
                    f"Processed {self.events_processed:,} events "
                    f"({self.events_processed // len(events)} timestamps) at {timestamp}"
                )

    def _process_time_slice(self, timestamp: datetime, events: list[MarketEvent]) -> None:
        """Process all events for a single timestamp in three phases.

        Args:
            timestamp: The current timestamp being processed
            events: All MarketEvents for this timestamp (one per asset)

        Processing phases:
            1. Fill old orders: Process pending orders using new market data
            2. Mark-to-market: Update portfolio values with current prices
            3. Strategy logic: Allow strategy to make new decisions

        This ensures correct sequencing: fills happen before strategy sees state.
        """
        # Create market data map for efficient lookup
        market_map = {event.asset_id: event for event in events}

        # Get or create context for this timestamp
        context_dict = self.context_data.get(timestamp, {})
        context = self.context_cache.get_or_create(timestamp, context_dict)

        # --- PHASE 1: Fill old orders (from previous timestamp) ---
        # This must happen BEFORE strategy sees portfolio state
        # Uses the new market data to fill pending orders
        if hasattr(self.broker, 'process_batch_fills'):
            self.broker.process_batch_fills(timestamp, market_map)
        else:
            # Fallback: Process each event individually through broker
            for event in events:
                self.broker.on_market_event(event)

        # --- PHASE 2: Mark-to-market (update portfolio values) ---
        # Update portfolio with current market prices for accurate equity/margin
        for event in events:
            self.portfolio.on_market_event(event)

        # Check for corporate actions at this timestamp
        current_date = timestamp.date()
        pending_actions = self.corporate_action_processor.get_pending_actions(current_date)
        if pending_actions:
            self._process_corporate_actions(current_date, timestamp)

        # Hook C: Risk management exit checks BEFORE strategy
        if self.risk_manager:
            for event in events:
                self._current_market_event = event
                exit_orders = self.risk_manager.check_position_exits(
                    market_event=event,
                    broker=self.broker,
                    portfolio=self.portfolio,
                )
                for order in exit_orders:
                    self.broker.submit_order(order)

        # --- PHASE 3: Strategy logic (new decisions) ---
        # Strategy sees post-fill portfolio state and can make new decisions
        # Update strategy's market data cache for helper methods
        self.strategy._current_market_data = market_map

        # Call batch handler if available
        if hasattr(self.strategy, 'on_data_batch'):
            self.strategy.on_data_batch(timestamp, market_map, context.data)
        elif hasattr(self.strategy, 'on_timestamp_batch'):
            # Backward compatibility with old batch API
            self.strategy.on_timestamp_batch(timestamp, events, context.data)
        else:
            # Fallback: Call on_market_event for each event
            for event in events:
                self._current_market_event = event
                self.strategy.on_market_event(event, context.data)

        # Dispatch to reporter for logging
        for event in events:
            self.reporter.on_event(event)

    def _process_corporate_actions(self, current_date, timestamp) -> None:
        """Process corporate actions for a given date.

        Args:
            current_date: Date to check for corporate actions
            timestamp: Timestamp for logging
        """
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
                logger.warning(
                    f"Corporate action created new position for {asset_id} "
                    f"with quantity {new_quantity}. Manual position creation not yet supported."
                )

        # Apply cash adjustment (dividends, merger proceeds, etc.)
        cash_change = updated_cash - cash
        if cash_change != 0:
            self.broker.adjust_cash(cash_change)

        # Log notifications
        for notification in notifications:
            logger.info(f"Corporate action: {notification}")
            if self.reporter and hasattr(self.reporter, 'add_note'):
                self.reporter.add_note(timestamp, notification)

    def _compile_results(self) -> dict[str, Any]:
        """Compile backtest results from all components.

        Returns:
            Dictionary with comprehensive backtest results
        """
        # Get data from components
        trades = self.broker.trades
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
            "final_value": self.broker._internal_portfolio.equity,  # Use broker's portfolio (source of truth)
            "total_return": (self.broker._internal_portfolio.equity / self.initial_capital - 1) * 100,
        }

        # Add reporter data if available
        if hasattr(self.reporter, "get_report"):
            results["report"] = self.reporter.get_report()

        return results

    def get_results(self) -> "BacktestResultsExporter":
        """Get results exporter for trade and return data.

        Returns:
            BacktestResultsExporter with access to trades and returns

        Example:
            >>> engine = BacktestEngine(...)
            >>> engine.run()
            >>> results = engine.get_results()
            >>> results.export_all("results/")
        """
        from ml4t.backtest.results import BacktestResults as BacktestResultsExporter

        # Get performance analyzer from portfolio (may be None if analytics disabled)
        analyzer = getattr(self.broker.portfolio, "_analyzer", None)

        return BacktestResultsExporter(
            trade_tracker=self.broker.trade_tracker,
            performance_analyzer=analyzer,
        )

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

        # Reset batch mode state
        self._event_batch = []
        self._current_batch_timestamp = None

    def _collect_or_process_batch(
        self,
        event: MarketEvent,
        context: dict[str, Any],
    ) -> None:
        """
        Collect events into batch or process batch when timestamp changes.

        Args:
            event: Market event to add to batch
            context: Market-wide context for this timestamp
        """
        # Check if this is a new timestamp
        if self._current_batch_timestamp is None:
            # First event - start new batch
            self._current_batch_timestamp = event.timestamp
            self._event_batch = [event]
            self._batch_context = context
        elif event.timestamp == self._current_batch_timestamp:
            # Same timestamp - add to batch
            self._event_batch.append(event)
        else:
            # New timestamp - process previous batch first
            self._process_event_batch()

            # Start new batch
            self._current_batch_timestamp = event.timestamp
            self._event_batch = [event]
            self._batch_context = context

    def _process_event_batch(self) -> None:
        """Process collected batch of events for same timestamp."""
        if not self._event_batch:
            return

        # Call strategy's batch handler
        self.strategy.on_timestamp_batch(
            timestamp=self._current_batch_timestamp,
            events=self._event_batch,
            context=self._batch_context,
        )

        # Also call on_event for backward compatibility
        for event in self._event_batch:
            self.strategy.on_event(event)

        # Clear batch
        self._event_batch = []
        self._current_batch_timestamp = None
        self._batch_context = None


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
            # Generate HTML report (requires ml4t-diagnostic)
            try:
                from ml4t.backtest.reporting.html import generate_html_report
                html = generate_html_report(self)
                with open(path, "w") as f:
                    f.write(html)
            except ImportError:
                raise ImportError(
                    "HTML report generation requires ml4t-diagnostic. "
                    "Install it or use .parquet/.json format."
                )
        else:
            raise ValueError(f"Unsupported file format: {path}")


__all__ = [
    "BacktestEngine",
    "BacktestResults",
]
