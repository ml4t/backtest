"""Base strategy class and interfaces for QEngine."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from qengine.core.event import Event, FillEvent, MarketEvent, SignalEvent
from qengine.core.types import AssetId, EventType, OrderSide


class StrategyState(Enum):
    """Strategy lifecycle states."""

    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class StrategyContext:
    """Context object containing strategy runtime information."""

    start_time: datetime
    end_time: datetime
    initial_capital: float
    commission_model: Any | None = None
    slippage_model: Any | None = None
    data_feeds: list[Any] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.

    This class defines the interface that all strategies must implement.
    Strategies receive events through the on_event method and can submit
    orders through the broker interface.
    """

    def __init__(self, name: str | None = None):
        """
        Initialize the strategy.

        Args:
            name: Optional name for the strategy
        """
        self.name = name or self.__class__.__name__
        self.state = StrategyState.INITIALIZED
        self.broker = None  # Will be injected by the engine
        self.data = None  # PIT data accessor
        self.context = None  # Strategy context
        self._subscriptions: set[tuple] = set()
        self._positions: dict[AssetId, float] = {}
        self._orders: list[Any] = []
        self._trades: list[Any] = []

    def on_start(self, portfolio=None, event_bus=None) -> None:
        """
        Called once when the strategy starts.

        Args:
            portfolio: Portfolio instance for position tracking (optional)
            event_bus: Event bus for publishing orders (optional)

        Override this method to perform one-time initialization tasks like:
        - Setting up indicators
        - Subscribing to data feeds
        - Initializing internal state
        """
        # Store references if provided
        if portfolio is not None:
            self.portfolio = portfolio
        if event_bus is not None:
            self.event_bus = event_bus

    @abstractmethod
    def on_event(self, event: Event) -> None:
        """
        Process an event.

        This is the main method where strategy logic is implemented.
        It's called for every event the strategy is subscribed to.

        Event routing is now properly fixed to ensure strategies receive
        all relevant events including MarketEvent, FillEvent, etc.

        Args:
            event: The event to process (MarketEvent, FillEvent, etc.)
        """

    def on_market_event(self, event: MarketEvent) -> None:
        """
        Process a market data event.

        Override this for specialized market data handling.

        Args:
            event: Market data event
        """

    def on_signal_event(self, event: SignalEvent) -> None:
        """
        Process an ML signal event.

        Override this for signal-based strategies.

        Args:
            event: Signal event from ML model
        """

    def on_fill_event(self, event: FillEvent) -> None:
        """
        Process an order fill event.

        Override this to track fills and update internal state.

        Args:
            event: Fill event
        """
        # Default implementation updates position tracking
        if event.side in [OrderSide.BUY, OrderSide.COVER]:
            self._positions[event.asset_id] = (
                self._positions.get(event.asset_id, 0) + event.fill_quantity
            )
        else:
            self._positions[event.asset_id] = (
                self._positions.get(event.asset_id, 0) - event.fill_quantity
            )

    def on_end(self) -> None:
        """
        Called once when the strategy stops.

        Override this to perform cleanup tasks like:
        - Closing positions
        - Saving state
        - Final analysis
        """

    def before_trading_start(self) -> None:
        """
        Called before the start of each trading day.

        Override this for daily preparation tasks like:
        - Updating universe
        - Recalculating signals
        - Adjusting parameters
        """

    def after_trading_end(self) -> None:
        """
        Called after the end of each trading day.

        Override this for end-of-day tasks like:
        - Recording metrics
        - Rebalancing
        - Risk calculations
        """

    def subscribe(
        self,
        asset: AssetId | None = None,
        event_type: EventType | None = None,
        **kwargs,
    ) -> None:
        """
        Subscribe to specific events.

        Args:
            asset: Asset to subscribe to (None for all)
            event_type: Type of events to receive
            **kwargs: Additional subscription parameters
        """
        subscription = (asset, event_type, tuple(kwargs.items()))
        self._subscriptions.add(subscription)

    def unsubscribe(
        self,
        asset: AssetId | None = None,
        event_type: EventType | None = None,
        **kwargs,
    ) -> None:
        """
        Unsubscribe from specific events.

        Args:
            asset: Asset to unsubscribe from
            event_type: Type of events to stop receiving
            **kwargs: Additional parameters
        """
        subscription = (asset, event_type, tuple(kwargs.items()))
        self._subscriptions.discard(subscription)

    @property
    def current_positions(self) -> dict[AssetId, float]:
        """Get current position quantities by asset."""
        return self._positions.copy()

    @property
    def is_flat(self) -> bool:
        """Check if strategy has no positions."""
        return all(qty == 0 for qty in self._positions.values())

    def log(self, message: str, level: str = "INFO", timestamp: datetime | None = None) -> None:
        """
        Log a message.

        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR, DEBUG)
            timestamp: Simulation timestamp (if None, uses wall clock time for compatibility)
        """
        # Use simulation time if provided, otherwise fall back to wall clock time
        if timestamp:
            time_str = timestamp.isoformat()
        else:
            time_str = datetime.now().isoformat()
        print(f"[{time_str}] [{self.name}] [{level}] {message}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', state={self.state})"


class SignalStrategy(Strategy):
    """
    Base class for signal-based strategies.

    This provides a simpler interface for strategies that primarily
    react to ML signals rather than raw market data.
    """

    def __init__(self, name: str | None = None, signal_threshold: float = 0.5):
        """
        Initialize signal strategy.

        Args:
            name: Strategy name
            signal_threshold: Threshold for acting on signals
        """
        super().__init__(name)
        self.signal_threshold = signal_threshold
        self._signal_history: dict[AssetId, list[float]] = {}

    def on_event(self, event: Event) -> None:
        """Route events to appropriate handlers."""
        if isinstance(event, SignalEvent):
            self.on_signal_event(event)
        elif isinstance(event, MarketEvent):
            self.on_market_event(event)
        elif isinstance(event, FillEvent):
            self.on_fill_event(event)

    def on_signal_event(self, event: SignalEvent) -> None:
        """
        Process signal and make trading decision.

        Args:
            event: Signal event
        """
        # Track signal history
        if event.asset_id not in self._signal_history:
            self._signal_history[event.asset_id] = []
        self._signal_history[event.asset_id].append(event.signal_value)

        # Make trading decision based on signal
        self.process_signal(event.asset_id, event.signal_value, event.confidence)

    @abstractmethod
    def process_signal(
        self,
        asset_id: AssetId,
        signal_value: float,
        confidence: float | None,
    ) -> None:
        """
        Process a signal and decide on action.

        Args:
            asset_id: Asset the signal is for
            signal_value: Signal value (typically -1 to 1)
            confidence: Optional confidence score
        """


class IndicatorStrategy(Strategy):
    """
    Base class for indicator-based strategies.

    Provides utilities for managing technical indicators.
    """

    def __init__(self, name: str | None = None):
        """Initialize indicator strategy."""
        super().__init__(name)
        self._indicators: dict[str, Any] = {}

    def add_indicator(self, name: str, indicator: Any) -> None:
        """
        Add a technical indicator.

        Args:
            name: Name for the indicator
            indicator: Indicator instance
        """
        self._indicators[name] = indicator

    def get_indicator(self, name: str) -> Any:
        """
        Get an indicator by name.

        Args:
            name: Indicator name

        Returns:
            Indicator instance
        """
        return self._indicators.get(name)

    def update_indicators(self, price: float) -> None:
        """
        Update all indicators with new price.

        Args:
            price: Latest price
        """
        for indicator in self._indicators.values():
            if hasattr(indicator, "update"):
                indicator.update(price)
