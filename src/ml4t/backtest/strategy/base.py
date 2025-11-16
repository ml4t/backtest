"""Base strategy class and interfaces for ml4t.backtest."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ml4t.backtest.core.event import Event, FillEvent, MarketEvent, SignalEvent
from ml4t.backtest.core.types import AssetId, EventType, OrderSide


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

    def on_market_event(self, event: MarketEvent, context: dict[str, Any] | None = None) -> None:
        """
        Process a market data event.

        Override this for specialized market data handling.

        Args:
            event: Market data event
            context: Market-wide context data (VIX, SPY, regime indicators, etc.)
                    Dictionary with timestamp-specific values shared across all assets.
                    Example: {'VIX': 18.5, 'SPY': 485.0, 'regime': 'bull'}
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

    # ===== Trading Helper Methods =====
    def get_position(self, asset_id: AssetId) -> float:
        """
        Get current position quantity for an asset.

        Args:
            asset_id: Asset identifier

        Returns:
            Current quantity (positive for long, negative for short, 0 for no position)

        Raises:
            ValueError: If broker is not initialized
        """
        if self.broker is None:
            raise ValueError("Broker not initialized. Strategy must be connected to broker.")

        position = self.broker.get_position(asset_id)
        return position if position is not None else 0.0

    def get_cash(self) -> float:
        """
        Get current available cash.

        Returns:
            Available cash balance

        Raises:
            ValueError: If broker is not initialized
        """
        if self.broker is None:
            raise ValueError("Broker not initialized. Strategy must be connected to broker.")

        return self.broker.get_cash()

    def get_portfolio_value(self) -> float:
        """
        Get total portfolio value (cash + positions).

        Returns:
            Total equity value

        Raises:
            ValueError: If broker is not initialized
        """
        if self.broker is None:
            raise ValueError("Broker not initialized. Strategy must be connected to broker.")

        return self.broker._internal_portfolio.equity

    def buy_percent(
        self,
        asset_id: AssetId,
        percent: float,
        price: float,
        limit_price: float | None = None,
    ) -> None:
        """
        Buy an asset using a percentage of portfolio value.

        Args:
            asset_id: Asset to buy
            percent: Percentage of portfolio to allocate (0.0 to 1.0, e.g., 0.10 for 10%)
            price: Current market price (for quantity calculation)
            limit_price: Optional limit price for limit order (if None, uses market order)

        Raises:
            ValueError: If broker is not initialized or parameters are invalid

        Example:
            # Buy 10% of portfolio at current price
            self.buy_percent("AAPL", 0.10, event.close)

            # Buy 10% with limit order
            self.buy_percent("AAPL", 0.10, event.close, limit_price=150.0)
        """
        if self.broker is None:
            raise ValueError("Broker not initialized. Strategy must be connected to broker.")

        if not 0.0 <= percent <= 1.0:
            raise ValueError(f"Percent must be between 0.0 and 1.0, got {percent}")

        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")

        # Calculate quantity based on portfolio value
        portfolio_value = self.get_portfolio_value()
        dollars_to_spend = portfolio_value * percent
        quantity = dollars_to_spend / price

        if quantity <= 0:
            return  # Nothing to buy

        # Import Order class here to avoid circular imports
        from ml4t.backtest.execution.order import Order
        from ml4t.backtest.core.types import OrderType

        # Create and submit order
        if limit_price is not None:
            order = Order(
                asset_id=asset_id,
                side=OrderSide.BUY,
                quantity=quantity,
                order_type=OrderType.LIMIT,
                limit_price=limit_price,
            )
        else:
            order = Order(
                asset_id=asset_id,
                side=OrderSide.BUY,
                quantity=quantity,
                order_type=OrderType.MARKET,
            )

        self.broker.submit_order(order)

    def sell_percent(
        self,
        asset_id: AssetId,
        percent: float,
        limit_price: float | None = None,
    ) -> None:
        """
        Sell a percentage of current position.

        Args:
            asset_id: Asset to sell
            percent: Percentage of position to sell (0.0 to 1.0, e.g., 0.50 for 50%)
            limit_price: Optional limit price for limit order (if None, uses market order)

        Raises:
            ValueError: If broker is not initialized or parameters are invalid

        Example:
            # Sell 50% of position
            self.sell_percent("AAPL", 0.50)

            # Sell 50% with limit order
            self.sell_percent("AAPL", 0.50, limit_price=155.0)
        """
        if self.broker is None:
            raise ValueError("Broker not initialized. Strategy must be connected to broker.")

        if not 0.0 <= percent <= 1.0:
            raise ValueError(f"Percent must be between 0.0 and 1.0, got {percent}")

        # Get current position
        current_position = self.get_position(asset_id)

        if current_position <= 0:
            return  # No position to sell

        # Calculate quantity to sell
        quantity = abs(current_position) * percent

        if quantity <= 0:
            return  # Nothing to sell

        # Import Order class here to avoid circular imports
        from ml4t.backtest.execution.order import Order
        from ml4t.backtest.core.types import OrderType

        # Create and submit order
        if limit_price is not None:
            order = Order(
                asset_id=asset_id,
                side=OrderSide.SELL,
                quantity=quantity,
                order_type=OrderType.LIMIT,
                limit_price=limit_price,
            )
        else:
            order = Order(
                asset_id=asset_id,
                side=OrderSide.SELL,
                quantity=quantity,
                order_type=OrderType.MARKET,
            )

        self.broker.submit_order(order)

    def close_position(self, asset_id: AssetId, limit_price: float | None = None) -> None:
        """
        Close entire position for an asset.

        Args:
            asset_id: Asset to close
            limit_price: Optional limit price for limit order (if None, uses market order)

        Raises:
            ValueError: If broker is not initialized

        Example:
            # Close position at market
            self.close_position("AAPL")

            # Close position with limit order
            self.close_position("AAPL", limit_price=155.0)
        """
        if self.broker is None:
            raise ValueError("Broker not initialized. Strategy must be connected to broker.")

        # Get current position
        current_position = self.get_position(asset_id)

        if current_position == 0:
            return  # No position to close

        # Determine side and quantity
        if current_position > 0:
            # Long position - sell to close
            side = OrderSide.SELL
            quantity = abs(current_position)
        else:
            # Short position - buy to cover
            side = OrderSide.BUY
            quantity = abs(current_position)

        # Import Order class here to avoid circular imports
        from ml4t.backtest.execution.order import Order
        from ml4t.backtest.core.types import OrderType

        # Create and submit order
        if limit_price is not None:
            order = Order(
                asset_id=asset_id,
                side=side,
                quantity=quantity,
                order_type=OrderType.LIMIT,
                limit_price=limit_price,
            )
        else:
            order = Order(
                asset_id=asset_id,
                side=side,
                quantity=quantity,
                order_type=OrderType.MARKET,
            )

        self.broker.submit_order(order)

    # ===== ML-Specific Helper Methods =====
    def size_by_confidence(
        self,
        asset_id: AssetId,
        confidence: float,
        max_percent: float,
        price: float,
        limit_price: float | None = None,
    ) -> None:
        """
        Size position based on ML model confidence score.

        Uses Kelly-like scaling: position_size = max_percent * confidence

        Args:
            asset_id: Asset to trade
            confidence: ML model confidence (0.0 to 1.0)
            max_percent: Maximum portfolio percentage at full confidence (0.0 to 1.0)
            price: Current market price (for quantity calculation)
            limit_price: Optional limit price for limit order (if None, uses market order)

        Raises:
            ValueError: If broker is not initialized or parameters are invalid

        Example:
            # With 80% confidence, allocate 80% of max 20% = 16% of portfolio
            self.size_by_confidence("AAPL", confidence=0.80, max_percent=0.20, price=event.close)
        """
        if self.broker is None:
            raise ValueError("Broker not initialized. Strategy must be connected to broker.")

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")

        if not 0.0 <= max_percent <= 1.0:
            raise ValueError(f"Max percent must be between 0.0 and 1.0, got {max_percent}")

        # Calculate position size scaled by confidence
        position_percent = max_percent * confidence

        # Use existing buy_percent helper
        self.buy_percent(asset_id, position_percent, price, limit_price)

    def rebalance_to_weights(
        self,
        target_weights: dict[AssetId, float],
        current_prices: dict[AssetId, float],
        tolerance: float = 0.01,
    ) -> None:
        """
        Rebalance portfolio to target weights.

        For each asset:
        - If current weight > target weight + tolerance: sell excess
        - If current weight < target weight - tolerance: buy deficit

        Args:
            target_weights: Dictionary mapping asset_id to target weight (0.0 to 1.0)
                          Weights should sum to <= 1.0 (remainder stays in cash)
            current_prices: Dictionary mapping asset_id to current price
            tolerance: Rebalancing tolerance (default: 1% = 0.01)

        Raises:
            ValueError: If broker is not initialized or parameters are invalid

        Example:
            target_weights = {"AAPL": 0.40, "MSFT": 0.30, "GOOGL": 0.20}  # 10% cash
            current_prices = {"AAPL": event1.close, "MSFT": event2.close, "GOOGL": event3.close}
            self.rebalance_to_weights(target_weights, current_prices, tolerance=0.02)
        """
        if self.broker is None:
            raise ValueError("Broker not initialized. Strategy must be connected to broker.")

        # Validate target weights sum to <= 1.0
        total_weight = sum(target_weights.values())
        if total_weight > 1.0:
            raise ValueError(f"Target weights sum to {total_weight:.3f}, must be <= 1.0")

        # Get portfolio value and current positions
        portfolio_value = self.get_portfolio_value()

        # Calculate current weights
        current_weights = {}
        for asset_id in set(list(target_weights.keys()) + list(current_prices.keys())):
            position = self.get_position(asset_id)
            if asset_id in current_prices and position != 0:
                position_value = abs(position) * current_prices[asset_id]
                current_weights[asset_id] = position_value / portfolio_value
            else:
                current_weights[asset_id] = 0.0

        # Import Order class here to avoid circular imports
        from ml4t.backtest.execution.order import Order
        from ml4t.backtest.core.types import OrderType

        # Rebalance each asset
        for asset_id, target_weight in target_weights.items():
            current_weight = current_weights.get(asset_id, 0.0)
            weight_diff = target_weight - current_weight

            # Check if rebalancing is needed (outside tolerance band)
            if abs(weight_diff) <= tolerance:
                continue  # Within tolerance, no rebalancing needed

            if asset_id not in current_prices:
                continue  # Can't rebalance without price

            price = current_prices[asset_id]

            # Calculate target position value and quantity
            target_value = portfolio_value * target_weight
            target_quantity = target_value / price

            # Get current position
            current_position = self.get_position(asset_id)

            # Calculate quantity to trade
            quantity_diff = target_quantity - abs(current_position)

            if abs(quantity_diff) < 1e-6:  # Negligible difference
                continue

            # Create and submit rebalancing order
            if quantity_diff > 0:
                # Need to buy more
                order = Order(
                    asset_id=asset_id,
                    side=OrderSide.BUY,
                    quantity=abs(quantity_diff),
                    order_type=OrderType.MARKET,
                )
                self.broker.submit_order(order)
            else:
                # Need to sell some
                order = Order(
                    asset_id=asset_id,
                    side=OrderSide.SELL,
                    quantity=abs(quantity_diff),
                    order_type=OrderType.MARKET,
                )
                self.broker.submit_order(order)

    def get_unrealized_pnl_pct(self, asset_id: AssetId) -> float | None:
        """
        Get unrealized P&L percentage for a position.

        Calculates: (current_value - cost_basis) / cost_basis

        Args:
            asset_id: Asset identifier

        Returns:
            Unrealized P&L as percentage (e.g., 0.15 for 15% gain, -0.10 for 10% loss)
            Returns None if no position exists

        Raises:
            ValueError: If broker is not initialized

        Example:
            pnl_pct = self.get_unrealized_pnl_pct("AAPL")
            if pnl_pct and pnl_pct > 0.20:  # 20% gain
                self.close_position("AAPL")  # Take profit
        """
        if self.broker is None:
            raise ValueError("Broker not initialized. Strategy must be connected to broker.")

        # Get position from broker's internal portfolio
        position_obj = self.broker._internal_portfolio.get_position(asset_id)

        if position_obj is None or position_obj.quantity == 0:
            return None  # No position

        # Calculate P&L percentage
        # Position object should have unrealized_pnl property
        cost_basis = position_obj.cost_basis
        if cost_basis == 0:
            return None  # Avoid division by zero

        unrealized_pnl = position_obj.unrealized_pnl
        pnl_pct = unrealized_pnl / abs(cost_basis)

        return pnl_pct

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
