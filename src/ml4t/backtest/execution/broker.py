"""Broker implementations for ml4t.backtest."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any

from ml4t.backtest.core.assets import AssetRegistry
from ml4t.backtest.core.event import FillEvent, MarketEvent
from ml4t.backtest.core.types import (
    AssetId,
    OrderId,
    OrderSide,
    OrderStatus,
    OrderType,
    Price,
    Quantity,
)
from ml4t.backtest.execution.bracket_manager import BracketOrderManager
from ml4t.backtest.execution.fill_simulator import FillSimulator
from ml4t.backtest.execution.order import Order, OrderState
from ml4t.backtest.execution.order_router import OrderRouter
from ml4t.backtest.execution.trade_tracker import TradeTracker
from ml4t.backtest.portfolio.margin import MarginAccount
from ml4t.backtest.portfolio.portfolio import Portfolio

if TYPE_CHECKING:
    from ml4t.backtest.execution.commission import CommissionModel
    from ml4t.backtest.execution.liquidity import LiquidityModel
    from ml4t.backtest.execution.market_impact import MarketImpactModel
    from ml4t.backtest.execution.slippage import SlippageModel

logger = logging.getLogger(__name__)


class Broker(ABC):
    """Abstract base class for broker implementations."""

    @abstractmethod
    def submit_order(self, order: Order) -> OrderId:
        """Submit an order for execution."""

    @abstractmethod
    def cancel_order(self, order_id: OrderId) -> bool:
        """Cancel an existing order."""

    @abstractmethod
    def get_order(self, order_id: OrderId) -> Order | None:
        """Get order by ID."""

    @abstractmethod
    def get_open_orders(self, asset_id: AssetId | None = None) -> list[Order]:
        """Get all open orders, optionally filtered by asset."""

    @abstractmethod
    def on_market_event(self, event: MarketEvent) -> list[FillEvent]:
        """Process market event and generate fills."""


class SimulationBroker(Broker):
    """
    Simulated broker for backtesting.

    Handles order execution with configurable realism models.
    Supports multiple asset classes including equities, futures, options, FX, and crypto.

    Features:
    - Execution delay: Prevents lookahead bias by delaying order fills by one market event
    - Multiple order types: Market, Limit, Stop, Trailing Stop, Bracket
    - Realistic fills: Slippage, commission, and market impact models
    - Position tracking: Multi-asset position management
    - Margin support: For derivatives trading
    """

    def __init__(
        self,
        initial_cash: float = 100000.0,
        asset_registry: AssetRegistry | None = None,
        commission_model: "CommissionModel | None" = None,
        slippage_model: "SlippageModel | None" = None,
        market_impact_model: "MarketImpactModel | None" = None,
        liquidity_model: "LiquidityModel | None" = None,
        fill_model: Any | None = None,
        enable_margin: bool = True,
        execution_delay: bool = True,  # Add execution delay to prevent lookahead (default True to prevent bias)
        max_leverage: float = 1.0,  # Maximum leverage allowed (1.0 = no leverage)
        allow_immediate_reentry: bool = True,  # Allow re-entry on same bar as exit (default True for realism)
    ):
        """
        Initialize simulation broker with specialized components.

        Args:
            initial_cash: Starting cash balance
            asset_registry: Registry of asset specifications
            commission_model: Model for calculating commissions
            slippage_model: Model for calculating slippage
            market_impact_model: Model for calculating market impact
            liquidity_model: Model for liquidity constraints and volume limits
            fill_model: Model for determining fills
            enable_margin: Whether to enable margin trading for derivatives
            execution_delay: Whether to delay order execution by one market event (default True to prevent lookahead bias)
            max_leverage: Maximum leverage allowed (default 1.0 = no leverage).
                Prevents unlimited leverage as capital depletes. Example: max_leverage=2.0 allows
                positions up to 2x cash balance.
            allow_immediate_reentry: Whether to allow re-entering a position on the same bar where it was exited.
                True (default) = realistic behavior, allows intrabar exit+entry if liquidity exists.
                False = VectorBT-compatible mode, prevents same-bar re-entry for validation purposes.
        """
        # Store configuration
        self.initial_cash = initial_cash
        self.asset_registry = asset_registry or AssetRegistry()
        self.commission_model = commission_model
        self.slippage_model = slippage_model
        self.market_impact_model = market_impact_model
        self.liquidity_model = liquidity_model
        self.fill_model = fill_model
        self.enable_margin = enable_margin
        self.execution_delay = execution_delay
        self.max_leverage = max_leverage
        self.allow_immediate_reentry = allow_immediate_reentry

        # Initialize portfolio (replaces old PositionTracker)
        # Note: Portfolio handles multiple assets with default precision
        # Asset-specific precision is handled in FillSimulator
        from ml4t.backtest.core.precision import PrecisionManager
        cash_precision_manager = PrecisionManager(
            position_decimals=8,  # Support fractional positions (crypto, fractional shares)
            price_decimals=2,
            cash_decimals=2,
        )
        # Internal portfolio for position tracking (replaces old PositionTracker)
        self._internal_portfolio = Portfolio(
            initial_cash=initial_cash,
            precision_manager=cash_precision_manager,
            track_analytics=False,  # Broker doesn't need performance analytics
        )
        self.order_router = OrderRouter(execution_delay)
        self.bracket_manager = BracketOrderManager(self.submit_order)

        # Margin account for derivatives
        if enable_margin:
            self.margin_account = MarginAccount(initial_cash, self.asset_registry)
        else:
            self.margin_account = None

        # Fill simulator
        self.fill_simulator = FillSimulator(
            asset_registry=self.asset_registry,
            commission_model=self.commission_model,
            slippage_model=self.slippage_model,
            market_impact_model=self.market_impact_model,
            liquidity_model=self.liquidity_model,
            margin_account=self.margin_account,
            max_leverage=self.max_leverage,
        )

        # Latest market prices
        self._last_prices: dict[AssetId, Price] = {}

        # Track last exit timestamp per asset (for allow_immediate_reentry=False)
        self._last_exit_time: dict[AssetId, datetime] = {}

        # VectorBT compatibility: Track newly-created brackets to skip checking on creation bar
        self._newly_created_brackets: set[str] = set()

        # Trade tracking (highly efficient) with cash precision manager
        self.trade_tracker = TradeTracker(precision_manager=cash_precision_manager)

        # Statistics
        self._total_commission = 0.0
        self._total_slippage = 0.0
        self._fill_count = 0

        logger.debug(
            f"SimulationBroker initialized with ${initial_cash:,.2f}, "
            f"execution_delay={execution_delay}, enable_margin={enable_margin}, "
            f"max_leverage={max_leverage}, allow_immediate_reentry={allow_immediate_reentry}"
        )

    # Properties for backward compatibility
    @property
    def cash(self) -> float:
        """Get current cash balance from Portfolio."""
        return self._internal_portfolio.cash

    @cash.setter
    def cash(self, value: float) -> None:
        """Set cash balance in Portfolio."""
        self._internal_portfolio.cash = value

    @property
    def _positions(self) -> dict[AssetId, Quantity]:
        """Get positions from Portfolio for backward compatibility."""
        return self._internal_portfolio.positions

    @property
    def _orders(self) -> dict[OrderId, Order]:
        """Get orders from OrderRouter for backward compatibility."""
        return self.order_router._orders

    @property
    def _open_orders(self) -> dict[AssetId, list[Order]]:
        """Get open orders from OrderRouter for backward compatibility."""
        return self.order_router._open_orders

    @property
    def _stop_orders(self) -> dict[AssetId, list[Order]]:
        """Get stop orders from OrderRouter for backward compatibility."""
        return self.order_router._stop_orders

    @property
    def _trailing_stops(self) -> dict[AssetId, list[Order]]:
        """Get trailing stops from OrderRouter for backward compatibility."""
        return self.order_router._trailing_stops

    @property
    def _pending_orders(self) -> dict[AssetId, list[tuple[Order, datetime]]]:
        """Get pending orders from OrderRouter for backward compatibility."""
        return self.order_router._pending_orders

    @property
    def trades(self) -> "pl.DataFrame":
        """
        Get completed trades as a Polars DataFrame.

        Returns DataFrame with columns:
        - trade_id: Unique trade identifier
        - asset_id: Asset symbol
        - entry_dt: Entry timestamp
        - entry_price: Entry execution price
        - entry_quantity: Position size
        - entry_commission: Entry fees
        - entry_slippage: Entry slippage cost
        - entry_order_id: Entry order ID
        - exit_dt: Exit timestamp
        - exit_price: Exit execution price
        - exit_quantity: Exit position size
        - exit_commission: Exit fees
        - exit_slippage: Exit slippage cost
        - exit_order_id: Exit order ID
        - pnl: Net profit/loss
        - return_pct: Return percentage
        - duration_bars: Trade duration in bars
        - direction: "long" or "short"

        Example:
            >>> broker = SimulationBroker(initial_cash=100000)
            >>> # ... run backtest ...
            >>> trades_df = broker.trades
            >>> print(f"Total trades: {len(trades_df)}")
            >>> print(f"Total PnL: ${trades_df['pnl'].sum():.2f}")
        """
        import polars as pl

        return self.trade_tracker.get_trades_df()

    def _get_asset_precision_manager(self, asset_id: AssetId) -> "PrecisionManager | None":
        """Get PrecisionManager for a specific asset.

        Args:
            asset_id: Asset identifier

        Returns:
            PrecisionManager for the asset, or None if asset not registered
        """
        from ml4t.backtest.core.precision import PrecisionManager

        asset_spec = self.asset_registry.get(asset_id)
        if asset_spec:
            return asset_spec.get_precision_manager()
        return None

    def can_enter_position(self, asset_id: AssetId, current_time: datetime) -> bool:
        """
        Check if a new position can be entered for the given asset.

        When allow_immediate_reentry=False, prevents entering a position on the same
        bar where the previous position was exited (VectorBT-compatible behavior).

        Args:
            asset_id: Asset to check
            current_time: Current bar timestamp

        Returns:
            True if entry is allowed, False otherwise
        """
        if self.allow_immediate_reentry:
            return True

        # Check if we exited on this same bar
        if asset_id in self._last_exit_time:
            return self._last_exit_time[asset_id] != current_time

        return True

    def submit_order(self, order: Order, timestamp: datetime | None = None) -> OrderId:
        """
        Submit an order for execution.

        Args:
            order: Order to submit
            timestamp: Current simulation time (if None, uses datetime.now() for compatibility)

        Returns:
            Order ID

        Raises:
            ValueError: If order is invalid
        """
        # Validate order
        if order.quantity <= 0:
            raise ValueError(f"Order quantity must be positive, got {order.quantity}")

        if order.order_type == OrderType.LIMIT and order.limit_price is None:
            raise ValueError("LIMIT order requires limit_price")

        if order.order_type == OrderType.STOP and order.stop_price is None:
            raise ValueError("STOP order requires stop_price")

        if order.order_type == OrderType.STOP_LIMIT:
            if order.limit_price is None:
                raise ValueError("STOP_LIMIT order requires limit_price")
            if order.stop_price is None:
                raise ValueError("STOP_LIMIT order requires stop_price")

        if order.limit_price is not None and order.limit_price <= 0:
            raise ValueError(f"Limit price must be positive, got {order.limit_price}")

        if order.stop_price is not None and order.stop_price <= 0:
            raise ValueError(f"Stop price must be positive, got {order.stop_price}")

        # Update order state
        order.state = OrderState.SUBMITTED
        order.status = OrderStatus.SUBMITTED
        order.submitted_time = timestamp if timestamp else datetime.now()

        # Initialize trailing stop price if applicable
        if order.order_type == OrderType.TRAILING_STOP and order.asset_id in self._last_prices:
            order.update_trailing_stop(self._last_prices[order.asset_id])

        # Route order (delegates to OrderRouter)
        self.order_router.route_order(order, order.submitted_time)

        # Legacy immediate execution for market orders (no delay mode)
        if not self.execution_delay and order.order_type == OrderType.MARKET:
            if order.asset_id in self._last_prices:
                # Create minimal MarketEvent for legacy immediate execution
                from ml4t.backtest.core.event import MarketEvent
                fill_event = MarketEvent(
                    timestamp=timestamp,
                    asset_id=order.asset_id,
                    data_type="trade",
                    price=self._last_prices[order.asset_id],
                    close=self._last_prices[order.asset_id],
                )
                fill_result = self.fill_simulator.try_fill_order(
                    order,
                    fill_event,
                    current_cash=self._internal_portfolio.cash,
                    current_position=self.get_position(order.asset_id),
                )
                if fill_result:
                    # Note: order.update_fill() is already called by FillSimulator.try_fill_order()

                    # Update position tracker (convert side to quantity_change)
                    quantity_change = fill_result.fill_quantity if order.side == OrderSide.BUY else -fill_result.fill_quantity
                    self._internal_portfolio.update_position(
                        order.asset_id,
                        quantity_change,
                        fill_result.fill_price,
                        fill_result.commission,
                        fill_result.slippage,
                    )
                    # Update statistics
                    self._total_commission += fill_result.commission
                    self._total_slippage += fill_result.slippage
                    self._fill_count += 1

                    # Track trade (same as delayed execution path)
                    self.trade_tracker.on_fill(fill_result.fill_event)

                    # Publish fill event immediately if order was filled
                    if hasattr(self, "event_bus") and self.event_bus:
                        self.event_bus.publish(fill_result.fill_event)

                    # Remove filled orders and handle bracket completions (same as delayed path)
                    if order.is_filled:
                        self.order_router.remove_order(order)

                        # Handle bracket order completion
                        if order.order_type == OrderType.BRACKET:
                            self._handle_bracket_fill(order, fill_result.fill_event)

        logger.debug(f"Submitted order: {order}")
        print(f"BROKER submit_order: {order.asset_id[:12]:<12} {order.side.name} {order.quantity:>6.0f} {order.order_type.name} "
              f"status={order.status.name} id={order.order_id}")
        return order.order_id

    def cancel_order(self, order_id: OrderId) -> bool:
        """
        Cancel an existing order (delegates to OrderRouter).

        Args:
            order_id: ID of order to cancel

        Returns:
            True if cancelled successfully
        """
        order = self.order_router.get_order(order_id)
        if not order or not order.is_active:
            return False

        # Remove from queues (delegates to OrderRouter)
        removed = self.order_router.remove_order(order)

        if removed:
            # Update order state
            order.cancel()
            logger.debug(f"Cancelled order: {order}")
            return True

        return False

    def get_order(self, order_id: OrderId) -> Order | None:
        """Get order by ID (delegates to OrderRouter)."""
        return self.order_router.get_order(order_id)

    def get_open_orders(self, asset_id: AssetId | None = None) -> list[Order]:
        """Get all open orders, optionally filtered by asset (delegates to OrderRouter)."""
        return [o for o in self.order_router.get_open_orders(asset_id) if o.is_active]

    def _get_exit_priority(self, bracket_type: str) -> int:
        """
        Get VectorBT exit priority for bracket leg type.

        Priority Order (lower number = higher priority):
            1. Stop Loss (SL)
            2. Trailing Stop Loss (TSL)
            3. Take Profit (TP)

        Args:
            bracket_type: Type from order.metadata["bracket_type"]

        Returns:
            Priority value (1 = highest, 3 = lowest)
        """
        priority_map = {
            "stop_loss": 1,      # Highest priority
            "trailing_stop": 2,  # Medium priority
            "take_profit": 3,    # Lowest priority
        }
        return priority_map.get(bracket_type, 999)  # Unknown types get lowest priority

    def _collect_triggered_bracket_exits(
        self,
        asset_id: AssetId,
        event: MarketEvent,
        price: Price,
    ) -> list[tuple[int, str, Order]]:
        """
        Collect all triggered bracket exit orders without filling them.

        Returns list of (priority, bracket_type, order) tuples sorted by priority.
        Only includes bracket leg orders (stop_loss, trailing_stop, take_profit).

        Args:
            asset_id: Asset identifier
            event: Market event with OHLC data
            price: Current market price

        Returns:
            List of (priority, bracket_type, order) tuples, sorted by priority (lowest first)
        """
        triggered = []

        # Check TP orders (LIMIT orders in open_orders with bracket_type="take_profit")
        for order in list(self._open_orders[asset_id]):
            bracket_type = order.metadata.get("bracket_type")
            if bracket_type == "take_profit":
                # VectorBT: Skip newly-created brackets (created in current event)
                if order.order_id in self._newly_created_brackets:
                    continue

                # TP triggers when high >= limit_price
                if order.can_fill(price=price, high=event.high, low=event.low):
                    priority = self._get_exit_priority(bracket_type)
                    triggered.append((priority, bracket_type, order))

        # Check SL orders (STOP orders in stop_orders with bracket_type="stop_loss")
        for order in list(self._stop_orders[asset_id]):
            bracket_type = order.metadata.get("bracket_type")
            if bracket_type == "stop_loss":
                # VectorBT: Skip newly-created brackets (created in current event)
                if order.order_id in self._newly_created_brackets:
                    continue

                # SL triggers when low <= stop_price
                if self._should_trigger_stop(order, price):
                    priority = self._get_exit_priority(bracket_type)
                    triggered.append((priority, bracket_type, order))

        # Check TSL orders (TRAILING_STOP orders with bracket_type="trailing_stop")
        for order in list(self._trailing_stops[asset_id]):
            bracket_type = order.metadata.get("bracket_type")
            if bracket_type == "trailing_stop":
                # VectorBT: Skip newly-created brackets (created in current event)
                if order.order_id in self._newly_created_brackets:
                    continue

                # Initialize peak tracking if not present
                if "peak_price" not in order.metadata:
                    order.metadata["peak_price"] = order.metadata.get("base_price", price)

                # Initialize TSL activation flag (for threshold support)
                if "tsl_activated" not in order.metadata:
                    order.metadata["tsl_activated"] = False

                # VectorBT 4-stage per-bar process (from TASK-018)
                # Capture peak before any updates (for trigger logic)
                peak_before_open = order.metadata["peak_price"]

                # Stage 1: Update peak with open
                peak_price = order.metadata["peak_price"]
                if event.open > peak_price:
                    peak_price = event.open
                    order.metadata["peak_price"] = peak_price

                # Check TSL threshold activation (if threshold specified)
                base_price = order.metadata.get("base_price", price)
                if order.tsl_threshold_pct is not None and not order.metadata["tsl_activated"]:
                    # TSL only activates if peak exceeds base_price * (1 + threshold)
                    threshold_price = base_price * (1.0 + order.tsl_threshold_pct)
                    if peak_price >= threshold_price:
                        order.metadata["tsl_activated"] = True
                elif order.tsl_threshold_pct is None:
                    # No threshold - TSL always active
                    order.metadata["tsl_activated"] = True

                # Stage 2: Calculate TSL level (only if activated)
                if order.trail_percent is not None and order.metadata["tsl_activated"]:
                    trail_amount = peak_price * (order.trail_percent / 100.0)
                    # For SELL orders (exiting long): TSL = peak - trail_amount (stop BELOW)
                    # For BUY orders (exiting short): TSL = peak + trail_amount (stop ABOVE)
                    if order.is_buy:
                        # BUY order (exiting short): TSL above peak
                        tsl_level = peak_price + trail_amount
                    else:
                        # SELL order (exiting long): TSL below peak
                        tsl_level = peak_price - trail_amount
                    order.trailing_stop_price = tsl_level

                # Stage 3: Update peak with high
                if event.high > peak_price:
                    peak_price = event.high
                    order.metadata["peak_price"] = peak_price

                    # Re-check threshold activation with updated peak
                    if order.tsl_threshold_pct is not None and not order.metadata["tsl_activated"]:
                        threshold_price = base_price * (1.0 + order.tsl_threshold_pct)
                        if peak_price >= threshold_price:
                            order.metadata["tsl_activated"] = True

                # Stage 4: Recalculate TSL with updated peak and check trigger (only if activated)
                # Rule: High-based peak updates can trigger ONLY if open already moved the peak
                # This prevents "ghost triggers" when bar opens flat but high spikes
                if order.trail_percent is not None and order.metadata["tsl_activated"]:
                    trail_amount = peak_price * (order.trail_percent / 100.0)
                    if order.is_buy:
                        # BUY order (exiting short): TSL above peak
                        tsl_level = peak_price + trail_amount
                    else:
                        # SELL order (exiting long): TSL below peak
                        tsl_level = peak_price - trail_amount
                    order.trailing_stop_price = tsl_level

                    # Check trigger: Allow high-based peak to trigger ONLY if open already moved peak
                    # If open didn't move peak, use pre-open TSL level for trigger check
                    if not order.is_filled:
                        if event.open > peak_before_open:
                            # Open moved peak, so high update can trigger in same bar
                            can_fill_result = order.can_fill(price=price, high=event.high, low=event.low)
                            if can_fill_result:
                                priority = self._get_exit_priority(bracket_type)
                                triggered.append((priority, bracket_type, order))
                        else:
                            # Open didn't move peak, check trigger with pre-open TSL
                            if order.is_buy:
                                pre_open_tsl = peak_before_open + (peak_before_open * order.trail_percent / 100.0)
                            else:
                                pre_open_tsl = peak_before_open - (peak_before_open * order.trail_percent / 100.0)
                            # Manually check if low/high reached pre-open TSL
                            if order.is_buy:
                                if event.high >= pre_open_tsl:
                                    priority = self._get_exit_priority(bracket_type)
                                    triggered.append((priority, bracket_type, order))
                            else:
                                if event.low <= pre_open_tsl:
                                    priority = self._get_exit_priority(bracket_type)
                                    triggered.append((priority, bracket_type, order))

        # Sort by priority (lowest number = highest priority)
        triggered.sort(key=lambda x: x[0])
        return triggered

    def on_market_event(self, event: MarketEvent) -> list[FillEvent]:
        """
        Process market event and generate fills.

        Args:
            event: Market data event

        Returns:
            List of fill events generated
        """
        # Increment bar counter for trade duration tracking
        self.trade_tracker.on_bar()

        fills = []
        asset_id = event.asset_id

        # Track orders activated in this event (to prevent same-event fills)
        newly_activated_orders = set()

        # FIX #1: Move pending orders to open FIRST (at START of event)
        # This allows orders placed on day T to fill on day T+1 (next bar), not T+2
        # Industry standard for daily data: signal T → fill T+1
        # IMPORTANT: Orders activated in this event should NOT fill until next event
        if self.execution_delay and asset_id in self._pending_orders:
            for order, _ in self._pending_orders[asset_id]:
                if order.order_type == OrderType.BRACKET:
                    # Bracket orders start as regular orders and create legs after fill
                    self._open_orders[asset_id].append(order)
                else:
                    self._open_orders[asset_id].append(order)
                # Mark as newly activated so we don't fill it this event
                newly_activated_orders.add(order.order_id)
            # Clear pending orders after moving them
            self._pending_orders[asset_id].clear()

        # FIX #2: Determine execution price - use OPEN for realistic daily backtesting
        # When execution_delay=True (next-bar fills), fill at next bar's OPEN (not CLOSE!)
        # This is the industry standard: you see day T's data, place order end-of-day,
        # order fills at day T+1's open (first available price)
        if self.execution_delay and event.open is not None:
            price = event.open  # Use next bar's open (realistic for daily data)
        elif event.close is not None:
            price = event.close  # Fallback to close (legacy behavior or intraday)
        elif event.price is not None:
            price = event.price  # Fallback to generic price
        else:
            return fills  # No price available

        # Update last known price
        self._last_prices[asset_id] = price

        # Process open orders (includes orders that were just moved from pending)
        # IMPORTANT: Skip bracket exit orders here - they're handled with priority logic below
        for order in list(self._open_orders[asset_id]):
            if not order.is_active:
                continue

            # Skip orders that were just activated in this event (execution delay)
            if self.execution_delay and order.order_id in newly_activated_orders:
                continue

            # Skip already-filled orders (prevent double-fill attempts)
            # DEBUG: Log when we're skipping filled orders
            if order.remaining_quantity <= 0:
                if not hasattr(self, '_debug_skip_count'):
                    self._debug_skip_count = 0
                if self._debug_skip_count < 10:
                    print(f"  [BROKER] Skipping filled order: {order.order_id} for {asset_id}, remaining={order.remaining_quantity}")
                    self._debug_skip_count += 1
                continue

            # Skip bracket exits (TP, SL, TSL) - they use priority-based processing
            if order.metadata.get("bracket_type") in ["take_profit", "stop_loss", "trailing_stop"]:
                continue

            # For next-bar fills (execution_delay=True), override close with open
            # Create a modified event if needed for next-bar execution
            if self.execution_delay:
                from ml4t.backtest.core.event import MarketEvent

                # Adjust bid/ask to be relative to open price (not close)
                # This is critical for correct slippage calculation with spread-aware models
                if event.bid_price is not None and event.ask_price is not None and event.close is not None:
                    # Calculate offset: how much open differs from close
                    offset = event.open - event.close
                    # Shift bid/ask by same offset to maintain spread but center at open
                    adjusted_bid = event.bid_price + offset
                    adjusted_ask = event.ask_price + offset
                else:
                    # No bid/ask data or can't adjust - pass through unchanged
                    adjusted_bid = event.bid_price
                    adjusted_ask = event.ask_price

                fill_event = MarketEvent(
                    timestamp=event.timestamp,
                    asset_id=event.asset_id,
                    data_type=event.data_type,
                    price=event.open,
                    open=event.open,
                    high=event.high,
                    low=event.low,
                    close=event.open,  # Override close with open for next-bar fills
                    volume=event.volume,
                    bid_price=adjusted_bid,
                    ask_price=adjusted_ask,
                    signals=event.signals,
                    context=event.context,
                )
            else:
                fill_event = event

            fill_result = self.fill_simulator.try_fill_order(
                order,
                fill_event,
                current_cash=self._internal_portfolio.cash,
                current_position=self.get_position(asset_id),
            )
            if fill_result:
                # Note: order.update_fill() is already called by FillSimulator.try_fill_order()

                # Check position before fill (for exit tracking)
                position_before = self._internal_portfolio.get_position(asset_id)

                # Update position tracker (convert side to quantity_change)
                quantity_change = fill_result.fill_quantity if order.side == OrderSide.BUY else -fill_result.fill_quantity
                self._internal_portfolio.update_position(
                    asset_id,
                    quantity_change,
                    fill_result.fill_price,
                    fill_result.commission,
                    fill_result.slippage,
                )

                # Track exit timestamp if position went to zero (for allow_immediate_reentry=False)
                position_after = self._internal_portfolio.get_position(asset_id)
                if position_before != 0 and position_after == 0:
                    self._last_exit_time[asset_id] = event.timestamp

                # Update statistics
                self._total_commission += fill_result.commission
                self._total_slippage += fill_result.slippage
                self._fill_count += 1

                # Track trade (efficient - minimal overhead)
                self.trade_tracker.on_fill(fill_result.fill_event)

                fills.append(fill_result.fill_event)

                # Remove filled orders
                if order.is_filled:
                    self.order_router.remove_order(order)

                    # Handle bracket order completion
                    if order.order_type == OrderType.BRACKET:
                        self._handle_bracket_fill(order, fill_result.fill_event)
                    # Handle OCO (One-Cancels-Other) logic for bracket legs (delegates to BracketOrderManager)
                    # Only cancel sibling orders if this is a bracket leg (not the parent)
                    elif order.child_order_ids and order.metadata.get("bracket_type"):
                        self.bracket_manager.handle_oco_fill(order, self.cancel_order)

        # Check stop orders for triggering
        # IMPORTANT: Skip bracket SL orders - they're handled with priority logic below
        triggered_stops = []
        for order in list(self._stop_orders[asset_id]):
            # Skip bracket exits - they use priority-based processing
            if order.metadata.get("bracket_type") == "stop_loss":
                continue

            if self._should_trigger_stop(order, price):
                triggered_stops.append(order)
                self._stop_orders[asset_id].remove(order)

        # Process triggered stops immediately
        for order in triggered_stops:
            # Skip already-filled orders (prevent double-fill attempts)
            if order.remaining_quantity <= 0:
                continue

            if order.order_type == OrderType.STOP:
                # Keep as STOP so FillSimulator uses stop_price as base (not market_price)
                # This ensures SL fills at stop level, not at bar's extreme (low)
                order.metadata["original_type"] = "STOP"
                fill_result = self.fill_simulator.try_fill_order(
                    order,
                    event,
                    current_cash=self._internal_portfolio.cash,
                    current_position=self.get_position(asset_id),
                )
                if fill_result:
                    # Note: order.update_fill() is already called by FillSimulator.try_fill_order()

                    # Update position tracker (convert side to quantity_change)
                    quantity_change = fill_result.fill_quantity if order.side == OrderSide.BUY else -fill_result.fill_quantity
                    self._internal_portfolio.update_position(
                        asset_id,
                        quantity_change,
                        fill_result.fill_price,
                        fill_result.commission,
                        fill_result.slippage,
                    )
                    # Update statistics
                    self._total_commission += fill_result.commission
                    self._total_slippage += fill_result.slippage
                    self._fill_count += 1

                    fills.append(fill_result.fill_event)

                    # FIX: Handle OCO logic for bracket legs that are filled as stop orders (delegates to BracketOrderManager)
                    if (
                        order.is_filled
                        and order.child_order_ids
                        and order.metadata.get("bracket_type")
                    ):
                        self.bracket_manager.handle_oco_fill(order, self.cancel_order)

                else:
                    # If couldn't fill as market order, restore type and add to open orders
                    order.order_type = original_type
                    del order.metadata["original_type"]
                    self._open_orders[asset_id].append(order)
            elif order.order_type == OrderType.STOP_LIMIT:
                # Keep as limit order after triggering
                order.order_type = OrderType.LIMIT
                self._open_orders[asset_id].append(order)

        # Update and check trailing stops
        # VectorBT TSL behavior (TASK-007): Trail from PEAK price, not current price
        # Peak tracking uses metadata-based approach for now (short-term workaround)
        # IMPORTANT: Skip bracket TSL orders - they're handled with priority logic below
        triggered_trailing = []
        for order in list(self._trailing_stops[asset_id]):
            # Skip bracket exits - they use priority-based processing
            if order.metadata.get("bracket_type") == "trailing_stop":
                continue

            # Initialize peak tracking if not present
            if "peak_price" not in order.metadata:
                # Initialize peak to entry price (base price)
                order.metadata["peak_price"] = order.metadata.get("base_price", price)

            # Initialize TSL activation flag (for threshold support)
            if "tsl_activated" not in order.metadata:
                order.metadata["tsl_activated"] = False

            # VectorBT 4-stage per-bar process:
            # Capture peak before any updates (for trigger logic)
            peak_before_open = order.metadata["peak_price"]

            # Stage 1: Update peak with open
            peak_price = order.metadata["peak_price"]
            if event.open > peak_price:
                peak_price = event.open
                order.metadata["peak_price"] = peak_price

            # Check TSL threshold activation (if threshold specified)
            base_price = order.metadata.get("base_price", price)
            if order.tsl_threshold_pct is not None and not order.metadata["tsl_activated"]:
                # TSL only activates if peak exceeds base_price * (1 + threshold)
                threshold_price = base_price * (1.0 + order.tsl_threshold_pct)
                if peak_price >= threshold_price:
                    order.metadata["tsl_activated"] = True
            elif order.tsl_threshold_pct is None:
                # No threshold - TSL always active
                order.metadata["tsl_activated"] = True

            # Stage 2: Calculate TSL level (only if activated)
            if order.trail_percent is not None and order.metadata["tsl_activated"]:
                trail_amount = peak_price * (order.trail_percent / 100.0)
                if order.is_buy:
                    # BUY order (exiting short): TSL above peak
                    tsl_level = peak_price + trail_amount
                else:
                    # SELL order (exiting long): TSL below peak
                    tsl_level = peak_price - trail_amount
                order.trailing_stop_price = tsl_level

            # Stage 3: Update peak with high
            if event.high > peak_price:
                peak_price = event.high
                order.metadata["peak_price"] = peak_price

                # Re-check threshold activation with updated peak
                if order.tsl_threshold_pct is not None and not order.metadata["tsl_activated"]:
                    threshold_price = base_price * (1.0 + order.tsl_threshold_pct)
                    if peak_price >= threshold_price:
                        order.metadata["tsl_activated"] = True

            # Stage 4: Recalculate TSL with updated peak and check trigger (only if activated)
            # Rule: High-based peak updates can trigger ONLY if open already moved the peak
            # This prevents "ghost triggers" when bar opens flat but high spikes
            if order.trail_percent is not None and order.metadata["tsl_activated"]:
                trail_amount = peak_price * (order.trail_percent / 100.0)
                if order.is_buy:
                    # BUY order (exiting short): TSL above peak
                    tsl_level = peak_price + trail_amount
                else:
                    # SELL order (exiting long): TSL below peak
                    tsl_level = peak_price - trail_amount
                order.trailing_stop_price = tsl_level

                # Check trigger: Allow high-based peak to trigger ONLY if open already moved peak
                # If open didn't move peak, use pre-open TSL level for trigger check
                if not order.is_filled:
                    if event.open > peak_before_open:
                        # Open moved peak, so high update can trigger in same bar
                        if order.can_fill(price=price, high=event.high, low=event.low):
                            triggered_trailing.append(order)
                            self._trailing_stops[asset_id].remove(order)
                    else:
                        # Open didn't move peak, check trigger with pre-open TSL
                        if order.is_buy:
                            pre_open_tsl = peak_before_open + (peak_before_open * order.trail_percent / 100.0)
                        else:
                            pre_open_tsl = peak_before_open - (peak_before_open * order.trail_percent / 100.0)
                        # Manually check if low/high reached pre-open TSL
                        if order.is_buy:
                            if event.high >= pre_open_tsl:
                                triggered_trailing.append(order)
                                self._trailing_stops[asset_id].remove(order)
                        else:
                            if event.low <= pre_open_tsl:
                                triggered_trailing.append(order)
                                self._trailing_stops[asset_id].remove(order)

        # Process triggered trailing stops immediately
        for order in triggered_trailing:
            # Skip already-filled orders (prevent double-fill attempts)
            if order.remaining_quantity <= 0:
                continue

            # Keep as TRAILING_STOP so FillSimulator uses correct fill price logic
            # (FillSimulator._calculate_fill_price handles TRAILING_STOP specially)
            order.metadata["original_type"] = "TRAILING_STOP"

            # For next-bar fills (execution_delay=True), override close with open
            # Create a modified event if needed for next-bar execution
            if self.execution_delay:
                from ml4t.backtest.core.event import MarketEvent

                # Adjust bid/ask to be relative to open price (not close)
                # This is critical for correct slippage calculation with spread-aware models
                if event.bid_price is not None and event.ask_price is not None and event.close is not None:
                    # Calculate offset: how much open differs from close
                    offset = event.open - event.close
                    # Shift bid/ask by same offset to maintain spread but center at open
                    adjusted_bid = event.bid_price + offset
                    adjusted_ask = event.ask_price + offset
                else:
                    # No bid/ask data or can't adjust - pass through unchanged
                    adjusted_bid = event.bid_price
                    adjusted_ask = event.ask_price

                fill_event = MarketEvent(
                    timestamp=event.timestamp,
                    asset_id=event.asset_id,
                    data_type=event.data_type,
                    price=event.open,
                    open=event.open,
                    high=event.high,
                    low=event.low,
                    close=event.open,  # Override close with open for next-bar fills
                    volume=event.volume,
                    bid_price=adjusted_bid,
                    ask_price=adjusted_ask,
                    signals=event.signals,
                    context=event.context,
                )
            else:
                fill_event = event

            fill_result = self.fill_simulator.try_fill_order(
                order,
                fill_event,
                current_cash=self._internal_portfolio.cash,
                current_position=self.get_position(asset_id),
            )
            if fill_result:
                # Note: order.update_fill() is already called by FillSimulator.try_fill_order()

                # Check position before fill (for exit tracking)
                position_before = self._internal_portfolio.get_position(asset_id)

                # Update position tracker (convert side to quantity_change)
                quantity_change = fill_result.fill_quantity if order.side == OrderSide.BUY else -fill_result.fill_quantity
                self._internal_portfolio.update_position(
                    asset_id,
                    quantity_change,
                    fill_result.fill_price,
                    fill_result.commission,
                    fill_result.slippage,
                )

                # Track exit timestamp if position went to zero (for allow_immediate_reentry=False)
                position_after = self._internal_portfolio.get_position(asset_id)
                if position_before != 0 and position_after == 0:
                    self._last_exit_time[asset_id] = event.timestamp

                # Update statistics
                self._total_commission += fill_result.commission
                self._total_slippage += fill_result.slippage
                self._fill_count += 1

                # Track trade (efficient - minimal overhead)
                self.trade_tracker.on_fill(fill_result.fill_event)

                fills.append(fill_result.fill_event)

                # FIX: Handle OCO logic for bracket legs that are filled as trailing stops (delegates to BracketOrderManager)
                if order.is_filled and order.child_order_ids and order.metadata.get("bracket_type"):
                    self.bracket_manager.handle_oco_fill(order, self.cancel_order)

            else:
                # If couldn't fill as market order, restore type and add to open orders
                order.order_type = original_type
                del order.metadata["original_type"]
                self._open_orders[asset_id].append(order)

        # TASK-019: Process bracket exits with VectorBT priority (SL > TSL > TP)
        # Collect all triggered bracket exits without filling them
        triggered_bracket_exits = self._collect_triggered_bracket_exits(asset_id, event, price)

        # Fill ONLY the highest priority exit (if any triggered)
        if triggered_bracket_exits:
            priority, bracket_type, winning_order = triggered_bracket_exits[0]

            # Remove winning order from its queue
            if bracket_type == "take_profit":
                if winning_order in self._open_orders[asset_id]:
                    self._open_orders[asset_id].remove(winning_order)
            elif bracket_type == "stop_loss":
                if winning_order in self._stop_orders[asset_id]:
                    self._stop_orders[asset_id].remove(winning_order)
            elif bracket_type == "trailing_stop":
                if winning_order in self._trailing_stops[asset_id]:
                    self._trailing_stops[asset_id].remove(winning_order)

            # Only fill if order has remaining quantity (prevent double-fill attempts)
            if winning_order.remaining_quantity > 0:
                # Keep original order type for stop/trailing stop to get correct fill price
                # (FillSimulator._calculate_fill_price handles STOP and TRAILING_STOP specially)
                if bracket_type in ["stop_loss", "trailing_stop"]:
                    winning_order.metadata["original_type"] = winning_order.order_type.value

                # Fill the winning exit
                fill_result = self.fill_simulator.try_fill_order(
                    winning_order,
                    event,
                    current_cash=self._internal_portfolio.cash,
                    current_position=self.get_position(asset_id),
                )

                if fill_result:
                    # Note: order.update_fill() is already called by FillSimulator.try_fill_order()

                    # Check position before fill (for exit tracking)
                    position_before = self._internal_portfolio.get_position(asset_id)

                    # Update position tracker (convert side to quantity_change)
                    quantity_change = fill_result.fill_quantity if winning_order.side == OrderSide.BUY else -fill_result.fill_quantity
                    self._internal_portfolio.update_position(
                        asset_id,
                        quantity_change,
                        fill_result.fill_price,
                        fill_result.commission,
                        fill_result.slippage,
                    )

                    # Track exit timestamp if position went to zero
                    position_after = self._internal_portfolio.get_position(asset_id)
                    if position_before != 0 and position_after == 0:
                        self._last_exit_time[asset_id] = event.timestamp

                    # Update statistics
                    self._total_commission += fill_result.commission
                    self._total_slippage += fill_result.slippage
                    self._fill_count += 1

                    # Track trade
                    self.trade_tracker.on_fill(fill_result.fill_event)

                    fills.append(fill_result.fill_event)

                    # Handle OCO logic to cancel sibling bracket orders
                    if winning_order.is_filled and winning_order.child_order_ids and winning_order.metadata.get("bracket_type"):
                        self.bracket_manager.handle_oco_fill(winning_order, self.cancel_order)

                else:
                    # If couldn't fill, restore order type and add back to queue
                    if bracket_type in ["stop_loss", "trailing_stop"]:
                        winning_order.order_type = original_type
                        if "original_type" in winning_order.metadata:
                            del winning_order.metadata["original_type"]

                    if bracket_type == "take_profit":
                        self._open_orders[asset_id].append(winning_order)
                    elif bracket_type == "stop_loss":
                        self._stop_orders[asset_id].append(winning_order)
                    elif bracket_type == "trailing_stop":
                        self._trailing_stops[asset_id].append(winning_order)

        # Publish fill events to the event bus (if available)
        if hasattr(self, "event_bus") and self.event_bus:
            for fill_event in fills:
                self.event_bus.publish(fill_event)

        # Clear newly-created brackets set for next event (VectorBT: brackets now active)
        self._newly_created_brackets.clear()

        return fills

    def process_batch_fills(
        self,
        timestamp: datetime,
        market_data_map: dict[AssetId, MarketEvent],
    ) -> list[FillEvent]:
        """
        Process fills for all assets simultaneously in batch mode.

        This method fixes Critical Issues #2 and #3 from the architecture review:
        - Issue #2: Per-asset event processing prevents portfolio-level decisions
        - Issue #3: Execution delay timing causes duplicate orders

        Unlike on_market_event() which processes ONE asset at a time, this method:
        1. Moves pending → open for ALL assets with market data
        2. Fills ALL orders against their corresponding market events
        3. Enables simultaneous cross-asset rebalancing

        Args:
            timestamp: Current timestamp for the batch
            market_data_map: Dictionary mapping asset_id to MarketEvent for this timestamp

        Returns:
            List of FillEvent objects generated from all assets

        Example:
            # Engine groups events by timestamp
            timestamp = datetime(2024, 1, 15)
            market_map = {
                "AAPL": MarketEvent(...),
                "MSFT": MarketEvent(...),
                "GOOGL": MarketEvent(...),
            }
            fills = broker.process_batch_fills(timestamp, market_map)

        Note:
            This method should be called BEFORE strategy logic to ensure fills
            from previous orders are applied before new decisions are made.
        """
        # Increment bar counter once per batch
        self.trade_tracker.on_bar()

        print(f"\nBATCH FILLS @ {timestamp.strftime('%Y-%m-%d')}: {len(market_data_map)} assets with data, "
              f"{len(self.order_router.get_open_orders())} open orders")

        all_fills = []

        # Phase 1: Move pending → open for ALL assets with market data
        # This enables execution delay while processing all assets together
        newly_activated_orders = set()

        if self.execution_delay:
            for asset_id in market_data_map.keys():
                # CRITICAL FIX: Use OrderRouter's method instead of direct dict manipulation
                activated = self.order_router.activate_pending_orders(asset_id)
                for order in activated:
                    # Mark as newly activated (won't fill until next batch)
                    newly_activated_orders.add(order.order_id)

        # Phase 2: Process fills for ALL assets
        for asset_id, event in market_data_map.items():
            # Determine execution price (use open for next-bar fills)
            if self.execution_delay and event.open is not None:
                price = event.open  # Next bar's open (realistic for daily)
            elif event.close is not None:
                price = event.close
            elif event.price is not None:
                price = event.price
            else:
                continue  # No price available for this asset

            # Update last known price
            self._last_prices[asset_id] = price

            # Process open orders for this asset
            # CRITICAL FIX: Access orders through order_router, not broker's own dict
            orders_for_asset = list(self.order_router._open_orders[asset_id])
            if orders_for_asset:
                print(f"  Processing {len(orders_for_asset)} orders for {asset_id[:12]}")

            for order in orders_for_asset:
                if not order.is_active:
                    print(f"    SKIP: Order {order.order_id[:8]} not active (status={order.status})")
                    continue

                # Skip newly activated orders (execution delay)
                if self.execution_delay and order.order_id in newly_activated_orders:
                    print(f"    SKIP: Order {order.order_id[:8]} newly activated (execution delay)")
                    continue

                # Skip already-filled orders
                if order.remaining_quantity <= 0:
                    print(f"    SKIP: Order {order.order_id[:8]} already filled")
                    continue

                # Skip bracket exits (handled separately with priority logic)
                if order.metadata.get("bracket_type") in ["take_profit", "stop_loss", "trailing_stop"]:
                    print(f"    SKIP: Order {order.order_id[:8]} is bracket exit")
                    continue

                print(f"    ATTEMPTING FILL: Order {order.order_id[:8]} {order.side.name} {order.quantity}")

                # Create modified event for next-bar execution if needed
                if self.execution_delay and event.open is not None:
                    # For next-bar fills, override close with open
                    # This ensures we fill at next bar's open, not close
                    event_for_fill = MarketEvent(
                        timestamp=event.timestamp,
                        asset_id=event.asset_id,
                        data_type=event.data_type,
                        open=event.open,
                        high=event.high,
                        low=event.low,
                        close=event.open,  # Override: use open as execution price
                        volume=event.volume,
                        signals=event.signals,
                        context=event.context,
                    )
                else:
                    event_for_fill = event

                # Attempt fill using FillSimulator
                # NOTE: try_fill_order returns FillResult (not FillEvent), extract fill_event from it
                # Get current position quantity (Position object has .quantity attribute)
                position = self._internal_portfolio.positions.get(asset_id)
                current_position_qty = position.quantity if position is not None else 0.0

                fill_result = self.fill_simulator.try_fill_order(
                    order, event_for_fill,
                    current_cash=self._internal_portfolio.cash,
                    current_position=current_position_qty
                )

                if fill_result is not None:
                    fill_event = fill_result.fill_event
                    print(f"  FILL: {order.asset_id[:12]:<12} {order.side.name} {fill_event.fill_quantity:>6.0f} @ ${fill_event.fill_price:>7.2f} "
                          f"status={order.status.name}")

                    # CRITICAL FIX: Do NOT apply fill directly to portfolio here!
                    # Fills will be published to event bus and portfolio will receive via subscription.
                    # Applying twice causes float precision drift and crashes.
                    # self._internal_portfolio.on_fill_event(fill_event)  # REMOVED - let event bus handle

                    # Track statistics
                    self._fill_count += 1
                    self._total_commission += fill_result.commission
                    self._total_slippage += fill_result.slippage

                    # Handle bracket orders if parent filled
                    if fill_event.fill_quantity == order.quantity and order.order_type == OrderType.BRACKET:
                        self._handle_bracket_fill(order, fill_event)

                    # Record trade
                    self.trade_tracker.on_fill(fill_event)

                    all_fills.append(fill_event)

            # Phase 3: Process stop orders for this asset
            # (Similar logic to on_market_event, but in batch context)
            triggered_stops = []
            for order in list(self._stop_orders[asset_id]):
                if self._should_trigger_stop(order, price):
                    triggered_stops.append(order)
                    self._stop_orders[asset_id].remove(order)

                    # Convert to market/limit order based on original type
                    if order.order_type == OrderType.STOP:
                        order.order_type = OrderType.MARKET
                        self._open_orders[asset_id].append(order)
                    elif order.order_type == OrderType.STOP_LIMIT:
                        order.order_type = OrderType.LIMIT
                        self._open_orders[asset_id].append(order)

        # Clear newly-activated orders set for next batch
        self._newly_created_brackets.clear()

        # Dispatch fills to event bus (subscribers will apply them to portfolio)
        if hasattr(self, 'event_bus') and self.event_bus:
            for fill_event in all_fills:
                self.event_bus.dispatch_event(fill_event)

        return all_fills

    def _should_trigger_stop(self, order: Order, price: Price) -> bool:
        """Check if stop order should be triggered."""
        if order.stop_price is None:
            return False

        if order.is_buy:
            return price >= order.stop_price
        return price <= order.stop_price

    def _handle_bracket_fill(self, parent_order: Order, fill_event: FillEvent) -> None:
        """
        Handle completion of bracket order by creating stop-loss and take-profit orders.
        (Delegates to BracketOrderManager)

        Args:
            parent_order: The filled bracket order
            fill_event: The fill event that completed the order
        """
        leg_orders = self.bracket_manager.handle_bracket_fill(parent_order, fill_event)

        # Track parent-child relationship and mark as newly created (VectorBT: skip same-bar checking)
        if leg_orders:
            parent_order.child_order_ids.extend([o.order_id for o in leg_orders])
            # Add to newly-created set so they won't be checked until next bar
            for leg_order in leg_orders:
                self._newly_created_brackets.add(leg_order.order_id)


    def get_position(self, asset_id: AssetId) -> Quantity:
        """Get current position for an asset (delegates to PositionTracker)."""
        position = self._internal_portfolio.get_position(asset_id)
        return position.quantity if position else 0.0

    def get_positions(self) -> dict[AssetId, Quantity]:
        """Get all current positions (delegates to PositionTracker)."""
        return self._internal_portfolio.get_all_positions()

    def get_cash(self) -> float:
        """Get current cash balance (delegates to PositionTracker)."""
        return self._internal_portfolio.cash

    def adjust_position_quantity(self, asset_id: AssetId, new_quantity: Quantity) -> None:
        """
        Adjust position quantity directly (for corporate actions like stock splits).

        This method sets the position quantity to a new value while keeping the
        cost_basis unchanged. This is appropriate for stock splits where the total
        investment value remains the same but the number of shares changes.

        Updates both internal portfolio (for broker operations) and external portfolio
        (for strategy/reporting).

        Args:
            asset_id: Asset to adjust
            new_quantity: New position quantity

        Example:
            >>> # 2:1 stock split: 100 shares @ $100 avg cost → 200 shares @ $50 avg cost
            >>> broker.adjust_position_quantity("AAPL", 200)
        """
        if asset_id in self._internal_portfolio.positions:
            # Update internal portfolio (used by broker for fills)
            internal_position = self._internal_portfolio.positions[asset_id]
            old_quantity = internal_position.quantity
            internal_position.quantity = new_quantity

            # Round quantity if precision manager available
            if internal_position.precision_manager:
                internal_position.quantity = internal_position.precision_manager.round_quantity(
                    internal_position.quantity
                )

            logger.info(
                f"Corporate action: Adjusted position quantity for {asset_id} "
                f"from {old_quantity:.2f} to {internal_position.quantity:.2f}"
            )

            # Update unrealized P&L with current price
            if internal_position.last_price > 0:
                internal_position.update_price(internal_position.last_price)

            # Also update external portfolio (if initialized)
            if hasattr(self, 'portfolio') and self.portfolio is not None:
                if asset_id in self.portfolio.positions:
                    external_position = self.portfolio.positions[asset_id]
                    external_position.quantity = internal_position.quantity
                    # Update unrealized P&L
                    if external_position.last_price > 0:
                        external_position.update_price(external_position.last_price)

            # Remove position if quantity is zero or effectively zero
            is_zero = internal_position.quantity == 0
            if internal_position.precision_manager:
                is_zero = is_zero or internal_position.precision_manager.is_position_zero(
                    internal_position.quantity
                )
            if is_zero:
                del self._internal_portfolio.positions[asset_id]
                # Also remove from external portfolio
                if hasattr(self, 'portfolio') and self.portfolio is not None:
                    if asset_id in self.portfolio.positions:
                        del self.portfolio.positions[asset_id]
                logger.info(f"Position in {asset_id} closed due to zero quantity after corporate action")
        else:
            # If no existing position and new_quantity > 0, this is unexpected for corporate actions
            # Corporate actions should only modify existing positions
            if new_quantity > 0:
                logger.warning(
                    f"Corporate action attempted to create new position for {asset_id} "
                    f"with quantity {new_quantity}, but no existing position found. Ignoring."
                )

    def adjust_cash(self, amount: float) -> None:
        """
        Adjust cash balance (for corporate actions like cash dividends).

        This method adds or subtracts cash from the portfolio. Positive amounts
        add cash (e.g., dividend payments), negative amounts subtract cash
        (e.g., rights offering subscription).

        Updates both internal portfolio (for broker operations) and external portfolio
        (for strategy/reporting).

        Args:
            amount: Amount to add (positive) or subtract (negative)

        Example:
            >>> # Dividend payment: 100 shares × $2.50/share = $250
            >>> broker.adjust_cash(250.0)
        """
        old_cash = self._internal_portfolio.cash
        self._internal_portfolio.cash += amount

        # Round cash if precision manager available
        if self._internal_portfolio.precision_manager:
            self._internal_portfolio.cash = self._internal_portfolio.precision_manager.round_cash(
                self._internal_portfolio.cash
            )

        # Also update external portfolio (if initialized)
        if hasattr(self, 'portfolio') and self.portfolio is not None:
            self.portfolio.cash += amount
            if self.portfolio.precision_manager:
                self.portfolio.cash = self.portfolio.precision_manager.round_cash(self.portfolio.cash)

        logger.info(
            f"Corporate action: Adjusted cash from ${old_cash:.2f} to "
            f"${self._internal_portfolio.cash:.2f} (change: ${amount:+.2f})"
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get broker statistics."""
        return {
            "total_commission": self._total_commission,
            "total_slippage": self._total_slippage,
            "fill_count": self._fill_count,
            "open_orders": sum(len(orders) for orders in self._open_orders.values()),
            "stop_orders": sum(len(orders) for orders in self._stop_orders.values()),
        }

    def initialize(self, portfolio, event_bus) -> None:
        """Initialize broker with portfolio and event bus.

        Args:
            portfolio: Portfolio instance for position tracking
            event_bus: Event bus for publishing fill events
        """
        # CRITICAL FIX: Use the provided portfolio as _internal_portfolio to avoid dual-portfolio drift
        # This ensures broker's internal tracking and engine's portfolio are the SAME object
        self._internal_portfolio = portfolio
        self.portfolio = portfolio
        self.event_bus = event_bus
        logger.debug("SimulationBroker initialized")

    def on_order_event(self, event) -> None:
        """Handle order event from strategy.

        Args:
            event: OrderEvent to process
        """
        from ml4t.backtest.execution.order import Order

        # Get asset-specific PrecisionManager for this order
        asset_precision_manager = self._get_asset_precision_manager(event.asset_id)

        # Create Order object from OrderEvent with PrecisionManager
        order = Order(
            order_id=event.order_id,
            asset_id=event.asset_id,
            order_type=event.order_type,
            side=event.side,
            quantity=event.quantity,
            limit_price=getattr(event, "limit_price", None),
            stop_price=getattr(event, "stop_price", None),
            time_in_force=getattr(event, "time_in_force", None),
            precision_manager=asset_precision_manager,
        )

        # Submit the order with the event's timestamp
        self.submit_order(order, timestamp=event.timestamp)

    def finalize(self) -> None:
        """Finalize broker at end of backtest."""
        # Cancel all remaining open orders
        for asset_orders in self._open_orders.values():
            for order in list(asset_orders):
                if order.is_active:
                    order.cancel()

        for asset_orders in self._stop_orders.values():
            for order in list(asset_orders):
                if order.is_active:
                    order.cancel()

        logger.info(f"SimulationBroker finalized. Total fills: {self._fill_count}")

    def get_trades(self) -> Any:
        """Get all executed trades.

        Returns:
            DataFrame or list of trades
        """
        import polars as pl

        trades = []
        for order_id, order in self._orders.items():
            if order.filled_quantity > 0:
                trades.append(
                    {
                        "order_id": order_id,
                        "asset_id": order.asset_id,
                        "side": order.side.value,
                        "quantity": order.filled_quantity,
                        "price": order.average_fill_price,
                        "commission": order.commission,
                        "status": order.status.value,
                        "submitted_time": order.submitted_time,
                        "filled_time": order.filled_time,
                    },
                )

        if trades:
            return pl.DataFrame(trades)
        return pl.DataFrame()

    def reset(self) -> None:
        """Reset broker to initial state (delegates to components).

        Raises:
            RuntimeError: If reset() called before initialization
        """
        if not hasattr(self, "initial_cash"):
            raise RuntimeError(
                "SimulationBroker.reset() called before initialization. "
                "Call __init__() with initial_cash parameter first."
            )

        # Reset components
        self._internal_portfolio.reset()
        self.order_router.reset()
        self.bracket_manager.reset()
        self.fill_simulator.reset()

        # Reset local state
        self._last_prices.clear()
        self._total_commission = 0.0
        self._total_slippage = 0.0
        self._fill_count = 0

        logger.debug("SimulationBroker reset (all components)")
