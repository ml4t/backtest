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
        """Initialize simulation broker with configurable execution models."""
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

        from ml4t.backtest.core.precision import PrecisionManager
        cash_precision_manager = PrecisionManager(
            position_decimals=8, price_decimals=2, cash_decimals=2,
        )
        self._internal_portfolio = Portfolio(
            initial_cash=initial_cash,
            precision_manager=cash_precision_manager,
            track_analytics=False,
        )
        self.order_router = OrderRouter(execution_delay)
        self.bracket_manager = BracketOrderManager(self.submit_order)

        self.margin_account = MarginAccount(initial_cash, self.asset_registry) if enable_margin else None

        self.fill_simulator = FillSimulator(
            asset_registry=self.asset_registry,
            commission_model=self.commission_model,
            slippage_model=self.slippage_model,
            market_impact_model=self.market_impact_model,
            liquidity_model=self.liquidity_model,
            margin_account=self.margin_account,
            max_leverage=self.max_leverage,
        )

        self._last_prices: dict[AssetId, Price] = {}
        self._last_exit_time: dict[AssetId, datetime] = {}
        self._newly_created_brackets: set[str] = set()
        self.trade_tracker = TradeTracker(precision_manager=cash_precision_manager)
        self._total_commission = 0.0
        self._total_slippage = 0.0
        self._fill_count = 0

        logger.debug(
            f"SimulationBroker initialized: ${initial_cash:,.2f}, "
            f"delay={execution_delay}, margin={enable_margin}, leverage={max_leverage}"
        )

    @property
    def cash(self) -> float:
        """Get current cash balance."""
        return self._internal_portfolio.cash

    @cash.setter
    def cash(self, value: float) -> None:
        """Set cash balance."""
        self._internal_portfolio.cash = value

    @property
    def _open_orders(self) -> dict[AssetId, list[Order]]:
        """Get open orders from OrderRouter."""
        return self.order_router._open_orders

    @property
    def _stop_orders(self) -> dict[AssetId, list[Order]]:
        """Get stop orders from OrderRouter."""
        return self.order_router._stop_orders

    @property
    def _trailing_stops(self) -> dict[AssetId, list[Order]]:
        """Get trailing stops from OrderRouter."""
        return self.order_router._trailing_stops

    @property
    def _pending_orders(self) -> dict[AssetId, list[tuple[Order, datetime]]]:
        """Get pending orders from OrderRouter."""
        return self.order_router._pending_orders

    @property
    def _orders(self) -> dict[OrderId, Order]:
        """Get all orders from OrderRouter."""
        return self.order_router._orders

    @property
    def trades(self) -> "pl.DataFrame":
        """Get completed trades as a Polars DataFrame."""
        return self.trade_tracker.get_trades_df()

    def _get_asset_precision_manager(self, asset_id: AssetId) -> "PrecisionManager | None":
        """Get PrecisionManager for a specific asset."""
        asset_spec = self.asset_registry.get(asset_id)
        return asset_spec.get_precision_manager() if asset_spec else None

    def _create_fill_event(self, event: MarketEvent) -> MarketEvent:
        """Create MarketEvent for next-bar fills with adjusted bid/ask to open."""
        if not self.execution_delay or event.open is None:
            return event

        # Adjust bid/ask to be relative to open price
        adjusted_bid = event.bid_price
        adjusted_ask = event.ask_price
        if event.bid_price is not None and event.ask_price is not None and event.close is not None:
            offset = event.open - event.close
            adjusted_bid = event.bid_price + offset
            adjusted_ask = event.ask_price + offset

        return MarketEvent(
            timestamp=event.timestamp,
            asset_id=event.asset_id,
            data_type=event.data_type,
            price=event.open,
            open=event.open,
            high=event.high,
            low=event.low,
            close=event.open,  # Override: use open as execution price
            volume=event.volume,
            bid_price=adjusted_bid,
            ask_price=adjusted_ask,
            signals=event.signals,
            context=event.context,
        )

    def can_enter_position(self, asset_id: AssetId, current_time: datetime) -> bool:
        """Check if entry allowed (respects allow_immediate_reentry setting)."""
        if self.allow_immediate_reentry:
            return True

        # Check if we exited on this same bar
        if asset_id in self._last_exit_time:
            return self._last_exit_time[asset_id] != current_time

        return True

    def _update_trailing_stop(
        self,
        order: Order,
        event: MarketEvent,
        price: Price,
    ) -> tuple[bool, float | None]:
        """Update trailing stop state and check if triggered (VectorBT 4-stage process)."""
        if "peak_price" not in order.metadata:
            order.metadata["peak_price"] = order.metadata.get("base_price", price)
        if "tsl_activated" not in order.metadata:
            order.metadata["tsl_activated"] = False

        peak_before_open = order.metadata["peak_price"]
        peak_price = order.metadata["peak_price"]

        # Stage 1: Update peak with open
        if event.open > peak_price:
            peak_price = event.open
            order.metadata["peak_price"] = peak_price

        # Check threshold activation
        base_price = order.metadata.get("base_price", price)
        if order.tsl_threshold_pct is not None and not order.metadata["tsl_activated"]:
            if peak_price >= base_price * (1.0 + order.tsl_threshold_pct):
                order.metadata["tsl_activated"] = True
        elif order.tsl_threshold_pct is None:
            order.metadata["tsl_activated"] = True

        # Stage 2: Calculate TSL level
        if order.trail_percent is not None and order.metadata["tsl_activated"]:
            trail_amount = peak_price * (order.trail_percent / 100.0)
            order.trailing_stop_price = peak_price + trail_amount if order.is_buy else peak_price - trail_amount

        # Stage 3: Update peak with high
        if event.high > peak_price:
            peak_price = event.high
            order.metadata["peak_price"] = peak_price
            if order.tsl_threshold_pct is not None and not order.metadata["tsl_activated"]:
                if peak_price >= base_price * (1.0 + order.tsl_threshold_pct):
                    order.metadata["tsl_activated"] = True

        # Stage 4: Recalculate and check trigger
        triggered = False
        pre_open_tsl = None
        if order.trail_percent is not None and order.metadata["tsl_activated"]:
            trail_amount = peak_price * (order.trail_percent / 100.0)
            order.trailing_stop_price = peak_price + trail_amount if order.is_buy else peak_price - trail_amount

            if not order.is_filled:
                if event.open > peak_before_open:
                    triggered = order.can_fill(price=price, high=event.high, low=event.low)
                else:
                    if order.is_buy:
                        pre_open_tsl = peak_before_open + (peak_before_open * order.trail_percent / 100.0)
                        triggered = event.high >= pre_open_tsl
                    else:
                        pre_open_tsl = peak_before_open - (peak_before_open * order.trail_percent / 100.0)
                        triggered = event.low <= pre_open_tsl

        return triggered, pre_open_tsl

    def _execute_fill(
        self,
        order: Order,
        event: MarketEvent,
        asset_id: AssetId,
        fills: list,
    ) -> bool:
        """Execute a fill: update portfolio, stats, track trade, handle bracket/OCO."""
        fill_result = self.fill_simulator.try_fill_order(
            order, event,
            current_cash=self._internal_portfolio.cash,
            current_position=self.get_position(asset_id),
        )
        if not fill_result:
            return False

        # Update portfolio
        position_before = self._internal_portfolio.get_position(asset_id)
        qty_change = fill_result.fill_quantity if order.side == OrderSide.BUY else -fill_result.fill_quantity
        self._internal_portfolio.update_position(
            asset_id, qty_change, fill_result.fill_price,
            fill_result.commission, fill_result.slippage,
        )

        # Track exit timestamp
        if position_before != 0 and self._internal_portfolio.get_position(asset_id) == 0:
            self._last_exit_time[asset_id] = event.timestamp

        # Update statistics and track trade
        self._total_commission += fill_result.commission
        self._total_slippage += fill_result.slippage
        self._fill_count += 1
        self.trade_tracker.on_fill(fill_result.fill_event)
        fills.append(fill_result.fill_event)

        # Handle bracket completion
        if order.is_filled:
            if order.order_type == OrderType.BRACKET:
                self._handle_bracket_fill(order, fill_result.fill_event)
            elif order.child_order_ids and order.metadata.get("bracket_type"):
                self.bracket_manager.handle_oco_fill(order, self.cancel_order)

        return True

    def submit_order(self, order: Order, timestamp: datetime | None = None) -> OrderId:
        """Submit an order for execution."""
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
                from ml4t.backtest.core.event import MarketEvent
                fill_event = MarketEvent(
                    timestamp=timestamp,
                    asset_id=order.asset_id,
                    data_type="trade",
                    price=self._last_prices[order.asset_id],
                    close=self._last_prices[order.asset_id],
                )
                fill_result = self.fill_simulator.try_fill_order(
                    order, fill_event,
                    current_cash=self._internal_portfolio.cash,
                    current_position=self.get_position(order.asset_id),
                )
                if fill_result:
                    quantity_change = fill_result.fill_quantity if order.side == OrderSide.BUY else -fill_result.fill_quantity
                    self._internal_portfolio.update_position(
                        order.asset_id, quantity_change, fill_result.fill_price,
                        fill_result.commission, fill_result.slippage,
                    )
                    self._total_commission += fill_result.commission
                    self._total_slippage += fill_result.slippage
                    self._fill_count += 1
                    self.trade_tracker.on_fill(fill_result.fill_event)
                    if hasattr(self, "event_bus") and self.event_bus:
                        self.event_bus.publish(fill_result.fill_event)

                    # Remove filled orders and handle bracket completions (same as delayed path)
                    if order.is_filled:
                        self.order_router.remove_order(order)

                        # Handle bracket order completion
                        if order.order_type == OrderType.BRACKET:
                            self._handle_bracket_fill(order, fill_result.fill_event)

        logger.debug(f"Submitted order: {order}")
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

        # Move pending→open at start of event (orders fill on next bar, not T+2)
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

        # Determine execution price (open for next-bar fills, close otherwise)
        if self.execution_delay and event.open is not None:
            price = event.open
        elif event.close is not None:
            price = event.close
        elif event.price is not None:
            price = event.price
        else:
            return fills

        # Update last known price
        self._last_prices[asset_id] = price

        # Process open orders (skip bracket exits - handled with priority logic below)
        for order in list(self._open_orders[asset_id]):
            if not order.is_active or order.remaining_quantity <= 0:
                continue
            if self.execution_delay and order.order_id in newly_activated_orders:
                continue
            if order.metadata.get("bracket_type") in ["take_profit", "stop_loss", "trailing_stop"]:
                continue

            fill_event = self._create_fill_event(event)
            if self._execute_fill(order, fill_event, asset_id, fills):
                if order.is_filled:
                    self.order_router.remove_order(order)

        # Check and process stop orders (skip bracket SL - handled below)
        for order in list(self._stop_orders[asset_id]):
            if order.metadata.get("bracket_type") == "stop_loss":
                continue
            if not self._should_trigger_stop(order, price):
                continue
            self._stop_orders[asset_id].remove(order)
            if order.remaining_quantity <= 0:
                continue

            if order.order_type == OrderType.STOP:
                order.metadata["original_type"] = "STOP"
                if not self._execute_fill(order, event, asset_id, fills):
                    del order.metadata["original_type"]
                    self._open_orders[asset_id].append(order)
            elif order.order_type == OrderType.STOP_LIMIT:
                order.order_type = OrderType.LIMIT
                self._open_orders[asset_id].append(order)

        # Check and process trailing stops (skip bracket TSL - handled below)
        for order in list(self._trailing_stops[asset_id]):
            if order.metadata.get("bracket_type") == "trailing_stop":
                continue
            tsl_triggered, _ = self._update_trailing_stop(order, event, price)
            if not tsl_triggered:
                continue
            self._trailing_stops[asset_id].remove(order)
            if order.remaining_quantity <= 0:
                continue

            order.metadata["original_type"] = "TRAILING_STOP"
            fill_event = self._create_fill_event(event)
            if not self._execute_fill(order, fill_event, asset_id, fills):
                del order.metadata["original_type"]
                self._open_orders[asset_id].append(order)

        # TASK-019: Process bracket exits with VectorBT priority (SL > TSL > TP)
        triggered_bracket_exits = self.bracket_manager.collect_triggered_exits(
            asset_id, event, price,
            self._open_orders[asset_id], self._stop_orders[asset_id], self._trailing_stops[asset_id],
            self._newly_created_brackets, self._update_trailing_stop, self._should_trigger_stop
        )

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

            # Only fill if order has remaining quantity
            if winning_order.remaining_quantity > 0:
                # Mark original type for stop/trailing stop (for fill price calculation)
                if bracket_type in ["stop_loss", "trailing_stop"]:
                    winning_order.metadata["original_type"] = winning_order.order_type.value

                if not self._execute_fill(winning_order, event, asset_id, fills):
                    # Fill failed - restore order to queue
                    if bracket_type in ["stop_loss", "trailing_stop"]:
                        if "original_type" in winning_order.metadata:
                            del winning_order.metadata["original_type"]
                    if bracket_type == "take_profit":
                        self._open_orders[asset_id].append(winning_order)
                    elif bracket_type == "stop_loss":
                        self._stop_orders[asset_id].append(winning_order)
                    elif bracket_type == "trailing_stop":
                        self._trailing_stops[asset_id].append(winning_order)

        # Publish fill events for other subscribers (strategy, risk_manager)
        # NOTE: Portfolio does NOT subscribe to FILL events - broker already updated
        # positions directly during processing via _internal_portfolio.update_position()
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
        """Process fills for all assets simultaneously in batch mode.

        Unlike on_market_event() which processes ONE asset at a time, this processes
        all assets together for portfolio-level decisions. Call BEFORE strategy logic.
        """
        # Increment bar counter once per batch
        self.trade_tracker.on_bar()

        all_fills = []

        # Phase 1: Move pending → open for all assets
        newly_activated_orders = set()
        if self.execution_delay:
            for asset_id in market_data_map.keys():
                activated = self.order_router.activate_pending_orders(asset_id)
                for order in activated:
                    newly_activated_orders.add(order.order_id)

        # Phase 2: Process fills for all assets
        for asset_id, event in market_data_map.items():
            # Determine execution price
            if self.execution_delay and event.open is not None:
                price = event.open
            elif event.close is not None:
                price = event.close
            elif event.price is not None:
                price = event.price
            else:
                continue

            self._last_prices[asset_id] = price

            for order in list(self.order_router._open_orders[asset_id]):
                if not order.is_active or order.remaining_quantity <= 0:
                    continue
                if self.execution_delay and order.order_id in newly_activated_orders:
                    continue
                if order.metadata.get("bracket_type") in ["take_profit", "stop_loss", "trailing_stop"]:
                    continue

                # Create modified event for next-bar execution (override close with open)
                if self.execution_delay and event.open is not None:
                    event_for_fill = MarketEvent(
                        timestamp=event.timestamp,
                        asset_id=event.asset_id,
                        data_type=event.data_type,
                        open=event.open,
                        high=event.high,
                        low=event.low,
                        close=event.open,
                        volume=event.volume,
                        signals=event.signals,
                        context=event.context,
                    )
                else:
                    event_for_fill = event

                position = self._internal_portfolio.positions.get(asset_id)
                current_position_qty = position.quantity if position is not None else 0.0
                fill_result = self.fill_simulator.try_fill_order(
                    order, event_for_fill,
                    current_cash=self._internal_portfolio.cash,
                    current_position=current_position_qty
                )

                if fill_result is not None:
                    fill_event = fill_result.fill_event

                    # Apply fill to portfolio immediately (update cash for next fill)
                    self._internal_portfolio.on_fill_event(fill_event)

                    self._fill_count += 1
                    self._total_commission += fill_result.commission
                    self._total_slippage += fill_result.slippage

                    if fill_event.fill_quantity == order.quantity and order.order_type == OrderType.BRACKET:
                        self._handle_bracket_fill(order, fill_event)

                    self.trade_tracker.on_fill(fill_event)
                    all_fills.append(fill_event)

                    # Remove filled orders from queue
                    if order.remaining_quantity <= 0:
                        if order in self.order_router._open_orders[asset_id]:
                            self.order_router._open_orders[asset_id].remove(order)

            # Cancel unfillable SELL orders (zero position)
            current_position = self._internal_portfolio.get_position(asset_id)
            current_qty = current_position.quantity if current_position else 0.0
            if current_qty <= 0:
                unfillable = [o for o in self.order_router._open_orders[asset_id]
                              if o.is_sell and o.is_active]
                for order in unfillable:
                    order.cancel()
                    self.order_router._open_orders[asset_id].remove(order)

            # Phase 3: Process stop orders
            for order in list(self._stop_orders[asset_id]):
                if self._should_trigger_stop(order, price):
                    self._stop_orders[asset_id].remove(order)
                    if order.order_type == OrderType.STOP:
                        order.order_type = OrderType.MARKET
                        self._open_orders[asset_id].append(order)
                    elif order.order_type == OrderType.STOP_LIMIT:
                        order.order_type = OrderType.LIMIT
                        self._open_orders[asset_id].append(order)

        self._newly_created_brackets.clear()
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

    def get_pending_order_quantity(self, asset_id: AssetId) -> Quantity:
        """Get net pending order quantity for an asset.

        Returns the sum of pending BUY quantities minus pending SELL quantities.
        This represents the expected position change when pending orders fill.

        Used by order_target_percent to prevent duplicate orders when
        execution_delay is enabled.
        """
        pending_qty = 0.0

        # Check pending orders (not yet activated)
        for order, _ in self.order_router._pending_orders.get(asset_id, []):
            if order.is_active:
                delta = order.remaining_quantity if order.is_buy else -order.remaining_quantity
                pending_qty += delta

        # Check open orders (activated, waiting to fill)
        for order in self.order_router._open_orders.get(asset_id, []):
            if order.is_active:
                delta = order.remaining_quantity if order.is_buy else -order.remaining_quantity
                pending_qty += delta

        return pending_qty

    def adjust_position_quantity(self, asset_id: AssetId, new_quantity: Quantity) -> None:
        """Adjust position quantity for corporate actions (keeps cost_basis unchanged)."""
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
        """Adjust cash balance for corporate actions (positive=add, negative=subtract)."""
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

    def _get_exit_priority(self, bracket_type: str) -> int:
        """Get exit priority for bracket type (delegates to BracketOrderManager)."""
        return self.bracket_manager.get_exit_priority(bracket_type)

    def initialize(self, portfolio, event_bus) -> None:
        """Initialize broker with portfolio and event bus."""
        self._internal_portfolio = portfolio
        self.portfolio = portfolio
        self.event_bus = event_bus
        logger.debug("SimulationBroker initialized")

    def on_order_event(self, event) -> None:
        """Handle order event from strategy."""
        from ml4t.backtest.execution.order import Order

        asset_precision_manager = self._get_asset_precision_manager(event.asset_id)
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

    def reset(self) -> None:
        """Reset broker to initial state."""
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
