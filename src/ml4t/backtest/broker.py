"""Broker for order execution and position management."""

from datetime import datetime

from .models import CommissionModel, NoCommission, NoSlippage, SlippageModel
from .types import ExecutionMode, Fill, Order, OrderSide, OrderStatus, OrderType, Position, Trade


class Broker:
    """Broker interface - same for backtest and live trading."""

    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission_model: CommissionModel | None = None,
        slippage_model: SlippageModel | None = None,
        execution_mode: ExecutionMode = ExecutionMode.SAME_BAR,
        account_type: str = "cash",
        initial_margin: float = 0.5,
        maintenance_margin: float = 0.25,
    ):
        # Import accounting classes here to avoid circular imports
        from .accounting import (
            AccountState,
            CashAccountPolicy,
            Gatekeeper,
            MarginAccountPolicy,
        )

        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_model = commission_model or NoCommission()
        self.slippage_model = slippage_model or NoSlippage()
        self.execution_mode = execution_mode

        # Create AccountState with appropriate policy
        if account_type == "cash":
            policy = CashAccountPolicy()
        elif account_type == "margin":
            policy = MarginAccountPolicy(
                initial_margin=initial_margin, maintenance_margin=maintenance_margin
            )
        else:
            raise ValueError(f"Unknown account_type: '{account_type}'. Must be 'cash' or 'margin'")

        self.account = AccountState(initial_cash=initial_cash, policy=policy)
        self.account_type = account_type
        self.initial_margin = initial_margin
        self.maintenance_margin = maintenance_margin

        # Create Gatekeeper for order validation
        self.gatekeeper = Gatekeeper(self.account, self.commission_model)

        self.positions: dict[str, Position] = {}
        self.orders: list[Order] = []
        self.pending_orders: list[Order] = []
        self.fills: list[Fill] = []
        self.trades: list[Trade] = []
        self._order_counter = 0
        self._current_time: datetime | None = None
        self._current_prices: dict[str, float] = {}  # close prices
        self._current_opens: dict[str, float] = {}  # open prices for next-bar execution
        self._current_volumes: dict[str, float] = {}
        self._current_signals: dict[str, dict[str, float]] = {}
        self._orders_this_bar: list[Order] = []  # Orders placed this bar (for next-bar mode)

    def get_position(self, asset: str) -> Position | None:
        return self.positions.get(asset)

    def get_cash(self) -> float:
        return self.cash

    def get_account_value(self) -> float:
        value = self.cash
        for asset, pos in self.positions.items():
            price = self._current_prices.get(asset, pos.entry_price)
            value += pos.quantity * price
        return value

    def submit_order(
        self,
        asset: str,
        quantity: float,
        side: OrderSide | None = None,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        trail_amount: float | None = None,
    ) -> Order:
        if side is None:
            side = OrderSide.BUY if quantity > 0 else OrderSide.SELL
            quantity = abs(quantity)

        self._order_counter += 1
        order = Order(
            asset=asset,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            trail_amount=trail_amount,
            order_id=f"ORD-{self._order_counter}",
            created_at=self._current_time,
        )
        self.orders.append(order)
        self.pending_orders.append(order)

        # Track orders placed this bar for next-bar execution mode
        if self.execution_mode == ExecutionMode.NEXT_BAR:
            self._orders_this_bar.append(order)

        return order

    def submit_bracket(
        self,
        asset: str,
        quantity: float,
        take_profit: float,
        stop_loss: float,
        entry_type: OrderType = OrderType.MARKET,
        entry_limit: float | None = None,
    ) -> tuple[Order, Order, Order]:
        """Submit entry with take-profit and stop-loss."""
        entry = self.submit_order(asset, quantity, order_type=entry_type, limit_price=entry_limit)

        tp = self.submit_order(
            asset, quantity, OrderSide.SELL, OrderType.LIMIT, limit_price=take_profit
        )
        tp.parent_id = entry.order_id

        sl = self.submit_order(
            asset, quantity, OrderSide.SELL, OrderType.STOP, stop_price=stop_loss
        )
        sl.parent_id = entry.order_id

        return entry, tp, sl

    def update_order(self, order_id: str, **kwargs) -> bool:
        """Update pending order parameters (stop_price, limit_price, quantity, trail_amount)."""
        for order in self.pending_orders:
            if order.order_id == order_id:
                for key, value in kwargs.items():
                    if hasattr(order, key):
                        setattr(order, key, value)
                return True
        return False

    def cancel_order(self, order_id: str) -> bool:
        for order in self.pending_orders:
            if order.order_id == order_id:
                order.status = OrderStatus.CANCELLED
                self.pending_orders.remove(order)
                return True
        return False

    def close_position(self, asset: str) -> Order | None:
        pos = self.positions.get(asset)
        if pos and pos.quantity != 0:
            side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY
            return self.submit_order(asset, abs(pos.quantity), side)
        return None

    def get_order(self, order_id: str) -> Order | None:
        """Get order by ID."""
        for order in self.orders:
            if order.order_id == order_id:
                return order
        return None

    def get_pending_orders(self, asset: str | None = None) -> list[Order]:
        """Get pending orders, optionally filtered by asset."""
        if asset is None:
            return list(self.pending_orders)
        return [o for o in self.pending_orders if o.asset == asset]

    def _is_exit_order(self, order: Order) -> bool:
        """Check if order is an exit (reducing existing position).

        Exit orders are:
        - SELL when we have a long position (reducing long)
        - BUY when we have a short position (covering short)
        - Does NOT reverse the position

        Args:
            order: Order to check

        Returns:
            True if order is reducing an existing position, False otherwise
        """
        pos = self.positions.get(order.asset)
        if pos is None or pos.quantity == 0:
            return False  # No position, so this is entry, not exit

        # Calculate signed quantity delta
        signed_qty = order.quantity if order.side == OrderSide.BUY else -order.quantity

        # Check if opposite sign (reducing) and doesn't reverse
        if pos.quantity > 0 and signed_qty < 0:
            # Long position, sell order
            new_qty = pos.quantity + signed_qty
            return new_qty >= 0  # Exit if still long or flat, not reversal
        elif pos.quantity < 0 and signed_qty > 0:
            # Short position, buy order
            new_qty = pos.quantity + signed_qty
            return new_qty <= 0  # Exit if still short or flat, not reversal
        else:
            # Same sign - adding to position, not exiting
            return False

    def _update_time(
        self,
        timestamp: datetime,
        prices: dict[str, float],
        opens: dict[str, float],
        volumes: dict[str, float],
        signals: dict[str, dict],
    ):
        self._current_time = timestamp
        self._current_prices = prices
        self._current_opens = opens
        self._current_volumes = volumes
        self._current_signals = signals

        # In next-bar mode, move orders from this bar to pending for next bar
        if self.execution_mode == ExecutionMode.NEXT_BAR:
            # Orders placed last bar are now eligible for execution
            pass  # They're already in pending_orders
            # Clear orders placed this bar (will be processed next bar)
            self._orders_this_bar = []

        for pos in self.positions.values():
            pos.bars_held += 1

    def _process_orders(self, use_open: bool = False):
        """Process pending orders against current prices with exit-first sequencing.

        Exit-first sequencing ensures capital efficiency:
        1. Process all exit orders first (closing positions frees capital)
        2. Update account equity after exits
        3. Process all entry orders with updated buying power

        This prevents rejecting entry orders when we have pending exits that
        would free up capital.

        Args:
            use_open: If True, use open prices (for next-bar mode at bar start)
        """
        # Split orders into exits and entries
        exit_orders = []
        entry_orders = []

        for order in self.pending_orders[:]:
            # In next-bar mode, skip orders placed this bar
            if self.execution_mode == ExecutionMode.NEXT_BAR and order in self._orders_this_bar:
                continue

            if self._is_exit_order(order):
                exit_orders.append(order)
            else:
                entry_orders.append(order)

        filled_orders = []

        # Phase 1: Process exit orders (always allowed - frees capital)
        for order in exit_orders:
            # Get execution price based on mode
            if use_open and self.execution_mode == ExecutionMode.NEXT_BAR:
                price = self._current_opens.get(order.asset)
            else:
                price = self._current_prices.get(order.asset)

            if price is None:
                continue

            fill_price = self._check_fill(order, price)
            if fill_price is not None:
                self._execute_fill(order, fill_price)
                filled_orders.append(order)

        # Phase 2: Update account equity after exits
        self.account.mark_to_market(self._current_prices)

        # Phase 3: Process entry orders (validated via Gatekeeper)
        for order in entry_orders:
            # Get execution price based on mode
            if use_open and self.execution_mode == ExecutionMode.NEXT_BAR:
                price = self._current_opens.get(order.asset)
            else:
                price = self._current_prices.get(order.asset)

            if price is None:
                continue

            fill_price = self._check_fill(order, price)
            if fill_price is not None:
                # CRITICAL: Validate order before executing
                valid, rejection_reason = self.gatekeeper.validate_order(order, fill_price)

                if valid:
                    self._execute_fill(order, fill_price)
                    filled_orders.append(order)
                else:
                    # Reject order
                    order.status = OrderStatus.REJECTED
                    # Note: rejection_reason could be logged here if needed

        # Remove filled/rejected orders from pending
        for order in filled_orders:
            if order in self.pending_orders:
                self.pending_orders.remove(order)
            if order in self._orders_this_bar:
                self._orders_this_bar.remove(order)

        # Also remove rejected orders
        for order in self.pending_orders[:]:
            if order.status == OrderStatus.REJECTED:
                self.pending_orders.remove(order)

    def _check_fill(self, order: Order, price: float) -> float | None:
        """Check if order should fill, return fill price or None."""
        if order.order_type == OrderType.MARKET:
            return price
        elif order.order_type == OrderType.LIMIT:
            if (
                order.side == OrderSide.BUY
                and price <= order.limit_price
                or order.side == OrderSide.SELL
                and price >= order.limit_price
            ):
                return order.limit_price
        elif order.order_type == OrderType.STOP:
            if (
                order.side == OrderSide.BUY
                and price >= order.stop_price
                or order.side == OrderSide.SELL
                and price <= order.stop_price
            ):
                return price
        elif order.order_type == OrderType.TRAILING_STOP and order.side == OrderSide.SELL:
            new_stop = price - order.trail_amount
            if order.stop_price is None or new_stop > order.stop_price:
                order.stop_price = new_stop
            if price <= order.stop_price:
                return price
        return None

    def _execute_fill(self, order: Order, base_price: float):
        """Execute a fill and update positions."""
        volume = self._current_volumes.get(order.asset)

        # Calculate slippage
        slippage = self.slippage_model.calculate(order.asset, order.quantity, base_price, volume)
        fill_price = base_price + slippage if order.side == OrderSide.BUY else base_price - slippage

        # Calculate commission
        commission = self.commission_model.calculate(order.asset, order.quantity, fill_price)

        fill = Fill(
            order_id=order.order_id,
            asset=order.asset,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            timestamp=self._current_time,
            commission=commission,
            slippage=slippage,
        )
        self.fills.append(fill)

        order.status = OrderStatus.FILLED
        order.filled_at = self._current_time
        order.filled_price = fill_price
        order.filled_quantity = order.quantity

        # Update position
        pos = self.positions.get(order.asset)
        signed_qty = order.quantity if order.side == OrderSide.BUY else -order.quantity

        if pos is None:
            if signed_qty != 0:
                self.positions[order.asset] = Position(
                    asset=order.asset,
                    quantity=signed_qty,
                    entry_price=fill_price,
                    entry_time=self._current_time,
                )
        else:
            old_qty = pos.quantity
            new_qty = old_qty + signed_qty

            if new_qty == 0:
                # Position closed
                pnl = (fill_price - pos.entry_price) * old_qty - commission
                trade = Trade(
                    asset=order.asset,
                    entry_time=pos.entry_time,
                    exit_time=self._current_time,
                    entry_price=pos.entry_price,
                    exit_price=fill_price,
                    quantity=abs(old_qty),
                    pnl=pnl,
                    pnl_percent=(fill_price - pos.entry_price) / pos.entry_price
                    if pos.entry_price
                    else 0,
                    bars_held=pos.bars_held,
                    commission=commission,
                    slippage=slippage,
                    entry_signals=self._current_signals.get(order.asset, {}),
                    exit_signals=self._current_signals.get(order.asset, {}),
                )
                self.trades.append(trade)
                del self.positions[order.asset]
            elif (old_qty > 0) != (new_qty > 0):
                # Position flipped
                pnl = (fill_price - pos.entry_price) * old_qty - commission
                self.trades.append(
                    Trade(
                        asset=order.asset,
                        entry_time=pos.entry_time,
                        exit_time=self._current_time,
                        entry_price=pos.entry_price,
                        exit_price=fill_price,
                        quantity=abs(old_qty),
                        pnl=pnl,
                        pnl_percent=(fill_price - pos.entry_price) / pos.entry_price
                        if pos.entry_price
                        else 0,
                        bars_held=pos.bars_held,
                        commission=commission,
                        slippage=slippage,
                    )
                )
                self.positions[order.asset] = Position(
                    asset=order.asset,
                    quantity=new_qty,
                    entry_price=fill_price,
                    entry_time=self._current_time,
                )
            else:
                # Position scaled
                if abs(new_qty) > abs(old_qty):
                    total_cost = pos.entry_price * abs(old_qty) + fill_price * abs(signed_qty)
                    pos.entry_price = total_cost / abs(new_qty)
                pos.quantity = new_qty

        # Update cash
        cash_change = -signed_qty * fill_price - commission
        self.cash += cash_change

        # Update AccountState (import Position from accounting)
        from .accounting import Position as AcctPosition

        # Sync position to AccountState
        broker_pos = self.positions.get(order.asset)
        if broker_pos is None:
            # Position was closed, remove from account
            if order.asset in self.account.positions:
                del self.account.positions[order.asset]
        else:
            # Update or create position in account
            self.account.positions[order.asset] = AcctPosition(
                asset=broker_pos.asset,
                quantity=broker_pos.quantity,
                avg_entry_price=broker_pos.entry_price,
                current_price=self._current_prices.get(order.asset, broker_pos.entry_price),
                entry_time=broker_pos.entry_time,
                bars_held=broker_pos.bars_held,
            )

        # Update account cash
        self.account.cash = self.cash

        # Cancel sibling bracket orders on fill
        if order.parent_id:
            for o in self.pending_orders[:]:
                if o.parent_id == order.parent_id and o.order_id != order.order_id:
                    o.status = OrderStatus.CANCELLED
                    self.pending_orders.remove(o)
