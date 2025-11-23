"""Broker for order execution and position management."""

from datetime import datetime

from .models import CommissionModel, NoCommission, NoSlippage, SlippageModel
from .types import ContractSpec, ExecutionMode, Fill, Order, OrderSide, OrderStatus, OrderType, Position, StopFillMode, StopLevelBasis, Trade


class Broker:
    """Broker interface - same for backtest and live trading."""

    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission_model: CommissionModel | None = None,
        slippage_model: SlippageModel | None = None,
        execution_mode: ExecutionMode = ExecutionMode.SAME_BAR,
        stop_fill_mode: StopFillMode = StopFillMode.STOP_PRICE,
        stop_level_basis: StopLevelBasis = StopLevelBasis.FILL_PRICE,
        account_type: str = "cash",
        initial_margin: float = 0.5,
        maintenance_margin: float = 0.25,
        execution_limits=None,
        market_impact_model=None,
        contract_specs: dict[str, ContractSpec] | None = None,
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
        self.stop_fill_mode = stop_fill_mode
        self.stop_level_basis = stop_level_basis

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
        self._current_highs: dict[str, float] = {}  # high prices for limit/stop checks
        self._current_lows: dict[str, float] = {}  # low prices for limit/stop checks
        self._current_volumes: dict[str, float] = {}
        self._current_signals: dict[str, dict[str, float]] = {}
        self._orders_this_bar: list[Order] = []  # Orders placed this bar (for next-bar mode)

        # Risk management
        self._position_rules = None  # Global position rules
        self._position_rules_by_asset: dict[str, any] = {}  # Per-asset rules
        self._pending_exits: dict[str, dict] = {}  # asset -> {reason, pct} for NEXT_BAR_OPEN mode

        # Execution model (volume limits and market impact)
        self.execution_limits = execution_limits  # ExecutionLimits instance
        self.market_impact_model = market_impact_model  # MarketImpactModel instance
        self._partial_orders: dict[str, float] = {}  # order_id -> remaining quantity
        self._filled_this_bar: set[str] = set()  # order_ids that had fills this bar

        # Contract specifications (for futures and other derivatives)
        self._contract_specs: dict[str, ContractSpec] = contract_specs or {}

    def get_contract_spec(self, asset: str) -> ContractSpec | None:
        """Get contract specification for an asset."""
        return self._contract_specs.get(asset)

    def get_multiplier(self, asset: str) -> float:
        """Get contract multiplier for an asset (1.0 for equities)."""
        spec = self._contract_specs.get(asset)
        return spec.multiplier if spec else 1.0

    def get_position(self, asset: str) -> Position | None:
        return self.positions.get(asset)

    def get_cash(self) -> float:
        return self.cash

    def get_account_value(self) -> float:
        """Calculate total account value (cash + position values)."""
        value = self.cash
        for asset, pos in self.positions.items():
            price = self._current_prices.get(asset, pos.entry_price)
            multiplier = self.get_multiplier(asset)
            value += pos.quantity * price * multiplier
        return value

    # === Risk Management ===

    def set_position_rules(self, rules, asset: str | None = None) -> None:
        """Set position rules globally or per-asset.

        Args:
            rules: PositionRule or RuleChain to apply
            asset: If provided, apply only to this asset; otherwise global
        """
        if asset:
            self._position_rules_by_asset[asset] = rules
        else:
            self._position_rules = rules

    def update_position_context(self, asset: str, context: dict) -> None:
        """Update context data for a position (used by signal-based rules).

        Args:
            asset: Asset symbol
            context: Dict of signal/indicator values (e.g., {'exit_signal': -0.5, 'atr': 2.5})
        """
        pos = self.positions.get(asset)
        if pos:
            pos.context.update(context)

    def _get_position_rules(self, asset: str):
        """Get applicable rules for an asset (per-asset or global)."""
        return self._position_rules_by_asset.get(asset) or self._position_rules

    def _build_position_state(self, pos: Position, current_price: float):
        """Build PositionState from Position for rule evaluation."""
        # Import here to avoid circular imports
        from .risk.types import PositionState

        asset = pos.asset

        # Merge stop configuration into context for rules to access
        context = {
            **pos.context,
            "stop_fill_mode": self.stop_fill_mode,
            "stop_level_basis": self.stop_level_basis,
        }

        return PositionState(
            asset=asset,
            side=pos.side,
            entry_price=pos.entry_price,
            current_price=current_price,
            quantity=abs(int(pos.quantity)),
            initial_quantity=abs(int(pos.initial_quantity)) if pos.initial_quantity else abs(int(pos.quantity)),
            unrealized_pnl=pos.unrealized_pnl(current_price),
            unrealized_return=pos.pnl_percent(current_price),
            bars_held=pos.bars_held,
            high_water_mark=pos.high_water_mark,
            low_water_mark=pos.low_water_mark,
            # Bar OHLC for intrabar stop/limit detection
            bar_open=self._current_opens.get(asset),
            bar_high=self._current_highs.get(asset),
            bar_low=self._current_lows.get(asset),
            max_favorable_excursion=pos.max_favorable_excursion,
            max_adverse_excursion=pos.max_adverse_excursion,
            entry_time=pos.entry_time,
            current_time=self._current_time,
            context=context,
        )

    def evaluate_position_rules(self) -> list[Order]:
        """Evaluate position rules for all open positions.

        Called by Engine before processing orders. Returns list of exit orders.
        Handles defer_fill=True by storing pending exits for next bar.
        """
        from .risk.types import ActionType

        exit_orders = []

        for asset, pos in list(self.positions.items()):
            rules = self._get_position_rules(asset)
            if rules is None:
                continue

            price = self._current_prices.get(asset)
            if price is None:
                continue

            # Build state and evaluate
            state = self._build_position_state(pos, price)
            action = rules.evaluate(state)

            if action.action == ActionType.EXIT_FULL:
                if action.defer_fill:
                    # NEXT_BAR_OPEN mode: defer exit to next bar
                    # Store pending exit info (will be processed at next bar's open)
                    self._pending_exits[asset] = {
                        "reason": action.reason,
                        "pct": 1.0,
                        "quantity": pos.quantity,
                    }
                else:
                    # Generate full exit order immediately
                    side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY
                    order = self.submit_order(asset, -pos.quantity, order_type=OrderType.MARKET)
                    if order:
                        order._risk_exit_reason = action.reason
                        # Store fill price for stop/limit triggered exits
                        # This is the price at which the stop/limit was triggered
                        order._risk_fill_price = action.fill_price
                        exit_orders.append(order)

            elif action.action == ActionType.EXIT_PARTIAL:
                if action.defer_fill:
                    # NEXT_BAR_OPEN mode: defer partial exit to next bar
                    exit_qty = int(abs(pos.quantity) * action.pct)
                    if exit_qty > 0:
                        self._pending_exits[asset] = {
                            "reason": action.reason,
                            "pct": action.pct,
                            "quantity": exit_qty if pos.quantity > 0 else -exit_qty,
                        }
                else:
                    # Generate partial exit order immediately
                    exit_qty = int(abs(pos.quantity) * action.pct)
                    if exit_qty > 0:
                        actual_qty = -exit_qty if pos.quantity > 0 else exit_qty
                        order = self.submit_order(asset, actual_qty, order_type=OrderType.MARKET)
                        if order:
                            order._risk_exit_reason = action.reason
                            order._risk_fill_price = action.fill_price
                            exit_orders.append(order)

        return exit_orders

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

        # Capture signal price (close at order time) for stop level calculation
        # This is used when stop_level_basis is SIGNAL_PRICE (Backtrader behavior)
        order._signal_price = self._current_prices.get(asset)

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

    def _process_pending_exits(self) -> list[Order]:
        """Process pending exits from NEXT_BAR_OPEN mode.

        Called at the start of a new bar to fill deferred exits at open price.
        Returns list of exit orders that were created and will be filled at open.
        """
        exit_orders = []

        for asset, pending in list(self._pending_exits.items()):
            pos = self.positions.get(asset)
            if pos is None:
                # Position no longer exists (shouldn't happen normally)
                del self._pending_exits[asset]
                continue

            open_price = self._current_opens.get(asset)
            if open_price is None:
                # No open price available, skip this bar
                continue

            # Create exit order with fill price = open price
            exit_qty = pending["quantity"]
            order = self.submit_order(asset, -exit_qty, order_type=OrderType.MARKET)
            if order:
                order._risk_exit_reason = pending["reason"]
                # Fill at current bar's open (this is the "next bar" from when stop triggered)
                order._risk_fill_price = open_price
                exit_orders.append(order)

            # Remove from pending
            del self._pending_exits[asset]

        return exit_orders

    def _update_time(
        self,
        timestamp: datetime,
        prices: dict[str, float],
        opens: dict[str, float],
        highs: dict[str, float],
        lows: dict[str, float],
        volumes: dict[str, float],
        signals: dict[str, dict],
    ):
        self._current_time = timestamp
        self._current_prices = prices
        self._current_opens = opens
        self._current_highs = highs
        self._current_lows = lows
        self._current_volumes = volumes
        self._current_signals = signals

        # Clear per-bar tracking at start of new bar
        self._filled_this_bar.clear()

        # In next-bar mode, move orders from this bar to pending for next bar
        if self.execution_mode == ExecutionMode.NEXT_BAR:
            # Orders placed last bar are now eligible for execution
            pass  # They're already in pending_orders
            # Clear orders placed this bar (will be processed next bar)
            self._orders_this_bar = []

        for asset, pos in self.positions.items():
            pos.bars_held += 1
            # Update water marks for risk tracking
            if asset in prices:
                pos.update_water_marks(prices[asset])

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
                fully_filled = self._execute_fill(order, fill_price)
                if fully_filled:
                    filled_orders.append(order)
                    # Clean up partial tracking
                    self._partial_orders.pop(order.order_id, None)
                else:
                    # Update order quantity to remaining
                    self._update_partial_order(order)

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
                    fully_filled = self._execute_fill(order, fill_price)
                    if fully_filled:
                        filled_orders.append(order)
                        # Clean up partial tracking
                        self._partial_orders.pop(order.order_id, None)
                    else:
                        # Update order quantity to remaining
                        self._update_partial_order(order)
                else:
                    # Reject order
                    order.status = OrderStatus.REJECTED
                    # Note: rejection_reason could be logged here if needed

        # Remove filled/rejected orders from pending (only fully filled ones)
        for order in filled_orders:
            if order in self.pending_orders:
                self.pending_orders.remove(order)
            if order in self._orders_this_bar:
                self._orders_this_bar.remove(order)

        # Also remove rejected orders
        for order in self.pending_orders[:]:
            if order.status == OrderStatus.REJECTED:
                self.pending_orders.remove(order)

    def _get_effective_quantity(self, order: Order) -> float:
        """Get effective order quantity (considering partial fills).

        For orders with partial fills in progress, returns the remaining quantity.
        """
        remaining = self._partial_orders.get(order.order_id)
        if remaining is not None:
            return remaining
        return order.quantity

    def _update_partial_order(self, order: Order) -> None:
        """Update order quantity after partial fill for next bar."""
        remaining = self._partial_orders.get(order.order_id)
        if remaining is not None:
            order.quantity = remaining

    def _check_fill(self, order: Order, price: float) -> float | None:
        """Check if order should fill, return fill price or None.

        Uses High/Low data to properly check if limit/stop prices were traded through.
        For limit orders: fill at limit price if bar's range touched it
        For stop orders: fill at stop price (or worse) if bar's range triggered it
        For risk-triggered market orders: fill at the stop/target price (intrabar execution)
        """
        high = self._current_highs.get(order.asset, price)
        low = self._current_lows.get(order.asset, price)

        if order.order_type == OrderType.MARKET:
            # Check if this is a risk-triggered exit with a specific fill price
            # (e.g., stop-loss or take-profit that was triggered intrabar)
            risk_fill_price = getattr(order, "_risk_fill_price", None)
            if risk_fill_price is not None:
                # Use the stop/target price as the base fill price
                # Slippage will be applied on top by the caller
                return risk_fill_price
            return price

        elif order.order_type == OrderType.LIMIT:
            # Limit buy fills if Low <= limit_price (price dipped to our level)
            # Limit sell fills if High >= limit_price (price rose to our level)
            if order.side == OrderSide.BUY and low <= order.limit_price or order.side == OrderSide.SELL and high >= order.limit_price:
                return order.limit_price

        elif order.order_type == OrderType.STOP:
            # Stop buy triggers if High >= stop_price (price rose to trigger)
            # Stop sell triggers if Low <= stop_price (price fell to trigger)
            if order.side == OrderSide.BUY and high >= order.stop_price:
                # Fill at stop price or worse (could gap through)
                return max(order.stop_price, low)  # At least stop price
            elif order.side == OrderSide.SELL and low <= order.stop_price:
                # Fill at stop price or worse (could gap through)
                return min(order.stop_price, high)  # At most stop price

        elif order.order_type == OrderType.TRAILING_STOP and order.side == OrderSide.SELL:
            # Update trailing stop based on high water mark
            new_stop = high - order.trail_amount
            if order.stop_price is None or new_stop > order.stop_price:
                order.stop_price = new_stop
            # Check if triggered
            if low <= order.stop_price:
                return min(order.stop_price, high)

        return None

    def _execute_fill(self, order: Order, base_price: float) -> bool:
        """Execute a fill and update positions.

        Returns:
            True if order is fully filled, False if partially filled (remainder pending)
        """
        volume = self._current_volumes.get(order.asset)

        # Get effective quantity (considering partial fills from previous bars)
        effective_quantity = self._get_effective_quantity(order)
        fill_quantity = effective_quantity

        # Apply execution limits (volume participation)
        if self.execution_limits is not None:
            # Skip if already filled this bar (for volume limits)
            if order.order_id in self._filled_this_bar:
                return False

            exec_result = self.execution_limits.calculate(effective_quantity, volume, base_price)
            fill_quantity = exec_result.fillable_quantity

            if fill_quantity <= 0:
                # Can't fill any this bar - keep order pending
                return False

            # Mark as filled this bar (to prevent double fills in same_bar mode)
            self._filled_this_bar.add(order.order_id)

            if exec_result.remaining_quantity > 0:
                # Partial fill - track remaining for future bars
                self._partial_orders[order.order_id] = exec_result.remaining_quantity
            else:
                # Fully filled - remove from partial tracking
                self._partial_orders.pop(order.order_id, None)

        # Apply market impact
        impact = 0.0
        if self.market_impact_model is not None:
            is_buy = order.side == OrderSide.BUY
            impact = self.market_impact_model.calculate(fill_quantity, base_price, volume, is_buy)
            base_price = base_price + impact  # Impact adjusts the base price

        # Calculate slippage (on top of market impact)
        slippage = self.slippage_model.calculate(order.asset, fill_quantity, base_price, volume)
        fill_price = base_price + slippage if order.side == OrderSide.BUY else base_price - slippage

        # Calculate commission
        commission = self.commission_model.calculate(order.asset, fill_quantity, fill_price)

        fill = Fill(
            order_id=order.order_id,
            asset=order.asset,
            side=order.side,
            quantity=fill_quantity,
            price=fill_price,
            timestamp=self._current_time,
            commission=commission,
            slippage=slippage,
        )
        self.fills.append(fill)

        # Determine if order is fully filled or partial
        is_partial = order.order_id in self._partial_orders
        if is_partial:
            # Update order for partial fill (will continue next bar)
            order.filled_quantity = (order.filled_quantity or 0) + fill_quantity
            # Don't change status - still pending for remainder
        else:
            order.status = OrderStatus.FILLED
            order.filled_at = self._current_time
            order.filled_price = fill_price
            order.filled_quantity = fill_quantity

        # Update position
        pos = self.positions.get(order.asset)
        signed_qty = fill_quantity if order.side == OrderSide.BUY else -fill_quantity

        if pos is None:
            if signed_qty != 0:
                # Create new position with signal_price in context for stop level calculation
                signal_price = getattr(order, "_signal_price", None)
                context = {"signal_price": signal_price} if signal_price is not None else {}
                self.positions[order.asset] = Position(
                    asset=order.asset,
                    quantity=signed_qty,
                    entry_price=fill_price,
                    entry_time=self._current_time,
                    context=context,
                    multiplier=self.get_multiplier(order.asset),
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
                    quantity=old_qty,  # Preserve sign: positive=long, negative=short
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
                        quantity=old_qty,  # Preserve sign: positive=long, negative=short
                        pnl=pnl,
                        pnl_percent=(fill_price - pos.entry_price) / pos.entry_price
                        if pos.entry_price
                        else 0,
                        bars_held=pos.bars_held,
                        commission=commission,
                        slippage=slippage,
                    )
                )
                # Create new position with signal_price in context
                signal_price = getattr(order, "_signal_price", None)
                context = {"signal_price": signal_price} if signal_price is not None else {}
                self.positions[order.asset] = Position(
                    asset=order.asset,
                    quantity=new_qty,
                    entry_price=fill_price,
                    entry_time=self._current_time,
                    context=context,
                    multiplier=self.get_multiplier(order.asset),
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

        # Cancel sibling bracket orders on fill (only for full fills)
        if order.parent_id and not is_partial:
            for o in self.pending_orders[:]:
                if o.parent_id == order.parent_id and o.order_id != order.order_id:
                    o.status = OrderStatus.CANCELLED
                    self.pending_orders.remove(o)

        # Return True for full fill, False for partial (order stays pending)
        return not is_partial
