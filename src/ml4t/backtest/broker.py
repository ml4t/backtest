"""Broker for order execution and position management."""

from datetime import datetime
from typing import TYPE_CHECKING, Any

from .config import InitialHwmSource, TrailHwmSource

if TYPE_CHECKING:
    from .accounting.policy import AccountPolicy
    from .execution import ExecutionLimits, MarketImpactModel

from .execution.fill_executor import FillExecutor
from .models import CommissionModel, NoCommission, NoSlippage, SlippageModel
from .types import (
    ContractSpec,
    ExecutionMode,
    ExitReason,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    StopFillMode,
    StopLevelBasis,
    Trade,
)


def _reason_to_exit_reason(reason: str) -> ExitReason:
    """Map reason string to ExitReason enum.

    Used when setting order._exit_reason from risk rule action.reason.

    Args:
        reason: Human-readable reason string (e.g., "stop_loss_5.0%")

    Returns:
        Corresponding ExitReason enum value
    """
    reason_lower = reason.lower()
    if "stop_loss" in reason_lower:
        return ExitReason.STOP_LOSS
    elif "take_profit" in reason_lower:
        return ExitReason.TAKE_PROFIT
    elif "trailing" in reason_lower:
        return ExitReason.TRAILING_STOP
    elif "time" in reason_lower:
        return ExitReason.TIME_STOP
    elif "end_of_data" in reason_lower:
        return ExitReason.END_OF_DATA
    else:
        return ExitReason.SIGNAL


class Broker:
    """Broker interface - same for backtest and live trading."""

    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission_model: CommissionModel | None = None,
        slippage_model: SlippageModel | None = None,
        stop_slippage_rate: float = 0.0,
        execution_mode: ExecutionMode = ExecutionMode.SAME_BAR,
        stop_fill_mode: StopFillMode = StopFillMode.STOP_PRICE,
        stop_level_basis: StopLevelBasis = StopLevelBasis.FILL_PRICE,
        trail_hwm_source: TrailHwmSource = TrailHwmSource.CLOSE,
        initial_hwm_source: InitialHwmSource = InitialHwmSource.FILL_PRICE,
        account_type: str = "cash",
        initial_margin: float = 0.5,
        long_maintenance_margin: float = 0.25,
        short_maintenance_margin: float = 0.30,
        fixed_margin_schedule: dict[str, tuple[float, float]] | None = None,
        execution_limits: "ExecutionLimits | None" = None,
        market_impact_model: "MarketImpactModel | None" = None,
        contract_specs: dict[str, ContractSpec] | None = None,
    ):
        # Runtime imports for accounting classes.
        # These are imported here rather than at module level because:
        # 1. The package __init__.py imports Broker, creating a potential import order issue
        # 2. TYPE_CHECKING block above provides type hints for static analysis
        # 3. This pattern allows mypy/pyright to validate types without runtime circular import
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
        self.stop_slippage_rate = stop_slippage_rate
        self.execution_mode = execution_mode
        self.stop_fill_mode = stop_fill_mode
        self.stop_level_basis = stop_level_basis
        self.trail_hwm_source = trail_hwm_source
        self.initial_hwm_source = initial_hwm_source

        # Create AccountState with appropriate policy
        policy: AccountPolicy
        if account_type == "cash":
            policy = CashAccountPolicy()
        elif account_type == "margin":
            policy = MarginAccountPolicy(
                initial_margin=initial_margin,
                long_maintenance_margin=long_maintenance_margin,
                short_maintenance_margin=short_maintenance_margin,
                fixed_margin_schedule=fixed_margin_schedule,
            )
        else:
            raise ValueError(f"Unknown account_type: '{account_type}'. Must be 'cash' or 'margin'")

        self.account = AccountState(initial_cash=initial_cash, policy=policy)
        self.account_type = account_type
        self.initial_margin = initial_margin
        self.long_maintenance_margin = long_maintenance_margin
        self.short_maintenance_margin = short_maintenance_margin
        self.fixed_margin_schedule = fixed_margin_schedule or {}

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
        self._position_rules: Any = None  # Global position rules
        self._position_rules_by_asset: dict[str, Any] = {}  # Per-asset rules
        self._pending_exits: dict[str, dict] = {}  # asset -> {reason, pct} for NEXT_BAR_OPEN mode

        # Execution model (volume limits and market impact)
        self.execution_limits = execution_limits  # ExecutionLimits instance
        self.market_impact_model = market_impact_model  # MarketImpactModel instance
        self._partial_orders: dict[str, float] = {}  # order_id -> remaining quantity
        self._filled_this_bar: set[str] = set()  # order_ids that had fills this bar

        # VBT Pro compatibility: prevent same-bar re-entry after stop exit
        self._stop_exits_this_bar: set[str] = set()  # assets that had stop exits this bar

        # VBT Pro compatibility: track positions created this bar
        # New positions should NOT have HWM updated from entry bar's high
        # VBT Pro uses CLOSE for initial HWM on entry bar, then updates from HIGH next bar
        self._positions_created_this_bar: set[str] = set()

        # Contract specifications (for futures and other derivatives)
        self._contract_specs: dict[str, ContractSpec] = contract_specs or {}

        # Fill execution (extracted from _execute_fill)
        self._fill_executor = FillExecutor(self)

    def get_contract_spec(self, asset: str) -> ContractSpec | None:
        """Get contract specification for an asset."""
        return self._contract_specs.get(asset)

    def get_multiplier(self, asset: str) -> float:
        """Get contract multiplier for an asset (1.0 for equities)."""
        spec = self._contract_specs.get(asset)
        return spec.multiplier if spec else 1.0

    def get_position(self, asset: str) -> Position | None:
        """Get the current position for an asset.

        Args:
            asset: Asset symbol

        Returns:
            Position object if position exists, None otherwise
        """
        return self.positions.get(asset)

    def get_positions(self) -> dict[str, Position]:
        """Get all current positions.

        Returns:
            Dictionary mapping asset symbols to Position objects
        """
        return self.positions

    def get_cash(self) -> float:
        """Get current cash balance.

        Returns:
            Current cash balance (can be negative for margin accounts)
        """
        return self.cash

    def get_account_value(self) -> float:
        """Calculate total account value (cash + position values)."""
        value = self.cash
        for asset, pos in self.positions.items():
            price = self._current_prices.get(asset, pos.entry_price)
            multiplier = self.get_multiplier(asset)
            value += pos.quantity * price * multiplier
        return value

    def get_rejected_orders(self, asset: str | None = None) -> list[Order]:
        """Get all rejected orders, optionally filtered by asset.

        Args:
            asset: If provided, filter to only this asset's rejected orders

        Returns:
            List of rejected Order objects with rejection_reason populated
        """
        rejected = [o for o in self.orders if o.status == OrderStatus.REJECTED]
        if asset is not None:
            rejected = [o for o in rejected if o.asset == asset]
        return rejected

    @property
    def last_rejection_reason(self) -> str | None:
        """Get reason for most recent order rejection.

        Returns:
            Rejection reason string, or None if no orders have been rejected
        """
        rejected = [o for o in self.orders if o.status == OrderStatus.REJECTED]
        return rejected[-1].rejection_reason if rejected else None

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
            initial_quantity=abs(int(pos.initial_quantity))
            if pos.initial_quantity
            else abs(int(pos.quantity)),
            unrealized_pnl=pos.unrealized_pnl(current_price),
            unrealized_return=pos.pnl_percent(current_price),
            bars_held=pos.bars_held,
            high_water_mark=pos.high_water_mark
            if pos.high_water_mark is not None
            else pos.entry_price,
            low_water_mark=pos.low_water_mark
            if pos.low_water_mark is not None
            else pos.entry_price,
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
                    order = self.submit_order(asset, -pos.quantity, order_type=OrderType.MARKET)
                    if order:
                        order._risk_exit_reason = action.reason
                        order._exit_reason = _reason_to_exit_reason(action.reason)
                        # Store fill price for stop/limit triggered exits
                        # This is the price at which the stop/limit was triggered
                        order._risk_fill_price = action.fill_price
                        exit_orders.append(order)
                        # VBT Pro compatibility: prevent same-bar re-entry
                        self._stop_exits_this_bar.add(asset)

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
                            order._exit_reason = _reason_to_exit_reason(action.reason)
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
    ) -> Order | None:
        """Submit a new order to the broker.

        Creates and queues an order for execution. Orders are validated by the
        Gatekeeper before fills to ensure account constraints are met.

        Args:
            asset: Asset symbol (e.g., "AAPL", "BTC-USD")
            quantity: Number of shares/units. Positive = buy, negative = sell
                     (if side is not specified)
            side: OrderSide.BUY or OrderSide.SELL. If None, inferred from quantity sign
            order_type: Order type (MARKET, LIMIT, STOP, TRAILING_STOP)
            limit_price: Limit price for LIMIT orders
            stop_price: Stop/trigger price for STOP orders
            trail_amount: Trail distance for TRAILING_STOP orders

        Returns:
            Order object if submitted successfully, None if rejected
            (e.g., same-bar re-entry after stop exit in VBT Pro mode)

        Examples:
            # Market buy
            order = broker.submit_order("AAPL", 100)

            # Market sell (using negative quantity)
            order = broker.submit_order("AAPL", -100)

            # Limit buy
            order = broker.submit_order("AAPL", 100, order_type=OrderType.LIMIT,
                                        limit_price=150.0)

            # Stop sell (stop-loss)
            order = broker.submit_order("AAPL", -100, order_type=OrderType.STOP,
                                        stop_price=145.0)
        """
        if side is None:
            side = OrderSide.BUY if quantity > 0 else OrderSide.SELL
            quantity = abs(quantity)

        # VBT Pro compatibility: prevent same-bar re-entry after stop exit
        # When a stop/trail exit occurs, don't allow new entry until next bar
        if side == OrderSide.BUY and asset in self._stop_exits_this_bar:
            return None  # Silently reject entry on same bar as stop exit

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
    ) -> tuple[Order, Order, Order] | None:
        """Submit entry with take-profit and stop-loss.

        Returns:
            Tuple of (entry_order, take_profit_order, stop_loss_order) or None if any fails.
        """
        entry = self.submit_order(asset, quantity, order_type=entry_type, limit_price=entry_limit)
        if entry is None:
            return None

        tp = self.submit_order(
            asset, quantity, OrderSide.SELL, OrderType.LIMIT, limit_price=take_profit
        )
        if tp is None:
            return None
        tp.parent_id = entry.order_id

        sl = self.submit_order(
            asset, quantity, OrderSide.SELL, OrderType.STOP, stop_price=stop_loss
        )
        if sl is None:
            return None
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
        """Close an open position for the given asset.

        Submits a market order to fully close the position.

        Args:
            asset: Asset symbol to close

        Returns:
            Order object if position exists and order submitted, None otherwise

        Example:
            # Close AAPL position
            order = broker.close_position("AAPL")
        """
        pos = self.positions.get(asset)
        if pos and pos.quantity != 0:
            side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY
            return self.submit_order(asset, abs(pos.quantity), side)
        return None

    def get_buying_power(self) -> float:
        """Get current buying power.

        Returns:
            Available buying power based on account policy:
            - Cash account: max(0, cash)
            - Margin account: (NLV - maintenance_margin) / initial_margin_rate
        """
        return self.account.buying_power

    def order_target_percent(
        self,
        asset: str,
        target_percent: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
    ) -> Order | None:
        """Order to achieve target portfolio weight.

        Calculates the order quantity needed to reach the target percentage
        of total portfolio value for this asset.

        Args:
            asset: Asset symbol
            target_percent: Target weight as decimal (0.10 = 10% of portfolio)
            order_type: Order type (default MARKET)
            limit_price: Limit price for LIMIT orders

        Returns:
            Submitted order, or None if no order needed or rejected

        Example:
            # Target 10% of portfolio in AAPL
            broker.order_target_percent("AAPL", 0.10)

            # Target 0% (close position)
            broker.order_target_percent("AAPL", 0.0)
        """
        if target_percent < -1.0 or target_percent > 1.0:
            # Allow up to 100% long or 100% short
            return None

        portfolio_value = self.get_account_value()
        if portfolio_value <= 0:
            return None

        price = self._current_prices.get(asset)
        if price is None or price <= 0:
            return None

        target_value = portfolio_value * target_percent
        return self._order_to_target_value(asset, target_value, price, order_type, limit_price)

    def order_target_value(
        self,
        asset: str,
        target_value: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
    ) -> Order | None:
        """Order to achieve target position value.

        Calculates the order quantity needed to reach the target dollar value
        for this position.

        Args:
            asset: Asset symbol
            target_value: Target position value in dollars (negative for short)
            order_type: Order type (default MARKET)
            limit_price: Limit price for LIMIT orders

        Returns:
            Submitted order, or None if no order needed or rejected

        Example:
            # Target $10,000 position in AAPL
            broker.order_target_value("AAPL", 10000)

            # Target short $5,000
            broker.order_target_value("AAPL", -5000)
        """
        price = self._current_prices.get(asset)
        if price is None or price <= 0:
            return None

        return self._order_to_target_value(asset, target_value, price, order_type, limit_price)

    def _order_to_target_value(
        self,
        asset: str,
        target_value: float,
        price: float,
        order_type: OrderType,
        limit_price: float | None,
    ) -> Order | None:
        """Internal helper to order toward a target value."""
        # Get current position value
        pos = self.positions.get(asset)
        current_value = 0.0
        if pos and pos.quantity != 0:
            current_value = pos.quantity * price

        # Calculate delta
        delta_value = target_value - current_value
        if abs(delta_value) < 0.01:  # Less than 1 cent, no trade needed
            return None

        # Convert to quantity
        delta_qty = delta_value / price

        # Submit order
        if delta_qty > 0:
            return self.submit_order(
                asset, delta_qty, OrderSide.BUY, order_type, limit_price=limit_price
            )
        elif delta_qty < 0:
            return self.submit_order(
                asset, abs(delta_qty), OrderSide.SELL, order_type, limit_price=limit_price
            )
        return None

    def rebalance_to_weights(
        self,
        target_weights: dict[str, float],
        order_type: OrderType = OrderType.MARKET,
    ) -> list[Order]:
        """Rebalance portfolio to target weights.

        Calculates orders needed to achieve target portfolio allocation.
        Processes sells before buys to free up capital.

        Args:
            target_weights: Dict of {asset: weight} where weights are decimals
                           (0.10 = 10%). Weights should sum to <= 1.0.
            order_type: Order type for all orders (default MARKET)

        Returns:
            List of submitted orders (may include None for rejected orders)

        Example:
            # Equal weight three stocks
            broker.rebalance_to_weights({
                "AAPL": 0.33,
                "GOOGL": 0.33,
                "MSFT": 0.34,
            })
        """
        portfolio_value = self.get_account_value()
        if portfolio_value <= 0:
            return []

        orders: list[Order] = []
        sells: list[tuple[str, float]] = []  # (asset, target_value)
        buys: list[tuple[str, float]] = []  # (asset, target_value)

        # Calculate target values and categorize as buys or sells
        for asset, weight in target_weights.items():
            price = self._current_prices.get(asset)
            if price is None or price <= 0:
                continue

            target_value = portfolio_value * weight

            pos = self.positions.get(asset)
            current_value = pos.quantity * price if pos and pos.quantity != 0 else 0.0

            delta = target_value - current_value
            if abs(delta) < 0.01:  # Less than 1 cent
                continue

            if delta < 0:
                sells.append((asset, target_value))
            else:
                buys.append((asset, target_value))

        # Also close positions not in target weights
        for asset, pos in self.positions.items():
            if pos.quantity != 0 and asset not in target_weights:
                sells.append((asset, 0.0))

        # Process sells first (frees capital for buys)
        for asset, target_value in sells:
            price = self._current_prices.get(asset)
            if price and price > 0:
                order = self._order_to_target_value(asset, target_value, price, order_type, None)
                if order:
                    orders.append(order)

        # Then process buys
        for asset, target_value in buys:
            price = self._current_prices.get(asset)
            if price and price > 0:
                order = self._order_to_target_value(asset, target_value, price, order_type, None)
                if order:
                    orders.append(order)

        return orders

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
                order._exit_reason = _reason_to_exit_reason(pending["reason"])
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
        self._stop_exits_this_bar.clear()  # VBT Pro: allow re-entry on next bar
        self._positions_created_this_bar.clear()  # VBT Pro: update HWM from next bar

        # In next-bar mode, move orders from this bar to pending for next bar
        if self.execution_mode == ExecutionMode.NEXT_BAR:
            # Orders placed last bar are now eligible for execution
            pass  # They're already in pending_orders
            # Clear orders placed this bar (will be processed next bar)
            self._orders_this_bar = []

        for _asset, pos in self.positions.items():
            pos.bars_held += 1
            # NOTE: Water marks are updated AFTER position rules are evaluated
            # via _update_water_marks(). VBT Pro evaluates trailing stops using
            # HWM from PREVIOUS bar, then updates HWM with current bar's high.

    def _update_water_marks(self):
        """Update water marks for all positions after position rules are evaluated.

        This must be called AFTER evaluate_position_rules() to match VBT Pro behavior:
        VBT Pro calculates trailing stop using HWM from previous bar, then updates HWM.

        CRITICAL VBT Pro behavior: For new positions, the entry bar's HIGH is NOT used
        to update HWM. VBT Pro uses CLOSE for initial HWM, then only starts updating
        from bar HIGHs on the NEXT bar after entry. This is because VBT Pro's vectorized
        calculation computes HWM as max(highs[entry_bar+1:current_bar+1]).
        """
        for asset, pos in self.positions.items():
            if asset in self._current_prices:
                # For new positions (created this bar), skip updating from entry bar's HIGH
                # VBT Pro only updates HWM from HIGHs starting on the bar AFTER entry
                is_new_position = asset in self._positions_created_this_bar
                pos.update_water_marks(
                    current_price=self._current_prices[asset],
                    bar_high=self._current_highs.get(asset),
                    bar_low=self._current_lows.get(asset),
                    use_high_for_hwm=(
                        self.trail_hwm_source == TrailHwmSource.HIGH and not is_new_position
                    ),
                )

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
                    # Reject order and store reason
                    order.status = OrderStatus.REJECTED
                    order.rejection_reason = rejection_reason

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

    def _check_gap_through(
        self, side: OrderSide, stop_price: float, bar_open: float
    ) -> float | None:
        """Check if bar gapped through stop level.

        If the bar opened beyond our stop level, we must fill at the open price
        (worse execution due to gap).

        Args:
            side: Order side (BUY or SELL)
            stop_price: The stop price level
            bar_open: The bar's opening price

        Returns:
            bar_open if gapped through, None if normal trigger
        """
        if side == OrderSide.SELL and bar_open <= stop_price:
            return bar_open  # Gapped down through stop
        elif side == OrderSide.BUY and bar_open >= stop_price:
            return bar_open  # Gapped up through stop
        return None

    def _check_market_fill(self, order: Order, price: float) -> float:
        """Check fill price for market order.

        For risk-triggered exits (stop-loss, take-profit), checks for gap-through
        scenarios where we must fill at worse price than the stop level.
        Also applies additional stop slippage if configured.

        Args:
            order: Market order to check
            price: Current market price (close)

        Returns:
            Fill price for the market order
        """
        risk_fill_price = getattr(order, "_risk_fill_price", None)
        if risk_fill_price is None:
            return price

        bar_open = self._current_opens.get(order.asset, price)
        gap_price = self._check_gap_through(order.side, risk_fill_price, bar_open)
        fill_price = gap_price if gap_price is not None else risk_fill_price

        # Apply additional stop slippage if configured
        # This models the reality that stop orders often fill at worse prices in fast markets
        if self.stop_slippage_rate > 0:
            if order.side == OrderSide.SELL:
                # Selling: slippage makes price worse (lower)
                fill_price = fill_price * (1 - self.stop_slippage_rate)
            else:
                # Buying (covering short): slippage makes price worse (higher)
                fill_price = fill_price * (1 + self.stop_slippage_rate)

        return fill_price

    def _check_limit_fill(self, order: Order, high: float, low: float) -> float | None:
        """Check if limit order should fill.

        Limit buy fills if price dipped to our level (Low <= limit).
        Limit sell fills if price rose to our level (High >= limit).

        Args:
            order: Limit order to check
            high: Bar high price
            low: Bar low price

        Returns:
            Limit price if order should fill, None otherwise
        """
        if order.limit_price is None:
            return None

        if (
            order.side == OrderSide.BUY
            and low <= order.limit_price
            or order.side == OrderSide.SELL
            and high >= order.limit_price
        ):
            return order.limit_price
        return None

    def _check_stop_fill(
        self, order: Order, high: float, low: float, bar_open: float
    ) -> float | None:
        """Check if stop order should fill.

        Stop buy triggers if price rose to trigger (High >= stop).
        Stop sell triggers if price fell to trigger (Low <= stop).
        Handles gap-through scenarios.

        Args:
            order: Stop order to check
            high: Bar high price
            low: Bar low price
            bar_open: Bar open price

        Returns:
            Fill price if triggered, None otherwise
        """
        if order.stop_price is None:
            return None

        triggered = False
        if (
            order.side == OrderSide.BUY
            and high >= order.stop_price
            or order.side == OrderSide.SELL
            and low <= order.stop_price
        ):
            triggered = True

        if not triggered:
            return None

        gap_price = self._check_gap_through(order.side, order.stop_price, bar_open)
        return gap_price if gap_price is not None else order.stop_price

    def _update_and_check_trailing_stop(
        self, order: Order, high: float, low: float, bar_open: float
    ) -> float | None:
        """Update trailing stop level and check if triggered.

        Updates the stop price based on the high water mark, then checks
        if the stop has been triggered.

        Note: This method mutates order.stop_price as trailing stops
        must track the high water mark.

        Args:
            order: Trailing stop order (must be SELL side)
            high: Bar high price
            low: Bar low price
            bar_open: Bar open price

        Returns:
            Fill price if triggered, None otherwise
        """
        if order.side != OrderSide.SELL:
            return None  # Only SELL trailing stops supported currently

        # Update trailing stop based on high water mark
        if order.trail_amount is not None:
            new_stop = high - order.trail_amount
            if order.stop_price is None or new_stop > order.stop_price:
                order.stop_price = new_stop

        # Check if triggered
        if order.stop_price is None or low > order.stop_price:
            return None

        gap_price = self._check_gap_through(order.side, order.stop_price, bar_open)
        return gap_price if gap_price is not None else order.stop_price

    def _check_fill(self, order: Order, price: float) -> float | None:
        """Check if order should fill, return fill price or None.

        Delegates to specialized methods based on order type:
        - Market orders: immediate fill, with gap-through handling for risk exits
        - Limit orders: fill if bar range touched limit price
        - Stop orders: fill at stop price (or worse) if bar triggered it
        - Trailing stops: update stop level, then check for trigger

        Args:
            order: Order to check
            price: Current market price (close)

        Returns:
            Fill price if order should fill, None otherwise
        """
        high = self._current_highs.get(order.asset, price)
        low = self._current_lows.get(order.asset, price)
        bar_open = self._current_opens.get(order.asset, price)

        if order.order_type == OrderType.MARKET:
            return self._check_market_fill(order, price)
        elif order.order_type == OrderType.LIMIT:
            return self._check_limit_fill(order, high, low)
        elif order.order_type == OrderType.STOP:
            return self._check_stop_fill(order, high, low, bar_open)
        elif order.order_type == OrderType.TRAILING_STOP:
            return self._update_and_check_trailing_stop(order, high, low, bar_open)

        return None

    def _execute_fill(self, order: Order, base_price: float) -> bool:
        """Execute a fill and update positions.

        This method delegates to FillExecutor for the actual implementation.
        See execution/fill_executor.py for the detailed logic.

        Args:
            order: Order to fill
            base_price: Base fill price before adjustments

        Returns:
            True if order is fully filled, False if partially filled (remainder pending)
        """
        return self._fill_executor.execute(order, base_price)
