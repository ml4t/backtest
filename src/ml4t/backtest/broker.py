"""Broker for order execution and position management."""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .config import (
    FillOrdering,
    InitialHwmSource,
    ShareType,
    StatsConfig,
    TrailStopTiming,
    WaterMarkSource,
)
from .core import (
    ExecutionEngine,
    OrderBook,
    PortfolioLedger,
    RiskEngine,
    SubmitOrderOptions,
)
from .execution.fill_executor import FillExecutor
from .models import CommissionModel, NoCommission, NoSlippage, SlippageModel
from .types import (
    AssetTradingStats,
    ContractSpec,
    ExecutionMode,
    Fill,
    Order,
    OrderSide,
    OrderType,
    Position,
    StopFillMode,
    StopLevelBasis,
    Trade,
)

if TYPE_CHECKING:
    from .accounting.policy import AccountPolicy
    from .config import BacktestConfig
    from .execution import ExecutionLimits, MarketImpactModel


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
        trail_hwm_source: WaterMarkSource = WaterMarkSource.CLOSE,
        initial_hwm_source: InitialHwmSource = InitialHwmSource.FILL_PRICE,
        trail_stop_timing: TrailStopTiming = TrailStopTiming.LAGGED,
        allow_short_selling: bool = False,
        allow_leverage: bool = False,
        initial_margin: float = 0.5,
        long_maintenance_margin: float = 0.25,
        short_maintenance_margin: float = 0.30,
        fixed_margin_schedule: dict[str, tuple[float, float]] | None = None,
        execution_limits: ExecutionLimits | None = None,
        market_impact_model: MarketImpactModel | None = None,
        contract_specs: dict[str, ContractSpec] | None = None,
        share_type: ShareType = ShareType.FRACTIONAL,
        fill_ordering: FillOrdering = FillOrdering.EXIT_FIRST,
        reject_on_insufficient_cash: bool = True,
        cash_buffer_pct: float = 0.0,
        partial_fills_allowed: bool = False,
    ):
        # Runtime imports for accounting classes.
        # These are imported here rather than at module level because:
        # 1. The package __init__.py imports Broker, creating a potential import order issue
        # 2. TYPE_CHECKING block above provides type hints for static analysis
        # 3. This pattern allows mypy/pyright to validate types without runtime circular import
        from .accounting import (
            AccountState,
            Gatekeeper,
            UnifiedAccountPolicy,
        )

        self.initial_cash = initial_cash
        # Note: self.cash is now a property delegating to self.account.cash (Bug #5 fix)
        self.commission_model = commission_model or NoCommission()
        self.slippage_model = slippage_model or NoSlippage()
        self.stop_slippage_rate = stop_slippage_rate
        self.execution_mode = execution_mode
        self.stop_fill_mode = stop_fill_mode
        self.stop_level_basis = stop_level_basis
        self.trail_hwm_source = trail_hwm_source
        self.initial_hwm_source = initial_hwm_source
        self.trail_stop_timing = trail_stop_timing
        self.share_type = share_type
        self.fill_ordering = fill_ordering
        self.reject_on_insufficient_cash = reject_on_insufficient_cash
        self.cash_buffer_pct = cash_buffer_pct
        self.partial_fills_allowed = partial_fills_allowed

        # Create AccountState with UnifiedAccountPolicy
        policy: AccountPolicy = UnifiedAccountPolicy(
            allow_short_selling=allow_short_selling,
            allow_leverage=allow_leverage,
            initial_margin=initial_margin,
            long_maintenance_margin=long_maintenance_margin,
            short_maintenance_margin=short_maintenance_margin,
            fixed_margin_schedule=fixed_margin_schedule,
        )

        self.account = AccountState(initial_cash=initial_cash, policy=policy)
        # Derive account_type string from flags for backward compat
        if allow_leverage:
            self.account_type = "margin"
        elif allow_short_selling:
            self.account_type = "crypto"
        else:
            self.account_type = "cash"
        self.allow_short_selling = allow_short_selling
        self.allow_leverage = allow_leverage
        self.initial_margin = initial_margin
        self.long_maintenance_margin = long_maintenance_margin
        self.short_maintenance_margin = short_maintenance_margin
        self.fixed_margin_schedule = fixed_margin_schedule or {}

        # Create Gatekeeper for order validation
        self.gatekeeper = Gatekeeper(
            self.account, self.commission_model, cash_buffer_pct=self.cash_buffer_pct
        )

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

        # Per-asset trading statistics for stateful decision-making
        self._asset_stats: dict[str, AssetTradingStats] = {}
        self._stats_config = StatsConfig()
        self._session_config = None  # Optional SessionConfig for session boundary detection
        self._last_session_id: int | None = None  # Track current session for boundary detection

        # Extracted orchestration components (Phase B1 alpha-reset)
        self._order_book = OrderBook(self)
        self._risk_engine = RiskEngine(self)
        self._execution_engine = ExecutionEngine(self)
        self._portfolio_ledger = PortfolioLedger(self)

    @classmethod
    def from_config(
        cls,
        config: BacktestConfig,
        execution_limits: ExecutionLimits | None = None,
        market_impact_model: MarketImpactModel | None = None,
        contract_specs: dict[str, ContractSpec] | None = None,
    ) -> Broker:
        """Create Broker from BacktestConfig.

        This is the recommended way to create a Broker. All settings come from
        the BacktestConfig, ensuring consistency with Engine and other components.

        Args:
            config: BacktestConfig with all behavioral settings
            execution_limits: Optional execution limits (not in config)
            market_impact_model: Optional market impact model (not in config)
            contract_specs: Optional contract specifications (not in config)

        Returns:
            Configured Broker instance

        Example:
            config = BacktestConfig.from_preset("backtrader")
            broker = Broker.from_config(config)
        """
        from .config import CommissionModel as CommModelEnum
        from .config import SlippageModel as SlipModelEnum
        from .models import (
            FixedSlippage,
            NoCommission,
            NoSlippage,
            PercentageCommission,
            PercentageSlippage,
            PerShareCommission,
        )

        # Build commission model from config
        commission_model = None
        if config.commission_model == CommModelEnum.PERCENTAGE:
            commission_model = PercentageCommission(rate=config.commission_rate)
        elif config.commission_model == CommModelEnum.PER_SHARE:
            commission_model = PerShareCommission(
                per_share=config.commission_per_share,
                minimum=config.commission_minimum,
            )
        elif config.commission_model == CommModelEnum.NONE:
            commission_model = NoCommission()

        # Build slippage model from config
        slippage_model = None
        if config.slippage_model == SlipModelEnum.PERCENTAGE:
            slippage_model = PercentageSlippage(rate=config.slippage_rate)
        elif config.slippage_model == SlipModelEnum.FIXED:
            slippage_model = FixedSlippage(amount=config.slippage_fixed)
        elif config.slippage_model == SlipModelEnum.NONE:
            slippage_model = NoSlippage()

        return cls(
            initial_cash=config.initial_cash,
            commission_model=commission_model,
            slippage_model=slippage_model,
            stop_slippage_rate=config.stop_slippage_rate,
            execution_mode=config.execution_mode,
            stop_fill_mode=config.stop_fill_mode,
            stop_level_basis=config.stop_level_basis,
            trail_hwm_source=config.trail_hwm_source,
            initial_hwm_source=config.initial_hwm_source,
            trail_stop_timing=config.trail_stop_timing,
            allow_short_selling=config.allow_short_selling,
            allow_leverage=config.allow_leverage,
            initial_margin=config.initial_margin,
            long_maintenance_margin=config.long_maintenance_margin,
            short_maintenance_margin=config.short_maintenance_margin,
            fixed_margin_schedule=config.fixed_margin_schedule,
            execution_limits=execution_limits,
            market_impact_model=market_impact_model,
            contract_specs=contract_specs,
            share_type=config.share_type,
            fill_ordering=config.fill_ordering,
            reject_on_insufficient_cash=config.reject_on_insufficient_cash,
            cash_buffer_pct=config.cash_buffer_pct,
            partial_fills_allowed=config.partial_fills_allowed,
        )

    # Phase 4.1: Make cash a property delegating to account to prevent state drift
    @property
    def cash(self) -> float:
        """Current cash balance (delegates to AccountState)."""
        return self.account.cash

    @cash.setter
    def cash(self, value: float) -> None:
        """Set cash balance (delegates to AccountState)."""
        self.account.cash = value

    def get_contract_spec(self, asset: str) -> ContractSpec | None:
        """Get contract specification for an asset."""
        return self._contract_specs.get(asset)

    def get_multiplier(self, asset: str) -> float:
        """Get contract multiplier for an asset (1.0 for equities)."""
        spec = self._contract_specs.get(asset)
        return spec.multiplier if spec else 1.0

    # === Trading Statistics ===

    def configure_stats(
        self,
        recent_window_size: int | None = None,
        track_session_stats: bool | None = None,
        enabled: bool | None = None,
        config: StatsConfig | None = None,
    ) -> None:
        """Configure trading statistics tracking.

        Can either pass individual parameters or a StatsConfig object.
        Individual parameters override config values if both are provided.

        Args:
            recent_window_size: Number of recent trades to track (default 50)
            track_session_stats: Whether to track per-session statistics
            enabled: Whether stats tracking is enabled
            config: StatsConfig object (alternative to individual params)

        Example:
            # Using individual parameters
            broker.configure_stats(recent_window_size=100)

            # Using StatsConfig
            broker.configure_stats(config=StatsConfig(
                recent_window_size=100,
                track_session_stats=True,
            ))
        """
        if config is not None:
            self._stats_config = config
        else:
            self._stats_config = StatsConfig()

        # Override with individual parameters if provided
        if recent_window_size is not None:
            self._stats_config.recent_window_size = recent_window_size
        if track_session_stats is not None:
            self._stats_config.track_session_stats = track_session_stats
        if enabled is not None:
            self._stats_config.enabled = enabled

        # Update existing stats deques to new window size
        new_size = self._stats_config.recent_window_size
        for stats in self._asset_stats.values():
            if stats.recent_pnls.maxlen != new_size:
                # Create new deque with updated maxlen, preserving recent data
                old_pnls = list(stats.recent_pnls)
                stats.recent_pnls = deque(old_pnls[-new_size:], maxlen=new_size)
                # Recalculate recent_wins from preserved data
                stats.recent_wins = sum(1 for pnl in stats.recent_pnls if pnl > 0)

    def get_asset_stats(self, asset: str) -> AssetTradingStats:
        """Get trading statistics for an asset.

        Returns the AssetTradingStats object for the given asset, creating
        one if it doesn't exist. Stats are automatically updated when
        positions are closed or scaled down.

        Args:
            asset: Asset symbol (e.g., "BTC", "AAPL")

        Returns:
            AssetTradingStats object with all-time and recent statistics

        Example:
            stats = broker.get_asset_stats("BTC")

            # Check recent performance
            if stats.recent_win_rate > 0.6:
                # Increase position size when winning
                size = base_size * 1.5
            elif stats.recent_win_rate < 0.4:
                # Reduce size when losing
                size = base_size * 0.5

            # Check session performance (intraday)
            if stats.session_trades > 3 and stats.session_win_rate < 0.25:
                # Stop trading this asset for today
                return
        """
        if asset not in self._asset_stats:
            self._asset_stats[asset] = AssetTradingStats(
                recent_pnls=deque(maxlen=self._stats_config.recent_window_size)
            )
        return self._asset_stats[asset]

    def _record_pnl_event(
        self,
        asset: str,
        pnl: float,
        is_partial: bool = False,
    ) -> None:
        """Record a P&L realization event (internal use).

        Called by FillExecutor when a position is closed or scaled down.
        Updates the AssetTradingStats for the asset.

        Args:
            asset: Asset symbol
            pnl: Realized P&L from the exit
            is_partial: True if this was a partial exit (scale down)
        """
        if not self._stats_config.enabled:
            return

        stats = self.get_asset_stats(asset)
        stats.record_pnl(pnl)

    def set_session_config(self, config) -> None:
        """Set session configuration for session-aware statistics.

        When a session config is set, trading statistics are reset at
        session boundaries. This is useful for intraday strategies that
        want to track performance within each trading session.

        Args:
            config: SessionConfig object from ml4t.backtest.sessions

        Example:
            from ml4t.backtest.sessions import SessionConfig

            # CME futures: sessions start 5pm CT previous day
            session_config = SessionConfig(
                calendar="CME_Equity",
                timezone="America/Chicago",
                session_start_time="17:00",
            )
            broker.set_session_config(session_config)
        """
        self._session_config = config
        self._last_session_id = None

    def _check_session_boundary(self, timestamp: datetime) -> None:
        """Check for session boundary and reset session stats if crossed.

        Called each bar in _update_time() when session_config is set.
        Computes the session date from the timestamp and resets session
        stats for all assets when the session changes.

        Args:
            timestamp: Current bar timestamp
        """
        if self._session_config is None:
            return

        if not self._stats_config.track_session_stats:
            return

        from zoneinfo import ZoneInfo

        from .sessions import assign_session_date

        # Get session timezone and times
        tz = ZoneInfo(self._session_config.timezone)
        session_start_hour = self._session_config.get_session_start_hour()
        session_start_minute = self._session_config.get_session_start_minute()

        # Compute session date for current timestamp
        session_date = assign_session_date(timestamp, tz, session_start_hour, session_start_minute)
        # Use ordinal as session ID for comparison
        current_session_id = session_date.toordinal()

        # Check if session changed
        if self._last_session_id is not None and current_session_id != self._last_session_id:
            # Session boundary crossed - reset session stats for all assets
            for stats in self._asset_stats.values():
                stats.reset_session(current_session_id)

        self._last_session_id = current_session_id

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
        return self._portfolio_ledger.get_account_value()

    def get_rejected_orders(self, asset: str | None = None) -> list[Order]:
        """Get all rejected orders, optionally filtered by asset.

        Args:
            asset: If provided, filter to only this asset's rejected orders

        Returns:
            List of rejected Order objects with rejection_reason populated
        """
        return self._portfolio_ledger.get_rejected_orders(asset=asset)

    @property
    def last_rejection_reason(self) -> str | None:
        """Get reason for most recent order rejection.

        Returns:
            Rejection reason string, or None if no orders have been rejected
        """
        return self._portfolio_ledger.last_rejection_reason

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

    def evaluate_position_rules(self) -> list[Order]:
        """Evaluate position rules for all open positions.

        Called by Engine before processing orders. Returns list of exit orders.
        Handles defer_fill=True by storing pending exits for next bar.
        """
        return self._risk_engine.evaluate_position_rules()

    def submit_order(
        self,
        asset: str,
        quantity: float,
        side: OrderSide | None = None,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        trail_amount: float | None = None,
        _options: SubmitOrderOptions | None = None,
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
        return self._order_book.submit_order(
            asset=asset,
            quantity=quantity,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            trail_amount=trail_amount,
            options=_options,
        )

    def submit_bracket(
        self,
        asset: str,
        quantity: float,
        take_profit: float,
        stop_loss: float,
        entry_type: OrderType = OrderType.MARKET,
        entry_limit: float | None = None,
        validate_prices: bool = True,
    ) -> tuple[Order, Order, Order] | None:
        """Submit entry with take-profit and stop-loss.

        Creates a bracket order with entry, take-profit limit, and stop-loss orders.
        The exit side is automatically determined from the entry direction.

        Args:
            asset: Asset symbol to trade
            quantity: Position size (positive for long, negative for short)
            take_profit: Take-profit price level (LIMIT order)
            stop_loss: Stop-loss price level (STOP order)
            entry_type: Entry order type (default MARKET)
            entry_limit: Entry limit price (if entry_type is LIMIT)
            validate_prices: If True, validate that TP/SL prices are sensible
                            for the position direction (default True)

        Returns:
            Tuple of (entry_order, take_profit_order, stop_loss_order) or None if any fails.

        Raises:
            ValueError: If validate_prices=True and prices are inverted for direction.

        Notes:
            For LONG entries (quantity > 0):
                - take_profit should be > reference_price (profit on up move)
                - stop_loss should be < reference_price (exit on down move)

            For SHORT entries (quantity < 0):
                - take_profit should be < reference_price (profit on down move)
                - stop_loss should be > reference_price (exit on up move)

            Reference price is entry_limit (if LIMIT order) or current market price.
        """
        import warnings

        entry = self.submit_order(asset, quantity, order_type=entry_type, limit_price=entry_limit)
        if entry is None:
            return None

        # Derive exit side from entry direction (Bug #4 fix)
        # Long entry (BUY) -> SELL to exit; Short entry (SELL) -> BUY to cover
        exit_side = OrderSide.SELL if entry.side == OrderSide.BUY else OrderSide.BUY
        exit_qty = abs(quantity)

        # Validate bracket prices if requested
        if validate_prices:
            ref_price = entry_limit if entry_limit is not None else self._current_prices.get(asset)
            if ref_price is not None:
                is_long = entry.side == OrderSide.BUY

                if is_long:
                    # Long: TP should be above entry, SL should be below
                    if take_profit <= ref_price:
                        warnings.warn(
                            f"Bracket order for LONG {asset}: take_profit ({take_profit}) <= "
                            f"entry ({ref_price}). TP should be above entry for longs.",
                            UserWarning,
                            stacklevel=2,
                        )
                    if stop_loss >= ref_price:
                        warnings.warn(
                            f"Bracket order for LONG {asset}: stop_loss ({stop_loss}) >= "
                            f"entry ({ref_price}). SL should be below entry for longs.",
                            UserWarning,
                            stacklevel=2,
                        )
                else:
                    # Short: TP should be below entry, SL should be above
                    if take_profit >= ref_price:
                        warnings.warn(
                            f"Bracket order for SHORT {asset}: take_profit ({take_profit}) >= "
                            f"entry ({ref_price}). TP should be below entry for shorts.",
                            UserWarning,
                            stacklevel=2,
                        )
                    if stop_loss <= ref_price:
                        warnings.warn(
                            f"Bracket order for SHORT {asset}: stop_loss ({stop_loss}) <= "
                            f"entry ({ref_price}). SL should be above entry for shorts.",
                            UserWarning,
                            stacklevel=2,
                        )

        tp = self.submit_order(asset, exit_qty, exit_side, OrderType.LIMIT, limit_price=take_profit)
        if tp is None:
            return None
        tp.parent_id = entry.order_id

        sl = self.submit_order(asset, exit_qty, exit_side, OrderType.STOP, stop_price=stop_loss)
        if sl is None:
            return None
        sl.parent_id = entry.order_id

        return entry, tp, sl

    def update_order(self, order_id: str, **kwargs) -> bool:
        """Update pending order parameters.

        Only the following fields can be updated:
        - quantity: Order size
        - limit_price: Limit price for LIMIT orders
        - stop_price: Stop/trigger price for STOP orders
        - trail_amount: Trail distance for TRAILING_STOP orders

        Args:
            order_id: ID of the order to update
            **kwargs: Fields to update

        Returns:
            True if order was found and updated, False otherwise

        Raises:
            ValueError: If attempting to update non-updatable fields
        """
        return self._order_book.update_order(order_id, **kwargs)

    def cancel_order(self, order_id: str) -> bool:
        return self._order_book.cancel_order(order_id)

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

    # === Position Modification (P1 Features) ===

    def reduce_position(
        self,
        asset: str,
        fraction: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
    ) -> Order | None:
        """Reduce an existing position by a fraction.

        Sells (for long) or covers (for short) a percentage of the current position.

        Args:
            asset: Asset symbol to reduce
            fraction: Fraction to exit (0.5 = sell half, 0.25 = sell quarter)
                     Must be between 0 and 1 (exclusive of 0, inclusive of 1)
            order_type: Order type (default MARKET)
            limit_price: Limit price for LIMIT orders

        Returns:
            Order object if position exists and order submitted, None otherwise

        Raises:
            ValueError: If fraction is not in (0, 1]

        Example:
            # Sell half of AAPL position
            order = broker.reduce_position("AAPL", fraction=0.5)

            # Sell 25% of position with limit
            order = broker.reduce_position("AAPL", fraction=0.25,
                                           order_type=OrderType.LIMIT,
                                           limit_price=155.0)
        """
        if fraction <= 0 or fraction > 1:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")

        pos = self.positions.get(asset)
        if pos is None or pos.quantity == 0:
            return None

        # Calculate quantity to exit
        exit_qty = abs(pos.quantity) * fraction

        # Determine exit side (opposite of position direction)
        side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY

        return self.submit_order(asset, exit_qty, side, order_type, limit_price)

    def buy(
        self,
        asset: str,
        shares: float | None = None,
        contracts: int | None = None,
        amount: float | None = None,
        dollars: float | None = None,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
    ) -> Order | None:
        """Buy an asset with explicit quantity specification.

        Provides a clean API for buying with different quantity types:
        - shares: Number of shares (equities)
        - contracts: Number of contracts (futures)
        - amount: Base currency amount (crypto)
        - dollars: Dollar value to buy (any asset)

        Exactly one quantity parameter must be provided.

        Args:
            asset: Asset symbol
            shares: Number of shares to buy (equities)
            contracts: Number of contracts to buy (futures)
            amount: Base currency amount to buy (crypto)
            dollars: Dollar value to buy (converted to quantity at current price)
            order_type: Order type (default MARKET)
            limit_price: Limit price for LIMIT orders

        Returns:
            Order object if submitted, None if no current price or invalid params

        Raises:
            ValueError: If zero or more than one quantity parameter is provided

        Example:
            # Buy 100 shares of AAPL
            broker.buy("AAPL", shares=100)

            # Buy 2 ES futures contracts
            broker.buy("ES", contracts=2)

            # Buy $5000 worth of BTC
            broker.buy("BTC", dollars=5000)

            # Buy 0.5 BTC
            broker.buy("BTC", amount=0.5)
        """
        # Count non-None parameters
        qty_params = [shares, contracts, amount, dollars]
        provided = sum(1 for p in qty_params if p is not None)

        if provided == 0:
            raise ValueError("Must provide one of: shares, contracts, amount, dollars")
        if provided > 1:
            raise ValueError("Must provide only one of: shares, contracts, amount, dollars")

        # Determine quantity
        quantity: float
        if shares is not None:
            quantity = shares
        elif contracts is not None:
            quantity = float(contracts)
        elif amount is not None:
            quantity = amount
        elif dollars is not None:
            price = self._current_prices.get(asset)
            if price is None or price <= 0:
                return None
            multiplier = self.get_multiplier(asset)
            quantity = dollars / (price * multiplier)
        else:
            return None  # Should not reach here

        if quantity <= 0:
            return None

        return self.submit_order(asset, quantity, OrderSide.BUY, order_type, limit_price)

    def sell(
        self,
        asset: str,
        shares: float | None = None,
        contracts: int | None = None,
        amount: float | None = None,
        dollars: float | None = None,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
    ) -> Order | None:
        """Sell an asset with explicit quantity specification.

        Provides a clean API for selling with different quantity types:
        - shares: Number of shares (equities)
        - contracts: Number of contracts (futures)
        - amount: Base currency amount (crypto)
        - dollars: Dollar value to sell (any asset)

        Exactly one quantity parameter must be provided.

        Args:
            asset: Asset symbol
            shares: Number of shares to sell (equities)
            contracts: Number of contracts to sell (futures)
            amount: Base currency amount to sell (crypto)
            dollars: Dollar value to sell (converted to quantity at current price)
            order_type: Order type (default MARKET)
            limit_price: Limit price for LIMIT orders

        Returns:
            Order object if submitted, None if no current price or invalid params

        Raises:
            ValueError: If zero or more than one quantity parameter is provided

        Example:
            # Sell 50 shares of AAPL
            broker.sell("AAPL", shares=50)

            # Sell 1 ES futures contract
            broker.sell("ES", contracts=1)

            # Sell $2500 worth of position
            broker.sell("BTC", dollars=2500)
        """
        # Count non-None parameters
        qty_params = [shares, contracts, amount, dollars]
        provided = sum(1 for p in qty_params if p is not None)

        if provided == 0:
            raise ValueError("Must provide one of: shares, contracts, amount, dollars")
        if provided > 1:
            raise ValueError("Must provide only one of: shares, contracts, amount, dollars")

        # Determine quantity
        quantity: float
        if shares is not None:
            quantity = shares
        elif contracts is not None:
            quantity = float(contracts)
        elif amount is not None:
            quantity = amount
        elif dollars is not None:
            price = self._current_prices.get(asset)
            if price is None or price <= 0:
                return None
            multiplier = self.get_multiplier(asset)
            quantity = dollars / (price * multiplier)
        else:
            return None  # Should not reach here

        if quantity <= 0:
            return None

        return self.submit_order(asset, quantity, OrderSide.SELL, order_type, limit_price)

    # === Trade History Access (P1 Features) ===

    def get_trades(
        self,
        asset: str | None = None,
        last_n: int | None = None,
        since_bar: int | None = None,
    ) -> list[Trade]:
        """Get completed trades, optionally filtered.

        Provides access to trade history during the backtest for stateful
        decision-making (e.g., adjusting position sizing based on recent
        wins/losses, implementing cooldown logic after stop-outs).

        Args:
            asset: Filter to only this asset's trades. If None, returns all trades.
            last_n: Return only the last N trades (after other filters)
            since_bar: Return trades with entry_bar >= since_bar

        Returns:
            List of Trade objects matching the filters, ordered by exit time

        Example:
            # Get all trades for BTC
            btc_trades = broker.get_trades(asset="BTC")

            # Get last 5 trades overall
            recent = broker.get_trades(last_n=5)

            # Get recent trades for cooldown logic
            last_trade = broker.get_last_trade("AAPL")
            if last_trade and last_trade.exit_reason == "stop_loss":
                # Implement cooldown after stop-out
                pass
        """
        result = self.trades

        # Filter by asset
        if asset is not None:
            result = [t for t in result if t.symbol == asset]

        # Filter by entry bar (if we had bar info - using entry_time as proxy)
        # Note: Trade doesn't have entry_bar, but bars_held is available
        # For now, since_bar is not fully implementable without entry_bar
        # but we keep the parameter for API consistency with spec

        # Apply last_n limit
        if last_n is not None and last_n > 0:
            result = result[-last_n:]

        return result

    def get_last_trade(self, asset: str | None = None) -> Trade | None:
        """Get the most recent completed trade.

        Convenience method for strategies that need to check the last trade
        (e.g., for cooldown logic after stop-outs).

        Args:
            asset: Filter to only this asset. If None, returns last trade overall.

        Returns:
            Most recent Trade object, or None if no trades

        Example:
            last = broker.get_last_trade("AAPL")
            if last and last.exit_reason == "stop_loss":
                # Was stopped out - implement cooldown
                cooldown_bars = 5
        """
        trades = self.get_trades(asset=asset, last_n=1)
        return trades[0] if trades else None

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
        # Bug #2 fix: Include contract multiplier in value calculations
        multiplier = self.get_multiplier(asset)
        unit_notional = price * multiplier  # Notional value per share/contract

        # Get current position value (with multiplier)
        pos = self.positions.get(asset)
        current_value = 0.0
        if pos and pos.quantity != 0:
            current_value = pos.quantity * unit_notional

        # Calculate delta
        delta_value = target_value - current_value
        if abs(delta_value) < 0.01:  # Less than 1 cent, no trade needed
            return None

        # Convert to quantity (accounting for multiplier)
        delta_qty = delta_value / unit_notional

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
            # Bug #2 fix: Include contract multiplier in value calculations
            multiplier = self.get_multiplier(asset)
            current_value = pos.quantity * price * multiplier if pos and pos.quantity != 0 else 0.0

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
        return self._order_book.get_order(order_id)

    def get_pending_orders(self, asset: str | None = None) -> list[Order]:
        """Get pending orders, optionally filtered by asset."""
        return self._order_book.get_pending_orders(asset=asset)

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

        Called at the start of a new bar to fill deferred exits.
        The fill price depends on stop_fill_mode:
        - STOP_PRICE: Fill at the stop price (with gap-through check)
        - NEXT_BAR_OPEN: Fill at open price
        - Other modes: Fill at open price

        Returns list of exit orders that were created and will be filled.
        """
        return self._risk_engine.process_pending_exits()

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

        # Check for session boundary (resets session stats if crossed)
        self._check_session_boundary(timestamp)

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

        Water mark source configuration:
        - trail_hwm_source == BAR_EXTREME: Update HWM from high, LWM from low (VBT Pro OHLC mode)
        - trail_hwm_source == CLOSE: Update HWM/LWM from close only (default)
        """
        for asset, pos in self.positions.items():
            if asset in self._current_prices:
                # For new positions (created this bar), skip updating from entry bar's HIGH/LOW
                # VBT Pro only updates water marks from bar extremes on the bar AFTER entry
                is_new_position = asset in self._positions_created_this_bar
                # BAR_EXTREME: use HIGH for HWM (longs), LOW for LWM (shorts)
                use_extremes = self.trail_hwm_source.value == "bar_extreme" and not is_new_position
                pos.update_water_marks(
                    current_price=self._current_prices[asset],
                    bar_high=self._current_highs.get(asset),
                    bar_low=self._current_lows.get(asset),
                    use_high_for_hwm=use_extremes,
                    use_low_for_lwm=use_extremes,
                )

    def _get_available_cash(self) -> float:
        """Get available cash after applying cash buffer.

        Returns:
            Cash available for new orders (total cash minus buffer reserve).
        """
        if self.cash_buffer_pct > 0:
            return self.account.cash * (1.0 - self.cash_buffer_pct)
        return self.account.cash

    def _apply_share_rounding(self, order: Order) -> None:
        """Round order quantity to integer if share_type is INTEGER.

        Modifies order in-place. If rounding reduces quantity to zero,
        the order will be rejected during validation (zero-quantity check).
        """
        if self.share_type == ShareType.INTEGER:
            order.quantity = float(int(order.quantity))

    def _try_partial_fill(self, order: Order, fill_price: float) -> bool:
        """Attempt a partial fill when full order cannot be afforded.

        Only used when partial_fills_allowed is True and the gatekeeper
        rejects an entry order for insufficient cash.

        Returns:
            True if a partial fill was executed, False otherwise.
        """
        available = self._get_available_cash()
        commission_rate = 0.0
        # Estimate commission rate for sizing
        test_commission = self.commission_model.calculate(order.asset, 1.0, fill_price)
        if fill_price > 0:
            commission_rate = test_commission / fill_price

        # Max affordable shares (accounting for commission)
        max_value = available / (1.0 + commission_rate) if commission_rate > 0 else available
        max_shares = max_value / fill_price if fill_price > 0 else 0

        if self.share_type == ShareType.INTEGER:
            max_shares = float(int(max_shares))

        if max_shares <= 0:
            return False

        # Modify order to affordable quantity and execute
        order.quantity = max_shares
        return bool(self._execute_fill(order, fill_price))

    def _process_orders(self, use_open: bool = False):
        """Process pending orders against current prices.

        Fill ordering is controlled by ``self.fill_ordering``:

        - EXIT_FIRST: All exits → mark-to-market → all entries (capital-efficient).
        - FIFO: Orders process in submission order with sequential cash updates.

        ``reject_on_insufficient_cash``, ``cash_buffer_pct``, ``partial_fills_allowed``,
        and ``share_type`` are all enforced here.

        Args:
            use_open: If True, use open prices (for next-bar mode at bar start).
        """
        self._execution_engine.process_orders(use_open=use_open)

    def _get_fill_price_for_order(self, order: Order, use_open: bool) -> float | None:
        """Get the fill price for an order based on execution mode."""
        if use_open and self.execution_mode == ExecutionMode.NEXT_BAR:
            return self._current_opens.get(order.asset)
        return self._current_prices.get(order.asset)

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

        For risk-triggered exits (stop-loss, take-profit, trailing stop), uses
        the fill price already computed by the position rule. Position rules
        handle gap-through scenarios correctly in their own logic.

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

        # Use the fill price computed by the position rule directly.
        # Position rules (TakeProfit, StopLoss, TrailingStop) already handle
        # gap-through scenarios correctly in their fill price calculation.
        # We should NOT override their logic with a generic gap-through check
        # because the direction differs for stop-loss vs take-profit orders.
        fill_price = risk_fill_price

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

        Updates the stop price based on the water mark, then checks
        if the stop has been triggered.

        Note: This method mutates order.stop_price as trailing stops
        must track the water mark.

        Args:
            order: Trailing stop order (SELL protects longs, BUY protects shorts)
            high: Bar high price
            low: Bar low price
            bar_open: Bar open price

        Returns:
            Fill price if triggered, None otherwise
        """
        if order.trail_amount is None:
            return None

        if order.side == OrderSide.SELL:
            # SELL trailing stop: protects long positions
            # Stop trails below the high water mark
            new_stop = high - order.trail_amount
            if order.stop_price is None or new_stop > order.stop_price:
                order.stop_price = new_stop

            # Check if triggered: low touched or crossed stop
            if order.stop_price is None or low > order.stop_price:
                return None

        else:  # OrderSide.BUY - Bug #5 fix
            # BUY trailing stop: protects short positions
            # Stop trails above the low water mark
            new_stop = low + order.trail_amount
            if order.stop_price is None or new_stop < order.stop_price:
                order.stop_price = new_stop

            # Check if triggered: high touched or crossed stop
            if order.stop_price is None or high < order.stop_price:
                return None

        # At this point stop_price is guaranteed non-None (set above, returned if None)
        assert order.stop_price is not None
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
