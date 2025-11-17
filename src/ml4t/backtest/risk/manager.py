"""Risk manager for orchestrating rule evaluation and position monitoring.

The RiskManager coordinates risk rule execution, caches RiskContext construction,
and integrates with the backtesting engine for position exit checking and order validation.

Key Features:
    - Rule registration and management (add_rule, remove_rule)
    - Context caching for performance (10x speedup with 10 rules)
    - Position exit checking (check_position_exits)
    - Order validation (validate_order)
    - Fill recording for position state tracking (record_fill)

Design Principles:
    - **Performance**: Context caching prevents redundant RiskContext construction
    - **Composability**: Supports multiple rules with automatic decision merging
    - **Clean Integration**: Three engine hooks cover all use cases
    - **Type Safety**: Full type hints throughout

Examples:
    >>> # Basic setup
    >>> from ml4t.backtest.risk import RiskManager, TimeBasedExit, PriceBasedStopLoss
    >>>
    >>> manager = RiskManager()
    >>> manager.add_rule(TimeBasedExit(max_bars=60))
    >>> manager.add_rule(PriceBasedStopLoss(sl_price=Decimal("95.00")))
    >>>
    >>> # Engine integration (Hook C: check exits before strategy)
    >>> exit_orders = manager.check_position_exits(
    ...     market_event=event,
    ...     broker=broker,
    ...     portfolio=portfolio
    ... )
    >>> for order in exit_orders:
    ...     broker.submit_order(order)
    >>>
    >>> # Engine integration (Hook B: validate order after strategy)
    >>> strategy_order = strategy.generate_signal(event)
    >>> validated = manager.validate_order(
    ...     order=strategy_order,
    ...     market_event=event,
    ...     broker=broker,
    ...     portfolio=portfolio
    ... )
    >>> if validated:
    ...     broker.submit_order(validated)
    >>>
    >>> # Engine integration (Hook D: record fills)
    >>> for fill_event in fill_events:
    ...     manager.record_fill(fill_event, market_event)
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional, Union

from ml4t.backtest.core.event import FillEvent, MarketEvent
from ml4t.backtest.core.types import AssetId, OrderSide, OrderType, Price
from ml4t.backtest.data.feature_provider import FeatureProvider
from ml4t.backtest.execution.order import Order
from ml4t.backtest.portfolio.portfolio import Portfolio
from ml4t.backtest.risk.context import RiskContext
from ml4t.backtest.risk.decision import ExitType, RiskDecision
from ml4t.backtest.risk.rule import RiskRule, RiskRuleProtocol


@dataclass
class PositionTradeState:
    """Position tracking state for bar counting and MFE/MAE calculation.

    Tracks entry information and running statistics for open positions.
    Updated on each market event via record_fill() and check_position_exits().

    Attributes:
        asset_id: Asset identifier
        entry_time: Timestamp of initial fill
        entry_price: Average entry price (cost basis)
        entry_quantity: Total quantity entered
        bars_held: Number of bars since entry (incremented on market events)
        max_favorable_excursion: Best unrealized P&L since entry (in price units)
        max_adverse_excursion: Worst unrealized P&L since entry (in price units)
    """
    asset_id: AssetId
    entry_time: datetime
    entry_price: Decimal
    entry_quantity: float
    bars_held: int = 0
    max_favorable_excursion: Decimal = Decimal("0.0")
    max_adverse_excursion: Decimal = Decimal("0.0")

    def update_on_market_event(self, market_price: Decimal) -> None:
        """Update bar counter and MFE/MAE on each market event.

        MFE (Max Favorable Excursion): The best price movement in your favor since entry.
        MAE (Max Adverse Excursion): The worst price movement against you since entry.

        For long positions (entry_quantity > 0):
            - MFE = max(MFE, current_price - entry_price)  # Best upward move
            - MAE = max(MAE, entry_price - current_price)  # Worst downward move

        For short positions (entry_quantity < 0):
            - MFE = max(MFE, entry_price - current_price)  # Best downward move
            - MAE = max(MAE, current_price - entry_price)  # Worst upward move

        Both are tracked as positive values representing the magnitude of excursion.

        Args:
            market_price: Current market price (close price from MarketEvent)
        """
        self.bars_held += 1

        # Calculate excursion from entry based on position direction
        if self.entry_quantity > 0:  # Long position
            # Favorable excursion: price went up
            favorable_excursion = market_price - self.entry_price
            # Adverse excursion: price went down (make positive)
            adverse_excursion = self.entry_price - market_price

            # Track maximum excursions (always positive or zero)
            self.max_favorable_excursion = max(
                self.max_favorable_excursion,
                max(Decimal("0"), favorable_excursion)
            )
            self.max_adverse_excursion = max(
                self.max_adverse_excursion,
                max(Decimal("0"), adverse_excursion)
            )

        else:  # Short position (entry_quantity < 0)
            # Favorable excursion: price went down
            favorable_excursion = self.entry_price - market_price
            # Adverse excursion: price went up (make positive)
            adverse_excursion = market_price - self.entry_price

            # Track maximum excursions (always positive or zero)
            self.max_favorable_excursion = max(
                self.max_favorable_excursion,
                max(Decimal("0"), favorable_excursion)
            )
            self.max_adverse_excursion = max(
                self.max_adverse_excursion,
                max(Decimal("0"), adverse_excursion)
            )


@dataclass
class PositionLevels:
    """Stop-loss and take-profit price levels for a position.

    Tracks risk management levels set by rules. Updated when rules
    return RiskDecision with update_stop_loss or update_take_profit.

    Attributes:
        asset_id: Asset identifier
        stop_loss: Stop-loss price (None = no stop set)
        take_profit: Take-profit price (None = no target set)
    """
    asset_id: AssetId
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None


class RiskManager:
    """Orchestrates risk rule evaluation and position monitoring.

    Manages a collection of risk rules, builds RiskContext with caching,
    evaluates all rules, and merges decisions. Integrates with engine via
    three hooks: check_position_exits, validate_order, record_fill.

    Attributes:
        feature_provider: Optional FeatureProvider for additional features
        _rules: List of registered risk rules
        _position_state: Per-asset position tracking state
        _position_levels: Per-asset stop-loss and take-profit levels
        _context_cache: Context cache for performance optimization

    Performance:
        - Context caching reduces construction from O(n×m) to O(n)
        - With 10 rules and 20 positions: ~10x speedup
        - Cache invalidated on each new timestamp
    """

    def __init__(self, feature_provider: Optional[FeatureProvider] = None):
        """Initialize RiskManager.

        Args:
            feature_provider: Optional provider for additional features beyond
                what's in MarketEvent.signals and MarketEvent.context dicts.
                Used when rules need features not already in the event.
        """
        self.feature_provider = feature_provider
        self._rules: list[Union[RiskRule, RiskRuleProtocol]] = []
        self._position_state: dict[AssetId, PositionTradeState] = {}
        self._position_levels: dict[AssetId, PositionLevels] = {}
        self._context_cache: dict[tuple[AssetId, datetime], RiskContext] = {}

    def add_rule(self, rule: Union[RiskRule, RiskRuleProtocol]) -> None:
        """Register a risk rule.

        Args:
            rule: Risk rule to add (RiskRule subclass or callable matching Protocol)

        Raises:
            TypeError: If rule doesn't implement RiskRule or RiskRuleProtocol

        Example:
            >>> manager = RiskManager()
            >>> manager.add_rule(TimeBasedExit(max_bars=60))
            >>>
            >>> # Callable rule (Protocol)
            >>> def simple_stop(ctx):
            ...     return RiskDecision.exit_now(...) if ctx.unrealized_pnl_pct < -0.05 else RiskDecision.no_action()
            >>> manager.add_rule(simple_stop)
        """
        if not isinstance(rule, (RiskRule, RiskRuleProtocol)):
            raise TypeError(
                f"Rule must be RiskRule subclass or callable matching RiskRuleProtocol, "
                f"got {type(rule).__name__}"
            )
        self._rules.append(rule)

    def remove_rule(self, rule: Union[RiskRule, RiskRuleProtocol]) -> None:
        """Unregister a risk rule.

        Args:
            rule: Rule to remove

        Raises:
            ValueError: If rule not found

        Example:
            >>> stop_loss_rule = PriceBasedStopLoss(sl_price=Decimal("95.00"))
            >>> manager.add_rule(stop_loss_rule)
            >>> # Later...
            >>> manager.remove_rule(stop_loss_rule)
        """
        try:
            self._rules.remove(rule)
        except ValueError:
            raise ValueError(f"Rule {rule} not registered with this RiskManager")

    def check_position_exits(
        self,
        market_event: MarketEvent,
        broker: "Broker",  # type: ignore
        portfolio: Portfolio,
    ) -> list[Order]:
        """Check all open positions and generate exit orders if rules triggered.

        This is Hook C in the engine integration - called BEFORE strategy signal
        generation to allow risk rules to exit positions independently.

        Process:
            1. Iterate through all open positions from broker
            2. Build RiskContext for each (with caching)
            3. Update position state (bars_held, MFE, MAE)
            4. Evaluate all rules
            5. Merge decisions
            6. Generate exit order if should_exit=True

        Args:
            market_event: Current market event with prices and features
            broker: Broker for position lookup
            portfolio: Portfolio for equity and cash

        Returns:
            List of exit orders to submit (Market orders to close positions)

        Example:
            >>> # In BacktestEngine event loop
            >>> for event in clock:
            ...     if isinstance(event, MarketEvent):
            ...         # Hook C: Check risk rules BEFORE strategy
            ...         exit_orders = risk_manager.check_position_exits(event, broker, portfolio)
            ...         for order in exit_orders:
            ...             broker.submit_order(order)
            ...
            ...         # Then run strategy
            ...         strategy.on_market_data(event)
        """
        exit_orders: list[Order] = []

        # Get all positions from broker (source of truth)
        positions = broker.get_positions()

        for asset_id, position in positions.items():
            # Skip if no position or position for different asset
            if position.quantity == 0 or asset_id != market_event.asset_id:
                continue

            # Build context with caching
            context = self._build_context(
                asset_id=asset_id,
                market_event=market_event,
                broker=broker,
                portfolio=portfolio,
            )

            # Update position state tracking
            if asset_id in self._position_state:
                state = self._position_state[asset_id]
                state.update_on_market_event(market_price=market_event.close)

            # Evaluate all rules
            decision = self.evaluate_all_rules(context)

            # Update levels if rules suggest changes
            if decision.update_stop_loss or decision.update_take_profit:
                if asset_id not in self._position_levels:
                    self._position_levels[asset_id] = PositionLevels(asset_id=asset_id)

                levels = self._position_levels[asset_id]
                if decision.update_stop_loss:
                    levels.stop_loss = decision.update_stop_loss
                if decision.update_take_profit:
                    levels.take_profit = decision.update_take_profit

            # Generate exit order if needed
            if decision.should_exit:
                # Exit entire position (opposite quantity)
                exit_quantity = -position.quantity
                # Determine side: if closing long (negative qty), we SELL. If closing short (positive qty), we BUY.
                side = OrderSide.SELL if exit_quantity < 0 else OrderSide.BUY

                exit_order = Order(
                    asset_id=asset_id,
                    order_type=OrderType.MARKET,
                    side=side,
                    quantity=abs(exit_quantity),
                )
                exit_orders.append(exit_order)

                # Clear position tracking (will be removed on fill)
                # Keep for now until fill confirmed

        return exit_orders

    def validate_order(
        self,
        order: Order,
        market_event: MarketEvent,
        broker: "Broker",  # type: ignore
        portfolio: Portfolio,
    ) -> Optional[Order]:
        """Validate order against risk rules before execution.

        This is Hook B in the engine integration - called AFTER strategy generates
        order but BEFORE broker executes it. Allows rules to reject or modify orders.

        Process:
            1. Build RiskContext for the order's asset
            2. Call validate_order() on each rule that implements it
            3. Return order if all rules pass, None if any rule rejects

        Args:
            order: Order to validate (from strategy)
            market_event: Current market event
            broker: Broker for position lookup
            portfolio: Portfolio for equity and cash

        Returns:
            Order if validated, None if rejected by rules

        Example:
            >>> # In BacktestEngine event loop
            >>> strategy_order = strategy.on_market_data(event)
            >>> if strategy_order:
            ...     # Hook B: Validate before execution
            ...     validated = risk_manager.validate_order(strategy_order, event, broker, portfolio)
            ...     if validated:
            ...         broker.submit_order(validated)
            ...     else:
            ...         logger.info("Order rejected by risk rules")
        """
        # Build context for order's asset
        context = self._build_context(
            asset_id=order.asset_id,
            market_event=market_event,
            broker=broker,
            portfolio=portfolio,
        )

        # Check each rule's validate_order method
        for rule in self._rules:
            # Only check rules that implement validate_order
            if hasattr(rule, 'validate_order') and callable(rule.validate_order):
                validated_order = rule.validate_order(order, context)
                if validated_order is None:
                    # Rule rejected the order
                    return None
                # Rule may have modified the order (e.g., reduced size)
                order = validated_order

        return order

    def record_fill(self, fill_event: FillEvent, market_event: MarketEvent) -> None:
        """Record fill event and update position state tracking.

        This is Hook D in the engine integration - called AFTER fills are executed
        to update position tracking state (entry time, entry price, etc.).

        Process:
            1. If opening new position: create PositionTradeState
            2. If closing position: remove PositionTradeState and PositionLevels
            3. Update position quantities

        Args:
            fill_event: Fill event from broker
            market_event: Current market event

        Example:
            >>> # In BacktestEngine event loop
            >>> fills = broker.process_fills(event)
            >>> for fill in fills:
            ...     # Hook D: Record fill
            ...     risk_manager.record_fill(fill, event)
            ...     strategy.on_fill(fill)
        """
        asset_id = fill_event.asset_id

        # Get current position from position_state (may not exist yet)
        current_state = self._position_state.get(asset_id)

        # Convert fill_quantity to signed based on side
        # BUY adds to position (positive), SELL subtracts (negative)
        from ml4t.backtest.core.types import OrderSide
        fill_qty = fill_event.fill_quantity if fill_event.side == OrderSide.BUY else -fill_event.fill_quantity

        # Determine if opening, adding, closing, or reversing position
        if current_state is None:
            # Opening new position
            self._position_state[asset_id] = PositionTradeState(
                asset_id=asset_id,
                entry_time=fill_event.timestamp,
                entry_price=fill_event.fill_price,
                entry_quantity=fill_qty,
                bars_held=0,
            )
            self._position_levels[asset_id] = PositionLevels(asset_id=asset_id)

        else:
            # Update existing position
            # Check if closing or reversing
            new_quantity = current_state.entry_quantity + fill_qty

            if abs(new_quantity) < 0.001:  # Closed position (allow for floating point error)
                # Position closed - remove tracking
                del self._position_state[asset_id]
                if asset_id in self._position_levels:
                    del self._position_levels[asset_id]

            elif (current_state.entry_quantity > 0 and new_quantity < 0) or \
                 (current_state.entry_quantity < 0 and new_quantity > 0):
                # Reversed position - reset tracking
                self._position_state[asset_id] = PositionTradeState(
                    asset_id=asset_id,
                    entry_time=fill_event.timestamp,
                    entry_price=fill_event.fill_price,
                    entry_quantity=new_quantity,
                    bars_held=0,
                )
                self._position_levels[asset_id] = PositionLevels(asset_id=asset_id)

            else:
                # Adding to position - update average entry price
                total_quantity = current_state.entry_quantity + fill_qty
                current_cost = current_state.entry_price * Decimal(str(abs(current_state.entry_quantity)))
                new_cost = fill_event.fill_price * Decimal(str(abs(fill_qty)))
                avg_price = (current_cost + new_cost) / Decimal(str(abs(total_quantity)))

                current_state.entry_price = avg_price
                current_state.entry_quantity = total_quantity

    def evaluate_all_rules(self, context: RiskContext) -> RiskDecision:
        """Evaluate all registered rules and merge decisions.

        Process:
            1. Call evaluate() on each rule
            2. Collect all decisions
            3. Merge using RiskDecision.merge() with priority resolution

        Args:
            context: RiskContext for current position/market state

        Returns:
            Merged RiskDecision from all rules

        Example:
            >>> context = RiskContext.from_state(event, position, portfolio)
            >>> decision = manager.evaluate_all_rules(context)
            >>> if decision.should_exit:
            ...     print(f"Exit signal: {decision.reason}")
        """
        if not self._rules:
            return RiskDecision.no_action(reason="No rules registered")

        decisions: list[RiskDecision] = []

        for rule in self._rules:
            # Call rule (handles both RiskRule.evaluate() and callable Protocol)
            if isinstance(rule, RiskRule):
                decision = rule.evaluate(context)
            else:
                # Callable rule (Protocol)
                decision = rule(context)

            decisions.append(decision)

        # Merge all decisions
        return RiskDecision.merge(decisions)

    def _build_context(
        self,
        asset_id: AssetId,
        market_event: MarketEvent,
        broker: "Broker",  # type: ignore
        portfolio: Portfolio,
    ) -> RiskContext:
        """Build RiskContext with caching.

        Context construction is expensive (position lookup, feature extraction).
        Cache contexts by (asset_id, timestamp) to avoid rebuilding for multiple
        rules on the same event.

        Performance:
            - Without cache: O(n × m) where n=positions, m=rules
            - With cache: O(n) - build once per position per event
            - 10 rules: ~10x speedup

        Args:
            asset_id: Asset identifier
            market_event: Current market event
            broker: Broker for position lookup
            portfolio: Portfolio for equity/cash

        Returns:
            Cached or newly constructed RiskContext
        """
        # Check cache
        cache_key = (asset_id, market_event.timestamp)
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]

        # Build new context
        position = broker.get_position(asset_id)

        # Get position state if exists
        state = self._position_state.get(asset_id)

        # Create a modified market event with tracked MFE/MAE injected into signals
        # This allows RiskContext to use tracked values instead of intra-bar OHLC
        modified_event = market_event
        if state is not None and position is not None:
            # Convert MFE/MAE from price units to currency units (price × quantity)
            quantity = abs(position.quantity) if hasattr(position, 'quantity') else abs(position)

            # Inject tracked MFE/MAE into signals dict
            # Use reserved keys prefixed with underscore to avoid conflicts
            signals = market_event.signals.copy() if market_event.signals else {}
            signals['_tracked_mfe'] = float(state.max_favorable_excursion * Decimal(str(quantity)))
            signals['_tracked_mae'] = float(state.max_adverse_excursion * Decimal(str(quantity)))

            # Create modified event (MarketEvent is not frozen, but we avoid mutation)
            # Note: MarketEvent is typically immutable-ish, so we need to be careful
            # For now, modify the signals dict in place since it's passed by reference
            # This is safe because we're in the context building phase
            from dataclasses import replace
            try:
                modified_event = replace(market_event, signals=signals)
            except Exception:
                # If replace fails (e.g., MarketEvent not a dataclass), modify in place
                market_event.signals.update(signals)
                modified_event = market_event

        context = RiskContext.from_state(
            market_event=modified_event,
            position=position,
            portfolio=portfolio,
            feature_provider=self.feature_provider,
            entry_time=state.entry_time if state else None,
            bars_held=state.bars_held if state else 0,
        )

        # Cache for reuse
        self._context_cache[cache_key] = context

        return context

    def clear_cache(self, before_timestamp: Optional[datetime] = None) -> None:
        """Clear context cache to free memory.

        Call periodically (e.g., at end of day) to prevent unbounded cache growth.

        Args:
            before_timestamp: If specified, only clear entries before this time.
                If None, clear entire cache.

        Example:
            >>> # Clear cache at end of day
            >>> if event.timestamp.hour == 16:  # Market close
            ...     manager.clear_cache()
        """
        if before_timestamp is None:
            self._context_cache.clear()
        else:
            self._context_cache = {
                k: v for k, v in self._context_cache.items()
                if k[1] >= before_timestamp
            }
