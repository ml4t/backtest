"""Fill execution orchestration.

This module provides FillExecutor which handles order fill execution,
extracting the logic from Broker._execute_fill() into a focused class
with helper methods for position creation, closing, flipping, and scaling.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from ..config import InitialHwmSource, ShareType
from ..core.shared import quantity_zero_tolerance
from ..core.state import ExecutionJournal, MarketState, OrderState, RiskState
from ..models import calculate_commission, calculate_slippage
from ..types import (
    ExitReason,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    Position,
    Trade,
)

if TYPE_CHECKING:
    from ..broker import Broker
    from ..core.fill_engine import FillEngine


def _get_exit_reason(order: Order) -> str:
    """Get exit reason from order.

    Args:
        order: Order with exit reason metadata

    Returns:
        ExitReason enum value as string
    """
    if order._exit_reason is not None:
        return order._exit_reason.value
    return ExitReason.SIGNAL.value


def _is_position_flip(old_quantity: float, new_quantity: float) -> bool:
    """Return whether a position crosses through zero to the opposite side."""
    return old_quantity > 0 > new_quantity or old_quantity < 0 < new_quantity


def _calculate_position_pnl(
    entry_price: float,
    exit_price: float,
    signed_quantity: float,
    multiplier: float,
) -> float:
    """Calculate PnL from entry and exit notionals in ledger operation order."""
    return (exit_price * signed_quantity - entry_price * signed_quantity) * multiplier


def _add_with_zero_cancellation(left: float, right: float) -> float:
    """Add values while collapsing tolerance-equivalent opposite amounts to zero."""
    if math.copysign(1.0, left) != math.copysign(1.0, right) and math.isclose(
        abs(left), abs(right), rel_tol=1e-9, abs_tol=1e-12
    ):
        return 0.0
    return left + right


@dataclass
class FillContext:
    """Context for a single fill execution.

    Encapsulates all the data needed to execute a fill without
    passing many individual parameters between methods.
    """

    order: Order
    current_time: datetime  # Validated timestamp for fill
    fill_quantity: float
    fill_price: float
    commission: float
    slippage: float
    signed_qty: float  # fill_quantity with sign (positive=buy, negative=sell)
    is_partial: bool
    price_source: str
    quote_context: dict[str, float | None]
    close_commission: float | None = None
    open_commission: float | None = None


class FillExecutor:
    """Orchestrates order fill execution.

    Extracts fill execution logic from Broker into a focused class with
    helper methods for each type of position change:
    - create_position: New position from flat
    - close_position: Close existing position to flat
    - flip_position: Reverse position (long→short or short→long)
    - scale_position: Add to or reduce existing position

    Example:
        >>> executor = FillExecutor(broker)
        >>> fully_filled = executor.execute(order, base_price=100.0)
    """

    def __init__(
        self,
        broker: Broker,
        *,
        account,
        market: MarketState,
        orders: OrderState,
        risk: RiskState,
        journal: ExecutionJournal,
        record_pnl: Callable[[str, float], None],
    ):
        """Initialize with broker instance.

        Args:
            broker: The Broker instance whose state we'll modify
        """
        self.broker = broker
        self.account = account
        self.market = market
        self.orders = orders
        self.risk = risk
        self.journal = journal
        self.record_pnl = record_pnl
        self.fill_engine: FillEngine | None = None

    def execute(self, order: Order, base_price: float) -> bool:
        """Execute a fill and update positions.

        This is the main entry point, replacing Broker._execute_fill().

        Args:
            order: Order to fill
            base_price: Base fill price before adjustments

        Returns:
            True if order is fully filled, False if partially filled

        Raises:
            ValueError: If an execution model returns a non-finite, non-positive,
                negative, or directionally favorable value outside its contract.
        """
        broker = self.broker
        current_time = self.market.time
        assert current_time is not None, "Cannot execute fill without current time"

        self._validate_execution_price(base_price, source="base execution price")

        available_size = broker.get_available_size(order.asset, order.side)

        # Get effective quantity (considering partial fills from previous bars)
        if self.fill_engine is None:
            raise RuntimeError("FillExecutor is not connected to a FillEngine")
        effective_quantity = self.fill_engine.get_effective_quantity(order)
        fill_quantity = effective_quantity

        # Apply execution limits (volume participation)
        remaining_quantity = 0.0
        if broker.execution_limits is not None:
            if order.order_id in self.orders.filled_this_bar:
                return False

            exec_result = broker.execution_limits.calculate(
                effective_quantity,
                available_size,
                base_price,
            )
            fill_quantity = exec_result.fillable_quantity

            if not math.isfinite(fill_quantity) or fill_quantity < 0:
                raise ValueError(
                    "Invalid execution quantity from "
                    f"{type(broker.execution_limits).__name__}: got {fill_quantity!r}"
                )
            if broker.share_type == ShareType.INTEGER:
                fill_quantity = float(int(fill_quantity))
            if fill_quantity == 0:
                return False

            remaining_quantity = max(0.0, effective_quantity - fill_quantity)
            if broker.share_type == ShareType.INTEGER:
                remaining_quantity = float(int(remaining_quantity))

        # Apply market impact
        if broker.market_impact_model is not None:
            is_buy = order.side == OrderSide.BUY
            impact = broker.market_impact_model.calculate(
                fill_quantity,
                base_price,
                available_size,
                is_buy,
            )
            self._validate_market_impact(impact, is_buy=is_buy)
            base_price = base_price + impact
            self._validate_execution_price(base_price, source="market-impact execution price")

        # Calculate slippage
        slippage = calculate_slippage(
            broker.slippage_model,
            order.asset,
            fill_quantity,
            base_price,
            available_size,
        )
        fill_price = base_price + slippage if order.side == OrderSide.BUY else base_price - slippage
        self._validate_execution_price(fill_price, source="execution price")

        # Calculate commission
        commission = calculate_commission(
            broker.commission_model, order.asset, fill_quantity, fill_price
        )
        quote_context = broker.get_quote_context(order.asset, order.side)

        signed_qty = fill_quantity if order.side == OrderSide.BUY else -fill_quantity
        close_commission = None
        open_commission = None
        position = self.account.positions.get(order.asset)
        is_exit_fill = position is not None and position.quantity * signed_qty < 0
        if position is not None:
            new_qty = position.quantity + signed_qty
            if _is_position_flip(position.quantity, new_qty):
                close_commission = calculate_commission(
                    broker.commission_model,
                    order.asset,
                    abs(position.quantity),
                    fill_price,
                )
                open_commission = calculate_commission(
                    broker.commission_model,
                    order.asset,
                    abs(new_qty),
                    fill_price,
                )

        if broker.execution_limits is not None:
            self.orders.filled_this_bar.add(order.order_id)
            if remaining_quantity > 0:
                self.orders.partial_quantities[order.order_id] = remaining_quantity
            else:
                self.orders.partial_quantities.pop(order.order_id, None)

        # Create fill record
        fill = Fill(
            order_id=order.order_id,
            rebalance_id=order.rebalance_id,
            asset=order.asset,
            side=order.side,
            quantity=fill_quantity,
            price=fill_price,
            timestamp=current_time,
            commission=commission,
            slippage=slippage,
            order_type=order.order_type.value,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            price_source=broker.execution_price.value,
            reference_price=quote_context["reference_price"],
            quote_mid_price=quote_context["quote_mid_price"],
            bid_price=quote_context["bid_price"],
            ask_price=quote_context["ask_price"],
            spread=quote_context["spread"],
            bid_size=quote_context["bid_size"],
            ask_size=quote_context["ask_size"],
            available_size=quote_context["available_size"],
            exit_reason=_get_exit_reason(order) if is_exit_fill else "",
            exit_reason_detail=order._risk_exit_reason,
        )
        self.journal.fills.append(fill)

        # Determine if partial fill
        is_partial = order.order_id in self.orders.partial_quantities
        previous_filled_quantity = order.filled_quantity
        cumulative_filled_quantity = previous_filled_quantity + fill_quantity
        previous_fill_notional = (order.filled_price or 0.0) * previous_filled_quantity
        order.filled_price = (
            previous_fill_notional + fill_price * fill_quantity
        ) / cumulative_filled_quantity
        order.filled_quantity = cumulative_filled_quantity
        if not is_partial:
            order.status = OrderStatus.FILLED
            order.filled_at = current_time

        # Build fill context
        ctx = FillContext(
            order=order,
            current_time=current_time,
            fill_quantity=fill_quantity,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage,
            signed_qty=signed_qty,
            is_partial=is_partial,
            price_source=broker.execution_price.value,
            quote_context=quote_context,
            close_commission=close_commission,
            open_commission=open_commission,
        )

        old_position = self.account.positions.get(order.asset)
        old_quantity = old_position.quantity if old_position is not None else 0.0
        old_entry_price = old_position.entry_price if old_position is not None else 0.0

        # Update position and get actual commission (may change for flips)
        actual_commission = self._update_position(ctx)
        fill.commission = actual_commission

        self._update_lock_notional_free_cash(
            ctx,
            old_quantity=old_quantity,
            old_entry_price=old_entry_price,
            commission=actual_commission,
        )

        # Update cash (include multiplier for futures/derivatives)
        multiplier = broker.get_multiplier(order.asset)
        cash_change = -signed_qty * fill_price * multiplier - actual_commission
        broker.cash += cash_change

        # Sync position to AccountState using execution price for this fill.
        # In next-bar/open execution this avoids close-price mark-to-market
        # leaking into same-cycle buying-power checks.
        self._sync_account_state(order.asset, current_price=ctx.fill_price)

        # Update account cash
        self.account.cash = broker.cash

        # Settlement delay: hold sale proceeds until settlement completes
        if broker.settlement_delay > 0 and cash_change > 0:
            self.account.add_settlement_hold(
                self.market.bar_index, broker.settlement_delay, cash_change
            )

        # Cancel sibling bracket orders on full fill
        if order.parent_id and not is_partial:
            for o in self.orders.pending[:]:
                if o.parent_id == order.parent_id and o.order_id != order.order_id:
                    o.status = OrderStatus.CANCELLED
                    self.orders.pending.remove(o)

        return not is_partial

    def _update_lock_notional_free_cash(
        self,
        ctx: FillContext,
        *,
        old_quantity: float,
        old_entry_price: float,
        commission: float,
    ) -> None:
        broker = self.broker
        policy = self.account.policy
        if policy.allow_leverage or policy.short_cash_policy != "lock_notional":
            return

        remaining = ctx.signed_qty
        free_cash = self.account._lock_notional_free_cash
        multiplier = broker.get_multiplier(ctx.order.asset)

        if old_quantity < 0.0 and remaining > 0.0:
            covered = min(remaining, abs(old_quantity))
            released_basis = covered * old_entry_price * multiplier
            required_cash = covered * ctx.fill_price * multiplier
            free_cash = _add_with_zero_cancellation(free_cash, 2.0 * released_basis - required_cash)
            remaining -= covered
        elif old_quantity > 0.0 and remaining < 0.0:
            closed = min(abs(remaining), old_quantity)
            free_cash = _add_with_zero_cancellation(free_cash, closed * ctx.fill_price * multiplier)
            remaining += closed

        if remaining != 0.0:
            required_cash = abs(remaining) * ctx.fill_price * multiplier
            free_cash = _add_with_zero_cancellation(free_cash, -required_cash)

        self.account._lock_notional_free_cash = _add_with_zero_cancellation(free_cash, -commission)

    @staticmethod
    def _validate_execution_price(value: float, *, source: str) -> None:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Invalid {source}: expected a finite positive number, got {value!r}")

    def _validate_market_impact(self, value: float, *, is_buy: bool) -> None:
        model_name = type(self.broker.market_impact_model).__name__
        if not math.isfinite(value):
            raise ValueError(
                f"Invalid market impact from {model_name}: expected a finite adverse value, "
                f"got {value!r}"
            )
        wrong_direction = (is_buy and value < 0.0) or (not is_buy and value > 0.0)
        if wrong_direction:
            expected = ">= 0 for buys" if is_buy else "<= 0 for sells"
            raise ValueError(
                f"Invalid market impact from {model_name}: expected {expected}, got {value!r}"
            )

    def _update_position(self, ctx: FillContext) -> float:
        """Update position based on fill.

        Args:
            ctx: Fill context with all execution details

        Returns:
            Actual commission charged (may differ from ctx.commission for flips)
        """
        pos = self.account.positions.get(ctx.order.asset)

        if pos is None:
            if ctx.signed_qty != 0:
                self._create_position(ctx)
            return ctx.commission
        else:
            old_qty = pos.quantity
            new_qty = old_qty + ctx.signed_qty
            if abs(new_qty) <= quantity_zero_tolerance(old_qty, ctx.signed_qty):
                new_qty = 0.0

            if new_qty == 0:
                self._close_position(ctx, pos, old_qty)
                return ctx.commission
            elif _is_position_flip(old_qty, new_qty):
                return self._flip_position(ctx, pos, old_qty, new_qty)
            else:
                self._scale_position(ctx, pos, old_qty, new_qty)
                return ctx.commission

    def _get_initial_hwm(self, asset: str, fill_price: float) -> float:
        """Get initial high water mark based on configuration.

        This is the single source of truth for HWM initialization,
        eliminating the duplication that existed in _execute_fill().

        Args:
            asset: Asset symbol
            fill_price: Fill price (default fallback)

        Returns:
            Initial HWM value based on configuration
        """
        broker = self.broker
        if broker.initial_hwm_source == InitialHwmSource.BAR_HIGH:
            return self.market.highs.get(asset, fill_price)
        elif broker.initial_hwm_source == InitialHwmSource.BAR_CLOSE:
            return self.market.closes.get(asset, self.market.prices.get(asset, fill_price))
        else:
            return fill_price

    def _get_initial_lwm(self, asset: str, fill_price: float) -> float:
        """Get initial low water mark based on configuration.

        For VBT Pro compatibility with OHLC data, LWM should be initialized
        from the entry bar's LOW price, not the high. This is critical for
        short positions where trailing stops use LWM.

        Args:
            asset: Asset symbol
            fill_price: Fill price (default fallback)

        Returns:
            Initial LWM value based on configuration
        """
        broker = self.broker
        # When using BAR_HIGH for HWM, use BAR_LOW for LWM
        if broker.initial_hwm_source == InitialHwmSource.BAR_HIGH:
            return self.market.lows.get(asset, fill_price)
        elif broker.initial_hwm_source == InitialHwmSource.BAR_CLOSE:
            return self.market.closes.get(asset, self.market.prices.get(asset, fill_price))
        else:
            return fill_price

    def _build_position_context(self, order: Order) -> dict:
        """Build position context with signal_price.

        This is the single source of truth for context building,
        eliminating the duplication that existed in _execute_fill().

        Args:
            order: Order with optional _signal_price

        Returns:
            Context dict for Position
        """
        broker = self.broker
        signal_price = getattr(order, "_signal_price", None)
        context = {
            "stop_fill_mode": broker.stop_fill_mode,
            "stop_level_basis": broker.stop_level_basis,
            "trail_hwm_source": broker.trail_hwm_source,
            "trail_stop_timing": broker.trail_stop_timing,
            "entry_quote_context": broker.get_quote_context(order.asset, order.side),
        }
        if signal_price is not None:
            context["signal_price"] = signal_price
        return context

    def _create_position(self, ctx: FillContext) -> None:
        """Create a new position from flat.

        Args:
            ctx: Fill context
        """
        broker = self.broker
        order = ctx.order

        initial_hwm = self._get_initial_hwm(order.asset, ctx.fill_price)
        initial_lwm = self._get_initial_lwm(order.asset, ctx.fill_price)
        context = self._build_position_context(order)

        pos = Position(
            asset=order.asset,
            quantity=ctx.signed_qty,
            entry_price=ctx.fill_price,
            entry_time=ctx.current_time,
            context=context,
            multiplier=broker.get_multiplier(order.asset),
            entry_commission=ctx.commission,
            entry_slippage=ctx.slippage,
            high_water_mark=initial_hwm,
            low_water_mark=initial_lwm,
        )
        self.account.positions[order.asset] = pos
        self.risk.positions_created_this_bar.add(order.asset)

    def _close_position(self, ctx: FillContext, pos: Position, old_qty: float) -> None:
        """Close an existing position to flat.

        Args:
            ctx: Fill context
            pos: Position being closed
            old_qty: Original position quantity
        """
        order = ctx.order

        # PnL includes both entry and exit commission, and multiplier for futures
        total_commission = pos.entry_commission + ctx.commission
        pnl = (
            _calculate_position_pnl(pos.entry_price, ctx.fill_price, old_qty, pos.multiplier)
            - total_commission
        )
        raw_pct = (ctx.fill_price - pos.entry_price) / pos.entry_price if pos.entry_price else 0.0
        pnl_pct = raw_pct if old_qty > 0 else -raw_pct
        entry_quote = pos.context.get("entry_quote_context", {})
        exit_quote = ctx.quote_context

        trade = Trade(
            symbol=order.asset,  # Order.asset -> Trade.symbol
            entry_time=pos.entry_time,
            exit_time=ctx.current_time,
            entry_price=pos.entry_price,
            exit_price=ctx.fill_price,
            quantity=old_qty,
            pnl=pnl,
            pnl_percent=pnl_pct,
            bars_held=pos.bars_held,
            fees=total_commission,
            exit_slippage=ctx.slippage,
            exit_reason=_get_exit_reason(order),
            exit_reason_detail=order._risk_exit_reason,
            mfe=pos.max_favorable_excursion,
            mae=pos.max_adverse_excursion,
            entry_slippage=pos.entry_slippage,
            multiplier=pos.multiplier,
            entry_quote_mid_price=entry_quote.get("quote_mid_price"),
            entry_bid_price=entry_quote.get("bid_price"),
            entry_ask_price=entry_quote.get("ask_price"),
            entry_spread=entry_quote.get("spread"),
            entry_available_size=entry_quote.get("available_size"),
            exit_quote_mid_price=exit_quote.get("quote_mid_price"),
            exit_bid_price=exit_quote.get("bid_price"),
            exit_ask_price=exit_quote.get("ask_price"),
            exit_spread=exit_quote.get("spread"),
            exit_available_size=exit_quote.get("available_size"),
        )
        self.journal.trades.append(trade)
        del self.account.positions[order.asset]

        # Record P&L event for trading stats
        self.record_pnl(order.asset, pnl)

    def _flip_position(
        self, ctx: FillContext, pos: Position, old_qty: float, new_qty: float
    ) -> float:
        """Handle position flip (long→short or short→long).

        Args:
            ctx: Fill context
            pos: Position being flipped
            old_qty: Original position quantity
            new_qty: New position quantity (opposite sign)

        Returns:
            Total commission charged (close + open portions)
        """
        broker = self.broker
        order = ctx.order

        # Calculate separate commissions for close and open portions
        if ctx.close_commission is None or ctx.open_commission is None:
            raise RuntimeError("Position flip commissions were not calculated before mutation")
        close_commission = ctx.close_commission
        open_commission = ctx.open_commission
        total_commission = close_commission + open_commission

        # Close the old position (include multiplier for futures)
        total_close_commission = pos.entry_commission + close_commission
        pnl = (
            _calculate_position_pnl(pos.entry_price, ctx.fill_price, old_qty, pos.multiplier)
            - total_close_commission
        )
        raw_pct = (ctx.fill_price - pos.entry_price) / pos.entry_price if pos.entry_price else 0.0
        pnl_pct = raw_pct if old_qty > 0 else -raw_pct
        entry_quote = pos.context.get("entry_quote_context", {})
        exit_quote = ctx.quote_context

        trade = Trade(
            symbol=order.asset,  # Order.asset -> Trade.symbol
            entry_time=pos.entry_time,
            exit_time=ctx.current_time,
            entry_price=pos.entry_price,
            exit_price=ctx.fill_price,
            quantity=old_qty,
            pnl=pnl,
            pnl_percent=pnl_pct,
            bars_held=pos.bars_held,
            fees=total_close_commission,
            exit_slippage=ctx.slippage,
            exit_reason=_get_exit_reason(order),
            exit_reason_detail=order._risk_exit_reason,
            mfe=pos.max_favorable_excursion,
            mae=pos.max_adverse_excursion,
            entry_slippage=pos.entry_slippage,
            multiplier=pos.multiplier,
            entry_quote_mid_price=entry_quote.get("quote_mid_price"),
            entry_bid_price=entry_quote.get("bid_price"),
            entry_ask_price=entry_quote.get("ask_price"),
            entry_spread=entry_quote.get("spread"),
            entry_available_size=entry_quote.get("available_size"),
            exit_quote_mid_price=exit_quote.get("quote_mid_price"),
            exit_bid_price=exit_quote.get("bid_price"),
            exit_ask_price=exit_quote.get("ask_price"),
            exit_spread=exit_quote.get("spread"),
            exit_available_size=exit_quote.get("available_size"),
        )
        self.journal.trades.append(trade)

        # Record P&L event for trading stats (flip = close old position)
        self.record_pnl(order.asset, pnl)

        # Create new position in opposite direction
        initial_hwm = self._get_initial_hwm(order.asset, ctx.fill_price)
        initial_lwm = self._get_initial_lwm(order.asset, ctx.fill_price)
        context = self._build_position_context(order)

        self.account.positions[order.asset] = Position(
            asset=order.asset,
            quantity=new_qty,
            entry_price=ctx.fill_price,
            entry_time=ctx.current_time,
            context=context,
            multiplier=broker.get_multiplier(order.asset),
            entry_commission=open_commission,
            entry_slippage=ctx.slippage,
            high_water_mark=initial_hwm,
            low_water_mark=initial_lwm,
        )
        self.risk.positions_created_this_bar.add(order.asset)

        # Cancel all other pending orders for this asset
        for pending_order in list(self.orders.pending):
            if pending_order.asset == order.asset and pending_order.order_id != order.order_id:
                pending_order.status = OrderStatus.CANCELLED
                self.orders.pending.remove(pending_order)

        return total_commission

    def _scale_position(
        self, ctx: FillContext, pos: Position, old_qty: float, new_qty: float
    ) -> None:
        """Scale an existing position up or down.

        Args:
            ctx: Fill context
            pos: Position being scaled
            old_qty: Original position quantity
            new_qty: New position quantity (same sign)
        """
        if abs(new_qty) < abs(old_qty):
            # Scaling down - this is a partial exit, calculate and record P&L
            exited_qty = abs(old_qty) - abs(new_qty)

            # Calculate P&L for the exited portion (include multiplier for futures)
            signed_exited_qty = math.copysign(exited_qty, old_qty)
            pnl = _calculate_position_pnl(
                pos.entry_price,
                ctx.fill_price,
                signed_exited_qty,
                pos.multiplier,
            )

            # Allocate the current position's entry costs in proportion to the quantity
            # removed. The residual cost remains attached to the residual position.
            exit_portion_ratio = exited_qty / abs(old_qty)
            proportional_entry_commission = pos.entry_commission * exit_portion_ratio
            pos.entry_commission -= proportional_entry_commission
            partial_exit_commission = ctx.commission
            total_commission = proportional_entry_commission + partial_exit_commission
            pnl -= total_commission

            raw_pct = (
                (ctx.fill_price - pos.entry_price) / pos.entry_price if pos.entry_price else 0.0
            )
            pnl_pct = raw_pct if old_qty > 0 else -raw_pct
            entry_quote = pos.context.get("entry_quote_context", {})
            exit_quote = ctx.quote_context
            self.journal.trades.append(
                Trade(
                    symbol=ctx.order.asset,
                    entry_time=pos.entry_time,
                    exit_time=ctx.current_time,
                    entry_price=pos.entry_price,
                    exit_price=ctx.fill_price,
                    quantity=math.copysign(exited_qty, old_qty),
                    pnl=pnl,
                    pnl_percent=pnl_pct,
                    bars_held=pos.bars_held,
                    fees=total_commission,
                    exit_slippage=ctx.slippage,
                    exit_reason=_get_exit_reason(ctx.order),
                    exit_reason_detail=ctx.order._risk_exit_reason,
                    status="partial",
                    mfe=pos.max_favorable_excursion,
                    mae=pos.max_adverse_excursion,
                    entry_slippage=pos.entry_slippage,
                    multiplier=pos.multiplier,
                    entry_quote_mid_price=entry_quote.get("quote_mid_price"),
                    entry_bid_price=entry_quote.get("bid_price"),
                    entry_ask_price=entry_quote.get("ask_price"),
                    entry_spread=entry_quote.get("spread"),
                    entry_available_size=entry_quote.get("available_size"),
                    exit_quote_mid_price=exit_quote.get("quote_mid_price"),
                    exit_bid_price=exit_quote.get("bid_price"),
                    exit_ask_price=exit_quote.get("ask_price"),
                    exit_spread=exit_quote.get("spread"),
                    exit_available_size=exit_quote.get("available_size"),
                )
            )

            # Record P&L event for trading stats
            self.record_pnl(ctx.order.asset, pnl)

        elif abs(new_qty) > abs(old_qty):
            # Scaling up - recalculate average entry price
            total_cost = pos.entry_price * abs(old_qty) + ctx.fill_price * abs(ctx.signed_qty)
            pos.entry_price = total_cost / abs(new_qty)
            # Accumulate entry-side costs so eventual close trade includes all entry legs.
            pos.entry_commission += ctx.commission
            if abs(new_qty) > 0:
                total_entry_slippage = pos.entry_slippage * abs(old_qty) + ctx.slippage * abs(
                    ctx.signed_qty
                )
                pos.entry_slippage = total_entry_slippage / abs(new_qty)

        pos.quantity = new_qty

    def _sync_account_state(self, asset: str, current_price: float | None = None) -> None:
        """Apply the fill mark to the canonical AccountState position.

        Args:
            asset: Asset to sync
            current_price: Optional mark price for account sync; defaults to latest
                broker close price when not provided.
        """
        broker = self.broker
        position = self.account.positions.get(asset)
        if position is None:
            return
        mark_price = (
            current_price
            if current_price is not None
            else broker.get_mark_price(asset, quantity=position.quantity) or position.entry_price
        )
        position.current_price = mark_price
