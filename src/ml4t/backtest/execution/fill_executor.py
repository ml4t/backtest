"""Fill execution orchestration.

This module provides FillExecutor which handles order fill execution,
extracting the logic from Broker._execute_fill() into a focused class
with helper methods for position creation, closing, flipping, and scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from ..config import InitialHwmSource
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


def _get_exit_reason(order: Order) -> str:
    """Get exit reason from order, preferring typed enum over string parsing.

    Priority:
    1. order._exit_reason (ExitReason enum) - preferred, set by broker
    2. order._risk_exit_reason (str) - legacy, parsed for backward compatibility
    3. ExitReason.SIGNAL - default for strategy-initiated exits

    Args:
        order: Order with exit reason metadata

    Returns:
        ExitReason enum value as string
    """
    # Prefer typed enum if available
    if order._exit_reason is not None:
        return order._exit_reason.value

    # Fall back to string parsing for backward compatibility
    reason = order._risk_exit_reason
    if reason is None:
        return ExitReason.SIGNAL.value

    reason_lower = reason.lower()
    if "stop_loss" in reason_lower:
        return ExitReason.STOP_LOSS.value
    elif "take_profit" in reason_lower:
        return ExitReason.TAKE_PROFIT.value
    elif "trailing" in reason_lower:
        return ExitReason.TRAILING_STOP.value
    elif "time" in reason_lower:
        return ExitReason.TIME_STOP.value
    elif "end_of_data" in reason_lower:
        return ExitReason.END_OF_DATA.value
    else:
        return ExitReason.SIGNAL.value


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

    def __init__(self, broker: Broker):
        """Initialize with broker instance.

        Args:
            broker: The Broker instance whose state we'll modify
        """
        self.broker = broker

    def execute(self, order: Order, base_price: float) -> bool:
        """Execute a fill and update positions.

        This is the main entry point, replacing Broker._execute_fill().

        Args:
            order: Order to fill
            base_price: Base fill price before adjustments

        Returns:
            True if order is fully filled, False if partially filled
        """
        broker = self.broker
        current_time = broker._current_time
        assert current_time is not None, "Cannot execute fill without current time"

        volume = broker._current_volumes.get(order.asset)

        # Get effective quantity (considering partial fills from previous bars)
        effective_quantity = broker._get_effective_quantity(order)
        fill_quantity = effective_quantity

        # Apply execution limits (volume participation)
        if broker.execution_limits is not None:
            if order.order_id in broker._filled_this_bar:
                return False

            exec_result = broker.execution_limits.calculate(effective_quantity, volume, base_price)
            fill_quantity = exec_result.fillable_quantity

            if fill_quantity <= 0:
                return False

            broker._filled_this_bar.add(order.order_id)

            if exec_result.remaining_quantity > 0:
                broker._partial_orders[order.order_id] = exec_result.remaining_quantity
            else:
                broker._partial_orders.pop(order.order_id, None)

        # Apply market impact
        if broker.market_impact_model is not None:
            is_buy = order.side == OrderSide.BUY
            impact = broker.market_impact_model.calculate(fill_quantity, base_price, volume, is_buy)
            base_price = base_price + impact

        # Calculate slippage
        slippage = broker.slippage_model.calculate(order.asset, fill_quantity, base_price, volume)
        fill_price = base_price + slippage if order.side == OrderSide.BUY else base_price - slippage

        # Calculate commission
        commission = broker.commission_model.calculate(order.asset, fill_quantity, fill_price)

        # Create fill record
        fill = Fill(
            order_id=order.order_id,
            asset=order.asset,
            side=order.side,
            quantity=fill_quantity,
            price=fill_price,
            timestamp=current_time,
            commission=commission,
            slippage=slippage,
        )
        broker.fills.append(fill)

        # Determine if partial fill
        is_partial = order.order_id in broker._partial_orders
        if is_partial:
            order.filled_quantity = (order.filled_quantity or 0) + fill_quantity
        else:
            order.status = OrderStatus.FILLED
            order.filled_at = current_time
            order.filled_price = fill_price
            order.filled_quantity = fill_quantity

        # Build fill context
        signed_qty = fill_quantity if order.side == OrderSide.BUY else -fill_quantity
        ctx = FillContext(
            order=order,
            current_time=current_time,
            fill_quantity=fill_quantity,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage,
            signed_qty=signed_qty,
            is_partial=is_partial,
        )

        # Update position and get actual commission (may change for flips)
        actual_commission = self._update_position(ctx)

        # Update cash
        cash_change = -signed_qty * fill_price - actual_commission
        broker.cash += cash_change

        # Sync position to AccountState
        self._sync_account_state(order.asset)

        # Update account cash
        broker.account.cash = broker.cash

        # Cancel sibling bracket orders on full fill
        if order.parent_id and not is_partial:
            for o in broker.pending_orders[:]:
                if o.parent_id == order.parent_id and o.order_id != order.order_id:
                    o.status = OrderStatus.CANCELLED
                    broker.pending_orders.remove(o)

        return not is_partial

    def _update_position(self, ctx: FillContext) -> float:
        """Update position based on fill.

        Args:
            ctx: Fill context with all execution details

        Returns:
            Actual commission charged (may differ from ctx.commission for flips)
        """
        broker = self.broker
        pos = broker.positions.get(ctx.order.asset)

        if pos is None:
            if ctx.signed_qty != 0:
                self._create_position(ctx)
            return ctx.commission
        else:
            old_qty = pos.quantity
            new_qty = old_qty + ctx.signed_qty

            if new_qty == 0:
                self._close_position(ctx, pos, old_qty)
                return ctx.commission
            elif (old_qty > 0) != (new_qty > 0):
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
            return broker._current_highs.get(asset, fill_price)
        elif broker.initial_hwm_source == InitialHwmSource.BAR_CLOSE:
            return broker._current_prices.get(asset, fill_price)
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
        signal_price = getattr(order, "_signal_price", None)
        return {"signal_price": signal_price} if signal_price is not None else {}

    def _create_position(self, ctx: FillContext) -> None:
        """Create a new position from flat.

        Args:
            ctx: Fill context
        """
        broker = self.broker
        order = ctx.order

        initial_hwm = self._get_initial_hwm(order.asset, ctx.fill_price)
        context = self._build_position_context(order)

        pos = Position(
            asset=order.asset,
            quantity=ctx.signed_qty,
            entry_price=ctx.fill_price,
            entry_time=ctx.current_time,
            context=context,
            multiplier=broker.get_multiplier(order.asset),
            entry_commission=ctx.commission,
            high_water_mark=initial_hwm,
            low_water_mark=initial_hwm,
        )
        broker.positions[order.asset] = pos
        broker._positions_created_this_bar.add(order.asset)

    def _close_position(self, ctx: FillContext, pos: Position, old_qty: float) -> None:
        """Close an existing position to flat.

        Args:
            ctx: Fill context
            pos: Position being closed
            old_qty: Original position quantity
        """
        broker = self.broker
        order = ctx.order

        # PnL includes both entry and exit commission
        total_commission = pos.entry_commission + ctx.commission
        pnl = (ctx.fill_price - pos.entry_price) * old_qty - total_commission
        pnl_pct = (ctx.fill_price - pos.entry_price) / pos.entry_price if pos.entry_price else 0

        trade = Trade(
            asset=order.asset,
            entry_time=pos.entry_time,
            exit_time=ctx.current_time,
            entry_price=pos.entry_price,
            exit_price=ctx.fill_price,
            quantity=old_qty,
            pnl=pnl,
            pnl_percent=pnl_pct,
            bars_held=pos.bars_held,
            commission=total_commission,
            slippage=ctx.slippage,
            exit_reason=_get_exit_reason(order),
            entry_signals=broker._current_signals.get(order.asset, {}),
            exit_signals=broker._current_signals.get(order.asset, {}),
            max_favorable_excursion=pos.max_favorable_excursion,
            max_adverse_excursion=pos.max_adverse_excursion,
        )
        broker.trades.append(trade)
        del broker.positions[order.asset]

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

        close_qty = abs(old_qty)
        open_qty = abs(new_qty)

        # Calculate separate commissions for close and open portions
        close_commission = broker.commission_model.calculate(order.asset, close_qty, ctx.fill_price)
        open_commission = broker.commission_model.calculate(order.asset, open_qty, ctx.fill_price)
        total_commission = close_commission + open_commission

        # Close the old position
        total_close_commission = pos.entry_commission + close_commission
        pnl = (ctx.fill_price - pos.entry_price) * old_qty - total_close_commission
        pnl_pct = (ctx.fill_price - pos.entry_price) / pos.entry_price if pos.entry_price else 0

        trade = Trade(
            asset=order.asset,
            entry_time=pos.entry_time,
            exit_time=ctx.current_time,
            entry_price=pos.entry_price,
            exit_price=ctx.fill_price,
            quantity=old_qty,
            pnl=pnl,
            pnl_percent=pnl_pct,
            bars_held=pos.bars_held,
            commission=total_close_commission,
            slippage=ctx.slippage * (close_qty / ctx.fill_quantity),
            exit_reason=_get_exit_reason(order),
            entry_signals=broker._current_signals.get(order.asset, {}),
            exit_signals=broker._current_signals.get(order.asset, {}),
            max_favorable_excursion=pos.max_favorable_excursion,
            max_adverse_excursion=pos.max_adverse_excursion,
        )
        broker.trades.append(trade)

        # Create new position in opposite direction
        initial_hwm = self._get_initial_hwm(order.asset, ctx.fill_price)
        context = self._build_position_context(order)

        broker.positions[order.asset] = Position(
            asset=order.asset,
            quantity=new_qty,
            entry_price=ctx.fill_price,
            entry_time=ctx.current_time,
            context=context,
            multiplier=broker.get_multiplier(order.asset),
            entry_commission=open_commission,
            high_water_mark=initial_hwm,
            low_water_mark=initial_hwm,
        )
        broker._positions_created_this_bar.add(order.asset)

        # Cancel all other pending orders for this asset
        for pending_order in list(broker.pending_orders):
            if pending_order.asset == order.asset and pending_order.order_id != order.order_id:
                pending_order.status = OrderStatus.CANCELLED
                broker.pending_orders.remove(pending_order)

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
        if abs(new_qty) > abs(old_qty):
            # Scaling up - recalculate average entry price
            total_cost = pos.entry_price * abs(old_qty) + ctx.fill_price * abs(ctx.signed_qty)
            pos.entry_price = total_cost / abs(new_qty)
        pos.quantity = new_qty

    def _sync_account_state(self, asset: str) -> None:
        """Sync broker position to AccountState.

        Args:
            asset: Asset to sync
        """
        broker = self.broker
        broker_pos = broker.positions.get(asset)

        if broker_pos is None:
            # Position was closed, remove from account
            if asset in broker.account.positions:
                del broker.account.positions[asset]
        else:
            # Update or create position in account
            broker.account.positions[asset] = Position(
                asset=broker_pos.asset,
                quantity=broker_pos.quantity,
                entry_price=broker_pos.entry_price,
                current_price=broker._current_prices.get(asset, broker_pos.entry_price),
                entry_time=broker_pos.entry_time,
                bars_held=broker_pos.bars_held,
            )
