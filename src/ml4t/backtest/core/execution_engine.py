"""Order execution sequencing extracted from Broker."""

from __future__ import annotations

import copy

from ..types import ExecutionMode, OrderSide, OrderStatus, OrderType, Position
from .shared import is_exit_order


class ExecutionEngine:
    """Executes pending orders using configured fill ordering."""

    def __init__(self, broker):
        self.broker = broker

    def process_orders(self, use_open: bool = False):
        if self._should_use_next_bar_queue_shadow_validation(use_open):
            self._process_orders_next_bar_queue_shadow(use_open)
            return

        ordering = self.broker.fill_ordering.value
        if ordering == "exit_first":
            self._process_orders_exit_first(use_open)
        elif ordering == "sequential":
            self._process_orders_sequential(use_open)
        else:
            self._process_orders_fifo(use_open)

    def _is_exit_order(self, order) -> bool:
        """Check if an order reduces an existing position without reversing."""
        return is_exit_order(order, self.broker.positions)

    def _process_orders_exit_first(self, use_open: bool = False):
        broker = self.broker
        fill = broker._fill_engine
        exit_orders = []
        entry_orders = []
        eligible_orders = []
        orders_this_bar_ids = broker._orders_this_bar_ids

        for order in broker.pending_orders[:]:
            if (
                broker.execution_mode is ExecutionMode.NEXT_BAR
                and order.order_id in orders_this_bar_ids
            ):
                continue
            eligible_orders.append(order)
            if self._is_exit_order(order):
                exit_orders.append(order)
            else:
                entry_orders.append(order)

        filled_orders: list = []
        deferred_entries: list = []

        for order in exit_orders:
            if order.status is not OrderStatus.PENDING:
                continue
            # Exit classification must be re-evaluated at fill time. Multiple
            # queued orders for the same asset can exhaust the current
            # position; later orders then become reversals/new entries and must
            # go through the normal entry validation path.
            if not self._is_exit_order(order):
                deferred_entries.append(order)
                continue
            # Exit-first processing needs the same integer-share quantization as
            # the generic per-order path; rebalance deltas often arrive as
            # fractional target values even when the broker disallows them.
            fill.apply_share_rounding(order)
            if order.quantity <= 0:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = "Quantity rounds to zero (share_type=INTEGER)"
                continue
            price = fill.get_fill_price_for_order(order, use_open)
            if price is None:
                continue
            fill_price = fill.check_fill(order, price)
            if fill_price is not None:
                fully_filled = fill.execute_fill(order, fill_price)
                if fully_filled:
                    filled_orders.append(order)
                    broker._partial_orders.pop(order.order_id, None)
                else:
                    fill.update_partial_order(order)

        broker.mark_account_positions(use_open=use_open)
        if deferred_entries:
            entry_order_ids = {order.order_id for order in entry_orders}
            entry_order_ids.update(order.order_id for order in deferred_entries)
            entry_orders = [order for order in eligible_orders if order.order_id in entry_order_ids]
        entry_orders = self._sort_entry_orders(entry_orders, use_open=use_open)

        for order in entry_orders:
            self._process_single_order(order, use_open, filled_orders)

        self._cleanup_filled_orders(filled_orders)

    def _should_use_next_bar_queue_shadow_validation(self, use_open: bool) -> bool:
        broker = self.broker
        if not (
            use_open
            and broker.execution_mode is ExecutionMode.NEXT_BAR
            and broker.next_bar_queue_shadow_validation
        ):
            return False

        current_bar_index = broker._bar_index
        for order in broker.pending_orders:
            if order.order_id in broker._orders_this_bar_ids:
                continue
            if getattr(order, "_created_bar_index", current_bar_index) < current_bar_index - 1:
                return True

        return False

    def _process_orders_next_bar_queue_shadow(self, use_open: bool = False):
        broker = self.broker
        fill = broker._fill_engine
        eligible_orders = []
        orders_this_bar_ids = broker._orders_this_bar_ids

        for order in broker.pending_orders[:]:
            if (
                broker.execution_mode is ExecutionMode.NEXT_BAR
                and order.order_id in orders_this_bar_ids
            ):
                continue
            eligible_orders.append(order)

        if not eligible_orders:
            return

        shadow_cash = broker.account.cash
        shadow_positions = {
            asset: copy.deepcopy(position) for asset, position in broker.account.positions.items()
        }
        for asset, position in shadow_positions.items():
            mark = broker._current_prices.get(asset)
            if mark is not None:
                position.current_price = mark

        accepted_orders: list[tuple[object, float]] = []
        filled_orders: list = []

        for order in eligible_orders:
            price = fill.get_fill_price_for_order(order, use_open)
            if price is None:
                continue

            fill.apply_share_rounding(order)
            if order.quantity <= 0:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = "Quantity rounds to zero (share_type=INTEGER)"
                continue

            fill_price = fill.check_fill(order, price)
            if fill_price is None:
                continue

            validation_price = broker._current_prices.get(order.asset, fill_price)
            valid, rejection_reason = self._validate_shadow_queue_order(
                order=order,
                validation_price=validation_price,
                shadow_cash=shadow_cash,
                shadow_positions=shadow_positions,
            )
            if not valid:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = rejection_reason
                continue

            accepted_orders.append((order, fill_price))
            shadow_cash = self._commit_shadow_queue_fill(
                order=order,
                fill_price=fill_price,
                shadow_cash=shadow_cash,
                shadow_positions=shadow_positions,
            )

        for order, fill_price in accepted_orders:
            fully_filled = fill.execute_fill(order, fill_price)
            if fully_filled:
                filled_orders.append(order)
                broker._partial_orders.pop(order.order_id, None)
            else:
                fill.update_partial_order(order)
            broker.mark_account_positions(use_open=False)

        self._cleanup_filled_orders(filled_orders)

    def _validate_shadow_queue_order(
        self,
        *,
        order,
        validation_price: float,
        shadow_cash: float,
        shadow_positions: dict[str, Position],
    ) -> tuple[bool, str]:
        broker = self.broker
        policy = broker.account.policy
        qty_delta = order.quantity if order.side is OrderSide.BUY else -order.quantity
        current_qty = (
            shadow_positions[order.asset].quantity if order.asset in shadow_positions else 0.0
        )
        new_qty = current_qty + qty_delta
        is_reversal = (
            abs(current_qty) > 1e-12
            and abs(new_qty) > 1e-12
            and ((current_qty > 0 and new_qty < 0) or (current_qty < 0 and new_qty > 0))
        )
        commission = broker.commission_model.calculate(
            order.asset, order.quantity, validation_price
        )

        if abs(current_qty) <= 1e-12:
            return policy.validate_new_position(
                asset=order.asset,
                quantity=qty_delta,
                price=validation_price,
                current_positions=shadow_positions,
                cash=shadow_cash - commission,
            )
        if is_reversal:
            return policy.handle_reversal(
                asset=order.asset,
                current_quantity=current_qty,
                order_quantity_delta=qty_delta,
                price=validation_price,
                current_positions=shadow_positions,
                cash=shadow_cash,
                commission=commission,
            )
        return policy.validate_position_change(
            asset=order.asset,
            current_quantity=current_qty,
            quantity_delta=qty_delta,
            price=validation_price,
            current_positions=shadow_positions,
            cash=shadow_cash - commission,
        )

    def _commit_shadow_queue_fill(
        self,
        *,
        order,
        fill_price: float,
        shadow_cash: float,
        shadow_positions: dict[str, Position],
    ) -> float:
        broker = self.broker
        qty_delta = order.quantity if order.side is OrderSide.BUY else -order.quantity
        current_qty = (
            shadow_positions[order.asset].quantity if order.asset in shadow_positions else 0.0
        )
        new_qty = current_qty + qty_delta
        commission = broker.commission_model.calculate(order.asset, order.quantity, fill_price)
        shadow_cash += -qty_delta * fill_price * broker.get_multiplier(order.asset) - commission

        if abs(new_qty) <= 1e-12:
            shadow_positions.pop(order.asset, None)
            return shadow_cash

        position = shadow_positions.get(order.asset)
        if position is None:
            shadow_positions[order.asset] = Position(
                asset=order.asset,
                quantity=new_qty,
                entry_price=fill_price,
                entry_time=broker._current_time,
                current_price=broker._current_prices.get(order.asset, fill_price),
                multiplier=broker.get_multiplier(order.asset),
            )
            return shadow_cash

        is_reversal = (
            abs(current_qty) > 1e-12
            and abs(new_qty) > 1e-12
            and ((current_qty > 0 and new_qty < 0) or (current_qty < 0 and new_qty > 0))
        )
        position.quantity = new_qty
        position.current_price = broker._current_prices.get(order.asset, fill_price)
        if abs(current_qty) <= 1e-12 or is_reversal:
            position.entry_price = fill_price

        return shadow_cash

    def _process_orders_fifo(self, use_open: bool = False):
        broker = self.broker
        eligible_orders = []
        orders_this_bar_ids = broker._orders_this_bar_ids
        for order in broker.pending_orders[:]:
            if (
                broker.execution_mode is ExecutionMode.NEXT_BAR
                and order.order_id in orders_this_bar_ids
            ):
                continue
            eligible_orders.append(order)

        filled_orders: list = []

        for order in eligible_orders:
            self._process_single_order(order, use_open, filled_orders)
            if filled_orders and filled_orders[-1] is order:
                broker.mark_account_positions(use_open=use_open)

        self._cleanup_filled_orders(filled_orders)

    def _process_orders_sequential(self, use_open: bool = False):
        """Process orders in submission order without exit/entry separation.

        Each order is processed individually with mark-to-market after each fill,
        interleaving exits and entries in their original submission order.  This
        matches LEAN's per-order sequential buying-power model where alphabetical
        order determines which orders see freed cash from prior exits.

        When buying_power_reservation is enabled, orders have already been validated
        at submission time by the shadow cash pool.  In that case entries fill
        directly (skipping the gatekeeper) since the shadow IS the validation.
        This avoids double-validation that causes discrepancies between the shadow's
        sequential accounting and the gatekeeper's point-in-time accounting.
        """
        broker = self.broker
        fill = broker._fill_engine
        eligible_orders = []
        orders_this_bar_ids = broker._orders_this_bar_ids
        for order in broker.pending_orders[:]:
            if (
                broker.execution_mode is ExecutionMode.NEXT_BAR
                and order.order_id in orders_this_bar_ids
            ):
                continue
            eligible_orders.append(order)

        filled_orders: list = []
        # When buying_power_reservation is on, the shadow already validated
        # these orders at submission time — skip gatekeeper at fill time.
        shadow_validated = broker.buying_power_reservation

        for order in eligible_orders:
            if order.status is not OrderStatus.PENDING:
                continue

            price = fill.get_fill_price_for_order(order, use_open)
            if price is None:
                continue

            is_exit = self._is_exit_order(order)

            if is_exit or shadow_validated:
                # Exits always fill (they free capital).
                # Entries fill directly when shadow-validated at submission.
                fill.apply_share_rounding(order)
                if order.quantity <= 0:
                    order.status = OrderStatus.REJECTED
                    order.rejection_reason = "Quantity rounds to zero (share_type=INTEGER)"
                    continue
                fill_price = fill.check_fill(order, price)
                if fill_price is not None:
                    fully_filled = fill.execute_fill(order, fill_price)
                    if fully_filled:
                        filled_orders.append(order)
                        broker._partial_orders.pop(order.order_id, None)
                    else:
                        fill.update_partial_order(order)
            else:
                # No shadow validation — use full gatekeeper path
                self._process_single_order(order, use_open, filled_orders)

            # Mark-to-market after every fill so the next order sees updated cash
            if filled_orders and filled_orders[-1] is order:
                broker.mark_account_positions(use_open=use_open)

        self._cleanup_filled_orders(filled_orders)

    def _process_single_order(self, order, use_open: bool, filled_orders: list) -> None:
        broker = self.broker
        fill = broker._fill_engine
        if order.status is not OrderStatus.PENDING:
            return
        price = fill.get_fill_price_for_order(order, use_open)
        if price is None:
            return

        skip_cash = broker.skip_cash_validation
        use_simple_cash_check = not skip_cash and self._use_simple_next_bar_cash_check(
            order, use_open
        )

        is_exit = self._is_exit_order(order)

        if is_exit:
            # Exit orders can originate from fractional target-value deltas during
            # rebalances. Integer-share brokers must still quantize queued next-bar
            # exits before fill, just like entry orders and immediate fills.
            fill.apply_share_rounding(order)
            if order.quantity <= 0:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = "Quantity rounds to zero (share_type=INTEGER)"
                return

            fill_price = fill.check_fill(order, price)
            if fill_price is not None:
                if use_simple_cash_check and not self._passes_simple_cash_check(order, fill_price):
                    order.status = OrderStatus.REJECTED
                    order.rejection_reason = "Insufficient cash (open cash check)"
                    return

                # Under locked-short-cash semantics, short covers/reversals can be
                # cash-constrained and may require partial fills.
                if (
                    order.side is OrderSide.BUY
                    and broker.short_cash_policy.value == "lock_notional"
                    and broker.account.get_position_quantity(order.asset) < 0
                ):
                    max_qty = fill.get_max_affordable_quantity(order, fill_price)
                    if broker.share_type.value == "integer":
                        max_qty = float(int(max_qty))
                    if max_qty <= 0:
                        order.status = OrderStatus.REJECTED
                        order.rejection_reason = "Insufficient cash to cover short"
                        return
                    if max_qty < order.quantity:
                        if broker.partial_fills_allowed:
                            order.quantity = max_qty
                        else:
                            order.status = OrderStatus.REJECTED
                            order.rejection_reason = "Insufficient cash to cover short"
                            return

                fully_filled = fill.execute_fill(order, fill_price)
                if fully_filled:
                    filled_orders.append(order)
                    broker._partial_orders.pop(order.order_id, None)
                else:
                    fill.update_partial_order(order)
        else:
            fill.apply_share_rounding(order)
            if order.quantity <= 0:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = "Quantity rounds to zero (share_type=INTEGER)"
                return

            fill_price = fill.check_fill(order, price)
            if fill_price is None:
                return

            if use_simple_cash_check and not self._passes_simple_cash_check(order, fill_price):
                order.status = OrderStatus.REJECTED
                order.rejection_reason = "Insufficient cash (open cash check)"
                return

            # Under locked-short-cash semantics, reversal entries can be
            # cash-constrained and must follow partial-fill sizing.
            current_qty = broker.account.get_position_quantity(order.asset)
            is_reversal_entry = broker.short_cash_policy.value == "lock_notional" and (
                (order.side is OrderSide.BUY and current_qty < 0)
                or (order.side is OrderSide.SELL and current_qty > 0)
            )
            if is_reversal_entry:
                max_qty = fill.get_max_affordable_quantity(order, fill_price)
                if broker.share_type.value == "integer":
                    max_qty = float(int(max_qty))
                if max_qty <= 0:
                    order.status = OrderStatus.REJECTED
                    order.rejection_reason = "Insufficient cash for reversal"
                    return
                if max_qty < order.quantity:
                    if broker.partial_fills_allowed:
                        order.quantity = max_qty
                    else:
                        order.status = OrderStatus.REJECTED
                        order.rejection_reason = "Insufficient cash for reversal"
                        return

            if skip_cash or use_simple_cash_check:
                valid, rejection_reason = True, ""
            else:
                valid, rejection_reason = broker.gatekeeper.validate_order(order, fill_price)

            if valid:
                fully_filled = fill.execute_fill(order, fill_price)
                if fully_filled:
                    filled_orders.append(order)
                    broker._partial_orders.pop(order.order_id, None)
                else:
                    fill.update_partial_order(order)
            elif (
                not broker.reject_on_insufficient_cash
                and "insufficient" in rejection_reason.lower()
            ):
                if broker.partial_fills_allowed and fill.try_partial_fill(order, fill_price):
                    filled_orders.append(order)
                    broker._partial_orders.pop(order.order_id, None)
                else:
                    # Permissive mode: silently skip unaffordable orders for this cycle
                    # by cancelling them instead of keeping them pending forever.
                    order.status = OrderStatus.CANCELLED
            elif (
                broker.partial_fills_allowed
                and "insufficient" in rejection_reason.lower()
                or (
                    order.rebalance_id is not None
                    and broker.share_type.value == "integer"
                    and "insufficient" in rejection_reason.lower()
                )
            ):
                if fill.try_partial_fill(order, fill_price):
                    filled_orders.append(order)
                    broker._partial_orders.pop(order.order_id, None)
                else:
                    order.status = OrderStatus.REJECTED
                    order.rejection_reason = rejection_reason
            else:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = rejection_reason

    def _use_simple_next_bar_cash_check(self, order, use_open: bool) -> bool:
        broker = self.broker
        return (
            use_open
            and broker.execution_mode is ExecutionMode.NEXT_BAR
            and broker.next_bar_simple_cash_check
            and order.order_type is OrderType.MARKET
        )

    def _passes_simple_cash_check(self, order, fill_price: float) -> bool:
        broker = self.broker
        if broker.share_type.value == "integer":
            order.quantity = float(int(order.quantity))
            if order.quantity <= 0:
                return False

        current_qty = broker.account.get_position_quantity(order.asset)
        is_opposite = (order.side is OrderSide.BUY and current_qty < 0) or (
            order.side is OrderSide.SELL and current_qty > 0
        )
        if is_opposite and order.quantity <= abs(current_qty):
            return True

        signed_qty = order.quantity if order.side is OrderSide.BUY else -order.quantity
        commission = broker.commission_model.calculate(order.asset, order.quantity, fill_price)
        projected_cash = broker.cash - signed_qty * fill_price - commission
        return projected_cash >= 0.0

    def _cleanup_filled_orders(self, filled_orders: list) -> None:
        broker = self.broker
        filled_ids = {o.order_id for o in filled_orders}
        if filled_ids:
            broker._orders_this_bar_ids.difference_update(filled_ids)

        new_pending = []
        for order in broker.pending_orders:
            if order.order_id in filled_ids or order.status in {
                OrderStatus.REJECTED,
                OrderStatus.CANCELLED,
            }:
                broker._orders_this_bar_ids.discard(order.order_id)
                continue
            new_pending.append(order)
        broker.pending_orders = new_pending

        if broker._orders_this_bar:
            broker._orders_this_bar = [
                o for o in broker._orders_this_bar if o.order_id in broker._orders_this_bar_ids
            ]

    def _sort_entry_orders(self, orders: list, use_open: bool) -> list:
        """Sort entry orders under EXIT_FIRST based on configured priority."""
        broker = self.broker
        fill = broker._fill_engine
        priority = broker.entry_order_priority.value
        if priority == "submission":
            return orders

        def notional(order) -> float:
            px = fill.get_fill_price_for_order(order, use_open)
            if px is None:
                px = broker._current_prices.get(
                    order.asset, broker._current_opens.get(order.asset, 0.0)
                )
            return abs(order.quantity * px)

        reverse = priority == "notional_desc"
        return sorted(orders, key=notional, reverse=reverse)
