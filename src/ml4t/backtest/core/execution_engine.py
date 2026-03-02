"""Order execution sequencing extracted from Broker."""

from __future__ import annotations

from ..types import ExecutionMode, OrderSide, OrderStatus, OrderType


class ExecutionEngine:
    """Executes pending orders using configured fill ordering."""

    def __init__(self, broker):
        self.broker = broker

    def process_orders(self, use_open: bool = False):
        ordering = self.broker.fill_ordering.value
        if ordering == "exit_first":
            self._process_orders_exit_first(use_open)
        elif ordering == "sequential":
            self._process_orders_sequential(use_open)
        else:
            self._process_orders_fifo(use_open)

    def _is_exit_order(self, order) -> bool:
        """Check if an order reduces an existing position without reversing."""
        broker = self.broker
        pos = broker.positions.get(order.asset)
        if pos is None or pos.quantity == 0:
            return False

        signed_qty = order.quantity if order.side is OrderSide.BUY else -order.quantity

        if pos.quantity > 0 and signed_qty < 0:
            new_qty = pos.quantity + signed_qty
            return new_qty >= 0
        if pos.quantity < 0 and signed_qty > 0:
            new_qty = pos.quantity + signed_qty
            return new_qty <= 0
        return False

    def _process_orders_exit_first(self, use_open: bool = False):
        broker = self.broker
        fill = broker._fill_engine
        mark_prices = broker._current_opens if use_open else broker._current_prices
        exit_orders = []
        entry_orders = []
        orders_this_bar_ids = broker._orders_this_bar_ids

        for order in broker.pending_orders[:]:
            if (
                broker.execution_mode is ExecutionMode.NEXT_BAR
                and order.order_id in orders_this_bar_ids
            ):
                continue
            if self._is_exit_order(order):
                exit_orders.append(order)
            else:
                entry_orders.append(order)

        filled_orders: list = []

        for order in exit_orders:
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

        broker.account.mark_to_market(mark_prices)
        entry_orders = self._sort_entry_orders(entry_orders, use_open=use_open)

        for order in entry_orders:
            self._process_single_order(order, use_open, filled_orders)

        self._cleanup_filled_orders(filled_orders)

    def _process_orders_fifo(self, use_open: bool = False):
        broker = self.broker
        mark_prices = broker._current_opens if use_open else broker._current_prices
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
                broker.account.mark_to_market(mark_prices)

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
        mark_prices = broker._current_opens if use_open else broker._current_prices
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
                broker.account.mark_to_market(mark_prices)

        self._cleanup_filled_orders(filled_orders)

    def _process_single_order(self, order, use_open: bool, filled_orders: list) -> None:
        broker = self.broker
        fill = broker._fill_engine
        price = fill.get_fill_price_for_order(order, use_open)
        if price is None:
            return

        skip_cash = broker.skip_cash_validation
        use_simple_cash_check = not skip_cash and self._use_simple_next_bar_cash_check(
            order, use_open
        )

        is_exit = self._is_exit_order(order)

        if is_exit:
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
            elif broker.partial_fills_allowed and "insufficient" in rejection_reason.lower():
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

    def _precheck_next_bar_orders_fifo(self, orders: list, use_open: bool) -> list:
        """Apply submission-price cash precheck for NEXT_BAR FIFO processing.

        This emulates frameworks that reject some next-bar market orders based on
        signal-bar pricing before open-price execution.
        """
        broker = self.broker
        fill = broker._fill_engine
        shadow_cash = broker.cash
        accepted: list = []

        for order in orders:
            if order.order_type is not OrderType.MARKET:
                accepted.append(order)
                continue

            if broker.share_type.value == "integer":
                order.quantity = float(int(order.quantity))
            if order.quantity <= 0:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = "Quantity rounds to zero (share_type=INTEGER)"
                continue

            signal_price = getattr(order, "_signal_price", None)
            check_price = (
                signal_price
                if signal_price is not None
                else fill.get_fill_price_for_order(order, use_open)
            )
            if check_price is None:
                accepted.append(order)
                continue

            signed_qty = order.quantity if order.side is OrderSide.BUY else -order.quantity
            commission = broker.commission_model.calculate(order.asset, order.quantity, check_price)
            projected_cash = shadow_cash - signed_qty * check_price - commission

            if projected_cash < 0.0:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = "Insufficient cash (submission precheck)"
                continue

            shadow_cash = projected_cash
            accepted.append(order)

        return accepted

    def _sort_entry_orders(self, orders: list, use_open: bool) -> list:
        """Sort entry orders under EXIT_FIRST based on configured priority."""
        broker = self.broker
        priority = broker.entry_order_priority.value
        if priority == "submission":
            return orders

        prices = broker._current_opens if use_open else broker._current_prices

        def notional(order) -> float:
            px = prices.get(order.asset)
            if px is None:
                px = broker._current_prices.get(
                    order.asset, broker._current_opens.get(order.asset, 0.0)
                )
            return abs(order.quantity * px)

        reverse = priority == "notional_desc"
        return sorted(orders, key=notional, reverse=reverse)
