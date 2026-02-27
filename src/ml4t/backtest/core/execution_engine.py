"""Order execution sequencing extracted from Broker."""

from __future__ import annotations

from ..types import OrderStatus


class ExecutionEngine:
    """Executes pending orders using configured fill ordering."""

    def __init__(self, broker):
        self.broker = broker

    def process_orders(self, use_open: bool = False):
        if self.broker.fill_ordering.value == "exit_first":
            self._process_orders_exit_first(use_open)
        else:
            self._process_orders_fifo(use_open)

    def _process_orders_exit_first(self, use_open: bool = False):
        broker = self.broker
        exit_orders = []
        entry_orders = []

        for order in broker.pending_orders[:]:
            if broker.execution_mode.value == "next_bar" and order in broker._orders_this_bar:
                continue
            if broker._is_exit_order(order):
                exit_orders.append(order)
            else:
                entry_orders.append(order)

        filled_orders: list = []

        for order in exit_orders:
            price = broker._get_fill_price_for_order(order, use_open)
            if price is None:
                continue
            fill_price = broker._check_fill(order, price)
            if fill_price is not None:
                fully_filled = broker._execute_fill(order, fill_price)
                if fully_filled:
                    filled_orders.append(order)
                    broker._partial_orders.pop(order.order_id, None)
                else:
                    broker._update_partial_order(order)

        broker.account.mark_to_market(broker._current_prices)

        for order in entry_orders:
            self._process_single_order(order, use_open, filled_orders)

        self._cleanup_filled_orders(filled_orders)

    def _process_orders_fifo(self, use_open: bool = False):
        broker = self.broker
        eligible_orders = []
        for order in broker.pending_orders[:]:
            if broker.execution_mode.value == "next_bar" and order in broker._orders_this_bar:
                continue
            eligible_orders.append(order)

        filled_orders: list = []

        for order in eligible_orders:
            self._process_single_order(order, use_open, filled_orders)
            if filled_orders and filled_orders[-1] is order:
                broker.account.mark_to_market(broker._current_prices)

        self._cleanup_filled_orders(filled_orders)

    def _process_single_order(self, order, use_open: bool, filled_orders: list) -> None:
        broker = self.broker
        price = broker._get_fill_price_for_order(order, use_open)
        if price is None:
            return

        is_exit = broker._is_exit_order(order)

        if is_exit:
            fill_price = broker._check_fill(order, price)
            if fill_price is not None:
                fully_filled = broker._execute_fill(order, fill_price)
                if fully_filled:
                    filled_orders.append(order)
                    broker._partial_orders.pop(order.order_id, None)
                else:
                    broker._update_partial_order(order)
        else:
            broker._apply_share_rounding(order)
            if order.quantity <= 0:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = "Quantity rounds to zero (share_type=INTEGER)"
                return

            fill_price = broker._check_fill(order, price)
            if fill_price is None:
                return

            valid, rejection_reason = broker.gatekeeper.validate_order(order, fill_price)

            if valid:
                fully_filled = broker._execute_fill(order, fill_price)
                if fully_filled:
                    filled_orders.append(order)
                    broker._partial_orders.pop(order.order_id, None)
                else:
                    broker._update_partial_order(order)
            elif (
                not broker.reject_on_insufficient_cash
                and "insufficient" in rejection_reason.lower()
            ):
                if broker.partial_fills_allowed and broker._try_partial_fill(order, fill_price):
                    filled_orders.append(order)
                    broker._partial_orders.pop(order.order_id, None)
            elif broker.partial_fills_allowed and "insufficient" in rejection_reason.lower():
                if broker._try_partial_fill(order, fill_price):
                    filled_orders.append(order)
                    broker._partial_orders.pop(order.order_id, None)
                else:
                    order.status = OrderStatus.REJECTED
                    order.rejection_reason = rejection_reason
            else:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = rejection_reason

    def _cleanup_filled_orders(self, filled_orders: list) -> None:
        broker = self.broker
        for order in filled_orders:
            if order in broker.pending_orders:
                broker.pending_orders.remove(order)
            if order in broker._orders_this_bar:
                broker._orders_this_bar.remove(order)

        for order in broker.pending_orders[:]:
            if order.status == OrderStatus.REJECTED:
                broker.pending_orders.remove(order)
