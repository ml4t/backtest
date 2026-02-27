"""Order-book operations extracted from Broker."""

from __future__ import annotations

from ..types import Order, OrderSide, OrderStatus, OrderType
from .shared import SubmitOrderOptions


class OrderBook:
    """Handles order submission/mutation/retrieval."""

    _UPDATABLE_ORDER_FIELDS: frozenset[str] = frozenset(
        {"quantity", "limit_price", "stop_price", "trail_amount"}
    )

    def __init__(self, broker):
        self.broker = broker

    def submit_order(
        self,
        asset: str,
        quantity: float,
        side: OrderSide | None = None,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        trail_amount: float | None = None,
        options: SubmitOrderOptions | None = None,
    ) -> Order | None:
        broker = self.broker

        if side is None:
            if quantity == 0:
                return None
            side = OrderSide.BUY if quantity > 0 else OrderSide.SELL
        quantity = abs(quantity)
        if quantity == 0:
            return None

        if asset in broker._stop_exits_this_bar:
            existing_pos = broker.positions.get(asset)
            if existing_pos is None:
                return None

        broker._order_counter += 1
        order = Order(
            asset=asset,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            trail_amount=trail_amount,
            order_id=f"ORD-{broker._order_counter}",
            created_at=broker._current_time,
        )

        order._signal_price = broker._current_prices.get(asset)

        broker.orders.append(order)
        broker.pending_orders.append(order)

        if broker.execution_mode.value == "next_bar" and (
            options is None or not options.eligible_in_next_bar_mode
        ):
            broker._orders_this_bar.append(order)

        return order

    def update_order(self, order_id: str, **kwargs) -> bool:
        invalid_fields = set(kwargs.keys()) - self._UPDATABLE_ORDER_FIELDS
        if invalid_fields:
            raise ValueError(
                f"Cannot update order fields: {invalid_fields}. "
                f"Updatable fields: {sorted(self._UPDATABLE_ORDER_FIELDS)}"
            )

        for order in self.broker.pending_orders:
            if order.order_id == order_id:
                for key, value in kwargs.items():
                    setattr(order, key, value)
                return True
        return False

    def cancel_order(self, order_id: str) -> bool:
        for order in self.broker.pending_orders:
            if order.order_id == order_id:
                order.status = OrderStatus.CANCELLED
                self.broker.pending_orders.remove(order)
                return True
        return False

    def get_order(self, order_id: str) -> Order | None:
        for order in self.broker.orders:
            if order.order_id == order_id:
                return order
        return None

    def get_pending_orders(self, asset: str | None = None) -> list[Order]:
        if asset is None:
            return list(self.broker.pending_orders)
        return [o for o in self.broker.pending_orders if o.asset == asset]
