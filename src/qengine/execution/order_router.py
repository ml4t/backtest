"""Order routing and queue management for broker simulation.

This module provides order queue management functionality extracted from
SimulationBroker to follow the Single Responsibility Principle.
"""

import logging
from collections import defaultdict
from datetime import datetime

from qengine.core.types import AssetId, OrderId, OrderType
from qengine.execution.order import Order

logger = logging.getLogger(__name__)


class OrderRouter:
    """Routes and manages orders in different queues.

    Responsibilities:
    - Route orders to appropriate queues (open, stop, trailing, bracket, pending)
    - Store all orders by ID
    - Query orders by various criteria
    - Remove orders from queues (cancel, fill)

    This class does NOT:
    - Execute orders (FillSimulator)
    - Track positions (PositionTracker)
    - Handle bracket logic (BracketOrderManager)
    """

    def __init__(self, execution_delay: bool = True) -> None:
        """Initialize order router.

        Args:
            execution_delay: If True, queue orders for next market event
        """
        self.execution_delay = execution_delay

        # All orders (history)
        self._orders: dict[OrderId, Order] = {}

        # Active queues organized by asset
        self._open_orders: dict[AssetId, list[Order]] = defaultdict(list)
        self._stop_orders: dict[AssetId, list[Order]] = defaultdict(list)
        self._trailing_stops: dict[AssetId, list[Order]] = defaultdict(list)
        self._bracket_orders: dict[OrderId, dict] = {}  # Parent ID -> bracket info
        self._pending_orders: dict[AssetId, list[tuple[Order, datetime]]] = defaultdict(
            list
        )

        logger.debug(
            f"OrderRouter initialized (execution_delay={execution_delay})"
        )

    def route_order(self, order: Order, timestamp: datetime) -> None:
        """Route an order to the appropriate queue.

        Args:
            order: Order to route
            timestamp: Current time
        """
        # Store in history
        self._orders[order.order_id] = order

        # Route based on type and execution delay setting
        if self.execution_delay:
            # With execution delay, route to appropriate queue
            if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
                self._stop_orders[order.asset_id].append(order)
                logger.debug(f"Routed STOP order {order.order_id} to stop queue")
            elif order.order_type == OrderType.TRAILING_STOP:
                self._trailing_stops[order.asset_id].append(order)
                logger.debug(
                    f"Routed TRAILING_STOP order {order.order_id} to trailing queue"
                )
            else:
                # Regular orders go to pending queue
                self._pending_orders[order.asset_id].append((order, timestamp))
                logger.debug(
                    f"Routed {order.order_type} order {order.order_id} to pending queue"
                )
        else:
            # Legacy immediate execution mode
            if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
                self._stop_orders[order.asset_id].append(order)
            elif order.order_type == OrderType.TRAILING_STOP:
                self._trailing_stops[order.asset_id].append(order)
            elif order.order_type == OrderType.BRACKET:
                self._open_orders[order.asset_id].append(order)
            else:
                self._open_orders[order.asset_id].append(order)

    def activate_pending_orders(self, asset_id: AssetId) -> list[Order]:
        """Activate pending orders for an asset (move to open queue).

        Called on market event after execution delay.

        Args:
            asset_id: Asset identifier

        Returns:
            List of orders that were activated
        """
        if asset_id not in self._pending_orders:
            return []

        activated = []
        for order, _ in self._pending_orders[asset_id]:
            self._open_orders[asset_id].append(order)
            activated.append(order)

        self._pending_orders[asset_id].clear()
        logger.debug(f"Activated {len(activated)} pending orders for {asset_id}")
        return activated

    def get_order(self, order_id: OrderId) -> Order | None:
        """Get order by ID.

        Args:
            order_id: Order identifier

        Returns:
            Order if found, None otherwise
        """
        return self._orders.get(order_id)

    def get_open_orders(self, asset_id: AssetId | None = None) -> list[Order]:
        """Get all open orders, optionally filtered by asset.

        Args:
            asset_id: Optional asset filter

        Returns:
            List of open orders
        """
        if asset_id:
            return list(self._open_orders.get(asset_id, []))

        # All open orders across all assets
        all_orders = []
        for orders in self._open_orders.values():
            all_orders.extend(orders)
        return all_orders

    def get_stop_orders(self, asset_id: AssetId) -> list[Order]:
        """Get stop orders for an asset.

        Args:
            asset_id: Asset identifier

        Returns:
            List of stop orders
        """
        return list(self._stop_orders.get(asset_id, []))

    def get_trailing_stops(self, asset_id: AssetId) -> list[Order]:
        """Get trailing stop orders for an asset.

        Args:
            asset_id: Asset identifier

        Returns:
            List of trailing stop orders
        """
        return list(self._trailing_stops.get(asset_id, []))

    def remove_order(self, order: Order) -> bool:
        """Remove an order from all queues.

        Args:
            order: Order to remove

        Returns:
            True if order was found and removed
        """
        found = False

        # Remove from open orders
        if order in self._open_orders[order.asset_id]:
            self._open_orders[order.asset_id].remove(order)
            found = True

        # Remove from stop orders
        if order in self._stop_orders[order.asset_id]:
            self._stop_orders[order.asset_id].remove(order)
            found = True

        # Remove from trailing stops
        if order in self._trailing_stops[order.asset_id]:
            self._trailing_stops[order.asset_id].remove(order)
            found = True

        # Remove from pending orders
        pending = self._pending_orders[order.asset_id]
        pending_to_remove = [(o, t) for o, t in pending if o == order]
        for item in pending_to_remove:
            pending.remove(item)
            found = True

        if found:
            logger.debug(f"Removed order {order.order_id} from queues")

        return found

    def register_bracket_order(self, parent_id: OrderId, bracket_info: dict) -> None:
        """Register a bracket order.

        Args:
            parent_id: Parent order ID
            bracket_info: Bracket order configuration
        """
        self._bracket_orders[parent_id] = bracket_info
        logger.debug(f"Registered bracket order for parent {parent_id}")

    def get_bracket_info(self, parent_id: OrderId) -> dict | None:
        """Get bracket order info.

        Args:
            parent_id: Parent order ID

        Returns:
            Bracket info if found
        """
        return self._bracket_orders.get(parent_id)

    def remove_bracket(self, parent_id: OrderId) -> None:
        """Remove bracket order registration.

        Args:
            parent_id: Parent order ID
        """
        if parent_id in self._bracket_orders:
            del self._bracket_orders[parent_id]
            logger.debug(f"Removed bracket order for parent {parent_id}")

    def reset(self) -> None:
        """Reset to initial state."""
        self._orders.clear()
        self._open_orders.clear()
        self._stop_orders.clear()
        self._trailing_stops.clear()
        self._bracket_orders.clear()
        self._pending_orders.clear()
        logger.debug("OrderRouter reset")

    def get_statistics(self) -> dict:
        """Get queue statistics.

        Returns:
            Dictionary with queue counts
        """
        return {
            "total_orders": len(self._orders),
            "open_orders": sum(len(orders) for orders in self._open_orders.values()),
            "stop_orders": sum(len(orders) for orders in self._stop_orders.values()),
            "trailing_stops": sum(
                len(orders) for orders in self._trailing_stops.values()
            ),
            "pending_orders": sum(
                len(orders) for orders in self._pending_orders.values()
            ),
            "bracket_orders": len(self._bracket_orders),
        }
