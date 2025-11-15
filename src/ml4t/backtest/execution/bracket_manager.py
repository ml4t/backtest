"""Bracket and OCO order management for broker simulation.

This module provides bracket order lifecycle management extracted from
SimulationBroker to follow the Single Responsibility Principle.

## VectorBT Compatibility - Base Price Calculation

**CRITICAL**: VectorBT calculates TP/SL/TSL levels from BASE PRICE (close before slippage),
NOT from fill_price (execution price after slippage). This matches real-world behavior
where stop levels are set relative to the market price, not the slippage-affected fill.

**Example** (from empirical testing - TASK-005):
```
Close at entry:     $43,885.00  ← BASE PRICE
Entry slippage:     +0.02%
Entry price:        $43,893.78  (= $43,885 * 1.0002)

TP = 2.5%:
TP level:           $44,982.12  (= $43,885 * 1.025)  ← Based on CLOSE, not entry_price
Exit slippage:      -0.02%
TP exit price:      $44,973.13  (= $44,982.12 * 0.9998)
```

**Implementation**:
When creating BRACKET orders, store the base price in order metadata:
```python
order = Order(
    ...,
    order_type=OrderType.BRACKET,
    tp_pct=0.025,  # 2.5% TP
    metadata={"base_price": market_event.close},  # ← CRITICAL for accuracy
)
```

If `base_price` is not provided, BracketOrderManager will estimate it by reversing
slippage from fill_price, but this may have rounding errors.
"""

import logging
from typing import Callable

from ml4t.backtest.core.event import FillEvent
from ml4t.backtest.core.types import OrderSide, OrderType
from ml4t.backtest.execution.order import Order

logger = logging.getLogger(__name__)


class BracketOrderManager:
    """Manages bracket orders and OCO (One-Cancels-Other) relationships.

    Responsibilities:
    - Create stop-loss and take-profit legs after parent fill
    - Link OCO orders
    - Cancel linked orders when one fills

    This class does NOT:
    - Execute orders (FillSimulator)
    - Route orders (OrderRouter)
    - Track positions (PositionTracker)
    """

    def __init__(self, submit_order_callback: Callable[[Order], None]) -> None:
        """Initialize bracket order manager.

        Args:
            submit_order_callback: Function to submit new orders (from broker)
        """
        self.submit_order = submit_order_callback
        self._bracket_relationships: dict[str, list[str]] = {}  # parent -> children

        logger.debug("BracketOrderManager initialized")

    def handle_bracket_fill(self, parent_order: Order, fill_event: FillEvent) -> list[Order]:
        """Create bracket legs after parent order fills.

        Supports both absolute prices (profit_target, stop_loss) and
        percentage-based levels (tp_pct, sl_pct, tsl_pct) for VectorBT compatibility.

        CRITICAL: VectorBT calculates TP/SL/TSL from BASE PRICE (close without slippage),
        NOT from fill_price. This matches real-world behavior where stops are set relative
        to the "true" price, not the execution price with slippage.

        Args:
            parent_order: The filled bracket order
            fill_event: The fill event

        Returns:
            List of created leg orders (stop-loss and take-profit)
        """
        if parent_order.order_type != OrderType.BRACKET:
            return []

        # Get base price for TP/SL/TSL calculations
        # VectorBT uses close (pre-slippage) as reference, not fill_price (post-slippage)
        fill_price = fill_event.fill_price
        is_buy = parent_order.is_buy

        # Extract base price from order metadata if available (preferred method)
        base_price = parent_order.metadata.get("base_price")

        if base_price is None:
            # Fallback: Reverse slippage from fill_price
            # Note: This is an approximation and may have rounding errors
            slippage_amount = fill_event.slippage
            if slippage_amount != 0:
                # Reverse the slippage application to get base price
                # For BUY: fill_price = base_price * (1 + slippage_rate)
                # For SELL: fill_price = base_price * (1 - slippage_rate)
                # We stored the dollar amount, so estimate: base ≈ fill - slippage
                base_price = fill_price - slippage_amount if is_buy else fill_price + slippage_amount
            else:
                # No slippage, fill_price is base_price
                base_price = fill_price

            logger.warning(
                f"Base price not found in order metadata for {parent_order.order_id}, "
                f"estimated from fill_price={fill_price} and slippage={slippage_amount}: "
                f"base_price={base_price}"
            )

        # Take profit level
        if parent_order.tp_pct is not None:
            # Percentage-based TP (calculated from BASE PRICE, not fill_price)
            if is_buy:
                profit_target = base_price * (1 + parent_order.tp_pct)
            else:  # Short position
                profit_target = base_price * (1 - parent_order.tp_pct)
        else:
            profit_target = parent_order.profit_target

        # Stop loss level
        if parent_order.sl_pct is not None:
            # Percentage-based SL (calculated from BASE PRICE, not fill_price)
            if is_buy:
                stop_loss = base_price * (1 - parent_order.sl_pct)
            else:  # Short position
                stop_loss = base_price * (1 + parent_order.sl_pct)
        elif parent_order.tsl_pct is not None:
            # Trailing stop loss (will use trail_percent parameter)
            stop_loss = None  # Will be handled by trailing stop logic
        else:
            stop_loss = parent_order.stop_loss

        # Validate we have at least one exit level
        if profit_target is None and stop_loss is None and parent_order.tsl_pct is None:
            logger.warning(
                f"Bracket order {parent_order.order_id} missing exit parameters "
                f"(need profit_target/tp_pct or stop_loss/sl_pct/tsl_pct)"
            )
            return []

        # Create exit orders (opposite side of entry)
        exit_side = OrderSide.SELL if parent_order.is_buy else OrderSide.BUY
        created_orders = []

        # Create stop-loss order (if SL specified)
        if stop_loss is not None:
            stop_order = Order(
                asset_id=parent_order.asset_id,
                order_type=OrderType.STOP,
                side=exit_side,
                quantity=parent_order.filled_quantity,
                stop_price=stop_loss,
                parent_order_id=parent_order.order_id,
                metadata={
                    "bracket_type": "stop_loss",
                    "creation_timestamp": fill_event.timestamp,  # VectorBT: Skip entry bar checking
                },
            )
            created_orders.append(stop_order)

        # Create trailing stop order (if TSL specified)
        if parent_order.tsl_pct is not None:
            tsl_order = Order(
                asset_id=parent_order.asset_id,
                order_type=OrderType.TRAILING_STOP,
                side=exit_side,
                quantity=parent_order.filled_quantity,
                trail_percent=parent_order.tsl_pct * 100.0,  # Convert decimal to percentage (0.01 -> 1.0)
                tsl_threshold_pct=parent_order.tsl_threshold_pct,  # Pass threshold from parent
                parent_order_id=parent_order.order_id,
                metadata={
                    "bracket_type": "trailing_stop",
                    "base_price": base_price,  # For VectorBT-compatible peak tracking (TASK-018)
                    "peak_price": base_price,  # Initialize peak to entry price
                    "creation_timestamp": fill_event.timestamp,  # VectorBT: Skip entry bar checking
                },
            )
            created_orders.append(tsl_order)

        # Create take-profit order (if TP specified)
        if profit_target is not None:
            profit_order = Order(
                asset_id=parent_order.asset_id,
                order_type=OrderType.LIMIT,
                side=exit_side,
                quantity=parent_order.filled_quantity,
                limit_price=profit_target,
                parent_order_id=parent_order.order_id,
                metadata={
                    "bracket_type": "take_profit",
                    "creation_timestamp": fill_event.timestamp,  # VectorBT: Skip entry bar checking
                },
            )
            created_orders.append(profit_order)

        # Link all orders as OCO (One-Cancels-Other)
        # Each order should cancel all others when it fills
        for order in created_orders:
            order.child_order_ids = [o.order_id for o in created_orders if o != order]

        # Track parent-child relationship
        self._bracket_relationships[parent_order.order_id] = [o.order_id for o in created_orders]

        # Submit all bracket legs
        for order in created_orders:
            self.submit_order(order)

        # Log created legs
        leg_types = ", ".join([o.metadata.get("bracket_type", "unknown") for o in created_orders])
        logger.info(
            f"Created {len(created_orders)} bracket legs for {parent_order.order_id}: {leg_types}"
        )

        return created_orders

    def handle_oco_fill(
        self, filled_order: Order, cancel_order_callback: Callable[[str], bool]
    ) -> list[str]:
        """Cancel linked OCO orders when one fills.

        Args:
            filled_order: The order that was filled
            cancel_order_callback: Function to cancel an order

        Returns:
            List of cancelled order IDs
        """
        if not filled_order.child_order_ids:
            return []

        cancelled = []
        for child_id in filled_order.child_order_ids:
            if cancel_order_callback(child_id):
                cancelled.append(child_id)
                logger.info(
                    f"Cancelled OCO order {child_id} due to fill of {filled_order.order_id}"
                )

        return cancelled

    def get_bracket_children(self, parent_id: str) -> list[str]:
        """Get child order IDs for a bracket parent.

        Args:
            parent_id: Parent order ID

        Returns:
            List of child order IDs
        """
        return self._bracket_relationships.get(parent_id, [])

    def reset(self) -> None:
        """Reset to initial state."""
        self._bracket_relationships.clear()
        logger.debug("BracketOrderManager reset")
