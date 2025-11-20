"""Order fill simulation with market realism.

This module provides order fill simulation functionality extracted from
SimulationBroker to follow the Single Responsibility Principle.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from ml4t.backtest.core.constants import MAX_COMMISSION_CALC_ITERATIONS, MIN_FILL_SIZE
from ml4t.backtest.core.event import FillEvent
from ml4t.backtest.core.types import Price, Quantity

if TYPE_CHECKING:
    from ml4t.backtest.core.event import MarketEvent
from ml4t.backtest.data.asset_registry import AssetRegistry, AssetSpec
from ml4t.backtest.execution.commission import CommissionModel
from ml4t.backtest.execution.liquidity import LiquidityModel
from ml4t.backtest.execution.market_impact import MarketImpactModel
from ml4t.backtest.execution.order import Order, OrderType
from ml4t.backtest.execution.slippage import SlippageModel
from ml4t.backtest.portfolio.margin import MarginAccount

logger = logging.getLogger(__name__)


@dataclass
class FillResult:
    """Result of a successful order fill.

    Attributes:
        fill_event: The generated fill event
        commission: Commission cost for this fill
        slippage: Slippage cost for this fill (tracking only, already in fill_price)
        fill_quantity: Actual quantity filled (may be less than order quantity)
        fill_price: Actual fill price (includes slippage)
    """

    fill_event: FillEvent
    commission: float
    slippage: float
    fill_quantity: Quantity
    fill_price: Price


class FillSimulator:
    """Simulates order fills with realistic market constraints.

    Responsibilities:
    - Determine if order can fill at market price
    - Apply market impact to market price
    - Calculate fill price with slippage
    - Apply liquidity constraints
    - Check margin requirements (derivatives)
    - Check cash constraints (equities)
    - Calculate commission and slippage costs
    - Update order state and model states
    - Generate FillEvent

    This class does NOT:
    - Track positions (PositionTracker)
    - Route orders (OrderRouter)
    - Manage bracket orders (BracketOrderManager)

    Design Notes:
    - Order mutation: FillSimulator calls order.update_fill() to update order state.
      This is acceptable coupling because fills conceptually update order lifecycle.

    - Model updates: FillSimulator updates MarketImpactModel and LiquidityModel state
      after fills. This keeps all fill-related side effects in one place.

    - Stateless operation: Position and cash are passed as parameters to try_fill_order,
      not stored in FillSimulator. This allows testing without broker context.
    """

    def __init__(
        self,
        asset_registry: AssetRegistry,
        commission_model: CommissionModel | None = None,
        slippage_model: SlippageModel | None = None,
        market_impact_model: MarketImpactModel | None = None,
        liquidity_model: LiquidityModel | None = None,
        margin_account: MarginAccount | None = None,
        max_leverage: float = 1.0,
    ) -> None:
        """Initialize fill simulator.

        Args:
            asset_registry: Registry for asset specifications
            commission_model: Optional commission calculator
            slippage_model: Optional slippage calculator
            market_impact_model: Optional market impact model
            liquidity_model: Optional liquidity constraint model
            margin_account: Optional margin account (for derivatives)
            max_leverage: Maximum leverage allowed (default 1.0 = no leverage).
                For cash-based trading, limits position size to max_leverage * available_cash.
                Example: max_leverage=2.0 allows positions up to 2x cash balance.
                This prevents unlimited leverage as capital depletes.
        """
        self.asset_registry = asset_registry
        self.commission_model = commission_model
        self.slippage_model = slippage_model
        self.market_impact_model = market_impact_model
        self.liquidity_model = liquidity_model
        self.margin_account = margin_account
        self.max_leverage = max_leverage

        # Track fill count for trade IDs
        self._fill_count = 0

        logger.debug(f"FillSimulator initialized with max_leverage={max_leverage}")

    def try_fill_order(
        self,
        order: Order,
        market_event: "MarketEvent | None" = None,  # noqa: F821
        *,
        current_cash: float = 0.0,
        current_position: Quantity = 0.0,
        # Legacy parameters (deprecated, for backward compatibility)
        market_price: Price | None = None,
        timestamp: datetime | None = None,
        high: Price | None = None,
        low: Price | None = None,
        close: Price | None = None,
    ) -> FillResult | None:
        """Attempt to fill an order using MarketEvent or legacy OHLC parameters.

        This method applies all market realism constraints:
        1. Validates order can fill at market price (using intrabar if high/low provided)
        2. Applies market impact to market price
        3. Calculates fill price with slippage
        4. Applies liquidity constraints
        5. Applies margin or cash constraints
        6. Calculates commission and slippage costs
        7. Updates order state
        8. Updates model states (market impact, liquidity)
        9. Generates FillEvent

        Args:
            order: Order to attempt filling
            market_event: MarketEvent containing OHLCV and other market data (PREFERRED)
            current_cash: Available cash for purchases
            current_position: Current position quantity for the asset
            market_price: (DEPRECATED) Current market price - use market_event instead
            timestamp: (DEPRECATED) Event timestamp - use market_event.timestamp instead
            high: (DEPRECATED) Bar's high price - use market_event.high instead
            low: (DEPRECATED) Bar's low price - use market_event.low instead
            close: (DEPRECATED) Bar's close price - use market_event.close instead

        Returns:
            FillResult if order was filled, None if order cannot be filled

        Note:
            **BACKWARD COMPATIBILITY**: This method supports both old and new signatures:

            NEW (recommended):
                fill_simulator.try_fill_order(order, market_event, current_cash=..., current_position=...)

            OLD (deprecated, will be removed in v0.6.0):
                fill_simulator.try_fill_order(order, market_price=..., high=..., low=..., close=...)

            Using legacy parameters will emit a deprecation warning.

            This method has side effects:
            - Modifies order state via order.update_fill()
            - Updates market impact model state
            - Updates liquidity model state
            - Increments internal fill counter

            For VectorBT Pro compatibility, prefer passing high/low/close for intrabar
            execution. If only market_price is provided, falls back to end-of-bar logic.
        """
        import warnings

        # Extract parameters from market_event or use legacy parameters
        if market_event is not None:
            # NEW SIGNATURE: Extract from MarketEvent
            _timestamp = market_event.timestamp
            _high = market_event.high
            _low = market_event.low
            _close = market_event.close
            _market_price = market_event.price or market_event.close
            _volume = market_event.volume
            _bid_price = market_event.bid_price
            _ask_price = market_event.ask_price
        else:
            # OLD SIGNATURE: Use legacy parameters with deprecation warning
            if any(p is not None for p in [market_price, timestamp, high, low, close]):
                warnings.warn(
                    "Passing OHLC parameters directly to try_fill_order() is deprecated "
                    "and will be removed in v0.6.0. "
                    "Please pass a MarketEvent instance instead: "
                    "fill_simulator.try_fill_order(order, market_event, current_cash=..., current_position=...)",
                    DeprecationWarning,
                    stacklevel=2,
                )
            _timestamp = timestamp
            _high = high
            _low = low
            _close = close
            _market_price = market_price
            _volume = None
            _bid_price = None
            _ask_price = None

        # Determine the price to use for checks and fills
        # Prefer close from OHLC, fallback to market_price for backward compatibility
        check_price = _close if _close is not None else _market_price
        if check_price is None:
            logger.warning("No price data provided to try_fill_order")
            return None

        # Check if order can be filled using intrabar execution if high/low available
        can_fill = order.can_fill(price=check_price, high=_high, low=_low)
        high_str = f"{_high:.2f}" if _high is not None else "None"
        low_str = f"{_low:.2f}" if _low is not None else "None"
        print(f"    FILL CHECK: {order.order_id[:8]} type={order.order_type.value} side={order.side.value} "
              f"check_price=${check_price:.2f} high={high_str} low={low_str} can_fill={can_fill}")
        if not can_fill:
            return None

        # Apply market impact to the market price
        impacted_market_price = self._get_market_price_with_impact(
            order,
            check_price,
            _timestamp,
        )

        # Determine fill price (with slippage on top of impact)
        fill_price = self._calculate_fill_price(order, impacted_market_price, market_event)

        # Determine fill quantity considering liquidity constraints
        fill_quantity = order.remaining_quantity

        # CRITICAL FIX: Skip orders with no remaining quantity (already filled)
        # This prevents zero-quantity fills and duplicate processing
        if fill_quantity <= 0:
            return None

        # Get asset specification
        asset_spec = self.asset_registry.get(order.asset_id)

        # Apply liquidity constraints if model is available
        if self.liquidity_model is not None:
            adjusted_quantity = self._apply_liquidity_constraints(order, fill_quantity, check_price)
            if adjusted_quantity is None:
                return None
            fill_quantity = adjusted_quantity

        # Check margin requirements for derivatives, or cash for equities
        if self.margin_account and asset_spec and getattr(asset_spec, 'requires_margin', False):
            # Margin trading for derivatives
            adjusted_quantity = self._apply_margin_constraints(
                order, fill_quantity, fill_price, asset_spec
            )
            if adjusted_quantity is None:
                return None
            fill_quantity = adjusted_quantity
        else:
            # Standard cash trading for equities
            adjusted_quantity = self._apply_cash_constraints(
                order, fill_quantity, fill_price, current_cash, current_position, asset_spec
            )
            if adjusted_quantity is None:
                return None
            fill_quantity = adjusted_quantity

        # Calculate costs (asset-specific for different classes)
        commission = self._calculate_commission(order, fill_quantity, fill_price, asset_spec)
        slippage = self._calculate_slippage(
            order,
            fill_quantity,
            check_price,
            fill_price,
            asset_spec,
        )

        # Update order
        order.update_fill(fill_quantity, fill_price, commission, _timestamp)

        # Update market impact after fill
        self._update_market_impact(
            order,
            fill_quantity,
            check_price,
            _timestamp,
        )

        # Update liquidity model after fill
        self._update_liquidity_model(order, fill_quantity, fill_price)

        # Increment fill count for trade ID
        self._fill_count += 1

        # Create fill event
        fill_event = FillEvent(
            timestamp=_timestamp,
            order_id=order.order_id,
            trade_id=f"T{self._fill_count:06d}",
            asset_id=order.asset_id,
            side=order.side,
            fill_quantity=fill_quantity,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage,
            metadata=order.metadata,  # Copy metadata from order to fill event
        )

        logger.debug(f"Filled order: {order} with {fill_event}")

        # Return fill result
        return FillResult(
            fill_event=fill_event,
            commission=commission,
            slippage=slippage,
            fill_quantity=fill_quantity,
            fill_price=fill_price,
        )

    def _apply_liquidity_constraints(
        self,
        order: Order,
        fill_quantity: Quantity,
        market_price: Price,
    ) -> Quantity | None:
        """Apply liquidity constraints to fill quantity.

        Args:
            order: Order being filled
            fill_quantity: Desired fill quantity
            market_price: Current market price

        Returns:
            Adjusted fill quantity, or None if liquidity too low
        """
        if not self.liquidity_model:
            return fill_quantity

        max_liquidity = self.liquidity_model.get_max_fill_quantity(order, market_price)
        adjusted_quantity = min(fill_quantity, max_liquidity)

        # If liquidity constraint results in very small fill, reject the order
        if adjusted_quantity < MIN_FILL_SIZE:
            return None

        return adjusted_quantity

    def _apply_margin_constraints(
        self,
        order: Order,
        fill_quantity: Quantity,
        fill_price: Price,
        asset_spec: AssetSpec,
    ) -> Quantity | None:
        """Apply margin requirements to fill quantity for derivatives.

        Args:
            order: Order being filled
            fill_quantity: Desired fill quantity
            fill_price: Fill price
            asset_spec: Asset specification

        Returns:
            Adjusted fill quantity, or None if insufficient margin
        """
        if not self.margin_account:
            return fill_quantity

        # Check margin for opening/increasing position
        if order.is_buy:
            has_margin, required_margin = self.margin_account.check_margin_requirement(
                order.asset_id,
                fill_quantity,
                fill_price,
            )
            if not has_margin:
                # Try partial fill within margin
                max_quantity = (
                    self.margin_account.available_margin / required_margin * fill_quantity
                )
                if max_quantity < asset_spec.min_quantity:
                    return None
                fill_quantity = max_quantity
        elif order.is_sell:
            # Short selling - check margin (only if not closing existing position)
            # The current_position parameter would need to be passed to check this
            has_margin, required_margin = self.margin_account.check_margin_requirement(
                order.asset_id,
                -fill_quantity,
                fill_price,
            )
            if not has_margin:
                return None

        return fill_quantity

    def _apply_cash_constraints(
        self,
        order: Order,
        fill_quantity: Quantity,
        fill_price: Price,
        current_cash: float,
        current_position: Quantity,
        asset_spec: AssetSpec | None,
    ) -> Quantity | None:
        """Apply cash constraints to fill quantity for cash-based trading.

        Args:
            order: Order being filled
            fill_quantity: Desired fill quantity
            fill_price: Fill price
            current_cash: Available cash
            current_position: Current position in asset
            asset_spec: Asset specification

        Returns:
            Adjusted fill quantity, or None if insufficient funds/shares
        """
        if order.is_buy:
            # Calculate maximum fill quantity considering cash and commission
            # We need to solve: quantity * price + commission(quantity) <= cash
            # For most commission models: commission = quantity * price * fee_rate
            # So: quantity * price * (1 + fee_rate) <= cash
            # Therefore: quantity <= cash / (price * (1 + fee_rate))

            # Get effective commission rate
            if self.commission_model:
                # Estimate commission rate as a fraction of notional
                test_quantity = 1.0
                test_commission = self._calculate_commission(order, test_quantity, fill_price, asset_spec)
                commission_rate = test_commission / (test_quantity * fill_price) if fill_price > 0 else 0
            else:
                # Use asset-specific fee or default
                commission_rate = getattr(asset_spec, "taker_fee", 0.001) if asset_spec else 0.001

            # Calculate max affordable quantity with leverage constraint
            # max_leverage=1.0 means no leverage (can only buy what you can afford with cash)
            # max_leverage=2.0 means 2x leverage (can buy 2x your cash balance)
            max_affordable_quantity = (current_cash * self.max_leverage) / (fill_price * (1 + commission_rate))

            if max_affordable_quantity < fill_quantity:
                fill_quantity = max_affordable_quantity

                # Ensure minimum fill size
                if fill_quantity < MIN_FILL_SIZE:
                    return None

                # Double-check we can afford this (in case of non-linear commission)
                actual_commission = self._calculate_commission(order, fill_quantity, fill_price, asset_spec)
                required_cash = fill_quantity * fill_price + actual_commission

                # If still over budget due to non-linear commission, reduce further
                # Account for leverage when checking if we can afford the position
                if required_cash > (current_cash * self.max_leverage):
                    # Binary search for the right quantity (more accurate for complex commission models)
                    low_qty = 0.0
                    high_qty = fill_quantity
                    max_budget = current_cash * self.max_leverage
                    for _ in range(MAX_COMMISSION_CALC_ITERATIONS):
                        mid_qty = (low_qty + high_qty) / 2
                        mid_commission = self._calculate_commission(order, mid_qty, fill_price, asset_spec)
                        mid_cost = mid_qty * fill_price + mid_commission

                        if mid_cost <= max_budget:
                            low_qty = mid_qty
                        else:
                            high_qty = mid_qty

                    fill_quantity = low_qty
                    if fill_quantity < MIN_FILL_SIZE:
                        return None

        # Check if we have enough shares for sell orders (non-short)
        # Only allow short selling if explicitly enabled in asset spec
        # Note: This check is only for regular orders, not for triggered stops
        if order.is_sell and (not asset_spec or not getattr(asset_spec, "short_enabled", False)):
            # Allow sell orders that were originally stop/trailing stops
            # (they would have been converted to MARKET by now)
            if order.metadata.get("original_type") not in ["STOP", "TRAILING_STOP"]:
                available_shares = current_position
                if available_shares < fill_quantity:
                    fill_quantity = available_shares
                    if fill_quantity <= 0:
                        return None

        # Round fill quantity to asset's precision (prevents float precision mismatches)
        # For equities: truncates to whole shares
        # For crypto: rounds to 8 decimals (satoshi precision)
        if asset_spec:
            precision_mgr = asset_spec.get_precision_manager()
            fill_quantity = precision_mgr.round_quantity(fill_quantity)

        return fill_quantity

    def _get_market_price_with_impact(
        self,
        order: Order,
        market_price: Price,
        timestamp: datetime,
    ) -> Price:
        """Get market price adjusted for market impact."""
        if not self.market_impact_model:
            return market_price

        # Get current cumulative impact for this asset
        current_impact = self.market_impact_model.get_current_impact(
            order.asset_id,
            timestamp,
        )

        # Apply existing impact to market price
        return market_price + current_impact

    def _calculate_fill_price(
        self,
        order: Order,
        market_price: Price,
        market_event: "MarketEvent | None" = None,
    ) -> Price:
        """Calculate the actual fill price including slippage."""
        # For STOP orders (including SL), use stop_price as the base price (not market_price)
        # This ensures stop orders fill at their stop level, not at the bar's extreme (low/high)
        if order.order_type == OrderType.STOP and order.stop_price is not None:
            base_price = order.stop_price
            if self.slippage_model:
                return self.slippage_model.calculate_fill_price(order, base_price, market_event)
            # Default simple slippage for stops
            if order.is_buy:
                return base_price * 1.0001  # Buy stops pay more
            return base_price * 0.9999  # Sell stops receive less

        # For TRAILING_STOP orders, use trailing_stop_price as base
        # TSL level is the actual target exit price (not just a trigger)
        if order.order_type == OrderType.TRAILING_STOP and order.trailing_stop_price is not None:
            base_price = order.trailing_stop_price
            if self.slippage_model:
                return self.slippage_model.calculate_fill_price(order, base_price, market_event)
            # Default simple slippage for trailing stops
            if order.is_buy:
                return base_price * 1.0001
            return base_price * 0.9999

        if self.slippage_model:
            return self.slippage_model.calculate_fill_price(order, market_price, market_event)

        # Default simple slippage: 0.01% for market orders
        if order.order_type == OrderType.MARKET:
            if order.is_buy:
                return market_price * 1.0001
            return market_price * 0.9999

        # Limit orders fill at limit price or better
        if order.order_type == OrderType.LIMIT and order.limit_price is not None:
            if order.is_buy:
                return min(order.limit_price, market_price)
            return max(order.limit_price, market_price)

        return market_price

    def _calculate_commission(
        self,
        order: Order,
        fill_quantity: Quantity,
        fill_price: Price,
        asset_spec: AssetSpec | None = None,
    ) -> float:
        """Calculate commission for the fill."""
        if self.commission_model:
            return self.commission_model.calculate(order, fill_quantity, fill_price)

        if asset_spec:
            # Use asset-specific fee structure
            notional = fill_quantity * fill_price * getattr(asset_spec, "contract_size", 1.0)
            if order.order_type == OrderType.MARKET:
                return notional * getattr(asset_spec, "taker_fee", 0.001)
            return notional * getattr(asset_spec, "maker_fee", 0.001)

        # Simple flat commission: $1 per trade for equities
        return 1.0

    def _calculate_slippage(
        self,
        order: Order,
        fill_quantity: Quantity,
        market_price: Price,
        fill_price: Price,
        asset_spec: AssetSpec | None = None,
    ) -> float:
        """Calculate slippage cost."""
        if self.slippage_model:
            return self.slippage_model.calculate_slippage_cost(
                order,
                fill_quantity,
                market_price,
                fill_price,
            )

        # Default asset-specific slippage
        if asset_spec:
            slippage_rate = 0.0001  # 1 bp default
            # Check if asset_class exists (full AssetSpec) or fallback to asset_type (simple AssetSpec)
            asset_class_value = getattr(asset_spec.asset_class, "value", None) if hasattr(asset_spec, "asset_class") else getattr(asset_spec, "asset_type", None)

            if asset_class_value == "crypto":
                slippage_rate = 0.001  # 10 bp for crypto
            elif asset_class_value == "fx":
                slippage_rate = 0.00005  # 0.5 bp for FX
            elif asset_class_value == "future":
                slippage_rate = 0.0002  # 2 bp for futures

            notional = fill_quantity * market_price * getattr(asset_spec, "contract_size", 1.0)
            return notional * slippage_rate

        # Simple calculation: difference between market and fill price
        return abs(fill_price - market_price) * fill_quantity

    def _update_market_impact(
        self,
        order: Order,
        fill_quantity: Quantity,
        market_price: Price,
        timestamp: datetime,
    ) -> None:
        """Update market impact state after a fill."""
        if not self.market_impact_model:
            return

        # Calculate new impact from this trade
        permanent_impact, temporary_impact = self.market_impact_model.calculate_impact(
            order,
            fill_quantity,
            market_price,
            timestamp,
        )

        # Update market state
        self.market_impact_model.update_market_state(
            order.asset_id,
            permanent_impact,
            temporary_impact,
            timestamp,
        )

    def _update_liquidity_model(
        self,
        order: Order,
        fill_quantity: Quantity,
        fill_price: Price,
    ) -> None:
        """Update liquidity model after a fill."""
        if not self.liquidity_model:
            return

        side = "buy" if order.is_buy else "sell"
        self.liquidity_model.update_volume(
            order.asset_id,
            fill_price,
            side,
            fill_quantity,
        )

    def reset(self) -> None:
        """Reset fill simulator state."""
        self._fill_count = 0
        logger.debug("FillSimulator reset")
