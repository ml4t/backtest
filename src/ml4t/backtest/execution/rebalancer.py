"""Portfolio rebalancing utilities for target weight execution.

This module provides utilities for converting portfolio target weights to orders,
enabling integration with external portfolio optimizers like riskfolio-lib,
PyPortfolioOpt, or cvxpy.

Example:
    from ml4t.backtest import TargetWeightExecutor, RebalanceConfig

    executor = TargetWeightExecutor(config=RebalanceConfig(
        min_trade_value=500,
        allow_fractional=True,
    ))

    # In strategy.on_data():
    target_weights = {'AAPL': 0.3, 'GOOG': 0.3, 'MSFT': 0.35}  # 5% cash
    orders = executor.execute(target_weights, data, broker)
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..broker import Broker
    from ..types import Order

from ..types import OrderSide


class WeightProvider(Protocol):
    """Protocol for anything that produces target weights."""

    def get_weights(self, data: dict, broker: "Broker") -> dict[str, float]:
        """Return target weights (asset -> weight, should sum to <= 1.0)."""
        ...


@dataclass
class RebalanceConfig:
    """Configuration for rebalancing behavior.

    Attributes:
        min_trade_value: Skip trades with absolute value smaller than this ($).
        min_weight_change: Skip if weight change is smaller than this (decimal).
        allow_fractional: Allow fractional shares (default: False, whole shares only).
        round_lots: Round to lot_size increments (e.g., 100-share lots).
        lot_size: Lot size for rounding (only used if round_lots=True).
        allow_short: Allow short positions via negative weights.
        max_single_weight: Maximum weight allowed for any single asset.
        cancel_before_rebalance: Cancel pending orders before rebalancing (safest).
        account_for_pending: Consider pending orders when calculating current weights.
    """

    # Trade thresholds
    min_trade_value: float = 100.0
    min_weight_change: float = 0.01

    # Share handling
    allow_fractional: bool = False
    round_lots: bool = False
    lot_size: int = 100

    # Position constraints
    allow_short: bool = False
    max_single_weight: float = 1.0

    # Order handling
    cancel_before_rebalance: bool = True
    account_for_pending: bool = True


class TargetWeightExecutor:
    """Convert target portfolio weights to orders.

    Handles the common pattern of rebalancing to target weights:
    - Computes required trades from current vs target positions
    - Accounts for pending orders to prevent double-allocation
    - Applies minimum trade thresholds
    - Handles lot rounding and fractional shares
    - Respects position limits

    Example:
        executor = TargetWeightExecutor(config=RebalanceConfig(
            min_trade_value=500,
            round_lots=True,
        ))

        # In strategy:
        target_weights = {'AAPL': 0.3, 'GOOG': 0.3, 'MSFT': 0.4}
        orders = executor.execute(target_weights, data, broker)
    """

    def __init__(self, config: RebalanceConfig | None = None):
        """Initialize the executor with optional configuration.

        Args:
            config: Rebalancing configuration. Uses defaults if not provided.
        """
        self.config = config or RebalanceConfig()

    def execute(
        self,
        target_weights: dict[str, float],
        data: dict[str, dict],
        broker: "Broker",
    ) -> list["Order"]:
        """Execute rebalancing to target weights.

        Args:
            target_weights: Dict of asset -> target weight (0.0 to 1.0).
                            Sum can be < 1.0 to hold cash.
            data: Current bar data (for prices). Format: {asset: {'close': price, ...}}
            broker: Broker instance for order submission.

        Returns:
            List of submitted orders.
        """
        # 1. Cancel pending orders if configured (prevents double-allocation)
        if self.config.cancel_before_rebalance:
            for pending_order in list(broker.pending_orders):
                broker.cancel_order(pending_order.order_id)

        equity = broker.get_account_value()
        if equity <= 0:
            return []

        orders: list[Order] = []

        # 2. Get current weights (effective or actual based on config)
        if self.config.account_for_pending and not self.config.cancel_before_rebalance:
            current_weights = self._get_effective_weights(broker, data)
        else:
            current_weights = self._get_current_weights(broker, data)

        # 3. Validate total weight <= 1.0 (allow cash targeting)
        total_target = sum(target_weights.values())
        if total_target > 1.0 + 1e-6:
            # Scale down to prevent over-allocation
            scale = 1.0 / total_target
            target_weights = {k: v * scale for k, v in target_weights.items()}

        # 4. Process each target asset
        for asset, target_wt in target_weights.items():
            order: Order | None = self._process_asset(
                asset, target_wt, current_weights, equity, data, broker
            )
            if order is not None:
                orders.append(order)

        # 5. Close positions not in target
        for asset in current_weights:
            if asset not in target_weights:
                pos = broker.get_position(asset)
                if pos and pos.quantity != 0:
                    close_order: Order | None = broker.close_position(asset)
                    if close_order:
                        orders.append(close_order)

        return orders

    def _process_asset(
        self,
        asset: str,
        target_wt: float,
        current_weights: dict[str, float],
        equity: float,
        data: dict[str, dict],
        broker: "Broker",
    ) -> "Order | None":
        """Process a single asset for rebalancing.

        Returns:
            Order if trade needed, None otherwise.
        """
        # Apply constraints
        target_wt = min(target_wt, self.config.max_single_weight)
        if target_wt < 0 and not self.config.allow_short:
            target_wt = 0

        current_wt = current_weights.get(asset, 0.0)
        weight_delta = target_wt - current_wt

        # Skip small weight changes
        if abs(weight_delta) < self.config.min_weight_change:
            return None

        # Get price
        price = data.get(asset, {}).get("close")
        if not price or price <= 0:
            return None

        # Compute trade value
        delta_value = equity * weight_delta

        # Skip small trades
        if abs(delta_value) < self.config.min_trade_value:
            return None

        # Compute shares
        shares = delta_value / price

        # Apply share rounding
        if self.config.round_lots:
            shares = round(shares / self.config.lot_size) * self.config.lot_size
        elif not self.config.allow_fractional:
            shares = int(shares)

        if shares == 0:
            return None

        # Submit order
        side = OrderSide.BUY if shares > 0 else OrderSide.SELL
        return broker.submit_order(asset, abs(shares), side)

    def _get_current_weights(self, broker: "Broker", data: dict[str, dict]) -> dict[str, float]:
        """Get current portfolio weights from held positions only.

        Args:
            broker: Broker instance.
            data: Current bar data for prices.

        Returns:
            Dict of asset -> current weight.
        """
        equity = broker.get_account_value()
        if equity <= 0:
            return {}

        weights = {}
        for asset, pos in broker.positions.items():
            price = data.get(asset, {}).get("close", pos.entry_price)
            value = pos.quantity * price
            weights[asset] = value / equity

        return weights

    def _get_effective_weights(self, broker: "Broker", data: dict[str, dict]) -> dict[str, float]:
        """Get effective weights including pending orders.

        This prevents double-allocation when execute() is called multiple times
        before orders fill (e.g., with ExecutionMode.NEXT_BAR or LIMIT orders).

        Args:
            broker: Broker instance.
            data: Current bar data for prices.

        Returns:
            Dict of asset -> effective weight (positions + pending orders).
        """
        equity = broker.get_account_value()
        if equity <= 0:
            return {}

        # Start with actual positions
        effective_value: dict[str, float] = {}
        for asset, pos in broker.positions.items():
            price = data.get(asset, {}).get("close", pos.entry_price)
            effective_value[asset] = pos.quantity * price

        # Add net value of pending orders
        for order in broker.pending_orders:
            price = order.limit_price or data.get(order.asset, {}).get("close")
            if price:
                # BUY adds value, SELL subtracts
                sign = 1 if order.side == OrderSide.BUY else -1
                delta = order.quantity * price * sign
                effective_value[order.asset] = effective_value.get(order.asset, 0) + delta

        return {k: v / equity for k, v in effective_value.items()}

    def preview(
        self,
        target_weights: dict[str, float],
        data: dict[str, dict],
        broker: "Broker",
    ) -> list[dict]:
        """Preview trades without executing.

        Useful for debugging and understanding what trades would be generated.

        Args:
            target_weights: Dict of asset -> target weight.
            data: Current bar data.
            broker: Broker instance.

        Returns:
            List of trade previews with asset, current_weight, target_weight,
            shares, value, and skip_reason (if applicable).
        """
        equity = broker.get_account_value()
        if equity <= 0:
            return []

        if self.config.account_for_pending and not self.config.cancel_before_rebalance:
            current_weights = self._get_effective_weights(broker, data)
        else:
            current_weights = self._get_current_weights(broker, data)

        previews = []

        for asset, target_wt in target_weights.items():
            current_wt = current_weights.get(asset, 0.0)
            price = data.get(asset, {}).get("close", 0)
            weight_delta = target_wt - current_wt

            if price > 0:
                delta_value = equity * weight_delta
                shares = delta_value / price

                # Determine if would be skipped
                skip_reason = None
                if abs(weight_delta) < self.config.min_weight_change:
                    skip_reason = "weight_change_too_small"
                elif abs(delta_value) < self.config.min_trade_value:
                    skip_reason = "trade_value_too_small"
                elif not self.config.allow_fractional and abs(int(shares)) == 0:
                    skip_reason = "rounds_to_zero_shares"

                previews.append(
                    {
                        "asset": asset,
                        "current_weight": current_wt,
                        "target_weight": target_wt,
                        "weight_delta": weight_delta,
                        "shares": shares,
                        "value": delta_value,
                        "skip_reason": skip_reason,
                    }
                )

        # Add positions not in target (will be closed)
        for asset in current_weights:
            if asset not in target_weights:
                pos = broker.get_position(asset)
                if pos and pos.quantity != 0:
                    price = data.get(asset, {}).get("close", pos.entry_price)
                    current_wt = current_weights.get(asset, 0.0)
                    previews.append(
                        {
                            "asset": asset,
                            "current_weight": current_wt,
                            "target_weight": 0.0,
                            "weight_delta": -current_wt,
                            "shares": -pos.quantity,
                            "value": -pos.quantity * price,
                            "skip_reason": None,
                            "action": "close_position",
                        }
                    )

        return previews
