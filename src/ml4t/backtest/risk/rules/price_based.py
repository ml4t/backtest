"""Price-based risk management rules (stop loss, take profit)."""

from typing import Optional

from ml4t.backtest.core.types import Price
from ml4t.backtest.risk.rule import RiskRule
from ml4t.backtest.risk.context import RiskContext
from ml4t.backtest.risk.decision import RiskDecision, ExitType


class PriceBasedStopLoss(RiskRule):
    """Exit position when price hits stop-loss level.

    This rule triggers an exit when the current price breaches the stop-loss
    level, protecting against adverse price movements. The rule supports both:
    - Long positions: Exit if price <= stop_loss_price
    - Short positions: Exit if price >= stop_loss_price

    Args:
        stop_loss_price: Price level to trigger stop loss (optional if using position levels)

    Examples:
        >>> # Fixed stop loss at $95 (for long position)
        >>> rule = PriceBasedStopLoss(stop_loss_price=95.0)
        >>> risk_manager.add_rule(rule)
        >>>
        >>> # Dynamic stop loss from position levels (set by strategy or other rules)
        >>> rule = PriceBasedStopLoss()  # Uses context.stop_loss_price
        >>> risk_manager.add_rule(rule)

    Note:
        - If stop_loss_price is None, uses context.stop_loss_price
        - If no stop loss price available, returns NO_ACTION
        - Exit triggers at market on next bar (realistic fill simulation)
    """

    def __init__(self, stop_loss_price: Optional[Price] = None):
        """Initialize PriceBasedStopLoss rule.

        Args:
            stop_loss_price: Fixed stop loss price, or None to use position levels
        """
        self.stop_loss_price = stop_loss_price

    def evaluate(self, context: RiskContext) -> RiskDecision:
        """Evaluate whether position should exit based on stop loss.

        Args:
            context: Current risk context with position and market data

        Returns:
            RiskDecision to exit if stop loss hit, otherwise NO_ACTION
        """
        # No position
        if context.position_quantity == 0:
            return RiskDecision.no_action(
                reason="No position to stop out",
                metadata={"stop_loss_price": self.stop_loss_price},
                asset_id=context.asset_id,
            )

        # Determine stop loss price (fixed or from position levels)
        sl_price = self.stop_loss_price
        if sl_price is None:
            sl_price = context.stop_loss_price

        # No stop loss configured
        if sl_price is None:
            return RiskDecision.no_action(
                reason="No stop loss price configured",
                metadata={},
                asset_id=context.asset_id,
            )

        # Check if stop loss triggered based on position direction
        is_long = context.position_quantity > 0
        stop_hit = False

        if is_long:
            # Long position: stop if price <= stop_loss_price
            stop_hit = context.current_price <= sl_price
        else:
            # Short position: stop if price >= stop_loss_price
            stop_hit = context.current_price >= sl_price

        if stop_hit:
            # Calculate stop loss distance for metadata
            sl_distance = abs(context.current_price - sl_price)
            sl_percent = (
                (sl_distance / context.entry_price * 100) if context.entry_price else None
            )

            return RiskDecision.exit_now(
                exit_type=ExitType.STOP_LOSS,
                reason=(
                    f"Stop loss hit: price={context.current_price:.2f}, "
                    f"stop={sl_price:.2f}, "
                    f"distance={sl_distance:.2f}"
                ),
                metadata={
                    "stop_loss_price": sl_price,
                    "current_price": context.current_price,
                    "stop_distance": sl_distance,
                    "stop_percent": sl_percent,
                    "position_direction": "long" if is_long else "short",
                    "mae": context.mae,
                },
                asset_id=context.asset_id,
            )

        # Stop not hit
        distance_to_stop = abs(context.current_price - sl_price)
        return RiskDecision.no_action(
            reason=f"Price above stop loss (distance: {distance_to_stop:.2f})",
            metadata={
                "stop_loss_price": sl_price,
                "current_price": context.current_price,
                "distance_to_stop": distance_to_stop,
            },
            asset_id=context.asset_id,
        )

    @property
    def priority(self) -> int:
        """Priority of this rule (higher = evaluated first).

        Returns:
            10 (high priority - stop losses should be checked first)
        """
        return 10


class PriceBasedTakeProfit(RiskRule):
    """Exit position when price hits take-profit level.

    This rule triggers an exit when the current price reaches the take-profit
    level, locking in gains. The rule supports both:
    - Long positions: Exit if price >= take_profit_price
    - Short positions: Exit if price <= take_profit_price

    Args:
        take_profit_price: Price level to trigger take profit (optional if using position levels)

    Examples:
        >>> # Fixed take profit at $110 (for long position)
        >>> rule = PriceBasedTakeProfit(take_profit_price=110.0)
        >>> risk_manager.add_rule(rule)
        >>>
        >>> # Dynamic take profit from position levels (set by strategy or other rules)
        >>> rule = PriceBasedTakeProfit()  # Uses context.take_profit_price
        >>> risk_manager.add_rule(rule)

    Note:
        - If take_profit_price is None, uses context.take_profit_price
        - If no take profit price available, returns NO_ACTION
        - Exit triggers at market on next bar (realistic fill simulation)
    """

    def __init__(self, take_profit_price: Optional[Price] = None):
        """Initialize PriceBasedTakeProfit rule.

        Args:
            take_profit_price: Fixed take profit price, or None to use position levels
        """
        self.take_profit_price = take_profit_price

    def evaluate(self, context: RiskContext) -> RiskDecision:
        """Evaluate whether position should exit based on take profit.

        Args:
            context: Current risk context with position and market data

        Returns:
            RiskDecision to exit if take profit hit, otherwise NO_ACTION
        """
        # No position
        if context.position_quantity == 0:
            return RiskDecision.no_action(
                reason="No position to take profit on",
                metadata={"take_profit_price": self.take_profit_price},
                asset_id=context.asset_id,
            )

        # Determine take profit price (fixed or from position levels)
        tp_price = self.take_profit_price
        if tp_price is None:
            tp_price = context.take_profit_price

        # No take profit configured
        if tp_price is None:
            return RiskDecision.no_action(
                reason="No take profit price configured",
                metadata={},
                asset_id=context.asset_id,
            )

        # Check if take profit triggered based on position direction
        is_long = context.position_quantity > 0
        profit_hit = False

        if is_long:
            # Long position: take profit if price >= take_profit_price
            profit_hit = context.current_price >= tp_price
        else:
            # Short position: take profit if price <= take_profit_price
            profit_hit = context.current_price <= tp_price

        if profit_hit:
            # Calculate profit for metadata
            profit_distance = abs(context.current_price - tp_price)
            profit_percent = (
                (profit_distance / context.entry_price * 100) if context.entry_price else None
            )

            return RiskDecision.exit_now(
                exit_type=ExitType.TAKE_PROFIT,
                reason=(
                    f"Take profit hit: price={context.current_price:.2f}, "
                    f"target={tp_price:.2f}, "
                    f"distance={profit_distance:.2f}"
                ),
                metadata={
                    "take_profit_price": tp_price,
                    "current_price": context.current_price,
                    "profit_distance": profit_distance,
                    "profit_percent": profit_percent,
                    "position_direction": "long" if is_long else "short",
                    "mfe": context.mfe,
                },
                asset_id=context.asset_id,
            )

        # Take profit not hit
        distance_to_target = abs(tp_price - context.current_price)
        return RiskDecision.no_action(
            reason=f"Price below take profit (distance: {distance_to_target:.2f})",
            metadata={
                "take_profit_price": tp_price,
                "current_price": context.current_price,
                "distance_to_target": distance_to_target,
            },
            asset_id=context.asset_id,
        )

    @property
    def priority(self) -> int:
        """Priority of this rule (higher = evaluated first).

        Returns:
            8 (medium-high priority - after stop losses but before time exits)
        """
        return 8
