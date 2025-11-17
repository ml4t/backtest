"""Time-based risk management rules."""

from ml4t.backtest.risk.rule import RiskRule
from ml4t.backtest.risk.context import RiskContext
from ml4t.backtest.risk.decision import RiskDecision, ExitType


class TimeBasedExit(RiskRule):
    """Exit position after holding for a maximum number of bars.

    This rule triggers an exit when a position has been held for a specified
    number of bars, regardless of profit/loss. Useful for implementing:
    - Maximum holding period strategies
    - Time-decay exits
    - Regime-rotation strategies

    Args:
        max_bars: Maximum number of bars to hold a position

    Examples:
        >>> # Exit after 60 bars (e.g., 60 days for daily data)
        >>> rule = TimeBasedExit(max_bars=60)
        >>> risk_manager.add_rule(rule)
        >>>
        >>> # Short-term mean reversion: exit after 5 bars
        >>> rule = TimeBasedExit(max_bars=5)

    Note:
        - The rule only triggers on positions that have time tracking
        - If bars_held is None, the rule returns NO_ACTION
        - The exit is at market price on the next bar
    """

    def __init__(self, max_bars: int):
        """Initialize TimeBasedExit rule.

        Args:
            max_bars: Maximum number of bars to hold position (must be >= 1)

        Raises:
            ValueError: If max_bars < 1
        """
        if max_bars < 1:
            raise ValueError(f"max_bars must be >= 1, got {max_bars}")
        self.max_bars = max_bars

    def evaluate(self, context: RiskContext) -> RiskDecision:
        """Evaluate whether position should exit based on holding period.

        Args:
            context: Current risk context with position and market data

        Returns:
            RiskDecision to exit if bars_held >= max_bars, otherwise NO_ACTION
        """
        # No position or no time tracking
        if context.position_quantity == 0 or context.bars_held is None:
            return RiskDecision.no_action(
                reason="No position or no time tracking",
                metadata={"max_bars": self.max_bars},
                asset_id=context.asset_id,
            )

        # Check if holding period exceeded
        if context.bars_held >= self.max_bars:
            return RiskDecision.exit_now(
                exit_type=ExitType.TIME_EXIT,
                reason=f"Time exit: held {context.bars_held}/{self.max_bars} bars",
                metadata={
                    "max_bars": self.max_bars,
                    "bars_held": context.bars_held,
                    "entry_price": context.entry_price,
                    "current_price": context.current_price,
                },
                asset_id=context.asset_id,
            )

        # Position within time limit
        return RiskDecision.no_action(
            reason=f"Position within time limit ({context.bars_held}/{self.max_bars} bars)",
            metadata={"max_bars": self.max_bars, "bars_held": context.bars_held},
            asset_id=context.asset_id,
        )

    @property
    def priority(self) -> int:
        """Priority of this rule (higher = evaluated first).

        Returns:
            5 (medium priority - evaluated after critical stops but before take profits)
        """
        return 5
