"""Static exit rules - fixed thresholds that don't change."""

from dataclasses import dataclass

from ..types import PositionAction, PositionState


@dataclass
class StopLoss:
    """Exit when unrealized loss exceeds threshold.

    Args:
        pct: Maximum loss as decimal (0.05 = 5% loss triggers exit)

    Example:
        rule = StopLoss(pct=0.05)  # Exit at -5%
    """

    pct: float

    def evaluate(self, state: PositionState) -> PositionAction:
        """Exit if loss exceeds threshold."""
        if state.unrealized_return <= -self.pct:
            return PositionAction.exit_full(f"stop_loss_{self.pct:.1%}")
        return PositionAction.hold()


@dataclass
class TakeProfit:
    """Exit when unrealized profit exceeds threshold.

    Args:
        pct: Target profit as decimal (0.10 = 10% profit triggers exit)

    Example:
        rule = TakeProfit(pct=0.10)  # Exit at +10%
    """

    pct: float

    def evaluate(self, state: PositionState) -> PositionAction:
        """Exit if profit exceeds threshold."""
        if state.unrealized_return >= self.pct:
            return PositionAction.exit_full(f"take_profit_{self.pct:.1%}")
        return PositionAction.hold()


@dataclass
class TimeExit:
    """Exit after holding for a specified number of bars.

    Args:
        max_bars: Maximum bars to hold position

    Example:
        rule = TimeExit(max_bars=20)  # Exit after 20 bars
    """

    max_bars: int

    def evaluate(self, state: PositionState) -> PositionAction:
        """Exit if held too long."""
        if state.bars_held >= self.max_bars:
            return PositionAction.exit_full(f"time_exit_{self.max_bars}bars")
        return PositionAction.hold()
