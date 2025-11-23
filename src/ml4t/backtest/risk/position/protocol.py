"""Protocol definition for position rules."""

from typing import Protocol, runtime_checkable

from ..types import PositionAction, PositionState


@runtime_checkable
class PositionRule(Protocol):
    """Protocol for position-level risk rules.

    All position rules must implement the evaluate() method which takes
    the current position state and returns an action.

    Rules are stateless by design - any state tracking (like triggered
    profit targets) should be stored on the Position object, not the rule.

    Example:
        class MyCustomRule:
            def __init__(self, threshold: float):
                self.threshold = threshold

            def evaluate(self, state: PositionState) -> PositionAction:
                if state.unrealized_return > self.threshold:
                    return PositionAction.exit_full("threshold_reached")
                return PositionAction.hold()
    """

    def evaluate(self, state: PositionState) -> PositionAction:
        """Evaluate rule against current position state.

        Args:
            state: Current position state with prices, P&L, etc.

        Returns:
            PositionAction indicating what to do (HOLD, EXIT_FULL, etc.)
        """
        ...
