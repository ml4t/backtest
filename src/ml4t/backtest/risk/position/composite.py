"""Composite rules for combining multiple position rules."""

from dataclasses import dataclass

from ..types import ActionType, PositionAction, PositionState
from .protocol import PositionRule


@dataclass
class RuleChain:
    """Evaluate rules in order, first non-HOLD action wins.

    This is the most common composition pattern - rules are checked
    in priority order and the first rule to trigger takes effect.

    Args:
        rules: List of rules to evaluate in order

    Example:
        chain = RuleChain([
            StopLoss(pct=0.05),           # Highest priority
            ScaledExit([(0.10, 0.5)]),    # Second priority
            TighteningTrailingStop([...]), # Third priority
            TimeExit(bars=20),             # Lowest priority
        ])
    """

    rules: list[PositionRule]

    def evaluate(self, state: PositionState) -> PositionAction:
        """Evaluate rules in order, return first non-HOLD action."""
        for rule in self.rules:
            action = rule.evaluate(state)
            if action.action != ActionType.HOLD:
                return action
        return PositionAction.hold()


@dataclass
class AllOf:
    """All rules must return non-HOLD for the action to trigger.

    Useful for requiring multiple conditions to be true before exiting.
    Returns the first rule's action details (pct, stop_price, etc.).

    Args:
        rules: List of rules that must all agree

    Example:
        # Exit only if both profitable AND held long enough
        rule = AllOf([
            TakeProfit(pct=0.0),   # Must be profitable
            TimeExit(bars=5),       # Must have held 5+ bars
        ])
    """

    rules: list[PositionRule]

    def evaluate(self, state: PositionState) -> PositionAction:
        """Return action only if ALL rules return non-HOLD."""
        actions = [rule.evaluate(state) for rule in self.rules]

        if all(a.action != ActionType.HOLD for a in actions):
            # All triggered - return first rule's action with combined reason
            reasons = [a.reason for a in actions if a.reason]
            first = actions[0]
            return PositionAction(
                action=first.action,
                pct=first.pct,
                stop_price=first.stop_price,
                reason=" AND ".join(reasons) if reasons else "",
            )

        return PositionAction.hold()


@dataclass
class AnyOf:
    """First rule to return non-HOLD wins (alias for RuleChain).

    Semantically equivalent to RuleChain but named for clarity
    when composing complex rule logic.

    Args:
        rules: List of rules where any can trigger

    Example:
        # Exit on stop-loss OR signal
        rule = AnyOf([
            StopLoss(pct=0.05),
            SignalExit(threshold=0.5),
        ])
    """

    rules: list[PositionRule]

    def evaluate(self, state: PositionState) -> PositionAction:
        """Return first non-HOLD action (same as RuleChain)."""
        for rule in self.rules:
            action = rule.evaluate(state)
            if action.action != ActionType.HOLD:
                return action
        return PositionAction.hold()
