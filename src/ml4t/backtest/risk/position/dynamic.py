"""Dynamic exit rules - thresholds that change based on position state."""

from dataclasses import dataclass, field

from ..types import PositionAction, PositionState


@dataclass
class TrailingStop:
    """Exit when price retraces from high water mark.

    For longs: Exit if price drops X% from highest price since entry
    For shorts: Exit if price rises X% from lowest price since entry

    Args:
        pct: Trail percentage as decimal (0.05 = 5% trail)

    Example:
        rule = TrailingStop(pct=0.05)  # 5% trailing stop
    """

    pct: float

    def evaluate(self, state: PositionState) -> PositionAction:
        """Exit if price retraces beyond trail."""
        if state.is_long:
            # Long: stop triggers when price drops below (high - trail%)
            stop_price = state.high_water_mark * (1 - self.pct)
            if state.current_price <= stop_price:
                return PositionAction.exit_full(f"trailing_stop_{self.pct:.1%}")
        else:
            # Short: stop triggers when price rises above (low + trail%)
            stop_price = state.low_water_mark * (1 + self.pct)
            if state.current_price >= stop_price:
                return PositionAction.exit_full(f"trailing_stop_{self.pct:.1%}")

        return PositionAction.hold()


@dataclass
class TighteningTrailingStop:
    """Trailing stop that tightens as profit increases.

    The trail percentage decreases at higher profit levels, locking in
    more gains as the position becomes more profitable.

    Args:
        schedule: List of (return_threshold, trail_pct) tuples.
                  Must be sorted by return_threshold ascending.

    Example:
        rule = TighteningTrailingStop([
            (0.0, 0.05),   # At 0% return: 5% trail
            (0.10, 0.03),  # At 10%+ return: 3% trail
            (0.20, 0.02),  # At 20%+ return: 2% trail
        ])
    """

    schedule: list[tuple[float, float]]

    def __post_init__(self):
        # Sort by return threshold descending for efficient lookup
        self._schedule = sorted(self.schedule, key=lambda x: x[0], reverse=True)

    def _get_trail_pct(self, unrealized_return: float) -> float:
        """Get applicable trail percentage based on current return."""
        for threshold, trail_pct in self._schedule:
            if unrealized_return >= threshold:
                return trail_pct
        # Default to last (loosest) trail if no threshold met
        return self._schedule[-1][1] if self._schedule else 0.05

    def evaluate(self, state: PositionState) -> PositionAction:
        """Exit if price retraces beyond dynamic trail."""
        trail_pct = self._get_trail_pct(state.unrealized_return)

        if state.is_long:
            stop_price = state.high_water_mark * (1 - trail_pct)
            if state.current_price <= stop_price:
                return PositionAction.exit_full(
                    f"tightening_trail_{trail_pct:.1%}_at_{state.unrealized_return:.1%}"
                )
        else:
            stop_price = state.low_water_mark * (1 + trail_pct)
            if state.current_price >= stop_price:
                return PositionAction.exit_full(
                    f"tightening_trail_{trail_pct:.1%}_at_{state.unrealized_return:.1%}"
                )

        return PositionAction.hold()


@dataclass
class ScaledExit:
    """Exit portions of position at profit targets.

    Allows scaling out of positions by exiting a percentage at each
    profit level. Tracks which levels have been triggered to avoid
    duplicate exits.

    Args:
        targets: List of (return_threshold, exit_pct) tuples.
                 exit_pct is relative to CURRENT position size.

    Example:
        rule = ScaledExit([
            (0.05, 0.25),  # At +5%: exit 25% of position
            (0.10, 0.33),  # At +10%: exit 33% of remaining
            (0.15, 0.50),  # At +15%: exit 50% of remaining
        ])

    Note:
        This rule tracks triggered levels internally. For proper operation
        in backtesting, create a new instance per position.
    """

    targets: list[tuple[float, float]]
    _triggered: set[float] = field(default_factory=set, repr=False)

    def __post_init__(self):
        # Sort by return threshold ascending
        self._targets = sorted(self.targets, key=lambda x: x[0])

    def reset(self):
        """Reset triggered levels (call when position closes)."""
        self._triggered = set()

    def evaluate(self, state: PositionState) -> PositionAction:
        """Check if any untriggered profit target is hit."""
        for threshold, exit_pct in self._targets:
            if threshold not in self._triggered:
                if state.unrealized_return >= threshold:
                    self._triggered.add(threshold)
                    return PositionAction.exit_partial(
                        exit_pct, f"scale_out_{threshold:.0%}_{exit_pct:.0%}"
                    )
        return PositionAction.hold()
