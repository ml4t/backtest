"""Dynamic trailing stop that tightens over position lifetime.

This rule implements a trailing stop that automatically tightens over time,
protecting profits while giving trends room to develop early in the trade.
The tightening is linear with position age (bars_held).
"""

import logging
from typing import Optional

from ml4t.backtest.core.types import Price
from ml4t.backtest.risk.context import RiskContext
from ml4t.backtest.risk.decision import RiskDecision, ExitType
from ml4t.backtest.risk.rule import RiskRule

logger = logging.getLogger(__name__)


class DynamicTrailingStop(RiskRule):
    """Trailing stop that tightens over position lifetime for profit protection.

    This rule implements a time-based tightening of the trailing stop, providing:
    - **Early flexibility**: Wide initial trail gives trends room to develop
    - **Automatic tightening**: Trail narrows as position ages (bars_held increases)
    - **Profit protection**: Locks in more profit as position matures
    - **MFE tracking**: Stop trails the peak price (entry + MFE), not current price
    - **Never backward**: Stop only moves in favorable direction

    The trailing stop distance tightens linearly:
    - **Trail percentage** = initial_trail_pct - (bars_held × tighten_rate)
    - **Minimum trail**: Never goes below 0.5% (prevents over-tightening)

    Stop level calculation:
    - **Long positions**: (entry_price + MFE) × (1 - current_trail_pct)
    - **Short positions**: (entry_price - MFE) × (1 + current_trail_pct)

    Args:
        initial_trail_pct: Starting trail distance as percentage (e.g., 0.05 = 5%)
        tighten_rate: How much to tighten per bar (e.g., 0.001 = 0.1% per bar)
        priority: Rule priority for conflict resolution (default: 100)

    Examples:
        >>> # Start with 5% trail, tighten by 0.1% per bar
        >>> # After 20 bars: 5% - (20 × 0.1%) = 3% trail
        >>> # After 40 bars: 5% - (40 × 0.1%) = 1% trail (tightening)
        >>> trailing_stop = DynamicTrailingStop(
        ...     initial_trail_pct=0.05,    # 5% initial trail
        ...     tighten_rate=0.001         # 0.1% tightening per bar
        ... )
        >>> risk_manager.add_rule(trailing_stop)
        >>>
        >>> # Aggressive: Lock in profits quickly
        >>> aggressive_trail = DynamicTrailingStop(
        ...     initial_trail_pct=0.03,    # 3% initial trail
        ...     tighten_rate=0.002         # 0.2% per bar
        ... )
        >>> # After 10 bars: 3% - 2% = 1% trail (very tight)
        >>>
        >>> # Patient: Let trends run longer
        >>> patient_trail = DynamicTrailingStop(
        ...     initial_trail_pct=0.08,    # 8% initial trail
        ...     tighten_rate=0.0005        # 0.05% per bar
        ... )
        >>> # After 40 bars: 8% - 2% = 6% trail (still wide)

    Recommended Settings:
        - **Aggressive**: initial=0.03 (3%), tighten=0.002 (0.2%/bar)
          - Locks profits quickly, exits early on reversals
          - Good for momentum strategies, volatile markets
        - **Balanced**: initial=0.05 (5%), tighten=0.001 (0.1%/bar) [default]
          - Standard tightening, suitable for most trend-following
          - After 30 bars: 2% trail (reasonable protection)
        - **Patient**: initial=0.08 (8%), tighten=0.0005 (0.05%/bar)
          - Lets trends run longer, fewer whipsaw exits
          - Good for strong trends, lower volatility markets

    Behavior Over Time (Balanced 5%/0.1% example):
        - **Bar 0-10**: 5.0% → 4.0% trail (early development phase)
        - **Bar 11-20**: 3.9% → 3.0% trail (starting to tighten)
        - **Bar 21-40**: 2.9% → 1.0% trail (active profit protection)
        - **Bar 41+**: 0.5% trail (minimum, locked in tight)

    Comparison to Fixed Trailing Stop:
        - **Fixed trail**: Constant 5% distance throughout trade
        - **Dynamic trail**: Starts at 5%, tightens to 0.5% over 45 bars
        - **Advantage**: Captures more of strong trends, exits faster on reversals
        - **Trade-off**: May exit prematurely if trend continues beyond typical duration

    Use Cases:
        - **Trend capture**: Let early trend develop, lock in later profits
        - **Mean-reversion protection**: Exit if trend stalls or reverses
        - **Time-based risk reduction**: Reduce risk as holding period extends
        - **Profit protection**: Automatically tighten as unrealized gains accumulate

    Note:
        - Requires position tracking (bars_held from TASK-INT-021)
        - Requires MFE tracking (_tracked_mfe in context.features)
        - Returns `update_stops()` not `exit_now()` - updates levels, doesn't exit
        - Minimum 0.5% trail prevents whipsaws from over-tightening
        - Works for both long and short positions
    """

    def __init__(
        self,
        initial_trail_pct: float,
        tighten_rate: float,
        *,
        minimum_trail_pct: float = 0.005,  # 0.5% minimum
        priority: int = 100,
    ):
        """Initialize DynamicTrailingStop rule.

        Args:
            initial_trail_pct: Starting trail distance as percentage (e.g., 0.05 = 5%)
            tighten_rate: How much to tighten per bar (e.g., 0.001 = 0.1% per bar)
            minimum_trail_pct: Minimum trail percentage (default: 0.005 = 0.5%)
            priority: Rule priority for conflict resolution (default: 100)

        Raises:
            ValueError: If parameters are invalid (negative, zero, or out of range)
        """
        if initial_trail_pct <= 0:
            raise ValueError(
                f"initial_trail_pct must be positive, got {initial_trail_pct}"
            )
        if tighten_rate < 0:  # Allow zero for testing (no tightening)
            raise ValueError(f"tighten_rate must be non-negative, got {tighten_rate}")
        if minimum_trail_pct <= 0:
            raise ValueError(
                f"minimum_trail_pct must be positive, got {minimum_trail_pct}"
            )
        if minimum_trail_pct >= initial_trail_pct:
            raise ValueError(
                f"minimum_trail_pct ({minimum_trail_pct}) must be less than "
                f"initial_trail_pct ({initial_trail_pct})"
            )

        self.initial_trail_pct = initial_trail_pct
        self.tighten_rate = tighten_rate
        self.minimum_trail_pct = minimum_trail_pct
        self._priority = priority

    @property
    def priority(self) -> int:
        """Priority for conflict resolution (higher = more important)."""
        return self._priority

    def evaluate(self, context: RiskContext) -> RiskDecision:
        """Evaluate dynamic trailing stop level.

        Calculates the current trail percentage based on bars_held, then
        computes the stop level trailing the peak price (entry + MFE).

        Args:
            context: Current risk context with position and market data

        Returns:
            RiskDecision with update_stop_loss if position exists,
            otherwise no_action
        """
        # No position - no stop needed
        if context.position_quantity == 0:
            return RiskDecision.no_action(
                reason="No position to apply trailing stop",
                asset_id=context.asset_id,
            )

        # Calculate current trail percentage (tightens with bars_held)
        current_trail_pct = self.initial_trail_pct - (
            context.bars_held * self.tighten_rate
        )

        # Enforce minimum trail (prevent over-tightening)
        current_trail_pct = max(current_trail_pct, self.minimum_trail_pct)

        # Calculate peak price from entry + MFE
        # MFE is already in currency units (position_quantity × price_move)
        # For long: peak = entry + (MFE / quantity)
        # For short: peak = entry - (MFE / abs(quantity))
        is_long = context.position_quantity > 0

        if is_long:
            # Long: Peak is highest price seen (entry + MFE per share)
            mfe_per_share = (
                context.max_favorable_excursion / context.position_quantity
                if context.position_quantity != 0
                else 0.0
            )
            peak_price = context.entry_price + mfe_per_share

            # Stop trails below the peak
            stop_level = peak_price * (1.0 - current_trail_pct)
        else:
            # Short: Peak is lowest price seen (entry - MFE per share)
            mfe_per_share = (
                context.max_favorable_excursion / abs(context.position_quantity)
                if context.position_quantity != 0
                else 0.0
            )
            peak_price = context.entry_price - mfe_per_share

            # Stop trails above the peak
            stop_level = peak_price * (1.0 + current_trail_pct)

        # Calculate trail distance for metadata
        trail_distance = abs(peak_price - stop_level)
        trail_distance_pct = (
            (trail_distance / context.entry_price) if context.entry_price > 0 else 0.0
        )

        # Check if trailing stop has been hit (only if context has close price)
        if hasattr(context, 'close'):
            current_price = context.close
            stop_hit = False

            if is_long:
                # Long position: stop if price <= stop_level
                stop_hit = current_price <= stop_level
            else:
                # Short position: stop if price >= stop_level
                stop_hit = current_price >= stop_level
        else:
            stop_hit = False
            current_price = context.entry_price  # Fallback for unit tests

        if stop_hit:
            # Exit immediately - trailing stop triggered
            return RiskDecision.exit_now(
                exit_type=ExitType.STOP_LOSS,
                reason=(
                    f"Dynamic trailing stop hit: {current_trail_pct*100:.2f}% trail "
                    f"at bar {context.bars_held}, price={current_price:.2f} <= SL={stop_level:.2f}"
                ),
                priority=self.priority,
                metadata={
                    "bars_held": context.bars_held,
                    "initial_trail_pct": self.initial_trail_pct,
                    "current_trail_pct": current_trail_pct,
                    "tighten_rate": self.tighten_rate,
                    "minimum_trail_pct": self.minimum_trail_pct,
                    "peak_price": peak_price,
                    "stop_level": stop_level,
                    "current_price": current_price,
                    "trail_distance": abs(current_price - stop_level),
                    "exit_type": "stop_loss",
                    "mfe": context.max_favorable_excursion,
                    "mfe_per_share": mfe_per_share,
                    "entry_price": context.entry_price,
                    "position_direction": "long" if is_long else "short",
                },
                asset_id=context.asset_id,
            )

        # Stop not hit - update trailing stop level for monitoring
        return RiskDecision.update_stops(
            update_stop_loss=stop_level,
            reason=(
                f"Dynamic trailing stop: {current_trail_pct*100:.2f}% trail "
                f"at bar {context.bars_held}, SL={stop_level:.2f}"
            ),
            priority=self.priority,
            metadata={
                "bars_held": context.bars_held,
                "initial_trail_pct": self.initial_trail_pct,
                "current_trail_pct": current_trail_pct,
                "tighten_rate": self.tighten_rate,
                "minimum_trail_pct": self.minimum_trail_pct,
                "peak_price": peak_price,
                "stop_level": stop_level,
                "trail_distance": trail_distance,
                "trail_distance_pct": trail_distance_pct,
                "mfe": context.max_favorable_excursion,
                "mfe_per_share": mfe_per_share,
                "entry_price": context.entry_price,
                "position_direction": "long" if is_long else "short",
            },
            asset_id=context.asset_id,
        )


__all__ = ["DynamicTrailingStop"]
