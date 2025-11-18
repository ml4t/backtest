"""Risk decision dataclass for risk management rules.

This module provides the RiskDecision dataclass representing the output of risk rules,
with factory methods for common decision types and logic for merging multiple decisions.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

from ml4t.backtest.core.types import AssetId, Price


class ExitType(Enum):
    """Type of exit signal."""

    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    TIME_EXIT = "time_exit"
    RISK_EXIT = "risk_exit"  # e.g., VIX spike, market regime change
    OTHER = "other"


@dataclass(frozen=True)
class RiskDecision:
    """Immutable decision output from a risk rule evaluation.

    Represents an action recommendation from a risk rule, such as:
    - Exit a position (immediately or at a specific price)
    - Update stop-loss or take-profit levels
    - No action required

    Multiple decisions can be merged using the `merge()` method, with conflicts
    resolved by priority (higher priority wins).

    Attributes:
        should_exit: Whether to exit the position immediately
        exit_type: Type of exit signal (if should_exit=True)
        exit_price: Suggested exit price (None = use market order)
        update_stop_loss: New stop-loss price (None = no change)
        update_take_profit: New take-profit price (None = no change)
        reason: Human-readable explanation of the decision
        priority: Priority for conflict resolution (higher = more important)
        metadata: Additional context for logging/debugging
        asset_id: Asset this decision applies to (None = universal)

    Example:
        >>> # Exit immediately due to stop-loss breach
        >>> decision = RiskDecision.exit_now(
        ...     exit_type=ExitType.STOP_LOSS,
        ...     reason="Price fell below stop-loss at $95.00"
        ... )

        >>> # Update trailing stop
        >>> decision = RiskDecision.update_stops(
        ...     update_stop_loss=Decimal("98.50"),
        ...     reason="Trailing stop raised to lock in profit"
        ... )

        >>> # No action
        >>> decision = RiskDecision.no_action(reason="Position within risk limits")

        >>> # Merge multiple decisions (higher priority wins)
        >>> merged = RiskDecision.merge([decision1, decision2, decision3])
    """

    should_exit: bool = False
    exit_type: Optional[ExitType] = None
    exit_price: Optional[Price] = None
    update_stop_loss: Optional[Price] = None
    update_take_profit: Optional[Price] = None
    reason: str = ""
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    asset_id: Optional[AssetId] = None

    def __post_init__(self) -> None:
        """Validate decision consistency."""
        if self.should_exit and self.exit_type is None:
            raise ValueError("exit_type must be specified when should_exit=True")

        if self.exit_type is not None and not self.should_exit:
            raise ValueError("should_exit must be True when exit_type is specified")

    @classmethod
    def no_action(
        cls,
        reason: str = "No action required",
        metadata: Optional[dict[str, Any]] = None,
        asset_id: Optional[AssetId] = None,
    ) -> "RiskDecision":
        """Create a no-action decision.

        Args:
            reason: Explanation for no action
            metadata: Additional context
            asset_id: Asset this decision applies to

        Returns:
            RiskDecision with all action flags set to False/None

        Example:
            >>> decision = RiskDecision.no_action(
            ...     reason="Position within risk limits",
            ...     metadata={"mae_pct": 0.02, "unrealized_pnl": 150.0}
            ... )
        """
        return cls(
            should_exit=False,
            reason=reason,
            priority=0,
            metadata=metadata or {},
            asset_id=asset_id,
        )

    @classmethod
    def exit_now(
        cls,
        exit_type: ExitType,
        reason: str,
        exit_price: Optional[Price] = None,
        priority: int = 10,
        metadata: Optional[dict[str, Any]] = None,
        asset_id: Optional[AssetId] = None,
    ) -> "RiskDecision":
        """Create an immediate exit decision.

        Args:
            exit_type: Type of exit (stop-loss, take-profit, etc.)
            reason: Explanation for exit
            exit_price: Specific exit price (None = use market order)
            priority: Priority for conflict resolution (default: 10)
            metadata: Additional context
            asset_id: Asset this decision applies to

        Returns:
            RiskDecision with should_exit=True

        Example:
            >>> decision = RiskDecision.exit_now(
            ...     exit_type=ExitType.STOP_LOSS,
            ...     reason="Price fell below stop-loss at $95.00",
            ...     exit_price=Decimal("95.00"),
            ...     metadata={"breach_pct": 0.05}
            ... )
        """
        return cls(
            should_exit=True,
            exit_type=exit_type,
            exit_price=exit_price,
            reason=reason,
            priority=priority,
            metadata=metadata or {},
            asset_id=asset_id,
        )

    @classmethod
    def update_stops(
        cls,
        update_stop_loss: Optional[Price] = None,
        update_take_profit: Optional[Price] = None,
        reason: str = "",
        priority: int = 5,
        metadata: Optional[dict[str, Any]] = None,
        asset_id: Optional[AssetId] = None,
    ) -> "RiskDecision":
        """Create a decision to update stop-loss or take-profit levels.

        Args:
            update_stop_loss: New stop-loss price (None = no change)
            update_take_profit: New take-profit price (None = no change)
            reason: Explanation for update
            priority: Priority for conflict resolution (default: 5)
            metadata: Additional context
            asset_id: Asset this decision applies to

        Returns:
            RiskDecision with updated stop levels

        Example:
            >>> decision = RiskDecision.update_stops(
            ...     update_stop_loss=Decimal("98.50"),
            ...     reason="Trailing stop raised to lock in profit",
            ...     metadata={"previous_sl": Decimal("96.00")}
            ... )
        """
        if update_stop_loss is None and update_take_profit is None:
            raise ValueError("At least one of update_stop_loss or update_take_profit must be specified")

        return cls(
            should_exit=False,
            update_stop_loss=update_stop_loss,
            update_take_profit=update_take_profit,
            reason=reason,
            priority=priority,
            metadata=metadata or {},
            asset_id=asset_id,
        )

    @classmethod
    def _resolve_sl_conflicts(
        cls,
        decisions: list["RiskDecision"],
    ) -> Optional[Price]:
        """Resolve stop-loss conflicts using priority + conservative logic.

        Resolution logic:
        1. Filter to decisions with stop-loss updates
        2. Find highest priority
        3. Among same priority: use max(sl_values) for long (tightest/most conservative)

        Args:
            decisions: List of risk decisions (assumed non-exit decisions)

        Returns:
            Resolved stop-loss price, or None if no stop-loss updates

        Examples:
            >>> # Different priorities: highest priority wins
            >>> d1 = RiskDecision.update_stops(update_stop_loss=98.0, priority=5)
            >>> d2 = RiskDecision.update_stops(update_stop_loss=99.0, priority=10)
            >>> assert _resolve_sl_conflicts([d1, d2]) == 99.0  # Priority 10 wins

            >>> # Same priority: most conservative (tightest, highest price for long)
            >>> d1 = RiskDecision.update_stops(update_stop_loss=98.0, priority=5)
            >>> d2 = RiskDecision.update_stops(update_stop_loss=99.0, priority=5)
            >>> assert _resolve_sl_conflicts([d1, d2]) == 99.0  # max() wins
        """
        # Filter to decisions with stop-loss updates
        sl_decisions = [d for d in decisions if d.update_stop_loss is not None]
        if not sl_decisions:
            return None

        # Find highest priority
        max_priority = max(d.priority for d in sl_decisions)

        # Get all decisions with highest priority
        top_priority_decisions = [d for d in sl_decisions if d.priority == max_priority]

        # If only one decision at highest priority, use it
        if len(top_priority_decisions) == 1:
            return top_priority_decisions[0].update_stop_loss

        # Multiple decisions at same priority: use most conservative (tightest)
        # For long positions: max(stop_loss) is tightest (closest to current price)
        # For short positions: min(stop_loss) is tightest
        # Since we don't have position direction here, assume long (most common)
        # Future enhancement: could pass position direction if needed
        return max(d.update_stop_loss for d in top_priority_decisions)

    @classmethod
    def _resolve_tp_conflicts(
        cls,
        decisions: list["RiskDecision"],
    ) -> Optional[Price]:
        """Resolve take-profit conflicts using priority + conservative logic.

        Resolution logic:
        1. Filter to decisions with take-profit updates
        2. Find highest priority
        3. Among same priority: use min(tp_values) for long (most conservative/nearest)

        Args:
            decisions: List of risk decisions (assumed non-exit decisions)

        Returns:
            Resolved take-profit price, or None if no take-profit updates

        Examples:
            >>> # Different priorities: highest priority wins
            >>> d1 = RiskDecision.update_stops(update_take_profit=105.0, priority=5)
            >>> d2 = RiskDecision.update_stops(update_take_profit=110.0, priority=10)
            >>> assert _resolve_tp_conflicts([d1, d2]) == 110.0  # Priority 10 wins

            >>> # Same priority: most conservative (nearest target for long)
            >>> d1 = RiskDecision.update_stops(update_take_profit=105.0, priority=5)
            >>> d2 = RiskDecision.update_stops(update_take_profit=110.0, priority=5)
            >>> assert _resolve_tp_conflicts([d1, d2]) == 105.0  # min() wins (nearest)
        """
        # Filter to decisions with take-profit updates
        tp_decisions = [d for d in decisions if d.update_take_profit is not None]
        if not tp_decisions:
            return None

        # Find highest priority
        max_priority = max(d.priority for d in tp_decisions)

        # Get all decisions with highest priority
        top_priority_decisions = [d for d in tp_decisions if d.priority == max_priority]

        # If only one decision at highest priority, use it
        if len(top_priority_decisions) == 1:
            return top_priority_decisions[0].update_take_profit

        # Multiple decisions at same priority: use most conservative (nearest target)
        # For long positions: min(take_profit) is most conservative (easier to hit)
        # For short positions: max(take_profit) is most conservative
        # Assume long (most common)
        return min(d.update_take_profit for d in top_priority_decisions)

    @classmethod
    def merge(
        cls,
        decisions: list["RiskDecision"],
        default_priority_order: Optional[list[ExitType]] = None,
    ) -> "RiskDecision":
        """Merge multiple risk decisions into a single decision.

        Merging logic:
        1. Exit decisions take precedence over stop updates and no-action
        2. Among exit decisions, highest priority wins
        3. If priorities are equal, use default_priority_order (if provided)
        4. Stop-loss/take-profit updates resolved via priority + conservative logic:
           - Highest priority wins
           - Same priority: most conservative value wins (tightest stop, nearest target)
        5. Metadata is merged (later decisions override earlier ones)

        Args:
            decisions: List of RiskDecisions to merge
            default_priority_order: Optional ordering of exit types for tie-breaking
                (e.g., [ExitType.STOP_LOSS, ExitType.RISK_EXIT, ExitType.TAKE_PROFIT])

        Returns:
            Single merged RiskDecision

        Raises:
            ValueError: If decisions list is empty

        Example:
            >>> decision1 = RiskDecision.update_stops(
            ...     update_stop_loss=Decimal("98.50"),
            ...     reason="Trailing stop"
            ... )
            >>> decision2 = RiskDecision.exit_now(
            ...     exit_type=ExitType.STOP_LOSS,
            ...     reason="Stop-loss breach",
            ...     priority=10
            ... )
            >>> merged = RiskDecision.merge([decision1, decision2])
            >>> assert merged.should_exit  # Exit takes precedence
            >>> assert merged.exit_type == ExitType.STOP_LOSS
        """
        if not decisions:
            raise ValueError("Cannot merge empty list of decisions")

        if len(decisions) == 1:
            return decisions[0]

        # Separate exit decisions from non-exit decisions
        exit_decisions = [d for d in decisions if d.should_exit]
        non_exit_decisions = [d for d in decisions if not d.should_exit]

        # If any decision says to exit, prioritize that
        if exit_decisions:
            # Sort by priority (highest first)
            exit_decisions.sort(key=lambda d: d.priority, reverse=True)

            # Get highest priority decisions (may be multiple with same priority)
            highest_priority = exit_decisions[0].priority
            top_decisions = [d for d in exit_decisions if d.priority == highest_priority]

            # If multiple decisions with same priority, use default ordering
            if len(top_decisions) > 1 and default_priority_order:
                for exit_type in default_priority_order:
                    matching = [d for d in top_decisions if d.exit_type == exit_type]
                    if matching:
                        selected = matching[0]
                        break
                else:
                    # No match in default ordering, use first one
                    selected = top_decisions[0]
            else:
                selected = top_decisions[0]

            # Merge metadata from all decisions
            merged_metadata = {}
            for d in decisions:
                merged_metadata.update(d.metadata)

            # Merge reasons
            reasons = [d.reason for d in decisions if d.reason]
            merged_reason = selected.reason
            if len(reasons) > 1:
                merged_reason += f" (also: {', '.join(r for r in reasons if r != selected.reason)})"

            return cls(
                should_exit=True,
                exit_type=selected.exit_type,
                exit_price=selected.exit_price,
                update_stop_loss=selected.update_stop_loss,
                update_take_profit=selected.update_take_profit,
                reason=merged_reason,
                priority=selected.priority,
                metadata=merged_metadata,
                asset_id=selected.asset_id,
            )

        # No exit decisions - merge stop updates
        # Resolve stop-loss conflicts: highest priority wins, same priority uses most conservative
        stop_loss = cls._resolve_sl_conflicts(non_exit_decisions)

        # Resolve take-profit conflicts: highest priority wins, same priority uses most conservative
        take_profit = cls._resolve_tp_conflicts(non_exit_decisions)

        # Merge metadata
        merged_metadata = {}
        for d in decisions:
            merged_metadata.update(d.metadata)

        # Merge reasons
        reasons = [d.reason for d in decisions if d.reason]
        merged_reason = "; ".join(reasons) if reasons else "No action required"

        # Use highest priority
        max_priority = max(d.priority for d in decisions)

        return cls(
            should_exit=False,
            update_stop_loss=stop_loss,
            update_take_profit=take_profit,
            reason=merged_reason,
            priority=max_priority,
            metadata=merged_metadata,
            asset_id=decisions[0].asset_id if decisions else None,
        )

    def is_action_required(self) -> bool:
        """Check if this decision requires any action.

        Returns:
            True if decision requires exit or stop update, False otherwise
        """
        return (
            self.should_exit
            or self.update_stop_loss is not None
            or self.update_take_profit is not None
        )

    def __str__(self) -> str:
        """Human-readable representation of the decision."""
        if self.should_exit:
            return f"RiskDecision(EXIT: {self.exit_type.value}, reason='{self.reason}', priority={self.priority})"
        elif self.update_stop_loss or self.update_take_profit:
            updates = []
            if self.update_stop_loss:
                updates.append(f"SL={self.update_stop_loss}")
            if self.update_take_profit:
                updates.append(f"TP={self.update_take_profit}")
            return f"RiskDecision(UPDATE: {', '.join(updates)}, reason='{self.reason}', priority={self.priority})"
        else:
            return f"RiskDecision(NO_ACTION, reason='{self.reason}')"
