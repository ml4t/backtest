"""Risk rule interface for composable risk management.

This module provides the RiskRule abstract base class and Protocol for implementing
custom risk rules that evaluate trading positions and generate risk decisions.
"""

from abc import ABC, abstractmethod
from typing import Optional, Protocol, runtime_checkable

from ml4t.backtest.execution.order import Order
from ml4t.backtest.risk.context import RiskContext
from ml4t.backtest.risk.decision import RiskDecision


@runtime_checkable
class RiskRuleProtocol(Protocol):
    """Protocol for callable risk rules.

    Allows any callable with the correct signature to be used as a risk rule,
    without requiring inheritance from RiskRule ABC.

    Example:
        >>> def simple_stop_loss(context: RiskContext) -> RiskDecision:
        ...     '''Simple 5% stop-loss rule.'''
        ...     if context.unrealized_pnl_pct < -0.05:
        ...         return RiskDecision.exit_now(
        ...             exit_type=ExitType.STOP_LOSS,
        ...             reason=f"Stop-loss breach: {context.unrealized_pnl_pct:.2%}"
        ...         )
        ...     return RiskDecision.no_action()
        ...
        >>> # Can be used directly with RiskManager
        >>> manager.add_rule(simple_stop_loss)  # No class needed!
    """

    def __call__(self, context: RiskContext) -> RiskDecision:
        """Evaluate risk context and return decision.

        Args:
            context: Current risk-relevant state

        Returns:
            RiskDecision indicating action to take
        """
        ...


class RiskRule(ABC):
    """Abstract base class for risk management rules.

    Risk rules evaluate position state and market conditions to generate
    risk decisions (exit signals, stop-loss updates, etc.).

    Rules should be:
    - **Stateless**: No mutable internal state (use RiskContext for state)
    - **Composable**: Can be combined with other rules via RiskManager
    - **Testable**: Pure functions of RiskContext → RiskDecision
    - **Fast**: Called on every market event, avoid expensive operations

    Subclasses must implement:
    - `evaluate(context)`: Generate risk decision from context

    Optional methods:
    - `validate_order(order, context)`: Pre-execution order validation
    - `priority` property: For conflict resolution (default: 0)

    Example Implementation:
        >>> class TrailingStopRule(RiskRule):
        ...     '''Trailing stop-loss that locks in profits.'''
        ...
        ...     def __init__(self, trail_pct: float = 0.02):
        ...         self.trail_pct = trail_pct
        ...
        ...     def evaluate(self, context: RiskContext) -> RiskDecision:
        ...         if not context.position or not context.position.is_long:
        ...             return RiskDecision.no_action()
        ...
        ...         # Calculate trailing stop based on max favorable excursion
        ...         if context.mfe_pct > self.trail_pct:
        ...             new_stop = context.current_price * (1 - self.trail_pct)
        ...             return RiskDecision.update_stops(
        ...                 update_stop_loss=new_stop,
        ...                 reason=f"Trailing stop: lock in {context.mfe_pct:.2%} gain"
        ...             )
        ...
        ...         return RiskDecision.no_action()
        ...
        ...     @property
        ...     def priority(self) -> int:
        ...         return 5  # Medium priority

    Usage:
        >>> # Add to RiskManager
        >>> manager = RiskManager()
        >>> manager.add_rule(TrailingStopRule(trail_pct=0.02))
        >>> manager.add_rule(VIXFilterRule())
        >>>
        >>> # Evaluate all rules on each market event
        >>> decision = manager.evaluate(context)
        >>> if decision.should_exit:
        ...     broker.submit_order(MarketOrder.sell(context.position.quantity))
    """

    @abstractmethod
    def evaluate(self, context: RiskContext) -> RiskDecision:
        """Evaluate risk context and generate decision.

        This is the core method that must be implemented by all risk rules.

        Args:
            context: Snapshot of risk-relevant state including:
                - Current position (quantity, entry price, unrealized P&L)
                - Latest market data (price, signals, market-wide context)
                - Portfolio state (equity, leverage, margin)
                - Computed metrics (MAE, MFE, exposure)

        Returns:
            RiskDecision indicating recommended action:
                - Exit position (immediately or at specific price)
                - Update stop-loss or take-profit levels
                - No action required

        Example:
            >>> def evaluate(self, context: RiskContext) -> RiskDecision:
            ...     # Exit if VIX spikes above 30
            ...     vix = context.market_context.get('vix')
            ...     if vix and vix > 30:
            ...         return RiskDecision.exit_now(
            ...             exit_type=ExitType.RISK_EXIT,
            ...             reason=f"VIX spike: {vix}",
            ...             priority=10
            ...         )
            ...     return RiskDecision.no_action()
        """
        pass

    def validate_order(
        self, order: Order, context: RiskContext
    ) -> Optional[Order]:
        """Optional pre-execution order validation and modification.

        Called before orders are submitted to the broker, allowing rules to:
        - Reject orders (return None)
        - Modify order size based on risk constraints
        - Add stop-loss/take-profit to orders
        - Prevent trades during adverse conditions

        Default implementation: pass-through (no validation)

        Args:
            order: Order about to be submitted
            context: Current risk context for decision making

        Returns:
            Modified order, original order (if valid), or None (to reject)

        Example:
            >>> def validate_order(self, order, context):
            ...     # Prevent new positions when VIX > 30
            ...     vix = context.market_context.get('vix')
            ...     if vix and vix > 30:
            ...         return None  # Reject order
            ...
            ...     # Reduce position size if leverage too high
            ...     if context.leverage > 2.0:
            ...         order.quantity = int(order.quantity * 0.5)
            ...
            ...     return order
        """
        return order  # Default: no validation, pass through

    @property
    def priority(self) -> int:
        """Priority for conflict resolution when merging decisions.

        Higher priority rules take precedence when multiple rules generate
        conflicting decisions. Common priority levels:

        - 0: Informational (default, e.g., logging, metrics)
        - 5: Stop updates (trailing stops, profit targets)
        - 10: Critical exits (stop-loss breaches, risk limits)
        - 15: Emergency exits (circuit breakers, system issues)

        Returns:
            Integer priority (default: 0)

        Example:
            >>> @property
            >>> def priority(self) -> int:
            ...     return 10  # Stop-loss rules are high priority
        """
        return 0  # Default: lowest priority

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"{self.__class__.__name__}(priority={self.priority})"


class CompositeRule(RiskRule):
    """Composite rule that combines multiple sub-rules.

    Useful for creating rule groups or applying multiple rules as a single unit.

    Example:
        >>> # Create a composite protective rule
        >>> protective = CompositeRule([
        ...     StopLossRule(stop_pct=0.05),
        ...     TakeProfitRule(target_pct=0.15),
        ...     TrailingStopRule(trail_pct=0.02)
        ... ])
        >>> manager.add_rule(protective)  # Add all 3 rules at once
    """

    def __init__(self, rules: list[RiskRule | RiskRuleProtocol]):
        """Initialize composite rule.

        Args:
            rules: List of sub-rules to evaluate
        """
        self.rules = rules

    def evaluate(self, context: RiskContext) -> RiskDecision:
        """Evaluate all sub-rules and merge their decisions.

        Args:
            context: Risk context to evaluate

        Returns:
            Merged decision from all sub-rules
        """
        decisions = []
        for rule in self.rules:
            if isinstance(rule, RiskRule):
                decision = rule.evaluate(context)
            elif callable(rule):  # RiskRuleProtocol
                decision = rule(context)
            else:
                raise TypeError(f"Invalid rule type: {type(rule)}")

            decisions.append(decision)

        # Merge all decisions using RiskDecision.merge()
        return RiskDecision.merge(decisions)

    def validate_order(
        self, order: Order, context: RiskContext
    ) -> Optional[Order]:
        """Run order validation through all sub-rules.

        If any rule rejects the order (returns None), the order is rejected.
        Otherwise, modifications are applied sequentially.

        Args:
            order: Order to validate
            context: Risk context

        Returns:
            Validated/modified order or None if rejected
        """
        current_order = order
        for rule in self.rules:
            if isinstance(rule, RiskRule):
                current_order = rule.validate_order(current_order, context)
                if current_order is None:
                    return None  # Rejected by this rule

        return current_order

    @property
    def priority(self) -> int:
        """Use maximum priority of all sub-rules."""
        if not self.rules:
            return 0

        return max(
            rule.priority if isinstance(rule, RiskRule) else 0
            for rule in self.rules
        )

    def __repr__(self) -> str:
        """String representation showing sub-rules."""
        return f"CompositeRule({len(self.rules)} rules, priority={self.priority})"


# Type alias for anything that can be used as a risk rule
RiskRuleLike = RiskRule | RiskRuleProtocol
