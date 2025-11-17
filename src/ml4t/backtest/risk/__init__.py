"""Risk management module for ml4t.backtest.

This module provides risk context, decision, and rule evaluation infrastructure
for position and portfolio risk management.

Main Components:
    - RiskContext: Immutable snapshot of position/portfolio state for risk evaluation
    - RiskDecision: Output of risk rule evaluation (exit, update stops, or no action)
    - RiskRule: Abstract base class for composable risk rules
    - RiskManager: Orchestrates rule evaluation with context caching
    - CompositeRule: Combine multiple rules into a single unit
    - RiskRuleProtocol: Protocol for callable risk rules (no inheritance needed)

Example:
    >>> from ml4t.backtest.risk import RiskManager, RiskContext, RiskDecision, RiskRule
    >>>
    >>> # Simple callable rule (no class needed)
    >>> def stop_loss_rule(context: RiskContext) -> RiskDecision:
    ...     if context.unrealized_pnl_pct < -0.05:
    ...         return RiskDecision.exit_now(
    ...             exit_type=ExitType.STOP_LOSS,
    ...             reason="5% stop-loss breach"
    ...         )
    ...     return RiskDecision.no_action()
    >>>
    >>> # Register with RiskManager
    >>> manager = RiskManager()
    >>> manager.add_rule(stop_loss_rule)
"""

from ml4t.backtest.risk.context import RiskContext
from ml4t.backtest.risk.decision import ExitType, RiskDecision
from ml4t.backtest.risk.manager import RiskManager
from ml4t.backtest.risk.rule import (
    CompositeRule,
    RiskRule,
    RiskRuleLike,
    RiskRuleProtocol,
)
from ml4t.backtest.risk.rules import (
    PriceBasedStopLoss,
    PriceBasedTakeProfit,
    TimeBasedExit,
)

__all__ = [
    "RiskContext",
    "RiskDecision",
    "ExitType",
    "RiskRule",
    "RiskRuleProtocol",
    "RiskRuleLike",
    "CompositeRule",
    "RiskManager",
    "TimeBasedExit",
    "PriceBasedStopLoss",
    "PriceBasedTakeProfit",
]
