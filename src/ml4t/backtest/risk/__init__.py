"""Risk management framework for ml4t-backtest.

Provides position-level and portfolio-level risk management including:
- Static exits (stop-loss, take-profit, time exit)
- Dynamic exits (trailing stop, tightening trail, scaled exit)
- Signal-based exits
- Rule composition (RuleChain, AllOf, AnyOf)
- Portfolio-level limits (drawdown, position count, exposure)
"""

# Portfolio-level risk
from .portfolio import (
    DailyLossLimit,
    GrossExposureLimit,
    MaxDrawdownLimit,
    MaxExposureLimit,
    MaxPositionsLimit,
    NetExposureLimit,
    PortfolioLimit,
    PortfolioState,
    RiskManager,
)

# Position rules
from .position import (
    AllOf,
    AnyOf,
    PositionRule,
    RuleChain,
    ScaledExit,
    SignalExit,
    StopLoss,
    TakeProfit,
    TighteningTrailingStop,
    TimeExit,
    TrailingStop,
    VolatilityStop,
    VolatilityTrailingStop,
)
from .types import ActionType, PositionAction, PositionState

__all__ = [
    # Types
    "ActionType",
    "PositionAction",
    "PositionState",
    # Protocol
    "PositionRule",
    # Static rules
    "StopLoss",
    "TakeProfit",
    "TimeExit",
    # Dynamic rules
    "TrailingStop",
    "TighteningTrailingStop",
    "ScaledExit",
    "VolatilityStop",
    "VolatilityTrailingStop",
    # Signal rules
    "SignalExit",
    # Composition
    "RuleChain",
    "AllOf",
    "AnyOf",
    # Portfolio risk
    "RiskManager",
    "PortfolioLimit",
    "PortfolioState",
    "MaxDrawdownLimit",
    "MaxPositionsLimit",
    "MaxExposureLimit",
    "DailyLossLimit",
    "GrossExposureLimit",
    "NetExposureLimit",
]
