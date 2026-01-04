"""Execution model for realistic order fills.

This module provides:
- Volume participation limits (max % of bar volume)
- Partial fills (fill what's possible, queue remainder)
- Market impact modeling (price impact based on size vs volume)
- Portfolio rebalancing utilities (target weight → orders)
"""

from .fill_executor import FillContext, FillExecutor
from .impact import (
    LinearImpact,
    MarketImpactModel,
    NoImpact,
    SquareRootImpact,
)
from .limits import (
    ExecutionLimits,
    NoLimits,
    VolumeParticipationLimit,
)
from .rebalancer import (
    RebalanceConfig,
    TargetWeightExecutor,
)
from .result import ExecutionResult

__all__ = [
    # Fill Execution
    "FillExecutor",
    "FillContext",
    # Limits
    "ExecutionLimits",
    "NoLimits",
    "VolumeParticipationLimit",
    # Impact
    "MarketImpactModel",
    "NoImpact",
    "LinearImpact",
    "SquareRootImpact",
    # Rebalancing
    "RebalanceConfig",
    "TargetWeightExecutor",
    # Result
    "ExecutionResult",
]
