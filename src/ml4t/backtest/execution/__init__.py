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
    AssetExecutionLimits,
    ExecutionLimits,
    NoLimits,
    PositiveVolumeLimit,
    SharedVolumeParticipationLimit,
    VolumeParticipationLimit,
)
from .rebalancer import (
    RebalanceConfig,
    TargetWeightExecutor,
)
from .result import ExecutionResult
from .schedule import (
    RebalanceCadence,
    RebalanceSchedule,
    is_rebalance_timestamp,
    resolve_rebalance_timestamps,
    session_date_for_timestamp,
)

__all__ = [
    # Fill Execution
    "FillExecutor",
    "FillContext",
    # Limits
    "ExecutionLimits",
    "AssetExecutionLimits",
    "NoLimits",
    "PositiveVolumeLimit",
    "VolumeParticipationLimit",
    "SharedVolumeParticipationLimit",
    # Impact
    "MarketImpactModel",
    "NoImpact",
    "LinearImpact",
    "SquareRootImpact",
    # Rebalancing
    "RebalanceConfig",
    "TargetWeightExecutor",
    "RebalanceCadence",
    "RebalanceSchedule",
    "is_rebalance_timestamp",
    "resolve_rebalance_timestamps",
    "session_date_for_timestamp",
    # Result
    "ExecutionResult",
]
