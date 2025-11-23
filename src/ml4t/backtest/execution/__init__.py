"""Execution model for realistic order fills.

This module provides:
- Volume participation limits (max % of bar volume)
- Partial fills (fill what's possible, queue remainder)
- Market impact modeling (price impact based on size vs volume)
"""

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
from .result import ExecutionResult

__all__ = [
    # Limits
    "ExecutionLimits",
    "NoLimits",
    "VolumeParticipationLimit",
    # Impact
    "MarketImpactModel",
    "NoImpact",
    "LinearImpact",
    "SquareRootImpact",
    # Result
    "ExecutionResult",
]
