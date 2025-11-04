"""
Framework adapters for cross-framework validation.

This module provides standardized interfaces for running identical
strategies across different backtesting frameworks to validate
QEngine's correctness and performance.
"""

from .base import (
    BaseFrameworkAdapter,
    BaseStrategy,
    MomentumStrategy,
    TradeRecord,
    ValidationResult,
)
from .qengine_adapter import QEngineAdapter
from .vectorbt_adapter import VectorBTAdapter

try:
    from .backtrader_adapter import BacktraderAdapter
except ImportError:
    BacktraderAdapter = None

try:
    from .zipline_adapter import ZiplineAdapter
except ImportError:
    ZiplineAdapter = None

__all__ = [
    "BacktraderAdapter",
    "BaseFrameworkAdapter",
    "BaseStrategy",
    "MomentumStrategy",
    "QEngineAdapter",
    "TradeRecord",
    "ValidationResult",
    "VectorBTAdapter",
    "ZiplineAdapter",
]
