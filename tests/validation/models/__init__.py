"""Validation-specific model implementations for cross-framework testing.

This module contains implementations that exist ONLY for validation purposes,
specifically to match the behavior of other backtesting frameworks (VectorBT,
Zipline, Backtrader) during comparative testing.

These models should NOT be used in production code. They serve as test fixtures
to ensure qengine produces identical results to reference implementations.

Available models:
- VectorBTCommission: VectorBT Pro two-component fee model (percentage + fixed)
- VectorBTInfiniteSizer: VectorBT's size=np.inf position sizing with granularity
- VectorBTSlippage: VectorBT's multiplicative slippage formula

Import these ONLY from tests/validation/, never from src/qengine production code.
"""

from .vectorbt_commission import VectorBTCommission
from .vectorbt_sizer import VectorBTInfiniteSizer
from .vectorbt_slippage import VectorBTSlippage

__all__ = [
    "VectorBTCommission",
    "VectorBTInfiniteSizer",
    "VectorBTSlippage",
]
