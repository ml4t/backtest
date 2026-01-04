"""ML4T Backtest - Event-driven backtesting engine.

A minimal, event-driven backtesting engine with:
- Point-in-time correctness
- Exit-first order processing
- Cash and margin account policies
- VectorBT-validated results
"""

from ml4t.backtest.types import (
    Order,
    OrderType,
    OrderSide,
    Fill,
    Position,
    Trade,
)
from ml4t.backtest.config import BacktestConfig

__version__ = "1.0.0"

__all__ = [
    "Order",
    "OrderType",
    "OrderSide",
    "Fill",
    "Position",
    "Trade",
    "BacktestConfig",
]
