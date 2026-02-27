"""ml4t.backtest - Minimal event-driven backtesting engine.

A clean, extensible backtesting engine with:
- Multi-asset support
- Polars-first data handling
- Pluggable commission/slippage models
- Same-bar and next-bar execution modes
- Live trading compatible interface
"""

try:
    from ml4t.backtest._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"

from .broker import Broker
from .config import BacktestConfig, Mode
from .datafeed import DataFeed
from .engine import Engine, run_backtest
from .result import BacktestResult

# Risk management rules (position-level)
from .risk.position.composite import RuleChain
from .risk.position.dynamic import TrailingStop
from .risk.position.static import StopLoss
from .strategy import Strategy
from .types import (
    ExecutionMode,
    ExitReason,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    StopFillMode,
    StopLevelBasis,
    Trade,
)

__all__ = [
    # Core API
    "DataFeed",
    "Broker",
    "Strategy",
    "Engine",
    "run_backtest",
    "BacktestConfig",
    "Mode",
    "BacktestResult",
    # Canonical domain types
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "ExecutionMode",
    "ExitReason",
    "StopFillMode",
    "StopLevelBasis",
    "Order",
    "Position",
    "Fill",
    "Trade",
    # Risk rules
    "StopLoss",
    "TrailingStop",
    "RuleChain",
]
