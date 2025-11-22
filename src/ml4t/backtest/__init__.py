"""ml4t.backtest - Minimal event-driven backtesting engine.

A clean, extensible backtesting engine with:
- Multi-asset support
- Polars-first data handling
- Pluggable commission/slippage models
- Same-bar and next-bar execution modes
- Live trading compatible interface
"""

__version__ = "0.2.0"

# Import from modules
from .types import (
    OrderType,
    OrderSide,
    OrderStatus,
    ExecutionMode,
    Order,
    Position,
    Fill,
    Trade,
)
from .models import (
    CommissionModel,
    SlippageModel,
    NoCommission,
    PercentageCommission,
    PerShareCommission,
    TieredCommission,
    CombinedCommission,
    NoSlippage,
    FixedSlippage,
    PercentageSlippage,
    VolumeShareSlippage,
)
from .datafeed import DataFeed
from .broker import Broker
from .strategy import Strategy
from .engine import Engine, run_backtest, BacktestEngine
from .config import (
    BacktestConfig,
    FillTiming,
    ExecutionPrice,
    ShareType,
    SizingMethod,
    SignalProcessing,
    CommissionModel as CommissionModelType,
    SlippageModel as SlippageModelType,
    PRESETS_DIR,
)

__all__ = [
    # Types
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "ExecutionMode",
    "Order",
    "Position",
    "Fill",
    "Trade",
    # Models
    "CommissionModel",
    "SlippageModel",
    "NoCommission",
    "PercentageCommission",
    "PerShareCommission",
    "TieredCommission",
    "CombinedCommission",
    "NoSlippage",
    "FixedSlippage",
    "PercentageSlippage",
    "VolumeShareSlippage",
    # Core
    "DataFeed",
    "Broker",
    "Strategy",
    "Engine",
    "BacktestEngine",  # Backward compatibility alias
    "run_backtest",
    # Configuration
    "BacktestConfig",
    "FillTiming",
    "ExecutionPrice",
    "ShareType",
    "SizingMethod",
    "SignalProcessing",
    "CommissionModelType",
    "SlippageModelType",
    "PRESETS_DIR",
]
