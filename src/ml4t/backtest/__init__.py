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
from .broker import Broker
from .config import (
    PRESETS_DIR,
    BacktestConfig,
    ExecutionPrice,
    FillTiming,
    ShareType,
    SignalProcessing,
    SizingMethod,
)
from .config import (
    CommissionModel as CommissionModelType,
)
from .config import (
    SlippageModel as SlippageModelType,
)
from .datafeed import DataFeed
from .engine import BacktestEngine, Engine, run_backtest
from .models import (
    CombinedCommission,
    CommissionModel,
    FixedSlippage,
    NoCommission,
    NoSlippage,
    PercentageCommission,
    PercentageSlippage,
    PerShareCommission,
    SlippageModel,
    TieredCommission,
    VolumeShareSlippage,
)
from .strategy import Strategy
from .types import (
    ExecutionMode,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Trade,
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
