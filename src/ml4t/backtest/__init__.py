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
# Analytics
from .analytics import (
    EquityCurve,
    TradeAnalyzer,
    cagr,
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    volatility,
)
from .broker import Broker

# Calendar functions (pandas_market_calendars integration)
from .calendar import (
    CALENDAR_ALIASES,
    filter_to_trading_days,
    filter_to_trading_sessions,
    generate_trading_minutes,
    get_calendar,
    get_early_closes,
    get_holidays,
    get_schedule,
    get_trading_days,
    is_market_open,
    is_trading_day,
    list_calendars,
    next_trading_day,
    previous_trading_day,
)
from .config import (
    PRESETS_DIR,
    BacktestConfig,
    DataFrequency,
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

# Execution model (volume limits, market impact)
from .execution import (
    ExecutionLimits,
    ExecutionResult,
    LinearImpact,
    MarketImpactModel,
    NoImpact,
    NoLimits,
    SquareRootImpact,
    VolumeParticipationLimit,
)
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
    StopFillMode,
    StopLevelBasis,
    Trade,
)

__all__ = [
    # Types
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "ExecutionMode",
    "StopFillMode",
    "StopLevelBasis",
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
    "DataFrequency",
    "FillTiming",
    "ExecutionPrice",
    "ShareType",
    "SizingMethod",
    "SignalProcessing",
    "CommissionModelType",
    "SlippageModelType",
    "PRESETS_DIR",
    # Analytics
    "EquityCurve",
    "TradeAnalyzer",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "cagr",
    "volatility",
    # Calendar functions
    "CALENDAR_ALIASES",
    "get_calendar",
    "get_schedule",
    "get_trading_days",
    "is_trading_day",
    "is_market_open",
    "next_trading_day",
    "previous_trading_day",
    "list_calendars",
    "get_holidays",
    "get_early_closes",
    "filter_to_trading_days",
    "filter_to_trading_sessions",
    "generate_trading_minutes",
    # Execution model
    "ExecutionLimits",
    "NoLimits",
    "VolumeParticipationLimit",
    "MarketImpactModel",
    "NoImpact",
    "LinearImpact",
    "SquareRootImpact",
    "ExecutionResult",
]
