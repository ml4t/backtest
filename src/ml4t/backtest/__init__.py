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

# Import from modules
# Analytics
# Deprecated: Import from analysis for backward compatibility
# These emit deprecation warnings when imported
import warnings as _warnings

from .analytics import (
    # Analytics classes
    EquityCurve,
    TradeAnalyzer,
    cagr,
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    # Bridge functions (diagnostic integration)
    to_equity_dataframe,
    to_returns_series,
    to_trade_record,
    to_trade_records,
    volatility,
)

with _warnings.catch_warnings():
    _warnings.filterwarnings("ignore", category=DeprecationWarning)
    from .analysis import BacktestAnalyzer, TradeStatistics

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
    InitialHwmSource,
    Mode,
    ShareType,
    SignalProcessing,
    SizingMethod,
    TrailHwmSource,
)
from .config import (
    CommissionModel as CommissionModelType,
)
from .config import (
    SlippageModel as SlippageModelType,
)
from .datafeed import DataFeed
from .engine import BacktestEngine, Engine, run_backtest

# Execution model (volume limits, market impact, rebalancing)
from .execution import (
    ExecutionLimits,
    ExecutionResult,
    LinearImpact,
    MarketImpactModel,
    NoImpact,
    NoLimits,
    RebalanceConfig,
    SquareRootImpact,
    TargetWeightExecutor,
    VolumeParticipationLimit,
)

# Export utilities
from .export import BacktestExporter
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

# Structured result
from .result import BacktestResult, enrich_trades_with_signals

# Risk management rules (position-level)
from .risk.position.composite import RuleChain
from .risk.position.dynamic import (
    ScaledExit,
    TighteningTrailingStop,
    TrailingStop,
    VolatilityStop,
    VolatilityTrailingStop,
)
from .risk.position.signal import SignalExit
from .risk.position.static import StopLoss, TimeExit

# Session alignment
from .sessions import SessionConfig, align_to_sessions, compute_session_pnl

# Strategy templates
from .strategies import (
    LongShortStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    SignalFollowingStrategy,
)
from .strategy import Strategy
from .types import (
    AssetClass,
    ContractSpec,
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
    # Types
    "AssetClass",
    "ContractSpec",
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
    "BacktestResult",
    "BacktestExporter",
    "enrich_trades_with_signals",
    # Strategy templates
    "SignalFollowingStrategy",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "LongShortStrategy",
    # Session alignment
    "SessionConfig",
    "compute_session_pnl",
    "align_to_sessions",
    # Configuration
    "BacktestConfig",
    "Mode",
    "DataFrequency",
    "FillTiming",
    "ExecutionPrice",
    "ShareType",
    "SizingMethod",
    "SignalProcessing",
    "TrailHwmSource",
    "InitialHwmSource",
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
    # Analysis (diagnostic integration)
    "BacktestAnalyzer",
    "TradeStatistics",
    "to_trade_record",
    "to_trade_records",
    "to_returns_series",
    "to_equity_dataframe",
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
    # Rebalancing
    "RebalanceConfig",
    "TargetWeightExecutor",
    # Risk management rules (position-level)
    "StopLoss",
    "TimeExit",
    "TrailingStop",
    "TighteningTrailingStop",
    "VolatilityStop",
    "VolatilityTrailingStop",
    "ScaledExit",
    "SignalExit",
    "RuleChain",
]
