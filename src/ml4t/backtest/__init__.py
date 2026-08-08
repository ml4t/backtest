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
from .config import BacktestConfig, CommissionType
from .datafeed import DataFeed
from .engine import Engine, run_backtest

# Execution: rebalancing
from .execution.rebalancer import RebalanceConfig, TargetWeightExecutor
from .execution.schedule import RebalanceCadence, RebalanceSchedule, resolve_rebalance_timestamps
from .result import (
    ArtifactDiagnostic,
    ArtifactError,
    ArtifactIncompleteError,
    ArtifactManifestError,
    ArtifactNotFoundError,
    ArtifactReadError,
    ArtifactWriteError,
    BacktestResult,
    UnsupportedArtifactVersionError,
)

# Risk management rules (position-level)
from .risk.position.composite import RuleChain
from .risk.position.dynamic import TrailingStop
from .risk.position.static import StopLoss, TakeProfit
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
    # Core API
    "DataFeed",
    "Broker",
    "Strategy",
    "Engine",
    "run_backtest",
    "BacktestConfig",
    "BacktestResult",
    "ArtifactDiagnostic",
    "ArtifactError",
    "ArtifactNotFoundError",
    "ArtifactManifestError",
    "ArtifactIncompleteError",
    "ArtifactReadError",
    "ArtifactWriteError",
    "UnsupportedArtifactVersionError",
    "CommissionType",
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
    # Asset specifications
    "AssetClass",
    "ContractSpec",
    # Execution: rebalancing
    "RebalanceConfig",
    "TargetWeightExecutor",
    "RebalanceCadence",
    "RebalanceSchedule",
    "resolve_rebalance_timestamps",
    # Risk rules
    "StopLoss",
    "TakeProfit",
    "TrailingStop",
    "RuleChain",
]
