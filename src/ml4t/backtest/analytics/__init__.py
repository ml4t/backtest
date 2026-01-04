"""Analytics module for backtest performance analysis."""

from .bridge import (
    to_equity_dataframe,
    to_returns_series,
    to_trade_record,
    to_trade_records,
)
from .equity import EquityCurve
from .metrics import (
    cagr,
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    volatility,
)
from .trades import MAEMFEAnalyzer, TradeAnalyzer

__all__ = [
    # Bridge functions (diagnostic integration)
    "to_trade_record",
    "to_trade_records",
    "to_returns_series",
    "to_equity_dataframe",
    # Analytics classes
    "EquityCurve",
    "TradeAnalyzer",
    "MAEMFEAnalyzer",
    # Metric functions
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "cagr",
    "volatility",
]
