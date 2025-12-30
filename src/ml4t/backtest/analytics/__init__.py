"""Analytics module for backtest performance analysis."""

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
    "EquityCurve",
    "TradeAnalyzer",
    "MAEMFEAnalyzer",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "cagr",
    "volatility",
]
