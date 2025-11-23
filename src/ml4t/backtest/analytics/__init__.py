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
from .trades import TradeAnalyzer

__all__ = [
    "EquityCurve",
    "TradeAnalyzer",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "cagr",
    "volatility",
]
