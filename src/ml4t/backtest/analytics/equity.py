"""Equity curve tracking and analysis."""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .metrics import (
    TRADING_DAYS_PER_YEAR,
    cagr,
    calmar_ratio,
    max_drawdown,
    returns_from_values,
    sharpe_ratio,
    sortino_ratio,
    volatility,
)


@dataclass
class EquityCurve:
    """Track portfolio equity over time with computed metrics.

    Attributes:
        timestamps: List of timestamps
        values: Portfolio values at each timestamp
    """

    timestamps: list[datetime] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def append(self, timestamp: datetime, value: float) -> None:
        """Add a data point."""
        self.timestamps.append(timestamp)
        self.values.append(value)

    def __len__(self) -> int:
        return len(self.values)

    @property
    def returns(self) -> np.ndarray:
        """Daily returns."""
        if len(self.values) < 2:
            return np.array([])
        return returns_from_values(self.values)

    @property
    def cumulative_returns(self) -> np.ndarray:
        """Cumulative returns from start."""
        if len(self.values) < 1:
            return np.array([])
        initial = self.values[0]
        return np.array(self.values) / initial - 1

    @property
    def initial_value(self) -> float:
        """Starting portfolio value."""
        return self.values[0] if self.values else 0.0

    @property
    def final_value(self) -> float:
        """Ending portfolio value."""
        return self.values[-1] if self.values else 0.0

    @property
    def total_return(self) -> float:
        """Total return as decimal."""
        if not self.values or self.values[0] == 0:
            return 0.0
        return self.values[-1] / self.values[0] - 1

    @property
    def years(self) -> float:
        """Duration in years based on trading days."""
        return len(self.values) / TRADING_DAYS_PER_YEAR if self.values else 0.0

    def sharpe(self, risk_free_rate: float = 0.0) -> float:
        """Annualized Sharpe ratio."""
        return sharpe_ratio(self.returns, risk_free_rate)

    def sortino(self, risk_free_rate: float = 0.0) -> float:
        """Annualized Sortino ratio."""
        return sortino_ratio(self.returns, risk_free_rate)

    def max_drawdown_info(self) -> tuple[float, int, int]:
        """Maximum drawdown with peak/trough indices."""
        return max_drawdown(self.values)

    @property
    def max_dd(self) -> float:
        """Maximum drawdown as negative decimal."""
        dd, _, _ = self.max_drawdown_info()
        return dd

    @property
    def cagr(self) -> float:
        """Compound Annual Growth Rate."""
        return cagr(self.initial_value, self.final_value, self.years)

    @property
    def calmar(self) -> float:
        """Calmar ratio (CAGR / Max Drawdown)."""
        return calmar_ratio(self.cagr, self.max_dd)

    @property
    def volatility(self) -> float:
        """Annualized volatility."""
        return volatility(self.returns)

    def drawdown_series(self) -> np.ndarray:
        """Drawdown at each point (for underwater chart)."""
        if len(self.values) < 2:
            return np.array([])
        arr = np.array(self.values)
        running_max = np.maximum.accumulate(arr)
        return (arr - running_max) / running_max

    def to_dict(self) -> dict:
        """Export metrics as dictionary."""
        return {
            "initial_value": self.initial_value,
            "final_value": self.final_value,
            "total_return": self.total_return,
            "cagr": self.cagr,
            "sharpe": self.sharpe(),
            "sortino": self.sortino(),
            "max_drawdown": self.max_dd,
            "calmar": self.calmar,
            "volatility": self.volatility,
            "trading_days": len(self.values),
            "years": self.years,
        }
