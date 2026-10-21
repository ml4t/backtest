"""Equity curve tracking and analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from ..capital import CapitalChange, flow_adjusted_returns
from .annualization import resolve_periods_per_year
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

if TYPE_CHECKING:
    from ..config import BacktestConfig


@dataclass
class EquityCurve:
    """Track portfolio equity over time with computed metrics.

    Attributes:
        timestamps: List of timestamps
        values: Portfolio values at each timestamp
    """

    timestamps: list[datetime] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    periods_per_year_override: float | None = None
    observation_based: bool = False
    initial_capital: float | None = None
    capital_changes: list[CapitalChange] = field(default_factory=list)

    def append(self, timestamp: datetime, value: float) -> None:
        """Add a data point."""
        self.timestamps.append(timestamp)
        self.values.append(value)

    def __len__(self) -> int:
        return len(self.values)

    @property
    def returns(self) -> np.ndarray:
        """Daily returns."""
        if self.capital_changes:
            return np.array(
                flow_adjusted_returns(
                    list(zip(self.timestamps, self.values)),
                    self.initial_value,
                    self.capital_changes,
                )
            )
        if self.initial_capital is not None and self.values:
            return returns_from_values([self.initial_capital, *self.values])
        if len(self.values) < 2:
            return np.array([])
        return returns_from_values(self.values)

    @property
    def cumulative_returns(self) -> np.ndarray:
        """Cumulative returns from start."""
        if len(self.values) < 1:
            return np.array([])
        initial = self.initial_value
        if self.capital_changes:
            return np.cumprod(1 + self.returns) - 1
        return np.array(self.values) / initial - 1

    @property
    def performance_values(self) -> np.ndarray:
        """Investment wealth at the initial capital scale, excluding external cash."""
        if self.capital_changes:
            return self.initial_value * np.cumprod(1 + self.returns)
        return np.array(self.values)

    @property
    def initial_value(self) -> float:
        """Starting portfolio value."""
        if self.initial_capital is not None:
            return self.initial_capital
        return self.values[0] if self.values else 0.0

    @property
    def final_value(self) -> float:
        """Ending portfolio value."""
        return self.values[-1] if self.values else 0.0

    @property
    def total_return(self) -> float:
        """Total return as decimal."""
        if not self.values or self.initial_value == 0:
            return 0.0
        if self.capital_changes:
            return float(self.cumulative_returns[-1])
        return self.values[-1] / self.initial_value - 1

    @property
    def years(self) -> float:
        """Duration in observation periods or elapsed wall-clock years."""
        if self.observation_based:
            return max(len(self.values) - 1, 0) / self.periods_per_year
        if len(self.timestamps) >= 2:
            elapsed_seconds = (self.timestamps[-1] - self.timestamps[0]).total_seconds()
            if elapsed_seconds > 0:
                return elapsed_seconds / (365.25 * 24 * 60 * 60)
        return len(self.values) / TRADING_DAYS_PER_YEAR if self.values else 0.0

    @property
    def periods_per_year(self) -> float:
        """Annualization factor, preferring configured cadence over elapsed-time inference."""
        if self.periods_per_year_override is not None:
            return float(self.periods_per_year_override)
        if self.observation_based:
            return float(TRADING_DAYS_PER_YEAR)
        if len(self.values) < 2 or len(self.timestamps) < 2:
            return float(TRADING_DAYS_PER_YEAR)
        elapsed_seconds = (self.timestamps[-1] - self.timestamps[0]).total_seconds()
        if elapsed_seconds <= 0:
            return float(TRADING_DAYS_PER_YEAR)
        periods = len(self.values) - 1
        inferred = periods * 365.25 * 24 * 60 * 60 / elapsed_seconds
        if not np.isfinite(inferred) or inferred <= 0:
            return float(TRADING_DAYS_PER_YEAR)
        return float(inferred)

    @classmethod
    def from_config(cls, config: BacktestConfig) -> EquityCurve:
        """Create an equity curve with annualization metadata derived from config."""
        feed_spec = config.resolved_feed_spec
        observation_based = config.time_contract_version == 2 and config.resolved_calendar is None
        periods_per_year = config.periods_per_year
        if periods_per_year is None and observation_based:
            periods_per_year = float(TRADING_DAYS_PER_YEAR)
        if periods_per_year is None:
            periods_per_year = resolve_periods_per_year(
                feed_spec.data_frequency,
                calendar=config.resolved_calendar,
            )
        return cls(
            periods_per_year_override=periods_per_year,
            observation_based=observation_based,
            initial_capital=config.initial_cash,
        )

    def max_drawdown_info(self) -> tuple[float, int, int]:
        """Maximum drawdown with peak/trough indices."""
        if self.initial_capital is not None:
            drawdown, peak, trough = max_drawdown([self.initial_capital, *self.performance_values])
            return drawdown, peak - 1, trough - 1
        return max_drawdown(self.values)

    @property
    def max_dd(self) -> float:
        """Maximum drawdown as negative decimal."""
        dd, _, _ = self.max_drawdown_info()
        return dd

    @property
    def cagr(self) -> float:
        """Compound Annual Growth Rate."""
        return cagr(self.initial_value, self.initial_value * (1 + self.total_return), self.years)

    @property
    def volatility(self) -> float:
        """Annualized volatility."""
        if len(self.returns) < 2:
            return 0.0
        base_vol = volatility(self.returns, annualize=False)
        return float(base_vol * np.sqrt(self.periods_per_year))

    @property
    def sharpe(self) -> float:
        """Annualized Sharpe ratio using inferred bar frequency."""
        if len(self.returns) < 2:
            return 0.0
        base_sharpe = sharpe_ratio(self.returns, annualize=False)
        return float(base_sharpe * np.sqrt(self.periods_per_year))

    @property
    def sortino(self) -> float:
        """Annualized Sortino ratio using inferred bar frequency."""
        if len(self.returns) < 2:
            return 0.0
        base_sortino = sortino_ratio(self.returns, annualize=False)
        if np.isinf(base_sortino):
            return float("inf")
        return float(base_sortino * np.sqrt(self.periods_per_year))

    def drawdown_series(self) -> np.ndarray:
        """Drawdown at each point (for underwater chart)."""
        if not self.values:
            return np.array([])
        arr = self.performance_values
        running_max = np.maximum.accumulate(arr)
        if self.initial_capital is not None:
            running_max = np.maximum(running_max, self.initial_capital)
        return (arr - running_max) / running_max

    def to_dict(self) -> dict:
        """Export metrics as dictionary."""
        return {
            "initial_value": self.initial_value,
            "final_value": self.final_value,
            "total_return": self.total_return,
            "cagr": self.cagr,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_dd,
            "calmar": calmar_ratio(self.cagr, self.max_dd),
            "volatility": self.volatility,
            "trading_days": len(self.values),
            "years": self.years,
        }
