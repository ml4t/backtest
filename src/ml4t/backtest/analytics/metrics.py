"""Performance metrics for backtest evaluation."""

import math
from collections.abc import Sequence

import numpy as np

# Type for arrays that can be returns/values
ReturnsLike = Sequence[float] | np.ndarray

# Annualization factors
TRADING_DAYS_PER_YEAR = 252


def returns_from_values(values: Sequence[float]) -> np.ndarray:
    """Calculate returns from a series of portfolio values."""
    arr = np.array(values)
    return np.diff(arr) / arr[:-1]


def volatility(returns: ReturnsLike, annualize: bool = True) -> float:
    """Calculate volatility (standard deviation of returns).

    Args:
        returns: Sequence of period returns
        annualize: If True, annualize assuming daily returns

    Returns:
        Volatility as decimal (0.15 = 15%)
    """
    arr = np.array(returns)
    if len(arr) < 2:
        return 0.0
    vol = np.std(arr, ddof=1)
    if annualize:
        vol *= math.sqrt(TRADING_DAYS_PER_YEAR)
    return float(vol)


def sharpe_ratio(
    returns: ReturnsLike,
    risk_free_rate: float = 0.0,
    annualize: bool = True,
) -> float:
    """Calculate Sharpe ratio.

    Args:
        returns: Sequence of period returns
        risk_free_rate: Annual risk-free rate (default 0)
        annualize: If True, annualize assuming daily returns

    Returns:
        Sharpe ratio
    """
    arr = np.array(returns)
    if len(arr) < 2:
        return 0.0

    # Convert annual risk-free to daily if annualizing
    if annualize:
        daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
        excess_returns = arr - daily_rf
    else:
        excess_returns = arr - risk_free_rate

    mean_excess = np.mean(excess_returns)
    std = np.std(arr, ddof=1)

    if std == 0:
        return 0.0

    sharpe = mean_excess / std
    if annualize:
        sharpe *= math.sqrt(TRADING_DAYS_PER_YEAR)
    return float(sharpe)


def sortino_ratio(
    returns: ReturnsLike,
    risk_free_rate: float = 0.0,
    annualize: bool = True,
) -> float:
    """Calculate Sortino ratio (uses downside deviation).

    Args:
        returns: Sequence of period returns
        risk_free_rate: Annual risk-free rate (default 0)
        annualize: If True, annualize assuming daily returns

    Returns:
        Sortino ratio
    """
    arr = np.array(returns)
    if len(arr) < 2:
        return 0.0

    # Convert annual risk-free to daily if annualizing
    if annualize:
        daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
        excess_returns = arr - daily_rf
    else:
        excess_returns = arr - risk_free_rate

    mean_excess = np.mean(excess_returns)

    # Downside deviation: std of returns below target (0)
    downside = arr[arr < 0]
    if len(downside) < 2:
        return float("inf") if mean_excess > 0 else 0.0

    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return float("inf") if mean_excess > 0 else 0.0

    sortino = mean_excess / downside_std
    if annualize:
        sortino *= math.sqrt(TRADING_DAYS_PER_YEAR)
    return float(sortino)


def max_drawdown(values: Sequence[float]) -> tuple[float, int, int]:
    """Calculate maximum drawdown from portfolio values.

    Args:
        values: Sequence of portfolio values (not returns)

    Returns:
        Tuple of (max_drawdown_pct, peak_idx, trough_idx)
        max_drawdown_pct is negative (e.g., -0.20 for 20% drawdown)
    """
    arr = np.array(values)
    if len(arr) < 2:
        return 0.0, 0, 0

    # Running maximum
    running_max = np.maximum.accumulate(arr)
    drawdowns = (arr - running_max) / running_max

    trough_idx = int(np.argmin(drawdowns))
    peak_idx = int(np.argmax(arr[: trough_idx + 1])) if trough_idx > 0 else 0

    return float(drawdowns[trough_idx]), peak_idx, trough_idx


def cagr(
    initial_value: float,
    final_value: float,
    years: float,
) -> float:
    """Calculate Compound Annual Growth Rate.

    Args:
        initial_value: Starting portfolio value
        final_value: Ending portfolio value
        years: Number of years

    Returns:
        CAGR as decimal (0.15 = 15% annual return)
    """
    if initial_value <= 0 or years <= 0:
        return 0.0
    if final_value <= 0:
        return -1.0  # Total loss

    return (final_value / initial_value) ** (1 / years) - 1


def calmar_ratio(cagr_value: float, max_dd: float) -> float:
    """Calculate Calmar ratio (CAGR / Max Drawdown).

    Args:
        cagr_value: Compound Annual Growth Rate
        max_dd: Maximum drawdown (should be negative)

    Returns:
        Calmar ratio (higher is better)
    """
    if max_dd >= 0:
        return float("inf") if cagr_value > 0 else 0.0
    return cagr_value / abs(max_dd)
