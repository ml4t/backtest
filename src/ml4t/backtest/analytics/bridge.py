"""Bridge functions for ml4t.diagnostic integration.

This module provides conversion functions to bridge ml4t.backtest results
to ml4t.diagnostic for comprehensive analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from ml4t.backtest.types import Trade


def to_trade_record(trade: Trade) -> dict[str, Any]:
    """Convert a backtest Trade to diagnostic TradeRecord format.

    This creates a dictionary compatible with ml4t.diagnostic.integration.TradeRecord.
    We use a dict to avoid hard dependency on diagnostic library.

    Args:
        trade: A completed Trade from backtest

    Returns:
        Dictionary matching TradeRecord schema

    Example:
        >>> from ml4t.backtest.analytics.bridge import to_trade_record
        >>> record = to_trade_record(trade)
        >>> # Use with diagnostic
        >>> from ml4t.diagnostic.integration import TradeRecord
        >>> tr = TradeRecord(**record)
    """
    return {
        "timestamp": trade.exit_time,
        "symbol": trade.asset,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "pnl": trade.pnl,
        "duration": trade.exit_time - trade.entry_time,
        "direction": "long" if trade.quantity > 0 else "short",
        "quantity": abs(trade.quantity),
        "entry_timestamp": trade.entry_time,
        "fees": trade.commission,
        "slippage": trade.slippage,
        "metadata": {
            "entry_signals": trade.entry_signals,
            "exit_signals": trade.exit_signals,
            "bars_held": trade.bars_held,
            "pnl_percent": trade.pnl_percent,
            "mfe": trade.max_favorable_excursion,
            "mae": trade.max_adverse_excursion,
        },
    }


def to_trade_records(trades: list[Trade]) -> list[dict[str, Any]]:
    """Convert list of backtest trades to diagnostic format.

    Args:
        trades: List of Trade objects from broker.trades

    Returns:
        List of dictionaries matching TradeRecord schema

    Example:
        >>> trades = engine.broker.trades
        >>> records = to_trade_records(trades)
        >>>
        >>> # Use with diagnostic TradeAnalysis
        >>> from ml4t.diagnostic.integration import TradeRecord
        >>> from ml4t.diagnostic.evaluation import TradeAnalysis
        >>> trade_records = [TradeRecord(**r) for r in records]
        >>> analyzer = TradeAnalysis(trade_records)
    """
    return [to_trade_record(t) for t in trades]


def to_returns_series(equity_curve: list[float] | np.ndarray) -> pl.Series:
    """Convert equity curve to returns series for diagnostic analysis.

    Args:
        equity_curve: List or array of portfolio values over time

    Returns:
        Polars Series of period returns

    Example:
        >>> returns = to_returns_series(engine.broker.equity_history)
        >>> # Use with diagnostic Sharpe analysis
        >>> from ml4t.diagnostic.evaluation import sharpe_ratio
        >>> sr = sharpe_ratio(returns, confidence_intervals=True)
    """
    values = np.array(equity_curve)
    if len(values) < 2:
        return pl.Series("returns", [], dtype=pl.Float64)
    returns = np.diff(values) / values[:-1]
    return pl.Series("returns", returns)


def to_equity_dataframe(
    equity_history: list[float],
    timestamps: list[Any] | None = None,
) -> pl.DataFrame:
    """Convert equity history to DataFrame with timestamps.

    Args:
        equity_history: List of portfolio values
        timestamps: Optional list of timestamps (same length)

    Returns:
        DataFrame with 'timestamp', 'equity', 'returns' columns
    """
    n = len(equity_history)
    if n == 0:
        return pl.DataFrame(
            schema={"timestamp": pl.Datetime, "equity": pl.Float64, "returns": pl.Float64}
        )

    # Calculate returns
    values = np.array(equity_history)
    returns = np.zeros(n)
    returns[1:] = np.diff(values) / values[:-1]

    data = {
        "equity": [float(x) for x in equity_history],  # Ensure consistent float type
        "returns": returns.tolist(),
    }

    if timestamps is not None:
        data["timestamp"] = timestamps
    else:
        # Generate integer index if no timestamps
        data["bar"] = list(range(n))

    return pl.DataFrame(data)
