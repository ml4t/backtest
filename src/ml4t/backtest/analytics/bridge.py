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

    With the aligned schema (v0.1.0a6+), field names now match between
    backtest Trade and diagnostic TradeRecord, simplifying this conversion.

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
        # Primary fields (aligned schema)
        "symbol": trade.symbol,
        "entry_time": trade.entry_time,
        "exit_time": trade.exit_time,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "quantity": trade.quantity,  # Signed (positive=long, negative=short)
        "pnl": trade.pnl,
        "pnl_percent": trade.pnl_percent,
        "bars_held": trade.bars_held,
        "fees": trade.fees,
        "slippage": trade.slippage,
        "exit_reason": trade.exit_reason,
        "status": trade.status,
        "mfe": trade.mfe,
        "mae": trade.mae,
        "metadata": trade.metadata,
        # Diagnostic-specific computed fields
        "duration": trade.exit_time - trade.entry_time,
        # Legacy field (diagnostic still expects this)
        "timestamp": trade.exit_time,  # Alias for exit_time
        "entry_timestamp": trade.entry_time,  # Alias for entry_time
        "direction": trade.direction,  # Derived from quantity sign
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
