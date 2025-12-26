"""Bridge ml4t.backtest results to ml4t.diagnostic for comprehensive analysis.

This module provides the integration layer between ml4t.backtest (execution)
and ml4t.diagnostic (analysis), enabling:

1. Trade-level statistics (win rate, profit factor, etc.)
2. SHAP-based error analysis for ML strategies
3. Benchmark comparison and statistical significance
4. Comprehensive backtest reports

The key insight: Backtest focuses on execution fidelity, Diagnostic focuses on
analytics. This adapter bridges them cleanly.

Example - Basic trade analysis:
    >>> from ml4t.backtest import Engine
    >>> from ml4t.backtest.analysis import BacktestAnalyzer
    >>>
    >>> # Run backtest
    >>> result = engine.run()
    >>>
    >>> # Analyze with diagnostic
    >>> analyzer = BacktestAnalyzer(engine)
    >>> stats = analyzer.trade_statistics()
    >>> print(f"Win rate: {stats.win_rate:.2%}")
    >>> print(f"Profit factor: {stats.profit_factor:.2f}")

Example - Full diagnostic integration:
    >>> from ml4t.backtest.analysis import to_trade_records, to_returns_series
    >>> from ml4t.diagnostic.evaluation import TradeAnalysis
    >>>
    >>> # Convert trades
    >>> trades = to_trade_records(engine.broker.trades)
    >>>
    >>> # Use diagnostic library directly
    >>> analyzer = TradeAnalysis(trades)
    >>> worst = analyzer.worst_trades(n=20)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from ml4t.backtest.types import Trade

if TYPE_CHECKING:
    from ml4t.backtest.engine import Engine


def to_trade_record(trade: Trade) -> dict[str, Any]:
    """Convert a backtest Trade to diagnostic TradeRecord format.

    This creates a dictionary compatible with ml4t.diagnostic.integration.TradeRecord.
    We use a dict to avoid hard dependency on diagnostic library.

    Args:
        trade: A completed Trade from backtest

    Returns:
        Dictionary matching TradeRecord schema

    Example:
        >>> from ml4t.backtest.analysis import to_trade_record
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
        "equity": equity_history,
        "returns": returns.tolist(),
    }

    if timestamps is not None:
        data["timestamp"] = timestamps
    else:
        # Generate integer index if no timestamps
        data["bar"] = list(range(n))

    return pl.DataFrame(data)


class TradeStatistics:
    """Compute comprehensive trade statistics from backtest results.

    This provides the same metrics as ml4t.diagnostic.evaluation.TradeStatistics
    but can be computed directly from backtest trades without the diagnostic
    library dependency.

    For more advanced analysis (SHAP, clustering, hypothesis generation),
    use the full diagnostic library via to_trade_records().

    Attributes:
        n_trades: Total number of completed trades
        n_winners: Number of profitable trades
        n_losers: Number of losing trades
        win_rate: Fraction of winning trades
        profit_factor: Gross profit / gross loss
        avg_pnl: Mean P&L per trade
        avg_winner: Average P&L of winning trades
        avg_loser: Average P&L of losing trades
        expectancy: Expected value per trade
        avg_bars_held: Average holding period in bars

    Example:
        >>> stats = TradeStatistics.from_trades(engine.broker.trades)
        >>> print(stats.summary())
    """

    def __init__(
        self,
        n_trades: int,
        n_winners: int,
        n_losers: int,
        win_rate: float,
        profit_factor: float | None,
        total_pnl: float,
        avg_pnl: float,
        pnl_std: float,
        avg_winner: float | None,
        avg_loser: float | None,
        max_winner: float,
        max_loser: float,
        avg_bars_held: float,
        avg_pnl_percent: float,
        total_commission: float,
        total_slippage: float,
    ):
        self.n_trades = n_trades
        self.n_winners = n_winners
        self.n_losers = n_losers
        self.win_rate = win_rate
        self.profit_factor = profit_factor
        self.total_pnl = total_pnl
        self.avg_pnl = avg_pnl
        self.pnl_std = pnl_std
        self.avg_winner = avg_winner
        self.avg_loser = avg_loser
        self.max_winner = max_winner
        self.max_loser = max_loser
        self.avg_bars_held = avg_bars_held
        self.avg_pnl_percent = avg_pnl_percent
        self.total_commission = total_commission
        self.total_slippage = total_slippage

    @property
    def expectancy(self) -> float:
        """Expected value per trade: win_rate * avg_winner - (1 - win_rate) * |avg_loser|"""
        if self.avg_winner is None or self.avg_loser is None:
            return self.avg_pnl
        return self.win_rate * self.avg_winner + (1 - self.win_rate) * self.avg_loser

    @property
    def payoff_ratio(self) -> float | None:
        """Ratio of average winner to average loser (in absolute terms)."""
        if self.avg_winner is None or self.avg_loser is None or self.avg_loser == 0:
            return None
        return self.avg_winner / abs(self.avg_loser)

    @classmethod
    def from_trades(cls, trades: list[Trade]) -> TradeStatistics:
        """Compute statistics from list of Trade objects.

        Args:
            trades: List of completed trades from broker.trades

        Returns:
            TradeStatistics instance with all computed metrics
        """
        if not trades:
            return cls(
                n_trades=0,
                n_winners=0,
                n_losers=0,
                win_rate=0.0,
                profit_factor=None,
                total_pnl=0.0,
                avg_pnl=0.0,
                pnl_std=0.0,
                avg_winner=None,
                avg_loser=None,
                max_winner=0.0,
                max_loser=0.0,
                avg_bars_held=0.0,
                avg_pnl_percent=0.0,
                total_commission=0.0,
                total_slippage=0.0,
            )

        pnls = np.array([t.pnl for t in trades])
        pnl_pcts = np.array([t.pnl_percent for t in trades])
        bars = np.array([t.bars_held for t in trades])
        commissions = np.array([t.commission for t in trades])
        slippages = np.array([t.slippage for t in trades])

        n_trades = len(trades)
        winners = pnls[pnls > 0]
        losers = pnls[pnls < 0]
        n_winners = len(winners)
        n_losers = len(losers)

        win_rate = n_winners / n_trades if n_trades > 0 else 0.0
        total_pnl = float(pnls.sum())
        avg_pnl = float(pnls.mean())
        pnl_std = float(pnls.std()) if n_trades > 1 else 0.0

        avg_winner = float(winners.mean()) if len(winners) > 0 else None
        avg_loser = float(losers.mean()) if len(losers) > 0 else None
        max_winner = float(pnls.max()) if n_trades > 0 else 0.0
        max_loser = float(pnls.min()) if n_trades > 0 else 0.0

        gross_profit = float(winners.sum()) if len(winners) > 0 else 0.0
        gross_loss = abs(float(losers.sum())) if len(losers) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

        avg_bars_held = float(bars.mean()) if n_trades > 0 else 0.0
        avg_pnl_percent = float(pnl_pcts.mean()) if n_trades > 0 else 0.0

        return cls(
            n_trades=n_trades,
            n_winners=n_winners,
            n_losers=n_losers,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_pnl=total_pnl,
            avg_pnl=avg_pnl,
            pnl_std=pnl_std,
            avg_winner=avg_winner,
            avg_loser=avg_loser,
            max_winner=max_winner,
            max_loser=max_loser,
            avg_bars_held=avg_bars_held,
            avg_pnl_percent=avg_pnl_percent,
            total_commission=float(commissions.sum()),
            total_slippage=float(slippages.sum()),
        )

    def summary(self) -> str:
        """Generate human-readable summary of trade statistics."""
        lines = [
            "Trade Statistics",
            "=" * 50,
            f"Total Trades: {self.n_trades}",
            f"Winners: {self.n_winners} | Losers: {self.n_losers}",
            f"Win Rate: {self.win_rate:.2%}",
            "",
            "P&L Metrics",
            "-" * 50,
            f"Total P&L: ${self.total_pnl:,.2f}",
            f"Average P&L: ${self.avg_pnl:,.2f} (±${self.pnl_std:,.2f})",
            f"Avg Return: {self.avg_pnl_percent:.2%}",
        ]

        if self.avg_winner is not None:
            lines.append(f"Avg Winner: ${self.avg_winner:,.2f}")
        if self.avg_loser is not None:
            lines.append(f"Avg Loser: ${self.avg_loser:,.2f}")
        if self.profit_factor is not None:
            lines.append(f"Profit Factor: {self.profit_factor:.2f}")
        if self.payoff_ratio is not None:
            lines.append(f"Payoff Ratio: {self.payoff_ratio:.2f}")

        lines.extend(
            [
                f"Expectancy: ${self.expectancy:,.2f}",
                f"Max Winner: ${self.max_winner:,.2f}",
                f"Max Loser: ${self.max_loser:,.2f}",
                "",
                "Execution Metrics",
                "-" * 50,
                f"Avg Holding Period: {self.avg_bars_held:.1f} bars",
                f"Total Commission: ${self.total_commission:,.2f}",
                f"Total Slippage: ${self.total_slippage:,.2f}",
            ]
        )

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Export statistics as dictionary."""
        return {
            "n_trades": self.n_trades,
            "n_winners": self.n_winners,
            "n_losers": self.n_losers,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_pnl": self.total_pnl,
            "avg_pnl": self.avg_pnl,
            "pnl_std": self.pnl_std,
            "avg_winner": self.avg_winner,
            "avg_loser": self.avg_loser,
            "max_winner": self.max_winner,
            "max_loser": self.max_loser,
            "expectancy": self.expectancy,
            "payoff_ratio": self.payoff_ratio,
            "avg_bars_held": self.avg_bars_held,
            "avg_pnl_percent": self.avg_pnl_percent,
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
        }


class BacktestAnalyzer:
    """High-level analyzer for backtest results.

    Provides convenient access to trade statistics and prepares data
    for the diagnostic library.

    Example:
        >>> engine = Engine(feed, strategy, initial_cash=100_000)
        >>> result = engine.run()
        >>>
        >>> analyzer = BacktestAnalyzer(engine)
        >>> print(analyzer.trade_statistics().summary())
        >>>
        >>> # For advanced analysis with diagnostic library
        >>> trade_records = analyzer.get_trade_records()
    """

    def __init__(self, engine: Engine):
        """Initialize analyzer with completed engine.

        Args:
            engine: Engine instance after run() has been called
        """
        self.engine = engine
        self.broker = engine.broker
        self._trade_stats: TradeStatistics | None = None

    @property
    def trades(self) -> list[Trade]:
        """Get list of completed trades."""
        return self.broker.trades

    @property
    def equity_history(self) -> list[float]:
        """Get equity curve (list of portfolio values)."""
        # Engine stores equity_curve as list of (timestamp, value) tuples
        if hasattr(self.engine, "equity_curve"):
            return [value for _, value in self.engine.equity_curve]
        # Fallback for older broker interface
        if hasattr(self.broker, "equity_history"):
            return self.broker.equity_history
        return []

    def trade_statistics(self) -> TradeStatistics:
        """Compute comprehensive trade statistics.

        Returns:
            TradeStatistics with all metrics
        """
        if self._trade_stats is None:
            self._trade_stats = TradeStatistics.from_trades(self.trades)
        return self._trade_stats

    def get_trade_records(self) -> list[dict[str, Any]]:
        """Get trades in diagnostic TradeRecord format.

        Returns:
            List of dicts compatible with ml4t.diagnostic TradeRecord
        """
        return to_trade_records(self.trades)

    def get_returns_series(self) -> pl.Series:
        """Get returns as Polars Series for diagnostic analysis.

        Returns:
            Series of period returns
        """
        return to_returns_series(self.equity_history)

    def get_equity_dataframe(self) -> pl.DataFrame:
        """Get equity curve as DataFrame.

        Returns:
            DataFrame with equity and returns columns
        """
        return to_equity_dataframe(self.equity_history)

    def get_trades_dataframe(self) -> pl.DataFrame:
        """Get trades as Polars DataFrame for analysis.

        Returns:
            DataFrame with one row per trade
        """
        if not self.trades:
            return pl.DataFrame()

        records = []
        for t in self.trades:
            records.append(
                {
                    "asset": t.asset,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "pnl": t.pnl,
                    "pnl_percent": t.pnl_percent,
                    "bars_held": t.bars_held,
                    "commission": t.commission,
                    "slippage": t.slippage,
                    "direction": "long" if t.quantity > 0 else "short",
                    "mfe": t.max_favorable_excursion,
                    "mae": t.max_adverse_excursion,
                }
            )

        return pl.DataFrame(records)

    def summary(self) -> str:
        """Generate comprehensive backtest summary.

        Returns:
            Formatted summary string
        """
        stats = self.trade_statistics()

        # Get backtest-level metrics
        equity = self.equity_history
        initial = equity[0] if equity else 0
        final = equity[-1] if equity else 0
        total_return = (final - initial) / initial if initial > 0 else 0

        lines = [
            "Backtest Summary",
            "=" * 60,
            f"Initial Capital: ${initial:,.2f}",
            f"Final Value: ${final:,.2f}",
            f"Total Return: {total_return:.2%}",
            "",
            stats.summary(),
        ]

        return "\n".join(lines)
