"""Bridge ml4t.backtest results to ml4t.diagnostic for comprehensive analysis.

.. deprecated:: 0.3.0
    This module is deprecated. Import from ml4t.backtest.analytics instead:
    - to_trade_record, to_trade_records -> from ml4t.backtest.analytics.bridge
    - TradeStatistics -> use ml4t.backtest.analytics.TradeAnalyzer
    - BacktestAnalyzer -> use BacktestResult directly

This module is kept for backward compatibility and re-exports from analytics.

Example - New recommended approach:
    >>> from ml4t.backtest import BacktestResult
    >>> result = engine.run()  # Returns BacktestResult
    >>> trades_df = result.to_trades_dataframe()
    >>> metrics = result.metrics

Example - For diagnostic integration:
    >>> from ml4t.backtest.analytics import to_trade_records, TradeAnalyzer
    >>> records = to_trade_records(engine.broker.trades)
    >>> analyzer = TradeAnalyzer(engine.broker.trades)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

# Re-export from analytics.bridge for backward compatibility
from ml4t.backtest.analytics.bridge import (
    to_equity_dataframe,
    to_returns_series,
    to_trade_record,
    to_trade_records,
)
from ml4t.backtest.types import Trade

if TYPE_CHECKING:
    from ml4t.backtest.engine import Engine

# Emit deprecation warning on import
warnings.warn(
    "ml4t.backtest.analysis is deprecated. Use ml4t.backtest.analytics instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "to_trade_record",
    "to_trade_records",
    "to_returns_series",
    "to_equity_dataframe",
    "TradeStatistics",
    "BacktestAnalyzer",
]


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
            equity: list[float] = getattr(self.broker, "equity_history", [])
            return equity
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
