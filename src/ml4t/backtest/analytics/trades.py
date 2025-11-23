"""Trade analysis and statistics."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..types import Trade


@dataclass
class TradeAnalyzer:
    """Analyze a collection of trades for performance statistics."""

    trades: Sequence["Trade"]

    def __post_init__(self):
        self._pnls = np.array([t.pnl for t in self.trades]) if self.trades else np.array([])
        self._returns = (
            np.array([t.pnl_percent for t in self.trades]) if self.trades else np.array([])
        )

    @property
    def num_trades(self) -> int:
        """Total number of trades."""
        return len(self.trades)

    @property
    def num_winners(self) -> int:
        """Number of winning trades (pnl > 0)."""
        return int(np.sum(self._pnls > 0))

    @property
    def num_losers(self) -> int:
        """Number of losing trades (pnl < 0)."""
        return int(np.sum(self._pnls < 0))

    @property
    def win_rate(self) -> float:
        """Percentage of winning trades."""
        if self.num_trades == 0:
            return 0.0
        return self.num_winners / self.num_trades

    @property
    def gross_profit(self) -> float:
        """Sum of all winning trade PnLs."""
        winners = self._pnls[self._pnls > 0]
        return float(np.sum(winners)) if len(winners) > 0 else 0.0

    @property
    def gross_loss(self) -> float:
        """Sum of all losing trade PnLs (negative)."""
        losers = self._pnls[self._pnls < 0]
        return float(np.sum(losers)) if len(losers) > 0 else 0.0

    @property
    def net_profit(self) -> float:
        """Total profit/loss."""
        return float(np.sum(self._pnls)) if len(self._pnls) > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        """Gross profit / |Gross loss|. Higher is better."""
        if self.gross_loss == 0:
            return float("inf") if self.gross_profit > 0 else 0.0
        return self.gross_profit / abs(self.gross_loss)

    @property
    def avg_win(self) -> float:
        """Average winning trade PnL."""
        winners = self._pnls[self._pnls > 0]
        return float(np.mean(winners)) if len(winners) > 0 else 0.0

    @property
    def avg_loss(self) -> float:
        """Average losing trade PnL (negative)."""
        losers = self._pnls[self._pnls < 0]
        return float(np.mean(losers)) if len(losers) > 0 else 0.0

    @property
    def avg_trade(self) -> float:
        """Average trade PnL (expectancy per trade)."""
        return float(np.mean(self._pnls)) if len(self._pnls) > 0 else 0.0

    @property
    def expectancy(self) -> float:
        """Mathematical expectancy: (win_rate * avg_win) + ((1 - win_rate) * avg_loss)."""
        return self.win_rate * self.avg_win + (1 - self.win_rate) * self.avg_loss

    @property
    def largest_win(self) -> float:
        """Largest single winning trade."""
        winners = self._pnls[self._pnls > 0]
        return float(np.max(winners)) if len(winners) > 0 else 0.0

    @property
    def largest_loss(self) -> float:
        """Largest single losing trade (most negative)."""
        losers = self._pnls[self._pnls < 0]
        return float(np.min(losers)) if len(losers) > 0 else 0.0

    @property
    def avg_return(self) -> float:
        """Average return per trade (as decimal)."""
        return float(np.mean(self._returns)) if len(self._returns) > 0 else 0.0

    @property
    def avg_bars_held(self) -> float:
        """Average number of bars positions were held."""
        if not self.trades:
            return 0.0
        bars = [t.bars_held for t in self.trades if hasattr(t, "bars_held")]
        return float(np.mean(bars)) if bars else 0.0

    @property
    def total_commission(self) -> float:
        """Total commission paid across all trades."""
        return sum(t.commission for t in self.trades)

    @property
    def total_slippage(self) -> float:
        """Total slippage cost across all trades."""
        return sum(t.slippage for t in self.trades)

    def by_side(self, side: str) -> "TradeAnalyzer":
        """Filter trades by side ('long' or 'short')."""
        filtered = [t for t in self.trades if t.side == side]
        return TradeAnalyzer(filtered)

    def by_asset(self, asset: str) -> "TradeAnalyzer":
        """Filter trades by asset."""
        filtered = [t for t in self.trades if t.asset == asset]
        return TradeAnalyzer(filtered)

    def to_dict(self) -> dict:
        """Export statistics as dictionary."""
        return {
            "num_trades": self.num_trades,
            "num_winners": self.num_winners,
            "num_losers": self.num_losers,
            "win_rate": self.win_rate,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "net_profit": self.net_profit,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "avg_trade": self.avg_trade,
            "expectancy": self.expectancy,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "avg_return": self.avg_return,
            "avg_bars_held": self.avg_bars_held,
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
        }
