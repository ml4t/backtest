"""Trade analysis and statistics."""

from collections.abc import Sequence
from dataclasses import dataclass, field
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

    # MFE/MAE Analysis Methods

    @property
    def avg_mfe(self) -> float:
        """Average maximum favorable excursion across trades."""
        if not self.trades:
            return 0.0
        mfes = [t.max_favorable_excursion for t in self.trades]
        return float(np.mean(mfes))

    @property
    def avg_mae(self) -> float:
        """Average maximum adverse excursion across trades."""
        if not self.trades:
            return 0.0
        maes = [t.max_adverse_excursion for t in self.trades]
        return float(np.mean(maes))

    @property
    def mfe_capture_ratio(self) -> float:
        """Average ratio of realized return to MFE.

        Values close to 1.0 indicate exits near peak profit.
        Values close to 0.0 indicate exits gave back most gains.
        """
        if not self.trades:
            return 0.0
        ratios = []
        for t in self.trades:
            if t.max_favorable_excursion > 0:
                ratios.append(t.pnl_percent / t.max_favorable_excursion)
        return float(np.mean(ratios)) if ratios else 0.0

    @property
    def mae_recovery_ratio(self) -> float:
        """Average ratio showing how much of MAE was recovered.

        Calculated as (MAE - final_loss) / MAE for losing trades.
        Higher values indicate better recovery from drawdowns.
        """
        if not self.trades:
            return 0.0
        ratios = []
        for t in self.trades:
            if t.max_adverse_excursion < 0 and t.pnl_percent < 0:
                # Both negative: MAE was -10%, final was -5% = recovered 50%
                recovery = (t.max_adverse_excursion - t.pnl_percent) / abs(t.max_adverse_excursion)
                ratios.append(recovery)
        return float(np.mean(ratios)) if ratios else 0.0

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
            # MFE/MAE metrics
            "avg_mfe": self.avg_mfe,
            "avg_mae": self.avg_mae,
            "mfe_capture_ratio": self.mfe_capture_ratio,
            "mae_recovery_ratio": self.mae_recovery_ratio,
        }


@dataclass
class MAEMFEAnalyzer:
    """Analyze MAE/MFE distributions for optimal stop/target discovery.

    Provides statistical analysis of Maximum Adverse Excursion (MAE) and
    Maximum Favorable Excursion (MFE) to help optimize exit strategies.

    Attributes:
        trades: Sequence of Trade objects with MAE/MFE data
        _maes: Array of MAE values (negative for losses)
        _mfes: Array of MFE values (positive for gains)

    Example:
        analyzer = MAEMFEAnalyzer(trades)

        # Find stop that would preserve 90% of winning trades
        optimal_stop = analyzer.suggest_stop_loss(percentile=90)

        # Find target that captures 75% of MFE
        optimal_target = analyzer.suggest_take_profit(percentile=75)

        # Get edge ratio
        edge = analyzer.edge_ratio

    Note:
        MAE values are typically negative (adverse = loss)
        MFE values are typically positive (favorable = gain)
    """

    trades: Sequence["Trade"]
    _maes: np.ndarray = field(init=False, repr=False)
    _mfes: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if self.trades:
            self._maes = np.array([t.max_adverse_excursion for t in self.trades])
            self._mfes = np.array([t.max_favorable_excursion for t in self.trades])
        else:
            self._maes = np.array([])
            self._mfes = np.array([])

    @property
    def num_trades(self) -> int:
        """Number of trades analyzed."""
        return len(self.trades)

    # --- MAE Statistics ---

    @property
    def mae_mean(self) -> float:
        """Mean MAE across all trades (typically negative)."""
        return float(np.mean(self._maes)) if len(self._maes) > 0 else 0.0

    @property
    def mae_median(self) -> float:
        """Median MAE across all trades."""
        return float(np.median(self._maes)) if len(self._maes) > 0 else 0.0

    @property
    def mae_std(self) -> float:
        """Standard deviation of MAE."""
        return float(np.std(self._maes)) if len(self._maes) > 0 else 0.0

    def mae_percentile(self, q: float) -> float:
        """MAE at given percentile (0-100).

        Args:
            q: Percentile (e.g., 10 = worst 10% of trades)

        Returns:
            MAE value at that percentile (negative)
        """
        if len(self._maes) == 0:
            return 0.0
        return float(np.percentile(self._maes, q))

    # --- MFE Statistics ---

    @property
    def mfe_mean(self) -> float:
        """Mean MFE across all trades (typically positive)."""
        return float(np.mean(self._mfes)) if len(self._mfes) > 0 else 0.0

    @property
    def mfe_median(self) -> float:
        """Median MFE across all trades."""
        return float(np.median(self._mfes)) if len(self._mfes) > 0 else 0.0

    @property
    def mfe_std(self) -> float:
        """Standard deviation of MFE."""
        return float(np.std(self._mfes)) if len(self._mfes) > 0 else 0.0

    def mfe_percentile(self, q: float) -> float:
        """MFE at given percentile (0-100).

        Args:
            q: Percentile (e.g., 75 = captured by 75% of trades)

        Returns:
            MFE value at that percentile
        """
        if len(self._mfes) == 0:
            return 0.0
        return float(np.percentile(self._mfes, q))

    # --- Combined Metrics ---

    @property
    def edge_ratio(self) -> float:
        """Ratio of average MFE to average |MAE|.

        Values > 1.0 indicate trades tend to go further in favor
        than against. Higher is better.
        """
        if self.mae_mean == 0:
            return float("inf") if self.mfe_mean > 0 else 0.0
        return self.mfe_mean / abs(self.mae_mean)

    @property
    def efficiency(self) -> float:
        """Average trade efficiency: realized_return / MFE.

        Values close to 1.0 indicate exits near peak profit.
        """
        if len(self.trades) == 0:
            return 0.0
        efficiencies = []
        for t in self.trades:
            if t.max_favorable_excursion > 0:
                efficiencies.append(t.pnl_percent / t.max_favorable_excursion)
        return float(np.mean(efficiencies)) if efficiencies else 0.0

    # --- Optimization Suggestions ---

    def suggest_stop_loss(self, percentile: float = 90) -> float:
        """Suggest stop loss level that preserves given % of winning trades.

        Args:
            percentile: Percentage of winning trades to preserve (default 90)

        Returns:
            Suggested stop loss as return percentage (e.g., -0.05 = -5%)

        Example:
            # Stop that would save 90% of winners
            stop = analyzer.suggest_stop_loss(percentile=90)
            # Returns e.g., -0.03 meaning -3% stop
        """
        if len(self.trades) == 0:
            return 0.0

        # Get MAE for winning trades only
        winner_maes = np.array([t.max_adverse_excursion for t in self.trades if t.pnl > 0])
        if len(winner_maes) == 0:
            return self.mae_percentile(100 - percentile)

        # Find MAE that preserves `percentile`% of winners
        # Lower percentile = tighter stop (fewer trades hit it)
        return float(np.percentile(winner_maes, 100 - percentile))

    def suggest_take_profit(self, percentile: float = 75) -> float:
        """Suggest take profit level based on MFE distribution.

        Args:
            percentile: Percentage of trades that reach this level (default 75)

        Returns:
            Suggested take profit as return percentage (e.g., 0.05 = 5%)

        Example:
            # Target that 75% of trades reach
            target = analyzer.suggest_take_profit(percentile=75)
            # Returns e.g., 0.05 meaning +5% target
        """
        if len(self._mfes) == 0:
            return 0.0

        # Find MFE that `percentile`% of trades achieve
        # Lower percentile = more conservative target
        return float(np.percentile(self._mfes, 100 - percentile))

    def optimal_exit_levels(
        self,
        stop_percentile: float = 90,
        target_percentile: float = 75,
    ) -> dict[str, float]:
        """Get optimized stop loss and take profit levels.

        Args:
            stop_percentile: % of winning trades to preserve (default 90)
            target_percentile: % of trades that reach target (default 75)

        Returns:
            Dictionary with 'stop_loss' and 'take_profit' levels
        """
        return {
            "stop_loss": self.suggest_stop_loss(stop_percentile),
            "take_profit": self.suggest_take_profit(target_percentile),
            "risk_reward": abs(
                self.suggest_take_profit(target_percentile)
                / self.suggest_stop_loss(stop_percentile)
            )
            if self.suggest_stop_loss(stop_percentile) != 0
            else float("inf"),
        }

    def distribution_data(self) -> dict[str, list[float]]:
        """Get MAE/MFE distribution data for visualization.

        Returns:
            Dictionary with 'mae' and 'mfe' lists for plotting
        """
        return {
            "mae": self._maes.tolist() if len(self._maes) > 0 else [],
            "mfe": self._mfes.tolist() if len(self._mfes) > 0 else [],
        }

    def to_dict(self) -> dict:
        """Export analysis results as dictionary."""
        return {
            "num_trades": self.num_trades,
            # MAE stats
            "mae_mean": self.mae_mean,
            "mae_median": self.mae_median,
            "mae_std": self.mae_std,
            "mae_p10": self.mae_percentile(10),
            "mae_p25": self.mae_percentile(25),
            "mae_p50": self.mae_percentile(50),
            "mae_p75": self.mae_percentile(75),
            "mae_p90": self.mae_percentile(90),
            # MFE stats
            "mfe_mean": self.mfe_mean,
            "mfe_median": self.mfe_median,
            "mfe_std": self.mfe_std,
            "mfe_p10": self.mfe_percentile(10),
            "mfe_p25": self.mfe_percentile(25),
            "mfe_p50": self.mfe_percentile(50),
            "mfe_p75": self.mfe_percentile(75),
            "mfe_p90": self.mfe_percentile(90),
            # Combined
            "edge_ratio": self.edge_ratio,
            "efficiency": self.efficiency,
            # Suggestions
            "suggested_stop_loss": self.suggest_stop_loss(),
            "suggested_take_profit": self.suggest_take_profit(),
        }
