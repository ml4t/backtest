"""Backtest results export and analysis.

This module provides a unified interface for accessing and exporting backtest results.
"""

from pathlib import Path
from typing import Optional

import polars as pl

from ml4t.backtest.execution.trade_tracker import TradeTracker
from ml4t.backtest.portfolio.analytics import PerformanceAnalyzer


class BacktestResults:
    """
    Unified interface for backtest results export and analysis.

    Provides access to:
    - Completed trades with entry/exit signals and reasons
    - Portfolio returns at various frequencies (daily, weekly, monthly)
    - Raw equity curve (event-based)

    Design philosophy:
    - Raw data only (no metrics like Sharpe, drawdown)
    - User-friendly export methods
    - Flexible frequency resampling
    """

    def __init__(
        self,
        trade_tracker: TradeTracker,
        performance_analyzer: Optional[PerformanceAnalyzer] = None,
    ):
        """Initialize results container.

        Args:
            trade_tracker: TradeTracker with completed trades
            performance_analyzer: Optional PerformanceAnalyzer for equity curve
        """
        self.trade_tracker = trade_tracker
        self.performance_analyzer = performance_analyzer

    def get_trades(self, include_metadata: bool = True) -> pl.DataFrame:
        """Get completed trades with entry/exit details.

        Args:
            include_metadata: If True, include metadata columns with entry/exit signals

        Returns:
            DataFrame with columns:
                - trade_id, asset_id
                - entry_dt, entry_price, entry_quantity, entry_commission, entry_slippage, entry_order_id
                - exit_dt, exit_price, exit_quantity, exit_commission, exit_slippage, exit_order_id
                - pnl, return_pct, duration_bars, direction
                - entry_metadata, exit_metadata (if include_metadata=True)

        Example:
            >>> results.get_trades()
            shape: (100, 20)
            ┌──────────┬──────────┬─────────────┬─────────────┬───┬────────────┬──────────┬───────────────┐
            │ trade_id ┆ asset_id ┆ entry_dt    ┆ entry_price ┆ … ┆ return_pct ┆ direction┆ entry_metadata│
            │ ---      ┆ ---      ┆ ---         ┆ ---         ┆   ┆ ---        ┆ ---      ┆ ---           │
            │ i64      ┆ str      ┆ datetime    ┆ f64         ┆   ┆ f64        ┆ str      ┆ object        │
            ╞══════════╪══════════╪═════════════╪═════════════╪═══╪════════════╪══════════╪═══════════════╡
            │ 0        ┆ AAPL     ┆ 2024-01-01  ┆ 150.00      ┆ … ┆ 2.5        ┆ long     ┆ {"reason":... │
            └──────────┴──────────┴─────────────┴─────────────┴───┴────────────┴──────────┴───────────────┘
        """
        trades_df = self.trade_tracker.get_trades_df()

        if trades_df.is_empty():
            return trades_df

        if not include_metadata:
            # Drop metadata columns if they exist
            cols_to_keep = [c for c in trades_df.columns if c != "metadata"]
            return trades_df.select(cols_to_keep)

        # Extract entry and exit metadata as separate columns
        # Note: metadata is stored as {"entry": {...}, "exit": {...}}
        # We'll keep the raw dict for now - users can parse as needed
        return trades_df

    def get_returns(self, frequency: str = "daily") -> pl.DataFrame:
        """Get portfolio returns at specified frequency.

        Args:
            frequency: Resampling frequency ("daily", "weekly", "monthly", "event")
                - "event": Raw equity curve (one row per fill/update)
                - "daily": End-of-day equity and daily returns
                - "weekly": End-of-week equity and weekly returns
                - "monthly": End-of-month equity and monthly returns

        Returns:
            DataFrame with columns: date, equity, returns

        Example:
            >>> results.get_returns("daily")
            shape: (252, 3)
            ┌────────────┬────────────┬──────────┐
            │ date       ┆ equity     ┆ returns  │
            │ ---        ┆ ---        ┆ ---      │
            │ date       ┆ f64        ┆ f64      │
            ╞════════════╪════════════╪══════════╡
            │ 2024-01-01 ┆ 100000.00  ┆ 0.0000   │
            │ 2024-01-02 ┆ 101250.00  ┆ 0.0125   │
            │ 2024-01-03 ┆ 100850.00  ┆ -0.0040  │
            └────────────┴────────────┴──────────┘

        Raises:
            ValueError: If performance_analyzer is None (analytics disabled)
        """
        if self.performance_analyzer is None:
            raise ValueError(
                "Performance analytics not available. "
                "Enable analytics when creating Portfolio: Portfolio(track_analytics=True)"
            )

        return self.performance_analyzer.get_returns(frequency)

    def export_trades(self, output_path: Path | str, include_metadata: bool = True) -> Path:
        """Export trades to Parquet file.

        Args:
            output_path: Path to output file (.parquet extension recommended)
            include_metadata: If True, include entry/exit metadata

        Returns:
            Path to exported file

        Example:
            >>> results.export_trades("results/trades.parquet")
            PosixPath('results/trades.parquet')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        trades_df = self.get_trades(include_metadata=include_metadata)
        trades_df.write_parquet(output_path)

        return output_path

    def export_returns(
        self,
        output_path: Path | str,
        frequency: str = "daily",
    ) -> Path:
        """Export returns to Parquet file.

        Args:
            output_path: Path to output file (.parquet extension recommended)
            frequency: Resampling frequency ("daily", "weekly", "monthly", "event")

        Returns:
            Path to exported file

        Example:
            >>> results.export_returns("results/returns_daily.parquet", frequency="daily")
            PosixPath('results/returns_daily.parquet')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        returns_df = self.get_returns(frequency)
        returns_df.write_parquet(output_path)

        return output_path

    def export_all(self, output_dir: Path | str) -> dict[str, Path]:
        """Export all results to directory.

        Creates:
        - trades.parquet: All completed trades with metadata
        - returns_daily.parquet: Daily returns
        - returns_event.parquet: Raw event-based equity curve

        Args:
            output_dir: Directory to save files

        Returns:
            Dictionary mapping file type to path

        Example:
            >>> paths = results.export_all("results/")
            >>> paths
            {
                'trades': PosixPath('results/trades.parquet'),
                'returns_daily': PosixPath('results/returns_daily.parquet'),
                'returns_event': PosixPath('results/returns_event.parquet'),
            }
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        # Export trades
        exported_files["trades"] = self.export_trades(output_dir / "trades.parquet")

        # Export returns (if analytics available)
        if self.performance_analyzer is not None:
            exported_files["returns_daily"] = self.export_returns(
                output_dir / "returns_daily.parquet", frequency="daily"
            )
            exported_files["returns_event"] = self.export_returns(
                output_dir / "returns_event.parquet", frequency="event"
            )

        return exported_files

    def summary(self) -> dict:
        """Get high-level summary statistics.

        Returns:
            Dictionary with:
                - num_trades: Total completed trades
                - num_open_positions: Current open positions
                - (if analytics enabled) final_equity, total_return

        Example:
            >>> results.summary()
            {
                'num_trades': 100,
                'num_open_positions': 5,
                'final_equity': 125430.50,
                'total_return': 0.2543,
            }
        """
        summary = {
            "num_trades": self.trade_tracker.get_trade_count(),
            "num_open_positions": self.trade_tracker.get_open_position_count(),
        }

        if self.performance_analyzer is not None:
            summary["final_equity"] = self.performance_analyzer.tracker.equity
            summary["total_return"] = self.performance_analyzer.tracker.returns

        return summary
