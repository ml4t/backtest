"""Structured backtest result with export capabilities.

This module provides a BacktestResult class that wraps the raw output from
Engine.run() with convenient DataFrame export methods and Parquet serialization.

Example:
    >>> from ml4t.backtest import Engine, DataFeed, Strategy
    >>> engine = Engine(feed, strategy)
    >>> result = engine.run()
    >>>
    >>> # Export trades to Parquet
    >>> result.to_parquet("./results/my_backtest")
    >>>
    >>> # Get DataFrames
    >>> trades_df = result.to_trades_dataframe()
    >>> equity_df = result.to_equity_dataframe()
    >>>
    >>> # Integration with ml4t.diagnostic
    >>> trade_records = result.to_trade_records()
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from .types import Fill, Trade

if TYPE_CHECKING:
    from .analytics import EquityCurve, TradeAnalyzer
    from .config import BacktestConfig


# Annualization factors for common trading calendars
_ANNUALIZATION_FACTORS: dict[str, int] = {
    "crypto": 365,  # 24/7 trading
    "NYSE": 252,
    "NASDAQ": 252,
    "CME_Equity": 252,
    "CME_Agriculture": 252,
    "CME_Globex_Energy_and_Metals": 252,
    "LSE": 253,
    "XETRA": 252,
    "TSX": 252,
    "HKEX": 252,
    "JPX": 245,
}


def _get_annualization_factor(calendar: str | None) -> int:
    """Get annualization factor for a trading calendar.

    Args:
        calendar: Trading calendar name. If None, defaults to 252.

    Returns:
        Number of trading days per year for the calendar.
    """
    if calendar is None:
        return 252  # Default for equities

    # Check known calendars (case-insensitive)
    cal_upper = calendar.upper()
    for name, factor in _ANNUALIZATION_FACTORS.items():
        if name.upper() == cal_upper:
            return factor

    # Try pandas_market_calendars for unknown calendars
    try:
        from pandas_market_calendars import get_calendar

        cal = get_calendar(calendar)
        # Estimate from typical year
        schedule = cal.schedule("2024-01-01", "2024-12-31")
        return len(schedule)
    except Exception:
        # Default fallback
        return 252


@dataclass
class BacktestResult:
    """Structured backtest result with export capabilities.

    This class wraps the raw output from Engine.run() and provides:
    - DataFrame conversion methods (trades, equity, daily P&L)
    - Parquet export/import for persistence
    - Integration with ml4t.diagnostic library
    - Backward-compatible dict export

    Attributes:
        trades: List of completed Trade objects
        equity_curve: List of (timestamp, portfolio_value) tuples
        fills: List of Fill objects (all order fills)
        metrics: Dictionary of computed performance metrics
        config: BacktestConfig used for the backtest (optional)
        equity: EquityCurve analytics object
        trade_analyzer: TradeAnalyzer analytics object
    """

    trades: list[Trade]
    equity_curve: list[tuple[datetime, float]]
    fills: list[Fill]
    metrics: dict[str, Any]
    config: BacktestConfig | None = None
    equity: EquityCurve | None = None
    trade_analyzer: TradeAnalyzer | None = None

    # Cached DataFrames (computed on demand)
    _trades_df: pl.DataFrame | None = field(default=None, repr=False)
    _equity_df: pl.DataFrame | None = field(default=None, repr=False)

    def to_trades_dataframe(self) -> pl.DataFrame:
        """Convert trades to Polars DataFrame.

        Returns DataFrame with columns:
            asset, entry_time, exit_time, entry_price, exit_price,
            quantity, direction, pnl, pnl_percent, bars_held,
            commission, slippage, mfe, mae, exit_reason, status

        The status column indicates "closed" (actually exited) or "open"
        (mark-to-market at end of backtest).

        Returns:
            Polars DataFrame with one row per trade
        """
        if self._trades_df is not None:
            return self._trades_df

        if not self.trades:
            return pl.DataFrame(schema=self._trades_schema())

        records = []
        for t in self.trades:
            records.append(
                {
                    "symbol": t.symbol,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "direction": t.direction,
                    "pnl": t.pnl,
                    "pnl_percent": t.pnl_percent,
                    "bars_held": t.bars_held,
                    "fees": t.fees,
                    "slippage": t.slippage,
                    "mfe": t.mfe,
                    "mae": t.mae,
                    "exit_reason": t.exit_reason,
                    "status": t.status,
                }
            )

        self._trades_df = pl.DataFrame(records, schema=self._trades_schema())
        return self._trades_df

    def to_equity_dataframe(self) -> pl.DataFrame:
        """Convert equity curve to Polars DataFrame.

        Returns DataFrame with columns:
            timestamp, equity, return, cumulative_return,
            drawdown, high_water_mark

        Returns:
            Polars DataFrame with one row per bar, sorted by timestamp
        """
        if self._equity_df is not None:
            return self._equity_df

        if not self.equity_curve:
            return pl.DataFrame(schema=self._equity_schema())

        timestamps = [ts for ts, _ in self.equity_curve]
        values = [float(v) for _, v in self.equity_curve]

        # Build base DataFrame and sort by timestamp
        df = pl.DataFrame({"timestamp": timestamps, "equity": values}).sort("timestamp")

        # Vectorized computation using Polars
        df = df.with_columns(
            [
                # Returns: percent change, first bar has no return
                pl.col("equity").pct_change().fill_null(0.0).alias("return"),
                # Cumulative return from initial equity
                (pl.col("equity") / pl.first("equity") - 1.0).alias("cumulative_return"),
                # High water mark (running maximum)
                pl.col("equity").cum_max().alias("high_water_mark"),
            ]
        ).with_columns(
            # Drawdown: (equity / hwm) - 1, handle division by zero
            pl.when(pl.col("high_water_mark") > 0)
            .then(pl.col("equity") / pl.col("high_water_mark") - 1.0)
            .otherwise(0.0)
            .alias("drawdown")
        )

        # Reorder columns to match expected schema
        self._equity_df = df.select(
            ["timestamp", "equity", "return", "cumulative_return", "drawdown", "high_water_mark"]
        )

        return self._equity_df

    def to_daily_pnl(self, session_aligned: bool = False) -> pl.DataFrame:
        """Get daily P&L DataFrame.

        Args:
            session_aligned: If True and session config is available,
                align P&L to trading sessions (e.g., CME 5pm-4pm CT).
                If False, use calendar day boundaries.

        Returns:
            DataFrame with columns:
                date, open_equity, close_equity, high_equity, low_equity,
                pnl, return_pct, cumulative_return, num_bars
        """
        if not self.equity_curve:
            return pl.DataFrame(
                schema={
                    "date": pl.Date,
                    "open_equity": pl.Float64,
                    "close_equity": pl.Float64,
                    "high_equity": pl.Float64,
                    "low_equity": pl.Float64,
                    "pnl": pl.Float64,
                    "return_pct": pl.Float64,
                    "cumulative_return": pl.Float64,
                    "num_bars": pl.Int32,
                }
            )

        # Build equity DataFrame
        equity_df = self.to_equity_dataframe()

        if session_aligned and self.config and self.config.calendar:
            # Use session alignment
            from .sessions import SessionConfig, compute_session_pnl

            session_config = SessionConfig(
                calendar=self.config.calendar,
                timezone=self.config.timezone,
                session_start_time=getattr(self.config, "session_start_time", None),
            )
            return compute_session_pnl(self.equity_curve, session_config)

        # Default: calendar day aggregation
        daily = (
            equity_df.with_columns(pl.col("timestamp").dt.date().alias("date"))
            .group_by("date")
            .agg(
                [
                    pl.col("equity").first().alias("open_equity"),
                    pl.col("equity").last().alias("close_equity"),
                    pl.col("equity").max().alias("high_equity"),
                    pl.col("equity").min().alias("low_equity"),
                    pl.len().alias("num_bars"),
                ]
            )
            .sort("date")
        )

        # Compute daily P&L and returns
        daily = daily.with_columns(
            [
                (pl.col("close_equity") - pl.col("open_equity")).alias("pnl"),
            ]
        )

        # Return percent (handle first day)
        prev_close = daily.select(pl.col("close_equity").shift(1)).to_series()
        return_pct = (daily["close_equity"] - prev_close) / prev_close
        return_pct = return_pct.fill_null(0.0)

        # Cumulative return from first open
        initial = daily["open_equity"][0] if len(daily) > 0 else 1.0
        cum_return = (daily["close_equity"] / initial) - 1.0

        daily = daily.with_columns(
            [
                return_pct.alias("return_pct"),
                cum_return.alias("cumulative_return"),
            ]
        )

        return daily

    def to_daily_returns(
        self,
        calendar: str | None = None,
        session_aligned: bool | None = None,
    ) -> pl.Series:
        """Get daily returns as Polars Series for ml4t-diagnostic integration.

        This method properly aggregates bar-level equity to daily returns,
        which is the correct input for computing risk metrics like Sharpe ratio.
        For intraday data, using bar-level returns would give incorrect results.

        Args:
            calendar: Trading calendar for context. If provided and known,
                enables session-aware aggregation. Common values:
                - "crypto": 365 days/year (24/7)
                - "NYSE", "NASDAQ": 252 days/year
                - "CME_Equity", etc: Uses pandas_market_calendars
                If None, uses config calendar or defaults to calendar day boundaries.
            session_aligned: If True, align to trading sessions (e.g., CME 5pm CT).
                If None, auto-detect from calendar (True for CME, False for crypto).
                If False, use calendar day boundaries.

        Returns:
            Series of daily returns (percentage, e.g., 0.01 = 1%)

        Example:
            >>> result = engine.run()
            >>> daily_returns = result.to_daily_returns(calendar="NYSE")
            >>> # Use with ml4t-diagnostic
            >>> from ml4t.diagnostic import sharpe_ratio
            >>> sharpe = sharpe_ratio(daily_returns.to_numpy(), annualization_factor=252)
        """
        # Determine session alignment
        if session_aligned is None:
            # Auto-detect: CME calendars typically need session alignment
            cal = calendar or (self.config.calendar if self.config else None)
            session_aligned = cal is not None and "CME" in str(cal).upper()

        # Get daily P&L DataFrame
        daily_df = self.to_daily_pnl(session_aligned=session_aligned)

        if daily_df.is_empty():
            return pl.Series("daily_return", [], dtype=pl.Float64)

        # Return the return_pct column as a Series
        return daily_df["return_pct"].alias("daily_return")

    def to_returns_series(self) -> pl.Series:
        """Get period returns as Polars Series.

        Note: This returns BAR-LEVEL returns, not daily returns.
        For risk metrics like Sharpe ratio, use to_daily_returns() instead.

        Returns:
            Series of period returns (one per bar)
        """
        equity_df = self.to_equity_dataframe()
        return equity_df["return"]

    def to_trade_records(self) -> list[dict[str, Any]]:
        """Convert trades to ml4t.diagnostic TradeRecord format.

        Returns list of dictionaries matching the TradeRecord schema
        from ml4t.diagnostic.integration.

        Returns:
            List of trade record dictionaries
        """
        from .analytics.bridge import to_trade_records

        return to_trade_records(self.trades)

    def compute_metrics(
        self,
        calendar: str | None = None,
        confidence_intervals: bool = False,
    ) -> dict[str, Any]:
        """Compute performance metrics via ml4t-diagnostic.

        This method provides properly computed risk-adjusted metrics using
        daily returns. For intraday backtests, this is critical to get
        correct Sharpe/Sortino ratios (bar-level returns give wrong values).

        Args:
            calendar: Trading calendar for annualization:
                - "crypto": 365 days/year (24/7)
                - "NYSE", "NASDAQ": 252 days/year
                - Any pandas_market_calendars calendar
                If None, uses config calendar or defaults to 252.
            confidence_intervals: If True, include bootstrap CI for Sharpe
                (requires ml4t-diagnostic with bootstrap support)

        Returns:
            Dictionary with metrics:
                - sharpe_ratio: Annualized Sharpe ratio
                - sortino_ratio: Annualized Sortino ratio
                - calmar_ratio: CAGR / Max Drawdown
                - max_drawdown: Maximum drawdown (negative)
                - cagr: Compound annual growth rate
                - total_return: Total return
                - num_trades: Number of trades
                - win_rate: Percentage of winning trades
                - profit_factor: Gross profits / Gross losses
                - expectancy: Average expected P&L per trade
                Plus trade statistics from TradeAnalyzer

        Raises:
            ImportError: If ml4t-diagnostic is not installed

        Example:
            >>> result = engine.run()
            >>> metrics = result.compute_metrics(calendar="crypto")
            >>> print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
        """
        # Import from ml4t-diagnostic (optional dependency)
        # Using dynamic import to avoid type checker issues with optional deps
        import importlib
        from collections.abc import Callable

        try:
            diagnostic = importlib.import_module("ml4t.diagnostic")
            sharpe_ratio: Callable[..., float] = diagnostic.sharpe_ratio
            sortino_ratio: Callable[..., float] = diagnostic.sortino_ratio
        except (ImportError, AttributeError) as e:
            raise ImportError(
                "ml4t-diagnostic is required for compute_metrics(). "
                "Install with: pip install ml4t-diagnostic"
            ) from e

        # Get calendar and annualization factor
        cal = calendar or (self.config.calendar if self.config else None)
        annualization_factor = _get_annualization_factor(cal)

        # Get daily returns
        daily_returns = self.to_daily_returns(calendar=cal)
        returns_array = daily_returns.to_numpy()

        # Compute metrics via diagnostic library
        metrics: dict[str, Any] = {}

        # Risk-adjusted returns
        if len(returns_array) > 0:
            metrics["sharpe_ratio"] = sharpe_ratio(
                returns_array, annualization_factor=annualization_factor
            )
            metrics["sortino_ratio"] = sortino_ratio(
                returns_array, annualization_factor=annualization_factor
            )
        else:
            metrics["sharpe_ratio"] = 0.0
            metrics["sortino_ratio"] = 0.0

        # Drawdown metrics (from equity curve)
        equity_df = self.to_equity_dataframe()
        if len(equity_df) > 0:
            # Get max drawdown as float
            dd_values = equity_df["drawdown"].to_list()
            max_dd: float = min(dd_values) if dd_values else 0.0
            metrics["max_drawdown"] = max_dd

            # CAGR from total return and days
            equity_values = equity_df["equity"].to_list()
            first_val: float = equity_values[0] if equity_values else 1.0
            last_val: float = equity_values[-1] if equity_values else 1.0
            total_return: float = (last_val / first_val) - 1.0
            metrics["total_return"] = total_return

            # Days in backtest
            timestamps = equity_df["timestamp"].to_list()
            first_ts = timestamps[0]
            last_ts = timestamps[-1]
            days: float = (last_ts - first_ts).total_seconds() / 86400
            years: float = days / 365.25
            if years > 0:
                cagr: float = (1 + total_return) ** (1 / years) - 1
                metrics["cagr"] = cagr
                if max_dd != 0:
                    metrics["calmar_ratio"] = cagr / abs(max_dd)
                else:
                    metrics["calmar_ratio"] = 0.0
            else:
                metrics["cagr"] = 0.0
                metrics["calmar_ratio"] = 0.0
        else:
            metrics["max_drawdown"] = 0.0
            metrics["total_return"] = 0.0
            metrics["cagr"] = 0.0
            metrics["calmar_ratio"] = 0.0

        # Trade statistics
        if self.trade_analyzer is not None:
            ta = self.trade_analyzer
            metrics["num_trades"] = ta.num_trades
            metrics["win_rate"] = ta.win_rate
            metrics["profit_factor"] = ta.profit_factor
            metrics["expectancy"] = ta.expectancy
            metrics["avg_trade"] = ta.avg_trade
            metrics["avg_winner"] = ta.avg_win
            metrics["avg_loser"] = ta.avg_loss
            metrics["total_fees"] = ta.total_fees
        elif self.trades:
            # Compute basic stats
            wins = [t for t in self.trades if t.pnl > 0]
            losses = [t for t in self.trades if t.pnl < 0]
            metrics["num_trades"] = len(self.trades)
            metrics["win_rate"] = len(wins) / len(self.trades) if self.trades else 0.0
            total_wins = sum(t.pnl for t in wins)
            total_losses = abs(sum(t.pnl for t in losses))
            metrics["profit_factor"] = total_wins / total_losses if total_losses > 0 else 0.0
            metrics["avg_trade"] = sum(t.pnl for t in self.trades) / len(self.trades)
            metrics["expectancy"] = metrics["avg_trade"]
        else:
            metrics["num_trades"] = 0
            metrics["win_rate"] = 0.0
            metrics["profit_factor"] = 0.0
            metrics["expectancy"] = 0.0
            metrics["avg_trade"] = 0.0

        return metrics

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary (backward compatible with Engine.run()).

        Returns:
            Dictionary with all metrics and raw data
        """
        result = dict(self.metrics)
        result.update(
            {
                "trades": self.trades,
                "equity_curve": self.equity_curve,
                "fills": self.fills,
            }
        )
        if self.equity is not None:
            result["equity"] = self.equity
        if self.trade_analyzer is not None:
            result["trade_analyzer"] = self.trade_analyzer
        return result

    def to_parquet(
        self,
        path: str | Path,
        include: list[str] | None = None,
        compression: Literal["lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"] = "zstd",
    ) -> dict[str, Path]:
        """Export backtest result to Parquet files.

        Creates directory structure:
            {path}/
                trades.parquet
                equity.parquet
                daily_pnl.parquet
                metrics.json
                config.yaml (if config available)

        Args:
            path: Directory path to write files
            include: Components to include. Default: all.
                Options: ["trades", "equity", "daily_pnl", "metrics", "config"]
            compression: Parquet compression codec (default: "zstd")

        Returns:
            Dict mapping component names to file paths
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if include is None:
            include = ["trades", "equity", "daily_pnl", "metrics", "config"]

        written: dict[str, Path] = {}

        if "trades" in include:
            trades_path = path / "trades.parquet"
            self.to_trades_dataframe().write_parquet(trades_path, compression=compression)
            written["trades"] = trades_path

        if "equity" in include:
            equity_path = path / "equity.parquet"
            self.to_equity_dataframe().write_parquet(equity_path, compression=compression)
            written["equity"] = equity_path

        if "daily_pnl" in include:
            daily_path = path / "daily_pnl.parquet"
            self.to_daily_pnl().write_parquet(daily_path, compression=compression)
            written["daily_pnl"] = daily_path

        if "metrics" in include:
            metrics_path = path / "metrics.json"
            # Filter to JSON-serializable metrics
            serializable = {}
            for k, v in self.metrics.items():
                if isinstance(v, (int, float, str, bool, type(None))):
                    serializable[k] = v
                elif isinstance(v, datetime):
                    serializable[k] = v.isoformat()
                else:
                    # Handle numpy scalars (np.float64, np.int64, etc.)
                    try:
                        import numpy as np

                        if isinstance(v, np.generic):
                            serializable[k] = v.item()
                    except (ImportError, AttributeError):
                        pass  # Skip if numpy not available or not a numpy type
            with open(metrics_path, "w") as f:
                json.dump(serializable, f, indent=2)
            written["metrics"] = metrics_path

        if "config" in include and self.config is not None:
            config_path = path / "config.yaml"
            try:
                import yaml

                with open(config_path, "w") as f:
                    yaml.dump(self.config.to_dict(), f, default_flow_style=False)
                written["config"] = config_path
            except (ImportError, AttributeError):
                pass  # Skip if yaml not available or config has no to_dict

        return written

    @classmethod
    def from_parquet(cls, path: str | Path) -> BacktestResult:
        """Load backtest result from Parquet directory.

        Args:
            path: Directory containing Parquet files from to_parquet()

        Returns:
            BacktestResult instance
        """
        path = Path(path)

        # Load trades
        trades_path = path / "trades.parquet"
        trades: list[Trade] = []
        if trades_path.exists():
            trades_df = pl.read_parquet(trades_path)
            for row in trades_df.iter_rows(named=True):
                # Support both old (asset/commission) and new (symbol/fees) column names
                symbol = row.get("symbol") or row.get("asset", "")
                fees = row.get("fees") or row.get("commission", 0.0)
                trades.append(
                    Trade(
                        symbol=symbol,
                        entry_time=row["entry_time"],
                        exit_time=row["exit_time"],
                        entry_price=row["entry_price"],
                        exit_price=row["exit_price"],
                        quantity=row["quantity"],
                        pnl=row["pnl"],
                        pnl_percent=row["pnl_percent"],
                        bars_held=row["bars_held"],
                        fees=fees,
                        slippage=row["slippage"],
                        exit_reason=row.get("exit_reason", "signal"),
                        mfe=row["mfe"],
                        mae=row["mae"],
                    )
                )

        # Load equity curve
        equity_curve: list[tuple[datetime, float]] = []
        equity_path = path / "equity.parquet"
        if equity_path.exists():
            equity_df = pl.read_parquet(equity_path)
            for row in equity_df.iter_rows(named=True):
                equity_curve.append((row["timestamp"], row["equity"]))

        # Load metrics
        metrics: dict[str, Any] = {}
        metrics_path = path / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)

        # Load config if available
        config = None
        config_path = path / "config.yaml"
        if config_path.exists():
            try:
                import yaml

                from .config import BacktestConfig

                with open(config_path) as f:
                    config_data = yaml.safe_load(f)
                config = BacktestConfig.from_dict(config_data)
            except (ImportError, Exception):
                pass  # Skip if yaml not available or config invalid

        return cls(
            trades=trades,
            equity_curve=equity_curve,
            fills=[],  # Fills not persisted by default
            metrics=metrics,
            config=config,
        )

    @staticmethod
    def _trades_schema() -> dict[str, pl.DataType]:
        """Schema for trades DataFrame.

        This schema is part of the cross-library API specification, designed to
        produce identical Parquet output across Python, Numba, and Rust implementations.

        Schema Alignment (v0.1.0a6):
            - symbol: Asset identifier (was 'asset')
            - fees: Total transaction fees (was 'commission')
        """
        return {
            "symbol": pl.String(),
            "entry_time": pl.Datetime(),
            "exit_time": pl.Datetime(),
            "entry_price": pl.Float64(),
            "exit_price": pl.Float64(),
            "quantity": pl.Float64(),
            "direction": pl.String(),
            "pnl": pl.Float64(),
            "pnl_percent": pl.Float64(),
            "bars_held": pl.Int32(),
            "fees": pl.Float64(),
            "slippage": pl.Float64(),
            "mfe": pl.Float64(),
            "mae": pl.Float64(),
            "exit_reason": pl.String(),
            "status": pl.String(),  # "closed" or "open"
        }

    @staticmethod
    def _equity_schema() -> dict[str, pl.DataType]:
        """Schema for equity DataFrame."""
        return {
            "timestamp": pl.Datetime(),
            "equity": pl.Float64(),
            "return": pl.Float64(),
            "cumulative_return": pl.Float64(),
            "drawdown": pl.Float64(),
            "high_water_mark": pl.Float64(),
        }

    # --- Backward compatibility: dict-like access ---

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access for backward compatibility.

        .. deprecated:: 0.3.0
            Dict-style access (result["key"]) is deprecated and will be removed
            in a future version. Use direct attribute access instead:
            - result.trades instead of result["trades"]
            - result.metrics["sharpe"] instead of result["sharpe"]

        Example:
            result["sharpe"]  # Same as result.metrics["sharpe"]
            result["trades"]  # Same as result.trades
        """
        warnings.warn(
            "Dict-style access (result['key']) is deprecated. "
            "Use result.trades, result.metrics['sharpe'], etc. instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Special keys that map to attributes
        attr_map = {
            "trades": self.trades,
            "equity_curve": self.equity_curve,
            "fills": self.fills,
            "equity": self.equity,
            "trade_analyzer": self.trade_analyzer,
        }
        if key in attr_map:
            return attr_map[key]
        # Everything else from metrics
        return self.metrics[key]

    def __contains__(self, key: str) -> bool:
        """Support 'key in result' checks.

        .. deprecated:: 0.3.0
            Use hasattr() or check result.metrics directly.
        """
        warnings.warn(
            "'key in result' is deprecated. Use hasattr() or check result.metrics.",
            DeprecationWarning,
            stacklevel=2,
        )
        if key in ("trades", "equity_curve", "fills", "equity", "trade_analyzer"):
            return True
        return key in self.metrics

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get() for backward compatibility.

        .. deprecated:: 0.3.0
            Use direct attribute access or result.metrics.get().
        """
        warnings.warn(
            "result.get() is deprecated. Use result.metrics.get() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Direct lookup to avoid double deprecation warning from __getitem__
        attr_map = {
            "trades": self.trades,
            "equity_curve": self.equity_curve,
            "fills": self.fills,
            "equity": self.equity,
            "trade_analyzer": self.trade_analyzer,
        }
        if key in attr_map:
            return attr_map[key]
        return self.metrics.get(key, default)

    def to_tearsheet(
        self,
        template: Literal["quant_trader", "hedge_fund", "risk_manager", "full"] = "full",
        theme: Literal["default", "dark", "print", "presentation"] = "default",
        title: str | None = None,
        output_path: str | Path | None = None,
        include_statistical: bool = True,
        calendar: str | None = None,
    ) -> str:
        """Generate an interactive HTML tearsheet for the backtest results.

        This method integrates with ml4t.diagnostic to create comprehensive
        backtest visualizations including:
        - Executive summary with KPI cards and traffic lights
        - Trade analysis (MFE/MAE, exit reasons, waterfall)
        - Cost attribution (commission, slippage breakdown)
        - Statistical validity (DSR, confidence intervals, RAS)

        Parameters
        ----------
        template : {"quant_trader", "hedge_fund", "risk_manager", "full"}
            Report template persona:
            - "quant_trader": Trade-level focus (MFE/MAE, exit reasons)
            - "hedge_fund": Risk-adjusted focus (drawdowns, costs)
            - "risk_manager": Statistical focus (DSR, CI, MinTRL)
            - "full": All sections enabled
        theme : {"default", "dark", "print", "presentation"}
            Visual theme for the report
        title : str, optional
            Report title. Defaults to "Backtest Tearsheet"
        output_path : str or Path, optional
            If provided, saves HTML to this path
        include_statistical : bool
            Whether to include statistical validity analysis (DSR, RAS).
            Requires sufficient trades for meaningful statistics.
        calendar : str, optional
            Trading calendar for session alignment (e.g. "NYSE", "crypto").
            Passed to to_daily_returns() for correct daily aggregation.

        Returns
        -------
        str
            HTML content of the tearsheet

        Raises
        ------
        ImportError
            If ml4t-diagnostic is not installed

        Examples
        --------
        >>> result = engine.run()
        >>> html = result.to_tearsheet(template="quant_trader", theme="dark")
        >>> # Or save directly to file
        >>> result.to_tearsheet(output_path="backtest_report.html")
        """
        try:
            from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet
        except ImportError as e:
            raise ImportError(
                "ml4t-diagnostic is required for tearsheet generation. "
                "Install it with: pip install ml4t-diagnostic[viz]"
            ) from e

        # Extract data for tearsheet
        trades_df = self.to_trades_dataframe()
        # Use daily returns (not bar-level) for correct annualized metrics
        returns = self.to_daily_returns(calendar=calendar).to_numpy()

        # Build metrics dict with all available metrics
        tearsheet_metrics = dict(self.metrics)

        # Ensure common metrics are present
        if "n_trades" not in tearsheet_metrics:
            tearsheet_metrics["n_trades"] = len(self.trades)
        if "total_pnl" not in tearsheet_metrics and self.trades:
            tearsheet_metrics["total_pnl"] = sum(t.pnl for t in self.trades)
        if "win_rate" not in tearsheet_metrics and self.trades:
            winners = sum(1 for t in self.trades if t.pnl > 0)
            tearsheet_metrics["win_rate"] = winners / len(self.trades) if self.trades else 0
        if "total_commission" not in tearsheet_metrics and self.trades:
            tearsheet_metrics["total_commission"] = sum(t.fees for t in self.trades)
        if "total_slippage" not in tearsheet_metrics and self.trades:
            tearsheet_metrics["total_slippage"] = sum(t.slippage for t in self.trades)

        # Extract equity curve for portfolio-level charts
        equity_df = self.to_equity_dataframe() if self.equity_curve else None

        # Generate tearsheet
        html = generate_backtest_tearsheet(
            metrics=tearsheet_metrics,
            trades=trades_df if len(trades_df) > 0 else None,
            returns=returns if len(returns) > 0 else None,
            equity_curve=equity_df,
            template=template,
            theme=theme,
            title=title or "Backtest Tearsheet",
        )

        # Save to file if path provided
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html)

        return html

    def __repr__(self) -> str:
        """String representation."""
        n_trades = len(self.trades)
        n_bars = len(self.equity_curve)
        final_value = self.metrics.get("final_value", 0)
        total_return = self.metrics.get("total_return_pct", 0)
        return (
            f"BacktestResult(trades={n_trades}, bars={n_bars}, "
            f"final_value=${final_value:,.2f}, return={total_return:+.2f}%)"
        )


def enrich_trades_with_signals(
    trades_df: pl.DataFrame,
    signals_df: pl.DataFrame,
    signal_columns: list[str] | None = None,
    timestamp_col: str = "timestamp",
    asset_col: str | None = None,
) -> pl.DataFrame:
    """Enrich trades DataFrame with signal values at entry/exit times via as-of join.

    This function performs a backward as-of join to add signal values from the
    signals DataFrame to each trade at both entry and exit times. This is the
    recommended way to add ML features/signals to trades for analysis, rather
    than storing signals during backtest execution.

    This function is part of the cross-library API specification and should
    produce identical results across Python, Numba, and Rust implementations.

    Args:
        trades_df: Trades DataFrame with entry_time, exit_time columns.
            Typically from BacktestResult.to_trades_dataframe().
        signals_df: Signals DataFrame with timestamp and signal columns.
            Should have the same timestamps as the backtest data.
        signal_columns: Signal columns to include. If None, uses all columns
            except timestamp_col and asset_col.
        timestamp_col: Name of timestamp column in signals_df.
        asset_col: Name of asset column in signals_df for multi-asset signals.
            If None, assumes single-asset or already filtered.

    Returns:
        Trades DataFrame with added columns:
        - entry_{signal_name} for each signal
        - exit_{signal_name} for each signal

    Example:
        >>> from ml4t.backtest import Engine, enrich_trades_with_signals
        >>>
        >>> # Run backtest
        >>> result = engine.run()
        >>> trades_df = result.to_trades_dataframe()
        >>>
        >>> # Load signals used in backtest
        >>> signals = pl.read_parquet("ml_signals.parquet")
        >>>
        >>> # Enrich trades with signal values at entry/exit
        >>> enriched = enrich_trades_with_signals(
        ...     trades_df,
        ...     signals,
        ...     signal_columns=["momentum", "rsi", "ml_score"]
        ... )
        >>>
        >>> # Analyze: What was the ML score when we exited via stop-loss?
        >>> stop_loss_trades = enriched.filter(pl.col("exit_reason") == "stop_loss")
        >>> print(stop_loss_trades.select(["exit_ml_score", "pnl"]).describe())
    """
    # Determine signal columns if not specified
    exclude_cols = {timestamp_col}
    if asset_col:
        exclude_cols.add(asset_col)

    if signal_columns is None:
        signal_columns = [c for c in signals_df.columns if c not in exclude_cols]

    if not signal_columns:
        return trades_df

    # Preserve original trade order (join_asof requires sorting which disrupts order)
    trades_df = trades_df.with_row_index("_original_order")

    # Ensure signals are sorted by timestamp for join_asof
    signals_sorted = signals_df.sort(timestamp_col)

    # Join for entry signals
    entry_cols = [timestamp_col] + signal_columns
    if asset_col and asset_col in signals_df.columns:
        entry_cols = [timestamp_col, asset_col] + signal_columns

    entry_signals = signals_sorted.select(entry_cols)
    entry_rename = {c: f"entry_{c}" for c in signal_columns}
    entry_signals = entry_signals.rename(entry_rename)

    if asset_col and asset_col in signals_df.columns:
        # Multi-asset: join on both timestamp and asset
        result = trades_df.join_asof(
            entry_signals,
            left_on="entry_time",
            right_on=timestamp_col,
            by_left="asset",
            by_right=asset_col,
            strategy="backward",
        )
    else:
        # Single-asset: join on timestamp only
        result = trades_df.join_asof(
            entry_signals,
            left_on="entry_time",
            right_on=timestamp_col,
            strategy="backward",
        )

    # Join for exit signals
    exit_signals = signals_sorted.select(entry_cols)
    exit_rename = {c: f"exit_{c}" for c in signal_columns}
    exit_signals = exit_signals.rename(exit_rename)

    if asset_col and asset_col in signals_df.columns:
        result = result.join_asof(
            exit_signals,
            left_on="exit_time",
            right_on=timestamp_col,
            by_left="asset",
            by_right=asset_col,
            strategy="backward",
        )
    else:
        result = result.join_asof(
            exit_signals,
            left_on="exit_time",
            right_on=timestamp_col,
            strategy="backward",
        )

    # Restore original trade order and remove temporary column
    return result.sort("_original_order").drop("_original_order")
