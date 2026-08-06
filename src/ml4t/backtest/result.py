"""Structured backtest result with export capabilities.

This module provides a BacktestResult class that wraps the raw output from
Engine.run() with convenient DataFrame export methods and Parquet serialization.

Example:
    >>> from ml4t.backtest import Engine, DataFeed, Strategy
    >>> engine = Engine(feed, strategy)
    >>> result = engine.run()
    >>>
    >>> # Export trades and raw predictions to Parquet
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
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import polars as pl
from ml4t.specs.market_data import FeedSpec

try:
    from ._version import __version__
except ImportError:  # pragma: no cover - fallback for local editable edge cases
    __version__ = "0.0.0.dev0"
from .analytics.annualization import should_session_align
from .types import Fill, Order, OrderSide, OrderStatus, OrderType, Trade

if TYPE_CHECKING:
    from .analytics import EquityCurve, TradeAnalyzer
    from .config import BacktestConfig


_ARTIFACT_TYPE = "ml4t-backtest-result"
_ARTIFACT_SCHEMA_VERSION = 2
_MANIFEST_FILE = "manifest.json"
_INCOMPLETE_MARKER = ".artifact-incomplete"
_NONFINITE_FLOAT_KEY = "__ml4t_nonfinite_float__"
_COMPONENT_FILES = {
    "trades": "trades.parquet",
    "fills": "fills.parquet",
    "rejected_orders": "rejected_orders.parquet",
    "predictions": "predictions.parquet",
    "equity": "equity.parquet",
    "portfolio_state": "portfolio_state.parquet",
    "daily_pnl": "daily_pnl.parquet",
    "metrics": "metrics.json",
    "config": "config.yaml",
    "spec": "spec.yaml",
}
_REQUIRED_RESULT_COMPONENTS = frozenset(
    {"trades", "fills", "rejected_orders", "equity", "portfolio_state", "daily_pnl", "metrics"}
)


def _serialize_metric_value(value: Any, *, path: str) -> Any:
    """Convert a metric value to JSON-safe built-in containers and scalars."""
    if isinstance(value, bool | str | type(None) | int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            label = "nan"
        elif value > 0:
            label = "positive_infinity"
        else:
            label = "negative_infinity"
        return {_NONFINITE_FLOAT_KEY: label}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list | tuple):
        return [
            _serialize_metric_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ArtifactWriteError(f"{path} contains a non-string mapping key")
        return {
            key: _serialize_metric_value(item, path=f"{path}.{key}") for key, item in value.items()
        }
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return _serialize_metric_value(value.item(), path=path)
        if isinstance(value, np.ndarray):
            return _serialize_metric_value(value.tolist(), path=path)
    except (ImportError, AttributeError):
        pass
    if isinstance(value, pl.Series):
        return _serialize_metric_value(value.to_list(), path=path)
    raise ArtifactWriteError(f"{path} has unsupported value type {type(value).__name__}")


def _deserialize_metric_value(value: Any) -> Any:
    """Restore tagged non-finite floats from a portable JSON payload."""
    if isinstance(value, list):
        return [_deserialize_metric_value(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {_NONFINITE_FLOAT_KEY}:
            labels = {
                "nan": float("nan"),
                "positive_infinity": float("inf"),
                "negative_infinity": float("-inf"),
            }
            label = value[_NONFINITE_FLOAT_KEY]
            if label not in labels:
                raise ValueError(f"Unknown non-finite metric label: {label!r}")
            return labels[label]
        return {key: _deserialize_metric_value(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class ArtifactDiagnostic:
    """Structured description of one omission or recovery action."""

    code: str
    component: str
    message: str


class ArtifactError(ValueError):
    """Base class for result-artifact failures."""


class ArtifactNotFoundError(ArtifactError):
    """Raised when an artifact path does not contain any result data."""


class ArtifactManifestError(ArtifactError):
    """Raised when the artifact manifest is missing or malformed."""


class ArtifactIncompleteError(ArtifactError):
    """Raised when a current artifact is incomplete."""


class ArtifactReadError(ArtifactError):
    """Raised when a declared artifact component cannot be decoded."""


class ArtifactWriteError(ArtifactError):
    """Raised when a requested artifact component cannot be written."""


class UnsupportedArtifactVersionError(ArtifactError):
    """Raised when an artifact uses an unsupported schema version."""


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
        rejected_orders: Orders that reached the rejected terminal state. Orders
            cancelled under permissive insufficient-cash handling are not included.
        predictions: Raw prediction DataFrame passed into the backtest (optional)
        metrics: Dictionary of computed performance metrics
        config: BacktestConfig used for the backtest (optional)
        equity: EquityCurve analytics object
        trade_analyzer: TradeAnalyzer analytics object
        artifact_diagnostics: Structured omissions and recovery actions. Empty for
            artifacts loaded successfully in strict mode.
    """

    trades: list[Trade]
    equity_curve: list[tuple[datetime, float]]
    fills: list[Fill]
    metrics: dict[str, Any]
    predictions: pl.DataFrame | None = None
    config: BacktestConfig | None = None
    equity: EquityCurve | None = None
    trade_analyzer: TradeAnalyzer | None = None
    portfolio_state: list[tuple[datetime, float, float, float, float, int]] = field(
        default_factory=list
    )
    rejected_orders: list[Order] = field(default_factory=list)
    artifact_diagnostics: tuple[ArtifactDiagnostic, ...] = field(default_factory=tuple)

    # Cached DataFrames (computed on demand)
    _trades_df: pl.DataFrame | None = field(default=None, repr=False)
    _equity_df: pl.DataFrame | None = field(default=None, repr=False)
    _fills_df: pl.DataFrame | None = field(default=None, repr=False)
    _portfolio_state_df: pl.DataFrame | None = field(default=None, repr=False)
    _rejected_orders_df: pl.DataFrame | None = field(default=None, repr=False)

    def _feed_spec(self) -> FeedSpec | None:
        if self.config is None:
            return None
        return self.config.resolved_feed_spec

    def _auto_session_aligned(self, calendar: str | None = None) -> bool:
        timestamps = [ts for ts, _ in self.equity_curve]
        resolved_calendar = calendar or (self.config.resolved_calendar if self.config else None)
        return should_session_align(
            calendar=resolved_calendar,
            feed_spec=self._feed_spec(),
            timestamps=timestamps,
        )

    def to_trades_dataframe(self) -> pl.DataFrame:
        """Convert trades to Polars DataFrame.

        Returns DataFrame with columns:
            symbol, entry_time, exit_time, entry_price, exit_price,
            quantity, direction, pnl, pnl_percent, bars_held,
            fees, exit_slippage, mfe, mae, entry_slippage, multiplier,
            gross_pnl, net_return, total_slippage_cost, cost_drag,
            exit_reason, exit_reason_detail, status

        Cost decomposition columns:
            gross_pnl: Price-move P&L before fees
            net_return: Direction-aware net return including fees
            total_slippage_cost: Entry + exit slippage in dollars
            cost_drag: Total cost as fraction of notional

        The status column indicates "closed" (flat-to-flat completion), "partial"
        (realized reduction), or "open" (mark-to-market at end of backtest).

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
                    "exit_slippage": t.exit_slippage,
                    "mfe": t.mfe,
                    "mae": t.mae,
                    "entry_slippage": t.entry_slippage,
                    "multiplier": t.multiplier,
                    "entry_quote_mid_price": t.entry_quote_mid_price,
                    "entry_bid_price": t.entry_bid_price,
                    "entry_ask_price": t.entry_ask_price,
                    "entry_spread": t.entry_spread,
                    "entry_available_size": t.entry_available_size,
                    "exit_quote_mid_price": t.exit_quote_mid_price,
                    "exit_bid_price": t.exit_bid_price,
                    "exit_ask_price": t.exit_ask_price,
                    "exit_spread": t.exit_spread,
                    "exit_available_size": t.exit_available_size,
                    "gross_pnl": t.gross_pnl,
                    "net_return": t.net_return,
                    "total_slippage_cost": t.total_slippage_cost,
                    "cost_drag": t.cost_drag,
                    "exit_reason": t.exit_reason,
                    "exit_reason_detail": t.exit_reason_detail,
                    "status": t.status,
                }
            )

        self._trades_df = pl.DataFrame(records, schema=self._trades_schema())
        return self._trades_df

    def to_fills_dataframe(self) -> pl.DataFrame:
        """Convert fills to Polars DataFrame."""
        if self._fills_df is not None:
            return self._fills_df

        if not self.fills:
            return pl.DataFrame(schema=self._fills_schema())

        records = []
        for fill in self.fills:
            records.append(
                {
                    "order_id": fill.order_id,
                    "rebalance_id": fill.rebalance_id,
                    "asset": fill.asset,
                    "side": fill.side.value,
                    "quantity": fill.quantity,
                    "price": fill.price,
                    "timestamp": fill.timestamp,
                    "commission": fill.commission,
                    "slippage": fill.slippage,
                    "order_type": fill.order_type,
                    "limit_price": fill.limit_price,
                    "stop_price": fill.stop_price,
                    "price_source": fill.price_source,
                    "reference_price": fill.reference_price,
                    "quote_mid_price": fill.quote_mid_price,
                    "bid_price": fill.bid_price,
                    "ask_price": fill.ask_price,
                    "spread": fill.spread,
                    "bid_size": fill.bid_size,
                    "ask_size": fill.ask_size,
                    "available_size": fill.available_size,
                    "exit_reason": fill.exit_reason,
                    "exit_reason_detail": fill.exit_reason_detail,
                }
            )

        self._fills_df = pl.DataFrame(records, schema=self._fills_schema())
        return self._fills_df

    def to_rejected_orders_dataframe(self) -> pl.DataFrame:
        """Convert rejected orders to a stable, machine-readable DataFrame."""
        if self._rejected_orders_df is not None:
            return self._rejected_orders_df
        if not self.rejected_orders:
            return pl.DataFrame(schema=self._rejected_orders_schema())

        records = [
            {
                "order_id": order.order_id,
                "symbol": order.asset,
                "timestamp": order.created_at,
                "requested_quantity": order.requested_quantity,
                "filled_quantity": order.filled_quantity,
                "remaining_quantity": order.quantity,
                "side": order.side.value,
                "order_type": order.order_type.value,
                "limit_price": order.limit_price,
                "stop_price": order.stop_price,
                "trail_amount": order.trail_amount,
                "parent_id": order.parent_id,
                "rebalance_id": order.rebalance_id,
                "status": order.status.value,
                "rejection_code": order.rejection_code,
                "rejection_reason": order.rejection_reason,
            }
            for order in self.rejected_orders
        ]
        self._rejected_orders_df = pl.DataFrame(
            records,
            schema=self._rejected_orders_schema(),
        )
        return self._rejected_orders_df

    def to_predictions_dataframe(self) -> pl.DataFrame:
        """Return the raw prediction DataFrame used as backtest input."""
        if self.predictions is None:
            return pl.DataFrame()
        return self.predictions

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

    def to_portfolio_state_dataframe(self) -> pl.DataFrame:
        """Convert portfolio state snapshots to Polars DataFrame.

        Returns DataFrame with columns:
            timestamp, equity, cash, gross_exposure, net_exposure, open_positions

        Returns:
            Polars DataFrame with one row per bar, sorted by timestamp
        """
        if self._portfolio_state_df is not None:
            return self._portfolio_state_df

        if not self.portfolio_state:
            return pl.DataFrame(schema=self._portfolio_state_schema())

        self._portfolio_state_df = (
            pl.DataFrame(
                self.portfolio_state,
                schema=[
                    "timestamp",
                    "equity",
                    "cash",
                    "gross_exposure",
                    "net_exposure",
                    "open_positions",
                ],
                orient="row",
            )
            .sort("timestamp")
            .cast(self._portfolio_state_schema())
        )
        return self._portfolio_state_df

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

        if session_aligned and self.config and self.config.resolved_calendar:
            # Use session alignment
            from .sessions import SessionConfig, compute_session_pnl

            session_config = SessionConfig(
                calendar=self.config.resolved_calendar,
                timezone=self.config.resolved_timezone,
                session_start_time=self.config.resolved_session_start_time,
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
            >>> from ml4t.diagnostic.evaluation.metrics.risk_adjusted import sharpe_ratio
            >>> sharpe = sharpe_ratio(daily_returns.to_numpy(), annualization_factor=252)
        """
        # Determine session alignment
        if session_aligned is None:
            cal = calendar or (self.config.resolved_calendar if self.config else None)
            session_aligned = self._auto_session_aligned(cal)

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
                "portfolio_state": self.portfolio_state,
            }
        )
        if self.predictions is not None:
            result["predictions"] = self.predictions
        if self.equity is not None:
            result["equity"] = self.equity
        if self.trade_analyzer is not None:
            result["trade_analyzer"] = self.trade_analyzer
        return result

    def to_spec_dict(self) -> dict[str, Any]:
        """Export a resolved runtime spec for reproducibility.

        Returns:
            Dictionary containing the fully resolved config, library version,
            and realized run window. The nested ``config`` payload remains
            compatible with ``BacktestConfig.from_dict()``.
        """
        config_dict = self.config.to_dict() if self.config is not None else {}
        start = self.equity_curve[0][0].isoformat() if self.equity_curve else None
        end = self.equity_curve[-1][0].isoformat() if self.equity_curve else None
        return {
            "version": 1,
            "library_version": __version__,
            "config": config_dict,
            "window": {
                "start": start,
                "end": end,
            },
        }

    # Dict-like access keeps validation scripts and older notebook code working.
    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def keys(self):
        return self.to_dict().keys()

    def items(self):
        return self.to_dict().items()

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
                fills.parquet
                rejected_orders.parquet
                predictions.parquet
                equity.parquet
                portfolio_state.parquet
                daily_pnl.parquet
                metrics.json
                config.yaml (if config available)
                spec.yaml (if config available)
                manifest.json

        Args:
            path: Directory path to write files
            include: Components to include. Default: all.
                Options: ["trades", "fills", "rejected_orders", "predictions", "equity",
                    "portfolio_state", "daily_pnl", "metrics", "config", "spec"]
            compression: Parquet compression codec (default: "zstd")

        Returns:
            Dict mapping requested component names to file paths. The always-written
            manifest is returned under the additional ``"manifest"`` key; it is not
            a selectable component.

        Raises:
            ArtifactWriteError: If a requested component is unavailable or cannot be written.
        """
        explicitly_selected = include is not None
        requested = list(include) if include is not None else list(_COMPONENT_FILES)
        unknown = sorted(set(requested) - _COMPONENT_FILES.keys() - {"manifest"})
        if unknown:
            raise ArtifactWriteError(f"Unknown artifact components requested: {unknown}")
        requested = [name for name in requested if name != "manifest"]

        unavailable: dict[str, str] = {}
        if self.predictions is None:
            unavailable["predictions"] = "result has no predictions"
        if self.config is None:
            unavailable["config"] = "result has no config"
            unavailable["spec"] = "result has no config for a runtime spec"

        explicitly_unavailable = sorted(set(requested) & unavailable.keys())
        if explicitly_selected and explicitly_unavailable:
            details = ", ".join(f"{name}: {unavailable[name]}" for name in explicitly_unavailable)
            raise ArtifactWriteError(f"Requested artifact components are unavailable: {details}")

        selected = [name for name in requested if name not in unavailable]

        text_payloads: dict[str, str] = {}
        if "metrics" in selected:
            try:
                serializable_metrics = {
                    key: _serialize_metric_value(value, path=f"metrics[{key!r}]")
                    for key, value in self.metrics.items()
                }
                text_payloads["metrics"] = json.dumps(
                    serializable_metrics,
                    indent=2,
                    allow_nan=False,
                )
            except ArtifactWriteError:
                raise
            except Exception as exc:
                raise ArtifactWriteError(f"Failed to serialize metrics: {exc}") from exc

        if "config" in selected or "spec" in selected:
            try:
                import yaml
            except ImportError as exc:
                raise ArtifactWriteError("PyYAML is required to serialize config or spec") from exc
            if "config" in selected:
                try:
                    text_payloads["config"] = yaml.safe_dump(
                        self.config.to_dict(),
                        default_flow_style=False,
                    )
                except Exception as exc:
                    raise ArtifactWriteError(
                        f"Failed to serialize config component: {exc}"
                    ) from exc
            if "spec" in selected:
                try:
                    text_payloads["spec"] = yaml.safe_dump(
                        self.to_spec_dict(),
                        default_flow_style=False,
                        sort_keys=False,
                    )
                except Exception as exc:
                    raise ArtifactWriteError(f"Failed to serialize spec component: {exc}") from exc

        path = Path(path)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise ArtifactWriteError(f"Failed to create artifact directory {path}: {exc}") from exc

        def write_component(name: str, writer) -> None:
            try:
                writer()
            except Exception as exc:
                raise ArtifactWriteError(f"Failed to write {name} component: {exc}") from exc

        marker_path = path / _INCOMPLETE_MARKER
        write_component(
            "incomplete marker",
            lambda: marker_path.write_text("Result artifact write did not complete.\n"),
        )
        manifest_path = path / _MANIFEST_FILE
        write_component("stale manifest removal", lambda: manifest_path.unlink(missing_ok=True))

        written: dict[str, Path] = {}

        if "trades" in selected:
            trades_path = path / "trades.parquet"
            write_component(
                "trades",
                lambda: self.to_trades_dataframe().write_parquet(
                    trades_path,
                    compression=compression,
                ),
            )
            written["trades"] = trades_path

        if "fills" in selected:
            fills_path = path / "fills.parquet"
            write_component(
                "fills",
                lambda: self.to_fills_dataframe().write_parquet(
                    fills_path,
                    compression=compression,
                ),
            )
            written["fills"] = fills_path

        if "rejected_orders" in selected:
            rejected_orders_path = path / "rejected_orders.parquet"
            write_component(
                "rejected_orders",
                lambda: self.to_rejected_orders_dataframe().write_parquet(
                    rejected_orders_path,
                    compression=compression,
                ),
            )
            written["rejected_orders"] = rejected_orders_path

        if "predictions" in selected:
            predictions_path = path / "predictions.parquet"
            write_component(
                "predictions",
                lambda: self.to_predictions_dataframe().write_parquet(
                    predictions_path,
                    compression=compression,
                ),
            )
            written["predictions"] = predictions_path

        if "equity" in selected:
            equity_path = path / "equity.parquet"
            write_component(
                "equity",
                lambda: self.to_equity_dataframe().write_parquet(
                    equity_path,
                    compression=compression,
                ),
            )
            written["equity"] = equity_path

        if "portfolio_state" in selected:
            portfolio_state_path = path / "portfolio_state.parquet"
            write_component(
                "portfolio_state",
                lambda: self.to_portfolio_state_dataframe().write_parquet(
                    portfolio_state_path,
                    compression=compression,
                ),
            )
            written["portfolio_state"] = portfolio_state_path

        if "daily_pnl" in selected:
            daily_path = path / "daily_pnl.parquet"
            write_component(
                "daily_pnl",
                lambda: self.to_daily_pnl().write_parquet(
                    daily_path,
                    compression=compression,
                ),
            )
            written["daily_pnl"] = daily_path

        for name in ("metrics", "config", "spec"):
            if name not in selected:
                continue
            component_path = path / _COMPONENT_FILES[name]
            write_component(
                name,
                lambda component_path=component_path, payload=text_payloads[name]: (
                    component_path.write_text(payload)
                ),
            )
            written[name] = component_path

        manifest = {
            "artifact_type": _ARTIFACT_TYPE,
            "schema_version": _ARTIFACT_SCHEMA_VERSION,
            "library_version": __version__,
            "complete": written.keys() >= _REQUIRED_RESULT_COMPONENTS,
            "components": {
                name: _COMPONENT_FILES[name] for name in _COMPONENT_FILES if name in written
            },
            "omitted_components": {
                name: reason for name, reason in unavailable.items() if name in requested
            },
        }
        manifest_payload = json.dumps(manifest, indent=2, allow_nan=False)
        write_component("manifest", lambda: manifest_path.write_text(manifest_payload))
        written["manifest"] = manifest_path
        write_component("incomplete marker removal", marker_path.unlink)

        return written

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        *,
        recovery: bool = False,
    ) -> BacktestResult:
        """Load a validated result artifact.

        Args:
            path: Directory containing files written by :meth:`to_parquet`.
            recovery: Permit manifest-free beta artifacts and omit unreadable components.
                Every omission is reported through ``artifact_diagnostics``.

        Returns:
            BacktestResult instance.

        Raises:
            ArtifactError: If strict validation or component decoding fails.
        """
        path = Path(path)
        artifact_path = path
        if not path.exists():
            raise ArtifactNotFoundError(f"Result artifact path does not exist: {path}")
        if not path.is_dir():
            raise ArtifactNotFoundError(f"Result artifact path is not a directory: {path}")

        diagnostics: list[ArtifactDiagnostic] = []
        entries = list(path.iterdir())
        if not entries and not recovery:
            raise ArtifactNotFoundError(f"Result artifact directory is empty: {path}")

        marker_path = path / _INCOMPLETE_MARKER
        if marker_path.exists():
            if not recovery:
                raise ArtifactIncompleteError(
                    f"Result artifact contains {_INCOMPLETE_MARKER}; its write did not complete"
                )
            diagnostics.append(
                ArtifactDiagnostic(
                    code="incomplete_write",
                    component="manifest",
                    message="Artifact write did not complete.",
                )
            )

        def discover_legacy_components() -> dict[str, str]:
            discovered = {
                name: filename
                for name, filename in _COMPONENT_FILES.items()
                if (artifact_path / filename).exists()
            }
            if "predictions" not in discovered and (artifact_path / "signals.parquet").exists():
                discovered["predictions"] = "signals.parquet"
            return discovered

        manifest_path = path / _MANIFEST_FILE
        components: dict[str, str] = {}
        manifest: dict[str, Any] | None = None
        if not manifest_path.exists():
            if not recovery:
                raise ArtifactManifestError(
                    "Result artifact manifest is missing; pass recovery=True only for retained "
                    "beta artifacts"
                )
            diagnostics.append(
                ArtifactDiagnostic(
                    code="manifest_missing",
                    component="manifest",
                    message="Loaded a manifest-free beta artifact.",
                )
            )
            components = discover_legacy_components()
        else:
            try:
                with open(manifest_path) as file:
                    manifest_data = json.load(file)
                if not isinstance(manifest_data, dict):
                    raise TypeError("manifest root must be an object")
                manifest = manifest_data
            except Exception as exc:
                if not recovery:
                    raise ArtifactManifestError(
                        f"Failed to read {_MANIFEST_FILE}: {type(exc).__name__}: {exc}"
                    ) from exc
                diagnostics.append(
                    ArtifactDiagnostic(
                        code="manifest_invalid",
                        component="manifest",
                        message=f"Ignored malformed manifest ({type(exc).__name__}).",
                    )
                )
                components = discover_legacy_components()

        if manifest is not None:
            artifact_type = manifest.get("artifact_type")
            if artifact_type != _ARTIFACT_TYPE:
                message = f"Unsupported artifact type: {artifact_type!r}"
                if not recovery:
                    raise ArtifactManifestError(message)
                diagnostics.append(ArtifactDiagnostic("manifest_invalid", "manifest", message))
                components = discover_legacy_components()
                manifest = None

        if manifest is not None:
            schema_version = manifest.get("schema_version")
            if schema_version != _ARTIFACT_SCHEMA_VERSION:
                raise UnsupportedArtifactVersionError(
                    f"Unsupported result artifact schema version {schema_version!r}; "
                    f"supported version is {_ARTIFACT_SCHEMA_VERSION}"
                )
            component_data = manifest.get("components")
            if not isinstance(component_data, dict) or not all(
                isinstance(name, str) and isinstance(filename, str)
                for name, filename in component_data.items()
            ):
                if not recovery:
                    raise ArtifactManifestError("Manifest components must be a string mapping")
                diagnostics.append(
                    ArtifactDiagnostic(
                        "manifest_invalid",
                        "manifest",
                        "Ignored invalid component mapping.",
                    )
                )
                components = discover_legacy_components()
            else:
                unknown = sorted(set(component_data) - _COMPONENT_FILES.keys())
                noncanonical = sorted(
                    name
                    for name, filename in component_data.items()
                    if name in _COMPONENT_FILES and filename != _COMPONENT_FILES[name]
                )
                if unknown or noncanonical:
                    details = f"unknown={unknown}, noncanonical={noncanonical}"
                    if not recovery:
                        raise ArtifactManifestError(f"Invalid manifest components: {details}")
                    diagnostics.append(
                        ArtifactDiagnostic(
                            "manifest_invalid",
                            "manifest",
                            f"Ignored invalid manifest components: {details}.",
                        )
                    )
                    components = discover_legacy_components()
                else:
                    components = dict(component_data)

        declared_incomplete = manifest is not None and manifest.get("complete") is not True
        if declared_incomplete and not recovery:
            raise ArtifactIncompleteError("Result artifact manifest marks the export incomplete")
        if declared_incomplete:
            diagnostics.append(
                ArtifactDiagnostic(
                    code="manifest_incomplete",
                    component="manifest",
                    message="Manifest marks this as a selective or incomplete export.",
                )
            )

        missing_required_components = sorted(_REQUIRED_RESULT_COMPONENTS - components.keys())
        missing_files = sorted(
            name for name, filename in components.items() if not (path / filename).is_file()
        )
        if not recovery and (missing_required_components or missing_files):
            raise ArtifactIncompleteError(
                "Result artifact is incomplete: "
                f"missing components={missing_required_components}, missing files={missing_files}"
            )
        if recovery:
            missing_components = sorted(_COMPONENT_FILES.keys() - components.keys())
            diagnostics.extend(
                ArtifactDiagnostic(
                    code="component_missing",
                    component=name,
                    message=f"Component {_COMPONENT_FILES[name]} is absent.",
                )
                for name in missing_components
            )
            for name in missing_files:
                diagnostics.append(
                    ArtifactDiagnostic(
                        code="component_missing_file",
                        component=name,
                        message=f"Declared component {components[name]} is absent.",
                    )
                )
                components.pop(name)

        def read_component(name: str, reader, default):
            filename = components.get(name)
            if filename is None:
                return default
            try:
                return reader(path / filename)
            except Exception as exc:
                if not recovery:
                    raise ArtifactReadError(
                        f"Failed to read {filename}: {type(exc).__name__}: {exc}"
                    ) from exc
                diagnostics.append(
                    ArtifactDiagnostic(
                        code="component_invalid",
                        component=name,
                        message=f"Ignored unreadable {filename} ({type(exc).__name__}).",
                    )
                )
                return default

        def read_trades(component_path: Path) -> list[Trade]:
            result: list[Trade] = []
            for row in pl.read_parquet(component_path).iter_rows(named=True):
                symbol = row.get("symbol") or row.get("asset", "")
                fees = row.get("fees")
                if fees is None:
                    fees = row.get("commission", 0.0)
                result.append(
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
                        exit_slippage=row.get("exit_slippage", row.get("slippage", 0.0)),
                        exit_reason=row.get("exit_reason", "signal"),
                        exit_reason_detail=row.get("exit_reason_detail"),
                        status=row.get("status", "closed"),
                        mfe=row.get("mfe", 0.0),
                        mae=row.get("mae", 0.0),
                        entry_slippage=row.get("entry_slippage", 0.0),
                        multiplier=row.get("multiplier", 1.0),
                        entry_quote_mid_price=row.get("entry_quote_mid_price"),
                        entry_bid_price=row.get("entry_bid_price"),
                        entry_ask_price=row.get("entry_ask_price"),
                        entry_spread=row.get("entry_spread"),
                        entry_available_size=row.get("entry_available_size"),
                        exit_quote_mid_price=row.get("exit_quote_mid_price"),
                        exit_bid_price=row.get("exit_bid_price"),
                        exit_ask_price=row.get("exit_ask_price"),
                        exit_spread=row.get("exit_spread"),
                        exit_available_size=row.get("exit_available_size"),
                    )
                )
            return result

        def read_fills(component_path: Path) -> list[Fill]:
            result: list[Fill] = []
            for row in pl.read_parquet(component_path).iter_rows(named=True):
                result.append(
                    Fill(
                        order_id=row["order_id"],
                        rebalance_id=row.get("rebalance_id"),
                        asset=row["asset"],
                        side=OrderSide(row["side"]),
                        quantity=row["quantity"],
                        price=row["price"],
                        timestamp=row["timestamp"],
                        commission=row.get("commission", 0.0),
                        slippage=row.get("slippage", 0.0),
                        order_type=row.get("order_type", ""),
                        limit_price=row.get("limit_price"),
                        stop_price=row.get("stop_price"),
                        price_source=row.get("price_source", ""),
                        reference_price=row.get("reference_price"),
                        quote_mid_price=row.get("quote_mid_price"),
                        bid_price=row.get("bid_price"),
                        ask_price=row.get("ask_price"),
                        spread=row.get("spread"),
                        bid_size=row.get("bid_size"),
                        ask_size=row.get("ask_size"),
                        available_size=row.get("available_size"),
                        exit_reason=row.get("exit_reason", ""),
                        exit_reason_detail=row.get("exit_reason_detail"),
                    )
                )
            return result

        def read_rejected_orders(component_path: Path) -> list[Order]:
            result: list[Order] = []
            for row in pl.read_parquet(component_path).iter_rows(named=True):
                result.append(
                    Order(
                        order_id=row["order_id"],
                        asset=row["symbol"],
                        created_at=row["timestamp"],
                        requested_quantity=row["requested_quantity"],
                        quantity=row.get("remaining_quantity", row["requested_quantity"]),
                        filled_quantity=row.get("filled_quantity", 0.0),
                        side=OrderSide(row["side"]),
                        order_type=OrderType(row["order_type"]),
                        limit_price=row.get("limit_price"),
                        stop_price=row.get("stop_price"),
                        trail_amount=row.get("trail_amount"),
                        parent_id=row.get("parent_id"),
                        rebalance_id=row.get("rebalance_id"),
                        status=OrderStatus(row["status"]),
                        rejection_reason=row.get("rejection_reason"),
                        _rejection_code=row.get("rejection_code"),
                    )
                )
            return result

        def read_equity(component_path: Path) -> list[tuple[datetime, float]]:
            return [
                (row["timestamp"], row["equity"])
                for row in pl.read_parquet(component_path).iter_rows(named=True)
            ]

        def read_portfolio_state(
            component_path: Path,
        ) -> list[tuple[datetime, float, float, float, float, int]]:
            return [
                (
                    row["timestamp"],
                    row["equity"],
                    row["cash"],
                    row["gross_exposure"],
                    row["net_exposure"],
                    row["open_positions"],
                )
                for row in pl.read_parquet(component_path).iter_rows(named=True)
            ]

        def read_metrics(component_path: Path) -> dict[str, Any]:
            with open(component_path) as file:
                data = json.load(file)
            if not isinstance(data, dict):
                raise TypeError("metrics root must be an object")
            return _deserialize_metric_value(data)

        def read_config(component_path: Path):
            import yaml

            from .config import BacktestConfig

            with open(component_path) as file:
                data = yaml.safe_load(file)
            if not isinstance(data, dict):
                raise TypeError("config root must be a mapping")
            return BacktestConfig.from_dict(data)

        def read_spec_config(component_path: Path):
            import yaml

            from .config import BacktestConfig

            with open(component_path) as file:
                data = yaml.safe_load(file)
            if not isinstance(data, dict):
                raise TypeError("spec root must be a mapping")
            if data.get("version") != 1:
                raise ValueError(f"unsupported spec version {data.get('version')!r}")
            config_data = data.get("config")
            if not isinstance(config_data, dict):
                raise TypeError("spec config must be a mapping")
            return BacktestConfig.from_dict(config_data)

        trades = read_component("trades", read_trades, [])
        fills = read_component("fills", read_fills, [])
        rejected_orders = read_component("rejected_orders", read_rejected_orders, [])
        equity_curve = read_component("equity", read_equity, [])
        portfolio_state = read_component("portfolio_state", read_portfolio_state, [])
        metrics = read_component("metrics", read_metrics, {})
        predictions = read_component("predictions", pl.read_parquet, None)
        daily_pnl = read_component("daily_pnl", pl.read_parquet, None)
        config = read_component("config", read_config, None)
        spec_config = read_component("spec", read_spec_config, None)
        if config is None:
            config = spec_config

        if daily_pnl is not None:
            expected_daily_pnl = cls(
                trades=[],
                equity_curve=equity_curve,
                fills=[],
                metrics={},
            ).to_daily_pnl()
            if not daily_pnl.equals(expected_daily_pnl):
                message = "daily_pnl.parquet is inconsistent with equity.parquet"
                if not recovery:
                    raise ArtifactReadError(message)
                diagnostics.append(
                    ArtifactDiagnostic(
                        code="component_inconsistent",
                        component="daily_pnl",
                        message=message,
                    )
                )

        return cls(
            trades=trades,
            equity_curve=equity_curve,
            fills=fills,
            predictions=predictions,
            portfolio_state=portfolio_state,
            rejected_orders=rejected_orders,
            metrics=metrics,
            config=config,
            artifact_diagnostics=tuple(diagnostics),
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
            "exit_slippage": pl.Float64(),
            "mfe": pl.Float64(),
            "mae": pl.Float64(),
            "entry_slippage": pl.Float64(),
            "multiplier": pl.Float64(),
            "entry_quote_mid_price": pl.Float64(),
            "entry_bid_price": pl.Float64(),
            "entry_ask_price": pl.Float64(),
            "entry_spread": pl.Float64(),
            "entry_available_size": pl.Float64(),
            "exit_quote_mid_price": pl.Float64(),
            "exit_bid_price": pl.Float64(),
            "exit_ask_price": pl.Float64(),
            "exit_spread": pl.Float64(),
            "exit_available_size": pl.Float64(),
            "gross_pnl": pl.Float64(),
            "net_return": pl.Float64(),
            "total_slippage_cost": pl.Float64(),
            "cost_drag": pl.Float64(),
            "exit_reason": pl.String(),
            "exit_reason_detail": pl.String(),
            "status": pl.String(),  # "closed", "partial", or "open"
        }

    @staticmethod
    def _fills_schema() -> dict[str, pl.DataType]:
        """Schema for fills DataFrame."""
        return {
            "order_id": pl.String(),
            "rebalance_id": pl.String(),
            "asset": pl.String(),
            "side": pl.String(),
            "quantity": pl.Float64(),
            "price": pl.Float64(),
            "timestamp": pl.Datetime(),
            "commission": pl.Float64(),
            "slippage": pl.Float64(),
            "order_type": pl.String(),
            "limit_price": pl.Float64(),
            "stop_price": pl.Float64(),
            "price_source": pl.String(),
            "reference_price": pl.Float64(),
            "quote_mid_price": pl.Float64(),
            "bid_price": pl.Float64(),
            "ask_price": pl.Float64(),
            "spread": pl.Float64(),
            "bid_size": pl.Float64(),
            "ask_size": pl.Float64(),
            "available_size": pl.Float64(),
            "exit_reason": pl.String(),
            "exit_reason_detail": pl.String(),
        }

    @staticmethod
    def _rejected_orders_schema() -> dict[str, pl.DataType]:
        """Schema for rejected order records added compatibly in v0.1.0."""
        return {
            "order_id": pl.String(),
            "symbol": pl.String(),
            "timestamp": pl.Datetime(),
            "requested_quantity": pl.Float64(),
            "filled_quantity": pl.Float64(),
            "remaining_quantity": pl.Float64(),
            "side": pl.String(),
            "order_type": pl.String(),
            "limit_price": pl.Float64(),
            "stop_price": pl.Float64(),
            "trail_amount": pl.Float64(),
            "parent_id": pl.String(),
            "rebalance_id": pl.String(),
            "status": pl.String(),
            "rejection_code": pl.String(),
            "rejection_reason": pl.String(),
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

    @staticmethod
    def _portfolio_state_schema() -> dict[str, pl.DataType]:
        """Schema for portfolio state DataFrame."""
        return {
            "timestamp": pl.Datetime(),
            "equity": pl.Float64(),
            "cash": pl.Float64(),
            "gross_exposure": pl.Float64(),
            "net_exposure": pl.Float64(),
            "open_positions": pl.Int32(),
        }

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
    trades_asset_col: str | None = None,
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
        trades_asset_col: Name of asset column in trades_df.
            If None, auto-detects: "symbol" first, then "asset".

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

    # Detect trade-side asset column when doing multi-asset enrichment
    if asset_col and asset_col in signals_df.columns:
        if trades_asset_col is None:
            if "symbol" in trades_df.columns:
                trades_asset_col = "symbol"
            elif "asset" in trades_df.columns:
                trades_asset_col = "asset"
            else:
                raise ValueError(
                    "Multi-asset enrichment requires trades_df to include 'symbol' or 'asset', "
                    "or set trades_asset_col explicitly."
                )
        if trades_asset_col not in trades_df.columns:
            raise ValueError(f"trades_asset_col '{trades_asset_col}' not found in trades_df")

    # Ensure sortedness for join_asof
    signals_sorted = (
        signals_df.sort([asset_col, timestamp_col])
        if asset_col and asset_col in signals_df.columns
        else signals_df.sort(timestamp_col)
    )
    trades_sorted = (
        trades_df.sort([trades_asset_col, "entry_time"])
        if trades_asset_col
        else trades_df.sort("entry_time")
    )

    # Join for entry signals
    entry_cols = [timestamp_col] + signal_columns
    if asset_col and asset_col in signals_df.columns:
        entry_cols = [timestamp_col, asset_col] + signal_columns

    entry_signals = signals_sorted.select(entry_cols)
    entry_rename = {c: f"entry_{c}" for c in signal_columns}
    entry_signals = entry_signals.rename(entry_rename)

    if asset_col and asset_col in signals_df.columns:
        # Multi-asset: join on both timestamp and asset
        result = trades_sorted.join_asof(
            entry_signals,
            left_on="entry_time",
            right_on=timestamp_col,
            by_left=trades_asset_col,
            by_right=asset_col,
            strategy="backward",
            check_sortedness=False,
        )
    else:
        # Single-asset: join on timestamp only
        result = trades_sorted.join_asof(
            entry_signals,
            left_on="entry_time",
            right_on=timestamp_col,
            strategy="backward",
        )

    # Join for exit signals
    exit_signals = signals_sorted.select(entry_cols)
    exit_rename = {c: f"exit_{c}" for c in signal_columns}
    exit_signals = exit_signals.rename(exit_rename)
    result_for_exit = (
        result.sort([trades_asset_col, "exit_time"])
        if trades_asset_col
        else result.sort("exit_time")
    )

    if asset_col and asset_col in signals_df.columns:
        result = result_for_exit.join_asof(
            exit_signals,
            left_on="exit_time",
            right_on=timestamp_col,
            by_left=trades_asset_col,
            by_right=asset_col,
            strategy="backward",
            check_sortedness=False,
        )
    else:
        result = result_for_exit.join_asof(
            exit_signals,
            left_on="exit_time",
            right_on=timestamp_col,
            strategy="backward",
        )

    # Restore original trade order and remove temporary column
    return result.sort("_original_order").drop("_original_order")
