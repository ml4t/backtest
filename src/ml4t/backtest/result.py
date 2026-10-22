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
from collections.abc import ItemsView, Iterable, KeysView
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import polars as pl
from ml4t.specs import UNSET_CALENDAR, CalendarUnset
from ml4t.specs.market_data import FeedSpec

from ._action_artifacts import action_digest, action_frame, validate_action_tables
from ._futures_artifacts import (
    ACCOUNTING_COLUMNS,
    futures_payload,
    settlement_frame,
    validate_futures_tables,
    validate_instrument_accounting,
)
from ._input_provenance import provenance_digest, validate_input_provenance
from ._rolls import roll_digest, validate_rolls
from ._temporal import temporal_frame, temporal_rows, timestamp_dtype

try:
    from ._version import __version__
except ImportError:  # pragma: no cover - fallback for local editable edge cases
    __version__ = "0.0.0.dev0"
from .analytics.annualization import should_session_align
from .benchmark import _validate_benchmark_frame, _validate_benchmark_metadata
from .capital import CapitalChange, flow_adjusted_returns
from .controls import AccountControlContext
from .funding import FundingPayment
from .types import Fill, Order, OrderSide, OrderStatus, OrderType, Trade

if TYPE_CHECKING:
    from .analytics import EquityCurve, TradeAnalyzer
    from .config import BacktestConfig


_ARTIFACT_TYPE = "ml4t-backtest-result"
_ARTIFACT_SCHEMA_VERSION = 3
_BENCHMARK_ARTIFACT_SCHEMA_VERSION = 4
_ACTION_ARTIFACT_SCHEMA_VERSION = 5
_FUTURES_ARTIFACT_SCHEMA_VERSION = 6
_ROLL_ARTIFACT_SCHEMA_VERSION = 7
_MANIFEST_FILE = "manifest.json"
_INCOMPLETE_MARKER = ".artifact-incomplete"
_NONFINITE_FLOAT_KEY = "__ml4t_nonfinite_float__"
_COMPONENT_FILES = {
    "trades": "trades.parquet",
    "fills": "fills.parquet",
    "funding": "funding.parquet",
    "rejected_orders": "rejected_orders.parquet",
    "predictions": "predictions.parquet",
    "equity": "equity.parquet",
    "portfolio_state": "portfolio_state.parquet",
    "holdings": "holdings.parquet",
    "capital_flows": "capital_flows.parquet",
    "recorded_variables": "recorded_variables.parquet",
    "benchmark": "benchmark.parquet",
    "corporate_actions": "corporate_actions.parquet",
    "corporate_entitlements": "corporate_entitlements.parquet",
    "futures_settlements": "futures_settlements.parquet",
    "futures_rolls": "futures_rolls.parquet",
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


def _decode_recorded_value(payload: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Nonstandard JSON constant {value!r}; use tagged values")

    return _deserialize_metric_value(json.loads(payload, parse_constant=reject_constant))


def _validate_complete_holdings(
    account: pl.DataFrame,
    holdings: pl.DataFrame,
    error_type: type[ArtifactError] = ArtifactReadError,
) -> None:
    counts = dict(temporal_rows(holdings.group_by("observation_id").len()))
    for row in temporal_rows(account, named=True):
        actual = counts.pop(row["observation_id"], 0)
        if row["holdings_complete"]:
            if actual != row["open_positions"]:
                raise error_type("Declared complete holdings omit or add observed positions")
        elif actual:
            raise error_type("Uncaptured account observations cannot declare complete holdings")
    if counts:
        raise error_type("Holdings do not refer to recorded account observations")


def _benchmark_digest(frame: pl.DataFrame) -> str:
    rows = zip(
        frame["timestamp"].cast(pl.Int64).to_list(),
        [value.hex() for value in frame["benchmark_return"].to_list()],
        strict=True,
    )
    return sha256(json.dumps(list(rows), separators=(",", ":")).encode()).hexdigest()


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
        funding_payments: Timestamped cash flows separate from trades and fills
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
    funding_payments: list[FundingPayment] = field(default_factory=list)
    artifact_diagnostics: tuple[ArtifactDiagnostic, ...] = field(default_factory=tuple)
    initial_capital: float | None = None
    holdings: list[tuple[Any, ...]] | None = None
    holdings_sampling: str = "none"
    capital_changes: list[CapitalChange] = field(default_factory=list)
    returns_include_commissions: bool | None = None
    observation_metadata: list[tuple[str, bool, int | None]] = field(default_factory=list)
    recorded_variables: list[tuple[datetime, int, str, str]] | None = None
    account_method: str | None = None
    benchmark: pl.DataFrame | None = None
    benchmark_metadata: dict[str, Any] | None = None
    corporate_actions: list[dict[str, Any]] | None = None
    corporate_entitlements: list[dict[str, Any]] | None = None
    corporate_receivables: list[float] | None = None
    instrument_accounting: dict[str, dict[str, Any]] | None = None
    accounting_observations: list[AccountControlContext] | None = None
    futures_settlements: list[dict[str, Any]] | None = None
    futures_rolls: list[dict[str, Any]] | None = None
    input_provenance: dict[str, Any] | None = None

    def to_rolls_dataframe(self) -> pl.DataFrame:
        """Return actual close/open order outcomes; paired execution is not atomic."""
        if self.futures_rolls is None:
            raise ValueError("Result has no captured futures roll intents")
        from ._rolls import roll_frame

        return roll_frame(self.futures_rolls)

    def to_futures_settlements_dataframe(self) -> pl.DataFrame:
        """Return actual mark, scheduled and closing variation transfers, excluding fees."""
        if self.futures_settlements is None:
            raise ValueError("Result has no captured futures variation ledger")
        return settlement_frame(self.futures_settlements)

    def to_actions_dataframe(self) -> pl.DataFrame:
        """Return actual account/order action applications, including cash movements."""
        if self.corporate_actions is None:
            raise ValueError("Result has no captured corporate-action ledger")
        return action_frame(self.corporate_actions)

    def to_entitlements_dataframe(self) -> pl.DataFrame:
        """Return earned obligations independently of later trading, including paid status."""
        if self.corporate_entitlements is None:
            raise ValueError("Result has no captured corporate entitlements")
        return action_frame(self.corporate_entitlements, entitlements=True)

    # Cached DataFrames (computed on demand)
    _trades_df: pl.DataFrame | None = field(default=None, repr=False)
    _equity_df: pl.DataFrame | None = field(default=None, repr=False)
    _fills_df: pl.DataFrame | None = field(default=None, repr=False)
    _portfolio_state_df: pl.DataFrame | None = field(default=None, repr=False)
    _rejected_orders_df: pl.DataFrame | None = field(default=None, repr=False)

    def to_benchmark_dataframe(self) -> pl.DataFrame:
        """Return prepared period returns with their source and financial definition."""
        if self.benchmark is None:
            raise ValueError("Result has no prepared benchmark")
        _validate_benchmark_metadata(self.benchmark_metadata)
        frame = _validate_benchmark_frame(self.benchmark)
        timezone = self.config.resolved_timezone if self.config is not None else "UTC"
        import pandas as pd

        expected = {
            (
                pd.Timestamp(point).tz_localize(timezone)
                if pd.Timestamp(point).tz is None
                else pd.Timestamp(point)
            )
            .tz_convert("UTC")
            .value
            for point, _ in self.equity_curve
        }
        if set(frame["timestamp"].cast(pl.Int64).to_list()) != expected:
            raise ValueError("Benchmark periods do not match observed equity periods")
        return frame

    def _feed_spec(self) -> FeedSpec | None:
        if self.config is None:
            return None
        return self.config.resolved_feed_spec

    def _auto_session_aligned(self, calendar: str | None | CalendarUnset = UNSET_CALENDAR) -> bool:
        timestamps = [ts for ts, _ in self.equity_curve]
        resolved_calendar = (
            self._feed_spec() or FeedSpec(time_contract_version=2)
        ).resolve_calendar(calendar=calendar)
        return should_session_align(
            calendar=resolved_calendar,
            feed_spec=self._feed_spec(),
            timestamps=timestamps,
        )

    def _timestamp_dtype(self, timestamps: Iterable[datetime | None]) -> pl.DataType:
        """Use the recorded zone, or the equity timeline for an empty export."""
        return timestamp_dtype(chain(timestamps, (timestamp for timestamp, _ in self.equity_curve)))

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

        schema = self._trades_schema()
        schema["entry_time"] = self._timestamp_dtype(t.entry_time for t in self.trades)
        schema["exit_time"] = self._timestamp_dtype(t.exit_time for t in self.trades)
        if not self.trades:
            return temporal_frame(schema=schema)

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

        self._trades_df = temporal_frame(records, schema=schema)
        return self._trades_df

    def to_fills_dataframe(self) -> pl.DataFrame:
        """Convert every execution fill to a stable Polars DataFrame.

        The result includes order identity, quantity, execution costs, price
        source, nullable quote context, available size, and exit-reason fields.
        An empty result retains the same typed schema.
        """
        if self._fills_df is not None:
            return self._fills_df

        schema = self._fills_schema()
        schema["timestamp"] = self._timestamp_dtype(fill.timestamp for fill in self.fills)
        schema["stop_triggered_at"] = schema["timestamp"]
        if not self.fills:
            return temporal_frame(schema=schema)

        records = []
        for sequence, fill in enumerate(self.fills):
            records.append(
                {
                    "order_id": fill.order_id,
                    "fill_sequence": sequence,
                    "rebalance_id": fill.rebalance_id,
                    "asset": fill.asset,
                    "side": fill.side.value,
                    "quantity": fill.quantity,
                    "signed_quantity": fill.quantity
                    if fill.side is OrderSide.BUY
                    else -fill.quantity,
                    "multiplier": fill.multiplier,
                    "currency": fill.currency,
                    "price": fill.price,
                    "timestamp": fill.timestamp,
                    "commission": fill.commission,
                    "slippage": fill.slippage,
                    "order_type": fill.order_type,
                    "limit_price": fill.limit_price,
                    "stop_price": fill.stop_price,
                    "stop_triggered_at": fill.stop_triggered_at,
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

        self._fills_df = temporal_frame(records, schema=schema)
        return self._fills_df

    def to_funding_dataframe(self) -> pl.DataFrame:
        """Return timestamped funding cash flows separately from trades and fills."""
        schema = {
            "timestamp": self._timestamp_dtype(p.timestamp for p in self.funding_payments),
            "asset": pl.String(),
            "quantity": pl.Float64(),
            "mark_price": pl.Float64(),
            "multiplier": pl.Float64(),
            "rate": pl.Float64(),
            "amount_per_unit": pl.Float64(),
            "cash_delta": pl.Float64(),
        }
        return temporal_frame(
            [
                {
                    "timestamp": payment.timestamp,
                    "asset": payment.asset,
                    "quantity": payment.quantity,
                    "mark_price": payment.mark_price,
                    "multiplier": payment.multiplier,
                    "rate": payment.rate,
                    "amount_per_unit": payment.amount_per_unit,
                    "cash_delta": payment.cash_delta,
                }
                for payment in self.funding_payments
            ],
            schema=schema,
        )

    def to_rejected_orders_dataframe(self) -> pl.DataFrame:
        """Convert rejected orders to a stable, machine-readable DataFrame."""
        if self._rejected_orders_df is not None:
            return self._rejected_orders_df
        schema = self._rejected_orders_schema()
        schema["timestamp"] = self._timestamp_dtype(
            order.created_at for order in self.rejected_orders
        )
        if not self.rejected_orders:
            return temporal_frame(schema=schema)

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
        self._rejected_orders_df = temporal_frame(
            records,
            schema=schema,
        )
        return self._rejected_orders_df

    def to_predictions_dataframe(self) -> pl.DataFrame:
        """Return the raw prediction DataFrame used as backtest input."""
        if self.predictions is None:
            return temporal_frame()
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
            return temporal_frame(schema=self._equity_schema())

        timestamps = [ts for ts, _ in self.equity_curve]
        values = [float(v) for _, v in self.equity_curve]

        # Build base DataFrame and sort by timestamp
        df = temporal_frame({"timestamp": timestamps, "equity": values}).sort("timestamp")
        initial = self.initial_capital if self.initial_capital is not None else df["equity"][0]
        previous = pl.col("equity").shift(1).fill_null(initial)
        if self.capital_changes:
            if self.initial_capital is None:
                raise ArtifactError("Capital-flow returns require recorded initial_capital")
            returns = flow_adjusted_returns(
                list(temporal_rows(df)), self.initial_capital, self.capital_changes
            )
            df = df.with_columns(pl.Series("return", returns))
            df = df.with_columns(
                ((pl.col("return") + 1).cum_prod() * initial).alias("performance_equity")
            )
        else:
            df = df.with_columns(
                (pl.col("equity") / previous - 1.0).alias("return"),
                pl.col("equity").alias("performance_equity"),
            )

        # Vectorized computation using Polars
        df = df.with_columns(
            [
                (pl.col("performance_equity") / initial - 1.0).alias("cumulative_return"),
                pl.col("performance_equity")
                .cum_max()
                .clip(lower_bound=initial)
                .alias("high_water_mark"),
            ]
        ).with_columns(
            # Drawdown: (equity / hwm) - 1, handle division by zero
            pl.when(pl.col("high_water_mark") > 0)
            .then(pl.col("performance_equity") / pl.col("high_water_mark") - 1.0)
            .otherwise(0.0)
            .alias("drawdown")
        )

        # Reorder columns to match expected schema
        self._equity_df = df.select(
            [
                "timestamp",
                "equity",
                "return",
                "cumulative_return",
                "drawdown",
                "high_water_mark",
                "performance_equity",
            ]
        )

        return self._equity_df

    def to_portfolio_state_dataframe(self) -> pl.DataFrame:
        """Return ordered account observations and explicit holdings completeness."""
        if self._portfolio_state_df is not None:
            return self._portfolio_state_df
        schema = self._portfolio_state_schema()
        schema["timestamp"] = self._timestamp_dtype(row[0] for row in self.portfolio_state)
        frame = (
            temporal_frame(self.portfolio_state, schema=schema, orient="row")
            .sort("timestamp", maintain_order=True)
            .with_row_index("observation_id")
        )
        metadata = self.observation_metadata
        if metadata and len(metadata) != len(frame):
            raise ArtifactError("Account observation metadata must match the recorded timeline")
        if not metadata:
            metadata = [
                ("bar", self.holdings_sampling == "bar" and self.holdings is not None, None)
            ] * len(frame)
        self._portfolio_state_df = frame.with_columns(
            pl.Series("observation_kind", [row[0] for row in metadata], dtype=pl.String),
            pl.Series("holdings_complete", [row[1] for row in metadata], dtype=pl.Boolean),
            pl.Series("fill_sequence", [row[2] for row in metadata], dtype=pl.UInt32),
        )
        if self.corporate_receivables is not None:
            if len(self.corporate_receivables) != frame.height:
                raise ArtifactError("Corporate receivables must match actual account observations")
            self._portfolio_state_df = self._portfolio_state_df.with_columns(
                pl.Series("receivables_value", self.corporate_receivables, dtype=pl.Float64)
            )
        if self.accounting_observations is not None:
            if len(self.accounting_observations) != frame.height:
                raise ArtifactError("Futures accounting must match actual account observations")
            self._portfolio_state_df = self._portfolio_state_df.with_columns(
                *(
                    pl.Series(
                        name,
                        [getattr(row, name) for row in self.accounting_observations],
                        dtype=dtype,
                    )
                    for name, dtype in ACCOUNTING_COLUMNS.items()
                )
            )
        return self._portfolio_state_df

    def to_holdings_dataframe(self) -> pl.DataFrame:
        """Return complete bar snapshots, with account keys retaining all-cash observations.

        Enable with ``Engine(..., holdings_sampling="bar")``. Missing legacy or
        disabled capture is unavailable, rather than a complete empty portfolio.
        Currency is null when no contract specification declares it. Under the
        current signed-notional account method, liquidation value equals signed
        notional; future account methods must report their own value separately.
        """
        if self.holdings is None:
            raise ArtifactError("Holdings were not captured; enable holdings_sampling='bar'")
        schema = self._holdings_schema()
        schema["timestamp"] = self._timestamp_dtype(row[0] for row in self.holdings)
        schema["mark_timestamp"] = self._timestamp_dtype(row[-1] for row in self.holdings)
        return temporal_frame(self.holdings, schema=schema, orient="row")

    def to_capital_flows_dataframe(self) -> pl.DataFrame:
        """Return ordered external capital applications, with marked pre/post-flow NAV."""
        schema = {
            "timestamp": self._timestamp_dtype(change.timestamp for change in self.capital_changes),
            "event_id": pl.String(),
            "kind": pl.String(),
            "value": pl.Float64(),
            "cash_delta": pl.Float64(),
            "nav_before": pl.Float64(),
            "nav_after": pl.Float64(),
        }
        return temporal_frame(
            [
                (
                    change.timestamp,
                    change.event_id,
                    change.kind,
                    change.value,
                    change.cash_delta,
                    change.nav_before,
                    change.nav_after,
                )
                for change in self.capital_changes
            ],
            schema=schema,
            orient="row",
        ).with_row_index("flow_sequence")

    def to_recorded_dataframe(self) -> pl.DataFrame:
        """Return bar-end recorded values in portable tagged JSON, outside account fields."""
        if self.recorded_variables is None:
            raise ArtifactError("Strategy variables were not recorded")
        schema = {
            "timestamp": self._timestamp_dtype(row[0] for row in self.recorded_variables),
            "bar_index": pl.UInt32,
            "name": pl.String,
            "value_json": pl.String,
        }
        return temporal_frame(self.recorded_variables, schema=schema, orient="row")

    def to_recorded_records(self) -> list[dict[str, Any]]:
        """Return decoded strategy values grouped by their actual bar observation."""
        records: dict[int, dict[str, Any]] = {}
        for row in temporal_rows(self.to_recorded_dataframe(), named=True):
            record = records.setdefault(
                row["bar_index"],
                {"timestamp": row["timestamp"], "bar_index": row["bar_index"], "values": {}},
            )
            record["values"][row["name"]] = _decode_recorded_value(row["value_json"])
        return list(records.values())

    @staticmethod
    def _holdings_schema() -> dict[str, pl.DataType]:
        return {
            "timestamp": pl.Datetime(),
            "observation_id": pl.UInt32(),
            "asset": pl.String(),
            "quantity": pl.Float64(),
            "mark": pl.Float64(),
            "multiplier": pl.Float64(),
            "currency": pl.String(),
            "signed_notional": pl.Float64(),
            "liquidation_value": pl.Float64(),
            "mark_source": pl.String(),
            "mark_timestamp": pl.Datetime(),
        }

    def _session_pnl(self, calendar: str) -> pl.DataFrame:
        from .sessions import SessionConfig, compute_session_pnl

        session_config = SessionConfig(
            calendar=calendar,
            timezone=self.config.resolved_timezone if self.config else "UTC",
            session_start_time=self.config.resolved_session_start_time if self.config else None,
            timestamp_semantics=self.config.resolved_timestamp_semantics if self.config else None,
            bar_label=self.config.resolved_feed_spec.bar_label if self.config else None,
            data_frequency=self.config.resolved_data_frequency if self.config else None,
        )
        daily = compute_session_pnl(
            self.equity_curve,
            session_config,
            initial_capital=self.initial_capital,
            drawdown_values=self.to_equity_dataframe()["performance_equity"].to_list(),
        )
        return self._adjust_flow_pnl(daily, session_config)

    def _adjust_flow_pnl(self, daily: pl.DataFrame, session_config=None) -> pl.DataFrame:
        if not self.capital_changes or daily.is_empty():
            return daily

        def key(timestamp):
            return (
                timestamp.date()
                if session_config is None
                else session_config.date_for_timestamp(timestamp).date()
            )

        factors: dict[Any, float] = {}
        external: dict[Any, float] = {}
        for (timestamp, _), value in zip(self.equity_curve, self.to_returns_series()):
            day = key(timestamp)
            factors[day] = factors.get(day, 1.0) * (1 + value)
        for change in self.capital_changes:
            day = key(change.timestamp)
            external[day] = external.get(day, 0.0) + change.cash_delta
        dates = daily["date"] if session_config is None else daily["session_date"].dt.date()
        returns = pl.Series("return_pct", [factors[day] - 1.0 for day in dates])
        flows = pl.Series("external_flow", [external.get(day, 0.0) for day in dates])
        return daily.with_columns(
            returns,
            ((returns + 1).cum_prod() - 1).alias("cumulative_return"),
            (daily["pnl"] - flows).alias("pnl"),
            flows,
        )

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
            return temporal_frame(
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
            return self._session_pnl(self.config.resolved_calendar)

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
        if self.initial_capital is not None:
            daily = daily.with_columns(
                pl.col("close_equity").shift(1).fill_null(self.initial_capital).alias("open_equity")
            )

        # Compute daily P&L and returns
        daily = daily.with_columns(
            [
                (pl.col("close_equity") - pl.col("open_equity")).alias("pnl"),
            ]
        )

        # Return percent (handle first day)
        prev_close = daily.select(pl.col("close_equity").shift(1)).to_series()
        if self.initial_capital is not None:
            prev_close = prev_close.fill_null(self.initial_capital)
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

        return self._adjust_flow_pnl(daily)

    def to_daily_returns(
        self,
        calendar: str | None | CalendarUnset = UNSET_CALENDAR,
        session_aligned: bool | None = None,
    ) -> pl.Series:
        """Get daily returns as Polars Series for ml4t-diagnostic integration.

        Omission or a named calendar aggregates bar-level equity to daily returns.
        Explicit None instead retains observation-level returns;
        use the matching declared annualization scale for downstream metrics.

        Args:
            calendar: Trading calendar for context. If provided and known,
                enables session-aware aggregation. Common values:
                - "crypto": 365 days/year (24/7)
                - "NYSE", "NASDAQ": 252 days/year
                - "CME_Equity", etc: Uses pandas_market_calendars
                Omission resolves the feed calendar, then NYSE. Explicit None returns
                one return per observation without daily aggregation.
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
        resolver = self._feed_spec() or FeedSpec(time_contract_version=2)
        cal = resolver.resolve_calendar(calendar=calendar)
        if cal is None:
            if session_aligned:
                raise ValueError("session_aligned=True requires a calendar")
            return self.to_returns_series().alias("daily_return")
        if session_aligned is None:
            session_aligned = self._auto_session_aligned(cal)

        # Get daily P&L DataFrame
        daily_df = (
            self._session_pnl(cal)
            if session_aligned and cal is not None
            else self.to_daily_pnl(session_aligned=False)
        )

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
                "funding_payments": self.funding_payments,
                "portfolio_state": self.portfolio_state,
                "initial_capital": self.initial_capital,
                "account_method": self.account_method,
                "instrument_accounting": self.instrument_accounting,
                "accounting_observations": self.accounting_observations,
                "futures_settlements": self.futures_settlements,
                "futures_rolls": self.futures_rolls,
                "holdings": self.holdings,
                "holdings_sampling": self.holdings_sampling,
                "observation_metadata": self.observation_metadata,
                "capital_changes": self.capital_changes,
                "recorded_variables": self.recorded_variables,
                "returns_include_commissions": self.returns_include_commissions,
            }
        )
        if self.predictions is not None:
            result["predictions"] = self.predictions
        if self.input_provenance is not None:
            result["input_provenance"] = self.input_provenance
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
            "initial_capital": self.initial_capital,
            "account_method": self.account_method,
            "instrument_accounting": self.instrument_accounting,
            "returns_include_commissions": self.returns_include_commissions,
            "lifecycle_version": self.metrics.get("lifecycle_version"),
            "lifecycle_callback_counts": self.metrics.get("lifecycle_callback_counts", {}),
            "lifecycle_invocations": self.metrics.get("lifecycle_invocations", []),
            "execution_policy": self.metrics.get("execution_policy"),
            "target_intent_count": self.metrics.get("target_intent_count", 0),
            "child_order_intent_count": self.metrics.get("child_order_intent_count", 0),
            "intent_reconciliation_count": self.metrics.get("intent_reconciliation_count", 0),
            "target_rule_reconciliation_count": self.metrics.get(
                "target_rule_reconciliation_count", 0
            ),
            "target_intents": self.metrics.get("target_intents", []),
            "child_order_intents": self.metrics.get("child_order_intents", []),
            "intent_reconciliations": self.metrics.get("intent_reconciliations", []),
            "target_rule_reconciliations": self.metrics.get("target_rule_reconciliations", []),
            "config": config_dict,
            "window": {
                "start": start,
                "end": end,
            },
        }

    # Dict-like access keeps validation scripts and older notebook code working.
    def __getitem__(self, key: str) -> Any:
        """Return a metric or raw result component, raising KeyError if absent."""
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Return a metric or raw component, or default when the key is absent."""
        return self.to_dict().get(key, default)

    def keys(self) -> KeysView[str]:
        """Return keys from the backward-compatible dictionary representation."""
        return self.to_dict().keys()

    def items(self) -> ItemsView[str, Any]:
        """Return items from the backward-compatible dictionary representation."""
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
                funding.parquet
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
                Options: ["trades", "fills", "funding", "rejected_orders", "predictions", "equity",
                    "portfolio_state", "daily_pnl", "metrics", "config", "spec"]
            compression: Parquet compression codec (default: "zstd")

        Returns:
            Dict mapping requested component names to file paths. The always-written
            manifest is returned under the additional ``"manifest"`` key; it is not
            a selectable component.

        Raises:
            ArtifactWriteError: If a requested component is unavailable or cannot be written.
        """
        if self.input_provenance is not None:
            try:
                validate_input_provenance(self.input_provenance)
            except (TypeError, ValueError, KeyError) as exc:
                raise ArtifactWriteError(f"Invalid input provenance: {exc}") from exc
        explicitly_selected = include is not None
        if self.account_method not in (None, "signed_notional", "per_instrument"):
            raise ArtifactWriteError("Unknown account_method")
        if (self.account_method == "per_instrument") != (self.instrument_accounting is not None):
            raise ArtifactWriteError("Per-instrument account requires instrument declarations")
        if self.holdings_sampling not in ("none", "bar", "session", "fill_and_session"):
            raise ArtifactWriteError("Unknown holdings_sampling")
        if self.holdings is not None:
            _validate_complete_holdings(
                self.to_portfolio_state_dataframe(),
                self.to_holdings_dataframe(),
                ArtifactWriteError,
            )
        requested = list(include) if include is not None else list(_COMPONENT_FILES)
        unknown = sorted(set(requested) - _COMPONENT_FILES.keys() - {"manifest"})
        if unknown:
            raise ArtifactWriteError(f"Unknown artifact components requested: {unknown}")
        requested = [name for name in requested if name != "manifest"]

        unavailable: dict[str, str] = {}
        if self.holdings is None:
            unavailable["holdings"] = "holdings were not captured"
        if self.recorded_variables is None:
            unavailable["recorded_variables"] = "strategy variables were not recorded"
        if self.predictions is None:
            unavailable["predictions"] = "result has no predictions"
        if self.benchmark is None:
            unavailable["benchmark"] = "result has no prepared benchmark"
        if self.corporate_actions is None:
            unavailable["corporate_actions"] = "corporate-action applications were not captured"
        if self.corporate_entitlements is None:
            unavailable["corporate_entitlements"] = "corporate entitlements were not captured"
        if self.futures_settlements is None:
            unavailable["futures_settlements"] = "futures variation was not captured"
        if self.futures_rolls is None:
            unavailable["futures_rolls"] = "futures roll intents were not captured"
        if self.config is None:
            unavailable["config"] = "result has no config"
            unavailable["spec"] = "result has no config for a runtime spec"

        explicitly_unavailable = sorted(set(requested) & unavailable.keys())
        if explicitly_selected and explicitly_unavailable:
            details = ", ".join(f"{name}: {unavailable[name]}" for name in explicitly_unavailable)
            raise ArtifactWriteError(f"Requested artifact components are unavailable: {details}")

        selected = [name for name in requested if name not in unavailable]
        rolls_hash = None
        if self.futures_rolls is not None:
            try:
                rolls = self.to_rolls_dataframe()
                validate_rolls(rolls, self.fills)
                rolls_hash = roll_digest(rolls)
            except (TypeError, ValueError, KeyError) as exc:
                raise ArtifactWriteError(f"Invalid futures roll ledger: {exc}") from exc
        futures_hash = None
        if self.instrument_accounting is not None:
            try:
                settlement_table = self.to_futures_settlements_dataframe()
                account_table = self.to_portfolio_state_dataframe()
                validate_futures_tables(
                    self.instrument_accounting, account_table, settlement_table, self.fills
                )
                futures_hash = sha256(
                    futures_payload(
                        self.instrument_accounting, account_table, settlement_table
                    ).encode()
                ).hexdigest()
            except (TypeError, ValueError, KeyError) as exc:
                raise ArtifactWriteError(f"Invalid futures variation ledger: {exc}") from exc
        elif self.futures_settlements is not None or self.accounting_observations is not None:
            raise ArtifactWriteError("Futures observations require per-instrument declarations")
        corporate_hash = None
        if self.corporate_actions is not None:
            try:
                if self.corporate_receivables is None:
                    raise ValueError("Corporate actions require observed receivables")
                actions_frame, entitlements_frame = (
                    self.to_actions_dataframe(),
                    self.to_entitlements_dataframe(),
                )
                validate_action_tables(
                    actions_frame,
                    entitlements_frame,
                    self.to_portfolio_state_dataframe(),
                    self.corporate_receivables,
                    account_method=self.account_method,
                )
                corporate_hash = action_digest(
                    actions_frame, entitlements_frame, self.corporate_receivables
                )
            except (TypeError, ValueError) as exc:
                raise ArtifactWriteError(f"Invalid corporate-action ledger: {exc}") from exc
        benchmark_frame = None
        if self.benchmark is not None:
            try:
                benchmark_frame = self.to_benchmark_dataframe()
                json.dumps(self.benchmark_metadata, allow_nan=False)
            except Exception as exc:
                raise ArtifactWriteError(f"Invalid prepared benchmark: {exc}") from exc

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
            config = self.config
            if config is None:
                raise ArtifactWriteError("Config and spec components require a runtime config")
            try:
                import yaml
            except ImportError as exc:
                raise ArtifactWriteError("PyYAML is required to serialize config or spec") from exc
            if "config" in selected:
                try:
                    text_payloads["config"] = yaml.safe_dump(
                        config.to_dict(),
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

        if "funding" in selected:
            funding_path = path / "funding.parquet"
            write_component(
                "funding",
                lambda: self.to_funding_dataframe().write_parquet(
                    funding_path,
                    compression=compression,
                ),
            )
            written["funding"] = funding_path

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

        if "holdings" in selected:
            holdings_path = path / _COMPONENT_FILES["holdings"]
            write_component(
                "holdings",
                lambda: self.to_holdings_dataframe().write_parquet(
                    holdings_path, compression=compression
                ),
            )
            written["holdings"] = holdings_path

        if "capital_flows" in selected:
            flows_path = path / _COMPONENT_FILES["capital_flows"]
            write_component(
                "capital_flows",
                lambda: self.to_capital_flows_dataframe().write_parquet(
                    flows_path, compression=compression
                ),
            )
            written["capital_flows"] = flows_path

        if "recorded_variables" in selected:
            recorded_path = path / _COMPONENT_FILES["recorded_variables"]
            write_component(
                "recorded_variables",
                lambda: self.to_recorded_dataframe().write_parquet(
                    recorded_path, compression=compression
                ),
            )
            written["recorded_variables"] = recorded_path

        if "benchmark" in selected:
            benchmark_path = path / _COMPONENT_FILES["benchmark"]
            assert benchmark_frame is not None
            write_component(
                "benchmark",
                lambda: benchmark_frame.write_parquet(benchmark_path, compression=compression),
            )
            written["benchmark"] = benchmark_path

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

        for component in (
            "corporate_actions",
            "corporate_entitlements",
            "futures_settlements",
            "futures_rolls",
        ):
            if component in selected:
                target = path / _COMPONENT_FILES[component]
                frame = (
                    self.to_actions_dataframe()
                    if component == "corporate_actions"
                    else (
                        self.to_entitlements_dataframe()
                        if component == "corporate_entitlements"
                        else (
                            self.to_futures_settlements_dataframe()
                            if component == "futures_settlements"
                            else self.to_rolls_dataframe()
                        )
                    )
                )
                write_component(
                    component,
                    lambda target=target, frame=frame: frame.write_parquet(
                        target, compression=compression
                    ),
                )
                written[component] = target

        manifest = {
            "artifact_type": _ARTIFACT_TYPE,
            "schema_version": _ROLL_ARTIFACT_SCHEMA_VERSION
            if self.futures_rolls is not None
            else (
                _FUTURES_ARTIFACT_SCHEMA_VERSION
                if self.instrument_accounting is not None
                else (
                    _ACTION_ARTIFACT_SCHEMA_VERSION
                    if self.corporate_actions is not None
                    else (
                        _BENCHMARK_ARTIFACT_SCHEMA_VERSION
                        if self.benchmark is not None
                        else _ARTIFACT_SCHEMA_VERSION
                    )
                )
            ),
            **({"roll_content_sha256": rolls_hash} if rolls_hash is not None else {}),
            **(
                {
                    "instrument_accounting": self.instrument_accounting,
                    "futures_content_sha256": futures_hash,
                }
                if futures_hash is not None
                else {}
            ),
            **(
                {"corporate_action_content_sha256": corporate_hash}
                if corporate_hash is not None
                else {}
            ),
            **(
                {
                    "benchmark_metadata": self.benchmark_metadata,
                    "benchmark_content_sha256": _benchmark_digest(benchmark_frame),
                }
                if benchmark_frame is not None
                else {}
            ),
            **(
                {
                    "input_provenance": self.input_provenance,
                    "input_provenance_content_sha256": provenance_digest(self.input_provenance),
                }
                if self.input_provenance is not None
                else {}
            ),
            "library_version": __version__,
            "account_method": self.account_method,
            "initial_capital": self.initial_capital,
            "holdings_sampling": self.holdings_sampling,
            "recorded_sampling": "bar_end" if self.recorded_variables is not None else None,
            "returns_include_commissions": self.returns_include_commissions,
            "complete": written.keys()
            >= (
                _REQUIRED_RESULT_COMPONENTS
                | ({"funding"} if self.funding_payments else set())
                | ({"capital_flows"} if self.capital_changes else set())
                | ({"holdings"} if self.holdings_sampling != "none" else set())
                | ({"recorded_variables"} if self.recorded_variables is not None else set())
                | ({"benchmark"} if self.benchmark is not None else set())
                | ({"futures_settlements"} if self.instrument_accounting is not None else set())
                | ({"futures_rolls"} if self.futures_rolls is not None else set())
                | (
                    {"corporate_actions", "corporate_entitlements"}
                    if self.corporate_actions is not None
                    else set()
                )
            ),
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
            if schema_version not in (
                2,
                _ARTIFACT_SCHEMA_VERSION,
                _BENCHMARK_ARTIFACT_SCHEMA_VERSION,
                _ACTION_ARTIFACT_SCHEMA_VERSION,
                _FUTURES_ARTIFACT_SCHEMA_VERSION,
                _ROLL_ARTIFACT_SCHEMA_VERSION,
            ):
                raise UnsupportedArtifactVersionError(
                    f"Unsupported result artifact schema version {schema_version!r}; "
                    "supported versions are 2, 3, 4, 5, 6 and 7"
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

        input_provenance = manifest.get("input_provenance") if manifest else None
        input_digest = manifest.get("input_provenance_content_sha256") if manifest else None
        if input_provenance is not None:
            try:
                validate_input_provenance(input_provenance)
                if provenance_digest(input_provenance) != input_digest:
                    raise ValueError(
                        "Input provenance does not match its retained content identity"
                    )
            except (TypeError, ValueError, KeyError) as exc:
                raise ArtifactManifestError(f"Invalid input provenance: {exc}") from exc
        elif input_digest is not None:
            raise ArtifactManifestError("Input provenance declaration is missing")
        initial_capital = manifest.get("initial_capital") if manifest is not None else None
        account_method = manifest.get("account_method") if manifest else None
        if account_method not in (None, "signed_notional", "per_instrument"):
            raise ArtifactManifestError("Unknown account_method")
        rolls_declared = (
            manifest is not None and manifest.get("schema_version") == _ROLL_ARTIFACT_SCHEMA_VERSION
        )
        futures_declared = manifest is not None and (
            manifest.get("schema_version") == _FUTURES_ARTIFACT_SCHEMA_VERSION
            or rolls_declared
            and account_method == "per_instrument"
        )
        instrument_accounting = manifest.get("instrument_accounting") if manifest else None
        if futures_declared:
            try:
                validate_instrument_accounting(instrument_accounting)
                if account_method != "per_instrument":
                    raise ValueError("Futures artifacts require per-instrument accounting")
            except ValueError as exc:
                raise ArtifactManifestError(str(exc)) from exc
        elif account_method == "per_instrument" or instrument_accounting is not None:
            raise ArtifactManifestError("Per-instrument accounting requires schema version 6 or 7")
        returns_include_commissions = (
            manifest.get("returns_include_commissions") if manifest else None
        )
        if returns_include_commissions is not None and not isinstance(
            returns_include_commissions, bool
        ):
            raise ArtifactManifestError("returns_include_commissions must be a boolean or null")
        if initial_capital is not None and (
            isinstance(initial_capital, bool)
            or not isinstance(initial_capital, (int, float))
            or not math.isfinite(initial_capital)
            or initial_capital <= 0
        ):
            raise ArtifactManifestError("initial_capital must be a finite positive number or null")
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

        benchmark_metadata = manifest.get("benchmark_metadata") if manifest else None
        benchmark_declared = benchmark_metadata is not None or (
            manifest is not None
            and manifest.get("schema_version") == _BENCHMARK_ARTIFACT_SCHEMA_VERSION
        )
        if benchmark_declared and benchmark_metadata is None and not recovery:
            raise ArtifactManifestError("Benchmark artifact has no provenance declaration")
        required_components = (
            _REQUIRED_RESULT_COMPONENTS
            | ({"benchmark"} if benchmark_declared else set())
            | ({"futures_settlements"} if futures_declared else set())
        ) | ({"futures_rolls"} if rolls_declared else set())
        actions_declared = manifest is not None and (
            manifest.get("schema_version") == _ACTION_ARTIFACT_SCHEMA_VERSION
            or manifest.get("schema_version")
            in {_FUTURES_ARTIFACT_SCHEMA_VERSION, _ROLL_ARTIFACT_SCHEMA_VERSION}
            and manifest.get("corporate_action_content_sha256") is not None
        )
        if actions_declared:
            required_components |= {"corporate_actions", "corporate_entitlements"}
        elif {"corporate_actions", "corporate_entitlements"} & components.keys():
            raise ArtifactManifestError("Corporate actions require schema version 5 provenance")
        if not futures_declared and "futures_settlements" in components:
            raise ArtifactManifestError("Futures settlements require schema version 6 provenance")
        if not rolls_declared and (
            "futures_rolls" in components or manifest and "roll_content_sha256" in manifest
        ):
            raise ArtifactManifestError("Futures rolls require schema version 7 provenance")
        missing_required_components = sorted(required_components - components.keys())
        missing_files = sorted(
            name for name, filename in components.items() if not (path / filename).is_file()
        )
        if not recovery and (missing_required_components or missing_files):
            raise ArtifactIncompleteError(
                "Result artifact is incomplete: "
                f"missing components={missing_required_components}, missing files={missing_files}"
            )
        if recovery:
            missing_components = sorted(
                (
                    _COMPONENT_FILES.keys()
                    - {
                        "funding",
                        "holdings",
                        "capital_flows",
                        "recorded_variables",
                        "benchmark",
                        "corporate_actions",
                        "corporate_entitlements",
                        "futures_settlements",
                        "futures_rolls",
                    }
                )
                - components.keys()
            )
            diagnostics.extend(
                ArtifactDiagnostic(
                    code="component_missing",
                    component=name,
                    message=f"Component {_COMPONENT_FILES[name]} is absent.",
                )
                for name in missing_components
            )
            if benchmark_declared and "benchmark" not in components:
                diagnostics.append(
                    ArtifactDiagnostic(
                        "component_missing", "benchmark", "Declared benchmark.parquet is absent."
                    )
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

        component_read_ok: dict[str, bool] = {}

        def read_component(name: str, reader, default):
            filename = components.get(name)
            if filename is None:
                component_read_ok[name] = False
                return default
            try:
                value = reader(path / filename)
                component_read_ok[name] = True
                return value
            except Exception as exc:
                component_read_ok[name] = False
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
            for row in temporal_rows(pl.read_parquet(component_path), named=True):
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
            frame = pl.read_parquet(component_path)
            if "fill_sequence" in frame.columns:
                sequence = frame["fill_sequence"]
                if not sequence.dtype.is_integer() or sorted(sequence.to_list()) != list(
                    range(len(frame))
                ):
                    raise ValueError("fill_sequence must uniquely order every fill from zero")
                frame = frame.sort("fill_sequence")
            for row in temporal_rows(frame, named=True):
                triggered_at = row.get("stop_triggered_at")
                if triggered_at is not None and (
                    not isinstance(triggered_at, datetime) or triggered_at > row["timestamp"]
                ):
                    raise ValueError("Stop trigger instant must not follow its fill")
                result.append(
                    Fill(
                        order_id=row["order_id"],
                        rebalance_id=row.get("rebalance_id"),
                        asset=row["asset"],
                        side=OrderSide(row["side"]),
                        quantity=row["quantity"],
                        multiplier=row.get("multiplier"),
                        currency=row.get("currency"),
                        price=row["price"],
                        timestamp=row["timestamp"],
                        commission=row.get("commission", 0.0),
                        slippage=row.get("slippage", 0.0),
                        order_type=row.get("order_type", ""),
                        limit_price=row.get("limit_price"),
                        stop_price=row.get("stop_price"),
                        stop_triggered_at=row.get("stop_triggered_at"),
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

        def read_funding(component_path: Path) -> list[FundingPayment]:
            return [
                FundingPayment(
                    timestamp=row["timestamp"],
                    asset=row["asset"],
                    quantity=row["quantity"],
                    mark_price=row["mark_price"],
                    multiplier=row["multiplier"],
                    rate=row["rate"],
                    amount_per_unit=row["amount_per_unit"],
                    cash_delta=row["cash_delta"],
                )
                for row in temporal_rows(pl.read_parquet(component_path), named=True)
            ]

        def read_rejected_orders(component_path: Path) -> list[Order]:
            result: list[Order] = []
            for row in temporal_rows(pl.read_parquet(component_path), named=True):
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
                for row in temporal_rows(pl.read_parquet(component_path), named=True)
            ]

        observation_metadata: list[tuple[str, bool, int | None]] = []
        corporate_receivables: list[float] = []
        accounting_observations: list[AccountControlContext] = []

        def read_portfolio_state(
            component_path: Path,
        ) -> list[tuple[datetime, float, float, float, float, int]]:
            frame = pl.read_parquet(component_path)
            if futures_declared:
                if not set(ACCOUNTING_COLUMNS) <= set(frame.columns):
                    raise ValueError("Variation artifact omits actual margin observations")
                accounting_observations.extend(
                    AccountControlContext(
                        row["timestamp"],
                        row["equity"],
                        row["cash"],
                        row["gross_exposure"],
                        row["net_exposure"],
                        **{name: row[name] for name in ACCOUNTING_COLUMNS},
                    )
                    for row in temporal_rows(frame, named=True)
                )
            elif set(ACCOUNTING_COLUMNS) & set(frame.columns):
                raise ValueError("Variation observations require schema version 6 provenance")
            if "receivables_value" in frame.columns:
                if not actions_declared:
                    raise ValueError(
                        "Observed receivables require a declared corporate-action ledger"
                    )
                corporate_receivables.extend(frame["receivables_value"].to_list())
            elif actions_declared:
                raise ValueError("Corporate actions require actual observed receivables")
            metadata_columns = {"observation_kind", "holdings_complete", "fill_sequence"}
            if metadata_columns & set(frame.columns):
                if not metadata_columns <= set(frame.columns):
                    raise ValueError("Account observation metadata is incomplete")
                metadata = []
                for row in temporal_rows(frame, named=True):
                    kind, complete, sequence = (
                        row["observation_kind"],
                        row["holdings_complete"],
                        row["fill_sequence"],
                    )
                    if kind not in {"bar", "after_fill", "session_last_observed"} or not isinstance(
                        complete, bool
                    ):
                        raise ValueError("Account observation metadata is invalid")
                    if kind == "after_fill":
                        if (
                            not isinstance(sequence, int)
                            or not 0 <= sequence < len(fills)
                            or fills[sequence].timestamp != row["timestamp"]
                        ):
                            raise ValueError(
                                "Account fill observation must refer to its actual fill"
                            )
                    elif sequence is not None:
                        raise ValueError("Only fill observations may identify a fill sequence")
                    metadata.append((kind, complete, sequence))
                observation_metadata.extend(metadata)
            return [
                (
                    row["timestamp"],
                    row["equity"],
                    row["cash"],
                    row["gross_exposure"],
                    row["net_exposure"],
                    row["open_positions"],
                )
                for row in temporal_rows(frame, named=True)
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
        funding_payments = read_component("funding", read_funding, [])
        flows_frame = read_component("capital_flows", pl.read_parquet, None)
        capital_changes = []
        if flows_frame is not None:
            try:
                if "flow_sequence" not in flows_frame.columns:
                    raise ValueError("flow_sequence is required")
                sequence = flows_frame["flow_sequence"]
                if not sequence.dtype.is_integer() or sorted(sequence.to_list()) != list(
                    range(len(flows_frame))
                ):
                    raise ValueError("flow_sequence must uniquely order every capital change")
                for row in temporal_rows(
                    flows_frame.sort("flow_sequence").drop("flow_sequence"), named=True
                ):
                    capital_changes.append(CapitalChange(**row))
            except Exception as exc:
                raise ArtifactReadError("Invalid capital_flows.parquet") from exc
        rejected_orders = read_component("rejected_orders", read_rejected_orders, [])
        equity_curve = read_component("equity", read_equity, [])

        def read_benchmark(component_path):
            frame = _validate_benchmark_frame(pl.read_parquet(component_path))
            if manifest is None or manifest.get("schema_version") not in (
                _BENCHMARK_ARTIFACT_SCHEMA_VERSION,
                _ACTION_ARTIFACT_SCHEMA_VERSION,
                _FUTURES_ARTIFACT_SCHEMA_VERSION,
                _ROLL_ARTIFACT_SCHEMA_VERSION,
            ):
                raise ValueError("Prepared benchmark requires artifact schema version 4, 5, 6 or 7")
            if _benchmark_digest(frame) != manifest.get("benchmark_content_sha256"):
                raise ValueError("Prepared benchmark content does not match its retained digest")
            probe = cls(
                trades=[],
                fills=[],
                metrics={},
                equity_curve=equity_curve,
                config=config,
                benchmark=frame,
                benchmark_metadata=benchmark_metadata,
            )
            return probe.to_benchmark_dataframe()

        portfolio_state = read_component("portfolio_state", read_portfolio_state, [])
        holdings_frame = read_component("holdings", pl.read_parquet, None)
        holdings = None
        holdings_sampling = manifest.get("holdings_sampling", "none") if manifest else "none"
        if holdings_sampling not in ("none", "bar", "session", "fill_and_session"):
            raise ArtifactManifestError("Unknown holdings_sampling declaration")
        if holdings_sampling != "none" and holdings_frame is None:
            if not recovery:
                raise ArtifactReadError("declared holdings sampling requires holdings.parquet")
            diagnostics.append(
                ArtifactDiagnostic(
                    "component_unavailable", "holdings", "Declared holdings could not be loaded."
                )
            )
            holdings_sampling = "none"
        if holdings_frame is not None:
            if holdings_sampling == "none":
                raise ArtifactManifestError(
                    "holdings component requires a declared sampling method"
                )
            for name, dtype in {"mark_source": pl.String, "mark_timestamp": pl.Datetime}.items():
                if name not in holdings_frame.columns:
                    holdings_frame = holdings_frame.with_columns(
                        pl.lit(None, dtype=dtype).alias(name)
                    )
            required = set(cls._holdings_schema())
            if not required <= set(holdings_frame.columns):
                raise ArtifactReadError("holdings.parquet is missing required fields")
            if holdings_frame.select(
                pl.struct("observation_id", "asset").is_duplicated().any()
            ).item():
                raise ArtifactReadError("holdings observation/asset keys must be unique")
            for row in temporal_rows(holdings_frame, named=True):
                observation = row["observation_id"]
                if (
                    not isinstance(observation, int)
                    or not 0 <= observation < len(portfolio_state)
                    or row["timestamp"] != portfolio_state[observation][0]
                    or not isinstance(row["asset"], str)
                    or not row["asset"]
                ):
                    raise ArtifactReadError(
                        "holdings keys must refer to actual account observations"
                    )
                if (
                    any(
                        not isinstance(row[name], (int, float)) or not math.isfinite(row[name])
                        for name in (
                            "quantity",
                            "mark",
                            "multiplier",
                            "signed_notional",
                            "liquidation_value",
                        )
                    )
                    or row["multiplier"] <= 0
                ):
                    raise ArtifactReadError(
                        "holdings quantities and values must be finite with a positive multiplier"
                    )
            holdings = list(temporal_rows(holdings_frame.select(list(cls._holdings_schema()))))
        recorded_frame = read_component("recorded_variables", pl.read_parquet, None)
        recorded_variables = None
        recorded_sampling = manifest.get("recorded_sampling") if manifest else None
        if recorded_sampling not in (None, "bar_end"):
            raise ArtifactManifestError("Unknown recorded-variable sampling declaration")
        if recorded_frame is not None:
            if recorded_sampling != "bar_end":
                raise ArtifactManifestError("Recorded variables require declared bar-end sampling")
            required = ["timestamp", "bar_index", "name", "value_json"]
            if not set(required) <= set(recorded_frame.columns):
                raise ArtifactReadError("Recorded variables are missing required fields")
            if recorded_frame.select(pl.struct("bar_index", "name").is_duplicated().any()).item():
                raise ArtifactReadError("Recorded variable keys must be unique")
            for row in temporal_rows(recorded_frame, named=True):
                index = row["bar_index"]
                if (
                    not isinstance(index, int)
                    or not 0 <= index < len(equity_curve)
                    or equity_curve[index][0] != row["timestamp"]
                    or not isinstance(row["name"], str)
                    or not row["name"]
                ):
                    raise ArtifactReadError(
                        "Recorded variables must refer to actual bar observations"
                    )
                try:
                    _decode_recorded_value(row["value_json"])
                except (TypeError, ValueError) as exc:
                    raise ArtifactReadError(
                        "Recorded values must contain valid tagged JSON"
                    ) from exc
            recorded_variables = list(temporal_rows(recorded_frame.select(required)))
        elif recorded_sampling == "bar_end":
            if not recovery:
                raise ArtifactReadError("Declared strategy variables are missing")
            diagnostics.append(
                ArtifactDiagnostic(
                    "component_unavailable",
                    "recorded_variables",
                    "Declared strategy variables could not be loaded.",
                )
            )
        metrics = read_component("metrics", read_metrics, {})
        if metrics.get("num_capital_changes", 0) and flows_frame is None:
            raise ArtifactReadError(
                "capital_flows.parquet is required for an artifact with external flows"
            )
        if capital_changes and (
            metrics.get("num_capital_changes") != len(capital_changes)
            or metrics.get("total_external_flow")
            != sum(change.cash_delta for change in capital_changes)
            or len({change.event_id for change in capital_changes}) != len(capital_changes)
            or any(
                change.kind not in {"delta", "target"}
                or not isinstance(change.event_id, str)
                or not change.event_id
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in (
                        change.value,
                        change.cash_delta,
                        change.nav_before,
                        change.nav_after,
                    )
                )
                or not math.isclose(
                    change.nav_before + change.cash_delta, change.nav_after, rel_tol=1e-12
                )
                for change in capital_changes
            )
        ):
            raise ArtifactReadError(
                "capital flows are inconsistent with recorded valuations or metrics"
            )
        if metrics.get("num_funding_events", 0) and not component_read_ok["funding"]:
            message = "funding.parquet is required for an artifact with funding events"
            if not recovery:
                raise ArtifactReadError(message)
            diagnostics.append(ArtifactDiagnostic("component_missing", "funding", message))
        if component_read_ok["funding"] and "num_funding_events" in metrics:
            reported_count = metrics["num_funding_events"]
            reported_total = metrics.get("total_funding")
            actual_total = sum(payment.cash_delta for payment in funding_payments)
            if (
                reported_count != len(funding_payments)
                or not isinstance(reported_total, (int, float))
                or not math.isclose(reported_total, actual_total, abs_tol=1e-9)
            ):
                message = "funding.parquet is inconsistent with funding metrics"
                if not recovery:
                    raise ArtifactReadError(message)
                diagnostics.append(ArtifactDiagnostic("component_inconsistent", "funding", message))
        predictions = read_component("predictions", pl.read_parquet, None)
        daily_pnl = read_component("daily_pnl", pl.read_parquet, None)
        config = read_component("config", read_config, None)
        spec_config = read_component("spec", read_spec_config, None)
        if config is None:
            config = spec_config

        benchmark = read_component("benchmark", read_benchmark, None)

        if daily_pnl is not None:
            if not component_read_ok["equity"]:
                diagnostics.append(
                    ArtifactDiagnostic(
                        code="component_unverified",
                        component="daily_pnl",
                        message="daily_pnl.parquet could not be verified without equity.parquet",
                    )
                )
            else:
                expected_daily_pnl = cls(
                    trades=[],
                    equity_curve=equity_curve,
                    fills=[],
                    metrics={},
                    initial_capital=initial_capital,
                    capital_changes=capital_changes,
                ).to_daily_pnl()
            if component_read_ok["equity"] and not daily_pnl.equals(expected_daily_pnl):
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

        corporate_actions = corporate_entitlements = None
        if actions_declared:
            actions_table = read_component("corporate_actions", pl.read_parquet, None)
            entitlements_table = read_component("corporate_entitlements", pl.read_parquet, None)
            if actions_table is None or entitlements_table is None:
                raise ArtifactReadError("Declared corporate-action ledger components are missing")
            corporate_actions = list(temporal_rows(actions_table, named=True))
            corporate_entitlements = list(temporal_rows(entitlements_table, named=True))

        futures_settlements = None
        if futures_declared:
            settlement_table = read_component("futures_settlements", pl.read_parquet, None)
            if settlement_table is None:
                raise ArtifactReadError("Declared futures variation ledger is missing")
            futures_settlements = list(temporal_rows(settlement_table, named=True))
        futures_rolls = None
        if rolls_declared:
            rolls_table = read_component("futures_rolls", pl.read_parquet, None)
            if rolls_table is None:
                raise ArtifactReadError("Declared futures roll ledger is missing")
            try:
                validate_rolls(rolls_table, fills)
                if manifest is None or roll_digest(rolls_table) != manifest.get(
                    "roll_content_sha256"
                ):
                    raise ValueError("Roll content identity does not match its manifest")
            except (TypeError, ValueError, KeyError) as exc:
                raise ArtifactReadError(f"Invalid futures roll ledger: {exc}") from exc
            futures_rolls = list(temporal_rows(rolls_table, named=True))

        result = cls(
            trades=trades,
            equity_curve=equity_curve,
            fills=fills,
            predictions=predictions,
            portfolio_state=portfolio_state,
            rejected_orders=rejected_orders,
            funding_payments=funding_payments,
            metrics=metrics,
            config=config,
            artifact_diagnostics=tuple(diagnostics),
            initial_capital=initial_capital,
            holdings=holdings,
            holdings_sampling=holdings_sampling,
            capital_changes=capital_changes,
            returns_include_commissions=returns_include_commissions,
            observation_metadata=observation_metadata,
            recorded_variables=recorded_variables,
            account_method=account_method,
            instrument_accounting=instrument_accounting,
            accounting_observations=accounting_observations if futures_declared else None,
            futures_settlements=futures_settlements,
            futures_rolls=futures_rolls,
            input_provenance=input_provenance,
            benchmark=benchmark,
            benchmark_metadata=benchmark_metadata if benchmark is not None else None,
            corporate_actions=corporate_actions,
            corporate_entitlements=corporate_entitlements,
            corporate_receivables=corporate_receivables if actions_declared else None,
        )
        if actions_declared:
            try:
                actions_table, entitlements_table = (
                    result.to_actions_dataframe(),
                    result.to_entitlements_dataframe(),
                )
                validate_action_tables(
                    actions_table,
                    entitlements_table,
                    result.to_portfolio_state_dataframe(),
                    corporate_receivables,
                    account_method=account_method,
                )
                if manifest is None or action_digest(
                    actions_table, entitlements_table, corporate_receivables
                ) != manifest.get("corporate_action_content_sha256"):
                    raise ValueError(
                        "Corporate-action content identity does not match its manifest"
                    )
            except (TypeError, ValueError, KeyError) as exc:
                raise ArtifactReadError(f"Invalid corporate-action ledger: {exc}") from exc
        if result.holdings is not None:
            if result.holdings_sampling in {"session", "fill_and_session"} and len(
                result.observation_metadata
            ) != len(result.portfolio_state):
                raise ArtifactReadError("Sampled holdings require account observation metadata")
            _validate_complete_holdings(
                result.to_portfolio_state_dataframe(), result.to_holdings_dataframe()
            )
        if futures_declared:
            try:
                accounts, settlements = (
                    result.to_portfolio_state_dataframe(),
                    result.to_futures_settlements_dataframe(),
                )
                validate_futures_tables(instrument_accounting, accounts, settlements, fills)
                if manifest is None or sha256(
                    futures_payload(instrument_accounting, accounts, settlements).encode()
                ).hexdigest() != manifest.get("futures_content_sha256"):
                    raise ValueError("Futures content identity does not match its manifest")
            except (TypeError, ValueError, KeyError) as exc:
                raise ArtifactReadError(f"Invalid futures variation ledger: {exc}") from exc
        return result

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
            "fill_sequence": pl.UInt32(),
            "rebalance_id": pl.String(),
            "asset": pl.String(),
            "side": pl.String(),
            "quantity": pl.Float64(),
            "signed_quantity": pl.Float64(),
            "multiplier": pl.Float64(),
            "currency": pl.String(),
            "price": pl.Float64(),
            "timestamp": pl.Datetime(),
            "commission": pl.Float64(),
            "slippage": pl.Float64(),
            "order_type": pl.String(),
            "limit_price": pl.Float64(),
            "stop_price": pl.Float64(),
            "stop_triggered_at": pl.Datetime(),
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
            "performance_equity": pl.Float64(),
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


def _align_join_time_zones(
    trades_df: pl.DataFrame,
    signals_df: pl.DataFrame,
    timestamp_col: str,
    trade_time_cols: tuple[str, ...] = ("entry_time", "exit_time"),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Put the trade times and the signal timestamps on one time zone before an as-of join.

    ``join_asof`` refuses a naive left key against a time-zone-aware right key, and the two
    sides of this join reach it from different places: trade times come out of the engine,
    which carries the bar clock without its zone, while a signal or prediction panel is read
    from parquet with its zone intact. The join then raises ``datatypes of join keys don't
    match`` and every caller that wraps this in a ``try`` loses its ML columns silently.

    **A naive column here is the same wall clock with the zone dropped, so it is localized
    and never shifted.** Measured 2026-09-15 on ``nasdaq100_microstructure``, whose trades
    carry ``Datetime(us)`` and whose predictions carry ``Datetime(us, UTC)``: reading the
    naive times as UTC matches 703 of 905 trades to an exact ``(symbol, instant)`` row in the
    prediction panel, and reading them as ``America/New_York`` matches 199. The residual 202
    are trades entered between two prediction instants, which is what a backward as-of join
    is for. Shifting instead of localizing would move every trade by the offset and the join
    would still return rows, so this failure would be silent rather than loud.

    Where both sides are aware but disagree, the signals are *converted* to the trades' zone:
    two aware columns already name instants, so converting preserves them and localizing
    would not.
    """
    left_dtype = trades_df.schema.get(trade_time_cols[0])
    right_dtype = signals_df.schema.get(timestamp_col)
    if not isinstance(left_dtype, pl.Datetime) or not isinstance(right_dtype, pl.Datetime):
        return trades_df, signals_df

    left_tz, right_tz = left_dtype.time_zone, right_dtype.time_zone
    if left_tz == right_tz:
        return trades_df, signals_df

    if left_tz is None:
        present = [c for c in trade_time_cols if c in trades_df.columns]
        trades_df = trades_df.with_columns(
            [pl.col(c).dt.replace_time_zone(right_tz) for c in present]
        )
    elif right_tz is None:
        signals_df = signals_df.with_columns(pl.col(timestamp_col).dt.replace_time_zone(left_tz))
    else:
        signals_df = signals_df.with_columns(pl.col(timestamp_col).dt.convert_time_zone(left_tz))
    return trades_df, signals_df


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

    # The two sides reach this join from different places and need not agree on a time zone.
    trades_df, signals_df = _align_join_time_zones(trades_df, signals_df, timestamp_col)

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
