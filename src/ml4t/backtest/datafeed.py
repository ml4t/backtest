"""Polars-based multi-asset data feed with O(1) timestamp lookups."""

from datetime import datetime
from typing import Any

import numpy as np
import polars as pl


class DataFeed:
    """Polars-based multi-asset data feed with signals and context.

    Pre-partitions data by timestamp at initialization for O(1) lookups
    during iteration (vs O(N) filter per bar).
    """

    def __init__(
        self,
        prices_path: str | None = None,
        signals_path: str | None = None,
        context_path: str | None = None,
        prices_df: pl.DataFrame | None = None,
        signals_df: pl.DataFrame | None = None,
        context_df: pl.DataFrame | None = None,
    ):
        self.prices = (
            prices_df
            if prices_df is not None
            else (pl.scan_parquet(prices_path).collect() if prices_path else None)
        )
        self.signals = (
            signals_df
            if signals_df is not None
            else (pl.scan_parquet(signals_path).collect() if signals_path else None)
        )
        self.context = (
            context_df
            if context_df is not None
            else (pl.scan_parquet(context_path).collect() if context_path else None)
        )

        if self.prices is None:
            raise ValueError("prices_path or prices_df required")

        # Pre-partition data by timestamp for O(1) lookups
        self._prices_by_ts = self._partition_by_timestamp(self.prices)
        self._signals_by_ts = (
            self._partition_by_timestamp(self.signals) if self.signals is not None else {}
        )
        self._context_by_ts = (
            self._partition_by_timestamp(self.context) if self.context is not None else {}
        )

        self._timestamps = self._get_timestamps()
        self._idx = 0

        # Asset ID mapping for array-based access (TASK-001)
        # Extract unique assets from prices data, sorted for deterministic ordering
        self._assets: list[str] = sorted(self.prices["asset"].unique().to_list())
        self._asset_to_idx: dict[str, int] = {
            asset: idx for idx, asset in enumerate(self._assets)
        }
        self._n_assets: int = len(self._assets)

        # Pre-materialize OHLCV as NumPy arrays (TASK-002)
        # Shape: (n_bars, n_assets) - allows slicing by time index
        self._ohlcv_arrays = self._build_ohlcv_arrays()
        self._timestamps_array: np.ndarray = np.array(self._timestamps)

    def _partition_by_timestamp(self, df: pl.DataFrame) -> dict[datetime, pl.DataFrame]:
        """Partition DataFrame into dict keyed by timestamp for O(1) access."""
        result = {}
        for ts_df in df.partition_by("timestamp", maintain_order=True):
            ts = ts_df["timestamp"][0]
            result[ts] = ts_df
        return result

    def _get_timestamps(self) -> list[datetime]:
        # Combine timestamps from all sources
        all_ts = set(self._prices_by_ts.keys())
        all_ts.update(self._signals_by_ts.keys())
        all_ts.update(self._context_by_ts.keys())
        return sorted(all_ts)

    def _build_ohlcv_arrays(self) -> dict[str, np.ndarray]:
        """Build pre-materialized OHLCV arrays from prices DataFrame.

        Returns dict with keys: 'open', 'high', 'low', 'close', 'volume'
        Each array has shape (n_bars, n_assets) with NaN for missing data.
        """
        n_bars = len(self._timestamps)
        n_assets = self._n_assets

        # Initialize arrays with NaN
        arrays = {
            "open": np.full((n_bars, n_assets), np.nan, dtype=np.float64),
            "high": np.full((n_bars, n_assets), np.nan, dtype=np.float64),
            "low": np.full((n_bars, n_assets), np.nan, dtype=np.float64),
            "close": np.full((n_bars, n_assets), np.nan, dtype=np.float64),
            "volume": np.full((n_bars, n_assets), np.nan, dtype=np.float64),
        }

        # Build timestamp to index mapping
        ts_to_idx = {ts: i for i, ts in enumerate(self._timestamps)}

        # Fill arrays from prices DataFrame using Polars pivot
        # This is a one-time cost at initialization
        for field in ["open", "high", "low", "close", "volume"]:
            if field not in self.prices.columns:
                continue

            # Pivot: rows=timestamp, cols=asset, values=field
            pivoted = self.prices.pivot(
                index="timestamp",
                on="asset",
                values=field,
                aggregate_function="first",
            )

            # Fill the array in timestamp order
            for row in pivoted.iter_rows(named=True):
                ts = row["timestamp"]
                t_idx = ts_to_idx.get(ts)
                if t_idx is None:
                    continue

                for asset in self._assets:
                    if asset in row and row[asset] is not None:
                        a_idx = self._asset_to_idx[asset]
                        arrays[field][t_idx, a_idx] = row[asset]

        return arrays

    def __iter__(self):
        self._idx = 0
        return self

    def __len__(self) -> int:
        return len(self._timestamps)

    @property
    def assets(self) -> list[str]:
        """Ordered list of unique assets (index → asset name)."""
        return self._assets

    @property
    def asset_to_idx(self) -> dict[str, int]:
        """Mapping from asset name to integer index."""
        return self._asset_to_idx

    @property
    def n_assets(self) -> int:
        """Number of unique assets."""
        return self._n_assets

    @property
    def n_bars(self) -> int:
        """Number of unique timestamps/bars."""
        return len(self._timestamps)

    def get_asset_idx(self, asset: str) -> int:
        """Get integer index for an asset name. Raises KeyError if not found."""
        return self._asset_to_idx[asset]

    def get_asset_name(self, idx: int) -> str:
        """Get asset name for an integer index. Raises IndexError if out of bounds."""
        return self._assets[idx]

    # OHLCV Array Accessors (TASK-002)
    @property
    def opens(self) -> np.ndarray:
        """Open prices array, shape (n_bars, n_assets)."""
        return self._ohlcv_arrays["open"]

    @property
    def highs(self) -> np.ndarray:
        """High prices array, shape (n_bars, n_assets)."""
        return self._ohlcv_arrays["high"]

    @property
    def lows(self) -> np.ndarray:
        """Low prices array, shape (n_bars, n_assets)."""
        return self._ohlcv_arrays["low"]

    @property
    def closes(self) -> np.ndarray:
        """Close prices array, shape (n_bars, n_assets)."""
        return self._ohlcv_arrays["close"]

    @property
    def volumes(self) -> np.ndarray:
        """Volume array, shape (n_bars, n_assets)."""
        return self._ohlcv_arrays["volume"]

    @property
    def timestamps_array(self) -> np.ndarray:
        """Timestamps as numpy array."""
        return self._timestamps_array

    def get_bar_arrays(self, t_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get OHLCV arrays for a specific time index.

        Returns (opens, highs, lows, closes, volumes) - each shape (n_assets,)
        """
        return (
            self._ohlcv_arrays["open"][t_idx],
            self._ohlcv_arrays["high"][t_idx],
            self._ohlcv_arrays["low"][t_idx],
            self._ohlcv_arrays["close"][t_idx],
            self._ohlcv_arrays["volume"][t_idx],
        )

    def __next__(self) -> tuple[datetime, dict[str, dict], dict[str, Any]]:
        if self._idx >= len(self._timestamps):
            raise StopIteration

        ts = self._timestamps[self._idx]
        self._idx += 1

        # O(1) lookup - no filtering needed
        assets_data = {}
        prices_df = self._prices_by_ts.get(ts)
        if prices_df is not None:
            for row in prices_df.iter_rows(named=True):
                asset = row["asset"]
                assets_data[asset] = {
                    "open": row.get("open"),
                    "high": row.get("high"),
                    "low": row.get("low"),
                    "close": row.get("close"),
                    "volume": row.get("volume"),
                    "signals": {},
                }

        # Add signals for each asset - O(1) lookup
        signals_df = self._signals_by_ts.get(ts)
        if signals_df is not None:
            for row in signals_df.iter_rows(named=True):
                asset = row["asset"]
                if asset in assets_data:
                    for k, v in row.items():
                        if k not in ("timestamp", "asset"):
                            assets_data[asset]["signals"][k] = v

        # Get context at this timestamp - O(1) lookup
        context_data = {}
        ctx_df = self._context_by_ts.get(ts)
        if ctx_df is not None and len(ctx_df) > 0:
            row = ctx_df.row(0, named=True)
            for k, v in row.items():
                if k != "timestamp":
                    context_data[k] = v

        return ts, assets_data, context_data
