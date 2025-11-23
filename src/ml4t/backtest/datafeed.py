"""Polars-based multi-asset data feed with O(1) timestamp lookups."""

from datetime import datetime
from typing import Any

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
        # Use to_dicts() for faster iteration (vs iter_rows)
        self._prices_by_ts = self._partition_by_timestamp_dicts(self.prices)
        self._signals_by_ts = (
            self._partition_by_timestamp_dicts(self.signals)
            if self.signals is not None
            else {}
        )
        self._context_by_ts = (
            self._partition_by_timestamp(self.context) if self.context is not None else {}
        )

        self._timestamps = self._get_timestamps()
        self._idx = 0

    def _partition_by_timestamp(self, df: pl.DataFrame) -> dict[datetime, pl.DataFrame]:
        """Partition DataFrame into dict keyed by timestamp for O(1) access."""
        result = {}
        for ts_df in df.partition_by("timestamp", maintain_order=True):
            ts = ts_df["timestamp"][0]
            result[ts] = ts_df
        return result

    def _partition_by_timestamp_dicts(
        self, df: pl.DataFrame
    ) -> dict[datetime, list[dict[str, Any]]]:
        """Partition DataFrame into dict of lists of row dicts for faster iteration."""
        result: dict[datetime, list[dict[str, Any]]] = {}
        for row in df.to_dicts():
            ts = row["timestamp"]
            if ts not in result:
                result[ts] = []
            result[ts].append(row)
        return result

    def _get_timestamps(self) -> list[datetime]:
        # Combine timestamps from all sources
        all_ts = set(self._prices_by_ts.keys())
        all_ts.update(self._signals_by_ts.keys())
        all_ts.update(self._context_by_ts.keys())
        return sorted(all_ts)

    def __iter__(self):
        self._idx = 0
        return self

    def __len__(self) -> int:
        return len(self._timestamps)

    @property
    def n_bars(self) -> int:
        """Number of unique timestamps/bars."""
        return len(self._timestamps)

    def __next__(self) -> tuple[datetime, dict[str, dict], dict[str, Any]]:
        if self._idx >= len(self._timestamps):
            raise StopIteration

        ts = self._timestamps[self._idx]
        self._idx += 1

        # O(1) lookup - using pre-converted dicts (faster than iter_rows)
        assets_data = {}
        price_rows = self._prices_by_ts.get(ts)
        if price_rows is not None:
            for row in price_rows:
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
        signal_rows = self._signals_by_ts.get(ts)
        if signal_rows is not None:
            for row in signal_rows:
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
