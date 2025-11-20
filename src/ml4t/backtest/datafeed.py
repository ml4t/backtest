"""Polars-based multi-asset data feed."""

from datetime import datetime
from typing import Any
import polars as pl


class DataFeed:
    """Polars-based multi-asset data feed with signals and context."""

    def __init__(
        self,
        prices_path: str | None = None,
        signals_path: str | None = None,
        context_path: str | None = None,
        prices_df: pl.DataFrame | None = None,
        signals_df: pl.DataFrame | None = None,
        context_df: pl.DataFrame | None = None,
    ):
        self.prices = prices_df if prices_df is not None else (
            pl.scan_parquet(prices_path).collect() if prices_path else None
        )
        self.signals = signals_df if signals_df is not None else (
            pl.scan_parquet(signals_path).collect() if signals_path else None
        )
        self.context = context_df if context_df is not None else (
            pl.scan_parquet(context_path).collect() if context_path else None
        )

        if self.prices is None:
            raise ValueError("prices_path or prices_df required")

        self._timestamps = self._get_timestamps()
        self._idx = 0

    def _get_timestamps(self) -> list[datetime]:
        ts = self.prices.select("timestamp").unique()
        if self.signals is not None:
            ts = ts.vstack(self.signals.select("timestamp").unique())
        if self.context is not None:
            ts = ts.vstack(self.context.select("timestamp").unique())
        return sorted(ts.unique().to_series().to_list())

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self) -> tuple[datetime, dict[str, dict], dict[str, Any]]:
        if self._idx >= len(self._timestamps):
            raise StopIteration

        ts = self._timestamps[self._idx]
        self._idx += 1

        # Get prices for all assets at this timestamp
        assets_data = {}
        prices_at_ts = self.prices.filter(pl.col("timestamp") == ts)
        for row in prices_at_ts.iter_rows(named=True):
            asset = row["asset"]
            assets_data[asset] = {
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
                "volume": row.get("volume"),
                "signals": {},
            }

        # Add signals for each asset
        if self.signals is not None:
            signals_at_ts = self.signals.filter(pl.col("timestamp") == ts)
            for row in signals_at_ts.iter_rows(named=True):
                asset = row["asset"]
                if asset in assets_data:
                    for k, v in row.items():
                        if k not in ("timestamp", "asset"):
                            assets_data[asset]["signals"][k] = v

        # Get context at this timestamp
        context_data = {}
        if self.context is not None:
            ctx_at_ts = self.context.filter(pl.col("timestamp") == ts)
            if len(ctx_at_ts) > 0:
                row = ctx_at_ts.row(0, named=True)
                for k, v in row.items():
                    if k != "timestamp":
                        context_data[k] = v

        return ts, assets_data, context_data
