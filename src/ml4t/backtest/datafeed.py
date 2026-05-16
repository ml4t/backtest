"""Polars-based multi-asset data feed with O(1) timestamp lookups.

Memory-efficient implementation that stores the original DataFrames plus
timestamp-to-slice indexes, then converts only the current bar to dicts at
iteration time.
"""

from datetime import datetime
from typing import Any

import polars as pl

from ml4t.specs.market_data import FeedSpec


class _AssetsData(dict[str, dict[str, Any]]):
    """Internal per-bar payload with pre-extracted broker views."""

    __slots__ = (
        "_prices",
        "_opens",
        "_highs",
        "_lows",
        "_closes",
        "_volumes",
        "_bids",
        "_asks",
        "_mids",
        "_bid_sizes",
        "_ask_sizes",
        "_signals",
    )

    def __init__(self):
        super().__init__()
        self._prices: dict[str, float] = {}
        self._opens: dict[str, Any] = {}
        self._highs: dict[str, Any] = {}
        self._lows: dict[str, Any] = {}
        self._closes: dict[str, Any] = {}
        self._volumes: dict[str, Any] = {}
        self._bids: dict[str, Any] = {}
        self._asks: dict[str, Any] = {}
        self._mids: dict[str, Any] = {}
        self._bid_sizes: dict[str, Any] = {}
        self._ask_sizes: dict[str, Any] = {}
        self._signals: dict[str, dict[str, Any]] = {}


class DataFeed:
    """Polars-based multi-asset data feed with signals and context.

    Pre-indexes data by timestamp at initialization for O(1) lookups during
    iteration. DataFrames are kept in their native format and converted to
    dicts only for the active bar, avoiding the large memory overhead of
    materializing one child DataFrame per timestamp.

    Memory Efficiency:
        - 1M bars: ~100 MB (was ~1 GB with pre-converted dicts)
        - 10M bars: ~1 GB (vs ~10+ GB with dicts)

    Usage:
        feed = DataFeed(prices_df=prices, signals_df=signals)
        for timestamp, assets_data, context in feed:
            # assets_data: {"AAPL": {"close": 150.0, "signals": {...}}, ...}
            process(timestamp, assets_data)
    """

    #: Column names checked (in order) when auto-detecting the entity column.
    ENTITY_COL_CANDIDATES = ("symbol", "asset", "product", "ticker")

    def __init__(
        self,
        prices_path: str | None = None,
        signals_path: str | None = None,
        context_path: str | None = None,
        prices_df: pl.DataFrame | None = None,
        signals_df: pl.DataFrame | None = None,
        context_df: pl.DataFrame | None = None,
        *,
        feed_spec: FeedSpec | Any | None = None,
        contract: FeedSpec | Any | None = None,
        entity_col: str | None = None,
        timestamp_col: str | None = None,
        price_col: str | None = None,
        open_col: str | None = None,
        high_col: str | None = None,
        low_col: str | None = None,
        close_col: str | None = None,
        volume_col: str | None = None,
        bid_col: str | None = None,
        ask_col: str | None = None,
        mid_col: str | None = None,
        bid_size_col: str | None = None,
        ask_size_col: str | None = None,
    ):
        if feed_spec is not None and contract is not None:
            raise ValueError("Pass either feed_spec or contract, not both")

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

        raw_spec = FeedSpec.from_any(feed_spec if feed_spec is not None else contract)
        self.feed_spec = raw_spec.with_overrides(
            entity_col=entity_col,
            timestamp_col=timestamp_col,
            price_col=price_col,
            open_col=open_col,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col,
            volume_col=volume_col,
            bid_col=bid_col,
            ask_col=ask_col,
            mid_col=mid_col,
            bid_size_col=bid_size_col,
            ask_size_col=ask_size_col,
        ).resolve(self.prices.columns, self.ENTITY_COL_CANDIDATES)
        self.contract = self.feed_spec
        self._timestamp_col = self.feed_spec.timestamp_col
        self._entity_col = self.feed_spec.entity_col
        self._price_col = self.feed_spec.price_col
        self._open_col = self.feed_spec.open_col
        self._high_col = self.feed_spec.high_col
        self._low_col = self.feed_spec.low_col
        self._close_col = self.feed_spec.close_col
        self._volume_col = self.feed_spec.volume_col
        self._bid_col = self.feed_spec.bid_col
        self._ask_col = self.feed_spec.ask_col
        self._mid_col = self.feed_spec.mid_col
        self._bid_size_col = self.feed_spec.bid_size_col
        self._ask_size_col = self.feed_spec.ask_size_col

        self.prices, self._price_ranges_by_ts = self._index_by_timestamp(self.prices)
        if self.signals is not None:
            self.signals, self._signal_ranges_by_ts = self._index_by_timestamp(self.signals)
        else:
            self._signal_ranges_by_ts = {}
        if self.context is not None:
            self.context, self._context_ranges_by_ts = self._index_by_timestamp(self.context)
        else:
            self._context_ranges_by_ts = {}

        self._timestamps = self._get_timestamps()
        self._idx = 0
        self._signal_columns = (
            [c for c in self.signals.columns if c not in (self._timestamp_col, self._entity_col)]
            if self.signals is not None
            else []
        )
        self._context_columns = (
            [c for c in self.context.columns if c != self._timestamp_col]
            if self.context is not None
            else []
        )

        price_cols = self.prices.columns
        self._price_asset_idx = price_cols.index(self._entity_col)
        self._price_open_idx = (
            price_cols.index(self._open_col) if self._open_col in price_cols else -1
        )
        self._price_high_idx = (
            price_cols.index(self._high_col) if self._high_col in price_cols else -1
        )
        self._price_low_idx = price_cols.index(self._low_col) if self._low_col in price_cols else -1
        self._price_close_idx = (
            price_cols.index(self._close_col) if self._close_col in price_cols else -1
        )
        self._price_price_idx = (
            price_cols.index(self._price_col)
            if self._price_col in price_cols
            else self._price_close_idx
        )
        self._price_volume_idx = (
            price_cols.index(self._volume_col) if self._volume_col in price_cols else -1
        )
        self._price_bid_idx = price_cols.index(self._bid_col) if self._bid_col in price_cols else -1
        self._price_ask_idx = price_cols.index(self._ask_col) if self._ask_col in price_cols else -1
        self._price_mid_idx = price_cols.index(self._mid_col) if self._mid_col in price_cols else -1
        self._price_bid_size_idx = (
            price_cols.index(self._bid_size_col) if self._bid_size_col in price_cols else -1
        )
        self._price_ask_size_idx = (
            price_cols.index(self._ask_size_col) if self._ask_size_col in price_cols else -1
        )

        if self.signals is not None:
            signal_cols = self.signals.columns
            if self._timestamp_col not in signal_cols:
                raise ValueError(
                    f"timestamp_col={self._timestamp_col!r} not found in signal columns {signal_cols}"
                )
            self._signal_asset_idx = signal_cols.index(self._entity_col)
            self._signal_col_indices = [signal_cols.index(c) for c in self._signal_columns]
        else:
            self._signal_asset_idx = -1
            self._signal_col_indices = []

        if self.context is not None:
            context_cols = self.context.columns
            if self._timestamp_col not in context_cols:
                raise ValueError(
                    f"timestamp_col={self._timestamp_col!r} not found in context columns {context_cols}"
                )
            self._context_col_indices = [context_cols.index(c) for c in self._context_columns]
        else:
            self._context_col_indices = []

    @classmethod
    def _resolve_entity_col(cls, explicit: str | None, columns: list[str]) -> str:
        """Determine the entity identifier column.

        If *explicit* is given, validate it exists.  Otherwise auto-detect by
        checking ``ENTITY_COL_CANDIDATES`` in order.
        """
        if explicit is not None:
            if explicit not in columns:
                raise ValueError(f"entity_col={explicit!r} not found in columns {columns}")
            return explicit
        for candidate in cls.ENTITY_COL_CANDIDATES:
            if candidate in columns:
                return candidate
        raise ValueError(
            f"Cannot detect entity column. Expected one of "
            f"{cls.ENTITY_COL_CANDIDATES}, got columns {columns}"
        )

    def _index_by_timestamp(
        self, df: pl.DataFrame
    ) -> tuple[pl.DataFrame, dict[datetime, tuple[int, int]]]:
        """Return a timestamp -> (offset, length) index over a sorted DataFrame."""
        if self._timestamp_col not in df.columns:
            raise ValueError(
                f"timestamp_col={self._timestamp_col!r} not found in columns {df.columns}"
            )

        if not df[self._timestamp_col].is_sorted():
            df = df.sort(self._timestamp_col)

        counts = df.group_by(self._timestamp_col, maintain_order=True).agg(
            pl.len().alias("_row_count")
        )
        result: dict[datetime, tuple[int, int]] = {}
        offset = 0
        for ts, row_count in counts.iter_rows(named=False):
            count = int(row_count)
            result[ts] = (offset, count)
            offset += count
        return df, result

    @staticmethod
    def _slice_for_timestamp(
        df: pl.DataFrame | None,
        ranges_by_ts: dict[datetime, tuple[int, int]],
        ts: datetime,
    ) -> pl.DataFrame | None:
        """Return the zero-copy timestamp slice for the requested bar."""
        if df is None:
            return None
        bounds = ranges_by_ts.get(ts)
        if bounds is None:
            return None
        offset, length = bounds
        return df.slice(offset, length)

    def _get_timestamps(self) -> list[datetime]:
        """Get sorted list of all timestamps across all data sources."""
        all_ts = set(self._price_ranges_by_ts.keys())
        all_ts.update(self._signal_ranges_by_ts.keys())
        all_ts.update(self._context_ranges_by_ts.keys())
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

    @property
    def timestamps(self) -> tuple[datetime, ...]:
        """Unique feed timestamps in iteration order."""
        return tuple(self._timestamps)

    def __next__(self) -> tuple[datetime, dict[str, dict], dict[str, Any]]:
        if self._idx >= len(self._timestamps):
            raise StopIteration

        ts = self._timestamps[self._idx]
        self._idx += 1

        # O(1) lookup + lazy conversion to dicts (only for current bar)
        assets_data = _AssetsData()
        price_asset_idx = self._price_asset_idx
        price_open_idx = self._price_open_idx
        price_high_idx = self._price_high_idx
        price_low_idx = self._price_low_idx
        price_close_idx = self._price_close_idx
        price_price_idx = self._price_price_idx
        price_volume_idx = self._price_volume_idx
        price_bid_idx = self._price_bid_idx
        price_ask_idx = self._price_ask_idx
        price_mid_idx = self._price_mid_idx
        price_bid_size_idx = self._price_bid_size_idx
        price_ask_size_idx = self._price_ask_size_idx

        # Convert price DataFrame slice to dicts (lazy, only current bar)
        price_df = self._slice_for_timestamp(self.prices, self._price_ranges_by_ts, ts)
        if price_df is not None:
            for row in price_df.iter_rows(named=False):
                asset = row[price_asset_idx]
                close = row[price_close_idx] if price_close_idx >= 0 else None
                price = row[price_price_idx] if price_price_idx >= 0 else close
                open_ = row[price_open_idx] if price_open_idx >= 0 else close
                high = row[price_high_idx] if price_high_idx >= 0 else close
                low = row[price_low_idx] if price_low_idx >= 0 else close
                volume = row[price_volume_idx] if price_volume_idx >= 0 else 0.0
                bid = row[price_bid_idx] if price_bid_idx >= 0 else None
                ask = row[price_ask_idx] if price_ask_idx >= 0 else None
                mid = row[price_mid_idx] if price_mid_idx >= 0 else None
                bid_size = row[price_bid_size_idx] if price_bid_size_idx >= 0 else None
                ask_size = row[price_ask_size_idx] if price_ask_size_idx >= 0 else None

                if open_ is None:
                    open_ = close
                if high is None:
                    high = close
                if low is None:
                    low = close
                if volume is None:
                    volume = 0.0
                if price is None:
                    price = close
                if mid is None and bid is not None and ask is not None:
                    mid = (bid + ask) / 2.0

                assets_data[asset] = {
                    "price": price,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "signals": {},
                }
                if bid is not None:
                    assets_data[asset]["bid"] = bid
                if ask is not None:
                    assets_data[asset]["ask"] = ask
                if mid is not None:
                    assets_data[asset]["mid"] = mid
                if bid_size is not None:
                    assets_data[asset]["bid_size"] = bid_size
                if ask_size is not None:
                    assets_data[asset]["ask_size"] = ask_size
                if price is not None:
                    assets_data._prices[asset] = price
                assets_data._opens[asset] = open_
                assets_data._highs[asset] = high
                assets_data._lows[asset] = low
                assets_data._closes[asset] = close
                assets_data._volumes[asset] = volume
                if bid is not None:
                    assets_data._bids[asset] = bid
                if ask is not None:
                    assets_data._asks[asset] = ask
                if mid is not None:
                    assets_data._mids[asset] = mid
                if bid_size is not None:
                    assets_data._bid_sizes[asset] = bid_size
                if ask_size is not None:
                    assets_data._ask_sizes[asset] = ask_size
                assets_data._signals[asset] = assets_data[asset]["signals"]

        # Add signals for each asset - lazy conversion
        signal_df = self._slice_for_timestamp(self.signals, self._signal_ranges_by_ts, ts)
        if signal_df is not None:
            signal_asset_idx = self._signal_asset_idx
            signal_col_indices = self._signal_col_indices
            signal_columns = self._signal_columns
            for row in signal_df.iter_rows(named=False):
                asset = row[signal_asset_idx]
                if asset in assets_data:
                    asset_signals = assets_data._signals[asset]
                    for i, col_idx in enumerate(signal_col_indices):
                        asset_signals[signal_columns[i]] = row[col_idx]

        # Get context at this timestamp - lazy conversion
        context_data: dict[str, Any] = {}
        ctx_df = self._slice_for_timestamp(self.context, self._context_ranges_by_ts, ts)
        if ctx_df is not None and len(ctx_df) > 0:
            row = ctx_df.row(0)
            for i, col_idx in enumerate(self._context_col_indices):
                context_data[self._context_columns[i]] = row[col_idx]

        return ts, assets_data, context_data
