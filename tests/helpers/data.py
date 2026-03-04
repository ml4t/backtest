"""Shared data generation helpers for ml4t-backtest tests.

Consolidates duplicated price/OHLCV generators scattered across test files.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from ml4t.backtest import Broker


def make_prices(
    closes: list[float],
    *,
    asset: str = "TEST",
    start: datetime = datetime(2024, 1, 1),
    opens: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    volumes: list[float] | None = None,
    freq_days: int = 1,
) -> pl.DataFrame:
    """Create a price DataFrame suitable for DataFeed.

    Args:
        closes: List of close prices (one per bar).
        asset: Asset symbol.
        start: Starting timestamp.
        opens: Open prices (defaults to close).
        highs: High prices (defaults to close).
        lows: Low prices (defaults to close).
        volumes: Volumes (defaults to 1_000_000).
        freq_days: Days between bars.

    Returns:
        Polars DataFrame with columns: timestamp, asset, open, high, low, close, volume.
    """
    n = len(closes)
    timestamps = [start + timedelta(days=i * freq_days) for i in range(n)]

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "asset": [asset] * n,
            "open": opens if opens is not None else closes,
            "high": highs if highs is not None else closes,
            "low": lows if lows is not None else closes,
            "close": closes,
            "volume": volumes if volumes is not None else [1_000_000.0] * n,
        }
    )


def make_ohlcv_prices(
    bars: list[tuple[float, float, float, float]],
    *,
    asset: str = "TEST",
    start: datetime = datetime(2024, 1, 1),
    volumes: list[float] | None = None,
    freq_days: int = 1,
) -> pl.DataFrame:
    """Create a price DataFrame from explicit OHLC tuples.

    Args:
        bars: List of (open, high, low, close) tuples.
        asset: Asset symbol.
        start: Starting timestamp.
        volumes: Volumes (defaults to 1_000_000).
        freq_days: Days between bars.

    Returns:
        Polars DataFrame with columns: timestamp, asset, open, high, low, close, volume.
    """
    n = len(bars)
    timestamps = [start + timedelta(days=i * freq_days) for i in range(n)]
    opens, highs, lows, closes = zip(*bars)

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "asset": [asset] * n,
            "open": list(opens),
            "high": list(highs),
            "low": list(lows),
            "close": list(closes),
            "volume": volumes if volumes is not None else [1_000_000.0] * n,
        }
    )


def set_broker_bar(
    broker: Broker,
    price: float,
    *,
    asset: str = "TEST",
    ts: datetime = datetime(2024, 1, 1),
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
    volume: float = 1_000_000.0,
) -> None:
    """Set the current bar on a Broker instance for unit testing.

    Args:
        broker: Broker instance.
        price: Close price (also used for open/high/low if not specified).
        asset: Asset symbol.
        ts: Bar timestamp.
        open_: Open price (defaults to close).
        high: High price (defaults to close).
        low: Low price (defaults to close).
        volume: Volume.
    """
    o = open_ if open_ is not None else price
    h = high if high is not None else price
    lo = low if low is not None else price

    broker._update_time(
        ts,
        {asset: price},
        {asset: o},
        {asset: h},
        {asset: lo},
        {asset: volume},
        {asset: {}},
    )
