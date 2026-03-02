"""Hot-path benchmark tests for feed/engine performance.

These benchmarks compare the current DataFeed implementation against a
reference legacy implementation (captured from pre-optimization behavior).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from statistics import median
from time import perf_counter
from typing import Any

import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy


class _LegacyDataFeed:
    """Reference feed implementation mirroring the previous code path."""

    def __init__(
        self,
        prices_df: pl.DataFrame,
        signals_df: pl.DataFrame | None = None,
        context_df: pl.DataFrame | None = None,
    ):
        self.prices = prices_df
        self.signals = signals_df
        self.context = context_df
        self._prices_by_ts = self._partition_by_timestamp(self.prices)
        self._signals_by_ts = (
            self._partition_by_timestamp(self.signals) if self.signals is not None else {}
        )
        self._context_by_ts = (
            self._partition_by_timestamp(self.context) if self.context is not None else {}
        )
        all_ts = set(self._prices_by_ts.keys())
        all_ts.update(self._signals_by_ts.keys())
        all_ts.update(self._context_by_ts.keys())
        self._timestamps = sorted(all_ts)
        self._idx = 0

    def _partition_by_timestamp(self, df: pl.DataFrame) -> dict[datetime, pl.DataFrame]:
        result: dict[datetime, pl.DataFrame] = {}
        for ts_df in df.partition_by("timestamp", maintain_order=True):
            ts = ts_df["timestamp"][0]
            result[ts] = ts_df
        return result

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self) -> tuple[datetime, dict[str, dict], dict[str, Any]]:
        if self._idx >= len(self._timestamps):
            raise StopIteration

        ts = self._timestamps[self._idx]
        self._idx += 1

        assets_data: dict[str, dict[str, Any]] = {}
        price_df = self._prices_by_ts.get(ts)
        if price_df is not None:
            for row in price_df.iter_rows(named=True):
                asset = row["asset"]
                assets_data[asset] = {
                    "open": row.get("open"),
                    "high": row.get("high"),
                    "low": row.get("low"),
                    "close": row.get("close"),
                    "volume": row.get("volume"),
                    "signals": {},
                }

        signal_df = self._signals_by_ts.get(ts)
        if signal_df is not None:
            for row in signal_df.iter_rows(named=True):
                asset = row["asset"]
                if asset in assets_data:
                    for k, v in row.items():
                        if k not in ("timestamp", "asset"):
                            assets_data[asset]["signals"][k] = v

        context_data: dict[str, Any] = {}
        ctx_df = self._context_by_ts.get(ts)
        if ctx_df is not None and len(ctx_df) > 0:
            row = ctx_df.row(0, named=True)
            for k, v in row.items():
                if k != "timestamp":
                    context_data[k] = v

        return ts, assets_data, context_data


class _NoopStrategy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        return None


def _build_benchmark_data(n_bars: int, n_assets: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    start = datetime(2021, 1, 1)
    price_rows = []
    signal_rows = []

    for i in range(n_bars):
        ts = start + timedelta(minutes=i)
        signal = ((i % 20) - 10) / 10.0
        for a in range(n_assets):
            asset = f"A{a:03d}"
            close = 100.0 + (a * 0.03) + ((i % 15) * 0.01)
            price_rows.append(
                {
                    "timestamp": ts,
                    "asset": asset,
                    "open": close * 0.999,
                    "high": close * 1.001,
                    "low": close * 0.998,
                    "close": close,
                    "volume": 1_000_000.0 + a,
                }
            )
            signal_rows.append(
                {
                    "timestamp": ts,
                    "asset": asset,
                    "score": signal,
                }
            )

    return pl.DataFrame(price_rows), pl.DataFrame(signal_rows)


def _run_engine(feed_cls, prices: pl.DataFrame, signals: pl.DataFrame) -> float:
    feed = feed_cls(prices_df=prices, signals_df=signals)
    engine = Engine.from_config(feed, _NoopStrategy(), BacktestConfig.from_preset("fast"))
    start = perf_counter()
    _ = engine.run()
    return perf_counter() - start


@pytest.mark.benchmark
def test_optimized_feed_matches_legacy_output():
    prices, signals = _build_benchmark_data(n_bars=50, n_assets=5)

    optimized = list(DataFeed(prices_df=prices, signals_df=signals))
    legacy = list(_LegacyDataFeed(prices_df=prices, signals_df=signals))

    assert len(optimized) == len(legacy)
    for (opt_ts, opt_assets, opt_ctx), (legacy_ts, legacy_assets, legacy_ctx) in zip(
        optimized, legacy, strict=True
    ):
        assert opt_ts == legacy_ts
        assert dict(opt_assets) == legacy_assets
        assert opt_ctx == legacy_ctx


@pytest.mark.benchmark
def test_optimized_feed_runtime_vs_legacy_baseline():
    prices, signals = _build_benchmark_data(n_bars=3000, n_assets=20)

    # Warm-up for consistent timing
    _ = _run_engine(DataFeed, prices, signals)
    _ = _run_engine(_LegacyDataFeed, prices, signals)

    optimized_runs = [_run_engine(DataFeed, prices, signals) for _ in range(3)]
    legacy_runs = [_run_engine(_LegacyDataFeed, prices, signals) for _ in range(3)]

    optimized_median = median(optimized_runs)
    legacy_median = median(legacy_runs)
    ratio = optimized_median / legacy_median if legacy_median > 0 else 1.0

    # Benchmark regression guard: optimized path should not be materially slower.
    assert ratio <= 1.15, (
        f"Optimized median {optimized_median:.4f}s vs legacy {legacy_median:.4f}s "
        f"(ratio={ratio:.3f})."
    )
