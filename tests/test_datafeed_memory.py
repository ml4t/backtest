"""Memory efficiency tests for DataFeed.

These tests verify that the DataFeed implementation is memory-efficient
by storing DataFrames instead of pre-converted dicts.
"""

from datetime import datetime, timedelta

import polars as pl
import pytest

from ml4t.backtest import DataFeed


class TestDataFeedMemoryEfficiency:
    """Tests for DataFeed memory efficiency."""

    def _create_large_dataset(self, n_bars: int, n_assets: int) -> pl.DataFrame:
        """Create a test dataset with specified size."""
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_bars)]
        assets = [f"ASSET_{i:04d}" for i in range(n_assets)]

        rows = []
        for ts in dates:
            for asset in assets:
                rows.append(
                    {
                        "timestamp": ts,
                        "asset": asset,
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "close": 100.5,
                        "volume": 1_000_000,
                    }
                )

        return pl.DataFrame(rows)

    def test_datafeed_stores_dataframes_not_dicts(self):
        """Verify DataFeed stores DataFrames internally (not dicts)."""
        prices = self._create_large_dataset(10, 5)
        feed = DataFeed(prices_df=prices)

        # Check internal storage type
        first_ts = list(feed._prices_by_ts.keys())[0]
        stored_value = feed._prices_by_ts[first_ts]

        # Should be a DataFrame, not a list of dicts
        assert isinstance(stored_value, pl.DataFrame), (
            f"Expected pl.DataFrame, got {type(stored_value)}. "
            "DataFeed should store DataFrames for memory efficiency."
        )

    def test_datafeed_iteration_produces_correct_format(self):
        """Verify iteration produces the expected dict format."""
        prices = self._create_large_dataset(5, 3)
        feed = DataFeed(prices_df=prices)

        ts, assets_data, context = next(iter(feed))

        # Should have all 3 assets
        assert len(assets_data) == 3

        # Each asset should have OHLCV and signals dict
        for _asset, data in assets_data.items():
            assert "open" in data
            assert "high" in data
            assert "low" in data
            assert "close" in data
            assert "volume" in data
            assert "signals" in data
            assert isinstance(data["signals"], dict)

    def test_datafeed_memory_scales_with_unique_timestamps(self):
        """Verify memory usage scales with timestamps, not total rows.

        The key insight is that storing DataFrames per timestamp uses
        much less memory than storing dicts per row.
        """
        # Create dataset: 100 bars × 10 assets = 1000 rows
        prices = self._create_large_dataset(100, 10)
        feed = DataFeed(prices_df=prices)

        # Memory should be dominated by DataFrames, not dicts
        # We verify this by checking the storage structure
        assert len(feed._prices_by_ts) == 100  # One entry per timestamp
        assert feed.n_bars == 100

        # Each stored value is a DataFrame (compact) not list[dict] (bloated)
        for ts_df in feed._prices_by_ts.values():
            assert isinstance(ts_df, pl.DataFrame)
            assert len(ts_df) == 10  # 10 assets per timestamp

    def test_datafeed_with_signals(self):
        """Verify DataFeed correctly handles signals with lazy conversion."""
        prices = self._create_large_dataset(5, 2)
        signals = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1), datetime(2020, 1, 1)],
                "asset": ["ASSET_0000", "ASSET_0001"],
                "momentum": [0.5, -0.3],
                "rsi": [65.0, 35.0],
            }
        )

        feed = DataFeed(prices_df=prices, signals_df=signals)

        # First bar should have signals
        ts, assets_data, _ = next(iter(feed))

        assert "ASSET_0000" in assets_data
        assert assets_data["ASSET_0000"]["signals"]["momentum"] == 0.5
        assert assets_data["ASSET_0000"]["signals"]["rsi"] == 65.0
        assert assets_data["ASSET_0001"]["signals"]["momentum"] == -0.3

    def test_datafeed_with_context(self):
        """Verify DataFeed correctly handles context with lazy conversion."""
        prices = self._create_large_dataset(5, 2)
        context = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
                "vix": [20.5, 22.0],
                "spy_close": [300.0, 302.0],
            }
        )

        feed = DataFeed(prices_df=prices, context_df=context)

        ts, _, ctx = next(iter(feed))

        assert ctx["vix"] == 20.5
        assert ctx["spy_close"] == 300.0

    @pytest.mark.benchmark
    def test_datafeed_memory_benchmark(self):
        """Benchmark memory usage for medium-scale dataset.

        This test verifies the memory fix is working by checking
        that the internal storage uses DataFrames.

        For a proper memory measurement, run:
            python -c "
            import tracemalloc
            from datetime import datetime, timedelta
            import polars as pl
            from ml4t.backtest import DataFeed

            tracemalloc.start()

            # Create 10K bars × 100 assets = 1M rows
            dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10000)]
            assets = [f'ASSET_{i:04d}' for i in range(100)]
            rows = [
                {'timestamp': ts, 'asset': asset, 'open': 100.0, 'high': 101.0,
                 'low': 99.0, 'close': 100.5, 'volume': 1_000_000}
                for ts in dates for asset in assets
            ]
            prices = pl.DataFrame(rows)

            feed = DataFeed(prices_df=prices)

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            print(f'Current memory: {current / 1024 / 1024:.1f} MB')
            print(f'Peak memory: {peak / 1024 / 1024:.1f} MB')
            print(f'Expected: <500 MB (was >1 GB with dicts)')
            "
        """
        # Create modest dataset for CI
        prices = self._create_large_dataset(100, 50)  # 5000 rows
        feed = DataFeed(prices_df=prices)

        # Verify structure is correct
        assert feed.n_bars == 100
        assert all(isinstance(df, pl.DataFrame) for df in feed._prices_by_ts.values())

        # Iterate through all bars to verify lazy conversion works
        count = 0
        for _ts, data, _ctx in feed:
            count += 1
            assert len(data) == 50  # 50 assets per bar

        assert count == 100


class TestDataFeedEdgeCases:
    """Edge case tests for DataFeed."""

    def test_empty_signals(self):
        """DataFeed should work with empty signals."""
        prices = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1)],
                "asset": ["AAPL"],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1_000_000],
            }
        )

        feed = DataFeed(prices_df=prices)
        ts, data, ctx = next(iter(feed))

        assert "AAPL" in data
        assert data["AAPL"]["signals"] == {}

    def test_single_bar_single_asset(self):
        """DataFeed should handle minimal dataset."""
        prices = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1)],
                "asset": ["AAPL"],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1_000_000],
            }
        )

        feed = DataFeed(prices_df=prices)

        assert len(feed) == 1
        assert feed.n_bars == 1

        ts, data, ctx = next(iter(feed))
        assert ts == datetime(2020, 1, 1)
        assert data["AAPL"]["close"] == 100.5
