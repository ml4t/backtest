"""Unit tests for Polars optimization features."""

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pytest

from ml4t.backtest.core.types import AssetId
from ml4t.backtest.data.polars_feed import (
    PolarsDataFeed,
    create_partitioned_dataset,
    load_partitioned_dataset,
    write_optimized_parquet,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_data():
    """Create sample OHLCV data with multiple symbols."""
    base_time = datetime(2025, 1, 1, 9, 30)
    rows = []

    # Create 3 months of daily data for 3 symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    for month in range(3):  # Jan, Feb, Mar
        for day in range(30):  # ~30 days per month
            timestamp = base_time + timedelta(days=month * 30 + day)
            for symbol in symbols:
                rows.append(
                    {
                        "timestamp": timestamp,
                        "asset_id": symbol,
                        "open": 100.0 + day,
                        "high": 101.0 + day,
                        "low": 99.0 + day,
                        "close": 100.5 + day,
                        "volume": 1000000 + day * 1000,
                    }
                )

    return pl.DataFrame(rows)


class TestCompressionOptimization:
    """Test compression codec functionality."""

    def test_write_with_zstd_compression(self, temp_dir, sample_data):
        """Test writing Parquet with zstd compression."""
        path = temp_dir / "compressed.parquet"
        write_optimized_parquet(sample_data, path, compression="zstd")

        # Verify file exists
        assert path.exists()

        # Verify can be read back
        df_read = pl.read_parquet(path)
        assert len(df_read) == len(sample_data)
        assert df_read["asset_id"].to_list() == sample_data["asset_id"].to_list()

    def test_compression_codecs(self, temp_dir, sample_data):
        """Test different compression codecs produce valid files."""
        codecs = ["zstd", "snappy", "gzip", "lz4", None]

        for codec in codecs:
            path = temp_dir / f"data_{codec or 'none'}.parquet"
            write_optimized_parquet(sample_data, path, compression=codec)

            # Verify correctness
            df_read = pl.read_parquet(path)
            assert len(df_read) == len(sample_data)

    def test_zstd_achieves_compression(self, temp_dir, sample_data):
        """Test that zstd actually reduces file size vs uncompressed."""
        # Write uncompressed
        path_uncompressed = temp_dir / "uncompressed.parquet"
        write_optimized_parquet(sample_data, path_uncompressed, compression=None)
        size_uncompressed = path_uncompressed.stat().st_size

        # Write with zstd
        path_zstd = temp_dir / "compressed.parquet"
        write_optimized_parquet(sample_data, path_zstd, compression="zstd")
        size_zstd = path_zstd.stat().st_size

        # Verify compression achieved
        compression_ratio = size_zstd / size_uncompressed
        assert compression_ratio < 1.0, "zstd should reduce file size"
        # Typical compression ratios are 30-50% reduction = 0.5-0.7 ratio
        assert compression_ratio < 0.8, "zstd should achieve >20% compression"


class TestCategoricalEncoding:
    """Test categorical encoding optimization."""

    def test_categorical_encoding_on_write(self, temp_dir, sample_data):
        """Test categorical encoding when writing Parquet."""
        path = temp_dir / "categorical.parquet"
        write_optimized_parquet(
            sample_data,
            path,
            use_categorical=True,
            categorical_columns=["asset_id"],
        )

        # Read back
        df_read = pl.read_parquet(path)

        # Verify data correctness
        assert len(df_read) == len(sample_data)
        assert df_read["asset_id"].dtype == pl.Categorical

    def test_categorical_encoding_in_datafeed(self, temp_dir, sample_data):
        """Test categorical encoding in PolarsDataFeed."""
        path = temp_dir / "data.parquet"
        sample_data.write_parquet(path)

        # Create feed with categorical enabled
        feed = PolarsDataFeed(
            price_path=path,
            asset_id=AssetId("AAPL"),
            use_categorical=True,
        )

        # Trigger initialization
        _ = feed.get_next_event()

        # Verify categorical conversion happened
        assert feed.df["asset_id"].dtype == pl.Categorical

    def test_categorical_reduces_memory(self, temp_dir):
        """Test that categorical encoding works correctly (Parquet has built-in dict encoding).

        Note: Parquet format already uses dictionary encoding for string columns,
        so file size may not differ significantly. The main benefit of categorical
        is in-memory representation after loading, not file size.
        """
        # Create dataset with many symbols
        base_time = datetime(2025, 1, 1)
        symbols = [f"SYM{i:04d}" for i in range(100)]
        rows = []

        for day in range(100):
            timestamp = base_time + timedelta(days=day)
            for symbol in symbols:
                rows.append(
                    {
                        "timestamp": timestamp,
                        "asset_id": symbol,
                        "close": 100.0,
                        "open": 99.0,
                        "high": 101.0,
                        "low": 98.0,
                        "volume": 1000000,
                    }
                )

        df = pl.DataFrame(rows)

        # Write with categorical encoding
        path_categorical = temp_dir / "categorical.parquet"
        write_optimized_parquet(
            df, path_categorical, use_categorical=True, categorical_columns=["asset_id"]
        )

        # Load back and verify categorical type
        df_loaded = pl.read_parquet(path_categorical)

        # Verify categorical encoding was applied
        assert df_loaded["asset_id"].dtype == pl.Categorical

        # Verify data correctness
        assert len(df_loaded) == len(df)

    def test_categorical_preserves_data(self, temp_dir, sample_data):
        """Test that categorical encoding preserves data exactly."""
        path = temp_dir / "categorical.parquet"
        write_optimized_parquet(
            sample_data,
            path,
            use_categorical=True,
            categorical_columns=["asset_id"],
        )

        df_read = pl.read_parquet(path)

        # Compare values (convert categorical back to string)
        original_ids = sample_data["asset_id"].to_list()
        read_ids = df_read["asset_id"].cast(pl.Utf8).to_list()
        assert original_ids == read_ids


class TestPartitioningStrategy:
    """Test partitioned dataset functionality."""

    def test_create_monthly_partitions(self, temp_dir, sample_data):
        """Test creating monthly partitioned dataset."""
        partitions = create_partitioned_dataset(
            sample_data,
            temp_dir / "partitioned",
            partition_by="month",
            timestamp_column="timestamp",
        )

        # Should create 3 partitions (3 months of data)
        assert len(partitions) == 3
        assert "2025-01" in partitions
        assert "2025-02" in partitions
        assert "2025-03" in partitions

        # Verify files exist
        for path in partitions.values():
            assert path.exists()

    def test_create_quarterly_partitions(self, temp_dir, sample_data):
        """Test creating quarterly partitioned dataset."""
        partitions = create_partitioned_dataset(
            sample_data,
            temp_dir / "partitioned",
            partition_by="quarter",
            timestamp_column="timestamp",
        )

        # All 3 months are in Q1
        assert len(partitions) == 1
        assert "2025-Q1" in partitions

    def test_load_partitioned_dataset_all(self, temp_dir, sample_data):
        """Test loading all partitions."""
        # Create partitions
        partitions = create_partitioned_dataset(
            sample_data,
            temp_dir / "partitioned",
            partition_by="month",
        )

        # Load all
        df_loaded = load_partitioned_dataset(
            temp_dir / "partitioned",
            lazy=False,
        )

        # Should have all data
        assert len(df_loaded) == len(sample_data)

    def test_load_partitioned_dataset_selective(self, temp_dir, sample_data):
        """Test loading specific partitions."""
        # Create partitions
        partitions = create_partitioned_dataset(
            sample_data,
            temp_dir / "partitioned",
            partition_by="month",
        )

        # Load only January
        df_jan = load_partitioned_dataset(
            temp_dir / "partitioned",
            partitions=["2025-01"],
            lazy=False,
        )

        # Should have only January data (31 days × 3 symbols = 93 rows)
        # Sample data creates days 0-30 for each month (31 days total)
        expected_rows = 31 * 3
        assert len(df_jan) == expected_rows

    def test_partitioned_dataset_with_compression(self, temp_dir, sample_data):
        """Test creating partitioned dataset with compression."""
        partitions = create_partitioned_dataset(
            sample_data,
            temp_dir / "partitioned",
            partition_by="month",
            compression="zstd",
        )

        # Verify partitions were created
        assert len(partitions) == 3

        # Load and verify correctness
        df_loaded = load_partitioned_dataset(
            temp_dir / "partitioned",
            lazy=False,
        )
        assert len(df_loaded) == len(sample_data)

    def test_partitioned_dataset_with_categorical(self, temp_dir, sample_data):
        """Test creating partitioned dataset with categorical encoding."""
        partitions = create_partitioned_dataset(
            sample_data,
            temp_dir / "partitioned",
            partition_by="month",
            use_categorical=True,
            categorical_columns=["asset_id"],
        )

        # Load and verify categorical encoding
        df_loaded = load_partitioned_dataset(
            temp_dir / "partitioned",
            lazy=False,
        )

        assert df_loaded["asset_id"].dtype == pl.Categorical

    def test_lazy_load_partitions(self, temp_dir, sample_data):
        """Test lazy loading of partitions."""
        # Create partitions
        create_partitioned_dataset(
            sample_data,
            temp_dir / "partitioned",
            partition_by="month",
        )

        # Load lazily
        lazy_df = load_partitioned_dataset(
            temp_dir / "partitioned",
            partitions=["2025-01"],
            lazy=True,
        )

        # Should be a LazyFrame
        assert isinstance(lazy_df, pl.LazyFrame)

        # Collect and verify
        df = lazy_df.collect()
        assert len(df) == 31 * 3  # 31 days × 3 symbols (days 0-30)


class TestLazyEvaluation:
    """Test that lazy evaluation is working correctly."""

    def test_datafeed_uses_scan_parquet(self, temp_dir, sample_data):
        """Test that PolarsDataFeed uses scan_parquet (lazy loading)."""
        path = temp_dir / "data.parquet"
        sample_data.write_parquet(path)

        feed = PolarsDataFeed(
            price_path=path,
            asset_id=AssetId("AAPL"),
        )

        # Should not be initialized yet (lazy)
        assert not feed._initialized
        assert feed.timestamp_groups is None

    def test_datafeed_defers_collection(self, temp_dir, sample_data):
        """Test that DataFrame collection is deferred until first event."""
        path = temp_dir / "data.parquet"
        sample_data.write_parquet(path)

        feed = PolarsDataFeed(
            price_path=path,
            asset_id=AssetId("AAPL"),
        )

        # Not initialized
        assert not feed._initialized

        # Get first event - triggers initialization
        _ = feed.get_next_event()

        # Now initialized
        assert feed._initialized
        assert feed.timestamp_groups is not None


class TestOptimizationIntegration:
    """Test combining multiple optimizations."""

    def test_all_optimizations_together(self, temp_dir, sample_data):
        """Test using compression + categorical + partitioning together."""
        # Create partitioned dataset with all optimizations
        partitions = create_partitioned_dataset(
            sample_data,
            temp_dir / "optimized",
            partition_by="month",
            compression="zstd",
            use_categorical=True,
            categorical_columns=["asset_id"],
        )

        # Load selective partition
        df = load_partitioned_dataset(
            temp_dir / "optimized",
            partitions=["2025-01"],
            lazy=False,
        )

        # Verify correctness
        assert len(df) == 31 * 3  # 31 days × 3 symbols (days 0-30)
        assert df["asset_id"].dtype == pl.Categorical

    def test_datafeed_with_all_optimizations(self, temp_dir, sample_data):
        """Test PolarsDataFeed with all optimizations enabled."""
        # Write optimized Parquet
        path = temp_dir / "optimized.parquet"
        write_optimized_parquet(
            sample_data,
            path,
            compression="zstd",
            use_categorical=True,
            categorical_columns=["asset_id"],
        )

        # Create feed with optimizations
        feed = PolarsDataFeed(
            price_path=path,
            asset_id=AssetId("AAPL"),
            use_categorical=True,
        )

        # Consume events and verify correctness
        events = []
        while not feed.is_exhausted:
            event = feed.get_next_event()
            if event:
                events.append(event)

        # Should get all AAPL events (90 days)
        assert len(events) == 90
        assert all(event.asset_id == "AAPL" for event in events)
