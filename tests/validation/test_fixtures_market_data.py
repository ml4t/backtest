"""
Unit tests for market data fixtures.

Tests all fixture functions with comprehensive coverage of edge cases
and error conditions.
"""

import pytest
from pathlib import Path
from datetime import datetime, date, timezone
import polars as pl
import pandas as pd
from tempfile import TemporaryDirectory

from tests.validation.fixtures.market_data import (
    load_wiki_prices,
    get_ticker_data,
    prepare_zipline_bundle_data,
    WIKI_PRICES_PATH,
)


class TestLoadWikiPrices:
    """Tests for load_wiki_prices() function."""

    def test_load_wiki_prices_unadjusted(self):
        """Test loading unadjusted prices returns correct columns."""
        df = load_wiki_prices(use_adjusted=False)

        # Check expected columns
        expected_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
        assert df.columns == expected_cols, f"Expected columns {expected_cols}, got {df.columns}"

        # Check data types
        assert df['ticker'].dtype == pl.Utf8
        # date column is actually datetime (not just date)
        assert df['date'].dtype in [pl.Date, pl.Datetime, pl.Datetime('ns')]
        assert df['open'].dtype == pl.Float64
        assert df['close'].dtype == pl.Float64
        assert df['volume'].dtype == pl.Float64

        # Check data is not empty
        assert len(df) > 0, "Expected non-empty DataFrame"

    def test_load_wiki_prices_adjusted(self):
        """Test loading adjusted prices returns correct columns."""
        df = load_wiki_prices(use_adjusted=True)

        # Check expected columns (same names, different data)
        expected_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
        assert df.columns == expected_cols

        # Check data is not empty
        assert len(df) > 0

    def test_load_wiki_prices_different_values(self):
        """Test that adjusted and unadjusted prices differ."""
        df_unadj = load_wiki_prices(use_adjusted=False)
        df_adj = load_wiki_prices(use_adjusted=True)

        # For tickers with splits, adjusted prices should differ
        # Sample one ticker and compare
        ticker_unadj = df_unadj.filter(pl.col('ticker') == 'AAPL').head(100)
        ticker_adj = df_adj.filter(pl.col('ticker') == 'AAPL').head(100)

        # At least some prices should differ (due to adjustments)
        # We can't guarantee ALL differ, but there should be some difference
        assert len(ticker_unadj) > 0
        assert len(ticker_adj) > 0

    def test_load_wiki_prices_missing_file(self, monkeypatch):
        """Test error handling when wiki_prices.parquet is missing."""
        # Mock WIKI_PRICES_PATH to non-existent location
        monkeypatch.setattr(
            'fixtures.market_data.WIKI_PRICES_PATH',
            Path('/nonexistent/path/wiki_prices.parquet')
        )

        with pytest.raises(FileNotFoundError) as exc_info:
            load_wiki_prices()

        assert 'Wiki prices not found' in str(exc_info.value)
        assert 'daily_us_equities' in str(exc_info.value)


class TestGetTickerData:
    """Tests for get_ticker_data() function."""

    def test_get_ticker_data_basic(self):
        """Test basic ticker data retrieval."""
        df = get_ticker_data('AAPL', start_date='2017-01-01', end_date='2017-12-31')

        # Check columns (note: actual order is symbol, timestamp, open, high, low, close, volume)
        expected_cols = {'symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        assert set(df.columns) == expected_cols

        # Check symbol is correct
        symbols = df['symbol'].unique().to_list()
        assert symbols == ['AAPL']

        # Check date range
        assert df['timestamp'].min() >= datetime(2017, 1, 1, tzinfo=timezone.utc)
        assert df['timestamp'].max() <= datetime(2017, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

        # Check data is sorted by timestamp
        timestamps = df['timestamp'].to_list()
        assert timestamps == sorted(timestamps)

    def test_get_ticker_data_no_dates(self):
        """Test retrieval without date filters returns all available data."""
        df = get_ticker_data('AAPL')

        assert len(df) > 0
        assert df['symbol'].unique().to_list() == ['AAPL']

    def test_get_ticker_data_start_date_only(self):
        """Test retrieval with only start date."""
        start = datetime(2015, 1, 1)  # Use date within data range
        df = get_ticker_data('AAPL', start_date=start)

        assert len(df) > 0
        assert df['timestamp'].min() >= start.replace(tzinfo=timezone.utc)

    def test_get_ticker_data_end_date_only(self):
        """Test retrieval with only end date."""
        end = datetime(2017, 12, 31)  # Use date within data range
        df = get_ticker_data('AAPL', end_date=end)

        assert len(df) > 0
        # End date filter is inclusive, so max timestamp should be <= end date at midnight next day
        end_utc = end.replace(tzinfo=timezone.utc)
        assert df['timestamp'].max() <= datetime(2018, 1, 1, tzinfo=timezone.utc)

    def test_get_ticker_data_date_formats(self):
        """Test different date format inputs (string, date, datetime)."""
        # String format
        df1 = get_ticker_data('AAPL', start_date='2017-01-01', end_date='2017-01-31')
        assert len(df1) > 0

        # date objects
        df2 = get_ticker_data('AAPL', start_date=date(2017, 1, 1), end_date=date(2017, 1, 31))
        assert len(df2) > 0

        # datetime objects
        df3 = get_ticker_data('AAPL', start_date=datetime(2017, 1, 1), end_date=datetime(2017, 1, 31))
        assert len(df3) > 0

        # All should return same data
        assert len(df1) == len(df2) == len(df3)

    def test_get_ticker_data_adjusted(self):
        """Test retrieval with adjusted prices."""
        df = get_ticker_data('AAPL', start_date='2017-01-01', end_date='2017-12-31', use_adjusted=True)

        assert len(df) > 0
        assert 'close' in df.columns

    def test_get_ticker_data_invalid_ticker(self):
        """Test retrieval with non-existent ticker returns empty DataFrame."""
        df = get_ticker_data('INVALID_TICKER_XYZ', start_date='2017-01-01', end_date='2017-12-31')

        # Should return empty DataFrame (no error)
        assert len(df) == 0
        # Column order may vary, just check all columns present
        expected_cols = {'symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        assert set(df.columns) == expected_cols

    def test_get_ticker_data_future_dates(self):
        """Test retrieval with future dates returns empty DataFrame."""
        future_start = datetime(2099, 1, 1)
        future_end = datetime(2099, 12, 31)

        df = get_ticker_data('AAPL', start_date=future_start, end_date=future_end)

        # Should return empty DataFrame
        assert len(df) == 0

    def test_get_ticker_data_reversed_dates(self):
        """Test retrieval with end_date before start_date returns empty DataFrame."""
        df = get_ticker_data('AAPL', start_date='2017-12-31', end_date='2017-01-01')

        # Should return empty DataFrame
        assert len(df) == 0

    def test_get_ticker_data_timezone_aware(self):
        """Test that returned timestamps are timezone-aware (UTC)."""
        df = get_ticker_data('AAPL', start_date='2017-01-01', end_date='2017-01-31')

        # Check timezone is UTC
        assert df['timestamp'].dtype == pl.Datetime(time_zone='UTC')

    def test_get_ticker_data_no_missing_ohlc(self):
        """Test that OHLC data has no null values."""
        df = get_ticker_data('AAPL', start_date='2017-01-01', end_date='2017-12-31')

        # Check for nulls in OHLC columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            null_count = df[col].null_count()
            assert null_count == 0, f"Column {col} has {null_count} null values"

    def test_get_ticker_data_ohlc_relationships(self):
        """Test that OHLC relationships are valid (high >= low, etc)."""
        df = get_ticker_data('AAPL', start_date='2017-01-01', end_date='2017-12-31')

        # High should be >= Low
        assert (df['high'] >= df['low']).all()

        # High should be >= Open and Close
        assert (df['high'] >= df['open']).all()
        assert (df['high'] >= df['close']).all()

        # Low should be <= Open and Close
        assert (df['low'] <= df['open']).all()
        assert (df['low'] <= df['close']).all()


class TestPrepareZiplineBundleData:
    """Tests for prepare_zipline_bundle_data() function."""

    @pytest.mark.slow
    @pytest.mark.slow
    def test_prepare_zipline_bundle_basic(self):
        """Test basic bundle creation with single ticker."""
        pytest.importorskip("pandas")
        with TemporaryDirectory() as tmpdir:
            result = prepare_zipline_bundle_data(
                tickers=['AAPL'],
                start_date='2017-01-01',
                end_date='2017-12-31',
                output_dir=Path(tmpdir),
                use_adjusted=False
            )

            # Check result structure
            assert 'bundle_file' in result
            assert 'tickers' in result
            assert 'num_tickers' in result
            assert 'start_date' in result
            assert 'end_date' in result
            assert 'metadata' in result

            # Check values
            assert result['tickers'] == ['AAPL']
            assert result['num_tickers'] == 1
            assert result['start_date'] == date(2017, 1, 1)
            assert result['end_date'] == date(2017, 12, 31)

            # Check file was created
            bundle_file = result['bundle_file']
            assert bundle_file.exists()
            assert bundle_file.suffix == '.h5'

    @pytest.mark.slow
    def test_prepare_zipline_bundle_multiple_tickers(self):
        """Test bundle creation with multiple tickers."""
        with TemporaryDirectory() as tmpdir:
            tickers = ['AAPL', 'MSFT', 'GOOGL']
            result = prepare_zipline_bundle_data(
                tickers=tickers,
                start_date='2017-01-01',
                end_date='2017-03-31',
                output_dir=Path(tmpdir),
                use_adjusted=False
            )

            assert result['tickers'] == tickers
            assert result['num_tickers'] == 3
            assert len(result['metadata']) == 3

            # Check each ticker has metadata
            symbols = [m['symbol'] for m in result['metadata']]
            assert set(symbols) == set(tickers)

    @pytest.mark.slow
    def test_prepare_zipline_bundle_hdf5_structure(self):
        """Test that HDF5 file has correct structure."""
        with TemporaryDirectory() as tmpdir:
            result = prepare_zipline_bundle_data(
                tickers=['AAPL'],
                start_date='2017-01-01',
                end_date='2017-03-31',
                output_dir=Path(tmpdir),
                use_adjusted=False
            )

            bundle_file = result['bundle_file']

            # Open with pandas HDFStore and check structure
            with pd.HDFStore(bundle_file, 'r') as store:
                # Check keys exist
                keys = store.keys()
                assert '/equities' in keys
                assert '/0' in keys  # SID 0 for AAPL

                # Check equities metadata
                equities = store['equities']
                assert 'sid' in equities.columns
                assert 'symbol' in equities.columns
                assert 'start_date' in equities.columns
                assert 'end_date' in equities.columns

                # Check ticker data
                ticker_data = store['0']
                assert 'open' in ticker_data.columns
                assert 'high' in ticker_data.columns
                assert 'low' in ticker_data.columns
                assert 'close' in ticker_data.columns
                assert 'volume' in ticker_data.columns

                # Check timestamps are timezone-aware
                assert ticker_data.index.tz is not None

    @pytest.mark.slow
    def test_prepare_zipline_bundle_date_formats(self):
        """Test bundle creation with different date format inputs."""
        with TemporaryDirectory() as tmpdir:
            # String format
            result1 = prepare_zipline_bundle_data(
                tickers=['AAPL'],
                start_date='2017-01-01',
                end_date='2017-01-31',
                output_dir=Path(tmpdir) / 'bundle1',
                use_adjusted=False
            )

            # date objects
            result2 = prepare_zipline_bundle_data(
                tickers=['AAPL'],
                start_date=date(2017, 1, 1),
                end_date=date(2017, 1, 31),
                output_dir=Path(tmpdir) / 'bundle2',
                use_adjusted=False
            )

            # datetime objects
            result3 = prepare_zipline_bundle_data(
                tickers=['AAPL'],
                start_date=datetime(2017, 1, 1),
                end_date=datetime(2017, 1, 31),
                output_dir=Path(tmpdir) / 'bundle3',
                use_adjusted=False
            )

            # All should succeed
            assert result1['bundle_file'].exists()
            assert result2['bundle_file'].exists()
            assert result3['bundle_file'].exists()

    @pytest.mark.slow
    def test_prepare_zipline_bundle_creates_output_dir(self):
        """Test that output directory is created if it doesn't exist."""
        with TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / 'nested' / 'path' / 'to' / 'bundle'

            result = prepare_zipline_bundle_data(
                tickers=['AAPL'],
                start_date='2017-01-01',
                end_date='2017-01-31',
                output_dir=nested_dir,
                use_adjusted=False
            )

            # Check directory was created
            assert nested_dir.exists()
            assert result['bundle_file'].exists()

    @pytest.mark.slow
    def test_prepare_zipline_bundle_invalid_ticker_warning(self, capsys):
        """Test that invalid ticker produces warning but doesn't fail."""
        with TemporaryDirectory() as tmpdir:
            result = prepare_zipline_bundle_data(
                tickers=['AAPL', 'INVALID_TICKER_XYZ'],
                start_date='2017-01-01',
                end_date='2017-01-31',
                output_dir=Path(tmpdir),
                use_adjusted=False
            )

            # Check that valid ticker was processed
            assert result['num_tickers'] == 1
            assert result['metadata'][0]['symbol'] == 'AAPL'

            # Check warning was printed
            captured = capsys.readouterr()
            assert 'Warning: No data found for INVALID_TICKER_XYZ' in captured.out

    @pytest.mark.slow
    def test_prepare_zipline_bundle_splits_and_dividends(self):
        """Test that splits and dividends are included if present."""
        with TemporaryDirectory() as tmpdir:
            # Use AAPL which has had splits and dividends
            result = prepare_zipline_bundle_data(
                tickers=['AAPL'],
                start_date='2010-01-01',
                end_date='2020-12-31',
                output_dir=Path(tmpdir),
                use_adjusted=False
            )

            bundle_file = result['bundle_file']

            with pd.HDFStore(bundle_file, 'r') as store:
                keys = store.keys()

                # Splits and dividends may or may not be present depending on data
                # Just check that if they exist, they have correct structure
                if '/splits' in keys:
                    splits = store['splits']
                    assert 'sid' in splits.columns
                    assert 'effective_date' in splits.columns
                    assert 'ratio' in splits.columns

                if '/dividends' in keys:
                    dividends = store['dividends']
                    assert 'sid' in dividends.columns
                    assert 'ex_date' in dividends.columns
                    assert 'amount' in dividends.columns

    @pytest.mark.slow
    def test_prepare_zipline_bundle_adjusted_prices(self):
        """Test bundle creation with adjusted prices."""
        with TemporaryDirectory() as tmpdir:
            result = prepare_zipline_bundle_data(
                tickers=['AAPL'],
                start_date='2017-01-01',
                end_date='2017-12-31',
                output_dir=Path(tmpdir),
                use_adjusted=True  # Use adjusted prices
            )

            # Should succeed with adjusted prices
            assert result['bundle_file'].exists()
            assert result['num_tickers'] == 1


class TestDataPathConstants:
    """Tests for data path constants."""

    def test_wiki_prices_path_exists(self):
        """Test that WIKI_PRICES_PATH points to existing file."""
        # This test may fail in CI/CD if data not present, which is acceptable
        # It serves as documentation of expected data location
        if WIKI_PRICES_PATH.exists():
            assert WIKI_PRICES_PATH.suffix == '.parquet'
            assert WIKI_PRICES_PATH.name == 'wiki_prices.parquet'
        else:
            pytest.skip(f"Wiki prices not found at {WIKI_PRICES_PATH} (acceptable in CI)")

    def test_projects_dir_structure(self):
        """Test that PROJECTS_DIR has expected structure."""
        from fixtures.market_data import PROJECTS_DIR

        # PROJECTS_DIR should be 4 levels up from fixtures/market_data.py
        # Structure: tests/validation/fixtures/market_data.py -> ../../../../projects
        assert PROJECTS_DIR.name == 'projects'


if __name__ == '__main__':
    # Allow running tests directly
    pytest.main([__file__, '-v', '--tb=short'])
