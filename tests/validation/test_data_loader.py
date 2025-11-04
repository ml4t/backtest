"""
Unit tests for UniversalDataLoader.
"""

import pandas as pd
import pytest

from .data_loader import UniversalDataLoader


class TestUniversalDataLoader:
    """Test suite for UniversalDataLoader."""

    @pytest.fixture
    def loader(self):
        """Create data loader instance."""
        return UniversalDataLoader()

    def test_init_valid_path(self, loader):
        """Test initialization with valid path."""
        assert loader.data_root.exists()
        assert "ml4t/projects" in str(loader.data_root)

    def test_init_invalid_path(self):
        """Test initialization with invalid path."""
        with pytest.raises(ValueError, match="Data root does not exist"):
            UniversalDataLoader(data_root="/nonexistent/path")

    # ============================================================================
    # Daily Equities Tests
    # ============================================================================

    def test_load_daily_equities_wiki(self, loader):
        """Test loading daily equities from wiki source."""
        df = loader.load_daily_equities(
            tickers=["AAPL"], start_date="2017-01-01", end_date="2017-01-31", source="wiki"
        )

        assert isinstance(df, pd.DataFrame)
        assert "timestamp" in df.columns
        assert "ticker" in df.columns
        assert "open" in df.columns
        assert "close" in df.columns
        assert len(df) > 0
        assert (df["ticker"] == "AAPL").all()

    def test_load_daily_equities_date_filtering(self, loader):
        """Test date filtering works correctly."""
        df = loader.load_daily_equities(
            tickers=["AAPL"], start_date="2017-01-01", end_date="2017-01-10", source="wiki"
        )

        assert df["timestamp"].min() >= pd.Timestamp("2017-01-01")
        assert df["timestamp"].max() <= pd.Timestamp("2017-01-10")

    def test_load_daily_equities_multiple_tickers(self, loader):
        """Test loading multiple tickers."""
        df = loader.load_daily_equities(
            tickers=["AAPL", "MSFT"], start_date="2017-01-01", end_date="2017-01-31", source="wiki"
        )

        tickers = df["ticker"].unique()
        assert len(tickers) <= 2
        assert all(t in ["AAPL", "MSFT"] for t in tickers)

    # ============================================================================
    # Minute Bar Tests
    # ============================================================================

    def test_load_minute_bars(self, loader):
        """Test loading minute bar data."""
        df = loader.load_minute_bars(year=2021, start_date="2021-01-04", end_date="2021-01-05")

        assert isinstance(df, pd.DataFrame)
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
        assert len(df) > 0

    def test_load_minute_bars_date_filtering(self, loader):
        """Test minute bar date filtering."""
        df = loader.load_minute_bars(year=2021, start_date="2021-01-04", end_date="2021-01-05")

        if len(df) > 0:  # Only test if data exists
            assert df["timestamp"].min().date() >= pd.Timestamp("2021-01-04").date()
            assert df["timestamp"].max().date() <= pd.Timestamp("2021-01-05").date()

    # ============================================================================
    # Crypto Tests
    # ============================================================================

    def test_load_crypto_futures(self, loader):
        """Test loading crypto futures data."""
        df = loader.load_crypto(
            symbol="BTC", data_type="futures", start_date="2020-01-01", end_date="2020-01-31"
        )

        assert isinstance(df, pd.DataFrame)
        assert "timestamp" in df.columns
        assert "ticker" in df.columns
        assert "open" in df.columns
        assert "close" in df.columns
        assert len(df) > 0
        assert (df["ticker"] == "BTC").all()

    def test_load_crypto_spot(self, loader):
        """Test loading crypto spot data."""
        df = loader.load_crypto(
            symbol="BTC", data_type="spot", start_date="2020-01-01", end_date="2020-01-31"
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_crypto_date_filtering(self, loader):
        """Test crypto date filtering."""
        df = loader.load_crypto(
            symbol="BTC", data_type="futures", start_date="2020-01-01", end_date="2020-01-10"
        )

        assert df["timestamp"].min() >= pd.Timestamp("2020-01-01", tz="UTC")
        assert df["timestamp"].max() <= pd.Timestamp("2020-01-10", tz="UTC")

    # ============================================================================
    # Order Flow Tests
    # ============================================================================

    def test_load_order_flow(self, loader):
        """Test loading SPY order flow data."""
        df = loader.load_order_flow()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    # ============================================================================
    # Format Conversion Tests
    # ============================================================================

    def test_to_vectorbt_format(self, loader):
        """Test conversion to VectorBT format."""
        df = loader.load_daily_equities(
            tickers=["AAPL"], start_date="2017-01-01", end_date="2017-01-31", source="wiki"
        )

        vectorbt_df = loader.to_framework_format(df, framework="vectorbt", symbol="AAPL")

        assert isinstance(vectorbt_df.index, pd.DatetimeIndex)
        assert "open" in vectorbt_df.columns
        assert "close" in vectorbt_df.columns
        assert "volume" in vectorbt_df.columns
        assert "ticker" not in vectorbt_df.columns  # Should be removed for single-symbol
        assert vectorbt_df.index.is_monotonic_increasing

    def test_to_zipline_format(self, loader):
        """Test conversion to Zipline format."""
        df = loader.load_daily_equities(
            tickers=["AAPL"], start_date="2017-01-01", end_date="2017-01-31", source="wiki"
        )

        zipline_df = loader.to_framework_format(df, framework="zipline", symbol="AAPL")

        assert isinstance(zipline_df.index, pd.DatetimeIndex)
        assert zipline_df.index.tz is not None  # Should have timezone
        assert str(zipline_df.index.tz) == "UTC"
        assert "open" in zipline_df.columns
        assert "close" in zipline_df.columns

    def test_to_backtrader_format(self, loader):
        """Test conversion to Backtrader format."""
        df = loader.load_daily_equities(
            tickers=["AAPL"], start_date="2017-01-01", end_date="2017-01-31", source="wiki"
        )

        bt_df = loader.to_framework_format(df, framework="backtrader", symbol="AAPL")

        assert isinstance(bt_df.index, pd.DatetimeIndex)
        assert all(col.islower() for col in bt_df.columns)  # Lowercase columns
        assert "open" in bt_df.columns
        assert "close" in bt_df.columns

    def test_to_qengine_format(self, loader):
        """Test conversion to QEngine format."""
        df = loader.load_daily_equities(
            tickers=["AAPL", "MSFT"], start_date="2017-01-01", end_date="2017-01-31", source="wiki"
        )

        qengine_df = loader.to_framework_format(df, framework="qengine", symbol="AAPL")

        assert isinstance(qengine_df.index, pd.DatetimeIndex)
        assert "open" in qengine_df.columns
        assert "close" in qengine_df.columns
        # For QEngine, ticker column can be present for multi-asset support
        if "ticker" in qengine_df.columns:
            assert (qengine_df["ticker"] == "AAPL").all()

    def test_invalid_framework(self, loader):
        """Test error handling for invalid framework."""
        df = loader.load_daily_equities(
            tickers=["AAPL"], start_date="2017-01-01", end_date="2017-01-31", source="wiki"
        )

        with pytest.raises(ValueError, match="Unknown framework"):
            loader.to_framework_format(df, framework="invalid", symbol="AAPL")

    # ============================================================================
    # Convenience Method Tests
    # ============================================================================

    def test_load_simple_equity_data(self, loader):
        """Test convenience method for simple equity loading."""
        df = loader.load_simple_equity_data(
            ticker="AAPL", start_date="2017-01-01", end_date="2017-12-31", framework="vectorbt"
        )

        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert "open" in df.columns
        assert "close" in df.columns
        assert len(df) > 0

    def test_load_crypto_simple(self, loader):
        """Test convenience method for simple crypto loading."""
        df = loader.load_crypto_simple(
            symbol="BTC", start_date="2020-01-01", end_date="2020-12-31", framework="vectorbt"
        )

        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert "open" in df.columns
        assert "close" in df.columns
        assert len(df) > 0

    # ============================================================================
    # Data Quality Tests
    # ============================================================================

    def test_no_missing_ohlcv_columns(self, loader):
        """Test that converted data has all OHLCV columns."""
        df = loader.load_simple_equity_data(ticker="AAPL", framework="vectorbt")

        required_cols = ["open", "high", "low", "close", "volume"]
        assert all(col in df.columns for col in required_cols)

    def test_sorted_timestamps(self, loader):
        """Test that data is sorted by timestamp."""
        df = loader.load_simple_equity_data(ticker="AAPL", framework="vectorbt")

        assert df.index.is_monotonic_increasing

    def test_no_duplicate_timestamps(self, loader):
        """Test that there are no duplicate timestamps for single symbol."""
        df = loader.load_simple_equity_data(ticker="AAPL", framework="vectorbt")

        assert not df.index.duplicated().any()
