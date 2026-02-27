"""Tests for the analytics bridge module - trade record conversion and utility functions."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ml4t.backtest.analytics.bridge import (
    to_equity_dataframe,
    to_returns_series,
    to_trade_record,
    to_trade_records,
)
from ml4t.backtest.types import Trade

# === Test Fixtures ===


@pytest.fixture
def sample_winning_trade() -> Trade:
    """A profitable long trade."""
    return Trade(
        symbol="AAPL",
        entry_time=datetime(2024, 1, 10, 10, 0),
        exit_time=datetime(2024, 1, 15, 15, 30),
        entry_price=150.0,
        exit_price=160.0,
        quantity=100,
        pnl=1000.0,
        pnl_percent=6.67,
        bars_held=5,
        fees=10.0,
        slippage=5.0,
        mfe=8.0,
        mae=-2.0,
    )


@pytest.fixture
def sample_losing_trade() -> Trade:
    """A losing long trade."""
    return Trade(
        symbol="MSFT",
        entry_time=datetime(2024, 1, 20, 9, 30),
        exit_time=datetime(2024, 1, 25, 16, 0),
        entry_price=400.0,
        exit_price=380.0,
        quantity=50,
        pnl=-1000.0,
        pnl_percent=-5.0,
        bars_held=5,
        fees=8.0,
        slippage=4.0,
        mfe=2.0,
        mae=-6.0,
    )


@pytest.fixture
def sample_short_trade() -> Trade:
    """A profitable short trade."""
    return Trade(
        symbol="TSLA",
        entry_time=datetime(2024, 2, 1, 10, 0),
        exit_time=datetime(2024, 2, 5, 14, 0),
        entry_price=250.0,
        exit_price=240.0,
        quantity=-100,  # Negative = short
        pnl=1000.0,
        pnl_percent=4.0,
        bars_held=4,
        fees=12.0,
        slippage=6.0,
        mfe=5.0,
        mae=-3.0,
    )


@pytest.fixture
def mixed_trades(
    sample_winning_trade: Trade,
    sample_losing_trade: Trade,
    sample_short_trade: Trade,
) -> list[Trade]:
    """Mix of winning/losing, long/short trades."""
    return [sample_winning_trade, sample_losing_trade, sample_short_trade]


@pytest.fixture
def sample_equity_curve() -> list[float]:
    """Sample equity curve with gains and losses."""
    return [100000, 101000, 100500, 102000, 103500, 103000, 105000]


@pytest.fixture
def sample_timestamps() -> list[datetime]:
    """Timestamps matching sample_equity_curve."""
    base = datetime(2024, 1, 1)
    return [base + timedelta(days=i) for i in range(7)]


# === Tests for to_trade_record ===


class TestToTradeRecord:
    """Tests for converting Trade to diagnostic record format."""

    def test_basic_long_trade(self, sample_winning_trade: Trade):
        """Test conversion of a basic winning long trade."""
        record = to_trade_record(sample_winning_trade)

        assert record["symbol"] == "AAPL"
        assert record["entry_price"] == 150.0
        assert record["exit_price"] == 160.0
        assert record["pnl"] == 1000.0
        assert record["direction"] == "long"
        assert record["quantity"] == 100
        assert record["timestamp"] == sample_winning_trade.exit_time
        assert record["entry_timestamp"] == sample_winning_trade.entry_time
        assert record["fees"] == 10.0
        assert record["slippage"] == 5.0

    def test_short_trade(self, sample_short_trade: Trade):
        """Test conversion of a short trade."""
        record = to_trade_record(sample_short_trade)

        assert record["direction"] == "short"
        assert record["quantity"] == -100  # Signed (negative for short)
        assert record["pnl"] == 1000.0

    def test_metadata_fields(self, sample_winning_trade: Trade):
        """Test that fields are correctly populated in aligned schema."""
        record = to_trade_record(sample_winning_trade)

        # These are now top-level fields in aligned schema (not in metadata)
        assert record["bars_held"] == 5
        assert record["pnl_percent"] == 6.67
        assert record["mfe"] == 8.0
        assert record["mae"] == -2.0
        # metadata is optional extension point (None by default)
        assert record["metadata"] is None

    def test_duration_calculation(self, sample_winning_trade: Trade):
        """Test that duration is correctly calculated."""
        record = to_trade_record(sample_winning_trade)

        expected_duration = sample_winning_trade.exit_time - sample_winning_trade.entry_time
        assert record["duration"] == expected_duration


# === Tests for to_trade_records ===


class TestToTradeRecords:
    """Tests for batch conversion of trades."""

    def test_empty_list(self):
        """Test with no trades."""
        records = to_trade_records([])
        assert records == []

    def test_multiple_trades(self, mixed_trades: list[Trade]):
        """Test conversion of multiple trades."""
        records = to_trade_records(mixed_trades)

        assert len(records) == 3
        assert records[0]["symbol"] == "AAPL"
        assert records[1]["symbol"] == "MSFT"
        assert records[2]["symbol"] == "TSLA"

    def test_preserves_order(self, mixed_trades: list[Trade]):
        """Test that trade order is preserved."""
        records = to_trade_records(mixed_trades)

        for i, (record, trade) in enumerate(zip(records, mixed_trades)):
            assert record["pnl"] == trade.pnl, f"Trade {i} PnL mismatch"


# === Tests for to_returns_series ===


class TestToReturnsSeries:
    """Tests for equity to returns conversion."""

    def test_basic_returns(self, sample_equity_curve: list[float]):
        """Test basic returns calculation."""
        returns = to_returns_series(sample_equity_curve)

        assert len(returns) == len(sample_equity_curve) - 1
        assert returns.dtype == pl.Float64
        assert returns.name == "returns"

        # First return: (101000 - 100000) / 100000 = 0.01
        assert pytest.approx(returns[0], rel=1e-4) == 0.01

    def test_empty_curve(self):
        """Test with empty equity curve."""
        returns = to_returns_series([])

        assert len(returns) == 0
        assert returns.dtype == pl.Float64

    def test_single_value(self):
        """Test with single value (no returns possible)."""
        returns = to_returns_series([100000])

        assert len(returns) == 0

    def test_numpy_array_input(self, sample_equity_curve: list[float]):
        """Test with numpy array input."""
        arr = np.array(sample_equity_curve)
        returns = to_returns_series(arr)

        assert len(returns) == len(sample_equity_curve) - 1

    def test_declining_equity(self):
        """Test with declining equity (negative returns)."""
        equity = [100000, 95000, 90000]
        returns = to_returns_series(equity)

        assert returns[0] < 0  # First return is negative
        assert returns[1] < 0  # Second return is negative


# === Tests for to_equity_dataframe ===


class TestToEquityDataframe:
    """Tests for equity curve DataFrame conversion."""

    def test_with_timestamps(
        self, sample_equity_curve: list[float], sample_timestamps: list[datetime]
    ):
        """Test DataFrame creation with timestamps."""
        df = to_equity_dataframe(sample_equity_curve, sample_timestamps)

        assert len(df) == len(sample_equity_curve)
        assert "timestamp" in df.columns
        assert "equity" in df.columns
        assert "returns" in df.columns
        assert df["timestamp"].to_list() == sample_timestamps

    def test_without_timestamps(self, sample_equity_curve: list[float]):
        """Test DataFrame creation without timestamps (uses bar index)."""
        df = to_equity_dataframe(sample_equity_curve)

        assert len(df) == len(sample_equity_curve)
        assert "bar" in df.columns
        assert "equity" in df.columns
        assert "returns" in df.columns
        assert df["bar"].to_list() == list(range(len(sample_equity_curve)))

    def test_empty_history(self):
        """Test with empty equity history."""
        df = to_equity_dataframe([])

        assert len(df) == 0
        assert "equity" in df.columns
        assert "returns" in df.columns

    def test_returns_calculation(self, sample_equity_curve: list[float]):
        """Test that returns are correctly calculated in DataFrame."""
        df = to_equity_dataframe(sample_equity_curve)

        # First return should be 0 (no prior value)
        assert df["returns"][0] == 0.0

        # Second return: (101000 - 100000) / 100000 = 0.01
        assert pytest.approx(df["returns"][1], rel=1e-4) == 0.01
