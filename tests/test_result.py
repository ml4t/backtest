"""Tests for BacktestResult and signal enrichment."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from ml4t.backtest.result import BacktestResult, enrich_trades_with_signals
from ml4t.backtest.types import Fill, OrderSide, Trade


@pytest.fixture
def sample_trades() -> list[Trade]:
    """Create sample trades for testing."""
    base_time = datetime(2024, 1, 1, 10, 0)
    return [
        Trade(
            asset="AAPL",
            entry_time=base_time,
            exit_time=base_time + timedelta(hours=2),
            entry_price=150.0,
            exit_price=155.0,
            quantity=100.0,
            pnl=500.0,
            pnl_percent=3.33,
            bars_held=24,
            commission=10.0,
            slippage=5.0,
            exit_reason="signal",
            max_favorable_excursion=4.0,
            max_adverse_excursion=-1.0,
        ),
        Trade(
            asset="MSFT",
            entry_time=base_time + timedelta(hours=3),
            exit_time=base_time + timedelta(hours=6),
            entry_price=300.0,
            exit_price=295.0,
            quantity=-50.0,  # Short
            pnl=250.0,
            pnl_percent=1.67,
            bars_held=36,
            commission=8.0,
            slippage=3.0,
            exit_reason="stop_loss",
            max_favorable_excursion=2.5,
            max_adverse_excursion=-0.5,
        ),
    ]


@pytest.fixture
def sample_equity_curve() -> list[tuple[datetime, float]]:
    """Create sample equity curve for testing."""
    base_time = datetime(2024, 1, 1, 10, 0)
    return [
        (base_time, 100000.0),
        (base_time + timedelta(hours=1), 100100.0),
        (base_time + timedelta(hours=2), 100500.0),
        (base_time + timedelta(hours=3), 100400.0),
        (base_time + timedelta(hours=4), 100800.0),
        (base_time + timedelta(hours=5), 100750.0),
    ]


@pytest.fixture
def sample_fills() -> list[Fill]:
    """Create sample fills for testing."""
    base_time = datetime(2024, 1, 1, 10, 0)
    return [
        Fill(
            order_id="order_1",
            asset="AAPL",
            side=OrderSide.BUY,
            timestamp=base_time,
            quantity=100.0,
            price=150.0,
            commission=5.0,
            slippage=2.5,
        ),
        Fill(
            order_id="order_2",
            asset="AAPL",
            side=OrderSide.SELL,
            timestamp=base_time + timedelta(hours=2),
            quantity=100.0,
            price=155.0,
            commission=5.0,
            slippage=2.5,
        ),
    ]


@pytest.fixture
def backtest_result(
    sample_trades: list[Trade],
    sample_equity_curve: list[tuple[datetime, float]],
    sample_fills: list[Fill],
) -> BacktestResult:
    """Create BacktestResult for testing."""
    return BacktestResult(
        trades=sample_trades,
        equity_curve=sample_equity_curve,
        fills=sample_fills,
        metrics={
            "final_value": 100750.0,
            "total_return_pct": 0.75,
            "sharpe": 1.5,
            "max_drawdown": -0.001,
        },
    )


class TestBacktestResultTradesDataFrame:
    """Tests for to_trades_dataframe()."""

    def test_trades_dataframe_basic(self, backtest_result: BacktestResult):
        """Test basic trades DataFrame conversion."""
        df = backtest_result.to_trades_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2
        assert df.columns == [
            "asset",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "quantity",
            "direction",
            "pnl",
            "pnl_percent",
            "bars_held",
            "commission",
            "slippage",
            "mfe",
            "mae",
            "exit_reason",
        ]

    def test_trades_dataframe_values(self, backtest_result: BacktestResult):
        """Test trades DataFrame values are correct."""
        df = backtest_result.to_trades_dataframe()

        # First trade (long)
        assert df["asset"][0] == "AAPL"
        assert df["entry_price"][0] == 150.0
        assert df["exit_price"][0] == 155.0
        assert df["quantity"][0] == 100.0
        assert df["direction"][0] == "long"
        assert df["pnl"][0] == 500.0
        assert df["exit_reason"][0] == "signal"

        # Second trade (short)
        assert df["asset"][1] == "MSFT"
        assert df["direction"][1] == "short"
        assert df["exit_reason"][1] == "stop_loss"

    def test_trades_dataframe_empty(self):
        """Test empty trades returns empty DataFrame with schema."""
        result = BacktestResult(
            trades=[],
            equity_curve=[],
            fills=[],
            metrics={},
        )
        df = result.to_trades_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0
        # Should have correct schema even when empty
        assert "asset" in df.columns
        assert "exit_reason" in df.columns

    def test_trades_dataframe_caching(self, backtest_result: BacktestResult):
        """Test DataFrame is cached on repeated calls."""
        df1 = backtest_result.to_trades_dataframe()
        df2 = backtest_result.to_trades_dataframe()

        # Should be same object (cached)
        assert df1 is df2


class TestBacktestResultEquityDataFrame:
    """Tests for to_equity_dataframe()."""

    def test_equity_dataframe_basic(self, backtest_result: BacktestResult):
        """Test basic equity DataFrame conversion."""
        df = backtest_result.to_equity_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 6
        assert df.columns == [
            "timestamp",
            "equity",
            "return",
            "cumulative_return",
            "drawdown",
            "high_water_mark",
        ]

    def test_equity_dataframe_values(self, backtest_result: BacktestResult):
        """Test equity DataFrame values are correct."""
        df = backtest_result.to_equity_dataframe()

        # First row
        assert df["equity"][0] == 100000.0
        assert df["return"][0] == 0.0  # First bar has no return
        assert df["cumulative_return"][0] == 0.0
        assert df["drawdown"][0] == 0.0
        assert df["high_water_mark"][0] == 100000.0

        # After some gains
        assert df["equity"][2] == 100500.0
        assert df["high_water_mark"][2] == 100500.0

        # Check drawdown after peak
        assert df["drawdown"][3] < 0  # Should be negative after peak

    def test_equity_dataframe_empty(self):
        """Test empty equity curve returns empty DataFrame with schema."""
        result = BacktestResult(
            trades=[],
            equity_curve=[],
            fills=[],
            metrics={},
        )
        df = result.to_equity_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0
        assert "timestamp" in df.columns

    def test_equity_dataframe_caching(self, backtest_result: BacktestResult):
        """Test DataFrame is cached on repeated calls."""
        df1 = backtest_result.to_equity_dataframe()
        df2 = backtest_result.to_equity_dataframe()

        assert df1 is df2


class TestBacktestResultDailyPnL:
    """Tests for to_daily_pnl()."""

    def test_daily_pnl_basic(self, backtest_result: BacktestResult):
        """Test basic daily P&L aggregation."""
        df = backtest_result.to_daily_pnl()

        assert isinstance(df, pl.DataFrame)
        assert "date" in df.columns
        assert "pnl" in df.columns
        assert "return_pct" in df.columns

    def test_daily_pnl_empty(self):
        """Test empty equity curve returns empty DataFrame."""
        result = BacktestResult(
            trades=[],
            equity_curve=[],
            fills=[],
            metrics={},
        )
        df = result.to_daily_pnl()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

    def test_daily_pnl_multi_day(self):
        """Test daily P&L with multiple days."""
        equity_curve = [
            (datetime(2024, 1, 1, 10, 0), 100000.0),
            (datetime(2024, 1, 1, 14, 0), 100500.0),
            (datetime(2024, 1, 2, 10, 0), 100600.0),
            (datetime(2024, 1, 2, 14, 0), 101000.0),
            (datetime(2024, 1, 3, 10, 0), 100800.0),
        ]
        result = BacktestResult(
            trades=[],
            equity_curve=equity_curve,
            fills=[],
            metrics={},
        )
        df = result.to_daily_pnl()

        assert len(df) == 3  # 3 distinct days
        assert df["date"][0] == datetime(2024, 1, 1).date()
        assert df["date"][1] == datetime(2024, 1, 2).date()
        assert df["date"][2] == datetime(2024, 1, 3).date()


class TestBacktestResultReturnsSeries:
    """Tests for to_returns_series()."""

    def test_returns_series_basic(self, backtest_result: BacktestResult):
        """Test returns series extraction."""
        returns = backtest_result.to_returns_series()

        assert isinstance(returns, pl.Series)
        assert len(returns) == 6
        assert returns[0] == 0.0  # First bar has no return


class TestBacktestResultTradeRecords:
    """Tests for to_trade_records()."""

    def test_trade_records_basic(self, backtest_result: BacktestResult):
        """Test trade records conversion for diagnostics."""
        records = backtest_result.to_trade_records()

        assert isinstance(records, list)
        assert len(records) == 2

        # Check first record has diagnostic fields
        record = records[0]
        assert "timestamp" in record  # exit_time mapped to timestamp
        assert "symbol" in record  # asset mapped to symbol
        assert "entry_price" in record
        assert "exit_price" in record
        assert "pnl" in record
        assert "duration" in record
        assert "direction" in record


class TestBacktestResultDict:
    """Tests for to_dict() and dict-like access."""

    def test_to_dict_basic(self, backtest_result: BacktestResult):
        """Test dictionary conversion."""
        d = backtest_result.to_dict()

        assert isinstance(d, dict)
        assert "trades" in d
        assert "equity_curve" in d
        assert "fills" in d
        assert "sharpe" in d

    def test_getitem_metrics(self, backtest_result: BacktestResult):
        """Test dictionary-style access for metrics."""
        assert backtest_result["sharpe"] == 1.5
        assert backtest_result["final_value"] == 100750.0

    def test_getitem_attributes(self, backtest_result: BacktestResult):
        """Test dictionary-style access for attributes."""
        assert backtest_result["trades"] == backtest_result.trades
        assert backtest_result["equity_curve"] == backtest_result.equity_curve
        assert backtest_result["fills"] == backtest_result.fills

    def test_contains(self, backtest_result: BacktestResult):
        """Test 'in' operator."""
        assert "trades" in backtest_result
        assert "sharpe" in backtest_result
        assert "nonexistent" not in backtest_result

    def test_get_method(self, backtest_result: BacktestResult):
        """Test get() method with default."""
        assert backtest_result.get("sharpe") == 1.5
        assert backtest_result.get("nonexistent", 42) == 42

    def test_repr(self, backtest_result: BacktestResult):
        """Test string representation."""
        s = repr(backtest_result)
        assert "BacktestResult" in s
        assert "trades=2" in s


class TestBacktestResultParquet:
    """Tests for Parquet serialization."""

    def test_to_parquet_basic(self, backtest_result: BacktestResult):
        """Test basic Parquet export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_backtest"
            written = backtest_result.to_parquet(path)

            assert "trades" in written
            assert "equity" in written
            assert "daily_pnl" in written
            assert "metrics" in written

            assert written["trades"].exists()
            assert written["equity"].exists()
            assert written["metrics"].exists()

    def test_to_parquet_selective(self, backtest_result: BacktestResult):
        """Test selective Parquet export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_backtest"
            written = backtest_result.to_parquet(path, include=["trades", "metrics"])

            assert "trades" in written
            assert "metrics" in written
            assert "equity" not in written

    def test_from_parquet_roundtrip(self, backtest_result: BacktestResult):
        """Test Parquet save and load roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_backtest"
            backtest_result.to_parquet(path)

            loaded = BacktestResult.from_parquet(path)

            assert len(loaded.trades) == len(backtest_result.trades)
            assert len(loaded.equity_curve) == len(backtest_result.equity_curve)
            assert loaded.metrics["sharpe"] == backtest_result.metrics["sharpe"]

    def test_to_parquet_compression(self, backtest_result: BacktestResult):
        """Test different compression codecs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for codec in ["zstd", "lz4", "snappy"]:
                path = Path(tmpdir) / f"test_{codec}"
                written = backtest_result.to_parquet(path, compression=codec)
                assert written["trades"].exists()

    def test_from_parquet_empty_dir(self):
        """Test loading from directory without files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loaded = BacktestResult.from_parquet(tmpdir)
            assert len(loaded.trades) == 0
            assert len(loaded.equity_curve) == 0

    def test_metrics_json_serialization(self, backtest_result: BacktestResult):
        """Test metrics JSON contains only serializable values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_backtest"
            backtest_result.to_parquet(path)

            with open(path / "metrics.json") as f:
                metrics = json.load(f)

            assert isinstance(metrics["sharpe"], float)
            assert isinstance(metrics["final_value"], float)


class TestEnrichTradesWithSignals:
    """Tests for enrich_trades_with_signals()."""

    @pytest.fixture
    def trades_df(self) -> pl.DataFrame:
        """Create sample trades DataFrame."""
        return pl.DataFrame(
            {
                "asset": ["AAPL", "AAPL", "MSFT"],
                "entry_time": [
                    datetime(2024, 1, 1, 10, 0),
                    datetime(2024, 1, 2, 10, 0),
                    datetime(2024, 1, 1, 14, 0),
                ],
                "exit_time": [
                    datetime(2024, 1, 1, 14, 0),
                    datetime(2024, 1, 2, 16, 0),
                    datetime(2024, 1, 2, 10, 0),
                ],
                "pnl": [100.0, -50.0, 200.0],
            }
        )

    @pytest.fixture
    def signals_df(self) -> pl.DataFrame:
        """Create sample signals DataFrame."""
        return pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 10, 0),
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 1, 14, 0),
                    datetime(2024, 1, 2, 10, 0),
                    datetime(2024, 1, 2, 14, 0),
                    datetime(2024, 1, 2, 16, 0),
                ],
                "momentum": [0.5, 0.6, 0.7, 0.3, 0.4, 0.2],
                "rsi": [30.0, 45.0, 60.0, 55.0, 50.0, 40.0],
            }
        )

    def test_enrich_basic(self, trades_df: pl.DataFrame, signals_df: pl.DataFrame):
        """Test basic signal enrichment."""
        # Sort trades by entry_time for join_asof
        trades_sorted = trades_df.sort("entry_time")
        enriched = enrich_trades_with_signals(trades_sorted, signals_df)

        assert "entry_momentum" in enriched.columns
        assert "exit_momentum" in enriched.columns
        assert "entry_rsi" in enriched.columns
        assert "exit_rsi" in enriched.columns

    def test_enrich_selected_columns(self, trades_df: pl.DataFrame, signals_df: pl.DataFrame):
        """Test enrichment with selected signal columns."""
        trades_sorted = trades_df.sort("entry_time")
        enriched = enrich_trades_with_signals(
            trades_sorted, signals_df, signal_columns=["momentum"]
        )

        assert "entry_momentum" in enriched.columns
        assert "exit_momentum" in enriched.columns
        assert "entry_rsi" not in enriched.columns

    def test_enrich_values_correct(self, trades_df: pl.DataFrame, signals_df: pl.DataFrame):
        """Test enriched values are correctly joined."""
        trades_sorted = trades_df.sort("entry_time")
        enriched = enrich_trades_with_signals(
            trades_sorted, signals_df, signal_columns=["momentum"]
        )

        # First trade after sort: entry at 10:00 (momentum=0.5), exit at 14:00 (momentum=0.7)
        first_trade = enriched.filter(pl.col("entry_time") == datetime(2024, 1, 1, 10, 0))
        assert first_trade["entry_momentum"][0] == 0.5
        assert first_trade["exit_momentum"][0] == 0.7

    def test_enrich_empty_signals(self, trades_df: pl.DataFrame):
        """Test with no signal columns returns original."""
        signals_df = pl.DataFrame({"timestamp": [datetime(2024, 1, 1, 10, 0)]})
        enriched = enrich_trades_with_signals(trades_df, signals_df)

        # Should return original columns only
        assert len(enriched.columns) == len(trades_df.columns)

    def test_enrich_multi_asset(self):
        """Test enrichment with multi-asset signals."""
        trades_df = pl.DataFrame(
            {
                "asset": ["AAPL", "MSFT"],
                "entry_time": [
                    datetime(2024, 1, 1, 10, 0),
                    datetime(2024, 1, 1, 10, 0),
                ],
                "exit_time": [
                    datetime(2024, 1, 1, 14, 0),
                    datetime(2024, 1, 1, 14, 0),
                ],
                "pnl": [100.0, 200.0],
            }
        ).sort("entry_time")

        signals_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 10, 0),
                    datetime(2024, 1, 1, 10, 0),
                    datetime(2024, 1, 1, 14, 0),
                    datetime(2024, 1, 1, 14, 0),
                ],
                "asset": ["AAPL", "MSFT", "AAPL", "MSFT"],
                "momentum": [0.5, 0.3, 0.7, 0.4],
            }
        )

        enriched = enrich_trades_with_signals(
            trades_df, signals_df, signal_columns=["momentum"], asset_col="asset"
        )

        # AAPL entry momentum should be 0.5, MSFT should be 0.3
        aapl_row = enriched.filter(pl.col("asset") == "AAPL")
        msft_row = enriched.filter(pl.col("asset") == "MSFT")

        assert aapl_row["entry_momentum"][0] == 0.5
        assert msft_row["entry_momentum"][0] == 0.3


class TestBacktestResultSchemas:
    """Tests for schema definitions."""

    def test_trades_schema(self):
        """Test trades schema definition."""
        schema = BacktestResult._trades_schema()

        assert schema["asset"] == pl.String()
        assert schema["entry_time"] == pl.Datetime()
        assert schema["pnl"] == pl.Float64()
        assert schema["bars_held"] == pl.Int32()
        assert schema["exit_reason"] == pl.String()

    def test_equity_schema(self):
        """Test equity schema definition."""
        schema = BacktestResult._equity_schema()

        assert schema["timestamp"] == pl.Datetime()
        assert schema["equity"] == pl.Float64()
        assert schema["return"] == pl.Float64()
        assert schema["drawdown"] == pl.Float64()
