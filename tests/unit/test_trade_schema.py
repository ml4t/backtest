"""Tests for comprehensive trade recording schema."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from ml4t.backtest.reporting.trade_schema import (
    ExitReason,
    MLTradeRecord,
    append_trades,
    export_parquet,
    get_schema,
    import_parquet,
    polars_to_trades,
    trades_to_polars,
)


class TestMLTradeRecord:
    """Test MLTradeRecord dataclass."""

    def test_minimal_record_creation(self):
        """Test creating minimal trade record with required fields only."""
        trade = MLTradeRecord(
            trade_id=1,
            asset_id="BTC",
            direction="long",
            entry_dt=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            entry_price=50000.0,
            entry_quantity=1.0,
        )

        assert trade.trade_id == 1
        assert trade.asset_id == "BTC"
        assert trade.direction == "long"
        assert trade.entry_price == 50000.0
        assert trade.entry_quantity == 1.0
        assert trade.exit_dt is None
        assert trade.exit_reason == ExitReason.UNKNOWN
        assert trade.metadata == {}

    def test_full_record_creation(self):
        """Test creating comprehensive trade record with all fields."""
        trade = MLTradeRecord(
            # Core trade details
            trade_id=1,
            asset_id="BTC",
            direction="long",
            # Entry details
            entry_dt=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            entry_price=50000.0,
            entry_quantity=1.0,
            entry_commission=10.0,
            entry_slippage=5.0,
            entry_order_id="order_1",
            # Exit details
            exit_dt=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            exit_price=51000.0,
            exit_quantity=1.0,
            exit_commission=10.0,
            exit_slippage=5.0,
            exit_order_id="order_2",
            exit_reason=ExitReason.TAKE_PROFIT,
            # Trade metrics
            pnl=970.0,
            return_pct=1.94,
            duration_bars=2,
            duration_seconds=7200.0,
            # ML signals at entry
            ml_score_entry=0.85,
            predicted_return_entry=0.02,
            confidence_entry=0.9,
            # ML signals at exit
            ml_score_exit=0.15,
            predicted_return_exit=-0.01,
            confidence_exit=0.7,
            # Technical indicators at entry
            atr_entry=1000.0,
            volatility_entry=0.02,
            momentum_entry=0.05,
            rsi_entry=65.0,
            # Technical indicators at exit
            atr_exit=1100.0,
            volatility_exit=0.025,
            momentum_exit=-0.02,
            rsi_exit=70.0,
            # Risk management
            stop_loss_price=49000.0,
            take_profit_price=52000.0,
            risk_reward_ratio=2.0,
            position_size_pct=5.0,
            # Market context at entry
            vix_entry=15.0,
            market_regime_entry="bull",
            sector_performance_entry=0.03,
            # Market context at exit
            vix_exit=18.0,
            market_regime_exit="bull",
            sector_performance_exit=0.02,
            # Metadata
            metadata={"strategy": "ml_momentum", "version": "1.0"},
        )

        # Verify all fields
        assert trade.trade_id == 1
        assert trade.asset_id == "BTC"
        assert trade.direction == "long"
        assert trade.entry_price == 50000.0
        assert trade.exit_price == 51000.0
        assert trade.exit_reason == ExitReason.TAKE_PROFIT
        assert trade.ml_score_entry == 0.85
        assert trade.ml_score_exit == 0.15
        assert trade.atr_entry == 1000.0
        assert trade.vix_entry == 15.0
        assert trade.stop_loss_price == 49000.0
        assert trade.position_size_pct == 5.0
        assert trade.metadata["strategy"] == "ml_momentum"

    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        trade = MLTradeRecord(
            trade_id=1,
            asset_id="BTC",
            direction="long",
            entry_dt=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            entry_price=50000.0,
            entry_quantity=1.0,
            exit_reason=ExitReason.SIGNAL,
            ml_score_entry=0.85,
        )

        trade_dict = trade.to_dict()

        # Check dictionary contains all fields (not metadata)
        assert trade_dict["trade_id"] == 1
        assert trade_dict["asset_id"] == "BTC"
        assert trade_dict["direction"] == "long"
        assert trade_dict["entry_price"] == 50000.0
        assert trade_dict["exit_reason"] == "signal"
        assert trade_dict["ml_score_entry"] == 0.85
        assert "metadata" not in trade_dict  # Metadata excluded from dict

    def test_exit_reason_enum(self):
        """Test ExitReason enum values."""
        assert ExitReason.SIGNAL.value == "signal"
        assert ExitReason.STOP_LOSS.value == "stop_loss"
        assert ExitReason.TAKE_PROFIT.value == "take_profit"
        assert ExitReason.TIME_STOP.value == "time_stop"
        assert ExitReason.RISK_RULE.value == "risk_rule"
        assert ExitReason.POSITION_SIZE.value == "position_size"
        assert ExitReason.END_OF_DATA.value == "end_of_data"
        assert ExitReason.MANUAL.value == "manual"
        assert ExitReason.UNKNOWN.value == "unknown"


class TestPolarsConversion:
    """Test Polars DataFrame conversion functions."""

    def test_get_schema(self):
        """Test schema retrieval."""
        schema = get_schema()

        # Check core fields
        assert schema["trade_id"] == pl.Int64
        assert schema["asset_id"] == pl.Utf8
        assert schema["entry_dt"] == pl.Datetime
        assert schema["entry_price"] == pl.Float64
        assert schema["exit_reason"] == pl.Utf8

        # Check ML fields
        assert schema["ml_score_entry"] == pl.Float64
        assert schema["confidence_exit"] == pl.Float64

        # Check risk fields
        assert schema["stop_loss_price"] == pl.Float64
        assert schema["risk_reward_ratio"] == pl.Float64

        # Check context fields
        assert schema["vix_entry"] == pl.Float64
        assert schema["market_regime_entry"] == pl.Utf8

    def test_empty_trades_to_polars(self):
        """Test conversion of empty list to DataFrame."""
        df = trades_to_polars([])

        assert len(df) == 0
        # Check schema is correct
        schema = get_schema()
        for col_name, col_type in schema.items():
            assert col_name in df.columns
            assert df[col_name].dtype == col_type

    def test_single_trade_to_polars(self):
        """Test conversion of single trade to DataFrame."""
        trade = MLTradeRecord(
            trade_id=1,
            asset_id="BTC",
            direction="long",
            entry_dt=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            entry_price=50000.0,
            entry_quantity=1.0,
            exit_dt=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            exit_price=51000.0,
            exit_quantity=1.0,
            exit_reason=ExitReason.TAKE_PROFIT,
            pnl=970.0,
            ml_score_entry=0.85,
            vix_entry=15.0,
        )

        df = trades_to_polars([trade])

        assert len(df) == 1
        assert df["trade_id"][0] == 1
        assert df["asset_id"][0] == "BTC"
        assert df["entry_price"][0] == 50000.0
        assert df["exit_reason"][0] == "take_profit"
        assert df["ml_score_entry"][0] == 0.85
        assert df["vix_entry"][0] == 15.0

    def test_multiple_trades_to_polars(self):
        """Test conversion of multiple trades to DataFrame."""
        trades = [
            MLTradeRecord(
                trade_id=i,
                asset_id=f"ASSET_{i}",
                direction="long" if i % 2 == 0 else "short",
                entry_dt=datetime(2024, 1, i + 1, 10, 0, tzinfo=timezone.utc),
                entry_price=50000.0 + i * 1000,
                entry_quantity=1.0,
                ml_score_entry=0.5 + i * 0.1,
            )
            for i in range(5)
        ]

        df = trades_to_polars(trades)

        assert len(df) == 5
        assert df["trade_id"].to_list() == [0, 1, 2, 3, 4]
        assert df["asset_id"][0] == "ASSET_0"
        assert df["direction"][0] == "long"
        assert df["direction"][1] == "short"
        assert df["ml_score_entry"][0] == pytest.approx(0.5)
        assert df["ml_score_entry"][4] == pytest.approx(0.9)

    def test_polars_to_trades_roundtrip(self):
        """Test round-trip conversion: trades -> DataFrame -> trades."""
        original_trades = [
            MLTradeRecord(
                trade_id=1,
                asset_id="BTC",
                direction="long",
                entry_dt=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                entry_price=50000.0,
                entry_quantity=1.0,
                exit_reason=ExitReason.STOP_LOSS,
                ml_score_entry=0.85,
                atr_entry=1000.0,
                vix_entry=15.0,
                market_regime_entry="bull",
            ),
            MLTradeRecord(
                trade_id=2,
                asset_id="ETH",
                direction="short",
                entry_dt=datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc),
                entry_price=3000.0,
                entry_quantity=10.0,
                exit_reason=ExitReason.TAKE_PROFIT,
                confidence_entry=0.9,
                rsi_entry=70.0,
            ),
        ]

        # Convert to DataFrame and back
        df = trades_to_polars(original_trades)
        recovered_trades = polars_to_trades(df)

        # Check count
        assert len(recovered_trades) == len(original_trades)

        # Check first trade
        assert recovered_trades[0].trade_id == original_trades[0].trade_id
        assert recovered_trades[0].asset_id == original_trades[0].asset_id
        assert recovered_trades[0].direction == original_trades[0].direction
        assert recovered_trades[0].entry_price == original_trades[0].entry_price
        assert recovered_trades[0].exit_reason == ExitReason.STOP_LOSS
        assert recovered_trades[0].ml_score_entry == original_trades[0].ml_score_entry
        assert recovered_trades[0].vix_entry == original_trades[0].vix_entry
        assert recovered_trades[0].market_regime_entry == original_trades[0].market_regime_entry

        # Check second trade
        assert recovered_trades[1].trade_id == 2
        assert recovered_trades[1].asset_id == "ETH"
        assert recovered_trades[1].exit_reason == ExitReason.TAKE_PROFIT
        assert recovered_trades[1].confidence_entry == 0.9

    def test_polars_to_trades_empty_df(self):
        """Test conversion of empty DataFrame to trades."""
        df = pl.DataFrame(schema=get_schema())
        trades = polars_to_trades(df)

        assert len(trades) == 0
        assert isinstance(trades, list)


class TestParquetExport:
    """Test Parquet export and import functions."""

    def test_export_and_import_single_trade(self):
        """Test exporting and importing single trade."""
        trade = MLTradeRecord(
            trade_id=1,
            asset_id="BTC",
            direction="long",
            entry_dt=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            entry_price=50000.0,
            entry_quantity=1.0,
            ml_score_entry=0.85,
            vix_entry=15.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trades.parquet"

            # Export
            export_parquet([trade], path)
            assert path.exists()

            # Import
            df = import_parquet(path)
            assert len(df) == 1
            assert df["trade_id"][0] == 1
            assert df["asset_id"][0] == "BTC"
            assert df["ml_score_entry"][0] == 0.85

    def test_export_and_import_multiple_trades(self):
        """Test exporting and importing multiple trades."""
        trades = [
            MLTradeRecord(
                trade_id=i,
                asset_id=f"ASSET_{i}",
                direction="long",
                entry_dt=datetime(2024, 1, i + 1, 10, 0, tzinfo=timezone.utc),
                entry_price=50000.0 + i * 1000,
                entry_quantity=1.0,
                ml_score_entry=0.5 + i * 0.1,
                exit_reason=ExitReason.SIGNAL if i % 2 == 0 else ExitReason.STOP_LOSS,
            )
            for i in range(10)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trades.parquet"

            # Export
            export_parquet(trades, path)
            assert path.exists()

            # Import
            df = import_parquet(path)
            assert len(df) == 10
            assert df["trade_id"].to_list() == list(range(10))

            # Check exit reasons
            assert df["exit_reason"][0] == "signal"
            assert df["exit_reason"][1] == "stop_loss"

    def test_export_dataframe_directly(self):
        """Test exporting DataFrame directly (not just list)."""
        trades = [
            MLTradeRecord(
                trade_id=i,
                asset_id=f"ASSET_{i}",
                direction="long",
                entry_dt=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                entry_price=50000.0,
                entry_quantity=1.0,
            )
            for i in range(5)
        ]

        df = trades_to_polars(trades)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trades.parquet"

            # Export DataFrame
            export_parquet(df, path)
            assert path.exists()

            # Import and verify
            imported_df = import_parquet(path)
            assert len(imported_df) == 5

    def test_export_compression_options(self):
        """Test different compression options."""
        trade = MLTradeRecord(
            trade_id=1,
            asset_id="BTC",
            direction="long",
            entry_dt=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            entry_price=50000.0,
            entry_quantity=1.0,
        )

        compressions = ["zstd", "snappy", "gzip", "lz4", "uncompressed"]

        with tempfile.TemporaryDirectory() as tmpdir:
            for compression in compressions:
                path = Path(tmpdir) / f"trades_{compression}.parquet"

                # Export with specific compression
                export_parquet([trade], path, compression=compression)
                assert path.exists()

                # Import and verify
                df = import_parquet(path)
                assert len(df) == 1

    def test_export_creates_parent_directories(self):
        """Test that export creates parent directories if needed."""
        trade = MLTradeRecord(
            trade_id=1,
            asset_id="BTC",
            direction="long",
            entry_dt=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            entry_price=50000.0,
            entry_quantity=1.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "nested" / "trades.parquet"

            # Export (should create directories)
            export_parquet([trade], path)
            assert path.exists()
            assert path.parent.exists()

    def test_export_empty_list_raises_error(self):
        """Test that exporting empty list raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trades.parquet"

            with pytest.raises(ValueError, match="Cannot export empty trades DataFrame"):
                export_parquet([], path)

    def test_export_invalid_type_raises_error(self):
        """Test that exporting invalid type raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trades.parquet"

            with pytest.raises(ValueError, match="trades must be list"):
                export_parquet("invalid", path)  # type: ignore

    def test_import_nonexistent_file_raises_error(self):
        """Test that importing nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            import_parquet("/nonexistent/path/trades.parquet")


class TestIncrementalWrites:
    """Test incremental trade appending."""

    def test_append_to_new_file(self):
        """Test appending trades to new file (creates file)."""
        trades = [
            MLTradeRecord(
                trade_id=i,
                asset_id="BTC",
                direction="long",
                entry_dt=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                entry_price=50000.0,
                entry_quantity=1.0,
            )
            for i in range(3)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trades.parquet"

            # Append to new file
            append_trades(trades, path)
            assert path.exists()

            # Check contents
            df = import_parquet(path)
            assert len(df) == 3

    def test_append_to_existing_file(self):
        """Test appending trades to existing file."""
        initial_trades = [
            MLTradeRecord(
                trade_id=i,
                asset_id="BTC",
                direction="long",
                entry_dt=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                entry_price=50000.0,
                entry_quantity=1.0,
            )
            for i in range(3)
        ]

        new_trades = [
            MLTradeRecord(
                trade_id=i,
                asset_id="ETH",
                direction="short",
                entry_dt=datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc),
                entry_price=3000.0,
                entry_quantity=10.0,
            )
            for i in range(3, 6)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trades.parquet"

            # Create initial file
            export_parquet(initial_trades, path)
            df1 = import_parquet(path)
            assert len(df1) == 3

            # Append new trades
            append_trades(new_trades, path)

            # Check combined contents
            df2 = import_parquet(path)
            assert len(df2) == 6
            assert df2["asset_id"].to_list() == ["BTC"] * 3 + ["ETH"] * 3
            assert df2["trade_id"].to_list() == [0, 1, 2, 3, 4, 5]

    def test_append_dataframe(self):
        """Test appending DataFrame (not just list)."""
        initial_trades = [
            MLTradeRecord(
                trade_id=i,
                asset_id="BTC",
                direction="long",
                entry_dt=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                entry_price=50000.0,
                entry_quantity=1.0,
            )
            for i in range(3)
        ]

        new_trades_df = trades_to_polars(
            [
                MLTradeRecord(
                    trade_id=i,
                    asset_id="ETH",
                    direction="long",
                    entry_dt=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                    entry_price=3000.0,
                    entry_quantity=1.0,
                )
                for i in range(3, 5)
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trades.parquet"

            # Create initial file
            export_parquet(initial_trades, path)

            # Append DataFrame
            append_trades(new_trades_df, path)

            # Check combined
            df = import_parquet(path)
            assert len(df) == 5

    def test_append_empty_list_raises_error(self):
        """Test that appending empty list raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trades.parquet"

            with pytest.raises(ValueError, match="Cannot append empty trades DataFrame"):
                append_trades([], path)


class TestSchemaValidation:
    """Test schema validation and type safety."""

    def test_nullable_fields(self):
        """Test that optional fields can be None."""
        trade = MLTradeRecord(
            trade_id=1,
            asset_id="BTC",
            direction="long",
            entry_dt=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            entry_price=50000.0,
            entry_quantity=1.0,
            # All optional fields left as None
        )

        # Convert to DataFrame and back
        df = trades_to_polars([trade])
        recovered = polars_to_trades(df)

        # Check None values are preserved
        assert recovered[0].exit_dt is None
        assert recovered[0].exit_price is None
        assert recovered[0].ml_score_entry is None
        assert recovered[0].vix_entry is None

    def test_field_type_consistency(self):
        """Test that field types are consistent across conversions."""
        trade = MLTradeRecord(
            trade_id=1,
            asset_id="BTC",
            direction="long",
            entry_dt=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            entry_price=50000.0,
            entry_quantity=1.0,
            ml_score_entry=0.85,
            confidence_entry=0.9,
            atr_entry=1000.0,
            vix_entry=15.0,
            duration_bars=10,
        )

        df = trades_to_polars([trade])

        # Check types
        assert df["trade_id"].dtype == pl.Int64
        assert df["asset_id"].dtype == pl.Utf8
        assert df["entry_price"].dtype == pl.Float64
        assert df["ml_score_entry"].dtype == pl.Float64
        assert df["duration_bars"].dtype == pl.Int64
