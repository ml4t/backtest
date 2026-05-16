"""Tests for BacktestResult and signal enrichment."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from ml4t.backtest.config import BacktestConfig
from ml4t.backtest.result import (
    BacktestResult,
    enrich_trades_with_signals,
)
from ml4t.backtest.types import Fill, OrderSide, Trade
from ml4t.specs.market_data import FeedSpec


@pytest.fixture
def sample_trades() -> list[Trade]:
    """Create sample trades for testing."""
    base_time = datetime(2024, 1, 1, 10, 0)
    return [
        Trade(
            symbol="AAPL",
            entry_time=base_time,
            exit_time=base_time + timedelta(hours=2),
            entry_price=150.0,
            exit_price=155.0,
            quantity=100.0,
            pnl=500.0,
            pnl_percent=3.33,
            bars_held=24,
            fees=10.0,
            exit_slippage=5.0,
            exit_reason="signal",
            mfe=4.0,
            mae=-1.0,
        ),
        Trade(
            symbol="MSFT",
            entry_time=base_time + timedelta(hours=3),
            exit_time=base_time + timedelta(hours=6),
            entry_price=300.0,
            exit_price=295.0,
            quantity=-50.0,  # Short
            pnl=250.0,
            pnl_percent=1.67,
            bars_held=36,
            fees=8.0,
            exit_slippage=3.0,
            exit_reason="stop_loss",
            mfe=2.5,
            mae=-0.5,
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
            rebalance_id="rebalance-1",
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
            rebalance_id="rebalance-1",
            commission=5.0,
            slippage=2.5,
        ),
    ]


@pytest.fixture
def sample_predictions() -> pl.DataFrame:
    """Create sample raw predictions used by a backtest."""
    base_time = datetime(2024, 1, 1, 10, 0)
    return pl.DataFrame(
        {
            "timestamp": [
                base_time,
                base_time,
                base_time + timedelta(hours=1),
                base_time + timedelta(hours=1),
            ],
            "asset": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "prediction": [0.8, -0.2, 0.6, 0.1],
            "confidence": [0.9, 0.4, 0.85, 0.55],
        }
    )


@pytest.fixture
def sample_portfolio_state() -> list[tuple[datetime, float, float, float, float, int]]:
    """Create sample portfolio state snapshots for testing."""
    base_time = datetime(2024, 1, 1, 10, 0)
    return [
        (base_time, 100000.0, 85000.0, 15000.0, 15000.0, 1),
        (base_time + timedelta(hours=1), 100100.0, 85000.0, 15100.0, 15100.0, 1),
        (base_time + timedelta(hours=2), 100500.0, 100500.0, 0.0, 0.0, 0),
        (base_time + timedelta(hours=3), 100400.0, 100400.0, 0.0, 0.0, 0),
        (base_time + timedelta(hours=4), 100800.0, 100800.0, 0.0, 0.0, 0),
        (base_time + timedelta(hours=5), 100750.0, 100750.0, 0.0, 0.0, 0),
    ]


@pytest.fixture
def backtest_result(
    sample_trades: list[Trade],
    sample_equity_curve: list[tuple[datetime, float]],
    sample_fills: list[Fill],
    sample_predictions: pl.DataFrame,
    sample_portfolio_state: list[tuple[datetime, float, float, float, float, int]],
) -> BacktestResult:
    """Create BacktestResult for testing."""
    return BacktestResult(
        trades=sample_trades,
        equity_curve=sample_equity_curve,
        fills=sample_fills,
        predictions=sample_predictions,
        portfolio_state=sample_portfolio_state,
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
            "symbol",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "quantity",
            "direction",
            "pnl",
            "pnl_percent",
            "bars_held",
            "fees",
            "exit_slippage",
            "mfe",
            "mae",
            "entry_slippage",
            "multiplier",
            "entry_quote_mid_price",
            "entry_bid_price",
            "entry_ask_price",
            "entry_spread",
            "entry_available_size",
            "exit_quote_mid_price",
            "exit_bid_price",
            "exit_ask_price",
            "exit_spread",
            "exit_available_size",
            "gross_pnl",
            "net_return",
            "total_slippage_cost",
            "cost_drag",
            "exit_reason",
            "status",
        ]

    def test_trades_dataframe_values(self, backtest_result: BacktestResult):
        """Test trades DataFrame values are correct."""
        df = backtest_result.to_trades_dataframe()

        # First trade (long)
        assert df["symbol"][0] == "AAPL"
        assert df["entry_price"][0] == 150.0
        assert df["exit_price"][0] == 155.0
        assert df["quantity"][0] == 100.0
        assert df["direction"][0] == "long"
        assert df["pnl"][0] == 500.0
        assert df["exit_reason"][0] == "signal"

        # Second trade (short)
        assert df["symbol"][1] == "MSFT"
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
        assert "symbol" in df.columns
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


class TestBacktestResultFillsDataFrame:
    """Tests for to_fills_dataframe()."""

    def test_fills_dataframe_basic(self, backtest_result: BacktestResult):
        df = backtest_result.to_fills_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2
        assert df.columns == [
            "order_id",
            "rebalance_id",
            "asset",
            "side",
            "quantity",
            "price",
            "timestamp",
            "commission",
            "slippage",
            "order_type",
            "limit_price",
            "stop_price",
            "price_source",
            "reference_price",
            "quote_mid_price",
            "bid_price",
            "ask_price",
            "spread",
            "bid_size",
            "ask_size",
            "available_size",
        ]
        assert df["rebalance_id"].to_list() == ["rebalance-1", "rebalance-1"]

    def test_fills_dataframe_empty(self):
        result = BacktestResult(trades=[], equity_curve=[], fills=[], metrics={})
        df = result.to_fills_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0
        assert "order_id" in df.columns


class TestBacktestResultPredictionsDataFrame:
    """Tests for to_predictions_dataframe()."""

    def test_predictions_dataframe_basic(self, backtest_result: BacktestResult):
        df = backtest_result.to_predictions_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 4
        assert df.columns == ["timestamp", "asset", "prediction", "confidence"]
        assert df["prediction"].to_list() == [0.8, -0.2, 0.6, 0.1]

    def test_predictions_dataframe_empty_when_absent(self):
        result = BacktestResult(trades=[], equity_curve=[], fills=[], metrics={})
        df = result.to_predictions_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert len(df.columns) == 0


class TestBacktestResultPortfolioStateDataFrame:
    """Tests for to_portfolio_state_dataframe()."""

    def test_portfolio_state_dataframe_basic(self, backtest_result: BacktestResult):
        df = backtest_result.to_portfolio_state_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 6
        assert df.columns == [
            "timestamp",
            "equity",
            "cash",
            "gross_exposure",
            "net_exposure",
            "open_positions",
        ]

    def test_portfolio_state_dataframe_values(self, backtest_result: BacktestResult):
        df = backtest_result.to_portfolio_state_dataframe()

        assert df["equity"][0] == 100000.0
        assert df["cash"][0] == 85000.0
        assert df["gross_exposure"][0] == 15000.0
        assert df["net_exposure"][2] == 0.0
        assert df["open_positions"][0] == 1
        assert df["open_positions"][2] == 0

    def test_portfolio_state_dataframe_empty(self):
        result = BacktestResult(trades=[], equity_curve=[], fills=[], metrics={})
        df = result.to_portfolio_state_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0
        assert "gross_exposure" in df.columns

    def test_portfolio_state_dataframe_caching(self, backtest_result: BacktestResult):
        df1 = backtest_result.to_portfolio_state_dataframe()
        df2 = backtest_result.to_portfolio_state_dataframe()

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

    def test_daily_returns_auto_aligns_using_feed_session_metadata(self):
        """Auto alignment should follow feed session metadata, not just calendar name."""
        from ml4t.backtest.config import BacktestConfig

        result = BacktestResult(
            trades=[],
            equity_curve=[
                (datetime(2024, 1, 1, 18, 0), 100000.0),
                (datetime(2024, 1, 2, 10, 0), 101000.0),
            ],
            fills=[],
            metrics={},
            config=BacktestConfig(
                calendar="NYSE",
                timezone="America/New_York",
                feed_spec=FeedSpec(
                    calendar="NYSE",
                    session_start_time="17:00",
                    timestamp_semantics="event_time",
                ),
            ),
        )

        assert len(result.to_daily_pnl()) == 2
        assert len(result.to_daily_returns()) == 1


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
    """Tests for to_dict()."""

    def test_to_dict_basic(self, backtest_result: BacktestResult):
        """Test dictionary conversion."""
        d = backtest_result.to_dict()

        assert isinstance(d, dict)
        assert "trades" in d
        assert "equity_curve" in d
        assert "fills" in d
        assert "predictions" in d
        assert "portfolio_state" in d
        assert "sharpe" in d

    def test_repr(self, backtest_result: BacktestResult):
        """Test string representation."""
        s = repr(backtest_result)
        assert "BacktestResult" in s
        assert "trades=2" in s

    def test_dict_like_accessors(self, backtest_result: BacktestResult):
        """Test __getitem__, get, keys, and items helpers."""
        assert backtest_result["sharpe"] == 1.5
        assert backtest_result.get("missing", 42) == 42
        assert "sharpe" in dict(backtest_result.items())
        assert ("sharpe", 1.5) in list(backtest_result.items())

    def test_to_dict_includes_optional_analytics(self):
        """Test to_dict includes equity and trade_analyzer when set."""
        result = BacktestResult(
            trades=[],
            equity_curve=[],
            fills=[],
            metrics={},
            equity=SimpleNamespace(name="eq"),
            trade_analyzer=SimpleNamespace(name="ta"),
        )
        d = result.to_dict()
        assert "equity" in d
        assert "trade_analyzer" in d

    def test_to_spec_dict_returns_resolved_config_snapshot(self):
        """Test resolved config snapshot contains defaults, feed, metadata, and runtime window."""
        config = BacktestConfig(
            initial_cash=250000.0,
            commission_rate=0.0025,
            feed_spec=FeedSpec(
                timestamp_col="time",
                entity_col="ticker",
                price_col="mid_price",
                calendar="NYSE",
                timezone="America/New_York",
                data_frequency="minute",
            ),
            metadata={
                "strategy_id": "topk_monthly_v2",
                "signals_path": "/tmp/preds.parquet",
            },
        )
        result = BacktestResult(
            trades=[],
            equity_curve=[
                (datetime(2024, 1, 2, 9, 30), 100000.0),
                (datetime(2024, 1, 31, 16, 0), 101500.0),
            ],
            fills=[],
            metrics={},
            config=config,
        )

        spec = result.to_spec_dict()

        assert spec["version"] == 1
        assert isinstance(spec["library_version"], str)
        assert spec["config"]["cash"]["initial"] == 250000.0
        assert spec["config"]["commission"]["rate"] == 0.0025
        assert spec["config"]["feed"]["timestamp_col"] == "time"
        assert spec["config"]["feed"]["entity_col"] == "ticker"
        assert spec["config"]["feed"]["price_col"] == "mid_price"
        assert spec["config"]["metadata"]["strategy_id"] == "topk_monthly_v2"
        assert spec["window"]["start"] == "2024-01-02T09:30:00"
        assert spec["window"]["end"] == "2024-01-31T16:00:00"


class TestBacktestResultParquet:
    """Tests for Parquet serialization."""

    def test_to_parquet_basic(self, backtest_result: BacktestResult):
        """Test basic Parquet export."""
        backtest_result.config = BacktestConfig(metadata={"strategy_id": "default_export"})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_backtest"
            written = backtest_result.to_parquet(path)

            assert "trades" in written
            assert "fills" in written
            assert "predictions" in written
            assert "equity" in written
            assert "portfolio_state" in written
            assert "daily_pnl" in written
            assert "metrics" in written
            assert "config" in written
            assert "spec" in written

            assert written["trades"].exists()
            assert written["fills"].exists()
            assert written["predictions"].exists()
            assert written["equity"].exists()
            assert written["portfolio_state"].exists()
            assert written["metrics"].exists()
            assert written["config"].exists()
            assert written["spec"].exists()

    def test_to_parquet_selective(self, backtest_result: BacktestResult):
        """Test selective Parquet export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_backtest"
            written = backtest_result.to_parquet(path, include=["trades", "metrics"])

            assert "trades" in written
            assert "metrics" in written
            assert "fills" not in written
            assert "predictions" not in written
            assert "equity" not in written
            assert "portfolio_state" not in written

    def test_to_parquet_config_write_failure_is_non_fatal(self):
        """Test config export failure is swallowed (ImportError/AttributeError path)."""

        class _BadConfig:
            def to_dict(self):
                raise AttributeError("no to_dict")

        result = BacktestResult(
            trades=[],
            equity_curve=[],
            fills=[],
            metrics={},
            config=_BadConfig(),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_backtest"
            written = result.to_parquet(path, include=["config"])
            assert "config" not in written

    def test_to_parquet_writes_spec_snapshot(self):
        """Test resolved runtime spec export."""
        config = BacktestConfig(
            initial_cash=75000.0,
            metadata={"strategy_id": "demo"},
        )
        result = BacktestResult(
            trades=[],
            equity_curve=[(datetime(2024, 1, 1, 10, 0), 75000.0)],
            fills=[],
            metrics={},
            config=config,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_backtest"
            written = result.to_parquet(path, include=["spec"])

            assert "spec" in written
            assert written["spec"].exists()

            import yaml

            with open(written["spec"]) as f:
                spec = yaml.safe_load(f)

            assert spec["config"]["cash"]["initial"] == 75000.0
            assert spec["config"]["metadata"]["strategy_id"] == "demo"
            assert spec["window"]["start"] == "2024-01-01T10:00:00"

    def test_to_parquet_writes_predictions_snapshot(self, sample_predictions: pl.DataFrame):
        result = BacktestResult(
            trades=[],
            equity_curve=[],
            fills=[],
            predictions=sample_predictions,
            metrics={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_backtest"
            written = result.to_parquet(path, include=["predictions"])

            assert "predictions" in written
            assert written["predictions"].exists()

            loaded = pl.read_parquet(written["predictions"])
            assert loaded.equals(sample_predictions)

    def test_from_parquet_roundtrip(self, backtest_result: BacktestResult):
        """Test Parquet save and load roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_backtest"
            backtest_result.to_parquet(path)

            loaded = BacktestResult.from_parquet(path)

            assert len(loaded.trades) == len(backtest_result.trades)
            assert len(loaded.fills) == len(backtest_result.fills)
            assert loaded.predictions is not None
            assert loaded.predictions.equals(backtest_result.predictions)
            assert len(loaded.equity_curve) == len(backtest_result.equity_curve)
            assert len(loaded.portfolio_state) == len(backtest_result.portfolio_state)
            assert loaded.fills[0].rebalance_id == "rebalance-1"
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

    def test_from_parquet_invalid_config_is_non_fatal(self, monkeypatch):
        """Test config load failures are swallowed and config remains None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "config.yaml").write_text("bad: [")
            # Force yaml.safe_load failure branch
            import yaml

            monkeypatch.setattr(
                yaml,
                "safe_load",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad yaml")),
            )
            loaded = BacktestResult.from_parquet(path)
            assert loaded.config is None

    def test_from_parquet_loads_config_from_spec_when_config_yaml_missing(self):
        """Test spec.yaml fallback restores replayable config."""
        config = BacktestConfig(
            initial_cash=82000.0,
            metadata={"strategy_id": "spec_fallback"},
        )
        result = BacktestResult(
            trades=[],
            equity_curve=[(datetime(2024, 2, 1, 10, 0), 82000.0)],
            fills=[],
            metrics={},
            config=config,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_backtest"
            result.to_parquet(path, include=["spec"])

            loaded = BacktestResult.from_parquet(path)

            assert loaded.config is not None
            assert loaded.config.initial_cash == 82000.0
            assert loaded.config.metadata["strategy_id"] == "spec_fallback"

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
        """Create sample trades DataFrame matching to_trades_dataframe() output."""
        return pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "MSFT"],
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
                "symbol": ["AAPL", "MSFT"],
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
        aapl_row = enriched.filter(pl.col("symbol") == "AAPL")
        msft_row = enriched.filter(pl.col("symbol") == "MSFT")

        assert aapl_row["entry_momentum"][0] == 0.5
        assert msft_row["entry_momentum"][0] == 0.3

    def test_enrich_multi_asset_requires_trade_asset_column(self):
        """Test multi-asset enrichment fails if trades have no asset/symbol column."""
        trades_df = pl.DataFrame(
            {
                "entry_time": [datetime(2024, 1, 1, 10, 0)],
                "exit_time": [datetime(2024, 1, 1, 14, 0)],
            }
        )
        signals_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 10, 0)],
                "asset": ["AAPL"],
                "momentum": [0.5],
            }
        )
        with pytest.raises(ValueError, match="requires trades_df to include"):
            enrich_trades_with_signals(
                trades_df,
                signals_df,
                signal_columns=["momentum"],
                asset_col="asset",
            )

    def test_enrich_multi_asset_rejects_unknown_trade_asset_column(self):
        """Test multi-asset enrichment validates explicit trades_asset_col."""
        trades_df = pl.DataFrame(
            {
                "symbol": ["AAPL"],
                "entry_time": [datetime(2024, 1, 1, 10, 0)],
                "exit_time": [datetime(2024, 1, 1, 14, 0)],
            }
        )
        signals_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 10, 0)],
                "asset": ["AAPL"],
                "momentum": [0.5],
            }
        )
        with pytest.raises(ValueError, match="not found in trades_df"):
            enrich_trades_with_signals(
                trades_df,
                signals_df,
                signal_columns=["momentum"],
                asset_col="asset",
                trades_asset_col="ticker",
            )

    def test_enrich_from_to_trades_dataframe(self, backtest_result: BacktestResult):
        """Integration: chain to_trades_dataframe() into enrich_trades_with_signals()."""
        trades_df = backtest_result.to_trades_dataframe()

        # Build signals covering both AAPL and MSFT entry/exit times
        signals_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 10, 0),
                    datetime(2024, 1, 1, 10, 0),
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 1, 16, 0),
                    datetime(2024, 1, 1, 16, 0),
                ],
                "asset": ["AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT"],
                "score": [0.8, 0.6, 0.9, 0.5, 0.7, 0.4],
            }
        )

        enriched = enrich_trades_with_signals(
            trades_df,
            signals_df,
            signal_columns=["score"],
            asset_col="asset",
        )

        assert "entry_score" in enriched.columns
        assert "exit_score" in enriched.columns
        assert len(enriched) == len(trades_df)
        # Verify the auto-detected trades_asset_col="symbol" worked
        assert "symbol" in enriched.columns


class TestBacktestResultSchemas:
    """Tests for schema definitions."""

    def test_trades_schema(self):
        """Test trades schema definition."""
        schema = BacktestResult._trades_schema()

        assert schema["symbol"] == pl.String()
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
