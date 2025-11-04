"""Unit tests for reporting functionality."""

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from qengine.core.event import FillEvent
from qengine.core.types import OrderSide
from qengine.portfolio.accounting import PortfolioAccounting
from qengine.reporting.html import HTMLReportGenerator
from qengine.reporting.parquet import ParquetReportGenerator


class TestReportGenerators:
    """Test suite for report generators."""

    @pytest.fixture
    def sample_accounting(self):
        """Create sample portfolio accounting with some trades."""
        accounting = PortfolioAccounting(initial_cash=100000.0)

        # Add some sample fills
        fills = [
            FillEvent(
                timestamp=datetime(2024, 1, 1, 9, 30),
                order_id="order1",
                trade_id="trade1",
                asset_id="AAPL",
                side=OrderSide.BUY,
                fill_quantity=100,
                fill_price=150.0,
                commission=1.0,
                slippage=0.5,
            ),
            FillEvent(
                timestamp=datetime(2024, 1, 2, 10, 30),
                order_id="order2",
                trade_id="trade2",
                asset_id="AAPL",
                side=OrderSide.SELL,
                fill_quantity=50,
                fill_price=155.0,
                commission=1.0,
                slippage=0.25,
            ),
            FillEvent(
                timestamp=datetime(2024, 1, 3, 11, 30),
                order_id="order3",
                trade_id="trade3",
                asset_id="GOOGL",
                side=OrderSide.BUY,
                fill_quantity=10,
                fill_price=2500.0,
                commission=2.0,
                slippage=1.0,
            ),
        ]

        for fill in fills:
            accounting.process_fill(fill)

        # Update prices for unrealized P&L
        accounting.update_prices({"AAPL": 160.0, "GOOGL": 2520.0}, datetime(2024, 1, 4))

        return accounting

    def test_html_report_generation(self, sample_accounting):
        """Test HTML report generation."""
        with TemporaryDirectory() as temp_dir:
            generator = HTMLReportGenerator(output_dir=Path(temp_dir), report_name="test_report")

            strategy_params = {"strategy": "buy_and_hold", "lookback": 20}
            backtest_params = {"start_date": "2024-01-01", "end_date": "2024-01-04"}

            report_path = generator.generate(
                sample_accounting,
                strategy_params=strategy_params,
                backtest_params=backtest_params,
            )

            # Check file was created
            assert report_path.exists()
            assert report_path.suffix == ".html"

            # Check content
            content = report_path.read_text()
            assert "QEngine Backtest Report" in content
            assert "test_report" in content
            assert "Performance Summary" in content
            assert "AAPL" in content
            assert "GOOGL" in content

    def test_parquet_report_generation(self, sample_accounting):
        """Test Parquet report generation."""
        with TemporaryDirectory() as temp_dir:
            generator = ParquetReportGenerator(
                output_dir=Path(temp_dir),
                report_name="test_parquet_report",
            )

            strategy_params = {"strategy": "mean_reversion", "window": 10}
            backtest_params = {"universe": ["AAPL", "GOOGL"], "rebalance": "daily"}

            report_dir = generator.generate(
                sample_accounting,
                strategy_params=strategy_params,
                backtest_params=backtest_params,
            )

            # Check directory was created
            assert report_dir.exists()
            assert report_dir.is_dir()

            # Check expected files exist
            expected_files = [
                "metadata.json",
                "performance_metrics.parquet",
                "equity_curve.parquet",
                "trades.parquet",
                "positions.parquet",
                "summary.parquet",
            ]

            for filename in expected_files:
                file_path = report_dir / filename
                assert file_path.exists(), f"Missing file: {filename}"

    def test_parquet_report_loading(self, sample_accounting):
        """Test loading a Parquet report."""
        with TemporaryDirectory() as temp_dir:
            generator = ParquetReportGenerator(
                output_dir=Path(temp_dir),
                report_name="load_test_report",
            )

            # Generate report
            report_dir = generator.generate(sample_accounting)

            # Load report
            loaded_data = generator.load_report(report_dir)

            # Check loaded data structure
            assert "metadata" in loaded_data
            assert "performance_metrics" in loaded_data
            assert "equity_curve" in loaded_data
            assert "trades" in loaded_data
            assert "positions" in loaded_data
            assert "summary" in loaded_data

            # Check trades data
            trades_df = loaded_data["trades"]
            assert len(trades_df) == 3  # We created 3 fills
            assert "AAPL" in trades_df["asset_id"].to_list()
            assert "GOOGL" in trades_df["asset_id"].to_list()

            # Check summary data
            summary_df = loaded_data["summary"]
            assert len(summary_df) == 1  # Single summary row
            summary_row = summary_df.row(0, named=True)
            assert summary_row["report_name"] == "load_test_report"
            assert summary_row["num_trades"] == 3

    def test_empty_accounting_reports(self):
        """Test report generation with empty accounting."""
        with TemporaryDirectory() as temp_dir:
            # Empty accounting (no trades)
            empty_accounting = PortfolioAccounting(initial_cash=50000.0)

            # Test HTML report
            html_generator = HTMLReportGenerator(
                output_dir=Path(temp_dir),
                report_name="empty_html_report",
            )
            html_report = html_generator.generate(empty_accounting)
            assert html_report.exists()

            content = html_report.read_text()
            assert "No trades found" in content

            # Test Parquet report
            parquet_generator = ParquetReportGenerator(
                output_dir=Path(temp_dir),
                report_name="empty_parquet_report",
            )
            parquet_report_dir = parquet_generator.generate(empty_accounting)
            assert parquet_report_dir.exists()

            # Check that files exist even with empty data
            trades_file = parquet_report_dir / "trades.parquet"
            assert trades_file.exists()

            # Load and check empty trades
            loaded_data = parquet_generator.load_report(parquet_report_dir)
            trades_df = loaded_data["trades"]
            assert len(trades_df) == 0

    def test_report_data_preparation(self, sample_accounting):
        """Test the base report data preparation."""
        with TemporaryDirectory() as temp_dir:
            generator = HTMLReportGenerator(output_dir=Path(temp_dir), report_name="data_prep_test")

            report_data = generator._prepare_report_data(
                sample_accounting,
                strategy_params={"test_param": "test_value"},
                backtest_params={"backtest_param": "backtest_value"},
            )

            # Check structure
            assert "metadata" in report_data
            assert "performance" in report_data
            assert "costs" in report_data
            assert "portfolio" in report_data
            assert "risk" in report_data
            assert "trades" in report_data
            assert "equity_curve" in report_data
            assert "positions" in report_data

            # Check metadata
            metadata = report_data["metadata"]
            assert metadata["strategy_params"]["test_param"] == "test_value"
            assert metadata["backtest_params"]["backtest_param"] == "backtest_value"

            # Check performance metrics
            performance = report_data["performance"]
            assert "total_return" in performance
            assert "total_pnl" in performance
            assert "sharpe_ratio" in performance
            assert "win_rate" in performance
            assert "profit_factor" in performance

            # Check that we have positive returns from our sample trades
            assert performance["total_return"] > 0
            assert performance["total_pnl"] > 0
            assert performance["num_trades"] == 3
