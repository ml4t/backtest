"""Tests for backtest export utilities."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from ml4t.backtest.export import BacktestExporter
from ml4t.backtest.result import BacktestResult
from ml4t.backtest.types import Trade


@pytest.fixture
def sample_result() -> BacktestResult:
    """Create sample BacktestResult for testing."""
    base_time = datetime(2024, 1, 1, 10, 0)
    trades = [
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
        ),
    ]
    equity_curve = [
        (base_time, 100000.0),
        (base_time + timedelta(hours=1), 100200.0),
        (base_time + timedelta(hours=2), 100500.0),
    ]

    return BacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        fills=[],
        metrics={
            "final_value": 100500.0,
            "total_return_pct": 0.5,
            "sharpe": 1.5,
            "sortino": 2.0,
            "calmar": 3.0,
            "max_drawdown": -0.001,
            "max_drawdown_pct": -0.1,
            "num_trades": 1,
            "win_rate": 1.0,
            "profit_factor": 5.0,
            "cagr": 0.12,
            "expectancy": 500.0,
            "avg_trade": 500.0,
            "total_commission": 10.0,
            "total_slippage": 5.0,
        },
    )


@pytest.fixture
def multiple_results() -> list[BacktestResult]:
    """Create multiple BacktestResults for batch testing."""
    results = []
    base_time = datetime(2024, 1, 1, 10, 0)

    for i in range(3):
        trades = [
            Trade(
                asset="AAPL",
                entry_time=base_time,
                exit_time=base_time + timedelta(hours=2),
                entry_price=150.0,
                exit_price=150.0 + (i + 1) * 5,
                quantity=100.0,
                pnl=(i + 1) * 500.0,
                pnl_percent=(i + 1) * 3.33,
                bars_held=24,
                commission=10.0,
                slippage=5.0,
            ),
        ]
        equity_curve = [
            (base_time, 100000.0),
            (base_time + timedelta(hours=2), 100000.0 + (i + 1) * 500),
        ]

        results.append(
            BacktestResult(
                trades=trades,
                equity_curve=equity_curve,
                fills=[],
                metrics={
                    "final_value": 100000.0 + (i + 1) * 500,
                    "total_return_pct": (i + 1) * 0.5,
                    "sharpe": (i + 1) * 0.5,
                    "sortino": (i + 1) * 0.75,
                    "calmar": (i + 1) * 1.0,
                    "max_drawdown_pct": -0.1,
                    "num_trades": 1,
                    "win_rate": 1.0,
                    "profit_factor": 5.0,
                    "cagr": (i + 1) * 0.04,
                    "expectancy": (i + 1) * 500.0,
                    "avg_trade": (i + 1) * 500.0,
                    "total_commission": 10.0,
                    "total_slippage": 5.0,
                },
            )
        )

    return results


class TestBacktestExporterParquet:
    """Tests for Parquet export functions."""

    def test_to_parquet_delegation(self, sample_result: BacktestResult):
        """Test to_parquet delegates to BacktestResult."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_export"
            written = BacktestExporter.to_parquet(sample_result, path)

            assert "trades" in written
            assert "equity" in written
            assert written["trades"].exists()

    def test_from_parquet_delegation(self, sample_result: BacktestResult):
        """Test from_parquet delegates to BacktestResult."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_export"
            sample_result.to_parquet(path)

            loaded = BacktestExporter.from_parquet(path)

            assert len(loaded.trades) == 1
            assert loaded.metrics["sharpe"] == 1.5


class TestBacktestExporterBatch:
    """Tests for batch export functionality."""

    def test_batch_export_basic(self, multiple_results: list[BacktestResult]):
        """Test basic batch export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            param_values = [
                {"stop_pct": 0.02, "target_pct": 0.05},
                {"stop_pct": 0.03, "target_pct": 0.05},
                {"stop_pct": 0.02, "target_pct": 0.10},
            ]

            summary = BacktestExporter.batch_export(
                results=multiple_results,
                base_path=base_path,
                param_values=param_values,
            )

            # Check summary DataFrame
            assert isinstance(summary, pl.DataFrame)
            assert len(summary) == 3
            assert "stop_pct" in summary.columns
            assert "target_pct" in summary.columns
            assert "sharpe" in summary.columns
            assert "run_id" in summary.columns

            # Check files created
            assert (base_path / "sweep_summary.parquet").exists()
            assert (base_path / "sweep_summary.csv").exists()
            assert (base_path / "run_0001").is_dir()
            assert (base_path / "run_0002").is_dir()
            assert (base_path / "run_0003").is_dir()

    def test_batch_export_no_individual(self, multiple_results: list[BacktestResult]):
        """Test batch export without individual directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            param_values = [{"i": i} for i in range(3)]

            BacktestExporter.batch_export(
                results=multiple_results,
                base_path=base_path,
                param_values=param_values,
                export_individual=False,
            )

            # Summary should exist
            assert (base_path / "sweep_summary.parquet").exists()

            # Individual directories should NOT exist
            assert not (base_path / "run_0001").exists()

    def test_batch_export_mismatched_lengths(self, multiple_results: list[BacktestResult]):
        """Test batch export fails with mismatched lengths."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(ValueError, match="must have same length"),
        ):
            BacktestExporter.batch_export(
                results=multiple_results,
                base_path=tmpdir,
                param_values=[{"i": 1}],  # Only 1 param set for 3 results
            )

    def test_batch_export_summary_metrics(self, multiple_results: list[BacktestResult]):
        """Test batch export includes all metrics in summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            param_values = [{"i": i} for i in range(3)]

            summary = BacktestExporter.batch_export(
                results=multiple_results,
                base_path=tmpdir,
                param_values=param_values,
            )

            # Check all expected metrics are present
            expected_metrics = [
                "num_trades",
                "total_return_pct",
                "max_drawdown_pct",
                "sharpe",
                "sortino",
                "calmar",
                "cagr",
                "win_rate",
                "profit_factor",
                "expectancy",
                "avg_trade",
                "final_value",
                "total_commission",
                "total_slippage",
            ]

            for metric in expected_metrics:
                assert metric in summary.columns, f"Missing metric: {metric}"

    def test_load_sweep_summary(self, multiple_results: list[BacktestResult]):
        """Test loading sweep summary from exported data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            param_values = [{"stop": 0.02 + i * 0.01} for i in range(3)]

            BacktestExporter.batch_export(
                results=multiple_results,
                base_path=tmpdir,
                param_values=param_values,
            )

            loaded = BacktestExporter.load_sweep_summary(tmpdir)

            assert isinstance(loaded, pl.DataFrame)
            assert len(loaded) == 3
            assert "stop" in loaded.columns


class TestBacktestExporterJsonReport:
    """Tests for JSON report generation."""

    def test_generate_json_report_basic(self, multiple_results: list[BacktestResult]):
        """Test basic JSON report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"

            BacktestExporter.generate_json_report(
                results=multiple_results,
                output_path=output_path,
            )

            assert output_path.exists()

            with open(output_path) as f:
                report = json.load(f)

            assert "meta" in report
            assert "results" in report
            assert report["meta"]["num_results"] == 3
            assert len(report["results"]) == 3

    def test_generate_json_report_with_metadata(self, multiple_results: list[BacktestResult]):
        """Test JSON report with custom metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"

            BacktestExporter.generate_json_report(
                results=multiple_results,
                output_path=output_path,
                metadata={
                    "version": "1.0.0",
                    "git_hash": "abc123",
                },
            )

            with open(output_path) as f:
                report = json.load(f)

            assert report["meta"]["version"] == "1.0.0"
            assert report["meta"]["git_hash"] == "abc123"

    def test_generate_json_report_metrics(self, multiple_results: list[BacktestResult]):
        """Test JSON report contains correct metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"

            BacktestExporter.generate_json_report(
                results=multiple_results,
                output_path=output_path,
            )

            with open(output_path) as f:
                report = json.load(f)

            first_result = report["results"][0]
            assert "index" in first_result
            assert "num_trades" in first_result
            assert "total_return_pct" in first_result
            assert "sharpe" in first_result
            assert "win_rate" in first_result


class TestBacktestExporterMarkdownReport:
    """Tests for Markdown report generation."""

    def test_generate_markdown_report_basic(self, multiple_results: list[BacktestResult]):
        """Test basic Markdown report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.md"

            BacktestExporter.generate_markdown_report(
                results=multiple_results,
                output_path=output_path,
            )

            assert output_path.exists()

            content = output_path.read_text()
            assert "# Backtest Results" in content
            assert "## Summary" in content
            assert "## Results" in content

    def test_generate_markdown_report_custom_title(self, multiple_results: list[BacktestResult]):
        """Test Markdown report with custom title."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.md"

            BacktestExporter.generate_markdown_report(
                results=multiple_results,
                output_path=output_path,
                title="My Strategy Sweep",
            )

            content = output_path.read_text()
            assert "# My Strategy Sweep" in content

    def test_generate_markdown_report_with_params(self, multiple_results: list[BacktestResult]):
        """Test Markdown report with parameter sweep."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.md"
            param_names = ["stop_pct", "target_pct"]
            param_values = [
                {"stop_pct": 0.02, "target_pct": 0.05},
                {"stop_pct": 0.03, "target_pct": 0.05},
                {"stop_pct": 0.02, "target_pct": 0.10},
            ]

            BacktestExporter.generate_markdown_report(
                results=multiple_results,
                output_path=output_path,
                param_names=param_names,
                param_values=param_values,
            )

            content = output_path.read_text()
            # Should include parameter columns in header
            assert "stop_pct" in content
            assert "target_pct" in content
            # Should include parameter values
            assert "0.02" in content
            assert "0.05" in content

    def test_generate_markdown_report_empty_results(self):
        """Test Markdown report with empty results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.md"

            BacktestExporter.generate_markdown_report(
                results=[],
                output_path=output_path,
            )

            content = output_path.read_text()
            assert "Total runs: 0" in content

    def test_generate_markdown_report_summary_stats(self, multiple_results: list[BacktestResult]):
        """Test Markdown report includes summary statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.md"

            BacktestExporter.generate_markdown_report(
                results=multiple_results,
                output_path=output_path,
            )

            content = output_path.read_text()
            assert "Best Sharpe:" in content
            assert "Best Return:" in content
