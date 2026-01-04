"""Tests for ml4t-diagnostic integration (optional dependency)."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from ml4t.backtest import BacktestResult
from ml4t.backtest.types import Trade


def create_sample_result(n_trades: int = 50) -> BacktestResult:
    """Create a sample BacktestResult for testing."""
    np.random.seed(42)
    trades = []
    base_time = datetime(2023, 1, 1)

    for i in range(n_trades):
        entry_time = base_time + timedelta(days=i * 2)
        exit_time = entry_time + timedelta(days=np.random.randint(1, 10))
        pnl = np.random.normal(50, 200)
        trades.append(
            Trade(
                asset=f"ASSET_{i % 5}",
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=100.0,
                exit_price=100.0 + pnl / 100,
                quantity=100.0,
                pnl=pnl,
                pnl_percent=pnl / 10000,
                bars_held=np.random.randint(1, 10),
                commission=5.0,
                slippage=2.0,
                exit_reason="signal",
                max_favorable_excursion=abs(np.random.normal(0.02, 0.01)),
                max_adverse_excursion=-abs(np.random.normal(0.01, 0.005)),
            )
        )

    equity = [
        (base_time + timedelta(days=i), 100000 + i * 100 + np.random.normal(0, 500))
        for i in range(252)
    ]

    return BacktestResult(
        trades=trades,
        equity_curve=equity,
        fills=[],
        metrics={
            "sharpe_ratio": 1.85,
            "max_drawdown": -0.12,
            "total_return_pct": 25.5,
            "final_value": 125500,
        },
    )


def has_diagnostic_library() -> bool:
    """Check if ml4t-diagnostic is available."""
    import importlib.util

    return importlib.util.find_spec("ml4t.diagnostic") is not None


@pytest.mark.skipif(not has_diagnostic_library(), reason="ml4t-diagnostic not installed")
class TestTearsheetIntegration:
    """Tests for BacktestResult.to_tearsheet() integration."""

    def test_to_tearsheet_basic(self):
        """Test basic tearsheet generation."""
        result = create_sample_result()
        html = result.to_tearsheet()

        assert isinstance(html, str)
        assert len(html) > 0
        assert "plotly" in html.lower()

    def test_to_tearsheet_templates(self):
        """Test tearsheet with different templates."""
        result = create_sample_result()

        for template in ["quant_trader", "hedge_fund", "risk_manager", "full"]:
            html = result.to_tearsheet(template=template)
            assert isinstance(html, str)
            assert len(html) > 0

    def test_to_tearsheet_themes(self):
        """Test tearsheet with different themes."""
        result = create_sample_result()

        for theme in ["default", "dark"]:
            html = result.to_tearsheet(theme=theme)
            assert isinstance(html, str)
            assert len(html) > 0

    def test_to_tearsheet_custom_title(self):
        """Test tearsheet with custom title."""
        result = create_sample_result()
        html = result.to_tearsheet(title="My Custom Backtest Report")

        assert isinstance(html, str)
        assert "My Custom Backtest Report" in html

    def test_to_tearsheet_save_to_file(self, tmp_path):
        """Test saving tearsheet to file."""
        result = create_sample_result()
        output_path = tmp_path / "tearsheet.html"

        html = result.to_tearsheet(output_path=output_path)

        assert output_path.exists()
        assert output_path.read_text() == html

    def test_to_tearsheet_empty_trades(self):
        """Test tearsheet with no trades."""
        result = BacktestResult(
            trades=[],
            equity_curve=[(datetime.now(), 100000.0)],
            fills=[],
            metrics={"sharpe_ratio": 0.0},
        )

        html = result.to_tearsheet()
        assert isinstance(html, str)

    def test_to_tearsheet_metrics_extraction(self):
        """Test that metrics are properly extracted."""
        result = create_sample_result()

        # Should not raise - metrics should be auto-populated
        html = result.to_tearsheet()
        assert isinstance(html, str)


class TestTearsheetMissingDependency:
    """Test behavior when ml4t-diagnostic is not installed."""

    @pytest.mark.skipif(has_diagnostic_library(), reason="ml4t-diagnostic IS installed")
    def test_import_error_when_diagnostic_missing(self):
        """Test that ImportError is raised with helpful message."""
        result = create_sample_result()

        with pytest.raises(ImportError, match="ml4t-diagnostic is required"):
            result.to_tearsheet()
