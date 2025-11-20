"""Unit tests for trade visualization utilities."""

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import polars as pl
import pytest

from ml4t.backtest.reporting.visualizations import (
    plot_exit_reasons,
    plot_feature_importance,
    plot_hold_time_distribution,
    plot_mfe_mae_scatter,
    plot_rule_performance,
)


@pytest.fixture
def sample_trades():
    """Create sample trade data with comprehensive fields for testing."""
    return pl.DataFrame(
        {
            "trade_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "asset_id": ["AAPL", "MSFT", "GOOGL", "AAPL", "TSLA", "MSFT", "AMZN", "NVDA"],
            "exit_reason": [
                "stop_loss",
                "take_profit",
                "stop_loss",
                "time_stop",
                "take_profit",
                "risk_rule",
                "take_profit",
                "stop_loss",
            ],
            "pnl": [-100.0, 200.0, -50.0, 150.0, 300.0, -75.0, 250.0, -120.0],
            "return_pct": [-0.05, 0.10, -0.025, 0.075, 0.15, -0.0375, 0.125, -0.06],
            "duration_bars": [10, 20, 15, 25, 30, 12, 22, 18],
            "duration_seconds": [3600.0, 7200.0, 5400.0, 9000.0, 10800.0, 4320.0, 7920.0, 6480.0],
            "atr_entry": [2.5, 3.0, 2.0, 3.5, 4.0, 2.8, 3.2, 2.6],
            "volatility_entry": [0.02, 0.03, 0.015, 0.035, 0.04, 0.028, 0.032, 0.022],
            "ml_score_entry": [0.6, 0.8, 0.5, 0.7, 0.9, 0.4, 0.85, 0.55],
            "momentum_entry": [0.05, 0.08, 0.03, 0.09, 0.12, 0.02, 0.10, 0.04],
            "rsi_entry": [55.0, 65.0, 45.0, 70.0, 75.0, 40.0, 68.0, 50.0],
            "mfe": [50.0, 250.0, 30.0, 180.0, 350.0, 20.0, 300.0, 40.0],
            "mae": [-150.0, -30.0, -80.0, -40.0, -25.0, -100.0, -35.0, -140.0],
        }
    )


@pytest.fixture
def empty_trades():
    """Create empty DataFrame for edge case testing."""
    return pl.DataFrame()


@pytest.fixture
def minimal_trades():
    """Create minimal trade data (single trade)."""
    return pl.DataFrame(
        {
            "trade_id": [1],
            "exit_reason": ["take_profit"],
            "pnl": [100.0],
            "duration_bars": [20],
        }
    )


class TestPlotRulePerformance:
    """Tests for plot_rule_performance function."""

    def test_basic_plot(self, sample_trades):
        """Test basic rule performance plot generation."""
        fig = plot_rule_performance(sample_trades)

        assert fig is not None
        assert len(fig.axes) == 2  # Dual axes (win rate + P&L)

        plt.close(fig)

    def test_empty_dataframe(self, empty_trades):
        """Test with empty DataFrame."""
        fig = plot_rule_performance(empty_trades)

        assert fig is not None
        # Should create a figure with message, not crash
        assert len(fig.axes) >= 1

        plt.close(fig)

    def test_minimal_trade(self, minimal_trades):
        """Test with single trade."""
        fig = plot_rule_performance(minimal_trades)

        assert fig is not None
        assert len(fig.axes) >= 1

        plt.close(fig)

    def test_save_to_file(self, sample_trades):
        """Test saving plot to file."""
        with TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "rule_performance.png"
            fig = plot_rule_performance(sample_trades, save_path=save_path)

            assert save_path.exists()
            assert save_path.stat().st_size > 0

            plt.close(fig)

    def test_custom_figsize(self, sample_trades):
        """Test custom figure size."""
        fig = plot_rule_performance(sample_trades, figsize=(8, 4))

        assert fig is not None
        width, height = fig.get_size_inches()
        assert width == 8
        assert height == 4

        plt.close(fig)

    def test_missing_pnl_column(self):
        """Test with missing P&L column."""
        trades = pl.DataFrame(
            {
                "exit_reason": ["stop_loss", "take_profit"],
                # Missing 'pnl' column
            }
        )

        # Should not crash, should return figure with message
        fig = plot_rule_performance(trades)
        assert fig is not None

        plt.close(fig)

    def test_null_exit_reasons(self, sample_trades):
        """Test with some null exit reasons."""
        trades_with_nulls = sample_trades.with_columns(
            pl.when(pl.col("trade_id") == 1).then(None).otherwise(pl.col("exit_reason")).alias("exit_reason")  # type: ignore
        )

        fig = plot_rule_performance(trades_with_nulls)
        assert fig is not None

        plt.close(fig)


class TestPlotHoldTimeDistribution:
    """Tests for plot_hold_time_distribution function."""

    def test_basic_histogram(self, sample_trades):
        """Test basic hold time histogram."""
        fig = plot_hold_time_distribution(sample_trades)

        assert fig is not None
        assert len(fig.axes) == 1

        plt.close(fig)

    def test_empty_dataframe(self, empty_trades):
        """Test with empty DataFrame."""
        fig = plot_hold_time_distribution(empty_trades)

        assert fig is not None
        assert len(fig.axes) >= 1

        plt.close(fig)

    def test_custom_bins(self, sample_trades):
        """Test with custom number of bins."""
        fig = plot_hold_time_distribution(sample_trades, bins=10)

        assert fig is not None

        plt.close(fig)

    def test_missing_duration_column(self):
        """Test with missing duration_bars column."""
        trades = pl.DataFrame(
            {
                "trade_id": [1, 2, 3],
                "pnl": [100, -50, 200],
                # Missing 'duration_bars'
            }
        )

        fig = plot_hold_time_distribution(trades)
        assert fig is not None  # Should handle gracefully

        plt.close(fig)

    def test_null_durations(self, sample_trades):
        """Test with some null durations."""
        trades_with_nulls = sample_trades.with_columns(
            pl.when(pl.col("trade_id") == 1).then(None).otherwise(pl.col("duration_bars")).alias("duration_bars")  # type: ignore
        )

        fig = plot_hold_time_distribution(trades_with_nulls)
        assert fig is not None

        plt.close(fig)

    def test_single_duration_value(self):
        """Test with all trades having same duration."""
        trades = pl.DataFrame(
            {
                "trade_id": [1, 2, 3, 4],
                "duration_bars": [20, 20, 20, 20],
            }
        )

        fig = plot_hold_time_distribution(trades)
        assert fig is not None

        plt.close(fig)

    def test_save_to_file(self, sample_trades):
        """Test saving histogram to file."""
        with TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "hold_time_dist.png"
            fig = plot_hold_time_distribution(sample_trades, save_path=save_path)

            assert save_path.exists()
            assert save_path.stat().st_size > 0

            plt.close(fig)


class TestPlotFeatureImportance:
    """Tests for plot_feature_importance function."""

    def test_basic_feature_plot(self, sample_trades):
        """Test basic feature importance plot."""
        fig = plot_feature_importance(sample_trades)

        assert fig is not None
        assert len(fig.axes) == 1

        plt.close(fig)

    def test_empty_dataframe(self, empty_trades):
        """Test with empty DataFrame."""
        fig = plot_feature_importance(empty_trades)

        assert fig is not None
        assert len(fig.axes) >= 1

        plt.close(fig)

    def test_custom_top_n(self, sample_trades):
        """Test limiting to top N features."""
        fig = plot_feature_importance(sample_trades, top_n=3)

        assert fig is not None

        plt.close(fig)

    def test_no_features(self):
        """Test with no feature columns."""
        trades = pl.DataFrame(
            {
                "trade_id": [1, 2, 3],
                "pnl": [100, -50, 200],
                # No feature columns
            }
        )

        fig = plot_feature_importance(trades)
        assert fig is not None  # Should handle gracefully

        plt.close(fig)

    def test_single_feature(self):
        """Test with single feature column."""
        trades = pl.DataFrame(
            {
                "trade_id": [1, 2, 3, 4],
                "pnl": [100, -50, 200, -75],
                "atr_entry": [2.5, 3.0, 2.0, 3.5],
            }
        )

        fig = plot_feature_importance(trades)
        assert fig is not None

        plt.close(fig)

    def test_null_feature_values(self, sample_trades):
        """Test with some null feature values."""
        trades_with_nulls = sample_trades.with_columns(
            pl.when(pl.col("trade_id") == 1).then(None).otherwise(pl.col("atr_entry")).alias("atr_entry")  # type: ignore
        )

        fig = plot_feature_importance(trades_with_nulls)
        assert fig is not None

        plt.close(fig)

    def test_save_to_file(self, sample_trades):
        """Test saving feature importance to file."""
        with TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "feature_importance.png"
            fig = plot_feature_importance(sample_trades, save_path=save_path)

            assert save_path.exists()
            assert save_path.stat().st_size > 0

            plt.close(fig)


class TestPlotExitReasons:
    """Tests for plot_exit_reasons function."""

    def test_basic_pie_chart(self, sample_trades):
        """Test basic exit reasons pie chart."""
        fig = plot_exit_reasons(sample_trades)

        assert fig is not None
        assert len(fig.axes) == 1

        plt.close(fig)

    def test_empty_dataframe(self, empty_trades):
        """Test with empty DataFrame."""
        fig = plot_exit_reasons(empty_trades)

        assert fig is not None
        assert len(fig.axes) >= 1

        plt.close(fig)

    def test_single_exit_reason(self):
        """Test with all trades having same exit reason."""
        trades = pl.DataFrame(
            {
                "trade_id": [1, 2, 3, 4],
                "exit_reason": ["take_profit", "take_profit", "take_profit", "take_profit"],
            }
        )

        fig = plot_exit_reasons(trades)
        assert fig is not None

        plt.close(fig)

    def test_missing_exit_reason_column(self):
        """Test with missing exit_reason column."""
        trades = pl.DataFrame(
            {
                "trade_id": [1, 2, 3],
                "pnl": [100, -50, 200],
                # Missing 'exit_reason'
            }
        )

        fig = plot_exit_reasons(trades)
        assert fig is not None  # Should handle gracefully

        plt.close(fig)

    def test_null_exit_reasons(self, sample_trades):
        """Test with some null exit reasons."""
        trades_with_nulls = sample_trades.with_columns(
            pl.when(pl.col("trade_id") <= 2).then(None).otherwise(pl.col("exit_reason")).alias("exit_reason")  # type: ignore
        )

        fig = plot_exit_reasons(trades_with_nulls)
        assert fig is not None

        plt.close(fig)

    def test_save_to_file(self, sample_trades):
        """Test saving pie chart to file."""
        with TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "exit_reasons.png"
            fig = plot_exit_reasons(sample_trades, save_path=save_path)

            assert save_path.exists()
            assert save_path.stat().st_size > 0

            plt.close(fig)


class TestPlotMfeMaeScatter:
    """Tests for plot_mfe_mae_scatter function."""

    def test_basic_scatter(self, sample_trades):
        """Test basic MFE vs MAE scatter plot."""
        fig = plot_mfe_mae_scatter(sample_trades)

        assert fig is not None
        assert len(fig.axes) == 1

        plt.close(fig)

    def test_empty_dataframe(self, empty_trades):
        """Test with empty DataFrame."""
        fig = plot_mfe_mae_scatter(empty_trades)

        assert fig is not None
        assert len(fig.axes) >= 1

        plt.close(fig)

    def test_missing_mfe_mae_columns(self):
        """Test with missing MFE/MAE columns."""
        trades = pl.DataFrame(
            {
                "trade_id": [1, 2, 3],
                "pnl": [100, -50, 200],
                # Missing 'mfe' and 'mae'
            }
        )

        fig = plot_mfe_mae_scatter(trades)
        assert fig is not None  # Should handle gracefully

        plt.close(fig)

    def test_null_mfe_mae_values(self, sample_trades):
        """Test with some null MFE/MAE values."""
        trades_with_nulls = sample_trades.with_columns(
            [
                pl.when(pl.col("trade_id") == 1).then(None).otherwise(pl.col("mfe")).alias("mfe"),  # type: ignore
                pl.when(pl.col("trade_id") == 1).then(None).otherwise(pl.col("mae")).alias("mae"),  # type: ignore
            ]
        )

        fig = plot_mfe_mae_scatter(trades_with_nulls)
        assert fig is not None

        plt.close(fig)

    def test_single_trade(self):
        """Test with single trade."""
        trades = pl.DataFrame(
            {
                "trade_id": [1],
                "mfe": [150.0],
                "mae": [-30.0],
                "pnl": [100.0],
            }
        )

        fig = plot_mfe_mae_scatter(trades)
        assert fig is not None

        plt.close(fig)

    def test_winners_and_losers(self):
        """Test with mix of winning and losing trades."""
        trades = pl.DataFrame(
            {
                "trade_id": [1, 2, 3, 4],
                "mfe": [150.0, 80.0, 220.0, 50.0],
                "mae": [-30.0, -60.0, -40.0, -80.0],
                "pnl": [100.0, -50.0, 200.0, -75.0],  # 2 winners, 2 losers
            }
        )

        fig = plot_mfe_mae_scatter(trades)
        assert fig is not None

        plt.close(fig)

    def test_save_to_file(self, sample_trades):
        """Test saving scatter plot to file."""
        with TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "mfe_mae_scatter.png"
            fig = plot_mfe_mae_scatter(sample_trades, save_path=save_path)

            assert save_path.exists()
            assert save_path.stat().st_size > 0

            plt.close(fig)


class TestVisualizationIntegration:
    """Integration tests for all visualization functions together."""

    def test_all_plots_with_complete_data(self, sample_trades):
        """Test all visualization functions with complete trade data."""
        # Should not raise any exceptions
        fig1 = plot_rule_performance(sample_trades)
        fig2 = plot_hold_time_distribution(sample_trades)
        fig3 = plot_feature_importance(sample_trades)
        fig4 = plot_exit_reasons(sample_trades)
        fig5 = plot_mfe_mae_scatter(sample_trades)

        assert all(fig is not None for fig in [fig1, fig2, fig3, fig4, fig5])

        for fig in [fig1, fig2, fig3, fig4, fig5]:
            plt.close(fig)

    def test_all_plots_with_minimal_data(self, minimal_trades):
        """Test all visualization functions with minimal trade data."""
        # Should not raise any exceptions, even with minimal data
        fig1 = plot_rule_performance(minimal_trades)
        fig2 = plot_hold_time_distribution(minimal_trades)
        fig3 = plot_feature_importance(minimal_trades)
        fig4 = plot_exit_reasons(minimal_trades)

        # MFE/MAE will fail gracefully (missing columns)
        fig5 = plot_mfe_mae_scatter(minimal_trades)

        assert all(fig is not None for fig in [fig1, fig2, fig3, fig4, fig5])

        for fig in [fig1, fig2, fig3, fig4, fig5]:
            plt.close(fig)

    def test_save_all_plots(self, sample_trades):
        """Test saving all plots to files."""
        with TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            plot_rule_performance(sample_trades, save_path=tmppath / "rule_perf.png")
            plot_hold_time_distribution(sample_trades, save_path=tmppath / "hold_time.png")
            plot_feature_importance(sample_trades, save_path=tmppath / "features.png")
            plot_exit_reasons(sample_trades, save_path=tmppath / "exit_reasons.png")
            plot_mfe_mae_scatter(sample_trades, save_path=tmppath / "mfe_mae.png")

            # Check all files exist
            assert (tmppath / "rule_perf.png").exists()
            assert (tmppath / "hold_time.png").exists()
            assert (tmppath / "features.png").exists()
            assert (tmppath / "exit_reasons.png").exists()
            assert (tmppath / "mfe_mae.png").exists()

            # Check all files have content
            assert (tmppath / "rule_perf.png").stat().st_size > 0
            assert (tmppath / "hold_time.png").stat().st_size > 0
            assert (tmppath / "features.png").stat().st_size > 0
            assert (tmppath / "exit_reasons.png").stat().st_size > 0
            assert (tmppath / "mfe_mae.png").stat().st_size > 0


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_extremely_large_dataset(self):
        """Test with large number of trades (performance check)."""
        # Create 10,000 synthetic trades
        n_trades = 10000
        trades = pl.DataFrame(
            {
                "trade_id": list(range(n_trades)),
                "exit_reason": ["take_profit" if i % 2 == 0 else "stop_loss" for i in range(n_trades)],
                "pnl": [100.0 if i % 2 == 0 else -50.0 for i in range(n_trades)],
                "duration_bars": [20 + (i % 30) for i in range(n_trades)],
                "atr_entry": [2.5 + (i % 10) * 0.1 for i in range(n_trades)],
                "mfe": [150.0 if i % 2 == 0 else 80.0 for i in range(n_trades)],
                "mae": [-30.0 if i % 2 == 0 else -60.0 for i in range(n_trades)],
            }
        )

        # Should handle large dataset without crashing
        fig1 = plot_rule_performance(trades)
        fig2 = plot_hold_time_distribution(trades)
        fig3 = plot_exit_reasons(trades)
        fig4 = plot_mfe_mae_scatter(trades)

        assert all(fig is not None for fig in [fig1, fig2, fig3, fig4])

        for fig in [fig1, fig2, fig3, fig4]:
            plt.close(fig)

    def test_all_negative_pnl(self):
        """Test with all losing trades."""
        trades = pl.DataFrame(
            {
                "trade_id": [1, 2, 3, 4],
                "exit_reason": ["stop_loss", "risk_rule", "stop_loss", "time_stop"],
                "pnl": [-100.0, -50.0, -75.0, -25.0],
                "duration_bars": [10, 15, 12, 18],
                "atr_entry": [2.5, 3.0, 2.8, 2.2],
                "mfe": [20.0, 30.0, 15.0, 40.0],
                "mae": [-120.0, -80.0, -95.0, -60.0],
            }
        )

        fig1 = plot_rule_performance(trades)
        fig2 = plot_mfe_mae_scatter(trades)

        assert fig1 is not None
        assert fig2 is not None

        plt.close(fig1)
        plt.close(fig2)

    def test_all_positive_pnl(self):
        """Test with all winning trades."""
        trades = pl.DataFrame(
            {
                "trade_id": [1, 2, 3, 4],
                "exit_reason": ["take_profit", "time_stop", "take_profit", "take_profit"],
                "pnl": [100.0, 50.0, 75.0, 25.0],
                "duration_bars": [10, 15, 12, 18],
                "atr_entry": [2.5, 3.0, 2.8, 2.2],
                "mfe": [120.0, 80.0, 95.0, 60.0],
                "mae": [-20.0, -10.0, -15.0, -5.0],
            }
        )

        fig1 = plot_rule_performance(trades)
        fig2 = plot_mfe_mae_scatter(trades)

        assert fig1 is not None
        assert fig2 is not None

        plt.close(fig1)
        plt.close(fig2)
