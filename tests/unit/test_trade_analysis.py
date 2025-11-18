"""Unit tests for trade analysis utilities."""

from datetime import timedelta

import polars as pl
import pytest

from ml4t.backtest.reporting.trade_analysis import (
    analyze_trades,
    avg_hold_time_by_rule,
    feature_correlation,
    pnl_attribution,
    rule_effectiveness,
    win_rate_by_rule,
)


@pytest.fixture
def sample_trades():
    """Create sample trade data for testing."""
    return pl.DataFrame(
        {
            "trade_id": [1, 2, 3, 4, 5, 6],
            "asset_id": ["AAPL", "MSFT", "GOOGL", "AAPL", "TSLA", "MSFT"],
            "exit_reason": [
                "stop_loss",
                "take_profit",
                "stop_loss",
                "time_stop",
                "take_profit",
                "risk_rule",
            ],
            "pnl": [-100.0, 200.0, -50.0, 150.0, 300.0, -75.0],
            "return_pct": [-0.05, 0.10, -0.025, 0.075, 0.15, -0.0375],
            "duration_bars": [10, 20, 15, 25, 30, 12],
            "duration_seconds": [3600.0, 7200.0, 5400.0, 9000.0, 10800.0, 4320.0],
            "atr_entry": [2.5, 3.0, 2.0, 3.5, 4.0, 2.8],
            "volatility_entry": [0.02, 0.03, 0.015, 0.035, 0.04, 0.028],
            "ml_score_entry": [0.6, 0.8, 0.5, 0.7, 0.9, 0.4],
        }
    )


class TestWinRateByRule:
    """Tests for win_rate_by_rule function."""

    def test_basic_win_rate(self, sample_trades):
        """Test basic win rate calculation."""
        result = win_rate_by_rule(sample_trades)

        assert "stop_loss" in result
        assert "take_profit" in result
        assert "time_stop" in result
        assert "risk_rule" in result

        # stop_loss: 0 wins out of 2 trades
        assert result["stop_loss"] == 0.0

        # take_profit: 2 wins out of 2 trades
        assert result["take_profit"] == 1.0

        # time_stop: 1 win out of 1 trade
        assert result["time_stop"] == 1.0

        # risk_rule: 0 wins out of 1 trade
        assert result["risk_rule"] == 0.0

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pl.DataFrame()
        result = win_rate_by_rule(empty_df)
        assert result == {}

    def test_no_completed_trades(self):
        """Test with no completed trades (all exit_reason null)."""
        df = pl.DataFrame(
            {"exit_reason": [None, None], "pnl": [100.0, -50.0]  # type: ignore
            }
        )
        result = win_rate_by_rule(df)
        assert result == {}

    def test_partial_data(self):
        """Test with some null values."""
        df = pl.DataFrame(
            {
                "exit_reason": ["stop_loss", "take_profit", None],  # type: ignore
                "pnl": [-100.0, 200.0, 150.0],
            }
        )
        result = win_rate_by_rule(df)

        # Should only count completed trades
        assert len(result) == 2
        assert result["stop_loss"] == 0.0
        assert result["take_profit"] == 1.0


class TestAvgHoldTimeByRule:
    """Tests for avg_hold_time_by_rule function."""

    def test_basic_hold_time(self, sample_trades):
        """Test basic hold time calculation."""
        result = avg_hold_time_by_rule(sample_trades)

        assert "stop_loss" in result
        assert "take_profit" in result

        # stop_loss: (3600 + 5400) / 2 = 4500 seconds
        assert result["stop_loss"] == timedelta(seconds=4500.0)

        # take_profit: (7200 + 10800) / 2 = 9000 seconds
        assert result["take_profit"] == timedelta(seconds=9000.0)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pl.DataFrame()
        result = avg_hold_time_by_rule(empty_df)
        assert result == {}

    def test_no_duration_data(self):
        """Test with no duration data."""
        df = pl.DataFrame(
            {
                "exit_reason": ["stop_loss", "take_profit"],
                "duration_seconds": [None, None],  # type: ignore
            }
        )
        result = avg_hold_time_by_rule(df)
        assert result == {}

    def test_single_trade_per_rule(self):
        """Test with single trade per rule."""
        df = pl.DataFrame(
            {
                "exit_reason": ["stop_loss", "take_profit"],
                "duration_seconds": [3600.0, 7200.0],
            }
        )
        result = avg_hold_time_by_rule(df)

        assert result["stop_loss"] == timedelta(seconds=3600.0)
        assert result["take_profit"] == timedelta(seconds=7200.0)


class TestPnlAttribution:
    """Tests for pnl_attribution function."""

    def test_basic_attribution(self, sample_trades):
        """Test basic P&L attribution."""
        result = pnl_attribution(sample_trades)

        assert "stop_loss" in result
        assert "take_profit" in result

        # stop_loss: -100 + -50 = -150
        assert result["stop_loss"] == -150.0

        # take_profit: 200 + 300 = 500
        assert result["take_profit"] == 500.0

        # time_stop: 150
        assert result["time_stop"] == 150.0

        # risk_rule: -75
        assert result["risk_rule"] == -75.0

    def test_sum_equals_total(self, sample_trades):
        """Test that sum of attribution equals total P&L."""
        result = pnl_attribution(sample_trades)
        total_from_attribution = sum(result.values())
        total_from_trades = sample_trades["pnl"].sum()

        assert abs(total_from_attribution - total_from_trades) < 0.01

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pl.DataFrame()
        result = pnl_attribution(empty_df)
        assert result == {}

    def test_no_pnl_data(self):
        """Test with no P&L data."""
        df = pl.DataFrame(
            {"exit_reason": ["stop_loss", "take_profit"], "pnl": [None, None]}  # type: ignore
        )
        result = pnl_attribution(df)
        assert result == {}


class TestRuleEffectiveness:
    """Tests for rule_effectiveness function."""

    def test_basic_effectiveness(self, sample_trades):
        """Test basic rule effectiveness metrics."""
        result = rule_effectiveness(sample_trades)

        assert not result.is_empty()
        assert "exit_reason" in result.columns
        assert "trigger_count" in result.columns
        assert "win_count" in result.columns
        assert "win_rate" in result.columns
        assert "total_pnl" in result.columns
        assert "avg_pnl" in result.columns
        assert "avg_return_pct" in result.columns
        assert "avg_duration_bars" in result.columns

    def test_effectiveness_calculations(self, sample_trades):
        """Test specific effectiveness calculations."""
        result = rule_effectiveness(sample_trades)

        # Find stop_loss row
        stop_loss_row = result.filter(pl.col("exit_reason") == "stop_loss")
        assert len(stop_loss_row) == 1

        # Check stop_loss metrics
        assert stop_loss_row["trigger_count"][0] == 2
        assert stop_loss_row["win_count"][0] == 0
        assert stop_loss_row["win_rate"][0] == 0.0
        assert stop_loss_row["total_pnl"][0] == -150.0
        assert stop_loss_row["avg_pnl"][0] == -75.0

        # Find take_profit row
        tp_row = result.filter(pl.col("exit_reason") == "take_profit")
        assert len(tp_row) == 1

        # Check take_profit metrics
        assert tp_row["trigger_count"][0] == 2
        assert tp_row["win_count"][0] == 2
        assert tp_row["win_rate"][0] == 1.0
        assert tp_row["total_pnl"][0] == 500.0
        assert tp_row["avg_pnl"][0] == 250.0

    def test_sorted_by_total_pnl(self, sample_trades):
        """Test that results are sorted by total P&L descending."""
        result = rule_effectiveness(sample_trades)

        # Verify descending order
        for i in range(len(result) - 1):
            assert result["total_pnl"][i] >= result["total_pnl"][i + 1]

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pl.DataFrame()
        result = rule_effectiveness(empty_df)
        assert result.is_empty()

    def test_single_rule(self):
        """Test with single exit rule."""
        df = pl.DataFrame(
            {
                "exit_reason": ["stop_loss", "stop_loss"],
                "pnl": [-100.0, -50.0],
                "return_pct": [-0.05, -0.025],
                "duration_bars": [10, 15],
            }
        )
        result = rule_effectiveness(df)

        assert len(result) == 1
        assert result["exit_reason"][0] == "stop_loss"
        assert result["trigger_count"][0] == 2
        assert result["win_rate"][0] == 0.0


class TestFeatureCorrelation:
    """Tests for feature_correlation function."""

    def test_basic_correlation(self, sample_trades):
        """Test basic feature correlation calculation."""
        result = feature_correlation(sample_trades)

        assert not result.is_empty()
        assert "feature" in result.columns
        assert "corr_pnl" in result.columns
        assert "corr_return_pct" in result.columns
        assert "corr_duration_bars" in result.columns

        # Should have detected _entry features
        features = result["feature"].to_list()
        assert "atr_entry" in features
        assert "volatility_entry" in features
        assert "ml_score_entry" in features

    def test_specific_features(self, sample_trades):
        """Test with specific feature list."""
        result = feature_correlation(sample_trades, features=["atr_entry"])

        assert len(result) == 1
        assert result["feature"][0] == "atr_entry"
        assert result["corr_pnl"][0] is not None

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pl.DataFrame()
        result = feature_correlation(empty_df)
        assert result.is_empty()

    def test_no_features(self):
        """Test with DataFrame containing no features."""
        df = pl.DataFrame({"exit_reason": ["stop_loss"], "pnl": [100.0]})
        result = feature_correlation(df)
        assert result.is_empty()

    def test_sorted_by_abs_correlation(self, sample_trades):
        """Test that results are sorted by absolute correlation with P&L."""
        result = feature_correlation(sample_trades)

        # Calculate absolute correlations
        abs_corrs = [abs(x) if x is not None else 0 for x in result["corr_pnl"].to_list()]

        # Verify descending order
        for i in range(len(abs_corrs) - 1):
            assert abs_corrs[i] >= abs_corrs[i + 1]

    def test_missing_feature_column(self):
        """Test with feature that doesn't exist in DataFrame."""
        df = pl.DataFrame({"pnl": [100.0, -50.0]})
        result = feature_correlation(df, features=["nonexistent_feature"])
        assert result.is_empty()

    def test_all_null_feature(self, sample_trades):
        """Test with feature that has all null values."""
        df = sample_trades.with_columns(
            pl.lit(None).cast(pl.Float64).alias("null_feature_entry")
        )
        result = feature_correlation(df, features=["null_feature_entry"])
        # Should skip features with all null values
        assert result.is_empty()


class TestAnalyzeTrades:
    """Tests for analyze_trades comprehensive function."""

    def test_basic_analysis(self, sample_trades):
        """Test basic comprehensive analysis."""
        result = analyze_trades(sample_trades)

        assert "summary" in result
        assert "by_rule" in result
        assert "win_rates" in result
        assert "hold_times" in result
        assert "pnl_attribution" in result
        assert "feature_correlations" in result

    def test_summary_statistics(self, sample_trades):
        """Test summary statistics calculation."""
        result = analyze_trades(sample_trades)
        summary = result["summary"]

        assert summary["total_trades"] == 6
        assert summary["winning_trades"] == 3
        assert summary["losing_trades"] == 3
        assert summary["overall_win_rate"] == 0.5

        # Total P&L: -100 + 200 - 50 + 150 + 300 - 75 = 425
        assert summary["total_pnl"] == 425.0

        assert summary["avg_pnl"] == pytest.approx(425.0 / 6, rel=1e-6)
        assert "max_win" in summary
        assert summary["max_win"] == 300.0
        assert "max_loss" in summary
        assert summary["max_loss"] == -100.0

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pl.DataFrame()
        result = analyze_trades(empty_df)

        assert result["summary"] == {}
        assert result["by_rule"].is_empty()
        assert result["win_rates"] == {}
        assert result["hold_times"] == {}
        assert result["pnl_attribution"] == {}
        assert result["feature_correlations"].is_empty()

    def test_no_completed_trades(self):
        """Test with no completed trades."""
        df = pl.DataFrame(
            {"exit_reason": [None, None], "pnl": [None, None]}  # type: ignore
        )
        result = analyze_trades(df)

        assert result["summary"] == {}

    def test_with_return_pct(self, sample_trades):
        """Test that return_pct is included in summary."""
        result = analyze_trades(sample_trades)
        summary = result["summary"]

        assert "avg_return_pct" in summary
        assert "median_return_pct" in summary

    def test_with_duration_bars(self, sample_trades):
        """Test that duration_bars is included in summary."""
        result = analyze_trades(sample_trades)
        summary = result["summary"]

        assert "avg_duration_bars" in summary
        assert "median_duration_bars" in summary


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_all_wins(self):
        """Test with all winning trades."""
        df = pl.DataFrame(
            {
                "exit_reason": ["take_profit", "time_stop"],
                "pnl": [100.0, 200.0],
                "return_pct": [0.05, 0.10],
                "duration_bars": [10, 20],
            }
        )
        result = analyze_trades(df)

        assert result["summary"]["overall_win_rate"] == 1.0
        assert all(wr == 1.0 for wr in result["win_rates"].values())

    def test_all_losses(self):
        """Test with all losing trades."""
        df = pl.DataFrame(
            {
                "exit_reason": ["stop_loss", "stop_loss"],
                "pnl": [-100.0, -50.0],
                "return_pct": [-0.05, -0.025],
                "duration_bars": [10, 15],
            }
        )
        result = analyze_trades(df)

        assert result["summary"]["overall_win_rate"] == 0.0
        assert all(wr == 0.0 for wr in result["win_rates"].values())

    def test_single_trade(self):
        """Test with single trade."""
        df = pl.DataFrame(
            {
                "exit_reason": ["take_profit"],
                "pnl": [100.0],
                "return_pct": [0.05],
                "duration_bars": [10],
                "duration_seconds": [3600.0],
            }
        )
        result = analyze_trades(df)

        assert result["summary"]["total_trades"] == 1
        assert result["summary"]["overall_win_rate"] == 1.0
        assert len(result["by_rule"]) == 1

    def test_zero_duration(self):
        """Test with zero duration trades."""
        df = pl.DataFrame(
            {
                "exit_reason": ["stop_loss"],
                "pnl": [-100.0],
                "duration_seconds": [0.0],
            }
        )
        result = avg_hold_time_by_rule(df)

        assert result["stop_loss"] == timedelta(seconds=0.0)

    def test_very_large_pnl(self):
        """Test with very large P&L values."""
        df = pl.DataFrame(
            {
                "exit_reason": ["take_profit"],
                "pnl": [1_000_000.0],
                "return_pct": [1.0],
                "duration_bars": [100],
            }
        )
        result = pnl_attribution(df)

        assert result["take_profit"] == 1_000_000.0

    def test_negative_duration(self):
        """Test handling of negative duration (data error)."""
        df = pl.DataFrame(
            {
                "exit_reason": ["stop_loss"],
                "pnl": [-100.0],
                "duration_seconds": [-3600.0],  # Invalid but should not crash
            }
        )
        result = avg_hold_time_by_rule(df)

        # Should still compute (garbage in, garbage out)
        assert "stop_loss" in result
