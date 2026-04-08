"""Tests for EquityCurve annualization behavior."""

from datetime import datetime, timedelta

import polars as pl

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.analytics.equity import EquityCurve
from ml4t.backtest.config import DataFrequency
from ml4t.specs.market_data import FeedSpec


class TestEquityCurveAnnualization:
    """Tests for time-aware annualization on intraday bars."""

    def test_years_uses_elapsed_time_for_intraday_bars(self):
        """Years should be based on elapsed time, not raw bar count."""
        eq = EquityCurve()
        start = datetime(2025, 1, 2, 9, 30)
        for i in range(390):
            eq.append(start + timedelta(minutes=i), 100_000.0 + float(i))

        assert 0.0 < eq.years < 0.01

    def test_periods_per_year_infers_intraday_frequency(self):
        """Annualization factor should rise for high-frequency bars."""
        eq = EquityCurve()
        start = datetime(2025, 1, 2, 9, 30)
        for i in range(6):
            eq.append(start + timedelta(minutes=i), 100_000.0 + float(i))

        assert eq.periods_per_year > 252.0

    def test_engine_equity_uses_configured_frequency_metadata(self):
        """Engine-built equity should prefer configured cadence over elapsed-time inference."""

        class HoldStrategy(Strategy):
            def on_data(self, timestamp, data, context, broker):
                return None

        prices = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 2, 9, 30), datetime(2025, 1, 2, 9, 31)],
                "asset": ["AAPL", "AAPL"],
                "close": [100.0, 101.0],
            }
        )
        engine = Engine(
            DataFeed(
                prices_df=prices,
                feed_spec=FeedSpec(calendar="NYSE", data_frequency="minute"),
            ),
            HoldStrategy(),
            BacktestConfig(data_frequency=DataFrequency.MINUTE_1, calendar="NYSE"),
        )

        result = engine.run()

        assert result.equity is not None
        assert result.equity.periods_per_year == 252.0 * 390.0

    def test_engine_equity_uses_default_config_assumptions_without_calendar(self):
        """Configured intraday cadence should not fall back to elapsed-time inference."""

        class HoldStrategy(Strategy):
            def on_data(self, timestamp, data, context, broker):
                return None

        prices = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 2, 9, 30), datetime(2025, 1, 2, 9, 31)],
                "asset": ["AAPL", "AAPL"],
                "close": [100.0, 101.0],
            }
        )
        engine = Engine(
            DataFeed(prices_df=prices),
            HoldStrategy(),
            BacktestConfig(data_frequency=DataFrequency.MINUTE_1),
        )

        result = engine.run()

        assert result.equity is not None
        assert result.equity.periods_per_year == 252.0 * 390.0
