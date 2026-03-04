"""Tests for EquityCurve annualization behavior."""

from datetime import datetime, timedelta

from ml4t.backtest.analytics.equity import EquityCurve


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
