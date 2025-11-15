"""Unit tests for PerformanceAnalyzer."""

from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

from ml4t.backtest.portfolio.analytics import PerformanceAnalyzer
from ml4t.backtest.portfolio.core import PositionTracker
from ml4t.backtest.portfolio.state import Position


class TestPerformanceAnalyzerInitialization:
    """Test PerformanceAnalyzer initialization."""

    def test_init_with_tracker(self):
        """Test initialization with a PositionTracker."""
        tracker = PositionTracker(initial_cash=100000)
        analyzer = PerformanceAnalyzer(tracker)

        assert analyzer.tracker is tracker
        assert analyzer.high_water_mark == 100000
        assert analyzer.max_drawdown == 0.0
        assert analyzer.daily_returns == []
        assert analyzer.timestamps == []
        assert analyzer.equity_curve == []
        assert analyzer.max_leverage == 0.0
        assert analyzer.max_concentration == 0.0

    def test_init_with_mock_tracker(self):
        """Test initialization with a mock tracker."""
        mock_tracker = Mock()
        mock_tracker.initial_cash = 50000

        analyzer = PerformanceAnalyzer(mock_tracker)

        assert analyzer.tracker is mock_tracker
        assert analyzer.high_water_mark == 50000


class TestPerformanceAnalyzerUpdate:
    """Test PerformanceAnalyzer.update() method."""

    def test_update_with_no_positions(self):
        """Test update with no positions (cash only)."""
        tracker = PositionTracker(initial_cash=100000)
        analyzer = PerformanceAnalyzer(tracker)

        timestamp = datetime(2024, 1, 1, 9, 30)
        analyzer.update(timestamp)

        assert len(analyzer.timestamps) == 1
        assert analyzer.timestamps[0] == timestamp
        assert len(analyzer.equity_curve) == 1
        assert analyzer.equity_curve[0] == 100000
        assert len(analyzer.daily_returns) == 0  # No previous equity to compare
        assert analyzer.high_water_mark == 100000
        assert analyzer.max_drawdown == 0.0

    def test_update_tracks_equity_increase(self):
        """Test update tracks equity increase."""
        tracker = PositionTracker(initial_cash=100000)
        analyzer = PerformanceAnalyzer(tracker)

        # First update
        timestamp1 = datetime(2024, 1, 1, 9, 30)
        analyzer.update(timestamp1)

        # Simulate profit (manually increase equity)
        tracker.cash = 110000

        # Second update
        timestamp2 = datetime(2024, 1, 1, 9, 31)
        analyzer.update(timestamp2)

        assert len(analyzer.equity_curve) == 2
        assert analyzer.equity_curve[0] == 100000
        assert analyzer.equity_curve[1] == 110000
        assert len(analyzer.daily_returns) == 1
        assert analyzer.daily_returns[0] == pytest.approx(0.10)  # 10% return
        assert analyzer.high_water_mark == 110000
        assert analyzer.max_drawdown == 0.0

    def test_update_tracks_drawdown(self):
        """Test update tracks max drawdown."""
        tracker = PositionTracker(initial_cash=100000)
        analyzer = PerformanceAnalyzer(tracker)

        # Update 1: Start at 100k
        analyzer.update(datetime(2024, 1, 1, 9, 30))

        # Update 2: Rise to 120k (new high water mark)
        tracker.cash = 120000
        analyzer.update(datetime(2024, 1, 1, 9, 31))

        # Update 3: Drop to 100k (drawdown)
        tracker.cash = 100000
        analyzer.update(datetime(2024, 1, 1, 9, 32))

        assert analyzer.high_water_mark == 120000
        expected_drawdown = (120000 - 100000) / 120000  # ~0.1667
        assert analyzer.max_drawdown == pytest.approx(expected_drawdown, rel=1e-4)

    def test_update_with_positions(self):
        """Test update with active positions."""
        tracker = PositionTracker(initial_cash=100000)
        analyzer = PerformanceAnalyzer(tracker)

        # Buy stock
        tracker.update_position("AAPL", quantity_change=100, price=150.0)

        # First update
        analyzer.update(datetime(2024, 1, 1, 9, 30))

        # Update price
        tracker.update_prices({"AAPL": 160.0})

        # Second update
        analyzer.update(datetime(2024, 1, 1, 9, 31))

        assert len(analyzer.equity_curve) == 2
        # Initial: 100000 - (100 * 150) = 85000 cash + 15000 position = 100000
        assert analyzer.equity_curve[0] == pytest.approx(100000, rel=1e-4)
        # After price update: 85000 cash + 16000 position = 101000
        assert analyzer.equity_curve[1] == pytest.approx(101000, rel=1e-4)

    def test_update_calculates_leverage(self):
        """Test update calculates leverage correctly."""
        tracker = PositionTracker(initial_cash=100000)
        analyzer = PerformanceAnalyzer(tracker)

        # Buy stock worth 50% of portfolio
        tracker.update_position("AAPL", quantity_change=333, price=150.0)

        # Update
        analyzer.update(datetime(2024, 1, 1, 9, 30))

        # Expected leverage: position_value / equity
        # Position value = 333 * 150 = 49950
        # Equity = 100000 (unchanged in this simplified test)
        # Leverage = 49950 / 100000 ≈ 0.4995
        assert analyzer.max_leverage > 0
        assert analyzer.max_leverage < 1.0  # Less than 100%

    def test_update_calculates_concentration(self):
        """Test update calculates concentration correctly."""
        tracker = PositionTracker(initial_cash=100000)
        analyzer = PerformanceAnalyzer(tracker)

        # Buy two positions
        tracker.update_position("AAPL", quantity_change=100, price=150.0)  # 15000
        tracker.update_position("GOOGL", quantity_change=50, price=200.0)   # 10000

        # Update
        analyzer.update(datetime(2024, 1, 1, 9, 30))

        # Max position = 15000, equity = 100000
        # Concentration = 15000 / 100000 = 0.15
        assert analyzer.max_concentration == pytest.approx(0.15, rel=1e-2)


class TestPerformanceAnalyzerSharpeRatio:
    """Test Sharpe ratio calculation."""

    def test_sharpe_ratio_insufficient_data(self):
        """Test Sharpe ratio returns None with insufficient data."""
        tracker = PositionTracker(initial_cash=100000)
        analyzer = PerformanceAnalyzer(tracker)

        # Only one update (no returns yet)
        analyzer.update(datetime(2024, 1, 1, 9, 30))

        assert analyzer.calculate_sharpe_ratio() is None

    def test_sharpe_ratio_zero_std(self):
        """Test Sharpe ratio with zero standard deviation."""
        tracker = PositionTracker(initial_cash=100000)
        analyzer = PerformanceAnalyzer(tracker)

        # Three updates with same equity (all zero returns) - need at least 2 returns
        analyzer.update(datetime(2024, 1, 1, 9, 30))
        analyzer.update(datetime(2024, 1, 1, 9, 31))
        analyzer.update(datetime(2024, 1, 1, 9, 32))

        assert analyzer.calculate_sharpe_ratio() == 0.0

    def test_sharpe_ratio_positive_returns(self):
        """Test Sharpe ratio with positive returns."""
        tracker = PositionTracker(initial_cash=100000)
        analyzer = PerformanceAnalyzer(tracker)

        # Simulate daily returns: 1% each day for 5 days
        base_equity = 100000
        for i in range(6):
            tracker.cash = base_equity * (1.01 ** i)
            analyzer.update(datetime(2024, 1, 1) + timedelta(days=i))

        sharpe = analyzer.calculate_sharpe_ratio()
        assert sharpe is not None
        assert sharpe > 0  # Positive returns should give positive Sharpe


class TestPerformanceAnalyzerGetMetrics:
    """Test get_metrics() method."""

    def test_get_metrics_basic(self):
        """Test get_metrics returns expected fields."""
        tracker = PositionTracker(initial_cash=100000)
        analyzer = PerformanceAnalyzer(tracker)

        # Simulate some activity
        tracker.update_position("AAPL", quantity_change=100, price=150.0)
        tracker.total_commission = 10.0
        tracker.total_slippage = 5.0
        analyzer.update(datetime(2024, 1, 1, 9, 30))

        metrics = analyzer.get_metrics()

        # Check all required fields are present
        assert "total_return" in metrics
        assert "total_pnl" in metrics
        assert "realized_pnl" in metrics
        assert "unrealized_pnl" in metrics
        assert "max_drawdown" in metrics
        assert "current_equity" in metrics
        assert "current_cash" in metrics
        assert "total_commission" in metrics
        assert "total_slippage" in metrics
        assert "max_leverage" in metrics
        assert "max_concentration" in metrics

    def test_get_metrics_includes_sharpe_when_available(self):
        """Test get_metrics includes Sharpe ratio when sufficient data."""
        tracker = PositionTracker(initial_cash=100000)
        analyzer = PerformanceAnalyzer(tracker)

        # Three updates to generate at least 2 returns
        analyzer.update(datetime(2024, 1, 1, 9, 30))
        tracker.cash = 101000
        analyzer.update(datetime(2024, 1, 1, 9, 31))
        tracker.cash = 102000
        analyzer.update(datetime(2024, 1, 1, 9, 32))

        metrics = analyzer.get_metrics()

        assert "sharpe_ratio" in metrics


class TestPerformanceAnalyzerGetEquityCurve:
    """Test get_equity_curve() method."""

    def test_get_equity_curve_empty(self):
        """Test get_equity_curve with no data."""
        tracker = PositionTracker(initial_cash=100000)
        analyzer = PerformanceAnalyzer(tracker)

        df = analyzer.get_equity_curve()

        assert df.is_empty()

    def test_get_equity_curve_with_data(self):
        """Test get_equity_curve with data."""
        tracker = PositionTracker(initial_cash=100000)
        analyzer = PerformanceAnalyzer(tracker)

        # Three updates
        timestamps = [
            datetime(2024, 1, 1, 9, 30),
            datetime(2024, 1, 1, 9, 31),
            datetime(2024, 1, 1, 9, 32),
        ]

        for i, ts in enumerate(timestamps):
            tracker.cash = 100000 + i * 1000  # Increment by 1000 each time
            analyzer.update(ts)

        df = analyzer.get_equity_curve()

        assert df.shape[0] == 3
        assert "timestamp" in df.columns
        assert "equity" in df.columns
        assert "returns" in df.columns
        assert df["returns"][0] == 0.0  # First return is always 0


class TestPerformanceAnalyzerReset:
    """Test reset() method."""

    def test_reset_clears_all_data(self):
        """Test reset clears all tracking data."""
        tracker = PositionTracker(initial_cash=100000)
        analyzer = PerformanceAnalyzer(tracker)

        # Populate with data
        tracker.update_position("AAPL", quantity_change=100, price=150.0)
        for i in range(5):
            tracker.cash = 100000 + i * 1000
            analyzer.update(datetime(2024, 1, 1) + timedelta(days=i))

        # Reset
        analyzer.reset()

        # Verify everything is cleared
        assert analyzer.high_water_mark == 100000
        assert analyzer.max_drawdown == 0.0
        assert analyzer.daily_returns == []
        assert analyzer.timestamps == []
        assert analyzer.equity_curve == []
        assert analyzer.max_leverage == 0.0
        assert analyzer.max_concentration == 0.0
