"""Tests for per-asset trading statistics tracking.

Tests the AssetTradingStats class and its integration with Broker for
stateful decision-making during backtests.
"""

from collections import deque
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from ml4t.backtest.broker import Broker
from ml4t.backtest.config import StatsConfig
from ml4t.backtest.models import NoCommission, NoSlippage
from ml4t.backtest.sessions import SessionConfig
from ml4t.backtest.types import AssetTradingStats, OrderSide, Position


@pytest.fixture
def broker():
    """Create a basic broker for testing."""
    return Broker(
        initial_cash=100000.0,
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
    )


@pytest.fixture
def broker_with_position(broker):
    """Create broker with an existing long position."""
    pos = Position(
        asset="BTC",
        quantity=10.0,
        entry_price=50000.0,
        entry_time=datetime(2024, 1, 1, 9, 30),
        initial_quantity=10.0,
    )
    broker.positions["BTC"] = pos
    broker.account.positions["BTC"] = pos
    return broker


class TestAssetTradingStatsBasic:
    """Test AssetTradingStats dataclass methods."""

    def test_initial_state(self):
        """Test initial stats are all zeros."""
        stats = AssetTradingStats()
        assert stats.total_realized_pnl == 0.0
        assert stats.total_trades == 0
        assert stats.total_wins == 0
        assert stats.win_rate == 0.0
        assert stats.recent_win_rate == 0.0
        assert stats.recent_expectancy == 0.0
        assert stats.session_pnl == 0.0
        assert stats.session_trades == 0
        assert stats.session_wins == 0

    def test_record_winning_trade(self):
        """Test recording a winning trade updates stats correctly."""
        stats = AssetTradingStats(recent_pnls=deque(maxlen=50))
        stats.record_pnl(100.0)

        assert stats.total_realized_pnl == 100.0
        assert stats.total_trades == 1
        assert stats.total_wins == 1
        assert stats.win_rate == 1.0
        assert stats.recent_win_rate == 1.0
        assert stats.recent_expectancy == 100.0
        assert stats.session_pnl == 100.0
        assert stats.session_trades == 1
        assert stats.session_wins == 1

    def test_record_losing_trade(self):
        """Test recording a losing trade updates stats correctly."""
        stats = AssetTradingStats(recent_pnls=deque(maxlen=50))
        stats.record_pnl(-50.0)

        assert stats.total_realized_pnl == -50.0
        assert stats.total_trades == 1
        assert stats.total_wins == 0
        assert stats.win_rate == 0.0
        assert stats.recent_win_rate == 0.0
        assert stats.recent_expectancy == -50.0

    def test_win_rate_calculation(self):
        """Test win rate is calculated correctly."""
        stats = AssetTradingStats(recent_pnls=deque(maxlen=50))
        stats.record_pnl(100.0)  # Win
        stats.record_pnl(-50.0)  # Loss
        stats.record_pnl(75.0)  # Win
        stats.record_pnl(-25.0)  # Loss

        assert stats.total_trades == 4
        assert stats.total_wins == 2
        assert stats.win_rate == 0.5

    def test_recent_window_circular_buffer(self):
        """Test recent window drops oldest trades when full."""
        stats = AssetTradingStats(recent_pnls=deque(maxlen=3))

        # Fill the buffer
        stats.record_pnl(100.0)  # Win
        stats.record_pnl(-50.0)  # Loss
        stats.record_pnl(75.0)  # Win

        assert len(stats.recent_pnls) == 3
        assert stats.recent_wins == 2
        assert stats.recent_win_rate == 2 / 3

        # Add another - should drop the first (100.0 win)
        stats.record_pnl(-25.0)  # Loss

        assert len(stats.recent_pnls) == 3
        assert list(stats.recent_pnls) == [-50.0, 75.0, -25.0]
        assert stats.recent_wins == 1  # Only 75.0 is a win now
        assert stats.recent_win_rate == 1 / 3

        # All-time stats still track everything
        assert stats.total_trades == 4
        assert stats.total_wins == 2

    def test_session_reset(self):
        """Test session stats reset correctly."""
        stats = AssetTradingStats(recent_pnls=deque(maxlen=50))
        stats.record_pnl(100.0)
        stats.record_pnl(-50.0)

        assert stats.session_trades == 2
        assert stats.session_pnl == 50.0

        # Reset session
        stats.reset_session(new_session_id=12345)

        assert stats.session_trades == 0
        assert stats.session_pnl == 0.0
        assert stats.session_wins == 0
        assert stats.session_id == 12345

        # All-time stats are preserved
        assert stats.total_trades == 2
        assert stats.total_realized_pnl == 50.0

    def test_avg_pnl(self):
        """Test average P&L calculation."""
        stats = AssetTradingStats(recent_pnls=deque(maxlen=50))
        stats.record_pnl(100.0)
        stats.record_pnl(-50.0)
        stats.record_pnl(50.0)

        assert stats.avg_pnl == pytest.approx(100 / 3, rel=1e-6)

    def test_recent_total_pnl(self):
        """Test recent total P&L calculation."""
        stats = AssetTradingStats(recent_pnls=deque(maxlen=50))
        stats.record_pnl(100.0)
        stats.record_pnl(-50.0)
        stats.record_pnl(75.0)

        assert stats.recent_total_pnl == 125.0

    def test_repr(self):
        """Test string representation."""
        stats = AssetTradingStats(recent_pnls=deque(maxlen=50))
        stats.record_pnl(100.0)
        stats.record_pnl(-50.0)

        repr_str = repr(stats)
        assert "AssetTradingStats" in repr_str
        assert "trades=2" in repr_str
        assert "50.0%" in repr_str  # win_rate


class TestBrokerStatsIntegration:
    """Test broker integration with trading stats."""

    def test_get_asset_stats_creates_new(self, broker):
        """Test get_asset_stats creates new stats if none exist."""
        stats = broker.get_asset_stats("BTC")
        assert stats is not None
        assert stats.total_trades == 0

    def test_get_asset_stats_returns_existing(self, broker):
        """Test get_asset_stats returns same object on subsequent calls."""
        stats1 = broker.get_asset_stats("BTC")
        stats1.record_pnl(100.0)

        stats2 = broker.get_asset_stats("BTC")
        assert stats2 is stats1
        assert stats2.total_trades == 1

    def test_multi_asset_stats_independent(self, broker):
        """Test stats are tracked independently per asset."""
        btc_stats = broker.get_asset_stats("BTC")
        eth_stats = broker.get_asset_stats("ETH")

        btc_stats.record_pnl(100.0)

        assert btc_stats.total_trades == 1
        assert eth_stats.total_trades == 0

    def test_configure_stats_window_size(self, broker):
        """Test configure_stats updates window size."""
        broker.configure_stats(recent_window_size=10)

        stats = broker.get_asset_stats("BTC")
        assert stats.recent_pnls.maxlen == 10

    def test_configure_stats_with_config_object(self, broker):
        """Test configure_stats accepts StatsConfig object."""
        config = StatsConfig(
            recent_window_size=100,
            track_session_stats=True,
            enabled=True,
        )
        broker.configure_stats(config=config)

        assert broker._stats_config.recent_window_size == 100
        assert broker._stats_config.track_session_stats is True

    def test_configure_stats_updates_existing_deques(self, broker):
        """Test configure_stats updates existing stats deque sizes."""
        # Create stats with default window
        stats = broker.get_asset_stats("BTC")
        for i in range(5):
            stats.record_pnl(float(i * 10))

        # Reduce window size
        broker.configure_stats(recent_window_size=3)

        assert stats.recent_pnls.maxlen == 3
        assert len(stats.recent_pnls) == 3
        # Should keep most recent
        assert list(stats.recent_pnls) == [20.0, 30.0, 40.0]

    def test_stats_disabled(self, broker):
        """Test stats not recorded when disabled."""
        broker.configure_stats(enabled=False)

        # Manually call _record_pnl_event (simulating a close)
        broker._record_pnl_event("BTC", 100.0)

        # Stats should not be created/updated
        assert "BTC" not in broker._asset_stats


class TestStatsOnPositionClose:
    """Test stats are recorded when positions are closed."""

    def test_full_close_records_pnl(self, broker_with_position):
        """Test closing a full position records P&L in stats."""
        broker = broker_with_position
        timestamp = datetime(2024, 1, 2, 9, 30)

        # Update broker with current prices
        broker._update_time(
            timestamp=timestamp,
            prices={"BTC": 51000.0},  # $1000 profit per unit
            opens={"BTC": 50500.0},
            highs={"BTC": 51500.0},
            lows={"BTC": 50000.0},
            volumes={"BTC": 1000.0},
            signals={},
        )

        # Submit close order
        order = broker.submit_order("BTC", -10.0, OrderSide.SELL)
        assert order is not None

        # Process the order
        broker._process_orders()

        # Check stats were recorded
        stats = broker.get_asset_stats("BTC")
        assert stats.total_trades == 1
        assert stats.total_wins == 1  # Profit = win
        assert stats.total_realized_pnl == pytest.approx(10000.0, rel=1e-2)  # 10 * $1000

    def test_partial_exit_records_pnl(self, broker_with_position):
        """Test partial position exit records proportional P&L."""
        broker = broker_with_position
        timestamp = datetime(2024, 1, 2, 9, 30)

        # Update broker with current prices
        broker._update_time(
            timestamp=timestamp,
            prices={"BTC": 51000.0},  # $1000 profit per unit
            opens={"BTC": 50500.0},
            highs={"BTC": 51500.0},
            lows={"BTC": 50000.0},
            volumes={"BTC": 1000.0},
            signals={},
        )

        # Sell half the position (5 units)
        order = broker.submit_order("BTC", -5.0, OrderSide.SELL)
        assert order is not None

        # Process the order
        broker._process_orders()

        # Check stats were recorded
        stats = broker.get_asset_stats("BTC")
        assert stats.total_trades == 1
        assert stats.total_wins == 1
        # P&L for 5 units at $1000 profit each
        assert stats.total_realized_pnl == pytest.approx(5000.0, rel=1e-2)

        # Position should still exist with 5 units
        pos = broker.get_position("BTC")
        assert pos is not None
        assert pos.quantity == 5.0

    def test_losing_trade_records_loss(self, broker_with_position):
        """Test losing trade records negative P&L."""
        broker = broker_with_position
        timestamp = datetime(2024, 1, 2, 9, 30)

        # Update broker with current prices (loss)
        broker._update_time(
            timestamp=timestamp,
            prices={"BTC": 49000.0},  # $1000 loss per unit
            opens={"BTC": 49500.0},
            highs={"BTC": 50000.0},
            lows={"BTC": 48500.0},
            volumes={"BTC": 1000.0},
            signals={},
        )

        # Close position at a loss
        order = broker.submit_order("BTC", -10.0, OrderSide.SELL)
        assert order is not None

        # Process the order
        broker._process_orders()

        # Check stats recorded the loss
        stats = broker.get_asset_stats("BTC")
        assert stats.total_trades == 1
        assert stats.total_wins == 0  # Loss
        assert stats.total_realized_pnl == pytest.approx(-10000.0, rel=1e-2)
        assert stats.win_rate == 0.0


class TestShortPositionStats:
    """Test stats tracking for short positions."""

    def test_profitable_short_records_win(self, broker):
        """Test profitable short position records P&L correctly."""
        timestamp = datetime(2024, 1, 1, 9, 30)

        # Create a short position
        pos = Position(
            asset="BTC",
            quantity=-10.0,  # Short position
            entry_price=50000.0,
            entry_time=timestamp,
            initial_quantity=-10.0,
        )
        broker.positions["BTC"] = pos
        broker.account.positions["BTC"] = pos

        timestamp2 = datetime(2024, 1, 2, 9, 30)

        # Price dropped - profit for short
        broker._update_time(
            timestamp=timestamp2,
            prices={"BTC": 49000.0},  # $1000 profit per unit (short)
            opens={"BTC": 49500.0},
            highs={"BTC": 50000.0},
            lows={"BTC": 48500.0},
            volumes={"BTC": 1000.0},
            signals={},
        )

        # Cover the short (buy to close)
        order = broker.submit_order("BTC", 10.0, OrderSide.BUY)
        assert order is not None

        # Process the order
        broker._process_orders()

        # Check stats - profit because price dropped
        stats = broker.get_asset_stats("BTC")
        assert stats.total_trades == 1
        assert stats.total_wins == 1  # Profit on short
        assert stats.total_realized_pnl == pytest.approx(10000.0, rel=1e-2)


class TestSessionBoundaryDetection:
    """Test session boundary detection and stats reset."""

    def test_session_config_set(self, broker):
        """Test setting session configuration."""
        config = SessionConfig(
            calendar="CME_Equity",
            timezone="America/Chicago",
            session_start_time="17:00",
        )
        broker.set_session_config(config)

        assert broker._session_config is config

    def test_session_boundary_resets_session_stats(self, broker):
        """Test crossing session boundary resets session stats."""
        config = SessionConfig(
            calendar="CME_Equity",
            timezone="America/Chicago",
            session_start_time="17:00",
        )
        broker.set_session_config(config)

        # Create stats with some session data
        stats = broker.get_asset_stats("BTC")
        stats.record_pnl(100.0)

        assert stats.session_trades == 1
        assert stats.session_pnl == 100.0

        # First bar - establishes session
        ct = ZoneInfo("America/Chicago")
        # Monday 4pm CT = still Monday session
        ts1 = datetime(2024, 1, 8, 16, 0, tzinfo=ct)
        broker._update_time(
            timestamp=ts1,
            prices={"BTC": 50000.0},
            opens={"BTC": 50000.0},
            highs={"BTC": 50000.0},
            lows={"BTC": 50000.0},
            volumes={"BTC": 100.0},
            signals={},
        )

        # Session stats should NOT be reset (first update)
        assert stats.session_trades == 1

        # Monday 5pm CT = Tuesday session starts, should reset
        ts2 = datetime(2024, 1, 8, 17, 0, tzinfo=ct)
        broker._update_time(
            timestamp=ts2,
            prices={"BTC": 50000.0},
            opens={"BTC": 50000.0},
            highs={"BTC": 50000.0},
            lows={"BTC": 50000.0},
            volumes={"BTC": 100.0},
            signals={},
        )

        # Session stats should be reset
        assert stats.session_trades == 0
        assert stats.session_pnl == 0.0

        # All-time stats preserved
        assert stats.total_trades == 1
        assert stats.total_realized_pnl == 100.0

    def test_no_session_config_no_reset(self, broker):
        """Test without session config, stats are not reset."""
        stats = broker.get_asset_stats("BTC")
        stats.record_pnl(100.0)

        # Update time without session config
        broker._update_time(
            timestamp=datetime(2024, 1, 2, 17, 0),
            prices={"BTC": 50000.0},
            opens={"BTC": 50000.0},
            highs={"BTC": 50000.0},
            lows={"BTC": 50000.0},
            volumes={"BTC": 100.0},
            signals={},
        )

        # Session stats should NOT be reset
        assert stats.session_trades == 1
        assert stats.session_pnl == 100.0


class TestStatsDerivedProperties:
    """Test derived property calculations."""

    def test_session_win_rate(self):
        """Test session win rate calculation."""
        stats = AssetTradingStats(recent_pnls=deque(maxlen=50))

        # 3 wins, 2 losses in session
        for pnl in [100.0, -50.0, 75.0, -25.0, 50.0]:
            stats.record_pnl(pnl)

        assert stats.session_trades == 5
        assert stats.session_wins == 3
        assert stats.session_win_rate == 0.6

    def test_zero_trades_properties(self):
        """Test properties with zero trades don't divide by zero."""
        stats = AssetTradingStats()

        assert stats.win_rate == 0.0
        assert stats.recent_win_rate == 0.0
        assert stats.recent_expectancy == 0.0
        assert stats.session_win_rate == 0.0
        assert stats.avg_pnl == 0.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_pnl_trade(self):
        """Test zero P&L trade is not counted as a win."""
        stats = AssetTradingStats(recent_pnls=deque(maxlen=50))
        stats.record_pnl(0.0)

        assert stats.total_trades == 1
        assert stats.total_wins == 0  # Zero is not a win
        assert stats.win_rate == 0.0

    def test_very_small_window(self):
        """Test with window size of 1."""
        stats = AssetTradingStats(recent_pnls=deque(maxlen=1))

        stats.record_pnl(100.0)
        assert stats.recent_win_rate == 1.0

        stats.record_pnl(-50.0)
        assert stats.recent_win_rate == 0.0  # Only last trade in window
        assert stats.recent_pnls.maxlen == 1
        assert len(stats.recent_pnls) == 1

    def test_large_number_of_trades(self):
        """Test stats with many trades."""
        stats = AssetTradingStats(recent_pnls=deque(maxlen=50))

        # Simulate 1000 trades
        wins = 0
        total_pnl = 0.0
        for i in range(1000):
            pnl = 100.0 if i % 3 != 0 else -50.0  # ~66% win rate
            if pnl > 0:
                wins += 1
            total_pnl += pnl
            stats.record_pnl(pnl)

        assert stats.total_trades == 1000
        assert stats.total_wins == wins
        assert stats.total_realized_pnl == pytest.approx(total_pnl, rel=1e-6)

        # Recent window should have last 50
        assert len(stats.recent_pnls) == 50
