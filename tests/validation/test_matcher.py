"""
Unit tests for trade matcher.

These tests verify that the trade matching logic correctly:
1. Groups trades by entry timestamp (with tolerance)
2. Calculates deltas and differences
3. Classifies severity
4. Handles edge cases (unmatched trades, empty inputs, etc.)

Note: The matcher is already validated end-to-end in Phase 1 (test_all_platforms_scenario_001.py).
These unit tests provide regression coverage and test edge cases in isolation.
"""

import pytest
from datetime import datetime, timedelta, timezone

from comparison.matcher import match_trades, TradeMatch, _create_match_from_group
from core.trade import StandardTrade


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_trade_backtest():
    """Sample ml4t.backtest trade for testing."""
    return StandardTrade(
        trade_id=1,
        platform='ml4t.backtest',
        entry_timestamp=datetime(2017, 2, 6, 14, 30, tzinfo=timezone.utc),
        entry_price=73.50,
        entry_price_component='open',
        entry_bar_ohlc={'open': 73.50, 'high': 74.00, 'low': 73.00, 'close': 73.85},
        exit_timestamp=datetime(2017, 2, 7, 14, 30, tzinfo=timezone.utc),
        exit_price=75.20,
        exit_price_component='open',
        exit_bar_ohlc={'open': 75.20, 'high': 75.50, 'low': 74.80, 'close': 75.30},
        exit_reason='signal',
        symbol='AAPL',
        quantity=100.0,
        side='long',
        gross_pnl=170.0,
        entry_commission=0.50,
        exit_commission=0.50,
        slippage=0.10,
        net_pnl=168.90,
    )


@pytest.fixture
def sample_trade_vectorbt():
    """Sample VectorBT trade for testing (same entry time, slightly different price)."""
    return StandardTrade(
        trade_id=1,
        platform='vectorbt',
        entry_timestamp=datetime(2017, 2, 6, 14, 30, tzinfo=timezone.utc),
        entry_price=73.51,  # Slightly different (0.01% diff)
        entry_price_component='close',  # Different component
        entry_bar_ohlc={'open': 73.50, 'high': 74.00, 'low': 73.00, 'close': 73.51},
        exit_timestamp=datetime(2017, 2, 7, 14, 30, tzinfo=timezone.utc),
        exit_price=75.21,
        exit_price_component='close',
        exit_bar_ohlc={'open': 75.20, 'high': 75.50, 'low': 74.80, 'close': 75.21},
        exit_reason='signal',
        symbol='AAPL',
        quantity=100.0,
        side='long',
        gross_pnl=170.0,
        entry_commission=0.50,
        exit_commission=0.50,
        slippage=0.10,
        net_pnl=168.90,
    )


@pytest.fixture
def sample_trade_backtrader():
    """Sample Backtrader trade for testing (within tolerance, slightly earlier)."""
    return StandardTrade(
        trade_id=1,
        platform='backtrader',
        entry_timestamp=datetime(2017, 2, 6, 14, 29, 30, tzinfo=timezone.utc),  # 30 seconds earlier
        entry_price=73.50,
        entry_price_component='open',
        entry_bar_ohlc={'open': 73.50, 'high': 74.00, 'low': 73.00, 'close': 73.85},
        exit_timestamp=datetime(2017, 2, 7, 14, 30, tzinfo=timezone.utc),
        exit_price=75.20,
        exit_price_component='open',
        exit_bar_ohlc={'open': 75.20, 'high': 75.50, 'low': 74.80, 'close': 75.30},
        exit_reason='signal',
        symbol='AAPL',
        quantity=100.0,
        side='long',
        gross_pnl=170.0,
        entry_commission=0.50,
        exit_commission=0.50,
        slippage=0.10,
        net_pnl=168.90,
    )


@pytest.fixture
def sample_trade_different_entry():
    """Trade with significantly different entry time (should not match)."""
    return StandardTrade(
        trade_id=2,
        platform='ml4t.backtest',
        entry_timestamp=datetime(2017, 2, 6, 15, 30, tzinfo=timezone.utc),  # 1 hour later
        entry_price=74.50,
        entry_price_component='open',
        entry_bar_ohlc={'open': 74.50, 'high': 75.00, 'low': 74.00, 'close': 74.85},
        exit_timestamp=datetime(2017, 2, 7, 15, 30, tzinfo=timezone.utc),
        exit_price=76.20,
        exit_price_component='open',
        exit_bar_ohlc={'open': 76.20, 'high': 76.50, 'low': 75.80, 'close': 76.30},
        exit_reason='signal',
        symbol='AAPL',
        quantity=100.0,
        side='long',
        gross_pnl=170.0,
        entry_commission=0.50,
        exit_commission=0.50,
        slippage=0.10,
        net_pnl=168.90,
    )


# ============================================================================
# Test TradeMatch Properties
# ============================================================================

class TestTradeMatchProperties:
    """Test TradeMatch dataclass properties and methods."""

    def test_reference_trade_returns_first_non_none(self):
        """Test reference_trade property returns first non-None trade."""
        match = TradeMatch(
            backtest_trade=None,
            vectorbt_trade=None,
            backtrader_trade=StandardTrade(
                trade_id=1, platform='backtrader',
                entry_timestamp=datetime(2017, 2, 6, 14, 30, tzinfo=timezone.utc),
                entry_price=73.50, entry_price_component='open', entry_bar_ohlc={},
                exit_timestamp=None, exit_price=None, exit_price_component=None,
                exit_bar_ohlc=None, exit_reason=None,
                symbol='AAPL', quantity=100.0, side='long',
                gross_pnl=None, entry_commission=0.5, exit_commission=0.0,
                slippage=0.0, net_pnl=None,
            ),
            zipline_trade=None,
            entry_timestamp_deltas={},
            exit_timestamp_deltas={},
            entry_price_diffs={},
            exit_price_diffs={},
            entry_components={},
            exit_components={},
            differences=[],
            severity='none',
        )

        ref = match.reference_trade
        assert ref is not None
        assert ref.platform == 'backtrader'

    def test_reference_trade_returns_none_when_all_none(self):
        """Test reference_trade property returns None when all trades are None."""
        match = TradeMatch(
            backtest_trade=None,
            vectorbt_trade=None,
            backtrader_trade=None,
            zipline_trade=None,
            entry_timestamp_deltas={},
            exit_timestamp_deltas={},
            entry_price_diffs={},
            exit_price_diffs={},
            entry_components={},
            exit_components={},
            differences=[],
            severity='none',
        )

        assert match.reference_trade is None

    def test_all_trades_returns_non_none_trades(self, sample_trade_backtest, sample_trade_vectorbt):
        """Test all_trades property returns dict of non-None trades."""
        match = TradeMatch(
            backtest_trade=sample_trade_backtest,
            vectorbt_trade=sample_trade_vectorbt,
            backtrader_trade=None,
            zipline_trade=None,
            entry_timestamp_deltas={},
            exit_timestamp_deltas={},
            entry_price_diffs={},
            exit_price_diffs={},
            entry_components={},
            exit_components={},
            differences=[],
            severity='none',
        )

        trades = match.all_trades
        assert len(trades) == 2
        assert 'ml4t.backtest' in trades
        assert 'vectorbt' in trades
        assert 'backtrader' not in trades
        assert 'zipline' not in trades

    def test_all_trades_includes_all_four_platforms(
        self, sample_trade_backtest, sample_trade_vectorbt, sample_trade_backtrader
    ):
        """Test all_trades property includes all four platforms when present."""
        zipline_trade = StandardTrade(
            trade_id=1, platform='zipline',
            entry_timestamp=datetime(2017, 2, 6, 14, 30, tzinfo=timezone.utc),
            entry_price=73.50, entry_price_component='open', entry_bar_ohlc={},
            exit_timestamp=None, exit_price=None, exit_price_component=None,
            exit_bar_ohlc=None, exit_reason=None,
            symbol='AAPL', quantity=100.0, side='long',
            gross_pnl=None, entry_commission=0.5, exit_commission=0.0,
            slippage=0.0, net_pnl=None,
        )

        match = TradeMatch(
            backtest_trade=sample_trade_backtest,
            vectorbt_trade=sample_trade_vectorbt,
            backtrader_trade=sample_trade_backtrader,
            zipline_trade=zipline_trade,
            entry_timestamp_deltas={},
            exit_timestamp_deltas={},
            entry_price_diffs={},
            exit_price_diffs={},
            entry_components={},
            exit_components={},
            differences=[],
            severity='none',
        )

        trades = match.all_trades
        assert len(trades) == 4
        assert 'ml4t.backtest' in trades
        assert 'vectorbt' in trades
        assert 'backtrader' in trades
        assert 'zipline' in trades


# ============================================================================
# Test match_trades() - Basic Matching
# ============================================================================

class TestMatchTradesBasic:
    """Test basic trade matching functionality."""

    def test_match_single_platform(self, sample_trade_backtest):
        """Test matching with single platform (no comparison)."""
        trades_by_platform = {
            'ml4t.backtest': [sample_trade_backtest],
        }

        matches = match_trades(trades_by_platform)

        assert len(matches) == 1
        assert matches[0].backtest_trade == sample_trade_backtest
        assert matches[0].vectorbt_trade is None
        assert matches[0].backtrader_trade is None
        assert matches[0].zipline_trade is None

    def test_match_two_platforms_exact_timestamp(self, sample_trade_backtest, sample_trade_vectorbt):
        """Test matching two platforms with exact timestamp match."""
        trades_by_platform = {
            'ml4t.backtest': [sample_trade_backtest],
            'vectorbt': [sample_trade_vectorbt],
        }

        matches = match_trades(trades_by_platform)

        assert len(matches) == 1
        assert matches[0].backtest_trade == sample_trade_backtest
        assert matches[0].vectorbt_trade == sample_trade_vectorbt

    def test_match_three_platforms_within_tolerance(
        self, sample_trade_backtest, sample_trade_vectorbt, sample_trade_backtrader
    ):
        """Test matching three platforms within timestamp tolerance."""
        trades_by_platform = {
            'ml4t.backtest': [sample_trade_backtest],
            'vectorbt': [sample_trade_vectorbt],
            'backtrader': [sample_trade_backtrader],  # 30 seconds earlier
        }

        matches = match_trades(trades_by_platform, timestamp_tolerance_seconds=60)

        assert len(matches) == 1
        assert matches[0].backtest_trade == sample_trade_backtest
        assert matches[0].vectorbt_trade == sample_trade_vectorbt
        assert matches[0].backtrader_trade == sample_trade_backtrader

    def test_match_respects_tolerance(
        self, sample_trade_backtest, sample_trade_backtrader
    ):
        """Test that matching respects timestamp tolerance."""
        trades_by_platform = {
            'ml4t.backtest': [sample_trade_backtest],
            'backtrader': [sample_trade_backtrader],  # 30 seconds earlier
        }

        # With 60s tolerance - should match
        matches = match_trades(trades_by_platform, timestamp_tolerance_seconds=60)
        assert len(matches) == 1
        assert matches[0].backtest_trade is not None
        assert matches[0].backtrader_trade is not None

        # With 15s tolerance - should NOT match
        matches = match_trades(trades_by_platform, timestamp_tolerance_seconds=15)
        assert len(matches) == 2  # Two separate matches
        assert matches[0].backtrader_trade is not None
        assert matches[0].backtest_trade is None
        assert matches[1].backtest_trade is not None
        assert matches[1].backtrader_trade is None

    def test_match_separate_groups(
        self, sample_trade_backtest, sample_trade_different_entry
    ):
        """Test that trades outside tolerance create separate groups."""
        trades_by_platform = {
            'ml4t.backtest': [sample_trade_backtest, sample_trade_different_entry],
        }

        matches = match_trades(trades_by_platform, timestamp_tolerance_seconds=60)

        assert len(matches) == 2
        # First match (earlier trade)
        assert matches[0].backtest_trade.entry_timestamp == sample_trade_backtest.entry_timestamp
        # Second match (later trade)
        assert matches[1].backtest_trade.entry_timestamp == sample_trade_different_entry.entry_timestamp


# ============================================================================
# Test match_trades() - Edge Cases
# ============================================================================

class TestMatchTradesEdgeCases:
    """Test edge cases for trade matching."""

    def test_empty_input(self):
        """Test matching with empty input."""
        matches = match_trades({})
        assert len(matches) == 0

    def test_empty_platform_lists(self):
        """Test matching with empty trade lists."""
        trades_by_platform = {
            'ml4t.backtest': [],
            'vectorbt': [],
        }

        matches = match_trades(trades_by_platform)
        assert len(matches) == 0

    def test_single_platform_empty(self, sample_trade_backtest):
        """Test matching when one platform has no trades."""
        trades_by_platform = {
            'ml4t.backtest': [sample_trade_backtest],
            'vectorbt': [],
        }

        matches = match_trades(trades_by_platform)

        assert len(matches) == 1
        assert matches[0].backtest_trade == sample_trade_backtest
        assert matches[0].vectorbt_trade is None

    def test_unmatched_trades_different_platforms(
        self, sample_trade_backtest, sample_trade_different_entry
    ):
        """Test that unmatched trades from different platforms create separate groups."""
        trades_by_platform = {
            'ml4t.backtest': [sample_trade_backtest],
            'vectorbt': [sample_trade_different_entry],  # Different entry time
        }

        matches = match_trades(trades_by_platform, timestamp_tolerance_seconds=60)

        assert len(matches) == 2
        # Each trade should be in its own group
        assert sum(1 for m in matches if m.backtest_trade is not None) == 1
        assert sum(1 for m in matches if m.vectorbt_trade is not None) == 1


# ============================================================================
# Test _create_match_from_group() - Delta Calculations
# ============================================================================

class TestCreateMatchDeltas:
    """Test delta calculation in _create_match_from_group()."""

    def test_entry_timestamp_deltas(self, sample_trade_backtest, sample_trade_backtrader):
        """Test entry timestamp delta calculation."""
        group = [
            ('ml4t.backtest', sample_trade_backtest),
            ('backtrader', sample_trade_backtrader),  # 30 seconds earlier
        ]

        match = _create_match_from_group(group)

        # Reference is first trade (ml4t.backtest)
        assert match.entry_timestamp_deltas['ml4t.backtest'] == 0.0
        assert match.entry_timestamp_deltas['backtrader'] == -30.0  # 30 seconds earlier

    def test_exit_timestamp_deltas(self):
        """Test exit timestamp delta calculation."""
        trade1 = StandardTrade(
            trade_id=1, platform='ml4t.backtest',
            entry_timestamp=datetime(2017, 2, 6, 14, 30, tzinfo=timezone.utc),
            entry_price=73.50, entry_price_component='open', entry_bar_ohlc={},
            exit_timestamp=datetime(2017, 2, 7, 14, 30, tzinfo=timezone.utc),
            exit_price=75.20, exit_price_component='open',
            exit_bar_ohlc={}, exit_reason='signal',
            symbol='AAPL', quantity=100.0, side='long',
            gross_pnl=170.0, entry_commission=0.5, exit_commission=0.5,
            slippage=0.1, net_pnl=168.9,
        )
        trade2 = StandardTrade(
            trade_id=1, platform='vectorbt',
            entry_timestamp=datetime(2017, 2, 6, 14, 30, tzinfo=timezone.utc),
            entry_price=73.50, entry_price_component='open', entry_bar_ohlc={},
            exit_timestamp=datetime(2017, 2, 7, 14, 31, tzinfo=timezone.utc),  # 60 seconds later
            exit_price=75.20, exit_price_component='open',
            exit_bar_ohlc={}, exit_reason='signal',
            symbol='AAPL', quantity=100.0, side='long',
            gross_pnl=170.0, entry_commission=0.5, exit_commission=0.5,
            slippage=0.1, net_pnl=168.9,
        )

        group = [('ml4t.backtest', trade1), ('vectorbt', trade2)]
        match = _create_match_from_group(group)

        assert match.exit_timestamp_deltas['ml4t.backtest'] == 0.0
        assert match.exit_timestamp_deltas['vectorbt'] == 60.0

    def test_entry_price_diffs(self, sample_trade_backtest, sample_trade_vectorbt):
        """Test entry price difference calculation."""
        group = [
            ('ml4t.backtest', sample_trade_backtest),  # 73.50
            ('vectorbt', sample_trade_vectorbt),  # 73.51
        ]

        match = _create_match_from_group(group)

        # Reference is ml4t.backtest (73.50)
        assert match.entry_price_diffs['ml4t.backtest'] == 0.0
        # VectorBT is 0.01 higher = (73.51 - 73.50) / 73.50 * 100 = 0.0136%
        assert abs(match.entry_price_diffs['vectorbt'] - 0.0136) < 0.001

    def test_exit_price_diffs(self, sample_trade_backtest, sample_trade_vectorbt):
        """Test exit price difference calculation."""
        group = [
            ('ml4t.backtest', sample_trade_backtest),  # 75.20
            ('vectorbt', sample_trade_vectorbt),  # 75.21
        ]

        match = _create_match_from_group(group)

        # Reference is ml4t.backtest (75.20)
        assert match.exit_price_diffs['ml4t.backtest'] == 0.0
        # VectorBT is 0.01 higher = (75.21 - 75.20) / 75.20 * 100 = 0.0133%
        assert abs(match.exit_price_diffs['vectorbt'] - 0.0133) < 0.001

    def test_component_tracking(self, sample_trade_backtest, sample_trade_vectorbt):
        """Test that OHLC components are tracked."""
        group = [
            ('ml4t.backtest', sample_trade_backtest),  # open/open
            ('vectorbt', sample_trade_vectorbt),  # close/close
        ]

        match = _create_match_from_group(group)

        assert match.entry_components['ml4t.backtest'] == 'open'
        assert match.entry_components['vectorbt'] == 'close'
        assert match.exit_components['ml4t.backtest'] == 'open'
        assert match.exit_components['vectorbt'] == 'close'


# ============================================================================
# Test _create_match_from_group() - Difference Detection
# ============================================================================

class TestCreateMatchDifferences:
    """Test difference detection in _create_match_from_group()."""

    def test_no_differences_identical_trades(self):
        """Test that identical trades produce no differences."""
        trade1 = StandardTrade(
            trade_id=1, platform='ml4t.backtest',
            entry_timestamp=datetime(2017, 2, 6, 14, 30, tzinfo=timezone.utc),
            entry_price=73.50, entry_price_component='open', entry_bar_ohlc={},
            exit_timestamp=datetime(2017, 2, 7, 14, 30, tzinfo=timezone.utc),
            exit_price=75.20, exit_price_component='open',
            exit_bar_ohlc={}, exit_reason='signal',
            symbol='AAPL', quantity=100.0, side='long',
            gross_pnl=170.0, entry_commission=0.5, exit_commission=0.5,
            slippage=0.1, net_pnl=168.9,
        )
        trade2 = StandardTrade(
            trade_id=1, platform='vectorbt',
            entry_timestamp=datetime(2017, 2, 6, 14, 30, tzinfo=timezone.utc),
            entry_price=73.50, entry_price_component='open', entry_bar_ohlc={},
            exit_timestamp=datetime(2017, 2, 7, 14, 30, tzinfo=timezone.utc),
            exit_price=75.20, exit_price_component='open',
            exit_bar_ohlc={}, exit_reason='signal',
            symbol='AAPL', quantity=100.0, side='long',
            gross_pnl=170.0, entry_commission=0.5, exit_commission=0.5,
            slippage=0.1, net_pnl=168.9,
        )

        group = [('ml4t.backtest', trade1), ('vectorbt', trade2)]
        match = _create_match_from_group(group)

        assert len(match.differences) == 0

    def test_entry_timing_difference(self, sample_trade_backtest, sample_trade_backtrader):
        """Test detection of entry timing differences."""
        group = [
            ('ml4t.backtest', sample_trade_backtest),
            ('backtrader', sample_trade_backtrader),  # 30 seconds earlier
        ]

        match = _create_match_from_group(group)

        # 30 seconds is within tolerance (60s), so no difference reported
        assert not any('Entry timing varies' in d for d in match.differences)

    def test_entry_timing_difference_large(self):
        """Test detection of large entry timing differences."""
        trade1 = StandardTrade(
            trade_id=1, platform='ml4t.backtest',
            entry_timestamp=datetime(2017, 2, 6, 14, 30, tzinfo=timezone.utc),
            entry_price=73.50, entry_price_component='open', entry_bar_ohlc={},
            exit_timestamp=None, exit_price=None, exit_price_component=None,
            exit_bar_ohlc=None, exit_reason=None,
            symbol='AAPL', quantity=100.0, side='long',
            gross_pnl=None, entry_commission=0.5, exit_commission=0.0,
            slippage=0.0, net_pnl=None,
        )
        trade2 = StandardTrade(
            trade_id=1, platform='vectorbt',
            entry_timestamp=datetime(2017, 2, 6, 14, 32, tzinfo=timezone.utc),  # 120 seconds later
            entry_price=73.50, entry_price_component='open', entry_bar_ohlc={},
            exit_timestamp=None, exit_price=None, exit_price_component=None,
            exit_bar_ohlc=None, exit_reason=None,
            symbol='AAPL', quantity=100.0, side='long',
            gross_pnl=None, entry_commission=0.5, exit_commission=0.0,
            slippage=0.0, net_pnl=None,
        )

        group = [('ml4t.backtest', trade1), ('vectorbt', trade2)]
        match = _create_match_from_group(group)

        assert any('Entry timing varies' in d and '120' in d for d in match.differences)

    def test_exit_timing_difference_large(self):
        """Test detection of large exit timing differences."""
        trade1 = StandardTrade(
            trade_id=1, platform='ml4t.backtest',
            entry_timestamp=datetime(2017, 2, 6, 14, 30, tzinfo=timezone.utc),
            entry_price=73.50, entry_price_component='open', entry_bar_ohlc={},
            exit_timestamp=datetime(2017, 2, 7, 14, 30, tzinfo=timezone.utc),
            exit_price=75.20, exit_price_component='open',
            exit_bar_ohlc={}, exit_reason='signal',
            symbol='AAPL', quantity=100.0, side='long',
            gross_pnl=170.0, entry_commission=0.5, exit_commission=0.5,
            slippage=0.1, net_pnl=168.9,
        )
        trade2 = StandardTrade(
            trade_id=1, platform='vectorbt',
            entry_timestamp=datetime(2017, 2, 6, 14, 30, tzinfo=timezone.utc),
            entry_price=73.50, entry_price_component='open', entry_bar_ohlc={},
            exit_timestamp=datetime(2017, 2, 7, 14, 32, tzinfo=timezone.utc),  # 120 seconds later
            exit_price=75.20, exit_price_component='open',
            exit_bar_ohlc={}, exit_reason='signal',
            symbol='AAPL', quantity=100.0, side='long',
            gross_pnl=170.0, entry_commission=0.5, exit_commission=0.5,
            slippage=0.1, net_pnl=168.9,
        )

        group = [('ml4t.backtest', trade1), ('vectorbt', trade2)]
        match = _create_match_from_group(group)

        assert any('Exit timing varies' in d and '120' in d for d in match.differences)

    def test_price_difference_detection(self):
        """Test detection of price differences."""
        trade1 = StandardTrade(
            trade_id=1, platform='ml4t.backtest',
            entry_timestamp=datetime(2017, 2, 6, 14, 30, tzinfo=timezone.utc),
            entry_price=73.50, entry_price_component='open', entry_bar_ohlc={},
            exit_timestamp=datetime(2017, 2, 7, 14, 30, tzinfo=timezone.utc),
            exit_price=75.20, exit_price_component='open',
            exit_bar_ohlc={}, exit_reason='signal',
            symbol='AAPL', quantity=100.0, side='long',
            gross_pnl=170.0, entry_commission=0.5, exit_commission=0.5,
            slippage=0.1, net_pnl=168.9,
        )
        trade2 = StandardTrade(
            trade_id=1, platform='vectorbt',
            entry_timestamp=datetime(2017, 2, 6, 14, 30, tzinfo=timezone.utc),
            entry_price=73.60,  # 0.136% higher (above 0.1% threshold)
            entry_price_component='open', entry_bar_ohlc={},
            exit_timestamp=datetime(2017, 2, 7, 14, 30, tzinfo=timezone.utc),
            exit_price=75.30,  # 0.133% higher
            exit_price_component='open',
            exit_bar_ohlc={}, exit_reason='signal',
            symbol='AAPL', quantity=100.0, side='long',
            gross_pnl=170.0, entry_commission=0.5, exit_commission=0.5,
            slippage=0.1, net_pnl=168.9,
        )

        group = [('ml4t.backtest', trade1), ('vectorbt', trade2)]
        match = _create_match_from_group(group)

        assert any('Entry prices vary' in d for d in match.differences)
        assert any('Exit prices vary' in d for d in match.differences)

    def test_component_difference_detection(self, sample_trade_backtest, sample_trade_vectorbt):
        """Test detection of OHLC component differences."""
        group = [
            ('ml4t.backtest', sample_trade_backtest),  # open/open
            ('vectorbt', sample_trade_vectorbt),  # close/close
        ]

        match = _create_match_from_group(group)

        assert any('Entry uses different OHLC components' in d for d in match.differences)
        assert any('Exit uses different OHLC components' in d for d in match.differences)


# ============================================================================
# Test _create_match_from_group() - Severity Classification
# ============================================================================

class TestCreateMatchSeverity:
    """Test severity classification in _create_match_from_group()."""

    def test_severity_none_identical_trades(self):
        """Test severity is 'none' for identical trades."""
        trade1 = StandardTrade(
            trade_id=1, platform='ml4t.backtest',
            entry_timestamp=datetime(2017, 2, 6, 14, 30, tzinfo=timezone.utc),
            entry_price=73.50, entry_price_component='open', entry_bar_ohlc={},
            exit_timestamp=datetime(2017, 2, 7, 14, 30, tzinfo=timezone.utc),
            exit_price=75.20, exit_price_component='open',
            exit_bar_ohlc={}, exit_reason='signal',
            symbol='AAPL', quantity=100.0, side='long',
            gross_pnl=170.0, entry_commission=0.5, exit_commission=0.5,
            slippage=0.1, net_pnl=168.9,
        )
        trade2 = StandardTrade(
            trade_id=1, platform='vectorbt',
            entry_timestamp=datetime(2017, 2, 6, 14, 30, tzinfo=timezone.utc),
            entry_price=73.50, entry_price_component='open', entry_bar_ohlc={},
            exit_timestamp=datetime(2017, 2, 7, 14, 30, tzinfo=timezone.utc),
            exit_price=75.20, exit_price_component='open',
            exit_bar_ohlc={}, exit_reason='signal',
            symbol='AAPL', quantity=100.0, side='long',
            gross_pnl=170.0, entry_commission=0.5, exit_commission=0.5,
            slippage=0.1, net_pnl=168.9,
        )

        group = [('ml4t.backtest', trade1), ('vectorbt', trade2)]
        match = _create_match_from_group(group)

        assert match.severity == 'none'

    def test_severity_minor_small_price_diff(self):
        """Test severity is 'minor' for small price differences (<1%)."""
        trade1 = StandardTrade(
            trade_id=1, platform='ml4t.backtest',
            entry_timestamp=datetime(2017, 2, 6, 14, 30, tzinfo=timezone.utc),
            entry_price=73.50, entry_price_component='open', entry_bar_ohlc={},
            exit_timestamp=datetime(2017, 2, 7, 14, 30, tzinfo=timezone.utc),
            exit_price=75.20, exit_price_component='open',
            exit_bar_ohlc={}, exit_reason='signal',
            symbol='AAPL', quantity=100.0, side='long',
            gross_pnl=170.0, entry_commission=0.5, exit_commission=0.5,
            slippage=0.1, net_pnl=168.9,
        )
        trade2 = StandardTrade(
            trade_id=1, platform='vectorbt',
            entry_timestamp=datetime(2017, 2, 6, 14, 30, tzinfo=timezone.utc),
            entry_price=73.60,  # 0.136% higher
            entry_price_component='close', entry_bar_ohlc={},  # Different component
            exit_timestamp=datetime(2017, 2, 7, 14, 30, tzinfo=timezone.utc),
            exit_price=75.30,  # 0.133% higher
            exit_price_component='close',
            exit_bar_ohlc={}, exit_reason='signal',
            symbol='AAPL', quantity=100.0, side='long',
            gross_pnl=170.0, entry_commission=0.5, exit_commission=0.5,
            slippage=0.1, net_pnl=168.9,
        )

        group = [('ml4t.backtest', trade1), ('vectorbt', trade2)]
        match = _create_match_from_group(group)

        assert match.severity == 'minor'

    def test_severity_major_medium_price_diff(self):
        """Test severity is 'major' for medium price differences (1-5%)."""
        trade1 = StandardTrade(
            trade_id=1, platform='ml4t.backtest',
            entry_timestamp=datetime(2017, 2, 6, 14, 30, tzinfo=timezone.utc),
            entry_price=73.50, entry_price_component='open', entry_bar_ohlc={},
            exit_timestamp=datetime(2017, 2, 7, 14, 30, tzinfo=timezone.utc),
            exit_price=75.20, exit_price_component='open',
            exit_bar_ohlc={}, exit_reason='signal',
            symbol='AAPL', quantity=100.0, side='long',
            gross_pnl=170.0, entry_commission=0.5, exit_commission=0.5,
            slippage=0.1, net_pnl=168.9,
        )
        trade2 = StandardTrade(
            trade_id=1, platform='vectorbt',
            entry_timestamp=datetime(2017, 2, 6, 14, 30, tzinfo=timezone.utc),
            entry_price=75.00,  # 2.04% higher
            entry_price_component='open', entry_bar_ohlc={},
            exit_timestamp=datetime(2017, 2, 7, 14, 30, tzinfo=timezone.utc),
            exit_price=77.50,  # 3.06% higher
            exit_price_component='open',
            exit_bar_ohlc={}, exit_reason='signal',
            symbol='AAPL', quantity=100.0, side='long',
            gross_pnl=250.0, entry_commission=0.5, exit_commission=0.5,
            slippage=0.1, net_pnl=248.9,
        )

        group = [('ml4t.backtest', trade1), ('vectorbt', trade2)]
        match = _create_match_from_group(group)

        assert match.severity == 'major'

    def test_severity_critical_large_price_diff(self):
        """Test severity is 'critical' for large price differences (>5%)."""
        trade1 = StandardTrade(
            trade_id=1, platform='ml4t.backtest',
            entry_timestamp=datetime(2017, 2, 6, 14, 30, tzinfo=timezone.utc),
            entry_price=73.50, entry_price_component='open', entry_bar_ohlc={},
            exit_timestamp=datetime(2017, 2, 7, 14, 30, tzinfo=timezone.utc),
            exit_price=75.20, exit_price_component='open',
            exit_bar_ohlc={}, exit_reason='signal',
            symbol='AAPL', quantity=100.0, side='long',
            gross_pnl=170.0, entry_commission=0.5, exit_commission=0.5,
            slippage=0.1, net_pnl=168.9,
        )
        trade2 = StandardTrade(
            trade_id=1, platform='vectorbt',
            entry_timestamp=datetime(2017, 2, 6, 14, 30, tzinfo=timezone.utc),
            entry_price=80.00,  # 8.84% higher
            entry_price_component='open', entry_bar_ohlc={},
            exit_timestamp=datetime(2017, 2, 7, 14, 30, tzinfo=timezone.utc),
            exit_price=82.00,  # 9.04% higher
            exit_price_component='open',
            exit_bar_ohlc={}, exit_reason='signal',
            symbol='AAPL', quantity=100.0, side='long',
            gross_pnl=200.0, entry_commission=0.5, exit_commission=0.5,
            slippage=0.1, net_pnl=198.9,
        )

        group = [('ml4t.backtest', trade1), ('vectorbt', trade2)]
        match = _create_match_from_group(group)

        assert match.severity == 'critical'


# ============================================================================
# Test match_trades() - Multiple Trades
# ============================================================================

class TestMatchTradesMultiple:
    """Test matching with multiple trades."""

    def test_multiple_trades_same_platform(self):
        """Test matching multiple trades from same platform."""
        trade1 = StandardTrade(
            trade_id=1, platform='ml4t.backtest',
            entry_timestamp=datetime(2017, 2, 6, 14, 30, tzinfo=timezone.utc),
            entry_price=73.50, entry_price_component='open', entry_bar_ohlc={},
            exit_timestamp=None, exit_price=None, exit_price_component=None,
            exit_bar_ohlc=None, exit_reason=None,
            symbol='AAPL', quantity=100.0, side='long',
            gross_pnl=None, entry_commission=0.5, exit_commission=0.0,
            slippage=0.0, net_pnl=None,
        )
        trade2 = StandardTrade(
            trade_id=2, platform='ml4t.backtest',
            entry_timestamp=datetime(2017, 2, 7, 14, 30, tzinfo=timezone.utc),
            entry_price=75.50, entry_price_component='open', entry_bar_ohlc={},
            exit_timestamp=None, exit_price=None, exit_price_component=None,
            exit_bar_ohlc=None, exit_reason=None,
            symbol='AAPL', quantity=100.0, side='long',
            gross_pnl=None, entry_commission=0.5, exit_commission=0.0,
            slippage=0.0, net_pnl=None,
        )

        trades_by_platform = {'ml4t.backtest': [trade1, trade2]}
        matches = match_trades(trades_by_platform)

        assert len(matches) == 2

    def test_multiple_trades_multiple_platforms(self):
        """Test matching multiple trades across multiple platforms."""
        qe_trade1 = StandardTrade(
            trade_id=1, platform='ml4t.backtest',
            entry_timestamp=datetime(2017, 2, 6, 14, 30, tzinfo=timezone.utc),
            entry_price=73.50, entry_price_component='open', entry_bar_ohlc={},
            exit_timestamp=None, exit_price=None, exit_price_component=None,
            exit_bar_ohlc=None, exit_reason=None,
            symbol='AAPL', quantity=100.0, side='long',
            gross_pnl=None, entry_commission=0.5, exit_commission=0.0,
            slippage=0.0, net_pnl=None,
        )
        qe_trade2 = StandardTrade(
            trade_id=2, platform='ml4t.backtest',
            entry_timestamp=datetime(2017, 2, 7, 14, 30, tzinfo=timezone.utc),
            entry_price=75.50, entry_price_component='open', entry_bar_ohlc={},
            exit_timestamp=None, exit_price=None, exit_price_component=None,
            exit_bar_ohlc=None, exit_reason=None,
            symbol='AAPL', quantity=100.0, side='long',
            gross_pnl=None, entry_commission=0.5, exit_commission=0.0,
            slippage=0.0, net_pnl=None,
        )
        vbt_trade1 = StandardTrade(
            trade_id=1, platform='vectorbt',
            entry_timestamp=datetime(2017, 2, 6, 14, 30, tzinfo=timezone.utc),
            entry_price=73.50, entry_price_component='open', entry_bar_ohlc={},
            exit_timestamp=None, exit_price=None, exit_price_component=None,
            exit_bar_ohlc=None, exit_reason=None,
            symbol='AAPL', quantity=100.0, side='long',
            gross_pnl=None, entry_commission=0.5, exit_commission=0.0,
            slippage=0.0, net_pnl=None,
        )
        vbt_trade2 = StandardTrade(
            trade_id=2, platform='vectorbt',
            entry_timestamp=datetime(2017, 2, 7, 14, 30, tzinfo=timezone.utc),
            entry_price=75.50, entry_price_component='open', entry_bar_ohlc={},
            exit_timestamp=None, exit_price=None, exit_price_component=None,
            exit_bar_ohlc=None, exit_reason=None,
            symbol='AAPL', quantity=100.0, side='long',
            gross_pnl=None, entry_commission=0.5, exit_commission=0.0,
            slippage=0.0, net_pnl=None,
        )

        trades_by_platform = {
            'ml4t.backtest': [qe_trade1, qe_trade2],
            'vectorbt': [vbt_trade1, vbt_trade2],
        }
        matches = match_trades(trades_by_platform)

        assert len(matches) == 2
        # Both matches should have both platforms
        for match in matches:
            assert match.backtest_trade is not None
            assert match.vectorbt_trade is not None


# ============================================================================
# Coverage Note
# ============================================================================

"""
Test Coverage Strategy:

1. **TradeMatch Properties**: Test dataclass properties and methods
   - reference_trade property
   - all_trades property

2. **Basic Matching**: Test core matching logic
   - Single platform (no comparison)
   - Two platforms with exact timestamp match
   - Three platforms within tolerance
   - Tolerance enforcement

3. **Edge Cases**: Test boundary conditions
   - Empty inputs
   - Unmatched trades
   - Missing platform data

4. **Delta Calculations**: Test metric computation
   - Entry/exit timestamp deltas
   - Entry/exit price differences
   - Component tracking

5. **Difference Detection**: Test difference identification
   - No differences (identical trades)
   - Timing differences
   - Price differences
   - Component differences

6. **Severity Classification**: Test severity levels
   - none: Identical trades
   - minor: <1% price difference
   - major: 1-5% price difference
   - critical: >5% price difference

7. **Multiple Trades**: Test with multiple trades
   - Multiple trades same platform
   - Multiple trades multiple platforms

**Expected Coverage**: 90%+ of matcher.py

The matcher is already proven working in test_all_platforms_scenario_001.py.
These tests provide regression coverage and document expected behavior.
"""
