"""Integration tests for Portfolio facade.

This module tests the Portfolio facade that combines:
- PositionTracker (core position/cash tracking)
- PerformanceAnalyzer (metrics and analytics, optional)
- TradeJournal (trade history and persistence)

Test Coverage:
- Basic facade functionality with all three components
- HFT mode (track_analytics=False)
- Custom analyzer/journal classes
- Backward compatibility
- Component integration
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from ml4t.backtest.core.event import FillEvent, OrderSide
from ml4t.backtest.core.precision import PrecisionManager
from ml4t.backtest.portfolio.portfolio import Portfolio
from ml4t.backtest.portfolio.analytics import PerformanceAnalyzer, TradeJournal


# ===== Test Fixtures =====


@pytest.fixture
def portfolio():
    """Create default portfolio for testing."""
    return Portfolio(initial_cash=100000.0, currency="USD")


@pytest.fixture
def hft_portfolio():
    """Create HFT portfolio with analytics disabled."""
    return Portfolio(initial_cash=100000.0, track_analytics=False)


@pytest.fixture
def fill_buy_btc():
    """Create a buy fill event for BTC."""
    return FillEvent(
        timestamp=datetime(2025, 1, 1, 10, 0, 0),
        order_id="order_001",
        trade_id="trade_001",
        asset_id="BTC",
        side=OrderSide.BUY,
        fill_quantity=Decimal("1.0"),
        fill_price=Decimal("50000.0"),
        commission=10.0,
        slippage=5.0,
    )


@pytest.fixture
def fill_sell_btc():
    """Create a sell fill event for BTC."""
    return FillEvent(
        timestamp=datetime(2025, 1, 1, 12, 0, 0),
        order_id="order_002",
        trade_id="trade_002",
        asset_id="BTC",
        side=OrderSide.SELL,
        fill_quantity=Decimal("1.0"),
        fill_price=Decimal("51000.0"),
        commission=10.0,
        slippage=5.0,
    )


# ===== Basic Facade Tests =====


def test_portfolio_initialization(portfolio):
    """Test basic initialization."""
    assert portfolio.cash == 100000.0
    assert portfolio.equity == 100000.0
    assert portfolio.currency == "USD"
    assert portfolio.initial_cash == 100000.0
    assert len(portfolio.positions) == 0


def test_portfolio_has_all_components(portfolio):
    """Test that facade has all three components."""
    # PositionTracker
    assert portfolio.tracker is not None
    assert portfolio.tracker.initial_cash == 100000.0

    # PerformanceAnalyzer (enabled by default)
    assert portfolio.analyzer is not None
    assert isinstance(portfolio.analyzer, PerformanceAnalyzer)

    # TradeJournal
    assert portfolio.journal is not None
    assert isinstance(portfolio.journal, TradeJournal)


def test_on_fill_event_updates_all_components(portfolio, fill_buy_btc):
    """Test that on_fill_event() delegates to all components."""
    portfolio.on_fill_event(fill_buy_btc)

    # PositionTracker updated
    assert portfolio.cash < 100000.0  # Cash reduced
    position = portfolio.get_position("BTC")
    assert position is not None
    assert position.quantity == 1.0

    # PerformanceAnalyzer updated
    assert len(portfolio.analyzer.timestamps) == 1

    # TradeJournal updated
    trades = portfolio.get_trades()
    assert len(trades) == 1


def test_on_fill_event_buy_then_sell(portfolio, fill_buy_btc, fill_sell_btc):
    """Test buy then sell updates all components correctly."""
    # Buy
    portfolio.on_fill_event(fill_buy_btc)
    cash_after_buy = portfolio.cash

    # Sell
    portfolio.on_fill_event(fill_sell_btc)
    cash_after_sell = portfolio.cash

    # Position closed
    position = portfolio.get_position("BTC")
    assert position is None

    # Realized P&L recorded
    assert portfolio.total_realized_pnl > 0  # Made profit

    # Cash increased from sale
    assert cash_after_sell > cash_after_buy

    # Journal has 2 fills
    trades = portfolio.get_trades()
    assert len(trades) == 2


# ===== Delegation Tests =====


def test_delegate_to_position_tracker(portfolio, fill_buy_btc):
    """Test delegation to PositionTracker."""
    portfolio.on_fill_event(fill_buy_btc)

    # Properties delegated
    assert portfolio.cash == portfolio.tracker.cash
    assert portfolio.equity == portfolio.tracker.equity
    assert portfolio.returns == portfolio.tracker.returns
    assert portfolio.unrealized_pnl == portfolio.tracker.unrealized_pnl
    assert portfolio.total_realized_pnl == portfolio.tracker.total_realized_pnl
    assert portfolio.total_commission == portfolio.tracker.total_commission
    assert portfolio.total_slippage == portfolio.tracker.total_slippage

    # Methods delegated
    position_from_portfolio = portfolio.get_position("BTC")
    position_from_tracker = portfolio.tracker.get_position("BTC")
    assert position_from_portfolio == position_from_tracker

    # update_prices delegated
    portfolio.update_prices({"BTC": 52000.0})
    assert portfolio.tracker.positions["BTC"].last_price == 52000.0


def test_delegate_to_performance_analyzer(portfolio, fill_buy_btc):
    """Test delegation to PerformanceAnalyzer."""
    portfolio.on_fill_event(fill_buy_btc)

    # Metrics delegated
    metrics = portfolio.get_performance_metrics()
    assert "max_drawdown" in metrics
    assert "current_equity" in metrics
    assert "current_cash" in metrics

    # calculate_sharpe_ratio delegated
    sharpe = portfolio.calculate_sharpe_ratio()
    assert sharpe is None  # Insufficient data


def test_delegate_to_trade_journal(portfolio, fill_buy_btc, fill_sell_btc):
    """Test delegation to TradeJournal."""
    portfolio.on_fill_event(fill_buy_btc)
    portfolio.on_fill_event(fill_sell_btc)

    # get_trades delegated
    trades = portfolio.get_trades()
    assert len(trades) == 2
    assert trades["asset_id"][0] == "BTC"


# ===== HFT Mode Tests (track_analytics=False) =====


def test_hft_mode_no_analyzer(hft_portfolio):
    """Test that HFT mode has no analyzer."""
    assert hft_portfolio.analyzer is None


def test_hft_mode_get_performance_metrics_raises(hft_portfolio, fill_buy_btc):
    """Test that get_performance_metrics() raises error in HFT mode."""
    hft_portfolio.on_fill_event(fill_buy_btc)

    with pytest.raises(ValueError, match="Analytics disabled"):
        hft_portfolio.get_performance_metrics()


def test_hft_mode_calculate_sharpe_ratio_returns_none(hft_portfolio, fill_buy_btc):
    """Test that calculate_sharpe_ratio() returns None in HFT mode."""
    hft_portfolio.on_fill_event(fill_buy_btc)

    sharpe = hft_portfolio.calculate_sharpe_ratio()
    assert sharpe is None


def test_hft_mode_still_has_tracker_and_journal(hft_portfolio, fill_buy_btc):
    """Test that HFT mode still has PositionTracker and TradeJournal."""
    hft_portfolio.on_fill_event(fill_buy_btc)

    # PositionTracker works
    assert hft_portfolio.cash < 100000.0
    position = hft_portfolio.get_position("BTC")
    assert position is not None

    # TradeJournal works
    trades = hft_portfolio.get_trades()
    assert len(trades) == 1


# ===== Custom Components Tests =====


class CustomAnalyzer(PerformanceAnalyzer):
    """Custom analyzer for testing."""

    def custom_metric(self) -> float:
        """Custom metric for testing."""
        return 42.0


class CustomJournal(TradeJournal):
    """Custom journal for testing."""

    def custom_export(self) -> str:
        """Custom export for testing."""
        return "custom_export"


def test_custom_analyzer_class():
    """Test custom analyzer class."""
    portfolio = Portfolio(initial_cash=100000.0, analyzer_class=CustomAnalyzer)

    # Custom analyzer used
    assert isinstance(portfolio.analyzer, CustomAnalyzer)
    assert portfolio.analyzer.custom_metric() == 42.0


def test_custom_journal_class():
    """Test custom journal class."""
    portfolio = Portfolio(initial_cash=100000.0, journal_class=CustomJournal)

    # Custom journal used
    assert isinstance(portfolio.journal, CustomJournal)
    assert portfolio.journal.custom_export() == "custom_export"


# ===== Backward Compatibility Tests =====


def test_backward_compat_positions_property(portfolio, fill_buy_btc):
    """Test positions property (backward compatibility)."""
    portfolio.on_fill_event(fill_buy_btc)

    # Can access positions directly
    assert "BTC" in portfolio.positions
    assert portfolio.positions["BTC"].quantity == 1.0


def test_backward_compat_get_current_state(portfolio, fill_buy_btc):
    """Test get_current_state() (backward compatibility)."""
    portfolio.on_fill_event(fill_buy_btc)

    state = portfolio.get_current_state(datetime(2025, 1, 1, 10, 0, 0))

    # PortfolioState created
    assert state.cash == portfolio.cash
    assert state.total_commission == portfolio.total_commission
    assert "BTC" in state.positions


def test_backward_compat_save_state(portfolio, fill_buy_btc):
    """Test save_state() (backward compatibility)."""
    portfolio.on_fill_event(fill_buy_btc)
    portfolio.save_state(datetime(2025, 1, 1, 10, 0, 0))

    # State saved to history
    assert len(portfolio.state_history) == 1
    assert portfolio.state_history[0].cash == portfolio.cash


def test_backward_compat_get_position_summary(portfolio, fill_buy_btc):
    """Test get_position_summary() (backward compatibility)."""
    portfolio.on_fill_event(fill_buy_btc)

    summary = portfolio.get_position_summary()

    # Summary has expected keys
    assert "cash" in summary
    assert "equity" in summary
    assert "positions" in summary
    assert "realized_pnl" in summary
    assert summary["positions"] == 1


# ===== Precision Manager Tests =====


def test_precision_manager():
    """Test that precision manager is passed to tracker."""
    pm = PrecisionManager(position_decimals=8, price_decimals=2, cash_decimals=2)
    portfolio = Portfolio(initial_cash=100000.0, precision_manager=pm)

    # Precision manager used
    assert portfolio.tracker.precision_manager == pm
    assert portfolio.precision_manager == pm


# ===== Complex Scenario Tests =====


def test_multiple_assets(portfolio):
    """Test portfolio with multiple assets."""
    # Buy BTC
    fill_btc = FillEvent(
        timestamp=datetime(2025, 1, 1, 10, 0, 0),
        order_id="order_btc",
        trade_id="trade_btc",
        asset_id="BTC",
        side=OrderSide.BUY,
        fill_quantity=Decimal("1.0"),
        fill_price=Decimal("50000.0"),
        commission=10.0,
        slippage=5.0,
    )
    portfolio.on_fill_event(fill_btc)

    # Buy ETH
    fill_eth = FillEvent(
        timestamp=datetime(2025, 1, 1, 11, 0, 0),
        order_id="order_eth",
        trade_id="trade_eth",
        asset_id="ETH",
        side=OrderSide.BUY,
        fill_quantity=Decimal("10.0"),
        fill_price=Decimal("3000.0"),
        commission=5.0,
        slippage=2.0,
    )
    portfolio.on_fill_event(fill_eth)

    # Both positions exist
    assert portfolio.get_position("BTC") is not None
    assert portfolio.get_position("ETH") is not None
    assert len(portfolio.positions) == 2

    # Journal has both fills
    trades = portfolio.get_trades()
    assert len(trades) == 2

    # Analyzer tracked both events
    assert len(portfolio.analyzer.timestamps) == 2


def test_round_trip_trade(portfolio):
    """Test complete round trip: buy, price update, sell."""
    # Buy BTC
    fill_buy = FillEvent(
        timestamp=datetime(2025, 1, 1, 10, 0, 0),
        order_id="order_buy",
        trade_id="trade_buy",
        asset_id="BTC",
        side=OrderSide.BUY,
        fill_quantity=Decimal("1.0"),
        fill_price=Decimal("50000.0"),
        commission=10.0,
        slippage=5.0,
    )
    portfolio.on_fill_event(fill_buy)
    cash_after_buy = portfolio.cash

    # Price increases
    portfolio.update_prices({"BTC": 52000.0})
    assert portfolio.unrealized_pnl > 0

    # Sell BTC
    fill_sell = FillEvent(
        timestamp=datetime(2025, 1, 1, 12, 0, 0),
        order_id="order_sell",
        trade_id="trade_sell",
        asset_id="BTC",
        side=OrderSide.SELL,
        fill_quantity=Decimal("1.0"),
        fill_price=Decimal("52000.0"),
        commission=10.0,
        slippage=5.0,
    )
    portfolio.on_fill_event(fill_sell)
    cash_after_sell = portfolio.cash

    # Position closed
    assert portfolio.get_position("BTC") is None

    # Realized profit
    expected_profit = 52000.0 - 50000.0  # Price diff
    assert portfolio.total_realized_pnl == pytest.approx(expected_profit, abs=1.0)

    # Cash increased
    assert cash_after_sell > cash_after_buy

    # Journal has complete trade
    trades = portfolio.get_trades()
    assert len(trades) == 2


def test_save_state_during_trading(portfolio):
    """Test saving state snapshots during trading."""
    timestamps = [
        datetime(2025, 1, 1, 10, 0, 0),
        datetime(2025, 1, 1, 11, 0, 0),
        datetime(2025, 1, 1, 12, 0, 0),
    ]

    # Buy BTC
    fill = FillEvent(
        timestamp=timestamps[0],
        order_id="order_snapshot_buy",
        trade_id="trade_snapshot_buy",
        asset_id="BTC",
        side=OrderSide.BUY,
        fill_quantity=Decimal("1.0"),
        fill_price=Decimal("50000.0"),
        commission=10.0,
        slippage=5.0,
    )
    portfolio.on_fill_event(fill)
    portfolio.save_state(timestamps[0])

    # Price update
    portfolio.update_prices({"BTC": 51000.0})
    portfolio.save_state(timestamps[1])

    # Sell
    fill_sell = FillEvent(
        timestamp=timestamps[2],
        order_id="order_snapshot_sell",
        trade_id="trade_snapshot_sell",
        asset_id="BTC",
        side=OrderSide.SELL,
        fill_quantity=Decimal("1.0"),
        fill_price=Decimal("51000.0"),
        commission=10.0,
        slippage=5.0,
    )
    portfolio.on_fill_event(fill_sell)
    portfolio.save_state(timestamps[2])

    # Three snapshots saved
    assert len(portfolio.state_history) == 3

    # Snapshots have correct sequence
    assert portfolio.state_history[0].timestamp == timestamps[0]
    assert portfolio.state_history[1].timestamp == timestamps[1]
    assert portfolio.state_history[2].timestamp == timestamps[2]

    # Last snapshot has no positions (closed)
    assert len(portfolio.state_history[2].positions) == 0
