"""Test to reproduce Issue #2: Portfolio.get_position() returns None."""

from datetime import datetime

import pytest

from qengine.core.event import FillEvent
from qengine.core.types import OrderSide
from qengine.portfolio.simple import SimplePortfolio


class TestPortfolioGetPosition:
    """Test suite to reproduce and fix Issue #2."""

    @pytest.fixture
    def portfolio(self):
        """Create a simple portfolio."""
        return SimplePortfolio(initial_capital=100000.0)

    @pytest.fixture
    def fill_event(self):
        """Create a sample fill event."""
        return FillEvent(
            timestamp=datetime(2024, 1, 1, 10, 0, 0),
            order_id="order-001",
            trade_id="trade-001",
            asset_id="BTC",
            side=OrderSide.BUY,
            fill_quantity=1.0,
            fill_price=50000.0,
            commission=10.0,
            slippage=5.0,
        )

    def test_get_position_returns_none_before_fill(self, portfolio):
        """Test that get_position returns None when no position exists."""
        position = portfolio.get_position("BTC")
        assert position is None

    def test_get_position_after_fill(self, portfolio, fill_event):
        """Test that get_position returns position after fill event.

        This is the core test for Issue #2:
        - Strategy receives fill event
        - Portfolio updates position
        - Strategy queries get_position()
        - Should return Position object, not None
        """
        # Process fill event
        portfolio.on_fill_event(fill_event)

        # Query position - THIS IS THE BUG
        position = portfolio.get_position("BTC")

        # Should NOT be None
        assert position is not None, "get_position() returned None after fill event!"
        assert position.asset_id == "BTC"
        assert position.quantity == 1.0

    def test_get_position_with_multiple_fills(self, portfolio):
        """Test position tracking with multiple fills."""
        # First buy
        portfolio.on_fill_event(
            FillEvent(
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
                order_id="order-001",
                trade_id="trade-001",
                asset_id="BTC",
                side=OrderSide.BUY,
                fill_quantity=1.0,
                fill_price=50000.0,
                commission=10.0,
                slippage=5.0,
            )
        )

        position = portfolio.get_position("BTC")
        assert position is not None
        assert position.quantity == 1.0

        # Second buy
        portfolio.on_fill_event(
            FillEvent(
                timestamp=datetime(2024, 1, 1, 11, 0, 0),
                order_id="order-002",
                trade_id="trade-002",
                asset_id="BTC",
                side=OrderSide.BUY,
                fill_quantity=0.5,
                fill_price=51000.0,
                commission=5.0,
                slippage=2.5,
            )
        )

        position = portfolio.get_position("BTC")
        assert position is not None
        assert position.quantity == 1.5

    def test_get_position_after_full_close(self, portfolio):
        """Test that position is removed after full close."""
        # Open position
        portfolio.on_fill_event(
            FillEvent(
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
                order_id="order-003",
                trade_id="trade-003",
                asset_id="BTC",
                side=OrderSide.BUY,
                fill_quantity=1.0,
                fill_price=50000.0,
                commission=10.0,
                slippage=5.0,
            )
        )

        assert portfolio.get_position("BTC") is not None

        # Close position
        portfolio.on_fill_event(
            FillEvent(
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                order_id="order-004",
                trade_id="trade-004",
                asset_id="BTC",
                side=OrderSide.SELL,
                fill_quantity=1.0,
                fill_price=52000.0,
                commission=10.0,
                slippage=5.0,
            )
        )

        # Position should be None after full close
        position = portfolio.get_position("BTC")
        assert position is None

    def test_get_position_with_partial_close(self, portfolio):
        """Test position after partial close."""
        # Open position
        portfolio.on_fill_event(
            FillEvent(
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
                order_id="order-005",
                trade_id="trade-005",
                asset_id="BTC",
                side=OrderSide.BUY,
                fill_quantity=1.0,
                fill_price=50000.0,
                commission=10.0,
                slippage=5.0,
            )
        )

        # Partial close
        portfolio.on_fill_event(
            FillEvent(
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                order_id="order-006",
                trade_id="trade-006",
                asset_id="BTC",
                side=OrderSide.SELL,
                fill_quantity=0.3,
                fill_price=52000.0,
                commission=5.0,
                slippage=2.5,
            )
        )

        position = portfolio.get_position("BTC")
        assert position is not None
        assert position.quantity == pytest.approx(0.7, rel=1e-6)

    def test_multiple_assets(self, portfolio):
        """Test position tracking for multiple assets."""
        # BTC position
        portfolio.on_fill_event(
            FillEvent(
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
                order_id="order-007",
                trade_id="trade-007",
                asset_id="BTC",
                side=OrderSide.BUY,
                fill_quantity=1.0,
                fill_price=50000.0,
                commission=10.0,
                slippage=5.0,
            )
        )

        # ETH position
        portfolio.on_fill_event(
            FillEvent(
                timestamp=datetime(2024, 1, 1, 10, 30, 0),
                order_id="order-008",
                trade_id="trade-008",
                asset_id="ETH",
                side=OrderSide.BUY,
                fill_quantity=10.0,
                fill_price=3000.0,
                commission=10.0,
                slippage=5.0,
            )
        )

        # Check both positions
        btc_pos = portfolio.get_position("BTC")
        eth_pos = portfolio.get_position("ETH")

        assert btc_pos is not None
        assert btc_pos.quantity == 1.0

        assert eth_pos is not None
        assert eth_pos.quantity == 10.0

        # Non-existent asset
        assert portfolio.get_position("AAPL") is None


class TestSimplePortfolioMethods:
    """Test suite for SimplePortfolio methods."""

    @pytest.fixture
    def portfolio(self):
        """Create a simple portfolio."""
        return SimplePortfolio(initial_capital=100000.0)

    def test_initialize_logs_startup(self, portfolio):
        """Test that initialize method works without errors."""
        # Should not raise any exceptions
        portfolio.initialize()

    def test_get_total_value_initial(self, portfolio):
        """Test get_total_value returns initial capital."""
        assert portfolio.get_total_value() == 100000.0

    def test_get_total_value_with_position(self, portfolio):
        """Test get_total_value includes position value."""
        # Add position
        portfolio.on_fill_event(
            FillEvent(
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
                order_id="order-001",
                trade_id="trade-001",
                asset_id="BTC",
                side=OrderSide.BUY,
                fill_quantity=1.0,
                fill_price=50000.0,
                commission=10.0,
                slippage=5.0,
            )
        )

        # Update with market price
        from qengine.core.event import MarketEvent
        from qengine.core.types import MarketDataType

        market_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 11, 0, 0),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            price=52000.0,
            close=52000.0,
            volume=1000,
        )

        portfolio.update_market_value(market_event)

        # Total value should be cash + position value
        # Cash after buy: 100000 - 50000 - 10 - 5 = 49985
        # Position value: 1.0 * 52000 = 52000
        # Total: 49985 + 52000 = 101985
        total = portfolio.get_total_value()
        assert total == pytest.approx(101985.0, rel=1e-4)

    def test_get_positions_empty(self, portfolio):
        """Test get_positions returns empty DataFrame when no positions."""
        df = portfolio.get_positions()
        assert len(df) == 0

    def test_get_positions_with_data(self, portfolio):
        """Test get_positions returns DataFrame with position data."""
        portfolio.on_fill_event(
            FillEvent(
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
                order_id="order-001",
                trade_id="trade-001",
                asset_id="BTC",
                side=OrderSide.BUY,
                fill_quantity=1.0,
                fill_price=50000.0,
                commission=10.0,
                slippage=5.0,
            )
        )

        df = portfolio.get_positions()
        assert len(df) == 1
        assert df["asset_id"][0] == "BTC"
        assert df["quantity"][0] == 1.0

    def test_get_trades_empty(self, portfolio):
        """Test get_trades returns empty DataFrame when no trades."""
        df = portfolio.get_trades()
        assert len(df) == 0

    def test_get_trades_with_data(self, portfolio):
        """Test get_trades returns DataFrame with trade data."""
        portfolio.on_fill_event(
            FillEvent(
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
                order_id="order-001",
                trade_id="trade-001",
                asset_id="BTC",
                side=OrderSide.BUY,
                fill_quantity=1.0,
                fill_price=50000.0,
                commission=10.0,
                slippage=5.0,
            )
        )

        df = portfolio.get_trades()
        assert len(df) == 1
        assert df["asset_id"][0] == "BTC"
        assert df["quantity"][0] == 1.0
        assert df["price"][0] == 50000.0

    def test_get_returns_empty(self, portfolio):
        """Test get_returns returns empty series when no state history."""
        returns = portfolio.get_returns()
        assert len(returns) == 0

    def test_get_returns_with_history(self, portfolio):
        """Test get_returns calculates returns from state history."""
        # Create trades to generate state history
        portfolio.on_fill_event(
            FillEvent(
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
                order_id="order-001",
                trade_id="trade-001",
                asset_id="BTC",
                side=OrderSide.BUY,
                fill_quantity=1.0,
                fill_price=50000.0,
                commission=10.0,
                slippage=5.0,
            )
        )

        # Save state to create history
        portfolio.save_state(datetime(2024, 1, 1, 10, 0, 0))

        from qengine.core.event import MarketEvent
        from qengine.core.types import MarketDataType

        # Update with higher price
        market_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 11, 0, 0),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            price=52000.0,
            close=52000.0,
            volume=1000,
        )
        portfolio.update_market_value(market_event)
        portfolio.save_state(datetime(2024, 1, 1, 11, 0, 0))

        returns = portfolio.get_returns()
        assert len(returns) > 0

    def test_calculate_metrics_basic(self, portfolio):
        """Test calculate_metrics returns all expected metrics."""
        metrics = portfolio.calculate_metrics()

        assert "total_return" in metrics
        assert "total_trades" in metrics
        assert "winning_trades" in metrics
        assert "losing_trades" in metrics
        assert "total_commission" in metrics
        assert "total_slippage" in metrics
        assert "final_equity" in metrics
        assert "cash_remaining" in metrics

    def test_calculate_metrics_with_trades(self, portfolio):
        """Test calculate_metrics with actual trades."""
        # Buy
        portfolio.on_fill_event(
            FillEvent(
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
                order_id="order-001",
                trade_id="trade-001",
                asset_id="BTC",
                side=OrderSide.BUY,
                fill_quantity=1.0,
                fill_price=50000.0,
                commission=10.0,
                slippage=5.0,
            )
        )

        # Sell at profit
        portfolio.on_fill_event(
            FillEvent(
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                order_id="order-002",
                trade_id="trade-002",
                asset_id="BTC",
                side=OrderSide.SELL,
                fill_quantity=1.0,
                fill_price=52000.0,
                commission=10.0,
                slippage=5.0,
            )
        )

        metrics = portfolio.calculate_metrics()

        assert metrics["total_trades"] == 2
        assert metrics["total_commission"] == 20.0
        assert metrics["total_slippage"] == 10.0

    def test_finalize_saves_state(self, portfolio):
        """Test finalize saves final state."""
        portfolio.on_fill_event(
            FillEvent(
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
                order_id="order-001",
                trade_id="trade-001",
                asset_id="BTC",
                side=OrderSide.BUY,
                fill_quantity=1.0,
                fill_price=50000.0,
                commission=10.0,
                slippage=5.0,
            )
        )

        initial_history_len = len(portfolio.state_history)
        portfolio.finalize(datetime(2024, 1, 1, 15, 0, 0))

        # Should have added one more state
        assert len(portfolio.state_history) == initial_history_len + 1

    def test_finalize_calculates_pnl(self, portfolio):
        """Test finalize calculates P&L for trades."""
        # Buy
        portfolio.on_fill_event(
            FillEvent(
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
                order_id="order-001",
                trade_id="trade-001",
                asset_id="BTC",
                side=OrderSide.BUY,
                fill_quantity=1.0,
                fill_price=50000.0,
                commission=10.0,
                slippage=5.0,
            )
        )

        # Sell
        portfolio.on_fill_event(
            FillEvent(
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                order_id="order-002",
                trade_id="trade-002",
                asset_id="BTC",
                side=OrderSide.SELL,
                fill_quantity=1.0,
                fill_price=52000.0,
                commission=10.0,
                slippage=5.0,
            )
        )

        portfolio.finalize(datetime(2024, 1, 1, 15, 0, 0))

        # Sell trade should have P&L calculated
        sell_trade = portfolio.trades[1]
        assert "pnl" in sell_trade
        # P&L = (sell_price - buy_price) * quantity - commission
        # = (52000 - 50000) * 1.0 - 10 = 1990
        assert sell_trade["pnl"] == pytest.approx(1990.0, rel=1e-4)

    def test_reset_clears_all_state(self, portfolio):
        """Test reset clears all portfolio state."""
        # Add some trades and state
        portfolio.on_fill_event(
            FillEvent(
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
                order_id="order-001",
                trade_id="trade-001",
                asset_id="BTC",
                side=OrderSide.BUY,
                fill_quantity=1.0,
                fill_price=50000.0,
                commission=10.0,
                slippage=5.0,
            )
        )

        from qengine.core.event import MarketEvent
        from qengine.core.types import MarketDataType

        market_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 11, 0, 0),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            price=52000.0,
            close=52000.0,
            volume=1000,
        )
        portfolio.update_market_value(market_event)
        portfolio.save_state(datetime(2024, 1, 1, 11, 0, 0))

        # Reset
        portfolio.reset()

        # Verify everything is cleared
        assert portfolio.cash == 100000.0
        assert len(portfolio.positions) == 0
        assert len(portfolio.trades) == 0
        assert len(portfolio.current_prices) == 0
        assert len(portfolio.state_history) == 0
        assert portfolio.total_commission == 0.0
        assert portfolio.total_slippage == 0.0
        assert portfolio.total_realized_pnl == 0.0

    def test_update_market_value_with_close_price(self, portfolio):
        """Test update_market_value uses close price when available."""
        from qengine.core.event import MarketEvent
        from qengine.core.types import MarketDataType

        portfolio.on_fill_event(
            FillEvent(
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
                order_id="order-001",
                trade_id="trade-001",
                asset_id="BTC",
                side=OrderSide.BUY,
                fill_quantity=1.0,
                fill_price=50000.0,
                commission=10.0,
                slippage=5.0,
            )
        )

        market_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 11, 0, 0),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            price=51000.0,  # Price field
            close=52000.0,  # Close field (should be preferred)
            volume=1000,
        )

        portfolio.update_market_value(market_event)

        # Should use close price
        assert portfolio.current_prices["BTC"] == 52000.0

    def test_update_market_value_with_price_only(self, portfolio):
        """Test update_market_value uses price when close not available."""
        from qengine.core.event import MarketEvent
        from qengine.core.types import MarketDataType

        portfolio.on_fill_event(
            FillEvent(
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
                order_id="order-001",
                trade_id="trade-001",
                asset_id="BTC",
                side=OrderSide.BUY,
                fill_quantity=1.0,
                fill_price=50000.0,
                commission=10.0,
                slippage=5.0,
            )
        )

        # Market event with only price field (no close)
        market_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 11, 0, 0),
            asset_id="BTC",
            data_type=MarketDataType.TRADE,
            price=51500.0,
            volume=100,
        )

        portfolio.update_market_value(market_event)

        # Should use price field
        assert portfolio.current_prices["BTC"] == 51500.0

    def test_calculate_metrics_with_returns(self, portfolio):
        """Test calculate_metrics includes returns-based metrics."""
        # Create state history to enable returns calculation
        portfolio.on_fill_event(
            FillEvent(
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
                order_id="order-001",
                trade_id="trade-001",
                asset_id="BTC",
                side=OrderSide.BUY,
                fill_quantity=1.0,
                fill_price=50000.0,
                commission=10.0,
                slippage=5.0,
            )
        )

        portfolio.save_state(datetime(2024, 1, 1, 10, 0, 0))

        from qengine.core.event import MarketEvent
        from qengine.core.types import MarketDataType

        # Update with higher price
        market_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 11, 0, 0),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            price=52000.0,
            close=52000.0,
            volume=1000,
        )
        portfolio.update_market_value(market_event)
        portfolio.save_state(datetime(2024, 1, 1, 11, 0, 0))

        # Another update with even higher price
        market_event2 = MarketEvent(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            price=54000.0,
            close=54000.0,
            volume=1000,
        )
        portfolio.update_market_value(market_event2)
        portfolio.save_state(datetime(2024, 1, 1, 12, 0, 0))

        # Now we have multiple states, so returns calculation will work
        metrics = portfolio.calculate_metrics()

        # Should include returns-based metrics
        assert "avg_return" in metrics
        assert "std_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics

        # Add a trade to enable win_rate calculation
        portfolio.trades[0]["pnl"] = 1000.0  # Mark as winning trade

        metrics = portfolio.calculate_metrics()
        assert "win_rate" in metrics
