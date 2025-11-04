"""Unit tests for portfolio accounting."""

from datetime import datetime, timedelta

import pytest

from qengine.core.event import FillEvent
from qengine.core.types import OrderSide
from qengine.portfolio.accounting import PortfolioAccounting
from qengine.portfolio.portfolio import Portfolio, Position


class TestPosition:
    """Test suite for Position class."""

    def test_position_initialization(self):
        """Test position initialization."""
        pos = Position(asset_id="AAPL")
        assert pos.asset_id == "AAPL"
        assert pos.quantity == 0.0
        assert pos.cost_basis == 0.0
        assert pos.unrealized_pnl == 0.0

    def test_add_shares(self):
        """Test adding shares to position."""
        pos = Position(asset_id="AAPL")

        # Buy 100 shares at $150
        pos.add_shares(100, 150.0)
        assert pos.quantity == 100
        assert pos.cost_basis == 15000.0
        assert pos.last_price == 150.0

        # Buy 50 more at $155
        pos.add_shares(50, 155.0)
        assert pos.quantity == 150
        assert pos.cost_basis == 22750.0  # 15000 + 7750

    def test_remove_shares(self):
        """Test removing shares from position."""
        pos = Position(asset_id="AAPL")
        pos.add_shares(100, 150.0)

        # Sell 50 shares at $160
        realized = pos.remove_shares(50, 160.0)

        assert pos.quantity == 50
        assert pos.cost_basis == 7500.0  # Half of original
        assert realized == 500.0  # 50 * (160 - 150)
        # Note: realized_pnl is no longer tracked on Position, only returned

    def test_update_price(self):
        """Test updating position price."""
        pos = Position(asset_id="AAPL")
        pos.add_shares(100, 150.0)

        # Price goes up to $160
        pos.update_price(160.0)
        assert pos.last_price == 160.0
        assert pos.unrealized_pnl == 1000.0  # 100 * (160 - 150)
        assert pos.market_value == 16000.0

    def test_close_position(self):
        """Test closing a position completely."""
        pos = Position(asset_id="AAPL")
        pos.add_shares(100, 150.0)
        pos.update_price(160.0)

        # Close position
        pos.add_shares(-100, 160.0)

        assert pos.quantity == 0
        assert pos.unrealized_pnl == 0.0
        assert pos.cost_basis == 0.0
        # Note: Position no longer tracks realized_pnl (tracked at Portfolio level)


class TestPortfolio:
    """Test suite for Portfolio class."""

    def test_portfolio_initialization(self):
        """Test portfolio initialization."""
        portfolio = Portfolio(initial_cash=50000.0)
        assert portfolio.cash == 50000.0
        assert portfolio.initial_cash == 50000.0
        assert portfolio.equity == 50000.0
        assert len(portfolio.positions) == 0

    def test_update_position_buy(self):
        """Test buying a position."""
        portfolio = Portfolio(initial_cash=10000.0)

        # Buy 50 shares at $100
        portfolio.update_position("AAPL", 50, 100.0, commission=1.0)

        assert portfolio.cash == 4999.0  # 10000 - (50*100) - 1
        assert "AAPL" in portfolio.positions
        assert portfolio.positions["AAPL"].quantity == 50
        assert portfolio.total_commission == 1.0

    def test_update_position_sell(self):
        """Test selling a position."""
        portfolio = Portfolio(initial_cash=10000.0)

        # Buy then sell
        portfolio.update_position("AAPL", 50, 100.0, commission=1.0)
        portfolio.update_position("AAPL", -30, 110.0, commission=1.0)

        assert portfolio.cash == 8298.0  # 4999 + (30*110) - 1
        assert portfolio.positions["AAPL"].quantity == 20
        assert portfolio.total_realized_pnl == 300.0  # 30 * (110 - 100)
        assert portfolio.total_commission == 2.0

    def test_update_prices(self):
        """Test updating portfolio prices."""
        portfolio = Portfolio(initial_cash=10000.0)
        portfolio.update_position("AAPL", 50, 100.0)
        portfolio.update_position("GOOGL", 10, 200.0)

        # Update prices
        portfolio.update_prices({"AAPL": 110.0, "GOOGL": 190.0})

        assert portfolio.positions["AAPL"].last_price == 110.0
        assert portfolio.positions["AAPL"].unrealized_pnl == 500.0
        assert portfolio.positions["GOOGL"].last_price == 190.0
        assert portfolio.positions["GOOGL"].unrealized_pnl == -100.0

    def test_equity_calculation(self):
        """Test equity calculation."""
        portfolio = Portfolio(initial_cash=10000.0)
        portfolio.update_position("AAPL", 50, 100.0)  # Cost: 5000

        assert portfolio.cash == 5000.0
        assert portfolio.equity == 10000.0  # 5000 cash + 5000 position

        # Price increases
        portfolio.update_prices({"AAPL": 110.0})
        assert portfolio.equity == 10500.0  # 5000 cash + 5500 position

    def test_returns_calculation(self):
        """Test returns calculation."""
        portfolio = Portfolio(initial_cash=10000.0)
        portfolio.update_position("AAPL", 50, 100.0)
        portfolio.update_prices({"AAPL": 110.0})

        assert portfolio.returns == 0.05  # 5% return

    def test_position_summary(self):
        """Test getting position summary."""
        portfolio = Portfolio(initial_cash=10000.0)
        portfolio.update_position("AAPL", 50, 100.0, commission=10.0, slippage=5.0)
        portfolio.update_prices({"AAPL": 110.0})

        summary = portfolio.get_position_summary()

        assert summary["cash"] == 4990.0
        assert summary["equity"] == 10490.0
        assert summary["positions"] == 1
        assert summary["unrealized_pnl"] == 500.0
        assert summary["commission"] == 10.0
        assert summary["slippage"] == 5.0


class TestPortfolioAccounting:
    """Test suite for PortfolioAccounting."""

    def test_accounting_initialization(self):
        """Test portfolio accounting initialization."""
        accounting = PortfolioAccounting(initial_cash=100000.0)
        assert accounting.portfolio.cash == 100000.0
        assert accounting.high_water_mark == 100000.0
        assert accounting.max_drawdown == 0.0
        assert len(accounting.fills) == 0

    def test_process_fill(self):
        """Test processing a fill event."""
        accounting = PortfolioAccounting(initial_cash=10000.0)

        # Create a buy fill
        fill = FillEvent(
            timestamp=datetime.now(),
            order_id="order1",
            trade_id="trade1",
            asset_id="AAPL",
            side=OrderSide.BUY,
            fill_quantity=50,
            fill_price=100.0,
            commission=1.0,
            slippage=0.5,
        )

        accounting.process_fill(fill)

        assert len(accounting.fills) == 1
        assert accounting.portfolio.cash == 4999.0  # 10000 - (50 * 100) - 1
        assert "AAPL" in accounting.portfolio.positions
        assert accounting.portfolio.total_commission == 1.0
        assert accounting.portfolio.total_slippage == 0.5

    def test_process_sell_fill(self):
        """Test processing sell fills."""
        accounting = PortfolioAccounting(initial_cash=10000.0)

        # Buy fill
        buy_fill = FillEvent(
            timestamp=datetime.now(),
            order_id="order1",
            trade_id="trade1",
            asset_id="AAPL",
            side=OrderSide.BUY,
            fill_quantity=50,
            fill_price=100.0,
            commission=1.0,
        )
        accounting.process_fill(buy_fill)

        # Sell fill at profit
        sell_fill = FillEvent(
            timestamp=datetime.now() + timedelta(minutes=1),
            order_id="order2",
            trade_id="trade2",
            asset_id="AAPL",
            side=OrderSide.SELL,
            fill_quantity=30,
            fill_price=110.0,
            commission=1.0,
        )
        accounting.process_fill(sell_fill)

        assert len(accounting.fills) == 2
        assert accounting.portfolio.positions["AAPL"].quantity == 20
        assert accounting.portfolio.total_realized_pnl == 300.0  # 30 * (110 - 100)

    def test_update_prices_and_metrics(self):
        """Test updating prices and metrics."""
        accounting = PortfolioAccounting(initial_cash=10000.0)

        # Create position
        fill = FillEvent(
            timestamp=datetime.now(),
            order_id="order1",
            trade_id="trade1",
            asset_id="AAPL",
            side=OrderSide.BUY,
            fill_quantity=50,
            fill_price=100.0,
            commission=1.0,
        )
        accounting.process_fill(fill)

        # Update prices
        accounting.update_prices({"AAPL": 110.0}, datetime.now())

        # Check metrics updated
        assert accounting.portfolio.positions["AAPL"].last_price == 110.0
        assert accounting.portfolio.unrealized_pnl == 500.0
        assert len(accounting.equity_curve) > 1

    def test_drawdown_calculation(self):
        """Test drawdown calculation."""
        accounting = PortfolioAccounting(initial_cash=10000.0)

        # Simulate equity changes
        accounting.portfolio.cash = 11000.0
        accounting._update_metrics(datetime.now())
        assert accounting.high_water_mark == 11000.0

        # Drawdown
        accounting.portfolio.cash = 9900.0
        accounting._update_metrics(datetime.now())
        assert accounting.max_drawdown == pytest.approx(0.1)  # 10% drawdown

    def test_performance_metrics(self):
        """Test getting performance metrics."""
        accounting = PortfolioAccounting(initial_cash=10000.0)

        # Create some trades
        fill = FillEvent(
            timestamp=datetime.now(),
            order_id="order1",
            trade_id="trade1",
            asset_id="AAPL",
            side=OrderSide.BUY,
            fill_quantity=50,
            fill_price=100.0,
            commission=10.0,
            slippage=5.0,
        )
        accounting.process_fill(fill)
        accounting.update_prices({"AAPL": 110.0}, datetime.now())

        metrics = accounting.get_performance_metrics()

        assert "total_return" in metrics
        assert "total_pnl" in metrics
        assert "realized_pnl" in metrics
        assert "unrealized_pnl" in metrics
        assert "max_drawdown" in metrics
        assert "total_commission" in metrics
        assert "total_slippage" in metrics
        assert "num_trades" in metrics
        assert metrics["num_trades"] == 1
        assert metrics["total_commission"] == 10.0
        assert metrics["total_slippage"] == 5.0

    def test_dataframe_outputs(self):
        """Test DataFrame output methods."""
        accounting = PortfolioAccounting(initial_cash=10000.0)

        # Create trades
        for i in range(3):
            fill = FillEvent(
                timestamp=datetime.now() + timedelta(minutes=i),
                order_id=f"order{i}",
                trade_id=f"trade{i}",
                asset_id="AAPL",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                fill_quantity=10,
                fill_price=100.0 + i,
                commission=1.0,
            )
            accounting.process_fill(fill)

        # Test trades DataFrame
        trades_df = accounting.get_trades_df()
        assert trades_df is not None
        assert len(trades_df) == 3
        assert "timestamp" in trades_df.columns
        assert "asset_id" in trades_df.columns

        # Test equity curve DataFrame
        equity_df = accounting.get_equity_curve_df()
        assert equity_df is not None
        assert "timestamp" in equity_df.columns
        assert "equity" in equity_df.columns

        # Test positions DataFrame
        positions_df = accounting.get_positions_df()
        assert positions_df is not None
        assert "asset_id" in positions_df.columns
        assert "quantity" in positions_df.columns

    def test_reset(self):
        """Test resetting portfolio."""
        accounting = PortfolioAccounting(initial_cash=10000.0)

        # Create some state
        fill = FillEvent(
            timestamp=datetime.now(),
            order_id="order1",
            trade_id="trade1",
            asset_id="AAPL",
            side=OrderSide.BUY,
            fill_quantity=50,
            fill_price=100.0,
            commission=1.0,
        )
        accounting.process_fill(fill)

        # Reset
        accounting.reset()

        assert accounting.portfolio.cash == 10000.0
        assert len(accounting.fills) == 0
        assert len(accounting.portfolio.positions) == 0
        assert accounting.max_drawdown == 0.0
