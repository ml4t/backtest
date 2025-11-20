"""Unit tests for TradeJournal."""

from datetime import datetime

import pytest

from ml4t.backtest.core.event import FillEvent, OrderSide
from ml4t.backtest.portfolio.analytics import TradeJournal


class TestTradeJournalInitialization:
    """Test TradeJournal initialization."""

    def test_init_empty(self):
        """Test initialization with empty journal."""
        journal = TradeJournal()

        assert journal.fills == []


class TestTradeJournalRecordFill:
    """Test fill recording."""

    def test_record_single_fill(self):
        """Test recording a single fill event."""
        journal = TradeJournal()

        fill = FillEvent(
            timestamp=datetime(2024, 1, 1, 9, 30),
            order_id="order1",
            trade_id="trade1",
            asset_id="AAPL",
            side=OrderSide.BUY,
            fill_quantity=100,
            fill_price=150.0,
            commission=1.0,
            slippage=0.5,
        )

        journal.record_fill(fill)

        assert len(journal.fills) == 1
        assert journal.fills[0] == fill

    def test_record_multiple_fills(self):
        """Test recording multiple fill events."""
        journal = TradeJournal()

        fills = [
            FillEvent(
                timestamp=datetime(2024, 1, 1, 9, 30),
                order_id="order1",
                trade_id="trade1",
                asset_id="AAPL",
                side=OrderSide.BUY,
                fill_quantity=100,
                fill_price=150.0,
                commission=1.0,
                slippage=0.0,
            ),
            FillEvent(
                timestamp=datetime(2024, 1, 1, 10, 30),
                order_id="order2",
                trade_id="trade2",
                asset_id="AAPL",
                side=OrderSide.SELL,
                fill_quantity=100,
                fill_price=160.0,
                commission=1.0,
                slippage=0.0,
            ),
        ]

        for fill in fills:
            journal.record_fill(fill)

        assert len(journal.fills) == 2
        assert journal.fills == fills


class TestTradeJournalGetTrades:
    """Test get_trades DataFrame export."""

    def test_get_trades_empty(self):
        """Test get_trades with no fills."""
        journal = TradeJournal()

        df = journal.get_trades()

        assert df.is_empty()

    def test_get_trades_with_fills(self):
        """Test get_trades with fills."""
        journal = TradeJournal()

        fills = [
            FillEvent(
                timestamp=datetime(2024, 1, 1, 9, 30),
                order_id="order1",
                trade_id="trade1",
                asset_id="AAPL",
                side=OrderSide.BUY,
                fill_quantity=100,
                fill_price=150.0,
                commission=1.0,
                slippage=0.5,
            ),
            FillEvent(
                timestamp=datetime(2024, 1, 1, 10, 30),
                order_id="order2",
                trade_id="trade2",
                asset_id="GOOGL",
                side=OrderSide.SELL,
                fill_quantity=50,
                fill_price=200.0,
                commission=2.0,
                slippage=0.3,
            ),
        ]

        for fill in fills:
            journal.record_fill(fill)

        df = journal.get_trades()

        assert df.shape[0] == 2
        assert "timestamp" in df.columns
        assert "order_id" in df.columns
        assert "trade_id" in df.columns
        assert "asset_id" in df.columns
        assert "side" in df.columns
        assert "quantity" in df.columns
        assert "price" in df.columns
        assert "commission" in df.columns
        assert "slippage" in df.columns

        # Verify first row
        assert df["asset_id"][0] == "AAPL"
        assert df["side"][0] == "buy"
        assert df["quantity"][0] == 100
        assert df["price"][0] == 150.0
        assert df["commission"][0] == 1.0
        assert df["slippage"][0] == 0.5


class TestTradeJournalWinRate:
    """Test win_rate calculation."""

    def test_win_rate_empty(self):
        """Test win_rate with no fills."""
        journal = TradeJournal()

        assert journal.calculate_win_rate() == 0.0

    def test_win_rate_only_buys(self):
        """Test win_rate with only buy orders."""
        journal = TradeJournal()

        fills = [
            FillEvent(
                timestamp=datetime(2024, 1, 1, 9, 30),
                order_id="order1",
                trade_id="trade1",
                asset_id="AAPL",
                side=OrderSide.BUY,
                fill_quantity=100,
                fill_price=150.0,
                commission=1.0,
                slippage=0.0,
            ),
        ]

        for fill in fills:
            journal.record_fill(fill)

        assert journal.calculate_win_rate() == 0.0  # No closed trades yet

    def test_win_rate_all_winning_trades(self):
        """Test win_rate with all winning trades."""
        journal = TradeJournal()

        # Buy at 150, sell at 160 (win)
        fills = [
            FillEvent(
                timestamp=datetime(2024, 1, 1, 9, 30),
                order_id="order1",
                trade_id="trade1",
                asset_id="AAPL",
                side=OrderSide.BUY,
                fill_quantity=100,
                fill_price=150.0,
                commission=1.0,
                slippage=0.0,
            ),
            FillEvent(
                timestamp=datetime(2024, 1, 1, 10, 30),
                order_id="order2",
                trade_id="trade2",
                asset_id="AAPL",
                side=OrderSide.SELL,
                fill_quantity=100,
                fill_price=160.0,
                commission=1.0,
                slippage=0.0,
            ),
            # Buy at 200, sell at 210 (win)
            FillEvent(
                timestamp=datetime(2024, 1, 1, 11, 30),
                order_id="order3",
                trade_id="trade3",
                asset_id="GOOGL",
                side=OrderSide.BUY,
                fill_quantity=50,
                fill_price=200.0,
                commission=1.0,
                slippage=0.0,
            ),
            FillEvent(
                timestamp=datetime(2024, 1, 1, 12, 30),
                order_id="order4",
                trade_id="trade4",
                asset_id="GOOGL",
                side=OrderSide.SELL,
                fill_quantity=50,
                fill_price=210.0,
                commission=1.0,
                slippage=0.0,
            ),
        ]

        for fill in fills:
            journal.record_fill(fill)

        assert journal.calculate_win_rate() == 1.0  # 100% win rate

    def test_win_rate_mixed_trades(self):
        """Test win_rate with mixed winning/losing trades."""
        journal = TradeJournal()

        # Buy at 150, sell at 160 (win)
        # Buy at 200, sell at 190 (loss)
        fills = [
            FillEvent(
                timestamp=datetime(2024, 1, 1, 9, 30),
                order_id="order1",
                trade_id="trade1",
                asset_id="AAPL",
                side=OrderSide.BUY,
                fill_quantity=100,
                fill_price=150.0,
                commission=1.0,
                slippage=0.0,
            ),
            FillEvent(
                timestamp=datetime(2024, 1, 1, 10, 30),
                order_id="order2",
                trade_id="trade2",
                asset_id="AAPL",
                side=OrderSide.SELL,
                fill_quantity=100,
                fill_price=160.0,
                commission=1.0,
                slippage=0.0,
            ),
            FillEvent(
                timestamp=datetime(2024, 1, 1, 11, 30),
                order_id="order3",
                trade_id="trade3",
                asset_id="GOOGL",
                side=OrderSide.BUY,
                fill_quantity=50,
                fill_price=200.0,
                commission=1.0,
                slippage=0.0,
            ),
            FillEvent(
                timestamp=datetime(2024, 1, 1, 12, 30),
                order_id="order4",
                trade_id="trade4",
                asset_id="GOOGL",
                side=OrderSide.SELL,
                fill_quantity=50,
                fill_price=190.0,
                commission=1.0,
                slippage=0.0,
            ),
        ]

        for fill in fills:
            journal.record_fill(fill)

        assert journal.calculate_win_rate() == 0.5  # 50% win rate (1 win, 1 loss)


class TestTradeJournalProfitFactor:
    """Test profit_factor calculation."""

    def test_profit_factor_empty(self):
        """Test profit_factor with no fills."""
        journal = TradeJournal()

        assert journal.calculate_profit_factor() == 0.0

    def test_profit_factor_only_profits(self):
        """Test profit_factor with only profitable trades."""
        journal = TradeJournal()

        # Buy at 150, sell at 160 (profit = 1000)
        fills = [
            FillEvent(
                timestamp=datetime(2024, 1, 1, 9, 30),
                order_id="order1",
                trade_id="trade1",
                asset_id="AAPL",
                side=OrderSide.BUY,
                fill_quantity=100,
                fill_price=150.0,
                commission=1.0,
                slippage=0.0,
            ),
            FillEvent(
                timestamp=datetime(2024, 1, 1, 10, 30),
                order_id="order2",
                trade_id="trade2",
                asset_id="AAPL",
                side=OrderSide.SELL,
                fill_quantity=100,
                fill_price=160.0,
                commission=1.0,
                slippage=0.0,
            ),
        ]

        for fill in fills:
            journal.record_fill(fill)

        # Profit factor should be infinity (no losses)
        assert journal.calculate_profit_factor() == float("inf")

    def test_profit_factor_only_losses(self):
        """Test profit_factor with only losing trades."""
        journal = TradeJournal()

        # Buy at 160, sell at 150 (loss = -1000)
        fills = [
            FillEvent(
                timestamp=datetime(2024, 1, 1, 9, 30),
                order_id="order1",
                trade_id="trade1",
                asset_id="AAPL",
                side=OrderSide.BUY,
                fill_quantity=100,
                fill_price=160.0,
                commission=1.0,
                slippage=0.0,
            ),
            FillEvent(
                timestamp=datetime(2024, 1, 1, 10, 30),
                order_id="order2",
                trade_id="trade2",
                asset_id="AAPL",
                side=OrderSide.SELL,
                fill_quantity=100,
                fill_price=150.0,
                commission=1.0,
                slippage=0.0,
            ),
        ]

        for fill in fills:
            journal.record_fill(fill)

        # Profit factor should be 0 (no profits)
        assert journal.calculate_profit_factor() == 0.0

    def test_profit_factor_mixed_trades(self):
        """Test profit_factor with mixed winning/losing trades."""
        journal = TradeJournal()

        # Buy at 100, sell at 120 (profit = 2000)
        # Buy at 200, sell at 190 (loss = -500)
        fills = [
            FillEvent(
                timestamp=datetime(2024, 1, 1, 9, 30),
                order_id="order1",
                trade_id="trade1",
                asset_id="AAPL",
                side=OrderSide.BUY,
                fill_quantity=100,
                fill_price=100.0,
                commission=0.0,
                slippage=0.0,
            ),
            FillEvent(
                timestamp=datetime(2024, 1, 1, 10, 30),
                order_id="order2",
                trade_id="trade2",
                asset_id="AAPL",
                side=OrderSide.SELL,
                fill_quantity=100,
                fill_price=120.0,
                commission=0.0,
                slippage=0.0,
            ),
            FillEvent(
                timestamp=datetime(2024, 1, 1, 11, 30),
                order_id="order3",
                trade_id="trade3",
                asset_id="GOOGL",
                side=OrderSide.BUY,
                fill_quantity=50,
                fill_price=200.0,
                commission=0.0,
                slippage=0.0,
            ),
            FillEvent(
                timestamp=datetime(2024, 1, 1, 12, 30),
                order_id="order4",
                trade_id="trade4",
                asset_id="GOOGL",
                side=OrderSide.SELL,
                fill_quantity=50,
                fill_price=190.0,
                commission=0.0,
                slippage=0.0,
            ),
        ]

        for fill in fills:
            journal.record_fill(fill)

        # Profit factor = gross_profit / gross_loss = 2000 / 500 = 4.0
        assert journal.calculate_profit_factor() == pytest.approx(4.0, rel=1e-6)


class TestTradeJournalAverageCommission:
    """Test avg_commission calculation."""

    def test_avg_commission_empty(self):
        """Test avg_commission with no fills."""
        journal = TradeJournal()

        assert journal.calculate_avg_commission() == 0.0

    def test_avg_commission_with_fills(self):
        """Test avg_commission with fills."""
        journal = TradeJournal()

        fills = [
            FillEvent(
                timestamp=datetime(2024, 1, 1, 9, 30),
                order_id="order1",
                trade_id="trade1",
                asset_id="AAPL",
                side=OrderSide.BUY,
                fill_quantity=100,
                fill_price=150.0,
                commission=1.0,
                slippage=0.0,
            ),
            FillEvent(
                timestamp=datetime(2024, 1, 1, 10, 30),
                order_id="order2",
                trade_id="trade2",
                asset_id="AAPL",
                side=OrderSide.SELL,
                fill_quantity=100,
                fill_price=160.0,
                commission=2.0,
                slippage=0.0,
            ),
            FillEvent(
                timestamp=datetime(2024, 1, 1, 11, 30),
                order_id="order3",
                trade_id="trade3",
                asset_id="GOOGL",
                side=OrderSide.BUY,
                fill_quantity=50,
                fill_price=200.0,
                commission=3.0,
                slippage=0.0,
            ),
        ]

        for fill in fills:
            journal.record_fill(fill)

        # Average commission = (1.0 + 2.0 + 3.0) / 3 = 2.0
        assert journal.calculate_avg_commission() == pytest.approx(2.0, rel=1e-6)


class TestTradeJournalAverageSlippage:
    """Test avg_slippage calculation."""

    def test_avg_slippage_empty(self):
        """Test avg_slippage with no fills."""
        journal = TradeJournal()

        assert journal.calculate_avg_slippage() == 0.0

    def test_avg_slippage_with_fills(self):
        """Test avg_slippage with fills."""
        journal = TradeJournal()

        fills = [
            FillEvent(
                timestamp=datetime(2024, 1, 1, 9, 30),
                order_id="order1",
                trade_id="trade1",
                asset_id="AAPL",
                side=OrderSide.BUY,
                fill_quantity=100,
                fill_price=150.0,
                commission=1.0,
                slippage=0.5,
            ),
            FillEvent(
                timestamp=datetime(2024, 1, 1, 10, 30),
                order_id="order2",
                trade_id="trade2",
                asset_id="AAPL",
                side=OrderSide.SELL,
                fill_quantity=100,
                fill_price=160.0,
                commission=2.0,
                slippage=1.0,
            ),
            FillEvent(
                timestamp=datetime(2024, 1, 1, 11, 30),
                order_id="order3",
                trade_id="trade3",
                asset_id="GOOGL",
                side=OrderSide.BUY,
                fill_quantity=50,
                fill_price=200.0,
                commission=3.0,
                slippage=1.5,
            ),
        ]

        for fill in fills:
            journal.record_fill(fill)

        # Average slippage = (0.5 + 1.0 + 1.5) / 3 = 1.0
        assert journal.calculate_avg_slippage() == pytest.approx(1.0, rel=1e-6)


class TestTradeJournalReset:
    """Test reset functionality."""

    def test_reset_clears_fills(self):
        """Test that reset clears all fills."""
        journal = TradeJournal()

        # Add some fills
        fills = [
            FillEvent(
                timestamp=datetime(2024, 1, 1, 9, 30),
                order_id="order1",
                trade_id="trade1",
                asset_id="AAPL",
                side=OrderSide.BUY,
                fill_quantity=100,
                fill_price=150.0,
                commission=1.0,
                slippage=0.5,
            ),
        ]

        for fill in fills:
            journal.record_fill(fill)

        # Reset
        journal.reset()

        # Verify cleared
        assert journal.fills == []
        assert journal.calculate_win_rate() == 0.0
        assert journal.calculate_profit_factor() == 0.0
        assert journal.calculate_avg_commission() == 0.0
        assert journal.calculate_avg_slippage() == 0.0
