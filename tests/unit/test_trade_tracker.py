"""Tests for trade tracking functionality."""

from datetime import datetime, timezone

import polars as pl
import pytest

from ml4t.backtest.core.event import FillEvent
from ml4t.backtest.core.types import OrderSide
from ml4t.backtest.execution.trade_tracker import TradeTracker


class TestTradeTracker:
    """Test TradeTracker functionality."""

    def test_empty_tracker(self):
        """Test empty tracker returns empty DataFrame with correct schema."""
        tracker = TradeTracker()
        df = tracker.get_trades_df()

        assert len(df) == 0
        # Check expected columns exist
        expected_columns = {
            "trade_id",
            "asset_id",
            "entry_dt",
            "entry_price",
            "entry_quantity",
            "entry_commission",
            "entry_slippage",
            "entry_order_id",
            "exit_dt",
            "exit_price",
            "exit_quantity",
            "exit_commission",
            "exit_slippage",
            "exit_order_id",
            "pnl",
            "return_pct",
            "duration_bars",
            "direction",
        }
        assert set(df.columns) == expected_columns

    def test_single_long_trade(self):
        """Test single long trade (buy then sell)."""
        tracker = TradeTracker()

        # Entry
        tracker.on_bar()
        entry_fill = FillEvent(
            timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            order_id="order_1",
            trade_id="fill_1",
            asset_id="BTC",
            side=OrderSide.BUY,
            fill_quantity=1.0,
            fill_price=50000.0,
            commission=10.0,
            slippage=5.0,
        )
        trades = tracker.on_fill(entry_fill)
        assert len(trades) == 0  # Position still open

        # Exit
        tracker.on_bar()
        tracker.on_bar()  # 2 bars later
        exit_fill = FillEvent(
            timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            order_id="order_2",
            trade_id="fill_2",
            asset_id="BTC",
            side=OrderSide.SELL,
            fill_quantity=1.0,
            fill_price=51000.0,
            commission=10.0,
            slippage=5.0,
        )
        trades = tracker.on_fill(exit_fill)
        assert len(trades) == 1

        # Check trade details
        trade = trades[0]
        assert trade.trade_id == 0
        assert trade.asset_id == "BTC"
        assert trade.direction == "long"
        assert trade.entry_price == 50000.0
        assert trade.exit_price == 51000.0
        assert trade.entry_quantity == 1.0
        assert trade.exit_quantity == 1.0
        assert trade.duration_bars == 2

        # PnL = (exit - entry) * qty - costs
        # PnL = (51000 - 50000) * 1.0 - (10 + 5 + 10 + 5) = 1000 - 30 = 970
        assert trade.pnl == pytest.approx(970.0)

        # Return = 970 / (50000 * 1.0) * 100 = 1.94%
        assert trade.return_pct == pytest.approx(1.94)

    def test_single_short_trade(self):
        """Test single short trade (sell then buy)."""
        tracker = TradeTracker()

        # Entry (short)
        tracker.on_bar()
        entry_fill = FillEvent(
            timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            order_id="order_1",
            trade_id="fill_1",
            asset_id="ETH",
            side=OrderSide.SELL,
            fill_quantity=10.0,
            fill_price=3000.0,
            commission=10.0,
            slippage=5.0,
        )
        trades = tracker.on_fill(entry_fill)
        assert len(trades) == 0

        # Exit (cover)
        tracker.on_bar()
        exit_fill = FillEvent(
            timestamp=datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
            order_id="order_2",
            trade_id="fill_2",
            asset_id="ETH",
            side=OrderSide.BUY,
            fill_quantity=10.0,
            fill_price=2900.0,
            commission=10.0,
            slippage=5.0,
        )
        trades = tracker.on_fill(exit_fill)
        assert len(trades) == 1

        trade = trades[0]
        assert trade.direction == "short"
        # Short PnL = (entry - exit) * qty - costs
        # = (3000 - 2900) * 10 - 30 = 1000 - 30 = 970
        assert trade.pnl == pytest.approx(970.0)

    def test_fifo_matching(self):
        """Test FIFO (First-In-First-Out) position matching."""
        tracker = TradeTracker()

        # First entry
        tracker.on_bar()
        fill1 = FillEvent(
            timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            order_id="order_1",
            trade_id="fill_1",
            asset_id="BTC",
            side=OrderSide.BUY,
            fill_quantity=1.0,
            fill_price=50000.0,
            commission=10.0,
            slippage=0.0,
        )
        tracker.on_fill(fill1)

        # Second entry (add to position)
        tracker.on_bar()
        fill2 = FillEvent(
            timestamp=datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
            order_id="order_2",
            trade_id="fill_2",
            asset_id="BTC",
            side=OrderSide.BUY,
            fill_quantity=1.0,
            fill_price=51000.0,
            commission=10.0,
            slippage=0.0,
        )
        tracker.on_fill(fill2)

        # Partial exit (closes first position only)
        tracker.on_bar()
        fill3 = FillEvent(
            timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            order_id="order_3",
            trade_id="fill_3",
            asset_id="BTC",
            side=OrderSide.SELL,
            fill_quantity=1.0,
            fill_price=52000.0,
            commission=10.0,
            slippage=0.0,
        )
        trades = tracker.on_fill(fill3)

        # Should close first position (FIFO)
        assert len(trades) == 1
        assert trades[0].entry_price == 50000.0  # First entry
        assert trades[0].exit_price == 52000.0
        assert trades[0].pnl == pytest.approx(2000.0 - 20.0)  # 1980

        # Exit remaining position
        tracker.on_bar()
        fill4 = FillEvent(
            timestamp=datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc),
            order_id="order_4",
            trade_id="fill_4",
            asset_id="BTC",
            side=OrderSide.SELL,
            fill_quantity=1.0,
            fill_price=53000.0,
            commission=10.0,
            slippage=0.0,
        )
        trades = tracker.on_fill(fill4)

        assert len(trades) == 1
        assert trades[0].entry_price == 51000.0  # Second entry
        assert trades[0].exit_price == 53000.0
        assert trades[0].pnl == pytest.approx(2000.0 - 20.0)  # 1980

        # Total trades
        assert tracker.get_trade_count() == 2

    def test_reverse_position(self):
        """Test reversing position (long to short in one fill)."""
        tracker = TradeTracker()

        # Long position
        tracker.on_bar()
        fill1 = FillEvent(
            timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            order_id="order_1",
            trade_id="fill_1",
            asset_id="BTC",
            side=OrderSide.BUY,
            fill_quantity=1.0,
            fill_price=50000.0,
            commission=10.0,
            slippage=0.0,
        )
        tracker.on_fill(fill1)

        # Reverse to short (sell 2.0 = close 1.0 long + open 1.0 short)
        tracker.on_bar()
        fill2 = FillEvent(
            timestamp=datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
            order_id="order_2",
            trade_id="fill_2",
            asset_id="BTC",
            side=OrderSide.SELL,
            fill_quantity=2.0,
            fill_price=51000.0,
            commission=20.0,
            slippage=0.0,
        )
        trades = tracker.on_fill(fill2)

        # Should close long position
        assert len(trades) == 1
        assert trades[0].direction == "long"
        assert trades[0].entry_quantity == 1.0
        assert trades[0].exit_quantity == 1.0

        # Should have new short position open
        assert tracker.get_open_position_count() == 1

    def test_trades_dataframe_output(self):
        """Test DataFrame output format."""
        tracker = TradeTracker()

        # Create two complete trades
        tracker.on_bar()
        tracker.on_fill(
            FillEvent(
                datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                "o1",
                "f1",
                "BTC",
                OrderSide.BUY,
                1.0,
                50000.0,
                10.0,
                5.0,
            )
        )

        tracker.on_bar()
        tracker.on_fill(
            FillEvent(
                datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
                "o2",
                "f2",
                "BTC",
                OrderSide.SELL,
                1.0,
                51000.0,
                10.0,
                5.0,
            )
        )

        tracker.on_bar()
        tracker.on_fill(
            FillEvent(
                datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
                "o3",
                "f3",
                "ETH",
                OrderSide.BUY,
                10.0,
                3000.0,
                10.0,
                5.0,
            )
        )

        tracker.on_bar()
        tracker.on_fill(
            FillEvent(
                datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc),
                "o4",
                "f4",
                "ETH",
                OrderSide.SELL,
                10.0,
                3100.0,
                10.0,
                5.0,
            )
        )

        # Get DataFrame
        df = tracker.get_trades_df()

        assert len(df) == 2
        assert df["trade_id"].to_list() == [0, 1]
        assert df["asset_id"].to_list() == ["BTC", "ETH"]
        assert df["direction"].to_list() == ["long", "long"]
        assert all(df["pnl"] > 0)  # Both profitable

        # Test column access (Polars syntax)
        assert "entry_dt" in df.columns
        assert "exit_dt" in df.columns
        assert "pnl" in df.columns
        assert "return_pct" in df.columns

    def test_statistics(self):
        """Test tracker statistics."""
        tracker = TradeTracker()

        # Process 4 fills (2 complete trades)
        tracker.on_bar()
        tracker.on_fill(
            FillEvent(
                datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                "o1",
                "f1",
                "BTC",
                OrderSide.BUY,
                1.0,
                50000.0,
            )
        )

        tracker.on_bar()
        tracker.on_fill(
            FillEvent(
                datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
                "o2",
                "f2",
                "BTC",
                OrderSide.SELL,
                1.0,
                51000.0,
            )
        )

        tracker.on_bar()
        tracker.on_fill(
            FillEvent(
                datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
                "o3",
                "f3",
                "ETH",
                OrderSide.BUY,
                10.0,
                3000.0,
            )
        )

        tracker.on_bar()
        tracker.on_fill(
            FillEvent(
                datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc),
                "o4",
                "f4",
                "ETH",
                OrderSide.SELL,
                10.0,
                3100.0,
            )
        )

        stats = tracker.get_stats()
        assert stats["total_fills_processed"] == 4
        assert stats["total_trades_completed"] == 2
        assert stats["open_positions"] == 0
        assert stats["avg_fills_per_trade"] == 2.0

    def test_reset(self):
        """Test tracker reset functionality."""
        tracker = TradeTracker()

        # Create a trade
        tracker.on_bar()
        tracker.on_fill(
            FillEvent(
                datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                "o1",
                "f1",
                "BTC",
                OrderSide.BUY,
                1.0,
                50000.0,
            )
        )

        assert tracker.get_open_position_count() == 1

        # Reset
        tracker.reset()

        assert tracker.get_trade_count() == 0
        assert tracker.get_open_position_count() == 0
        assert len(tracker.get_trades_df()) == 0
        stats = tracker.get_stats()
        assert stats["total_fills_processed"] == 0
        assert stats["total_trades_completed"] == 0
