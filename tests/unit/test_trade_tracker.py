"""Tests for trade tracking functionality."""

from datetime import datetime, timezone

import polars as pl
import pytest

from ml4t.backtest.core.event import FillEvent, MarketEvent
from ml4t.backtest.core.types import MarketDataType, OrderSide
from ml4t.backtest.execution.trade_tracker import TradeTracker
from ml4t.backtest.reporting.trade_schema import ExitReason


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


class TestMLTradeRecording:
    """Test ML and risk attribution in trade recording."""

    def test_ml_signal_capture_at_entry_and_exit(self):
        """Test capturing ML signals at entry and exit."""
        tracker = TradeTracker()

        # Entry with ML signals
        entry_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            close=50000.0,
            signals={
                "ml_score": 0.85,
                "predicted_return": 0.03,
                "confidence": 0.9,
                "atr": 1200.0,
                "volatility": 0.02,
                "momentum": 0.05,
                "rsi": 65.0,
            },
            context={
                "vix": 15.0,
                "market_regime": "bull",
                "sector_performance": 0.02,
            }
        )

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
            metadata={"stop_loss_price": 48000.0, "take_profit_price": 52000.0}
        )
        tracker.on_fill(entry_fill, entry_event)

        # Exit with different ML signals
        exit_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            close=51000.0,
            signals={
                "ml_score": 0.45,
                "predicted_return": -0.01,
                "confidence": 0.7,
                "atr": 1250.0,
                "volatility": 0.025,
                "momentum": -0.02,
                "rsi": 70.0,
            },
            context={
                "vix": 18.0,
                "market_regime": "neutral",
                "sector_performance": 0.01,
            }
        )

        tracker.on_bar()
        tracker.on_bar()
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
            metadata={"exit_reason": "take_profit"}
        )
        trades = tracker.on_fill(exit_fill, exit_event)

        assert len(trades) == 1
        trade = trades[0]

        # Convert to MLTradeRecord
        ml_trade = tracker.to_ml_trade_record(trade)

        # Check ML signals at entry
        assert ml_trade.ml_score_entry == 0.85
        assert ml_trade.predicted_return_entry == 0.03
        assert ml_trade.confidence_entry == 0.9

        # Check ML signals at exit
        assert ml_trade.ml_score_exit == 0.45
        assert ml_trade.predicted_return_exit == -0.01
        assert ml_trade.confidence_exit == 0.7

        # Check features at entry
        assert ml_trade.atr_entry == 1200.0
        assert ml_trade.volatility_entry == 0.02
        assert ml_trade.momentum_entry == 0.05
        assert ml_trade.rsi_entry == 65.0

        # Check features at exit
        assert ml_trade.atr_exit == 1250.0
        assert ml_trade.volatility_exit == 0.025
        assert ml_trade.momentum_exit == -0.02
        assert ml_trade.rsi_exit == 70.0

        # Check context at entry
        assert ml_trade.vix_entry == 15.0
        assert ml_trade.market_regime_entry == "bull"
        assert ml_trade.sector_performance_entry == 0.02

        # Check context at exit
        assert ml_trade.vix_exit == 18.0
        assert ml_trade.market_regime_exit == "neutral"
        assert ml_trade.sector_performance_exit == 0.01

        # Check risk management
        assert ml_trade.stop_loss_price == 48000.0
        assert ml_trade.take_profit_price == 52000.0
        assert ml_trade.exit_reason == ExitReason.TAKE_PROFIT

        # Check timing
        assert ml_trade.duration_bars == 2
        assert ml_trade.duration_seconds == 7200.0  # 2 hours

    def test_ml_recording_without_market_events(self):
        """Test backward compatibility when no market events provided."""
        tracker = TradeTracker()

        # Entry without market event
        tracker.on_bar()
        entry_fill = FillEvent(
            timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            order_id="order_1",
            trade_id="fill_1",
            asset_id="BTC",
            side=OrderSide.BUY,
            fill_quantity=1.0,
            fill_price=50000.0,
        )
        tracker.on_fill(entry_fill)  # No market event

        # Exit without market event
        tracker.on_bar()
        exit_fill = FillEvent(
            timestamp=datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
            order_id="order_2",
            trade_id="fill_2",
            asset_id="BTC",
            side=OrderSide.SELL,
            fill_quantity=1.0,
            fill_price=51000.0,
        )
        trades = tracker.on_fill(exit_fill)  # No market event

        assert len(trades) == 1
        trade = trades[0]

        # Convert to MLTradeRecord
        ml_trade = tracker.to_ml_trade_record(trade)

        # All ML/risk fields should be None
        assert ml_trade.ml_score_entry is None
        assert ml_trade.ml_score_exit is None
        assert ml_trade.atr_entry is None
        assert ml_trade.atr_exit is None
        assert ml_trade.vix_entry is None
        assert ml_trade.vix_exit is None
        assert ml_trade.exit_reason == ExitReason.SIGNAL  # Default

    def test_get_ml_trades_bulk_conversion(self):
        """Test bulk conversion of trades to MLTradeRecords."""
        tracker = TradeTracker()

        # Create multiple trades with ML signals
        for i in range(3):
            entry_event = MarketEvent(
                timestamp=datetime(2024, 1, 1, 10 + i, 0, tzinfo=timezone.utc),
                asset_id=f"ASSET_{i}",
                data_type=MarketDataType.BAR,
                close=50000.0 + i * 1000,
                signals={"ml_score": 0.8 + i * 0.05},
            )

            tracker.on_bar()
            entry_fill = FillEvent(
                datetime(2024, 1, 1, 10 + i, 0, tzinfo=timezone.utc),
                f"order_{i}",
                f"fill_{i}",
                f"ASSET_{i}",
                OrderSide.BUY,
                1.0,
                50000.0 + i * 1000,
            )
            tracker.on_fill(entry_fill, entry_event)

            tracker.on_bar()
            exit_fill = FillEvent(
                datetime(2024, 1, 1, 11 + i, 0, tzinfo=timezone.utc),
                f"order_exit_{i}",
                f"fill_exit_{i}",
                f"ASSET_{i}",
                OrderSide.SELL,
                1.0,
                51000.0 + i * 1000,
            )
            tracker.on_fill(exit_fill)

        # Get all ML trades
        ml_trades = tracker.get_ml_trades()

        assert len(ml_trades) == 3
        for i, ml_trade in enumerate(ml_trades):
            assert ml_trade.asset_id == f"ASSET_{i}"
            assert ml_trade.ml_score_entry == 0.8 + i * 0.05
            assert ml_trade.exit_reason == ExitReason.SIGNAL

    def test_partial_ml_signals(self):
        """Test handling of partial ML signal availability."""
        tracker = TradeTracker()

        # Entry with only some ML signals
        entry_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            close=50000.0,
            signals={
                "ml_score": 0.85,
                # predicted_return missing
                "atr": 1200.0,
                # volatility missing
            },
            context={
                "vix": 15.0,
                # market_regime missing
            }
        )

        tracker.on_bar()
        entry_fill = FillEvent(
            datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            "order_1",
            "fill_1",
            "BTC",
            OrderSide.BUY,
            1.0,
            50000.0,
        )
        tracker.on_fill(entry_fill, entry_event)

        tracker.on_bar()
        exit_fill = FillEvent(
            datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
            "order_2",
            "fill_2",
            "BTC",
            OrderSide.SELL,
            1.0,
            51000.0,
        )
        trades = tracker.on_fill(exit_fill)  # No exit event

        ml_trade = tracker.to_ml_trade_record(trades[0])

        # Available fields should be populated
        assert ml_trade.ml_score_entry == 0.85
        assert ml_trade.atr_entry == 1200.0
        assert ml_trade.vix_entry == 15.0

        # Missing fields should be None
        assert ml_trade.predicted_return_entry is None
        assert ml_trade.volatility_entry is None
        assert ml_trade.market_regime_entry is None
        assert ml_trade.ml_score_exit is None
