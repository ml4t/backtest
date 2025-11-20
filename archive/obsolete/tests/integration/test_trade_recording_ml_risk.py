"""Integration test for ML and risk-attributed trade recording.

Tests end-to-end workflow of:
1. ML strategy generating signals
2. Risk manager enforcing rules
3. Trade tracker capturing full ML/risk context
4. Parquet export/import workflow
"""

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from ml4t.backtest.core.event import FillEvent, MarketEvent
from ml4t.backtest.core.types import MarketDataType, OrderSide
from ml4t.backtest.execution.trade_tracker import TradeTracker
from ml4t.backtest.reporting.trade_schema import (
    ExitReason,
    export_parquet,
    import_parquet,
    polars_to_trades,
)


class TestTradeRecordingMLRiskIntegration:
    """Integration tests for ML/risk trade recording workflow."""

    def test_end_to_end_ml_strategy_with_risk_rules(self, tmp_path):
        """Test complete workflow: ML strategy → risk rules → trade recording → export.

        Simulates:
        - ML strategy generating entry/exit signals with predictions
        - Risk manager setting stop-loss/take-profit levels
        - Multiple exits for different reasons (signal, stop-loss, take-profit)
        - Export to Parquet and re-import
        """
        tracker = TradeTracker()

        # Trade 1: Normal signal-based exit
        entry1_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000,
            signals={
                "ml_score": 0.82,
                "predicted_return": 0.025,
                "confidence": 0.88,
                "atr": 2.5,
                "volatility": 0.018,
                "momentum": 0.04,
                "rsi": 62.0,
            },
            context={
                "vix": 14.5,
                "market_regime": "bull",
                "sector_performance": 0.015,
            }
        )

        tracker.on_bar()
        entry1_fill = FillEvent(
            timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            order_id="order_1",
            trade_id="fill_1",
            asset_id="AAPL",
            side=OrderSide.BUY,
            fill_quantity=100.0,
            fill_price=151.0,
            commission=1.0,
            slippage=0.05,
            metadata={
                "stop_loss_price": 145.0,
                "take_profit_price": 160.0,
                "risk_reward_ratio": 2.25,
                "position_size_pct": 5.0,
            }
        )
        tracker.on_fill(entry1_fill, entry1_event)

        # Exit: ML signal says exit (score dropped)
        exit1_event = MarketEvent(
            timestamp=datetime(2024, 1, 2, 14, 0, tzinfo=timezone.utc),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            close=155.0,
            signals={
                "ml_score": 0.35,  # Score dropped
                "predicted_return": -0.01,
                "confidence": 0.65,
                "atr": 2.8,
                "volatility": 0.022,
                "momentum": -0.01,
                "rsi": 68.0,
            },
            context={
                "vix": 16.0,
                "market_regime": "neutral",
                "sector_performance": 0.008,
            }
        )

        tracker.on_bar()
        tracker.on_bar()
        exit1_fill = FillEvent(
            timestamp=datetime(2024, 1, 2, 14, 0, tzinfo=timezone.utc),
            order_id="order_2",
            trade_id="fill_2",
            asset_id="AAPL",
            side=OrderSide.SELL,
            fill_quantity=100.0,
            fill_price=155.0,
            commission=1.0,
            slippage=0.05,
            metadata={"exit_reason": "signal"}
        )
        trades1 = tracker.on_fill(exit1_fill, exit1_event)
        assert len(trades1) == 1

        # Trade 2: Stop-loss triggered
        entry2_event = MarketEvent(
            timestamp=datetime(2024, 1, 3, 10, 0, tzinfo=timezone.utc),
            asset_id="MSFT",
            data_type=MarketDataType.BAR,
            close=380.0,
            signals={
                "ml_score": 0.75,
                "predicted_return": 0.02,
                "confidence": 0.85,
                "atr": 5.0,
            },
            context={"vix": 15.0}
        )

        tracker.on_bar()
        entry2_fill = FillEvent(
            timestamp=datetime(2024, 1, 3, 10, 0, tzinfo=timezone.utc),
            order_id="order_3",
            trade_id="fill_3",
            asset_id="MSFT",
            side=OrderSide.BUY,
            fill_quantity=50.0,
            fill_price=380.0,
            metadata={"stop_loss_price": 370.0, "position_size_pct": 3.0}
        )
        tracker.on_fill(entry2_fill, entry2_event)

        # Exit: Stop-loss hit
        exit2_event = MarketEvent(
            timestamp=datetime(2024, 1, 4, 11, 0, tzinfo=timezone.utc),
            asset_id="MSFT",
            data_type=MarketDataType.BAR,
            close=369.0,
            signals={"ml_score": 0.55, "atr": 5.5},
            context={"vix": 22.0}  # VIX spiked
        )

        tracker.on_bar()
        exit2_fill = FillEvent(
            timestamp=datetime(2024, 1, 4, 11, 0, tzinfo=timezone.utc),
            order_id="order_4",
            trade_id="fill_4",
            asset_id="MSFT",
            side=OrderSide.SELL,
            fill_quantity=50.0,
            fill_price=369.0,
            metadata={"exit_reason": "stop_loss"}
        )
        trades2 = tracker.on_fill(exit2_fill, exit2_event)
        assert len(trades2) == 1

        # Trade 3: Take-profit hit
        entry3_event = MarketEvent(
            timestamp=datetime(2024, 1, 5, 10, 0, tzinfo=timezone.utc),
            asset_id="GOOGL",
            data_type=MarketDataType.BAR,
            close=140.0,
            signals={"ml_score": 0.90, "predicted_return": 0.05},
        )

        tracker.on_bar()
        entry3_fill = FillEvent(
            timestamp=datetime(2024, 1, 5, 10, 0, tzinfo=timezone.utc),
            order_id="order_5",
            trade_id="fill_5",
            asset_id="GOOGL",
            side=OrderSide.BUY,
            fill_quantity=75.0,
            fill_price=140.0,
            metadata={"take_profit_price": 147.0}
        )
        tracker.on_fill(entry3_fill, entry3_event)

        # Exit: Take-profit hit
        exit3_event = MarketEvent(
            timestamp=datetime(2024, 1, 6, 13, 0, tzinfo=timezone.utc),
            asset_id="GOOGL",
            data_type=MarketDataType.BAR,
            close=147.5,
            signals={"ml_score": 0.92},  # Still high
        )

        tracker.on_bar()
        tracker.on_bar()
        exit3_fill = FillEvent(
            timestamp=datetime(2024, 1, 6, 13, 0, tzinfo=timezone.utc),
            order_id="order_6",
            trade_id="fill_6",
            asset_id="GOOGL",
            side=OrderSide.SELL,
            fill_quantity=75.0,
            fill_price=147.5,
            metadata={"exit_reason": "take_profit"}
        )
        trades3 = tracker.on_fill(exit3_fill, exit3_event)
        assert len(trades3) == 1

        # Convert all trades to MLTradeRecords
        ml_trades = tracker.get_ml_trades()
        assert len(ml_trades) == 3

        # Verify Trade 1: Signal exit
        trade1 = ml_trades[0]
        assert trade1.asset_id == "AAPL"
        assert trade1.exit_reason == ExitReason.SIGNAL
        assert trade1.ml_score_entry == 0.82
        assert trade1.ml_score_exit == 0.35
        assert trade1.predicted_return_entry == 0.025
        assert trade1.atr_entry == 2.5
        assert trade1.atr_exit == 2.8
        assert trade1.vix_entry == 14.5
        assert trade1.vix_exit == 16.0
        assert trade1.market_regime_entry == "bull"
        assert trade1.market_regime_exit == "neutral"
        assert trade1.stop_loss_price == 145.0
        assert trade1.take_profit_price == 160.0
        assert trade1.risk_reward_ratio == 2.25
        assert trade1.position_size_pct == 5.0
        assert trade1.duration_bars == 2
        assert trade1.pnl > 0  # Profitable

        # Verify Trade 2: Stop-loss exit
        trade2 = ml_trades[1]
        assert trade2.asset_id == "MSFT"
        assert trade2.exit_reason == ExitReason.STOP_LOSS
        assert trade2.stop_loss_price == 370.0
        assert trade2.vix_entry == 15.0
        assert trade2.vix_exit == 22.0  # VIX spiked
        assert trade2.pnl < 0  # Loss (stop-loss hit)

        # Verify Trade 3: Take-profit exit
        trade3 = ml_trades[2]
        assert trade3.asset_id == "GOOGL"
        assert trade3.exit_reason == ExitReason.TAKE_PROFIT
        assert trade3.take_profit_price == 147.0
        assert trade3.ml_score_entry == 0.90
        assert trade3.ml_score_exit == 0.92  # Still bullish at exit
        assert trade3.pnl > 0  # Profitable (take-profit hit)

        # Export to Parquet
        parquet_path = tmp_path / "ml_trades.parquet"
        export_parquet(ml_trades, parquet_path)

        # Verify file was created
        assert parquet_path.exists()

        # Re-import and verify
        df = import_parquet(parquet_path)
        assert len(df) == 3

        # Verify column schema
        expected_columns = {
            "trade_id", "asset_id", "direction",
            "entry_dt", "entry_price", "entry_quantity",
            "exit_dt", "exit_price", "exit_reason",
            "pnl", "return_pct", "duration_bars", "duration_seconds",
            "ml_score_entry", "ml_score_exit",
            "predicted_return_entry", "predicted_return_exit",
            "atr_entry", "atr_exit",
            "vix_entry", "vix_exit",
            "stop_loss_price", "take_profit_price",
        }
        assert expected_columns.issubset(set(df.columns))

        # Convert back to MLTradeRecord
        reimported_trades = polars_to_trades(df)
        assert len(reimported_trades) == 3

        # Verify data integrity
        assert reimported_trades[0].asset_id == "AAPL"
        assert reimported_trades[0].ml_score_entry == 0.82
        assert reimported_trades[1].asset_id == "MSFT"
        assert reimported_trades[1].exit_reason == ExitReason.STOP_LOSS
        assert reimported_trades[2].asset_id == "GOOGL"
        assert reimported_trades[2].exit_reason == ExitReason.TAKE_PROFIT

    def test_parquet_roundtrip_with_nulls(self, tmp_path):
        """Test Parquet export/import handles None values correctly."""
        tracker = TradeTracker()

        # Create trade with minimal data (lots of None fields)
        tracker.on_bar()
        entry_fill = FillEvent(
            timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            order_id="order_1",
            trade_id="fill_1",
            asset_id="TEST",
            side=OrderSide.BUY,
            fill_quantity=100.0,
            fill_price=100.0,
        )
        tracker.on_fill(entry_fill)  # No market event

        tracker.on_bar()
        exit_fill = FillEvent(
            timestamp=datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
            order_id="order_2",
            trade_id="fill_2",
            asset_id="TEST",
            side=OrderSide.SELL,
            fill_quantity=100.0,
            fill_price=105.0,
        )
        tracker.on_fill(exit_fill)  # No market event

        ml_trades = tracker.get_ml_trades()
        assert len(ml_trades) == 1

        # Export and re-import
        parquet_path = tmp_path / "sparse_trades.parquet"
        export_parquet(ml_trades, parquet_path)

        df = import_parquet(parquet_path)
        reimported = polars_to_trades(df)

        # Verify None fields are preserved
        assert reimported[0].ml_score_entry is None
        assert reimported[0].atr_entry is None
        assert reimported[0].vix_entry is None
        assert reimported[0].stop_loss_price is None

    def test_trade_analysis_workflow(self):
        """Test typical post-backtest analysis workflow on ML trades.

        Demonstrates querying trades by:
        - Exit reason
        - ML score ranges
        - Market regime
        - Risk metrics
        """
        tracker = TradeTracker()

        # Create diverse set of trades
        test_cases = [
            {
                "asset": "A",
                "ml_score": 0.9,
                "regime": "bull",
                "exit_reason": "signal",
                "entry_price": 100.0,
                "exit_price": 110.0,
            },
            {
                "asset": "B",
                "ml_score": 0.5,
                "regime": "bear",
                "exit_reason": "stop_loss",
                "entry_price": 100.0,
                "exit_price": 95.0,
            },
            {
                "asset": "C",
                "ml_score": 0.85,
                "regime": "bull",
                "exit_reason": "take_profit",
                "entry_price": 100.0,
                "exit_price": 115.0,
            },
        ]

        for i, tc in enumerate(test_cases):
            entry_event = MarketEvent(
                timestamp=datetime(2024, 1, 1, 10 + i, 0, tzinfo=timezone.utc),
                asset_id=tc["asset"],
                data_type=MarketDataType.BAR,
                close=tc["entry_price"],
                signals={"ml_score": tc["ml_score"]},
                context={"market_regime": tc["regime"]}
            )

            tracker.on_bar()
            entry_fill = FillEvent(
                datetime(2024, 1, 1, 10 + i, 0, tzinfo=timezone.utc),
                f"order_{i}",
                f"fill_{i}",
                tc["asset"],
                OrderSide.BUY,
                100.0,
                tc["entry_price"],
            )
            tracker.on_fill(entry_fill, entry_event)

            tracker.on_bar()
            exit_fill = FillEvent(
                datetime(2024, 1, 1, 11 + i, 0, tzinfo=timezone.utc),
                f"order_exit_{i}",
                f"fill_exit_{i}",
                tc["asset"],
                OrderSide.SELL,
                100.0,
                tc["exit_price"],
                metadata={"exit_reason": tc["exit_reason"]}
            )
            tracker.on_fill(exit_fill)

        # Get ML trades as Polars DataFrame
        ml_trades = tracker.get_ml_trades()
        from ml4t.backtest.reporting.trade_schema import trades_to_polars
        df = trades_to_polars(ml_trades)

        # Analysis 1: Wins vs losses by exit reason
        wins = df.filter(pl.col("pnl") > 0)
        losses = df.filter(pl.col("pnl") <= 0)
        assert len(wins) == 2  # signal, take_profit
        assert len(losses) == 1  # stop_loss

        # Analysis 2: Filter by ML score threshold
        high_confidence = df.filter(pl.col("ml_score_entry") >= 0.8)
        assert len(high_confidence) == 2  # scores 0.9 and 0.85

        # Analysis 3: Filter by market regime
        bull_trades = df.filter(pl.col("market_regime_entry") == "bull")
        assert len(bull_trades) == 2

        # Analysis 4: Exit reason breakdown
        exit_reasons = df.group_by("exit_reason").agg(pl.count().alias("count"))
        assert len(exit_reasons) == 3  # signal, stop_loss, take_profit
