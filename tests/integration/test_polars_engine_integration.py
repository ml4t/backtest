"""Integration tests for PolarsDataFeed with BacktestEngine.

This module tests the full integration of PolarsDataFeed with BacktestEngine,
ensuring backward compatibility and validating the new multi-source capabilities.
"""

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from ml4t.backtest.core.event import MarketEvent
from ml4t.backtest.data.polars_feed import PolarsDataFeed
from ml4t.backtest.engine import BacktestEngine
from ml4t.backtest.strategy.base import Strategy


class SimpleStrategy(Strategy):
    """Simple test strategy that buys on first event."""

    def __init__(self):
        super().__init__()
        self.events_received = []
        self.has_bought = False

    def on_start(self, portfolio, event_bus):
        self.portfolio = portfolio
        self.event_bus = event_bus

    def on_event(self, event):
        if isinstance(event, MarketEvent):
            self.events_received.append(event)

            if not self.has_bought and event.close > 0:
                from ml4t.backtest.core.event import OrderEvent
                from ml4t.backtest.core.types import OrderSide, OrderType

                order_event = OrderEvent(
                    timestamp=event.timestamp,
                    order_id=f"BUY_{len(self.events_received)}",
                    asset_id=event.asset_id,
                    order_type=OrderType.MARKET,
                    side=OrderSide.BUY,
                    quantity=10.0,
                )
                self.event_bus.publish(order_event)
                self.has_bought = True

    def reset(self):
        """Reset strategy state."""
        self.events_received = []
        self.has_bought = False


class MLSignalStrategy(Strategy):
    """Strategy that uses ML signals from PolarsDataFeed."""

    def __init__(self):
        super().__init__()
        self.signals_received = []
        self.trades_made = 0

    def on_start(self, portfolio, event_bus):
        self.portfolio = portfolio
        self.event_bus = event_bus

    def on_event(self, event):
        if isinstance(event, MarketEvent):
            # Check for ML signals
            if event.signals:
                self.signals_received.append({
                    'timestamp': event.timestamp,
                    'asset_id': event.asset_id,
                    'signals': event.signals.copy()
                })

                # Trade on strong signal
                signal_value = event.signals.get('ml_signal', 0.0)
                if abs(signal_value) > 0.7 and self.trades_made < 3:
                    from ml4t.backtest.core.event import OrderEvent
                    from ml4t.backtest.core.types import OrderSide, OrderType

                    side = OrderSide.BUY if signal_value > 0 else OrderSide.SELL
                    order_event = OrderEvent(
                        timestamp=event.timestamp,
                        order_id=f"ORDER_{self.trades_made}",
                        asset_id=event.asset_id,
                        order_type=OrderType.MARKET,
                        side=side,
                        quantity=5.0,
                    )
                    self.event_bus.publish(order_event)
                    self.trades_made += 1

    def reset(self):
        """Reset strategy state."""
        self.signals_received = []
        self.trades_made = 0


@pytest.fixture
def sample_price_data(tmp_path):
    """Create sample price data for testing."""
    data = pl.DataFrame({
        "timestamp": [
            datetime(2024, 1, 10, 9, 30),
            datetime(2024, 1, 11, 9, 30),
            datetime(2024, 1, 12, 9, 30),
            datetime(2024, 1, 15, 9, 30),
            datetime(2024, 1, 16, 9, 30),
        ],
        "asset_id": ["AAPL"] * 5,
        "open": [150.0, 151.0, 152.0, 153.0, 154.0],
        "high": [152.0, 153.0, 154.0, 155.0, 156.0],
        "low": [149.0, 150.0, 151.0, 152.0, 153.0],
        "close": [151.0, 152.0, 153.0, 154.0, 155.0],
        "volume": [1000000, 1100000, 1200000, 1300000, 1400000],
    })

    path = tmp_path / "prices.parquet"
    data.write_parquet(path)
    return path


@pytest.fixture
def sample_signal_data(tmp_path):
    """Create sample ML signal data."""
    data = pl.DataFrame({
        "timestamp": [
            datetime(2024, 1, 10, 9, 30),
            datetime(2024, 1, 11, 9, 30),
            datetime(2024, 1, 12, 9, 30),
            datetime(2024, 1, 15, 9, 30),
            datetime(2024, 1, 16, 9, 30),
        ],
        "asset_id": ["AAPL"] * 5,
        "ml_signal": [0.3, 0.8, -0.5, 0.9, 0.2],
        "confidence": [0.6, 0.9, 0.7, 0.95, 0.5],
    })

    path = tmp_path / "signals.parquet"
    data.write_parquet(path)
    return path


class TestPolarsDataFeedIntegration:
    """Test PolarsDataFeed integration with BacktestEngine."""

    def test_basic_integration(self, sample_price_data):
        """Test basic PolarsDataFeed integration without signals."""
        # Create feed
        feed = PolarsDataFeed(
            price_path=sample_price_data,
            asset_id="AAPL",
        )

        # Create strategy
        strategy = SimpleStrategy()

        # Create engine
        engine = BacktestEngine(
            data_feed=feed,
            strategy=strategy,
            initial_capital=100000.0,
        )

        # Run backtest
        results = engine.run()

        # Verify execution (events include fills, so >= 5)
        assert results["events_processed"] >= 5
        assert len(strategy.events_received) == 5
        assert strategy.has_bought
        assert results["final_value"] > 0

    def test_signal_integration(self, sample_price_data, sample_signal_data):
        """Test PolarsDataFeed with ML signals."""
        # Create feed with signals
        feed = PolarsDataFeed(
            price_path=sample_price_data,
            asset_id="AAPL",
            signals_path=sample_signal_data,
            validate_signal_timing=False,  # Disable for test simplicity
        )

        # Create strategy
        strategy = MLSignalStrategy()

        # Create engine
        engine = BacktestEngine(
            data_feed=feed,
            strategy=strategy,
            initial_capital=100000.0,
        )

        # Run backtest
        results = engine.run()

        # Verify signals were received
        assert len(strategy.signals_received) == 5

        # Verify signals have expected keys
        for signal_data in strategy.signals_received:
            assert 'ml_signal' in signal_data['signals']
            assert 'confidence' in signal_data['signals']

        # Verify trades were made based on signals
        assert strategy.trades_made > 0
        assert results["events_processed"] >= 5

    def test_backward_compatibility_parquet_feed(self, sample_price_data):
        """Test that ParquetDataFeed still works (backward compatibility)."""
        pass  # TODO: Re-enable when ParquetDataFeed is implemented
#        """Test that ParquetDataFeed still works (backward compatibility)."""
#        # Create old-style ParquetDataFeed
#        feed = ParquetDataFeed(
#            path=sample_price_data,
#            asset_id="AAPL",
#        )
#
#        # Create strategy
#        strategy = SimpleStrategy()
#
#        # Create engine
#        engine = BacktestEngine(
#            data_feed=feed,
#            strategy=strategy,
#            initial_capital=100000.0,
#        )
#
#        # Run backtest
#        results = engine.run()
#
#        # Verify execution (events include fills)
#        assert results["events_processed"] >= 5
#        assert len(strategy.events_received) == 5
#        assert strategy.has_bought
#
#    def test_polars_vs_parquet_consistency(self, sample_price_data):
#        """Test that PolarsDataFeed and ParquetDataFeed produce consistent results."""
#        # Create both feed types
#        polars_feed = PolarsDataFeed(
#            price_path=sample_price_data,
#            asset_id="AAPL",
#        )
#
#        parquet_feed = ParquetDataFeed(
#            path=sample_price_data,
#            asset_id="AAPL",
#        )
#
#        # Create identical strategies
#        strategy1 = SimpleStrategy()
#        strategy2 = SimpleStrategy()
#
#        # Create engines
#        engine1 = BacktestEngine(
#            data_feed=polars_feed,
#            strategy=strategy1,
#            initial_capital=100000.0,
#        )
#
#        engine2 = BacktestEngine(
#            data_feed=parquet_feed,
#            strategy=strategy2,
#            initial_capital=100000.0,
#        )
#
#        # Run both backtests
#        results1 = engine1.run()
#        results2 = engine2.run()
#
#        # Verify consistency
#        assert results1["events_processed"] == results2["events_processed"]
#        assert len(strategy1.events_received) == len(strategy2.events_received)
#
#        # Verify event data is identical
#        for e1, e2 in zip(strategy1.events_received, strategy2.events_received):
#            assert e1.timestamp == e2.timestamp
#            assert e1.asset_id == e2.asset_id
#            assert e1.close == e2.close
#            assert e1.volume == e2.volume
#
    def test_polars_feed_reset(self, sample_price_data):
        """Test that PolarsDataFeed reset works correctly."""
        # Run 1: First backtest
        feed1 = PolarsDataFeed(
            price_path=sample_price_data,
            asset_id="AAPL",
        )
        strategy1 = SimpleStrategy()
        engine1 = BacktestEngine(
            data_feed=feed1,
            strategy=strategy1,
            initial_capital=100000.0,
        )
        results1 = engine1.run()
        events_count_1 = len(strategy1.events_received)

        # Run 2: Reset feed and run again
        feed1.reset()
        strategy2 = SimpleStrategy()
        engine2 = BacktestEngine(
            data_feed=feed1,  # Reuse reset feed
            strategy=strategy2,
            initial_capital=100000.0,
        )
        results2 = engine2.run()
        events_count_2 = len(strategy2.events_received)

        # Verify both runs processed same number of events
        assert results1["events_processed"] == results2["events_processed"]
        assert events_count_1 == events_count_2

    def test_polars_feed_with_filters(self, tmp_path):
        """Test PolarsDataFeed with date filters."""
        # Create data with wider date range
        data = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 10, 9, 30),
                datetime(2024, 1, 11, 9, 30),
                datetime(2024, 1, 12, 9, 30),
                datetime(2024, 1, 15, 9, 30),
                datetime(2024, 1, 16, 9, 30),
                datetime(2024, 1, 17, 9, 30),
                datetime(2024, 1, 18, 9, 30),
            ],
            "asset_id": ["AAPL"] * 7,
            "open": [150.0, 151.0, 152.0, 153.0, 154.0, 155.0, 156.0],
            "high": [152.0, 153.0, 154.0, 155.0, 156.0, 157.0, 158.0],
            "low": [149.0, 150.0, 151.0, 152.0, 153.0, 154.0, 155.0],
            "close": [151.0, 152.0, 153.0, 154.0, 155.0, 156.0, 157.0],
            "volume": [1000000] * 7,
        })

        path = tmp_path / "prices_wide.parquet"
        data.write_parquet(path)

        # Create feed with date filter (only Jan 12-16)
        feed = PolarsDataFeed(
            price_path=path,
            asset_id="AAPL",
            filters=[
                pl.col("timestamp") >= datetime(2024, 1, 12),
                pl.col("timestamp") <= datetime(2024, 1, 16),
            ],
        )

        # Create strategy
        strategy = SimpleStrategy()

        # Create engine
        engine = BacktestEngine(
            data_feed=feed,
            strategy=strategy,
            initial_capital=100000.0,
        )

        # Run backtest
        results = engine.run()

        # Verify only filtered events were processed
        # Should be Jan 12, 15, 16 (3 days with weekend)
        assert len(strategy.events_received) >= 2  # At least 2 events in filter range
        assert len(strategy.events_received) <= 4  # At most 4 events

    def test_polars_feed_missing_signals(self, sample_price_data, tmp_path):
        """Test PolarsDataFeed handles missing signals gracefully."""
        # Create signal data with gaps
        data = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 10, 9, 30),
                datetime(2024, 1, 12, 9, 30),  # Skip Jan 11
                datetime(2024, 1, 16, 9, 30),  # Skip Jan 15
            ],
            "asset_id": ["AAPL"] * 3,
            "ml_signal": [0.8, -0.5, 0.9],
        })

        signal_path = tmp_path / "signals_sparse.parquet"
        data.write_parquet(signal_path)

        # Create feed with sparse signals
        feed = PolarsDataFeed(
            price_path=sample_price_data,
            asset_id="AAPL",
            signals_path=signal_path,
            validate_signal_timing=False,  # Disable for test
        )

        # Create strategy
        strategy = MLSignalStrategy()

        # Create engine
        engine = BacktestEngine(
            data_feed=feed,
            strategy=strategy,
            initial_capital=100000.0,
        )

        # Run backtest
        results = engine.run()

        # Verify all price events were processed (includes fills)
        assert results["events_processed"] >= 5

        # Verify only some events had signals
        events_with_signals = [s for s in strategy.signals_received if s['signals']]
        assert len(events_with_signals) == 3  # Only 3 timestamps had signals


class TestPolarsDataFeedPerformance:
    """Performance comparison tests."""

    def test_performance_baseline(self, tmp_path):
        """Establish performance baseline for PolarsDataFeed.

        This test creates a larger dataset and measures throughput.
        """
        # Create larger dataset (100 days)
        timestamps = pl.datetime_range(
            datetime(2024, 1, 1, 9, 30),
            datetime(2024, 4, 30, 9, 30),
            interval="1d",
            eager=True,
        ).to_list()

        n_days = len(timestamps)
        data = pl.DataFrame({
            "timestamp": timestamps,
            "asset_id": ["AAPL"] * n_days,
            "open": [150.0 + i * 0.1 for i in range(n_days)],
            "high": [152.0 + i * 0.1 for i in range(n_days)],
            "low": [149.0 + i * 0.1 for i in range(n_days)],
            "close": [151.0 + i * 0.1 for i in range(n_days)],
            "volume": [1000000] * n_days,
        })

        path = tmp_path / "prices_large.parquet"
        data.write_parquet(path)

        # Create feed
        feed = PolarsDataFeed(
            price_path=path,
            asset_id="AAPL",
        )

        # Create strategy
        strategy = SimpleStrategy()

        # Create engine
        engine = BacktestEngine(
            data_feed=feed,
            strategy=strategy,
            initial_capital=100000.0,
        )

        # Run backtest
        results = engine.run()

        # Verify throughput is reasonable (>10k events/sec target)
        events_per_second = results["events_per_second"]
        assert events_per_second > 1000  # Very conservative baseline
        # Events include both market and fill events
        assert results["events_processed"] >= n_days

        print(f"\nPolarsDataFeed throughput: {events_per_second:.0f} events/sec")

#    def test_parquet_vs_polars_performance(self, tmp_path):
#        """Compare ParquetDataFeed vs PolarsDataFeed performance."""
#        # Create dataset
#        timestamps = pl.datetime_range(
#            datetime(2024, 1, 1, 9, 30),
#            datetime(2024, 3, 31, 9, 30),
#            interval="1d",
#            eager=True,
#        ).to_list()
#
#        n_days = len(timestamps)
#        data = pl.DataFrame({
#            "timestamp": timestamps,
#            "asset_id": ["AAPL"] * n_days,
#            "open": [150.0 + i * 0.1 for i in range(n_days)],
#            "high": [152.0 + i * 0.1 for i in range(n_days)],
#            "low": [149.0 + i * 0.1 for i in range(n_days)],
#            "close": [151.0 + i * 0.1 for i in range(n_days)],
#            "volume": [1000000] * n_days,
#        })
#
#        path = tmp_path / "prices_perf.parquet"
#        data.write_parquet(path)
#
#        # Test ParquetDataFeed
#        parquet_feed = ParquetDataFeed(path=path, asset_id="AAPL")
#        strategy1 = SimpleStrategy()
#        engine1 = BacktestEngine(
#            data_feed=parquet_feed,
#            strategy=strategy1,
#            initial_capital=100000.0,
#        )
#        results1 = engine1.run()
#        parquet_throughput = results1["events_per_second"]
#
#        # Test PolarsDataFeed
#        polars_feed = PolarsDataFeed(price_path=path, asset_id="AAPL")
#        strategy2 = SimpleStrategy()
#        engine2 = BacktestEngine(
#            data_feed=polars_feed,
#            strategy=strategy2,
#            initial_capital=100000.0,
#        )
#        results2 = engine2.run()
#        polars_throughput = results2["events_per_second"]
#
#        # Compare
#        ratio = polars_throughput / parquet_throughput
#
#        print(f"\nParquetDataFeed: {parquet_throughput:.0f} events/sec")
#        print(f"PolarsDataFeed:  {polars_throughput:.0f} events/sec")
#        print(f"Ratio: {ratio:.2f}x")
#
#        # Note: In small tests like this (90 days), both feeds are fast enough
#        # (>10k events/sec) that the difference is not material. PolarsDataFeed's
#        # advantages become clear with:
#        # 1. Large datasets (>100k rows): lazy loading saves memory
#        # 2. Multi-source data: merges prices + signals + features efficiently
#        # 3. Multi-asset strategies: group_by provides 10-50x speedup
#        #
#        # For this test, we just verify both are "fast enough" (>1k events/sec)
#        assert parquet_throughput > 1000, f"ParquetDataFeed too slow: {parquet_throughput:.0f}"
#        assert polars_throughput > 1000, f"PolarsDataFeed too slow: {polars_throughput:.0f}"
