"""Unit tests for Strategy API execution modes (simple and batch).

Tests:
- Mode detection (simple vs batch)
- Simple mode execution (on_market_event)
- Batch mode execution (on_timestamp_batch)
- Backward compatibility
- Helper method integration
- Context passing
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

from ml4t.backtest.core.event import MarketEvent
from ml4t.backtest.core.types import MarketDataType, OrderSide
from ml4t.backtest.execution.broker import SimulationBroker
from ml4t.backtest.strategy.base import Strategy


class SimpleTestStrategy(Strategy):
    """Test strategy using Simple Mode (on_market_event)."""

    def __init__(self):
        super().__init__()
        self.events_received = []
        self.contexts_received = []

    def on_event(self, event):
        """Required abstract method."""
        pass

    def on_market_event(self, event, context=None):
        """Override for simple mode."""
        self.events_received.append(event)
        self.contexts_received.append(context)


class BatchTestStrategy(Strategy):
    """Test strategy using Batch Mode (on_timestamp_batch)."""

    def __init__(self):
        super().__init__()
        self.batches_received = []
        self.timestamps_received = []
        self.contexts_received = []

    def on_event(self, event):
        """Required abstract method."""
        pass

    def on_timestamp_batch(self, timestamp, events, context=None):
        """Override for batch mode."""
        self.timestamps_received.append(timestamp)
        self.batches_received.append(events)
        self.contexts_received.append(context)


class TestModeDetection:
    """Test automatic execution mode detection."""

    def test_simple_mode_detected_by_default(self):
        """Test that simple mode is default when only on_market_event is overridden."""
        strategy = SimpleTestStrategy()
        assert strategy.execution_mode == "simple"

    def test_batch_mode_detected_when_on_timestamp_batch_overridden(self):
        """Test that batch mode is detected when on_timestamp_batch is overridden."""
        strategy = BatchTestStrategy()
        assert strategy.execution_mode == "batch"

    def test_mode_detection_via_property(self):
        """Test that execution_mode property works."""
        simple_strategy = SimpleTestStrategy()
        batch_strategy = BatchTestStrategy()

        assert simple_strategy.execution_mode == "simple"
        assert batch_strategy.execution_mode == "batch"

    def test_base_strategy_is_simple_mode(self):
        """Test that base Strategy class defaults to simple mode."""

        class MinimalStrategy(Strategy):
            def on_event(self, event):
                pass

        strategy = MinimalStrategy()
        assert strategy.execution_mode == "simple"


class TestSimpleModeExecution:
    """Test Simple Mode (on_market_event) execution."""

    @pytest.fixture
    def simple_strategy(self):
        """Create simple mode strategy."""
        return SimpleTestStrategy()

    @pytest.fixture
    def market_event(self):
        """Create sample market event."""
        return MarketEvent(
            timestamp=datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000,
            signals={"ml_score": 0.85, "sma_10": 150.5, "sma_50": 148.2},
            context={"VIX": 18.5},
        )

    def test_simple_mode_receives_single_events(self, simple_strategy, market_event):
        """Test that simple mode receives events one at a time."""
        context = {"VIX": 18.5, "SPY": 485.0}

        # Call strategy directly (simulating engine)
        simple_strategy.on_market_event(market_event, context)

        assert len(simple_strategy.events_received) == 1
        assert simple_strategy.events_received[0] == market_event
        assert simple_strategy.contexts_received[0] == context

    def test_simple_mode_receives_multiple_events_separately(
        self, simple_strategy, market_event
    ):
        """Test that simple mode processes each event independently."""
        event1 = market_event
        event2 = MarketEvent(
            timestamp=datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc),
            asset_id="GOOGL",
            data_type=MarketDataType.BAR,
            open=2800.0,
            high=2820.0,
            low=2790.0,
            close=2810.0,
            volume=500000,
        )

        context = {"VIX": 18.5}

        # Process events separately
        simple_strategy.on_market_event(event1, context)
        simple_strategy.on_market_event(event2, context)

        assert len(simple_strategy.events_received) == 2
        assert simple_strategy.events_received[0].asset_id == "AAPL"
        assert simple_strategy.events_received[1].asset_id == "GOOGL"

    def test_simple_mode_can_access_event_data(self, simple_strategy, market_event):
        """Test that simple mode can access all event data."""
        simple_strategy.on_market_event(market_event, {"VIX": 18.5})

        received_event = simple_strategy.events_received[0]
        assert received_event.close == 151.0
        assert received_event.signals.get("ml_score") == 0.85
        assert received_event.signals.get("sma_10") == 150.5

    def test_simple_mode_works_without_context(self, simple_strategy, market_event):
        """Test that simple mode works when context is None."""
        simple_strategy.on_market_event(market_event, context=None)

        assert len(simple_strategy.events_received) == 1
        assert simple_strategy.contexts_received[0] is None


class TestBatchModeExecution:
    """Test Batch Mode (on_timestamp_batch) execution."""

    @pytest.fixture
    def batch_strategy(self):
        """Create batch mode strategy."""
        return BatchTestStrategy()

    @pytest.fixture
    def event_batch(self):
        """Create batch of events at same timestamp."""
        timestamp = datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc)
        return [
            MarketEvent(
                timestamp=timestamp,
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                open=150.0,
                high=152.0,
                low=149.0,
                close=151.0,
                volume=1000000,
                signals={"momentum": 0.05},
            ),
            MarketEvent(
                timestamp=timestamp,
                asset_id="GOOGL",
                data_type=MarketDataType.BAR,
                open=2800.0,
                high=2820.0,
                low=2790.0,
                close=2810.0,
                volume=500000,
                signals={"momentum": 0.08},
            ),
            MarketEvent(
                timestamp=timestamp,
                asset_id="MSFT",
                data_type=MarketDataType.BAR,
                open=380.0,
                high=385.0,
                low=378.0,
                close=383.0,
                volume=800000,
                signals={"momentum": 0.03},
            ),
        ]

    def test_batch_mode_receives_event_list(self, batch_strategy, event_batch):
        """Test that batch mode receives list of events."""
        timestamp = event_batch[0].timestamp
        context = {"VIX": 18.5}

        batch_strategy.on_timestamp_batch(timestamp, event_batch, context)

        assert len(batch_strategy.batches_received) == 1
        assert len(batch_strategy.batches_received[0]) == 3
        assert batch_strategy.timestamps_received[0] == timestamp
        assert batch_strategy.contexts_received[0] == context

    def test_batch_mode_can_iterate_events(self, batch_strategy, event_batch):
        """Test that batch mode can iterate over event list."""
        timestamp = event_batch[0].timestamp
        batch_strategy.on_timestamp_batch(timestamp, event_batch, None)

        received_batch = batch_strategy.batches_received[0]
        asset_ids = [event.asset_id for event in received_batch]

        assert asset_ids == ["AAPL", "GOOGL", "MSFT"]

    def test_batch_mode_can_rank_assets(self, batch_strategy, event_batch):
        """Test that batch mode can rank assets by indicator."""
        timestamp = event_batch[0].timestamp
        batch_strategy.on_timestamp_batch(timestamp, event_batch, None)

        received_batch = batch_strategy.batches_received[0]

        # Rank by momentum (now in signals dict)
        scores = {e.asset_id: e.signals.get("momentum", 0) for e in received_batch}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        assert ranked[0][0] == "GOOGL"  # Highest momentum (0.08)
        assert ranked[1][0] == "AAPL"  # Second (0.05)
        assert ranked[2][0] == "MSFT"  # Lowest (0.03)

    def test_batch_mode_receives_multiple_batches(self, batch_strategy):
        """Test that batch mode can process multiple timestamp batches."""
        # First batch
        timestamp1 = datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc)
        batch1 = [
            MarketEvent(
                timestamp=timestamp1,
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                close=150.0,
            ),
        ]

        # Second batch (different timestamp)
        timestamp2 = datetime(2024, 1, 15, 9, 31, tzinfo=timezone.utc)
        batch2 = [
            MarketEvent(
                timestamp=timestamp2,
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                close=151.0,
            ),
        ]

        batch_strategy.on_timestamp_batch(timestamp1, batch1, None)
        batch_strategy.on_timestamp_batch(timestamp2, batch2, None)

        assert len(batch_strategy.batches_received) == 2
        assert batch_strategy.timestamps_received[0] == timestamp1
        assert batch_strategy.timestamps_received[1] == timestamp2


class TestHelperMethodIntegration:
    """Test that helper methods work in both modes."""

    @pytest.fixture
    def strategy_with_broker(self):
        """Create strategy with initialized broker."""

        class TestStrategyWithTrades(Strategy):
            def on_event(self, event):
                pass

            def on_market_event(self, event, context=None):
                # Use helper method
                if event.signals.get("buy_signal", 0) > 0.7:
                    self.buy_percent(event.asset_id, 0.10, event.close)

        strategy = TestStrategyWithTrades()
        broker = SimulationBroker()
        strategy.broker = broker
        return strategy, broker

    def test_simple_mode_can_use_helpers(self, strategy_with_broker):
        """Test that simple mode can use helper methods."""
        strategy, broker = strategy_with_broker

        event = MarketEvent(
            timestamp=datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000,
            signals={"buy_signal": 0.85},
        )

        with patch.object(broker, "submit_order") as mock_submit:
            strategy.on_market_event(event, None)
            assert mock_submit.called

    def test_batch_mode_can_use_rebalance(self):
        """Test that batch mode can use rebalance_to_weights."""

        class BatchStrategyWithRebalance(Strategy):
            def on_event(self, event):
                pass

            def on_timestamp_batch(self, timestamp, events, context=None):
                # Use rebalance helper
                weights = {e.asset_id: 1.0 / len(events) for e in events}
                prices = {e.asset_id: e.close for e in events}
                self.rebalance_to_weights(weights, prices, tolerance=0.05)

        strategy = BatchStrategyWithRebalance()
        broker = SimulationBroker()
        strategy.broker = broker

        events = [
            MarketEvent(
                timestamp=datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc),
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                close=150.0,
            ),
            MarketEvent(
                timestamp=datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc),
                asset_id="GOOGL",
                data_type=MarketDataType.BAR,
                close=2800.0,
            ),
        ]

        with patch.object(broker, "submit_order") as mock_submit:
            strategy.on_timestamp_batch(events[0].timestamp, events, None)
            # Should submit rebalancing orders
            assert mock_submit.call_count >= 1


class TestBackwardCompatibility:
    """Test backward compatibility with existing strategies."""

    def test_strategy_without_on_market_event_override(self):
        """Test that strategies not overriding on_market_event still work."""

        class LegacyStrategy(Strategy):
            def on_event(self, event):
                # Old pattern: everything in on_event
                self.processed = True

        strategy = LegacyStrategy()
        assert strategy.execution_mode == "simple"  # Defaults to simple

    def test_on_event_still_called(self):
        """Test that on_event is still called for backward compatibility."""

        class EventTrackingStrategy(Strategy):
            def __init__(self):
                super().__init__()
                self.on_event_called = False

            def on_event(self, event):
                self.on_event_called = True

            def on_market_event(self, event, context=None):
                pass

        strategy = EventTrackingStrategy()
        event = MarketEvent(
            timestamp=datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            close=150.0,
        )

        # Simulating engine: calls both on_market_event and on_event
        strategy.on_market_event(event, None)
        strategy.on_event(event)

        assert strategy.on_event_called


class TestContextPassing:
    """Test context dict passing in both modes."""

    def test_simple_mode_receives_context(self):
        """Test that simple mode receives context dict."""
        strategy = SimpleTestStrategy()
        event = MarketEvent(
            timestamp=datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            close=150.0,
        )
        context = {"VIX": 18.5, "SPY": 485.0, "regime": "bull"}

        strategy.on_market_event(event, context)

        assert strategy.contexts_received[0] == context
        assert strategy.contexts_received[0]["VIX"] == 18.5

    def test_batch_mode_receives_shared_context(self):
        """Test that batch mode receives shared context for all events."""
        strategy = BatchTestStrategy()
        timestamp = datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc)
        events = [
            MarketEvent(
                timestamp=timestamp,
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                close=150.0,
            ),
            MarketEvent(
                timestamp=timestamp,
                asset_id="GOOGL",
                data_type=MarketDataType.BAR,
                close=2800.0,
            ),
        ]
        context = {"VIX": 18.5, "SPY": 485.0}

        strategy.on_timestamp_batch(timestamp, events, context)

        assert strategy.contexts_received[0] == context
        # Context is shared (same object for all events at timestamp)

    def test_context_none_handling(self):
        """Test that strategies handle None context gracefully."""
        simple_strategy = SimpleTestStrategy()
        batch_strategy = BatchTestStrategy()

        event = MarketEvent(
            timestamp=datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            close=150.0,
        )

        # Simple mode with None context
        simple_strategy.on_market_event(event, None)
        assert simple_strategy.contexts_received[0] is None

        # Batch mode with None context
        batch_strategy.on_timestamp_batch(event.timestamp, [event], None)
        assert batch_strategy.contexts_received[0] is None


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_batch_handling(self):
        """Test that batch mode handles empty event list."""
        strategy = BatchTestStrategy()
        timestamp = datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc)

        # Should not raise error
        strategy.on_timestamp_batch(timestamp, [], None)

        assert len(strategy.batches_received) == 1
        assert len(strategy.batches_received[0]) == 0

    def test_single_event_batch(self):
        """Test that batch mode works with single event."""
        strategy = BatchTestStrategy()
        timestamp = datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc)
        event = MarketEvent(
            timestamp=timestamp,
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            close=150.0,
        )

        strategy.on_timestamp_batch(timestamp, [event], None)

        assert len(strategy.batches_received[0]) == 1
        assert strategy.batches_received[0][0].asset_id == "AAPL"

    def test_large_batch_handling(self):
        """Test that batch mode can handle large event batches."""
        strategy = BatchTestStrategy()
        timestamp = datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc)

        # Create 100 events
        events = [
            MarketEvent(
                timestamp=timestamp,
                asset_id=f"ASSET_{i:03d}",
                data_type=MarketDataType.BAR,
                close=100.0 + i,
            )
            for i in range(100)
        ]

        strategy.on_timestamp_batch(timestamp, events, None)

        assert len(strategy.batches_received[0]) == 100
        assert strategy.batches_received[0][0].asset_id == "ASSET_000"
        assert strategy.batches_received[0][99].asset_id == "ASSET_099"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
