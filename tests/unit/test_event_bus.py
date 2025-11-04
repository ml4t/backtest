"""Unit tests for the EventBus component."""

import time
from datetime import datetime, timedelta
from threading import Thread

from qengine.core.event import EventBus, MarketEvent
from qengine.core.types import AssetId, EventType, MarketDataType


class TestEventBus:
    """Test suite for EventBus functionality."""

    def test_event_bus_initialization(self):
        """Test EventBus can be initialized with different modes."""
        # Priority queue mode
        bus1 = EventBus(use_priority_queue=True)
        assert bus1.use_priority_queue is True
        assert bus1.pending_count == 0

        # FIFO mode
        bus2 = EventBus(use_priority_queue=False)
        assert bus2.use_priority_queue is False
        assert bus2.pending_count == 0

    def test_publish_and_process_single_event(self, event_bus):
        """Test publishing and processing a single event."""
        # Create a test event
        event = MarketEvent(
            timestamp=datetime.now(),
            asset_id=AssetId("AAPL"),
            data_type=MarketDataType.BAR,
            close=150.0,
        )

        # Track if handler was called
        handler_called = False

        def handler(e):
            nonlocal handler_called
            handler_called = True
            assert e == event

        # Subscribe and publish
        event_bus.subscribe(EventType.MARKET, handler)
        event_bus.publish(event)

        assert event_bus.pending_count == 1

        # Process the event
        processed_event = event_bus.process_next()

        assert processed_event == event
        assert handler_called is True
        assert event_bus.pending_count == 0

    def test_chronological_ordering_with_priority_queue(self):
        """Test that events are processed in chronological order."""
        bus = EventBus(use_priority_queue=True)

        # Create events with different timestamps
        base_time = datetime(2024, 1, 1, 9, 30)
        event1 = MarketEvent(
            timestamp=base_time + timedelta(minutes=2),
            asset_id=AssetId("AAPL"),
            data_type=MarketDataType.BAR,
            close=152.0,
        )
        event2 = MarketEvent(
            timestamp=base_time,
            asset_id=AssetId("AAPL"),
            data_type=MarketDataType.BAR,
            close=150.0,
        )
        event3 = MarketEvent(
            timestamp=base_time + timedelta(minutes=1),
            asset_id=AssetId("AAPL"),
            data_type=MarketDataType.BAR,
            close=151.0,
        )

        # Publish out of order
        bus.publish(event1)
        bus.publish(event2)
        bus.publish(event3)

        # Process and verify chronological order
        assert bus.process_next() == event2  # Earliest
        assert bus.process_next() == event3  # Middle
        assert bus.process_next() == event1  # Latest
        assert bus.process_next() is None

    def test_fifo_ordering_without_priority_queue(self):
        """Test FIFO ordering when priority queue is disabled."""
        bus = EventBus(use_priority_queue=False)

        events = []
        for i in range(3):
            event = MarketEvent(
                timestamp=datetime.now(),
                asset_id=AssetId("AAPL"),
                data_type=MarketDataType.BAR,
                close=150.0 + i,
            )
            events.append(event)
            bus.publish(event)

        # Process in FIFO order
        for expected_event in events:
            assert bus.process_next() == expected_event

        assert bus.process_next() is None

    def test_multiple_subscribers(self, event_bus):
        """Test multiple subscribers receive events."""
        handler1_called = False
        handler2_called = False

        def handler1(event):
            nonlocal handler1_called
            handler1_called = True

        def handler2(event):
            nonlocal handler2_called
            handler2_called = True

        # Subscribe both handlers
        event_bus.subscribe(EventType.MARKET, handler1)
        event_bus.subscribe(EventType.MARKET, handler2)

        # Publish event
        event = MarketEvent(
            timestamp=datetime.now(),
            asset_id=AssetId("AAPL"),
            data_type=MarketDataType.BAR,
            close=150.0,
        )
        event_bus.publish(event)
        event_bus.process_next()

        # Both handlers should be called
        assert handler1_called is True
        assert handler2_called is True

    def test_unsubscribe(self, event_bus):
        """Test unsubscribing from events."""
        handler_called = False

        def handler(event):
            nonlocal handler_called
            handler_called = True

        # Subscribe then unsubscribe
        event_bus.subscribe(EventType.MARKET, handler)
        event_bus.unsubscribe(EventType.MARKET, handler)

        # Publish and process event
        event = MarketEvent(
            timestamp=datetime.now(),
            asset_id=AssetId("AAPL"),
            data_type=MarketDataType.BAR,
            close=150.0,
        )
        event_bus.publish(event)
        event_bus.process_next()

        # Handler should not be called
        assert handler_called is False

    def test_process_all(self, event_bus):
        """Test processing all events at once."""
        events_processed = []

        def handler(event):
            events_processed.append(event)

        event_bus.subscribe(EventType.MARKET, handler)

        # Publish multiple events
        for i in range(5):
            event = MarketEvent(
                timestamp=datetime.now() + timedelta(seconds=i),
                asset_id=AssetId("AAPL"),
                data_type=MarketDataType.BAR,
                close=150.0 + i,
            )
            event_bus.publish(event)

        # Process all
        count = event_bus.process_all()

        assert count == 5
        assert len(events_processed) == 5
        assert event_bus.pending_count == 0

    def test_clear(self, event_bus):
        """Test clearing the event queue."""
        # Add some events
        for i in range(3):
            event = MarketEvent(
                timestamp=datetime.now(),
                asset_id=AssetId("AAPL"),
                data_type=MarketDataType.BAR,
                close=150.0 + i,
            )
            event_bus.publish(event)

        assert event_bus.pending_count == 3

        # Clear the queue
        event_bus.clear()

        assert event_bus.pending_count == 0
        assert event_bus.process_next() is None

    def test_peek(self, event_bus):
        """Test peeking at the next event without removing it."""
        event = MarketEvent(
            timestamp=datetime.now(),
            asset_id=AssetId("AAPL"),
            data_type=MarketDataType.BAR,
            close=150.0,
        )

        event_bus.publish(event)

        # Peek should not remove the event
        peeked_event = event_bus.peek()
        assert peeked_event == event
        assert event_bus.pending_count == 1

        # Process should remove it
        processed_event = event_bus.process_next()
        assert processed_event == event
        assert event_bus.pending_count == 0

    def test_thread_safety(self, event_bus):
        """Test thread-safe operations on EventBus."""
        events_received = []

        def handler(event):
            events_received.append(event)

        event_bus.subscribe(EventType.MARKET, handler)

        def publisher():
            for i in range(10):
                event = MarketEvent(
                    timestamp=datetime.now(),
                    asset_id=AssetId("AAPL"),
                    data_type=MarketDataType.BAR,
                    close=150.0 + i,
                )
                event_bus.publish(event)
                time.sleep(0.001)

        def processor():
            for _ in range(10):
                event_bus.process_next()
                time.sleep(0.001)

        # Run publisher and processor in parallel
        publisher_thread = Thread(target=publisher)
        processor_thread = Thread(target=processor)

        publisher_thread.start()
        processor_thread.start()

        publisher_thread.join()
        processor_thread.join()

        # All events should be processed
        assert len(events_received) == 10

    def test_error_handling_in_handler(self, event_bus):
        """Test that errors in handlers don't break the event bus."""
        successful_handler_called = False

        def failing_handler(event):
            raise ValueError("Handler error")

        def successful_handler(event):
            nonlocal successful_handler_called
            successful_handler_called = True

        # Subscribe both handlers
        event_bus.subscribe(EventType.MARKET, failing_handler)
        event_bus.subscribe(EventType.MARKET, successful_handler)

        # Publish and process event
        event = MarketEvent(
            timestamp=datetime.now(),
            asset_id=AssetId("AAPL"),
            data_type=MarketDataType.BAR,
            close=150.0,
        )
        event_bus.publish(event)

        # Should not raise exception
        processed_event = event_bus.process_next()

        assert processed_event == event
        assert successful_handler_called is True
