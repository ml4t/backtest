"""Pytest configuration and fixtures for QEngine tests."""

import tempfile
from collections.abc import Generator
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from qengine.core.clock import Clock, ClockMode
from qengine.core.event import Event, EventBus, MarketEvent
from qengine.core.types import AssetId, MarketDataType


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_market_data(temp_dir: Path) -> Path:
    """Create sample market data in Parquet format."""
    data = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, 9, 30) + timedelta(minutes=i) for i in range(100)],
            "open": [100.0 + i * 0.1 for i in range(100)],
            "high": [100.5 + i * 0.1 for i in range(100)],
            "low": [99.5 + i * 0.1 for i in range(100)],
            "close": [100.0 + i * 0.1 for i in range(100)],
            "volume": [1000000 + i * 1000 for i in range(100)],
        },
    )

    path = temp_dir / "sample_data.parquet"
    data.write_parquet(str(path))
    return path


@pytest.fixture
def sample_signal_data(temp_dir: Path) -> Path:
    """Create sample ML signal data."""
    data = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, 9, 30) + timedelta(minutes=i) for i in range(100)],
            "asset_id": ["AAPL"] * 100,
            "signal": [0.5 + 0.01 * i for i in range(100)],
            "confidence": [0.8] * 100,
        },
    )

    path = temp_dir / "sample_signals.parquet"
    data.write_parquet(str(path))
    return path


@pytest.fixture
def event_bus() -> EventBus:
    """Create an event bus for testing."""
    return EventBus(use_priority_queue=True)


@pytest.fixture
def clock() -> Clock:
    """Create a clock for testing."""
    return Clock(
        mode=ClockMode.BACKTEST,
        calendar="NYSE",
        start_time=datetime(2024, 1, 1, 9, 30),
        end_time=datetime(2024, 1, 31, 16, 0),
    )


@pytest.fixture
def sample_events() -> list[Event]:
    """Create a list of sample events for testing."""
    events = []
    base_time = datetime(2024, 1, 1, 9, 30)

    for i in range(10):
        event = MarketEvent(
            timestamp=base_time + timedelta(minutes=i),
            asset_id=AssetId("AAPL"),
            data_type=MarketDataType.BAR,
            open=100.0 + i,
            high=101.0 + i,
            low=99.0 + i,
            close=100.0 + i,
            volume=1000000,
        )
        events.append(event)

    return events


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton instances between tests."""
    # Add any singleton resets here
    yield
    # Cleanup after test


@pytest.fixture
def mock_broker():
    """Create a mock broker for testing strategies."""

    class MockBroker:
        def __init__(self):
            self.orders = []
            self.positions = {}
            self.cash = 100000.0

        def submit_order(self, order):
            self.orders.append(order)
            return order

        def get_positions(self):
            return self.positions.copy()

        def get_cash(self):
            return self.cash

        def target_position(self, asset, target):
            self.positions[asset] = target

    return MockBroker()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "benchmark: marks tests as benchmark tests")


# Pytest hooks for better test output
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add integration marker to integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add benchmark marker to benchmark tests
        if "benchmark" in str(item.fspath):
            item.add_marker(pytest.mark.benchmark)
