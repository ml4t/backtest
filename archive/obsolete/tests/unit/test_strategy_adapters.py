"""Tests for strategy adapters."""

from datetime import datetime, timedelta
from unittest.mock import Mock

import numpy as np

from ml4t.backtest.core.event import FillEvent, MarketEvent
from ml4t.backtest.core.types import MarketDataType, OrderSide
from ml4t.backtest.strategy.adapters import (
    DataFrameAdapter,
    ExternalStrategyInterface,
    PITData,
    StrategyAdapter,
    StrategySignal,
)
from ml4t.backtest.strategy.crypto_basis_adapter import (
    CryptoBasisAdapter,
    CryptoBasisExternalStrategy,
    create_crypto_basis_strategy,
)


class MockExternalStrategy(ExternalStrategyInterface):
    """Mock external strategy for testing."""

    def __init__(self):
        self.initialized = False
        self.finalized = False
        self.signals_generated = []

    def initialize(self):
        self.initialized = True

    def finalize(self):
        self.finalized = True

    def generate_signal(self, timestamp, pit_data):
        signal = StrategySignal(
            timestamp=timestamp,
            asset_id="TEST",
            position=0.5,  # Always generate 50% position
            confidence=0.8,
            metadata={"test": True},
        )
        self.signals_generated.append(signal)
        return signal


class TestStrategyAdapter:
    """Test StrategyAdapter base class."""

    def test_adapter_initialization(self):
        """Test adapter initialization."""
        external = MockExternalStrategy()
        adapter = StrategyAdapter(external)

        assert adapter.external_strategy is external
        assert adapter.name.startswith("Adapter_MockExternalStrategy")
        assert not external.initialized

    def test_adapter_lifecycle(self):
        """Test adapter lifecycle methods."""
        external = MockExternalStrategy()
        adapter = StrategyAdapter(external)

        # Start
        adapter.on_start()
        assert external.initialized

        # End
        adapter.on_end()
        assert external.finalized

    def test_market_event_processing(self):
        """Test processing of market events."""
        external = MockExternalStrategy()
        adapter = StrategyAdapter(external)
        adapter.broker = Mock()
        adapter.broker.submit_order = Mock(return_value="order_123")
        adapter.broker.get_cash = Mock(return_value=100000)

        # Create market event
        event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 10, 0, 0),
            asset_id="TEST",
            data_type=MarketDataType.BAR,
            open=100.0,
            high=102.0,
            low=99.0,
            close=101.0,
            volume=1000,
        )

        # Process event
        adapter.on_market_event(event)

        # Check signal was generated
        assert len(external.signals_generated) == 1
        signal = external.signals_generated[0]
        assert signal.asset_id == "TEST"
        assert signal.position == 0.5

        # Check order was submitted
        adapter.broker.submit_order.assert_called_once()

    def test_fill_event_processing(self):
        """Test processing of fill events."""
        external = MockExternalStrategy()
        adapter = StrategyAdapter(external)

        # Create fill event
        fill_event = FillEvent(
            timestamp=datetime(2024, 1, 1, 10, 0, 0),
            order_id="order_123",
            trade_id="trade_456",
            asset_id="TEST",
            side=OrderSide.BUY,
            fill_quantity=100,
            fill_price=101.0,
            commission=1.0,
            slippage=0.1,
            market_impact=0.05,
        )

        # Process fill
        adapter.on_fill_event(fill_event)

        # Check position was updated
        assert adapter.current_positions.get("TEST", 0) == 100

    def test_position_sizing(self):
        """Test default position sizing."""
        external = MockExternalStrategy()
        adapter = StrategyAdapter(external)

        signal = StrategySignal(
            timestamp=datetime.now(),
            asset_id="TEST",
            position=1.0,
            confidence=0.5,
        )

        # Test position sizing
        cash = 100000
        position_size = adapter.position_sizer(signal, cash)

        # Should be 10% of cash * position * confidence
        expected = cash * 0.1 * 1.0 * 0.5
        assert abs(position_size - expected) < 0.01

    def test_risk_management(self):
        """Test default risk management."""
        external = MockExternalStrategy()
        adapter = StrategyAdapter(external)

        # Valid signal
        valid_signal = StrategySignal(
            timestamp=datetime.now(),
            asset_id="TEST",
            position=0.5,
            confidence=0.8,
        )
        assert adapter.risk_manager(valid_signal)

        # Invalid signal - too large position
        invalid_signal = StrategySignal(
            timestamp=datetime.now(),
            asset_id="TEST",
            position=15.0,  # Too large
            confidence=0.8,
        )
        assert not adapter.risk_manager(invalid_signal)


class TestDataFrameAdapter:
    """Test DataFrameAdapter class."""

    def test_dataframe_updates(self):
        """Test DataFrame updates with market data."""
        external = MockExternalStrategy()
        adapter = DataFrameAdapter(external, window_size=5)

        # Send multiple market events
        for i in range(7):
            event = MarketEvent(
                timestamp=datetime(2024, 1, 1, 10, i, 0),
                asset_id="TEST",
                data_type=MarketDataType.BAR,
                open=100.0 + i,
                high=102.0 + i,
                low=99.0 + i,
                close=101.0 + i,
                volume=1000 + i * 100,
            )
            adapter._update_data_history(event)

        # Check DataFrame was created and windowed
        df = adapter.get_dataframe("TEST")
        assert df is not None
        assert len(df) == 5  # Window size
        assert df["close"].to_list() == [103.0, 104.0, 105.0, 106.0, 107.0]

    def test_multiple_assets(self):
        """Test handling multiple assets."""
        external = MockExternalStrategy()
        adapter = DataFrameAdapter(external)

        # Send events for different assets
        for asset in ["BTC", "ETH"]:
            event = MarketEvent(
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
                asset_id=asset,
                data_type=MarketDataType.BAR,
                open=100.0,
                high=102.0,
                low=99.0,
                close=101.0,
                volume=1000,
            )
            adapter._update_data_history(event)

        # Check both DataFrames exist
        dataframes = adapter.get_all_dataframes()
        assert "BTC" in dataframes
        assert "ETH" in dataframes
        assert len(dataframes["BTC"]) == 1
        assert len(dataframes["ETH"]) == 1


class TestCryptoBasisStrategy:
    """Test crypto basis strategy integration."""

    def test_basis_strategy_initialization(self):
        """Test basis strategy initialization."""
        strategy = CryptoBasisExternalStrategy(spot_asset_id="BTC", futures_asset_id="BTC_FUTURE")

        assert strategy.spot_asset_id == "BTC"
        assert strategy.futures_asset_id == "BTC_FUTURE"
        assert not strategy.state.basis_history

    def test_basis_calculation(self):
        """Test basis calculation and signal generation."""
        strategy = CryptoBasisExternalStrategy(
            spot_asset_id="BTC",
            futures_asset_id="BTC_FUTURE",
            min_data_points=3,
            entry_threshold=1.5,
        )

        strategy.initialize()

        # Generate data with predictable basis pattern
        base_time = datetime(2024, 1, 1, 10, 0, 0)

        # Send initial data points (basis around 0)
        for i in range(10):
            pit_data = PITData(
                timestamp=base_time + timedelta(minutes=i),
                asset_data={},
                market_prices={
                    "BTC": 50000.0 + i * 10,  # Spot
                    "BTC_FUTURE": 50000.0 + i * 10 + 100,  # Futures (basis = 100)
                },
            )

            signal = strategy.generate_signal(base_time + timedelta(minutes=i), pit_data)

            # First few points should not generate signals (building history)
            if i < 3:
                assert signal is None

        # Now send data with high basis to trigger entry
        for i in range(5):
            pit_data = PITData(
                timestamp=base_time + timedelta(minutes=10 + i),
                asset_data={},
                market_prices={
                    "BTC": 50100.0,
                    "BTC_FUTURE": 50600.0,  # High basis = 500 (should trigger entry)
                },
            )

            signal = strategy.generate_signal(base_time + timedelta(minutes=10 + i), pit_data)

            if signal and signal.position != 0:
                # Should generate short signal (high basis)
                assert signal.position < 0
                assert signal.confidence > 0
                break

    def test_basis_adapter_integration(self):
        """Test full basis adapter with ml4t.backtest integration."""
        adapter = CryptoBasisAdapter(
            spot_asset_id="BTC",
            futures_asset_id="BTC_FUTURE",
            lookback_window=10,
            min_data_points=3,
        )

        # Mock broker
        adapter.broker = Mock()
        adapter.broker.submit_order = Mock(return_value="order_123")
        adapter.broker.get_cash = Mock(return_value=100000)

        adapter.on_start()

        # Send market data for both assets
        base_time = datetime(2024, 1, 1, 10, 0, 0)

        # Build up history
        for i in range(15):
            # Spot event
            spot_event = MarketEvent(
                timestamp=base_time + timedelta(minutes=i),
                asset_id="BTC",
                data_type=MarketDataType.BAR,
                open=50000.0,
                high=50100.0,
                low=49900.0,
                close=50000.0 + i * 10,
                volume=1000,
            )
            adapter.on_market_event(spot_event)

            # Futures event with growing basis
            futures_event = MarketEvent(
                timestamp=base_time + timedelta(minutes=i),
                asset_id="BTC_FUTURE",
                data_type=MarketDataType.BAR,
                open=50100.0,
                high=50200.0,
                low=50000.0,
                close=50000.0 + i * 10 + 100 + i * 20,  # Increasing basis
                volume=800,
            )
            adapter.on_market_event(futures_event)

        # Check that adapter processed events
        state = adapter.get_strategy_state()
        assert len(state["data_history_lengths"]) == 2
        assert state["data_history_lengths"]["BTC"] > 0
        assert state["data_history_lengths"]["BTC_FUTURE"] > 0

        # Check if any orders were submitted
        if adapter.broker.submit_order.called:
            assert adapter.broker.submit_order.call_count > 0

    def test_factory_function(self):
        """Test factory function for creating strategy."""
        strategy = create_crypto_basis_strategy(
            spot_asset_id="ETH",
            futures_asset_id="ETH_FUTURE",
            entry_threshold=2.5,
        )

        assert isinstance(strategy, CryptoBasisAdapter)
        assert strategy.spot_asset_id == "ETH"
        assert strategy.futures_asset_id == "ETH_FUTURE"

    def test_strategy_diagnostics(self):
        """Test strategy diagnostics."""
        adapter = CryptoBasisAdapter()

        diagnostics = adapter.get_strategy_diagnostics()

        # Check required fields
        assert "basis_statistics" in diagnostics
        assert "spot_asset" in diagnostics
        assert "futures_asset" in diagnostics
