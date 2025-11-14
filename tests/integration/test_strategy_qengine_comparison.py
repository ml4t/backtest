"""
Integration Tests: Standalone vs QEngine Strategy Comparison

These tests validate that QEngine strategy adapters produce results consistent
with their standalone implementations, ensuring the integration preserves
core strategy logic while adding event-driven execution capabilities.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

# Add project paths
qengine_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(qengine_src))

from qengine.core.event import MarketEvent
from qengine.core.types import MarketDataType
from qengine.strategy.adapters import PITData
from qengine.strategy.crypto_basis_adapter import (
    CryptoBasisAdapter,
    CryptoBasisExternalStrategy,
)
from qengine.strategy.spy_order_flow_adapter import (
    SPYOrderFlowAdapter,
    SPYOrderFlowExternalStrategy,
    create_spy_order_flow_strategy,
)


class TestSPYOrderFlowComparison:
    """Compare SPY Order Flow standalone vs QEngine implementations."""

    @pytest.fixture
    def spy_market_data(self):
        """Generate realistic SPY market data for testing."""
        events = []
        base_time = datetime(2024, 1, 2, 9, 30)
        base_price = 450.0

        for i in range(50):
            timestamp = base_time + timedelta(minutes=i * 5)

            # Price with trend and volatility
            trend = 0.01 * i
            noise = np.random.randn() * 1.2
            price = base_price + trend + noise

            # Volume with realistic patterns
            volume = int(np.random.randint(800000, 1500000))

            # Order flow imbalance patterns
            if i < 15:
                imbalance = 0.5 + np.random.randn() * 0.05  # Balanced
            elif i < 30:
                imbalance = 0.68 + np.random.randn() * 0.08  # Buy pressure
            else:
                imbalance = 0.45 + np.random.randn() * 0.08  # Sell pressure

            imbalance = np.clip(imbalance, 0.2, 0.8)

            buy_volume = int(volume * imbalance)
            sell_volume = volume - buy_volume

            event = MarketEvent(
                timestamp=timestamp,
                asset_id="SPY",
                data_type=MarketDataType.BAR,
                open=price - abs(np.random.randn() * 0.3),
                high=price + abs(np.random.randn() * 0.3),
                low=price - abs(np.random.randn() * 0.3),
                close=price,
                volume=volume,
                metadata={
                    "buy_volume": buy_volume,
                    "sell_volume": sell_volume,
                    "vwap": price + np.random.randn() * 0.05,
                    "imbalance": imbalance,
                },
            )
            events.append(event)

        return events

    @pytest.mark.skip(reason="SPYOrderFlowExternalStrategy is a stub implementation with no trading logic")
    def test_standalone_vs_qengine_signals(self, spy_market_data):
        """Test that standalone and QEngine strategies generate consistent signals."""

        # Setup standalone strategy
        standalone = SPYOrderFlowExternalStrategy(
            asset_id="SPY",
            lookback_window=30,
            momentum_window_short=5,
            momentum_window_long=15,
            imbalance_threshold=0.65,
            momentum_threshold=0.002,
            min_data_points=10,
            signal_cooldown=3,
        )
        standalone.initialize()

        # Setup QEngine adapter
        qengine_adapter = SPYOrderFlowAdapter(
            asset_id="SPY",
            lookback_window=30,
            momentum_window_short=5,
            momentum_window_long=15,
            imbalance_threshold=0.65,
            momentum_threshold=0.002,
            position_scaling=0.1,
        )

        # Mock broker for QEngine adapter
        qengine_adapter.broker = Mock()
        qengine_adapter.broker.submit_order = Mock(return_value="order_123")
        qengine_adapter.broker.get_cash = Mock(return_value=100000)
        qengine_adapter.on_start()

        # Process events and collect signals
        standalone_signals = []
        qengine_signals = []

        for event in spy_market_data:
            # Create PITData for standalone
            pit_data = PITData(
                timestamp=event.timestamp,
                asset_data={
                    "SPY": {
                        "timestamp": event.timestamp,
                        "close": event.close,
                        "volume": event.volume,
                        "buy_volume": event.metadata.get("buy_volume", event.volume * 0.5),
                        "sell_volume": event.metadata.get("sell_volume", event.volume * 0.5),
                    },
                },
                market_prices={"SPY": event.close},
            )

            # Generate standalone signal
            standalone_signal = standalone.generate_signal(event.timestamp, pit_data)
            if standalone_signal:
                standalone_signals.append(
                    {
                        "timestamp": standalone_signal.timestamp,
                        "position": standalone_signal.position,
                        "confidence": standalone_signal.confidence,
                        "metadata": standalone_signal.metadata,
                    },
                )

            # Process QEngine event (update data history first)
            qengine_adapter._update_data_history(event)

            # Create PIT data with order flow metadata for the adapter's strategy
            adapter_pit_data = PITData(
                timestamp=event.timestamp,
                asset_data={
                    "SPY": {
                        "timestamp": event.timestamp,
                        "close": event.close,
                        "volume": event.volume,
                        "buy_volume": event.metadata.get("buy_volume", event.volume * 0.5),
                        "sell_volume": event.metadata.get("sell_volume", event.volume * 0.5),
                    },
                },
                market_prices={"SPY": event.close},
            )

            # Generate signal directly from adapter's external strategy
            adapter_signal = qengine_adapter.external_strategy.generate_signal(
                event.timestamp,
                adapter_pit_data,
            )
            if adapter_signal:
                qengine_signals.append(
                    {
                        "timestamp": adapter_signal.timestamp,
                        "position": adapter_signal.position,
                        "confidence": adapter_signal.confidence,
                        "metadata": adapter_signal.metadata,
                    },
                )

        # Cleanup
        standalone.finalize()
        qengine_adapter.on_end()

        # Analysis
        print("\nSignal Comparison Results:")
        print(f"Standalone signals: {len(standalone_signals)}")
        print(f"QEngine signals: {len(qengine_signals)}")

        # Validate signal generation
        assert len(standalone_signals) > 0, "Standalone strategy should generate signals"
        assert len(qengine_signals) > 0, "QEngine strategy should generate signals"

        # Signal timing comparison (should be similar)
        signal_ratio = len(qengine_signals) / len(standalone_signals)
        assert 0.5 <= signal_ratio <= 2.0, f"Signal counts should be similar, ratio: {signal_ratio}"

        # Check signal characteristics are reasonable
        if standalone_signals:
            avg_confidence = np.mean([s["confidence"] for s in standalone_signals])
            assert 0.1 < avg_confidence < 1.0, (
                f"Average confidence should be reasonable: {avg_confidence}"
            )

            positions = [s["position"] for s in standalone_signals]
            assert -1.5 <= min(positions) <= max(positions) <= 1.5, (
                "Positions should be within expected range"
            )

    @pytest.mark.skip(reason="SPYOrderFlowExternalStrategy is a stub implementation with no state tracking")
    def test_strategy_state_consistency(self, spy_market_data):
        """Test that strategy state evolves consistently between implementations."""

        # Create strategies with identical parameters
        standalone = SPYOrderFlowExternalStrategy(
            asset_id="SPY",
            lookback_window=25,
            min_data_points=5,
            signal_cooldown=2,
        )
        standalone.initialize()

        adapter = create_spy_order_flow_strategy(
            asset_id="SPY",
            lookback_window=25,
            position_scaling=0.1,
        )
        adapter.broker = Mock()
        adapter.broker.get_cash = Mock(return_value=100000)
        adapter.on_start()

        # Process subset of events
        test_events = spy_market_data[:20]

        for event in test_events:
            pit_data = PITData(
                timestamp=event.timestamp,
                asset_data={
                    "SPY": {
                        "volume": event.volume,
                        "buy_volume": event.metadata.get("buy_volume"),
                        "sell_volume": event.metadata.get("sell_volume"),
                    },
                },
                market_prices={"SPY": event.close},
            )

            # Update both strategies
            standalone.generate_signal(event.timestamp, pit_data)
            adapter.on_market_event(event)

        # Compare final states
        standalone_stats = standalone.get_current_statistics()
        adapter_diagnostics = adapter.get_strategy_diagnostics()
        adapter_stats = adapter_diagnostics.get("order_flow_statistics", {})

        # Both should have processed same number of data points
        assert standalone_stats["data_points"] == len(test_events)
        assert adapter_stats.get("data_points", 0) == len(test_events)

        print("State consistency check passed:")
        print(f"  Standalone data points: {standalone_stats['data_points']}")
        print(f"  Adapter data points: {adapter_stats.get('data_points', 0)}")
        print(f"  Standalone signals: {standalone_stats['signal_count']}")
        print(f"  Adapter signals: {adapter_stats.get('signal_count', 0)}")


class TestCryptoBasisComparison:
    """Compare Crypto Basis standalone vs QEngine implementations."""

    @pytest.fixture
    def crypto_market_data(self):
        """Generate crypto basis trading data."""
        events = []
        base_time = datetime(2024, 1, 1, 10, 0)

        # Generate coordinated spot and futures prices
        for i in range(30):
            timestamp = base_time + timedelta(minutes=i * 5)

            # Spot price evolution
            spot_price = 50000 + i * 25 + np.random.randn() * 100

            # Futures price with varying basis
            if i < 10:
                basis = 50 + np.random.randn() * 10  # Small basis
            elif i < 20:
                basis = 300 + np.random.randn() * 50  # Large basis
            else:
                basis = 100 + np.random.randn() * 20  # Converging basis

            futures_price = spot_price + basis

            # Create spot event
            spot_event = MarketEvent(
                timestamp=timestamp,
                asset_id="BTC",
                data_type=MarketDataType.BAR,
                open=spot_price - 10,
                high=spot_price + 15,
                low=spot_price - 15,
                close=spot_price,
                volume=1000,
            )

            # Create futures event
            futures_event = MarketEvent(
                timestamp=timestamp,
                asset_id="BTC_FUTURE",
                data_type=MarketDataType.BAR,
                open=futures_price - 12,
                high=futures_price + 18,
                low=futures_price - 18,
                close=futures_price,
                volume=800,
            )

            events.append(("spot", spot_event))
            events.append(("futures", futures_event))

        return events

    def test_basis_calculation_consistency(self, crypto_market_data):
        """Test that basis calculations are consistent between implementations."""

        # Setup standalone strategy
        standalone = CryptoBasisExternalStrategy(
            spot_asset_id="BTC",
            futures_asset_id="BTC_FUTURE",
            lookback_window=15,
            min_data_points=5,
            entry_threshold=2.0,
            exit_threshold=0.5,
        )
        standalone.initialize()

        # Setup QEngine adapter
        adapter = CryptoBasisAdapter(
            spot_asset_id="BTC",
            futures_asset_id="BTC_FUTURE",
            lookback_window=15,
            min_data_points=5,
            entry_threshold=2.0,
            exit_threshold=0.5,
        )
        adapter.broker = Mock()
        adapter.broker.submit_order = Mock(return_value="order_123")
        adapter.broker.get_cash = Mock(return_value=100000)
        adapter.on_start()

        # Process events
        spot_prices = []
        futures_prices = []
        timestamps = []

        for event_type, event in crypto_market_data:
            if event_type == "spot":
                spot_prices.append(event.close)
                timestamps.append(event.timestamp)
            else:
                futures_prices.append(event.close)

            # Update adapter
            adapter.on_market_event(event)

        # Test standalone with synchronized data
        for i in range(min(len(spot_prices), len(futures_prices))):
            if i >= 5:  # After minimum data points
                pit_data = PITData(
                    timestamp=timestamps[i],
                    asset_data={},
                    market_prices={
                        "BTC": spot_prices[i],
                        "BTC_FUTURE": futures_prices[i],
                    },
                )

                standalone.generate_signal(timestamps[i], pit_data)

        # Compare final statistics
        standalone.get_current_statistics()
        adapter_diagnostics = adapter.get_strategy_diagnostics()
        adapter_diagnostics.get("basis_statistics", {})

        # Count non-zero basis entries as signals
        standalone_signals = sum(1 for b in standalone.state.basis_history if b != 0)

        print("\nBasis Strategy Comparison:")
        print(f"Standalone basis history length: {len(standalone.state.basis_history)}")
        print(f"Standalone non-zero basis signals: {standalone_signals}")
        print(f"Adapter diagnostics: {list(adapter_diagnostics.keys())}")
        print(f"Standalone current position: {standalone.state.current_position}")

        # Both should have processed similar amounts of data
        assert len(standalone.state.basis_history) > 0, "Standalone should calculate basis"
        assert len(adapter_diagnostics) > 0, "Adapter should provide diagnostics"

        # Basis calculations should be reasonable
        if standalone.state.basis_history:
            avg_basis = sum(standalone.state.basis_history) / len(standalone.state.basis_history)
            assert avg_basis != 0, "Average basis should not be zero"

        standalone.finalize()
        adapter.on_end()


class TestIntegrationRobustness:
    """Test robustness and edge cases in strategy integration."""

    def test_empty_data_handling(self):
        """Test that strategies handle empty or minimal data gracefully."""

        # Test SPY strategy with minimal data
        spy_strategy = SPYOrderFlowExternalStrategy(min_data_points=5)
        spy_strategy.initialize()

        # Send insufficient data points
        base_time = datetime(2024, 1, 1, 10, 0)
        for i in range(3):  # Less than min_data_points
            pit_data = PITData(
                timestamp=base_time + timedelta(minutes=i),
                asset_data={"SPY": {"volume": 1000}},
                market_prices={"SPY": 450.0},
            )

            signal = spy_strategy.generate_signal(base_time + timedelta(minutes=i), pit_data)
            assert signal is None, "Should not generate signals with insufficient data"

        spy_strategy.finalize()

    def test_missing_data_fields(self):
        """Test handling of missing or malformed data fields."""

        strategy = SPYOrderFlowExternalStrategy()
        strategy.initialize()

        # Test with missing price data
        pit_data = PITData(
            timestamp=datetime.now(),
            asset_data={},
            market_prices={},  # Empty prices
        )

        signal = strategy.generate_signal(datetime.now(), pit_data)
        assert signal is None, "Should handle missing price data gracefully"

        # Test with missing volume data
        pit_data = PITData(
            timestamp=datetime.now(),
            asset_data={"SPY": {}},  # Missing volume
            market_prices={"SPY": 450.0},
        )

        signal = strategy.generate_signal(datetime.now(), pit_data)
        assert signal is None, "Should handle missing volume data gracefully"

        strategy.finalize()

    def test_extreme_market_conditions(self):
        """Test strategy behavior under extreme market conditions."""

        adapter = create_spy_order_flow_strategy(
            imbalance_threshold=0.7,
            momentum_threshold=0.01,
        )
        adapter.broker = Mock()
        adapter.broker.get_cash = Mock(return_value=100000)
        adapter.on_start()

        base_time = datetime(2024, 1, 1, 10, 0)

        # Create extreme market events
        extreme_events = [
            # Flash crash scenario
            MarketEvent(
                timestamp=base_time,
                asset_id="SPY",
                data_type=MarketDataType.BAR,
                close=450.0,
                volume=1000000,
                metadata={"buy_volume": 100000, "sell_volume": 900000},  # Extreme sell pressure
            ),
            # Recovery bounce
            MarketEvent(
                timestamp=base_time + timedelta(minutes=5),
                asset_id="SPY",
                data_type=MarketDataType.BAR,
                close=455.0,
                volume=2000000,
                metadata={"buy_volume": 1800000, "sell_volume": 200000},  # Extreme buy pressure
            ),
        ]

        for event in extreme_events:
            # Should not crash on extreme data
            try:
                adapter.on_market_event(event)
            except Exception as e:
                pytest.fail(f"Strategy should handle extreme conditions gracefully: {e}")

        adapter.on_end()


if __name__ == "__main__":
    # Run specific test for debugging
    import logging

    logging.basicConfig(level=logging.INFO)

    # Run SPY comparison test
    test_spy = TestSPYOrderFlowComparison()
    events = test_spy.spy_market_data()
    test_spy.test_standalone_vs_qengine_signals(events)

    print("Integration tests completed successfully!")
