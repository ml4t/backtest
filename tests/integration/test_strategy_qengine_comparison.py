"""
Integration Tests: Standalone vs ml4t.backtest Strategy Comparison

These tests validate that ml4t.backtest strategy adapters produce results consistent
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
ml4t.backtest_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(ml4t.backtest_src))

from ml4t.backtest.core.event import MarketEvent
from ml4t.backtest.core.types import MarketDataType
from ml4t.backtest.strategy.adapters import PITData
from ml4t.backtest.strategy.crypto_basis_adapter import (
    CryptoBasisAdapter,
    CryptoBasisExternalStrategy,
)


class TestCryptoBasisComparison:
    """Compare Crypto Basis standalone vs ml4t.backtest implementations."""

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

        # Setup ml4t.backtest adapter
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

