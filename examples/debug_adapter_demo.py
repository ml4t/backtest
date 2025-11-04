#!/usr/bin/env python3
"""
Debug Strategy Adapter Demo
===========================

This simplified demo isolates and tests individual components to verify
the Strategy-QEngine Integration Bridge is working correctly.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock

from qengine.core.event import MarketEvent
from qengine.core.types import MarketDataType
from qengine.strategy.adapters import PITData
from qengine.strategy.crypto_basis_adapter import CryptoBasisExternalStrategy


def test_external_strategy_directly():
    """Test the external strategy implementation directly."""
    print("🧪 Testing CryptoBasisExternalStrategy directly...")

    strategy = CryptoBasisExternalStrategy(
        spot_asset_id="BTC",
        futures_asset_id="BTC_FUTURE",
        lookback_window=10,
        entry_threshold=0.5,
        min_data_points=5,
    )

    strategy.initialize()

    # Send some price data with obvious basis patterns
    base_time = datetime(2024, 1, 1, 10, 0, 0)

    print("📊 Sending price data with varying basis...")

    for i in range(15):
        spot_price = 50000.0

        # Create obvious basis pattern
        if i < 5:
            futures_price = 50100.0  # Basis = 100 (normal)
        elif i < 10:
            futures_price = 50500.0  # Basis = 500 (high - should trigger entry)
        else:
            futures_price = 50050.0  # Basis = 50 (low - should trigger exit)

        pit_data = PITData(
            timestamp=base_time + timedelta(minutes=i),
            asset_data={},
            market_prices={
                "BTC": spot_price,
                "BTC_FUTURE": futures_price,
            },
        )

        signal = strategy.generate_signal(
            base_time + timedelta(minutes=i),
            pit_data,
        )

        # Get statistics
        stats = strategy.get_current_statistics()

        print(
            f"   Step {i + 1}: Basis={futures_price - spot_price:6.0f}, "
            f"Z-Score={stats.get('z_score', 0):6.2f}, "
            f"Pos={stats.get('current_position', 0):6.3f}"
            f"{' 🎯 SIGNAL!' if signal and signal.position != 0 else ''}",
        )

        if signal and abs(signal.position) > 0.001:
            print(f"      Signal: {signal.position:.3f} confidence={signal.confidence:.3f}")

    print("✅ External strategy test complete\n")


def test_adapter_integration():
    """Test the full adapter integration."""
    print("🧪 Testing full adapter integration...")

    from qengine.strategy import create_crypto_basis_strategy

    # Create strategy with same parameters as direct test
    strategy = create_crypto_basis_strategy(
        spot_asset_id="BTC",
        futures_asset_id="BTC_FUTURE",
        lookback_window=10,  # Match direct test
        entry_threshold=0.5,  # Match direct test
        exit_threshold=0.1,
        min_data_points=5,  # Match direct test
        position_scaling=0.1,
    )

    # Mock broker
    broker = Mock()
    broker.submit_order = Mock(return_value="order_123")
    broker.get_cash = Mock(return_value=100000)
    strategy.broker = broker

    strategy.on_start()

    # Send market events
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    orders_submitted = 0

    print("📊 Processing market events...")

    for i in range(20):
        timestamp = base_time + timedelta(minutes=i)

        # Spot event
        spot_event = MarketEvent(
            timestamp=timestamp,
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            close=50000.0,
            volume=1000,
        )

        # Futures event with the SAME pattern as direct test
        if i < 5:
            basis = 100.0  # Normal basis
        elif i < 10:
            basis = 500.0  # High basis (matches direct test)
        else:
            basis = 50.0  # Low basis

        futures_event = MarketEvent(
            timestamp=timestamp,
            asset_id="BTC_FUTURE",
            data_type=MarketDataType.BAR,
            close=50000.0 + basis,
            volume=800,
        )

        # Process both events - need both to calculate basis
        initial_calls = broker.submit_order.call_count

        # Process spot event first (updates data but won't generate signal yet)
        strategy.on_market_event(spot_event)

        # Process futures event (now both prices available, can generate signal)
        strategy.on_market_event(futures_event)

        new_calls = broker.submit_order.call_count - initial_calls

        if new_calls > 0:
            orders_submitted += new_calls
            diagnostics = strategy.get_strategy_diagnostics()
            basis_stats = diagnostics.get("basis_statistics", {})

            print(
                f"   Event {i + 1}: Basis={basis:6.0f}, "
                f"Z-Score={basis_stats.get('z_score', 0):6.2f} "
                f"🎯 ORDER SUBMITTED!",
            )
        else:
            diagnostics = strategy.get_strategy_diagnostics()
            basis_stats = diagnostics.get("basis_statistics", {})
            if i % 5 == 0:
                print(
                    f"   Event {i + 1}: Basis={basis:6.0f}, "
                    f"Z-Score={basis_stats.get('z_score', 0):6.2f}, "
                    f"Data={basis_stats.get('data_points', 0)}, "
                    f"Current Basis={basis_stats.get('current_basis', 0):6.0f}",
                )

    strategy.on_end()

    print("✅ Adapter integration test complete")
    print(f"   Orders submitted: {orders_submitted}")
    print(f"   Broker calls: {broker.submit_order.call_count}")

    return orders_submitted > 0


def main():
    """Run debug tests."""
    print("🔍 Strategy-QEngine Integration Bridge Debug")
    print("=" * 50)

    success = True

    try:
        # Test 1: External strategy directly
        test_external_strategy_directly()

        # Test 2: Full adapter integration
        integration_success = test_adapter_integration()

        if integration_success:
            print("\n✅ ALL TESTS PASSED!")
            print("   🔗 External strategy working correctly")
            print("   📊 Adapter integration functional")
            print("   🎯 Signal generation and order submission working")
        else:
            print("\n⚠️  Integration test didn't generate orders")
            print("   Strategy logic may need adjustment")
            success = False

    except Exception as e:
        print(f"\n❌ Debug tests failed: {e}")
        import traceback

        traceback.print_exc()
        success = False

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
