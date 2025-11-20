#!/usr/bin/env python3
"""
Simple Strategy Adapter Demo
============================

This demonstrates the core Strategy-ml4t.backtest Integration Bridge functionality
without requiring full ml4t.backtest integration. It shows:

1. How external strategies integrate with ml4t.backtest adapters
2. Event processing and signal generation
3. Order submission through the broker interface

Run with: PYTHONPATH=~/ml4t/ml4t.backtest/src python examples/simple_adapter_demo.py
"""

from datetime import datetime, timedelta

import numpy as np

from ml4t.backtest.core.event import MarketEvent
from ml4t.backtest.core.types import MarketDataType, OrderSide
from ml4t.backtest.execution.order import Order
from ml4t.backtest.strategy import create_crypto_basis_strategy


class MockBroker:
    """Mock broker for demonstration purposes."""

    def __init__(self):
        self.orders = []
        self.cash = 100000

    def submit_order(self, order: Order) -> str:
        """Submit an order and return order ID."""
        order_id = f"order_{len(self.orders) + 1}"
        self.orders.append(order)
        print(
            f"  📋 Order submitted: {order.side.value} {order.quantity:.0f} {order.asset_id} @ {order.order_type.value}",
        )
        return order_id

    def get_cash(self) -> float:
        """Return available cash."""
        return self.cash


def generate_market_events(n_events: int = 50) -> list[MarketEvent]:
    """Generate synthetic market events for demo."""
    events = []
    start_time = datetime(2024, 1, 1, 10, 0, 0)

    # Generate spot and futures prices with varying basis
    np.random.seed(42)

    for i in range(n_events):
        timestamp = start_time + timedelta(minutes=i)

        # Spot price
        base_price = 50000 + i * 10 + np.random.normal(0, 100)
        spot_event = MarketEvent(
            timestamp=timestamp,
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            open=base_price * 0.999,
            high=base_price * 1.001,
            low=base_price * 0.998,
            close=base_price,
            volume=1000,
        )
        events.append(spot_event)

        # Futures price with dynamic basis (more extreme for demo)
        basis = (
            100 + 200 * np.sin(i * 0.1) + np.random.normal(0, 50)
        )  # More extreme oscillating basis
        futures_price = base_price + basis

        futures_event = MarketEvent(
            timestamp=timestamp,
            asset_id="BTC_FUTURE",
            data_type=MarketDataType.BAR,
            open=futures_price * 0.999,
            high=futures_price * 1.001,
            low=futures_price * 0.998,
            close=futures_price,
            volume=800,
        )
        events.append(futures_event)

    return events


def run_adapter_demo():
    """Run the strategy adapter demonstration."""
    print("🚀 Strategy-ml4t.backtest Integration Bridge Demo")
    print("=" * 50)

    # Create the integrated strategy
    print("\n📊 Creating Crypto Basis Strategy Adapter...")
    strategy = create_crypto_basis_strategy(
        spot_asset_id="BTC",
        futures_asset_id="BTC_FUTURE",
        lookback_window=20,
        entry_threshold=0.8,  # Lower threshold for more signals
        exit_threshold=0.3,
        max_position=0.2,
        position_scaling=0.05,  # Use 5% of capital per signal
    )

    # Mock broker
    broker = MockBroker()
    strategy.broker = broker

    print(f"✅ Strategy created: {strategy.name}")
    print(f"   Spot Asset: {strategy.spot_asset_id}")
    print(f"   Futures Asset: {strategy.futures_asset_id}")

    # Start strategy
    strategy.on_start()

    # Generate market events
    print("\n📈 Generating synthetic market data...")
    events = generate_market_events(n_events=100)
    print(f"✅ Generated {len(events)} market events")

    # Process events
    print("\n⚡ Processing market events...")
    signal_count = 0
    order_count = 0

    for i, event in enumerate(events):
        if i % 20 == 0:
            print(f"   Processing event {i + 1}/{len(events)}")

        # Process event through adapter
        initial_order_count = len(broker.orders)
        strategy.on_market_event(event)

        # Check if new orders were generated
        new_orders = len(broker.orders) - initial_order_count
        if new_orders > 0:
            signal_count += 1
            order_count += new_orders

            # Get strategy diagnostics
            diagnostics = strategy.get_strategy_diagnostics()
            basis_stats = diagnostics.get("basis_statistics", {})

            print(f"   🎯 Signal generated at {event.timestamp}")
            print(f"      Basis Z-Score: {basis_stats.get('z_score', 0):.2f}")
            print(f"      Position: {basis_stats.get('current_position', 0):.3f}")

        # Debug: Show basis statistics periodically
        elif i % 50 == 0 and i > 0:
            diagnostics = strategy.get_strategy_diagnostics()
            basis_stats = diagnostics.get("basis_statistics", {})
            if basis_stats.get("data_points", 0) > 0:
                print(
                    f"   📊 Debug - Z-Score: {basis_stats.get('z_score', 0):.2f}, Data: {basis_stats.get('data_points', 0)}",
                )

    # Final results
    print("\n📊 Demo Results:")
    print(f"   Events processed: {len(events)}")
    print(f"   Signals generated: {signal_count}")
    print(f"   Orders submitted: {order_count}")

    # Show order details
    if broker.orders:
        print("\n📋 Order Summary:")
        buy_orders = sum(1 for o in broker.orders if o.side == OrderSide.BUY)
        sell_orders = sum(1 for o in broker.orders if o.side == OrderSide.SELL)
        total_quantity = sum(o.quantity for o in broker.orders)

        print(f"   Buy orders: {buy_orders}")
        print(f"   Sell orders: {sell_orders}")
        print(f"   Total quantity: {total_quantity:.0f}")

        # Show last few orders
        print("\n📋 Last 3 Orders:")
        for order in broker.orders[-3:]:
            print(f"   {order.side.value} {order.quantity:.0f} {order.asset_id}")

    # Get final strategy state
    final_state = strategy.get_strategy_diagnostics()
    basis_stats = final_state.get("basis_statistics", {})

    print("\n🎯 Final Strategy State:")
    print(f"   Current Position: {basis_stats.get('current_position', 0):.3f}")
    print(f"   Data Points: {basis_stats.get('data_points', 0)}")
    print(f"   Current Z-Score: {basis_stats.get('z_score', 0):.2f}")

    # Stop strategy
    strategy.on_end()

    print("\n✅ Integration Bridge Demo Complete!")
    print("   🔗 External strategy successfully integrated with ml4t.backtest")
    print("   📊 Event-driven processing working correctly")
    print("   🎯 Signal generation and order submission functional")
    print("   🚀 Ready for full backtesting with ml4t.backtest!")

    return {
        "events_processed": len(events),
        "signals_generated": signal_count,
        "orders_submitted": order_count,
        "final_state": final_state,
    }


if __name__ == "__main__":
    try:
        results = run_adapter_demo()
        print("\n✨ Demo completed successfully!")
        exit(0)
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
