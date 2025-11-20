#!/usr/bin/env python
"""
SPY Order Flow Strategy Demo

Demonstrates the integration of the SPY Order Flow momentum strategy
with ml4t.backtest's event-driven backtesting framework.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# Add ml4t.backtest to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from unittest.mock import Mock

from ml4t.backtest.core.event import MarketEvent
from ml4t.backtest.core.types import MarketDataType
from ml4t.backtest.strategy.spy_order_flow_adapter import create_spy_order_flow_strategy


def generate_spy_order_flow_data(n_events: int = 100):
    """Generate synthetic SPY order flow data with realistic patterns."""
    events = []
    base_time = datetime(2024, 1, 2, 9, 30)
    base_price = 450.0

    for i in range(n_events):
        timestamp = base_time + timedelta(minutes=5 * i)

        # Generate price with trend and noise
        trend = 0.02 * i  # Slight upward trend
        noise = np.random.randn() * 1.5
        price = base_price + trend + noise

        # Generate order flow imbalance patterns
        if i < 30:
            # Morning: balanced flow
            imbalance = 0.5 + np.random.randn() * 0.1
        elif i < 60:
            # Mid-day: buy pressure
            imbalance = 0.65 + np.random.randn() * 0.1
        else:
            # Afternoon: varying flow
            imbalance = 0.5 + np.sin(i * 0.2) * 0.3 + np.random.randn() * 0.05

        imbalance = np.clip(imbalance, 0.1, 0.9)

        # Calculate volumes
        total_volume = np.random.randint(800000, 1500000)
        buy_volume = int(total_volume * imbalance)
        sell_volume = total_volume - buy_volume

        # Create event with order flow metadata
        event = MarketEvent(
            timestamp=timestamp,
            asset_id="SPY",
            data_type=MarketDataType.BAR,
            open=price - abs(np.random.randn() * 0.5),
            high=price + abs(np.random.randn() * 0.5),
            low=price - abs(np.random.randn() * 0.5),
            close=price,
            volume=total_volume,
            metadata={
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
                "vwap": price + np.random.randn() * 0.1,
                "tick_count": np.random.randint(1000, 5000),
            },
        )
        events.append(event)

    return events


def main():
    """Run SPY order flow strategy demo."""
    print("=" * 60)
    print("SPY ORDER FLOW STRATEGY DEMO")
    print("=" * 60)
    print()

    # Create strategy
    print("📊 Creating SPY Order Flow Strategy...")
    strategy = create_spy_order_flow_strategy(
        asset_id="SPY",
        lookback_window=50,
        momentum_window_short=5,
        momentum_window_long=20,
        imbalance_threshold=0.65,
        momentum_threshold=0.002,
        position_scaling=0.15,
    )
    print(f"✅ Strategy created: {strategy.name}")
    print()

    # Mock broker for demo
    broker = Mock()
    broker.submit_order = Mock(return_value="order_123")
    broker.get_cash = Mock(return_value=100000)
    strategy.broker = broker

    # Start strategy
    strategy.on_start()

    # Generate synthetic order flow data
    print("📈 Generating synthetic order flow data...")
    events = generate_spy_order_flow_data(100)
    print(f"✅ Generated {len(events)} market events")
    print()

    # Process events
    print("⚡ Processing order flow events...")
    orders_submitted = 0

    for i, event in enumerate(events):
        # Process event
        initial_orders = broker.submit_order.call_count
        strategy.on_market_event(event)
        new_orders = broker.submit_order.call_count - initial_orders

        if new_orders > 0:
            orders_submitted += new_orders

            # Get current diagnostics
            diagnostics = strategy.get_strategy_diagnostics()
            stats = diagnostics.get("order_flow_statistics", {})

            print(f"  [{event.timestamp.strftime('%H:%M')}] Signal Generated!")
            print(f"    Imbalance: {stats.get('imbalance_ratio', 0):.3f}")
            print(f"    Momentum:  {stats.get('price_momentum_5', 0):.4f}")
            print(f"    Position:  {stats.get('current_position', 0):.3f}")

        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(events)} events...")

    # Final statistics
    strategy.on_end()

    diagnostics = strategy.get_strategy_diagnostics()
    stats = diagnostics.get("order_flow_statistics", {})

    print()
    print("=" * 60)
    print("DEMO RESULTS")
    print("=" * 60)
    print(f"📊 Events Processed:    {len(events)}")
    print(f"🎯 Signals Generated:   {stats.get('signal_count', 0)}")
    print(f"📝 Orders Submitted:    {orders_submitted}")
    print(f"📈 Data Points Used:    {stats.get('data_points', 0)}")
    print()

    # Show order details if any
    if broker.submit_order.called:
        print("Order Details:")
        for call in broker.submit_order.call_args_list[-5:]:  # Last 5 orders
            order = call[0][0]
            print(f"  - {order.side.value}: {order.quantity} shares at {order.order_type.value}")

    print()
    print("✨ Demo completed successfully!")
    print()
    print("Key Features Demonstrated:")
    print("  ✓ Order flow imbalance detection")
    print("  ✓ Price momentum confirmation")
    print("  ✓ Volume surge detection")
    print("  ✓ Mean reversion signals")
    print("  ✓ Risk-managed position sizing")
    print("  ✓ Signal cooldown periods")


if __name__ == "__main__":
    main()
