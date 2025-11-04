"""Debug TSL order creation."""

from datetime import datetime
from qengine.core.event import MarketEvent
from qengine.core.types import OrderSide, OrderType, MarketDataType
from qengine.execution.broker import SimulationBroker
from qengine.execution.order import Order


def test_tsl_debug():
    broker = SimulationBroker(initial_cash=10000.0)

    # Entry
    entry_order = Order(
        asset_id="BTC",
        order_type=OrderType.BRACKET,
        side=OrderSide.BUY,
        quantity=1.0,
        tsl_pct=0.01,
        metadata={"base_price": 100.0},
    )
    print(f"Entry order: {entry_order}")
    print(f"Order type: {entry_order.order_type}")
    print(f"TSL pct: {entry_order.tsl_pct}")
    
    broker.submit_order(entry_order)
    
    print(f"\nAfter submit:")
    print(f"Order state: {entry_order.state}")
    print(f"Order status: {entry_order.status}")
    print(f"Open orders: {len(broker._open_orders.get('BTC', []))}")
    
    # Fill entry
    event = MarketEvent(
        timestamp=datetime.now(),
        asset_id="BTC",
        data_type=MarketDataType.BAR,
        open=100.0,
        high=100.5,
        low=99.5,
        close=100.0,
        volume=1000.0,
    )
    broker.on_market_event(event)
    
    print(f"\nAfter market event:")
    print(f"Order filled: {entry_order.is_filled}")
    print(f"Filled quantity: {entry_order.filled_quantity}")
    print(f"Open orders: {len(broker._open_orders.get('BTC', []))}")
    print(f"Trailing stops: {len(broker._trailing_stops.get('BTC', []))}")
    print(f"Position: {broker.position_tracker.get_position('BTC')}")
    
    if broker._trailing_stops.get("BTC"):
        for tsl in broker._trailing_stops["BTC"]:
            print(f"\nTSL order:")
            print(f"  Type: {tsl.order_type}")
            print(f"  Side: {tsl.side}")
            print(f"  Quantity: {tsl.quantity}")
            print(f"  Trail %: {tsl.trail_percent}")
            print(f"  Metadata: {tsl.metadata}")


if __name__ == "__main__":
    test_tsl_debug()
