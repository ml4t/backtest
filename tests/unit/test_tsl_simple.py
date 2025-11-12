"""Simple TSL test to verify peak tracking."""

from datetime import datetime
from qengine.core.event import MarketEvent
from qengine.core.types import OrderSide, OrderType, MarketDataType
from qengine.execution.broker import SimulationBroker
from qengine.execution.order import Order


def test_tsl_tracks_peak():
    """Verify TSL tracks peak price, not current price."""
    broker = SimulationBroker(initial_cash=10000.0, execution_delay=False)

    # Entry at 100
    entry_order = Order(
        asset_id="BTC",
        order_type=OrderType.BRACKET,
        side=OrderSide.BUY,
        quantity=1.0,
        tsl_pct=0.01,  # 1% TSL
        metadata={"base_price": 100.0},
    )
    broker.submit_order(entry_order)

    # Fill entry
    event_t0 = MarketEvent(
        timestamp=datetime.now(),
        asset_id="BTC",
        data_type=MarketDataType.BAR,
        open=100.0,
        high=100.5,
        low=99.5,
        close=100.0,
        volume=1000.0,
    )
    broker.on_market_event(event_t0)

    # Get TSL order
    tsl_orders = broker._trailing_stops.get("BTC", [])
    assert len(tsl_orders) == 1, f"Expected 1 TSL order, got {len(tsl_orders)}"
    tsl_order = tsl_orders[0]

    # T1: Price rises to 105 (new peak)
    event_t1 = MarketEvent(
        timestamp=datetime.now(),
        asset_id="BTC",
        data_type=MarketDataType.BAR,
        open=104.0,
        high=105.0,  # Peak
        low=103.0,
        close=104.0,
        volume=1000.0,
    )
    broker.on_market_event(event_t1)

    # Peak should be 105
    peak = tsl_order.metadata.get("peak_price")
    print(f"Peak after T1: {peak}")
    assert peak == 105.0, f"Peak should be 105, got {peak}"

    # TSL should be 103.95
    tsl_level = tsl_order.trailing_stop_price
    print(f"TSL level after T1: {tsl_level}")
    expected_tsl = 105.0 * 0.99  # 103.95
    assert abs(tsl_level - expected_tsl) < 0.01, \
        f"TSL should be {expected_tsl}, got {tsl_level}"

    # T2: Price falls to 102 (below peak)
    event_t2 = MarketEvent(
        timestamp=datetime.now(),
        asset_id="BTC",
        data_type=MarketDataType.BAR,
        open=103.0,
        high=103.5,
        low=101.5,
        close=102.0,
        volume=1000.0,
    )
    broker.on_market_event(event_t2)

    # Peak should STAY at 105
    peak_after = tsl_order.metadata.get("peak_price")
    print(f"Peak after T2 (price fell): {peak_after}")
    assert peak_after == 105.0, f"Peak should NOT drop, expected 105, got {peak_after}"

    # TSL should STAY at 103.95
    tsl_after = tsl_order.trailing_stop_price
    print(f"TSL level after T2 (price fell): {tsl_after}")
    assert abs(tsl_after - expected_tsl) < 0.01, \
        f"TSL should remain at {expected_tsl}, got {tsl_after}"

    print("✅ TSL correctly tracks peak, not current price!")


if __name__ == "__main__":
    test_tsl_tracks_peak()
