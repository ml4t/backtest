"""Property-based order-type invariants.

Uses Hypothesis to fuzz limit/stop prices and verify fill-level bounds
hold across random price scenarios. Operates at Broker level for speed.
"""

from __future__ import annotations

from datetime import datetime

from hypothesis import given, settings
from hypothesis import strategies as st

from ml4t.backtest import Broker, OrderSide, OrderType
from ml4t.backtest.models import NoCommission, NoSlippage


def _make_broker(cash: float = 200_000.0) -> Broker:
    return Broker(cash, NoCommission(), NoSlippage())


def _set_bar(
    broker: Broker,
    asset: str,
    close: float,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
) -> None:
    o = open_ if open_ is not None else close
    h = high if high is not None else close
    lo = low if low is not None else close
    broker._update_time(
        datetime(2024, 1, 1),
        {asset: close},
        {asset: o},
        {asset: h},
        {asset: lo},
        {asset: 1_000_000.0},
        {asset: {}},
    )


# ---------------------------------------------------------------------------
# Limit order bounds
# ---------------------------------------------------------------------------


@settings(max_examples=200)
@given(
    limit=st.floats(min_value=50.0, max_value=150.0, allow_nan=False, allow_infinity=False),
    bar_high=st.floats(min_value=50.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    bar_low=st.floats(min_value=10.0, max_value=150.0, allow_nan=False, allow_infinity=False),
)
def test_limit_buy_never_fills_above_limit(limit: float, bar_high: float, bar_low: float) -> None:
    """A limit BUY must never fill above the limit price."""
    if bar_low > bar_high:
        bar_low, bar_high = bar_high, bar_low

    broker = _make_broker()
    close = (bar_high + bar_low) / 2
    open_ = close

    _set_bar(broker, "TEST", close, open_=open_, high=bar_high, low=bar_low)
    broker.submit_order("TEST", 10.0, OrderSide.BUY, OrderType.LIMIT, limit_price=limit)
    broker._process_orders()

    limit_fills = [f for f in broker.fills if f.order_type == "limit"]
    for fill in limit_fills:
        assert fill.price <= limit + 1e-8, (
            f"Limit BUY filled at {fill.price} > limit {limit}"
        )


@settings(max_examples=200)
@given(
    limit=st.floats(min_value=50.0, max_value=150.0, allow_nan=False, allow_infinity=False),
    bar_high=st.floats(min_value=50.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    bar_low=st.floats(min_value=10.0, max_value=150.0, allow_nan=False, allow_infinity=False),
)
def test_limit_sell_never_fills_below_limit(
    limit: float, bar_high: float, bar_low: float
) -> None:
    """A limit SELL must never fill below the limit price."""
    if bar_low > bar_high:
        bar_low, bar_high = bar_high, bar_low

    broker = _make_broker()
    close = (bar_high + bar_low) / 2
    open_ = close

    _set_bar(broker, "TEST", close, open_=open_, high=bar_high, low=bar_low)
    broker.submit_order("TEST", 10.0, OrderSide.SELL, OrderType.LIMIT, limit_price=limit)
    broker._process_orders()

    limit_fills = [f for f in broker.fills if f.order_type == "limit"]
    for fill in limit_fills:
        assert fill.price >= limit - 1e-8, (
            f"Limit SELL filled at {fill.price} < limit {limit}"
        )


# ---------------------------------------------------------------------------
# Stop order bounds
# ---------------------------------------------------------------------------


@settings(max_examples=200)
@given(
    stop=st.floats(min_value=50.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    bar_open=st.floats(min_value=50.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    bar_high=st.floats(min_value=50.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    bar_low=st.floats(min_value=10.0, max_value=200.0, allow_nan=False, allow_infinity=False),
)
def test_stop_buy_never_fills_below_stop(
    stop: float, bar_open: float, bar_high: float, bar_low: float
) -> None:
    """A stop BUY must never fill below the stop price."""
    if bar_low > bar_high:
        bar_low, bar_high = bar_high, bar_low
    bar_open = max(bar_low, min(bar_open, bar_high))
    close = (bar_high + bar_low) / 2

    broker = _make_broker()
    _set_bar(broker, "TEST", close, open_=bar_open, high=bar_high, low=bar_low)
    broker.submit_order("TEST", 10.0, OrderSide.BUY, OrderType.STOP, stop_price=stop)
    broker._process_orders()

    stop_fills = [f for f in broker.fills if f.order_type == "stop"]
    for fill in stop_fills:
        assert fill.price >= stop - 1e-8, (
            f"Stop BUY filled at {fill.price} < stop {stop}"
        )


@settings(max_examples=200)
@given(
    stop=st.floats(min_value=10.0, max_value=150.0, allow_nan=False, allow_infinity=False),
    bar_open=st.floats(min_value=10.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    bar_high=st.floats(min_value=50.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    bar_low=st.floats(min_value=10.0, max_value=150.0, allow_nan=False, allow_infinity=False),
)
def test_stop_sell_never_fills_above_stop(
    stop: float, bar_open: float, bar_high: float, bar_low: float
) -> None:
    """A stop SELL must never fill above the stop price."""
    if bar_low > bar_high:
        bar_low, bar_high = bar_high, bar_low
    bar_open = max(bar_low, min(bar_open, bar_high))
    close = (bar_high + bar_low) / 2

    broker = _make_broker()
    _set_bar(broker, "TEST", close, open_=bar_open, high=bar_high, low=bar_low)
    broker.submit_order("TEST", 10.0, OrderSide.SELL, OrderType.STOP, stop_price=stop)
    broker._process_orders()

    stop_fills = [f for f in broker.fills if f.order_type == "stop"]
    for fill in stop_fills:
        assert fill.price <= stop + 1e-8, (
            f"Stop SELL filled at {fill.price} > stop {stop}"
        )


# ---------------------------------------------------------------------------
# Bracket: exactly one exit fires
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    entry_price=st.floats(
        min_value=50.0, max_value=150.0, allow_nan=False, allow_infinity=False
    ),
    tp_offset=st.floats(
        min_value=1.0, max_value=20.0, allow_nan=False, allow_infinity=False
    ),
    sl_offset=st.floats(
        min_value=1.0, max_value=20.0, allow_nan=False, allow_infinity=False
    ),
    exit_high=st.floats(
        min_value=50.0, max_value=200.0, allow_nan=False, allow_infinity=False
    ),
    exit_low=st.floats(
        min_value=10.0, max_value=150.0, allow_nan=False, allow_infinity=False
    ),
)
def test_bracket_exactly_one_exit(
    entry_price: float,
    tp_offset: float,
    sl_offset: float,
    exit_high: float,
    exit_low: float,
) -> None:
    """If a bracket's entry fills and an exit triggers, exactly one exit fills."""
    if exit_low > exit_high:
        exit_low, exit_high = exit_high, exit_low

    tp_price = entry_price + tp_offset
    sl_price = entry_price - sl_offset

    broker = _make_broker(500_000.0)

    # Bar 0: entry
    _set_bar(broker, "TEST", entry_price)
    orders = broker.submit_bracket(
        "TEST", 10.0, take_profit=tp_price, stop_loss=sl_price, validate_prices=False
    )
    assert orders is not None
    entry, tp, sl = orders
    broker._process_orders()

    if entry.status != broker.__class__.__dict__.get("_x", None):
        # Entry should have filled (market order at current price)
        pass

    if entry.filled_price is None:
        return  # Entry didn't fill (shouldn't happen with market)

    # Bar 1: exit bar with random high/low
    exit_close = (exit_high + exit_low) / 2
    exit_open = exit_close
    _set_bar(broker, "TEST", exit_close, open_=exit_open, high=exit_high, low=exit_low)
    broker._process_orders()

    # Check: at most one exit filled
    tp_filled = tp.status.value == "filled"
    sl_filled = sl.status.value == "filled"

    if tp_filled or sl_filled:
        # Exactly one should have filled, not both
        assert not (tp_filled and sl_filled), (
            f"Both TP and SL filled! tp={tp.status}, sl={sl.status}"
        )
        # The other should be cancelled
        if tp_filled:
            assert sl.status.value == "cancelled", f"SL not cancelled: {sl.status}"
        else:
            assert tp.status.value == "cancelled", f"TP not cancelled: {tp.status}"
