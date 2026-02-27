from __future__ import annotations

from datetime import datetime

from hypothesis import given, settings
from hypothesis import strategies as st

from ml4t.backtest import Broker, OrderSide
from ml4t.backtest.models import NoCommission, NoSlippage


def _set_bar(broker: Broker, price: float) -> None:
    ts = datetime(2024, 1, 1)
    broker._update_time(
        ts,
        {"AAPL": price},
        {"AAPL": price},
        {"AAPL": price},
        {"AAPL": price},
        {"AAPL": 1_000_000.0},
        {"AAPL": {}},
    )


@settings(max_examples=60)
@given(
    entry=st.floats(min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    exit_=st.floats(min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    qty=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
)
def test_round_trip_pnl_reconciles_cash(entry: float, exit_: float, qty: float) -> None:
    initial_cash = 200_000.0
    broker = Broker(initial_cash, NoCommission(), NoSlippage())

    _set_bar(broker, entry)
    broker.submit_order("AAPL", qty, OrderSide.BUY)
    broker._process_orders()

    _set_bar(broker, exit_)
    broker.close_position("AAPL")
    broker._process_orders()

    assert broker.get_position("AAPL") is None
    assert broker.trades

    trade = broker.trades[-1]
    expected_pnl = (exit_ - entry) * qty
    assert abs(trade.pnl - expected_pnl) < 1e-8
    assert abs((initial_cash + expected_pnl) - broker.cash) < 1e-8
    assert abs(broker.get_account_value() - broker.cash) < 1e-8

