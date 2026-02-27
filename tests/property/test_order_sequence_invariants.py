from __future__ import annotations

from datetime import datetime

from hypothesis import given, settings
from hypothesis import strategies as st

from ml4t.backtest import Broker, OrderSide
from ml4t.backtest.config import FillOrdering
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


def _scenario(fill_ordering: FillOrdering, price: float, qty: float) -> tuple[float, float]:
    broker = Broker(
        initial_cash=price * qty,
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        fill_ordering=fill_ordering,
        reject_on_insufficient_cash=True,
    )

    _set_bar(broker, price)
    broker.submit_order("AAPL", qty, OrderSide.BUY)
    broker._process_orders()

    # Re-enter before close submission: ordering decides whether re-entry can execute.
    broker.submit_order("AAPL", qty, OrderSide.BUY)
    broker.close_position("AAPL")
    broker._process_orders()

    pos = broker.get_position("AAPL")
    position_qty = 0.0 if pos is None else pos.quantity
    return position_qty, broker.get_account_value()


@settings(max_examples=60)
@given(
    price=st.floats(min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    qty=st.floats(min_value=0.1, max_value=200.0, allow_nan=False, allow_infinity=False),
)
def test_exit_first_never_underfills_vs_fifo(price: float, qty: float) -> None:
    exit_first_qty, exit_first_value = _scenario(FillOrdering.EXIT_FIRST, price, qty)
    fifo_qty, fifo_value = _scenario(FillOrdering.FIFO, price, qty)

    assert exit_first_qty >= fifo_qty
    assert abs(exit_first_value - fifo_value) < 1e-8
