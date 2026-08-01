from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from ml4t.backtest import AssetClass, Broker, ContractSpec, OrderSide, OrderStatus
from ml4t.backtest.config import ShareType
from ml4t.backtest.models import NoSlippage, PercentageCommission


def _execute(broker: Broker, *, day: int, price: float, quantity: float) -> None:
    timestamp = datetime(2024, 1, 1) + timedelta(days=day)
    broker._update_time(
        timestamp=timestamp,
        prices={"TEST": price},
        opens={"TEST": price},
        highs={"TEST": price},
        lows={"TEST": price},
        volumes={"TEST": 1_000_000.0},
        signals={},
    )
    side = OrderSide.BUY if quantity > 0 else OrderSide.SELL
    order = broker.submit_order("TEST", abs(quantity), side)
    assert order is not None
    broker._process_orders()
    assert order.status == OrderStatus.FILLED


@pytest.mark.parametrize(
    ("share_type", "contract_specs", "steps", "expected_closed", "expected_remaining"),
    [
        (
            ShareType.INTEGER,
            None,
            [(100.0, 100.0), (110.0, -30.0), (90.0, -20.0)],
            2,
            50.0,
        ),
        (
            ShareType.FRACTIONAL,
            None,
            [(100.0, -10.5), (90.0, 3.25), (110.0, 2.0)],
            2,
            -5.25,
        ),
        (
            ShareType.FRACTIONAL,
            None,
            [(100.0, 10.0), (110.0, 5.0), (120.0, -6.0)],
            1,
            9.0,
        ),
        (
            ShareType.INTEGER,
            None,
            [(100.0, 10.0), (120.0, -15.0)],
            1,
            -5.0,
        ),
        (
            ShareType.INTEGER,
            {"TEST": ContractSpec("TEST", AssetClass.FUTURE, multiplier=50.0)},
            [(4_000.0, 4.0), (4_010.0, -1.0)],
            1,
            3.0,
        ),
    ],
)
def test_fill_costs_are_conserved_across_realized_and_residual_positions(
    share_type: ShareType,
    contract_specs: dict[str, ContractSpec] | None,
    steps: list[tuple[float, float]],
    expected_closed: int,
    expected_remaining: float,
) -> None:
    broker = Broker(
        initial_cash=10_000_000.0,
        commission_model=PercentageCommission(0.01),
        slippage_model=NoSlippage(),
        allow_short_selling=True,
        allow_leverage=True,
        share_type=share_type,
        contract_specs=contract_specs,
    )

    for day, (price, quantity) in enumerate(steps):
        _execute(broker, day=day, price=price, quantity=quantity)

    position = broker.get_position("TEST")
    assert position is not None
    assert position.quantity == pytest.approx(expected_remaining)
    assert len(broker.trades) == expected_closed

    fill_costs = sum(fill.commission for fill in broker.fills)
    realized_costs = sum(trade.fees for trade in broker.trades)
    residual_costs = position.entry_commission
    assert realized_costs + residual_costs == pytest.approx(fill_costs, abs=1e-9)

    for trade in broker.trades:
        assert trade.gross_pnl - trade.fees == pytest.approx(trade.pnl, abs=1e-9)
