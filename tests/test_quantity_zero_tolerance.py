"""Quantity-zero behavior at realistic position scales."""

import math
from datetime import datetime

import polars as pl
import pytest

from ml4t.backtest import (
    BacktestConfig,
    DataFeed,
    Engine,
    ExecutionMode,
    OrderSide,
    Strategy,
)
from ml4t.backtest.config import ShareType
from ml4t.backtest.core.order_book import OrderBook
from ml4t.backtest.core.shared import (
    QUANTITY_ZERO_FLOOR,
    QUANTITY_ZERO_ULPS,
    quantity_zero_tolerance,
)

POSITION_QUANTITY = 10_927.322882


class _CloseLargePosition(Strategy):
    def __init__(
        self,
        opening_side: OrderSide,
        *,
        closing_quantity: float | None = None,
    ) -> None:
        self.opening_side = opening_side
        self.closing_quantity = closing_quantity or math.nextafter(POSITION_QUANTITY, math.inf)

    def on_data(self, timestamp, data, context, broker) -> None:
        quantity = POSITION_QUANTITY
        if timestamp.day == 2:
            broker.submit_order("AAA", quantity, self.opening_side)
        elif timestamp.day == 3:
            closing_side = OrderSide.SELL if self.opening_side is OrderSide.BUY else OrderSide.BUY
            broker.submit_order(
                "AAA",
                self.closing_quantity,
                closing_side,
            )


def _prices(days: tuple[int, ...] = (2, 3)) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, day) for day in days],
            "asset": ["AAA"] * len(days),
            "open": [10.0] * len(days),
            "high": [10.0] * len(days),
            "low": [10.0] * len(days),
            "close": [10.0] * len(days),
            "volume": [1e12] * len(days),
        }
    )


def _config(execution_mode: ExecutionMode = ExecutionMode.SAME_BAR) -> BacktestConfig:
    return BacktestConfig(
        initial_cash=1e8,
        execution_mode=execution_mode,
        immediate_fill=execution_mode is ExecutionMode.SAME_BAR,
        allow_short_selling=True,
        allow_leverage=True,
        share_type=ShareType.FRACTIONAL,
        next_bar_queue_shadow_validation=True,
    )


@pytest.mark.parametrize("opening_side", [OrderSide.BUY, OrderSide.SELL])
def test_large_position_close_does_not_leave_a_residual_position(
    opening_side: OrderSide,
) -> None:
    result = Engine(
        DataFeed(prices_df=_prices()),
        _CloseLargePosition(opening_side),
        _config(),
    ).run()

    portfolio_state = result.to_portfolio_state_dataframe()
    assert portfolio_state["open_positions"].to_list() == [1, 0]
    assert result.to_trades_dataframe()["status"].to_list() == ["closed"]


@pytest.mark.parametrize("opening_side", [OrderSide.BUY, OrderSide.SELL])
def test_next_bar_shadow_validation_uses_the_same_zero_contract(
    opening_side: OrderSide,
) -> None:
    result = Engine(
        DataFeed(prices_df=_prices((2, 3, 4))),
        _CloseLargePosition(opening_side),
        _config(ExecutionMode.NEXT_BAR),
    ).run()

    assert result.to_portfolio_state_dataframe()["open_positions"].to_list() == [0, 1, 0]


def test_tolerance_scales_with_the_cancelled_operands() -> None:
    residual = math.ulp(POSITION_QUANTITY)

    assert residual > QUANTITY_ZERO_FLOOR
    assert quantity_zero_tolerance(-POSITION_QUANTITY, POSITION_QUANTITY) == (
        QUANTITY_ZERO_ULPS * residual
    )


def test_nonfinite_operands_cannot_widen_the_tolerance() -> None:
    assert quantity_zero_tolerance(math.inf, math.nan) == QUANTITY_ZERO_FLOOR


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize(
    ("residual_ulps", "expected_closed"),
    [(1, True), (QUANTITY_ZERO_ULPS, True), (QUANTITY_ZERO_ULPS + 1, False)],
)
def test_order_simulation_and_fill_use_the_same_boundary(
    sign: float,
    residual_ulps: int,
    expected_closed: bool,
) -> None:
    old_quantity = sign * POSITION_QUANTITY
    closing_quantity = -sign * (POSITION_QUANTITY + residual_ulps * math.ulp(POSITION_QUANTITY))

    simulated_quantity, _, _, _ = OrderBook._simulate_position_update(
        old_quantity,
        10.0,
        closing_quantity,
        10.0,
    )

    assert (simulated_quantity == 0.0) is expected_closed


@pytest.mark.parametrize("opening_side", [OrderSide.BUY, OrderSide.SELL])
def test_genuine_small_remaining_position_is_not_erased(opening_side: OrderSide) -> None:
    closing_quantity = POSITION_QUANTITY - 1e-9
    result = Engine(
        DataFeed(prices_df=_prices()),
        _CloseLargePosition(opening_side, closing_quantity=closing_quantity),
        _config(),
    ).run()

    assert result.to_portfolio_state_dataframe()["open_positions"].to_list() == [1, 1]
    assert result.to_trades_dataframe()["status"].to_list() == ["partial", "open"]
