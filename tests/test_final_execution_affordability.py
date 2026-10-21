"""Actual execution costs must not bypass the declared cash policy."""

from datetime import UTC, datetime

import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.config import ExecutionMode, ShareType, SlippageType
from ml4t.backtest.types import OrderStatus


def _run(
    *,
    partial=False,
    fractional=False,
    skip=False,
    leverage=False,
    fee=None,
    slippage="percentage",
    rate=0.01,
    price_model=None,
    impact=None,
    cash=1000,
):
    times = [datetime(2025, 1, day, 21, tzinfo=UTC) for day in [6, 7]]
    orders = []

    class Buy(Strategy):
        def on_start(self, broker):
            if price_model is not None:
                broker.slippage_model = price_model
            if fee is not None:
                broker.commission_model = fee
                broker.gatekeeper.commission_model = fee

        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                orders.append(broker.submit_order("TEST", 10))

    result = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {
                    "timestamp": times,
                    "asset": ["TEST"] * 2,
                    "close": [100.0] * 2,
                    "volume": [100.0] * 2,
                }
            )
        ),
        Buy(),
        BacktestConfig(
            calendar=None,
            initial_cash=cash,
            execution_mode=ExecutionMode.NEXT_BAR,
            slippage_type=SlippageType(slippage),
            slippage_rate=rate,
            partial_fills_allowed=partial,
            share_type=ShareType.FRACTIONAL if fractional else ShareType.INTEGER,
            skip_cash_validation=skip,
            allow_leverage=leverage,
        ),
        market_impact_model=impact,
    ).run()
    return result, orders[0]


def test_slippage_cannot_put_an_unleveraged_cash_account_into_debt():
    result, order = _run()
    assert result.to_fills_dataframe().is_empty()
    assert result.to_portfolio_state_dataframe()["cash"].to_list() == [1000.0, 1000.0]
    assert order.status is OrderStatus.REJECTED
    assert order.rejection_code == "insufficient_cash"


def test_integer_partial_fill_is_sized_at_the_final_execution_price():
    result, order = _run(partial=True)
    fills = result.to_fills_dataframe()
    assert fills["quantity"].to_list() == [9.0]
    assert fills["price"].to_list() == [101.0]
    assert result.to_portfolio_state_dataframe()["cash"].to_list() == [1000.0, 91.0]
    assert order.requested_quantity == 10 and order.filled_quantity == 9


def test_fractional_partial_fill_cannot_round_above_available_cash_after_slippage():
    result, _ = _run(partial=True, fractional=True)
    assert result.to_fills_dataframe()["quantity"].to_list() == pytest.approx([1000 / 101])
    assert result.to_portfolio_state_dataframe()["cash"][-1] >= 0


@pytest.mark.parametrize("setting", ["skip", "leverage"])
def test_explicit_unconstrained_or_leveraged_account_keeps_its_cost_behavior(setting):
    result, _ = _run(**{setting: True})
    assert result.to_fills_dataframe()["quantity"].to_list() == [10.0]
    assert result.to_portfolio_state_dataframe()["cash"][-1] == -10.0


def test_rejected_final_price_does_not_consume_a_stateful_commission():
    class Fee:
        calls = 0

        def calculate(self, asset, quantity, price):
            self.calls += 1
            return float(self.calls) if price > 100 else 0.0

    fee = Fee()
    result, order = _run(fee=fee)
    assert result.to_fills_dataframe().is_empty()
    assert order.status is OrderStatus.REJECTED
    assert fee.calls == 0


def test_partial_final_price_fee_quotes_only_commit_one_actual_commission():
    class Fee:
        calls = 0

        def calculate(self, asset, quantity, price):
            self.calls += 1
            return float(self.calls) if price > 100 else 0.0

    fee = Fee()
    result, _ = _run(partial=True, fee=fee)
    assert result.to_fills_dataframe()["quantity"].to_list() == [9.0]
    assert result.to_fills_dataframe()["commission"].to_list() == [1.0]
    assert result.to_portfolio_state_dataframe()["cash"][-1] == 90.0
    assert fee.calls == 1


def test_partial_fill_reprices_quantity_dependent_impact_after_cash_sizing():
    result, _ = _run(partial=True, slippage="cumulative_volume_share", rate=0.1)
    fills = result.to_fills_dataframe()
    assert fills["quantity"].to_list() == [9.0]
    assert fills["price"].to_list() == pytest.approx([100.081])
    assert result.to_portfolio_state_dataframe()["cash"][-1] == pytest.approx(99.271)


def test_final_validation_includes_market_impact_without_slippage():
    from ml4t.backtest.execution.impact import LinearImpact

    result, order = _run(slippage="none", impact=LinearImpact(coefficient=0.1))
    assert result.to_fills_dataframe().is_empty()
    assert order.status is OrderStatus.REJECTED
    assert result.to_portfolio_state_dataframe()["cash"][-1] == 1000


@pytest.mark.parametrize(
    "cash,partial,expected_calls", [(1000, False, 0), (1000, True, 1), (2000, False, 1)]
)
def test_price_model_estimates_do_not_advance_custom_model_execution_state(
    cash, partial, expected_calls
):
    class StatefulSlippage:
        calls = 0

        def calculate(self, asset, quantity, price, volume):
            self.calls += 1
            return float(self.calls)

    model = StatefulSlippage()
    result, _ = _run(cash=cash, partial=partial, price_model=model)
    assert model.calls == expected_calls
    if expected_calls:
        assert result.to_fills_dataframe()["price"].to_list() == [101.0]
    else:
        assert result.to_fills_dataframe().is_empty()
