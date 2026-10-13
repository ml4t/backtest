"""Prediction-market venue fees against the published schedules and worked examples."""

from copy import deepcopy
from datetime import datetime, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Decimal

import numpy as np
import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.models import CommissionModel, estimate_commission
from ml4t.backtest.prediction_market_fees import (
    ForecastExCommission,
    KalshiCommission,
    PolymarketUSCommission,
    forecastex_fee,
    kalshi_fee,
    kalshi_order_fees,
    polymarket_us_fee,
)
from ml4t.backtest.types import OrderSide

CENT_PRICES = [k / 100 for k in range(1, 100)]


def _kalshi_reference(cents: int, contracts: int, side: str, precision: str | None) -> Decimal:
    """Kalshi fee in exact decimal arithmetic, following the fee-rounding page step by step."""
    price = Decimal(cents) / 100
    trade_fee = (Decimal("0.07") * contracts * price * (1 - price)).quantize(
        Decimal("0.000001"), ROUND_CEILING
    )
    if precision is None:
        return trade_fee
    revenue = price * contracts * (-1 if side == "buy" else 1)
    aligned = (revenue - trade_fee).quantize(Decimal(precision), ROUND_FLOOR)
    return revenue - aligned


# === Kalshi: formula ===


def test_kalshi_taker_fee_matches_the_schedule_peak_and_tail():
    # Guards a wrong rate: 0.07 x P x (1 - P) is 0.0175 at 0.50 and 0.0063 at 0.90.
    assert kalshi_fee(0.50, 100).item() == 1.75
    assert kalshi_fee(0.50, 1).item() == 0.0175
    assert kalshi_fee(0.90, 1).item() == 0.0063


def test_kalshi_fee_is_symmetric_between_yes_and_no_prices():
    # Buying NO at q is a purchase at q; the fee must not depend on which side is YES.
    prices = np.array(CENT_PRICES)
    np.testing.assert_array_equal(kalshi_fee(prices, 37), kalshi_fee(1.0 - prices, 37))


def test_kalshi_series_fee_multiplier_scales_the_fee():
    # Guards a dropped multiplier M, scalar or joined per row from series metadata.
    assert kalshi_fee(0.50, 100, fee_multiplier=2.0).item() == 3.5
    np.testing.assert_array_equal(
        kalshi_fee([0.50, 0.50], [100, 100], fee_multiplier=[1.0, 0.5]), [1.75, 0.875]
    )


@pytest.mark.parametrize(
    ("fee_type", "expected"),
    [
        ("quadratic", 0.0),
        ("quadratic_with_maker_fees", 0.4375),
        ("quadratic_with_combo_maker_fees", 0.875),
    ],
)
def test_kalshi_maker_fee_follows_series_fee_type(fee_type, expected):
    # Guards charging makers on taker-only series and using the wrong maker factor.
    assert kalshi_fee(0.50, 100, liquidity="maker", fee_type=fee_type).item() == expected
    assert kalshi_fee(0.50, 100, liquidity="taker", fee_type=fee_type).item() == 1.75


def test_kalshi_per_row_liquidity_and_fee_type_from_polars():
    fees = kalshi_fee(
        pl.Series([0.50, 0.50, 0.50]),
        pl.Series([100, 100, 100]),
        liquidity=pl.Series(["taker", "maker", "maker"]),
        fee_type=pl.Series(["quadratic", "quadratic", "quadratic_with_maker_fees"]),
    )
    np.testing.assert_array_equal(fees, [1.75, 0.0, 0.4375])


# === Kalshi: rounding ===


def test_kalshi_trade_fee_rounds_up_to_a_micro_dollar():
    # 0.07 x 0.055 x 0.945 = 0.00363825: rounding up gives 0.003639, down 0.003638.
    assert kalshi_fee(0.055, 1).item() == 0.003639


def test_kalshi_exact_fee_on_the_micro_grid_is_not_bumped_by_float_noise():
    # 0.07 x 100 x 0.25 evaluates to 1.7500000000000002; a naive ceiling charges 1.750001.
    assert kalshi_fee(0.50, 100).item() == 1.75


def test_kalshi_cent_rounding_applies_to_the_fill_not_each_contract():
    # 100 contracts at 0.50 cost 50.00 + 1.75 = 51.75, already on the cent grid.
    # Rounding each contract's 0.0175 to 0.02 would charge 2.00.
    assert kalshi_fee(0.50, 100, rounding="cent").item() == 1.75
    assert kalshi_fee(0.50, 1, rounding="cent").item() == 0.02
    assert kalshi_fee(0.50, 1, rounding="exact").item() == 0.0175


def test_kalshi_cent_rounding_matches_the_published_fcm_example():
    # docs.kalshi.com fee_rounding: buy, revenue -0.055, trade fee 0.003639, total 0.005.
    assert kalshi_fee(0.055, 1, side="buy", rounding="cent").item() == 0.005


def test_kalshi_direct_member_precision_is_a_hundredth_of_a_cent():
    # -0.055 - 0.003639 aligns down to -0.0587 at 0.0001 precision: fee 0.0037.
    assert kalshi_fee(0.055, 1, rounding="direct").item() == 0.0037
    assert kalshi_fee(0.50, 1, rounding="direct").item() == 0.0175


@pytest.mark.parametrize("rounding", ["exact", "cent", "direct"])
@pytest.mark.parametrize("side", ["buy", "sell"])
def test_kalshi_fee_matches_decimal_reference_over_the_price_grid(rounding, side):
    precision = {"exact": None, "cent": "0.01", "direct": "0.0001"}[rounding]
    for contracts in (1, 3, 7, 100, 2500):
        expected = [
            float(_kalshi_reference(cents, contracts, side, precision)) for cents in range(1, 100)
        ]
        actual = kalshi_fee(CENT_PRICES, contracts, side=side, rounding=rounding)
        np.testing.assert_array_equal(actual, expected)


def test_kalshi_order_accumulator_rebates_whole_cents_of_overpayment():
    # Each 1-contract fill at 0.10 has trade fee 0.0063 and rounding fee 0.0037.
    # After the third fill the accumulator holds 0.0111: 0.01 is rebated, 0.0011 carries.
    fills = kalshi_order_fees([0.10, 0.10, 0.10], [1, 1, 1], rounding="cent")
    np.testing.assert_allclose(fills, [0.01, 0.01, 0.0], rtol=0, atol=1e-12)
    single_fill = kalshi_fee(0.10, 3, rounding="cent").item()
    assert fills.sum() == pytest.approx(single_fill, abs=1e-12)


def test_kalshi_order_without_alignment_has_no_rebate():
    fills = kalshi_order_fees([0.10, 0.10, 0.10], [1, 1, 1], rounding="exact")
    np.testing.assert_array_equal(fills, [0.0063, 0.0063, 0.0063])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"price": 1.2}, "price"),
        ({"price": float("nan")}, "price"),
        ({"contracts": -1}, "contracts"),
        ({"fee_type": "flat"}, "flat"),
        ({"fee_type": "linear"}, "fee_type"),
        ({"liquidity": "passive"}, "liquidity"),
        ({"side": "short"}, "side"),
        ({"fee_multiplier": -1.0}, "fee_multiplier"),
        ({"rounding": "up"}, "rounding"),
    ],
)
def test_kalshi_rejects_inputs_outside_the_schedule(kwargs, message):
    arguments = {"price": 0.5, "contracts": 1} | kwargs
    price = arguments.pop("price")
    contracts = arguments.pop("contracts")
    with pytest.raises(ValueError, match=message):
        kalshi_fee(price, contracts, **arguments)


# === ForecastEx ===


def test_forecastex_fee_is_one_cent_per_contract_at_any_price():
    np.testing.assert_array_equal(
        forecastex_fee([0.05, 0.50, 0.95], [1, 10, 100]), [0.01, 0.1, 1.0]
    )


def test_forecastex_embedded_fee_era_can_be_priced_at_zero():
    assert forecastex_fee(0.5, 10, exchange_fee=0.0).item() == 0.0


# === Polymarket US ===


def test_polymarket_us_taker_fee_matches_the_schedule():
    assert polymarket_us_fee(0.50, 1).item() == pytest.approx(0.017375, abs=1e-15)
    assert polymarket_us_fee(0.50, 1000, rounding="cent").item() == 17.38


def test_polymarket_us_maker_rebate_is_a_negative_fee():
    # Published example: maker at 0.10 on 1,000 contracts receives 1.12.
    assert polymarket_us_fee(0.10, 1000, liquidity="maker", rounding="cent").item() == -1.12


def test_polymarket_us_cent_rounding_is_half_to_even_on_exact_values():
    # 0.0695 x 1000 x 0.3 x 0.7 = 14.595 exactly; in floats 14.594999999999997,
    # which a naive round(x, 2) turns into 14.59. 6.255 rounds to 6.26 (even).
    fees = polymarket_us_fee([0.30, 0.10, 0.90], 1000, rounding="cent")
    np.testing.assert_array_equal(fees, [14.60, 6.26, 6.26])
    # 0.125 rounds down to the even cent, where half-up would give 0.13.
    assert polymarket_us_fee(0.5, 1, fee_coefficient=0.5, rounding="cent").item() == 0.12


def test_polymarket_us_cent_rounding_matches_decimal_reference():
    for contracts in (1, 10, 100, 1000, 12345):
        expected = [
            float(
                (
                    Decimal("0.0695") * contracts * Decimal(k) / 100 * (1 - Decimal(k) / 100)
                ).quantize(Decimal("0.01"), ROUND_HALF_EVEN)
            )
            for k in range(1, 100)
        ]
        np.testing.assert_array_equal(
            polymarket_us_fee(CENT_PRICES, contracts, rounding="cent"), expected
        )


def test_polymarket_us_uses_the_per_market_fee_coefficient():
    fees = polymarket_us_fee([0.5, 0.5], [100, 100], fee_coefficient=[0.0695, 0.10])
    np.testing.assert_allclose(fees, [1.7375, 2.5], rtol=0, atol=1e-12)


# === CommissionModel adapters ===


@pytest.mark.parametrize(
    ("model", "quantity", "price", "expected"),
    [
        (KalshiCommission(), 100, 0.50, 1.75),
        (KalshiCommission(rounding="cent"), 1, 0.50, 0.02),
        (KalshiCommission(rounding="cent"), -1, 0.055, 0.005),
        (KalshiCommission(liquidity="maker"), 100, 0.50, 0.0),
        (
            KalshiCommission(liquidity="maker", fee_type="quadratic_with_maker_fees"),
            100,
            0.5,
            0.4375,
        ),
        (KalshiCommission(fee_multiplier=2.0), 100, 0.50, 3.5),
        (ForecastExCommission(), -25, 0.40, 0.25),
        (PolymarketUSCommission(rounding="cent"), 1000, 0.50, 17.38),
        (PolymarketUSCommission(liquidity="maker"), 1000, 0.10, 0.0),
    ],
)
def test_commission_models_charge_the_venue_fee(model, quantity, price, expected):
    assert isinstance(model, CommissionModel)
    assert model.calculate("EVT", quantity, price) == pytest.approx(expected, abs=1e-12)
    assert estimate_commission(deepcopy(model), "EVT", quantity, price) == pytest.approx(
        expected, abs=1e-12
    )


def test_commission_model_rejects_invalid_configuration_at_construction():
    with pytest.raises(ValueError, match="flat"):
        KalshiCommission(fee_type="flat")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rounding"):
        PolymarketUSCommission(rounding="direct")  # type: ignore[arg-type]


def test_engine_charges_kalshi_cent_fee_on_each_fill():
    class BuyThenSell(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == datetime(2024, 1, 1):
                broker.submit_order("EVT", 1, OrderSide.BUY)
            elif timestamp == datetime(2024, 1, 3):
                broker.submit_order("EVT", 100, OrderSide.BUY)

    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)]
    prices = pl.DataFrame(
        {
            "timestamp": dates,
            "asset": ["EVT"] * len(dates),
            "open": [0.5] * len(dates),
            "high": [0.5] * len(dates),
            "low": [0.5] * len(dates),
            "close": [0.5] * len(dates),
            "volume": [100000] * len(dates),
        }
    )
    engine = Engine(
        feed=DataFeed(prices_df=prices),
        strategy=BuyThenSell(),
        config=BacktestConfig(initial_cash=1000),
    )
    engine.broker.commission_model = KalshiCommission(rounding="cent")
    result = engine.run()

    assert [fill.commission for fill in result.fills] == [0.02, 1.75]


def test_negative_fee_parameters_are_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        forecastex_fee(0.5, 1, exchange_fee=-0.01)
    with pytest.raises(ValueError, match="non-negative"):
        polymarket_us_fee(0.5, 1, maker_rebate_coefficient=-0.0125)


def test_kalshi_order_fees_takes_one_order_as_one_dimension():
    with pytest.raises(ValueError, match="1-D"):
        kalshi_order_fees([[0.5, 0.5]], [[1, 1]])
