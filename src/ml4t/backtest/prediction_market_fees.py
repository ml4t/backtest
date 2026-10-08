"""Trading-fee schedules for prediction-market venues.

Each venue has a vectorized function for research over arrays of fills and a
``CommissionModel`` for the engine:

| Venue | Function | Model |
|---|---|---|
| Kalshi | `kalshi_fee`, `kalshi_order_fees` | `KalshiCommission` |
| ForecastEx (via IBKR) | `forecastex_fee` | `ForecastExCommission` |
| Polymarket US | `polymarket_us_fee` | `PolymarketUSCommission` |

Prices are dollars per contract in ``[0, 1]`` and contracts pay 1.00 at
settlement. Buying NO at price ``q`` is a purchase at ``q``; the quadratic
formulas use ``q * (1 - q)``, which is symmetric, so a fee needs no YES/NO flag.
Contracts are non-negative counts. ``side`` (``"buy"``/``"sell"``) matters only
where a venue's rounding depends on the cash moved (Kalshi balance alignment).
``liquidity`` is ``"taker"`` (the fill removed a resting order) or ``"maker"``
(the fill executed a resting order).

The vectorized functions accept scalars, NumPy arrays, Python sequences and
Polars Series, broadcast them against each other and return a float64
``numpy.ndarray`` of fees in dollars (0-d for all-scalar input).

Fee figures were read from the venues' published schedules on 2026-10-08. They
change; the module constants name the value in force on that date.
"""

from typing import Any, Literal

import numpy as np

__all__ = [
    "FORECASTEX_EXCHANGE_FEE",
    "KALSHI_MAKER_FEE_FACTORS",
    "KALSHI_TAKER_RATE",
    "POLYMARKET_US_MAKER_REBATE_COEFFICIENT",
    "POLYMARKET_US_TAKER_COEFFICIENT",
    "ForecastExCommission",
    "KalshiCommission",
    "KalshiFeeType",
    "KalshiRounding",
    "Liquidity",
    "PolymarketUSCommission",
    "PolymarketUSRounding",
    "Side",
    "forecastex_fee",
    "kalshi_fee",
    "kalshi_order_fees",
    "polymarket_us_fee",
]

Liquidity = Literal["taker", "maker"]
Side = Literal["buy", "sell"]
KalshiFeeType = Literal["quadratic", "quadratic_with_maker_fees", "quadratic_with_combo_maker_fees"]
KalshiRounding = Literal["exact", "cent", "direct"]
PolymarketUSRounding = Literal["exact", "cent"]

KALSHI_TAKER_RATE = 0.07
"""Kalshi general trading fee rate: ``round_up(M * 0.07 * C * P * (1 - P))``."""

KALSHI_MAKER_FEE_FACTORS: dict[str, float] = {
    "quadratic": 0.0,
    "quadratic_with_maker_fees": 0.25,
    "quadratic_with_combo_maker_fees": 0.5,
}
"""Maker fee as a multiple of the taker fee, by Kalshi series ``fee_type``."""

FORECASTEX_EXCHANGE_FEE = 0.01
"""ForecastEx transaction fee per contract per side, in force from 2026-03-31."""

POLYMARKET_US_TAKER_COEFFICIENT = 0.0695
"""Polymarket US standard taker coefficient (theta), effective 2026-10-01."""

POLYMARKET_US_MAKER_REBATE_COEFFICIENT = 0.0125
"""Polymarket US maker rebate coefficient (magnitude of theta = -0.0125)."""

_MICRO = 1_000_000.0
_KALSHI_BALANCE_PRECISION_MICRO = {"cent": 10_000.0, "direct": 100.0}

ArrayInput = Any


# === Shared helpers ===


def _floats(name: str, value: ArrayInput) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _prices(value: ArrayInput) -> np.ndarray:
    price = _floats("price", value)
    if np.any((price < 0.0) | (price > 1.0)):
        raise ValueError("price must be in [0, 1] dollars per contract")
    return price


def _contracts(value: ArrayInput) -> np.ndarray:
    contracts = _floats("contracts", value)
    if np.any(contracts < 0.0):
        raise ValueError("contracts must be non-negative; pass the direction as side")
    return contracts


def _labels(name: str, value: ArrayInput, allowed: tuple[str, ...]) -> np.ndarray:
    labels = np.asarray(value, dtype=np.str_)
    unknown = sorted(set(np.unique(labels).tolist()) - set(allowed))
    if unknown:
        raise ValueError(f"{name} must be one of {allowed}, got {unknown}")
    return labels


def _snap(units: np.ndarray) -> np.ndarray:
    """Remove floating-point noise from values that are integers in exact arithmetic.

    ``0.07 * 100 * 0.25 * 1e6`` evaluates to ``1750000.0000000002``; ceiling it
    would charge one micro-dollar too much. A value within ``1e-12`` relative of
    an integer is treated as that integer.
    """
    nearest = np.rint(units)
    tolerance = 1e-12 * np.maximum(np.abs(units), 1e3)
    return np.where(np.abs(units - nearest) <= tolerance, nearest, units)


def _side_from_quantity(quantity: float) -> Side:
    return "buy" if quantity >= 0.0 else "sell"


# === Kalshi ===


def _kalshi_components(
    price: ArrayInput,
    contracts: ArrayInput,
    liquidity: ArrayInput,
    side: ArrayInput,
    fee_type: ArrayInput,
    fee_multiplier: ArrayInput,
    rounding: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return trade fee and rounding fee per fill in micro-dollars, and the precision."""
    p = _prices(price)
    c = _contracts(contracts)
    multiplier = _floats("fee_multiplier", fee_multiplier)
    if np.any(multiplier < 0.0):
        raise ValueError("fee_multiplier must be non-negative")
    liquidity_labels = _labels("liquidity", liquidity, ("taker", "maker"))
    side_labels = _labels("side", side, ("buy", "sell"))
    fee_types = np.asarray(fee_type, dtype=np.str_)
    if np.any(fee_types == "flat"):
        raise ValueError(
            "Kalshi fee_type 'flat' uses the Specific Trading Fees Table, which is not modeled"
        )
    fee_types = _labels("fee_type", fee_types, tuple(KALSHI_MAKER_FEE_FACTORS))
    if rounding not in ("exact", "cent", "direct"):
        raise ValueError(f"rounding must be 'exact', 'cent' or 'direct', got {rounding!r}")

    maker_factor = np.zeros(fee_types.shape)
    for name, factor in KALSHI_MAKER_FEE_FACTORS.items():
        maker_factor = np.where(fee_types == name, factor, maker_factor)
    factor = np.where(liquidity_labels == "taker", 1.0, maker_factor)
    model_fee = multiplier * KALSHI_TAKER_RATE * factor * c * p * (1.0 - p)
    trade_fee = np.ceil(_snap(model_fee * _MICRO))
    if rounding == "exact":
        return trade_fee, np.zeros_like(trade_fee), 0.0

    precision = _KALSHI_BALANCE_PRECISION_MICRO[rounding]
    notional = _snap(p * c * _MICRO)
    revenue = np.where(side_labels == "buy", -notional, notional)
    before_alignment = revenue - trade_fee
    aligned = np.floor(_snap(before_alignment / precision)) * precision
    rounding_fee = before_alignment - aligned
    trade_fee, rounding_fee = np.broadcast_arrays(trade_fee, rounding_fee)
    return trade_fee, rounding_fee, precision


def kalshi_fee(
    price: ArrayInput,
    contracts: ArrayInput,
    *,
    liquidity: ArrayInput = "taker",
    side: ArrayInput = "buy",
    fee_type: ArrayInput = "quadratic",
    fee_multiplier: ArrayInput = 1.0,
    rounding: KalshiRounding = "exact",
) -> np.ndarray:
    """Kalshi trading fee per fill, with each fill treated as its own order.

    Taker fee: ``round_up(M * 0.07 * C * P * (1 - P))`` with ``C`` contracts at
    price ``P`` and ``M`` the series ``fee_multiplier``. A maker fill pays a
    multiple of the taker formula set by the series ``fee_type``: 0 for
    ``quadratic``, 0.25 for ``quadratic_with_maker_fees`` and 0.5 for
    ``quadratic_with_combo_maker_fees``. Both ``fee_type`` and ``fee_multiplier``
    are on the series record of Kalshi's API (an event may override them).
    Kalshi does not publish the ``flat`` formula in machine-readable form; that
    type raises ``ValueError``.

    Rounding:

    - ``"exact"``: the trade fee, rounded up to 0.000001. 100 contracts at 0.50
      pay 1.75; 1 contract at 0.50 pays 0.0175.
    - ``"cent"``: what a non-direct (FCM-cleared, retail) member is charged. The
      balance change (cash for the contracts minus the trade fee) is aligned
      down to 0.01, and the difference is added to the fee. 1 contract at 0.50
      pays 0.02. Kalshi rebates accumulated overpayment across the fills of one
      order; this function treats every row as a one-fill order, where no rebate
      can arise. Use `kalshi_order_fees` for the fills of one multi-fill order.
    - ``"direct"``: as ``"cent"`` with the direct-member precision 0.0001.

    ``side`` changes the result only under ``"cent"`` and ``"direct"``, where a
    sub-cent notional is part of the alignment.

    Sources: Kalshi fee schedule, "Fee Schedule for July 2026 - 7.7.26 Update"
    (https://kalshi.com/docs/kalshi-fee-schedule.pdf); maker multipliers from
    the ``FeeType`` description in https://docs.kalshi.com/openapi.yaml;
    rounding from https://docs.kalshi.com/getting_started/fee_rounding. All read
    2026-10-08.

    Args:
        price: Execution price in dollars per contract, in ``[0, 1]``.
        contracts: Contracts filled, non-negative.
        liquidity: ``"taker"`` or ``"maker"``, scalar or per fill.
        side: ``"buy"`` or ``"sell"``, scalar or per fill.
        fee_type: Series fee type, scalar or per fill.
        fee_multiplier: Series fee multiplier ``M``, scalar or per fill.
        rounding: ``"exact"``, ``"cent"`` or ``"direct"``.

    Returns:
        Fee in dollars per fill.
    """
    trade_fee, rounding_fee, _ = _kalshi_components(
        price, contracts, liquidity, side, fee_type, fee_multiplier, rounding
    )
    return (trade_fee + rounding_fee) / _MICRO


def kalshi_order_fees(
    price: ArrayInput,
    contracts: ArrayInput,
    *,
    liquidity: ArrayInput = "taker",
    side: ArrayInput = "buy",
    fee_type: ArrayInput = "quadratic",
    fee_multiplier: ArrayInput = 1.0,
    rounding: KalshiRounding = "cent",
) -> np.ndarray:
    """Net Kalshi fee per fill for the fills of one order, in execution order.

    Each fill pays its trade fee plus a rounding fee that aligns the balance to
    the member's precision (see `kalshi_fee`). The rounding fees accumulate per
    order, across taker and maker fills alike. Whenever the accumulator holds at
    least one precision step, whole steps are rebated on that fill, capped so
    the fill's net fee stays non-negative. Three fills that each overpay 0.004
    pay 0.004, 0.004, and on the third fill 0.010 less (accumulator 0.012,
    0.002 carried forward).

    Source: https://docs.kalshi.com/getting_started/fee_rounding (read
    2026-10-08).

    Args:
        price: Execution price per fill, in ``[0, 1]``.
        contracts: Contracts per fill, non-negative.
        liquidity: ``"taker"`` or ``"maker"``, scalar or per fill.
        side: ``"buy"`` or ``"sell"``; one order has one side.
        fee_type: Series fee type.
        fee_multiplier: Series fee multiplier ``M``.
        rounding: ``"cent"`` (non-direct member), ``"direct"`` or ``"exact"``
            (no balance alignment, so no accumulator).

    Returns:
        One-dimensional array of net fees in dollars, one per fill.
    """
    trade_fee, rounding_fee, precision = _kalshi_components(
        price, contracts, liquidity, side, fee_type, fee_multiplier, rounding
    )
    trade_fee = np.atleast_1d(trade_fee)
    rounding_fee = np.atleast_1d(rounding_fee)
    if trade_fee.ndim != 1:
        raise ValueError("kalshi_order_fees takes the fills of one order as 1-D input")
    if precision == 0.0:
        return trade_fee / _MICRO

    net = np.empty_like(trade_fee)
    accumulator = 0.0
    for index, (fee, overpayment) in enumerate(zip(trade_fee, rounding_fee, strict=True)):
        charged = fee + overpayment
        accumulator += overpayment
        available = np.array([accumulator, charged], dtype=np.float64) / precision
        rebate = float(np.floor(_snap(available)).min()) * precision
        accumulator -= rebate
        net[index] = charged - rebate
    return net / _MICRO


class KalshiCommission:
    """Kalshi trading fee as an engine ``CommissionModel``.

    Each ``calculate`` call is one fill and is treated as a one-fill order: the
    per-order rebate of rounding overpayment (`kalshi_order_fees`) needs order
    identity, which the protocol does not pass. A positive quantity is a buy, a
    negative quantity a sell; buying NO is a buy at the NO price.

    Formula, rounding modes and sources: see `kalshi_fee`.

    Args:
        liquidity: ``"taker"`` or ``"maker"`` for every fill.
        fee_type: Series ``fee_type``.
        fee_multiplier: Series ``fee_multiplier``.
        rounding: ``"exact"``, ``"cent"`` (non-direct member) or ``"direct"``.
    """

    def __init__(
        self,
        liquidity: Liquidity = "taker",
        fee_type: KalshiFeeType = "quadratic",
        fee_multiplier: float = 1.0,
        rounding: KalshiRounding = "exact",
    ):
        kalshi_fee(
            0.5,
            1.0,
            liquidity=liquidity,
            fee_type=fee_type,
            fee_multiplier=fee_multiplier,
            rounding=rounding,
        )
        self.liquidity = liquidity
        self.fee_type = fee_type
        self.fee_multiplier = fee_multiplier
        self.rounding = rounding

    def calculate(self, asset: str, quantity: float, price: float) -> float:
        return float(
            kalshi_fee(
                price,
                abs(quantity),
                liquidity=self.liquidity,
                side=_side_from_quantity(quantity),
                fee_type=self.fee_type,
                fee_multiplier=self.fee_multiplier,
                rounding=self.rounding,
            )
        )


# === ForecastEx ===


def forecastex_fee(
    price: ArrayInput,
    contracts: ArrayInput,
    *,
    exchange_fee: ArrayInput = FORECASTEX_EXCHANGE_FEE,
    broker_commission: ArrayInput = 0.0,
) -> np.ndarray:
    """ForecastEx fee per fill: a flat amount per contract per side.

    ForecastEx charges 0.01 per contract per side at execution, "independent of
    contract resolution, netting, or settlement"; IBKR charges no commission on
    ForecastEx contracts. The fee does not depend on price or liquidity; price
    is validated so that the inputs match the other venues.

    Before 2026-03-31 the fee was embedded in prices (a YES and a NO paired at
    1.01) instead of charged explicitly. Prices recorded before that date
    already contain it; pass ``exchange_fee=0.0`` for them. Block trades are
    exempt from 2026-09-22 to 2026-12-01; pass ``exchange_fee=0.0`` for those
    too.

    Sources: ForecastEx fee schedule, CFTC filing
    https://www.cftc.gov/filings/orgrules/rules09082624298.pdf; change of model,
    https://www.cftc.gov/filings/orgrules/rules03022640231.pdf; IBKR commission,
    https://www.interactivebrokers.com/en/pricing/commissions-events.php. All
    read 2026-10-08.

    Args:
        price: Execution price in ``[0, 1]`` (validated, not used).
        contracts: Contracts filled, non-negative.
        exchange_fee: Exchange fee per contract.
        broker_commission: Broker commission per contract.

    Returns:
        Fee in dollars per fill.
    """
    p = _prices(price)
    c = _contracts(contracts)
    per_contract = _floats("exchange_fee", exchange_fee) + _floats(
        "broker_commission", broker_commission
    )
    if np.any(per_contract < 0.0):
        raise ValueError("exchange_fee + broker_commission must be non-negative")
    fee = per_contract * c
    return np.broadcast_arrays(fee, p)[0].astype(np.float64, copy=True)


class ForecastExCommission:
    """ForecastEx fee via IBKR as an engine ``CommissionModel``.

    Formula and sources: see `forecastex_fee`.

    Args:
        exchange_fee: Exchange fee per contract per side.
        broker_commission: Broker commission per contract per side.
    """

    def __init__(
        self,
        exchange_fee: float = FORECASTEX_EXCHANGE_FEE,
        broker_commission: float = 0.0,
    ):
        forecastex_fee(0.5, 1.0, exchange_fee=exchange_fee, broker_commission=broker_commission)
        self.exchange_fee = exchange_fee
        self.broker_commission = broker_commission

    def calculate(self, asset: str, quantity: float, price: float) -> float:
        return float(
            forecastex_fee(
                price,
                abs(quantity),
                exchange_fee=self.exchange_fee,
                broker_commission=self.broker_commission,
            )
        )


# === Polymarket US ===


def polymarket_us_fee(
    price: ArrayInput,
    contracts: ArrayInput,
    *,
    liquidity: ArrayInput = "taker",
    fee_coefficient: ArrayInput = POLYMARKET_US_TAKER_COEFFICIENT,
    maker_rebate_coefficient: ArrayInput = POLYMARKET_US_MAKER_REBATE_COEFFICIENT,
    rounding: PolymarketUSRounding = "exact",
) -> np.ndarray:
    """Polymarket US fee per fill; a maker rebate is a negative fee.

    Taker: ``theta * C * p * (1 - p)`` with ``theta = 0.0695`` (1,000 contracts
    at 0.50 pay 17.38 after rounding). Maker: ``-0.0125 * C * p * (1 - p)``,
    paid at the trade. ``fee_coefficient`` is the taker theta; pass the
    ``fee_coefficient`` column of ml4t-data's ``PolymarketUSProvider`` market
    table (the venue's ``feeCoefficient``) to apply a per-market value.

    Rounding: ``"exact"`` returns the formula value; ``"cent"`` rounds each fill
    to 0.01 with round-half-to-even, as the venue charges it, so small fees can
    round to 0. The venue also caps the sum over one aggressive order's fills
    at the rounded cumulative fee; per-fill rounding can exceed that cap by up
    to 0.005 per fill.

    Not modeled: the combo taker curve ``C * p * [0.0695 * (1 - p) + 0.06 *
    (1 - p)^4]`` and the prior-month taker volume rebates (10%, 25%, 50%).

    Source: Polymarket US fee schedule, https://docs.polymarket.us/fees (read
    2026-10-08).

    Args:
        price: Execution price in ``[0, 1]``.
        contracts: Contracts filled, non-negative.
        liquidity: ``"taker"`` or ``"maker"``, scalar or per fill.
        fee_coefficient: Taker theta, scalar or per fill.
        maker_rebate_coefficient: Maker rebate magnitude, scalar or per fill.
        rounding: ``"exact"`` or ``"cent"``.

    Returns:
        Signed fee in dollars per fill: positive is paid, negative is received.
    """
    p = _prices(price)
    c = _contracts(contracts)
    taker = _floats("fee_coefficient", fee_coefficient)
    maker = _floats("maker_rebate_coefficient", maker_rebate_coefficient)
    if np.any(taker < 0.0) or np.any(maker < 0.0):
        raise ValueError("fee_coefficient and maker_rebate_coefficient must be non-negative")
    liquidity_labels = _labels("liquidity", liquidity, ("taker", "maker"))
    if rounding not in ("exact", "cent"):
        raise ValueError(f"rounding must be 'exact' or 'cent', got {rounding!r}")
    coefficient = np.where(liquidity_labels == "taker", taker, -maker)
    fee = coefficient * c * p * (1.0 - p)
    if rounding == "cent":
        half_cents = _snap(fee * 200.0)
        fee = np.round(half_cents / 2.0) / 100.0
    return np.asarray(fee, dtype=np.float64)


class PolymarketUSCommission:
    """Polymarket US taker fee as an engine ``CommissionModel``.

    The engine charges only non-negative commissions, so a maker fill is
    charged 0 and its rebate (0.0125 x C x p x (1 - p)) is left out, which
    overstates maker costs. Use `polymarket_us_fee` where the rebate matters.

    Formula, rounding and sources: see `polymarket_us_fee`.

    Args:
        liquidity: ``"taker"`` or ``"maker"`` for every fill.
        fee_coefficient: Taker theta of the market.
        rounding: ``"exact"`` or ``"cent"``.
    """

    def __init__(
        self,
        liquidity: Liquidity = "taker",
        fee_coefficient: float = POLYMARKET_US_TAKER_COEFFICIENT,
        rounding: PolymarketUSRounding = "exact",
    ):
        polymarket_us_fee(
            0.5, 1.0, liquidity=liquidity, fee_coefficient=fee_coefficient, rounding=rounding
        )
        self.liquidity = liquidity
        self.fee_coefficient = fee_coefficient
        self.rounding = rounding

    def calculate(self, asset: str, quantity: float, price: float) -> float:
        if self.liquidity == "maker":
            return 0.0
        return float(
            polymarket_us_fee(
                price,
                abs(quantity),
                liquidity="taker",
                fee_coefficient=self.fee_coefficient,
                rounding=self.rounding,
            )
        )
