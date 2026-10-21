"""Migration cost parameters compiled to native commission methods."""

from collections.abc import Mapping
from dataclasses import dataclass

from ....models import PerOrderCommission
from ._futures_fees import EXCHANGE_FEES


@dataclass(frozen=True)
class PerShare:
    cost: float = 0.001
    min_trade_cost: float = 0.0


@dataclass(frozen=True)
class PerTrade:
    cost: float = 0.0


@dataclass(frozen=True)
class PerDollar:
    cost: float = 0.0015


@dataclass(frozen=True)
class PerContract:
    cost: float | Mapping[str, float]
    exchange_fee: float | Mapping[str, float]
    min_trade_cost: float | None = 0.0


@dataclass(frozen=True)
class PerFutureTrade:
    cost: float | Mapping[str, float] = 0.0


def _root_value(value: float | Mapping[str, float], root: str, *, fallback: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return value.get(root, fallback)


def _future_model(model: PerContract | PerFutureTrade, root: str) -> PerOrderCommission:
    if isinstance(model, PerContract):
        fee = model.exchange_fee
        if isinstance(fee, Mapping) and root not in fee and root not in EXCHANGE_FEES:
            raise ValueError(f"Declare an exchange fee for unknown legacy root {root!r}")
        return PerOrderCommission(
            per_share=_root_value(model.cost, root, fallback=0.85),
            per_trade=_root_value(fee, root, fallback=EXCHANGE_FEES.get(root, 0.0)),
            minimum=model.min_trade_cost or 0.0,
        )
    if isinstance(model.cost, Mapping) and root not in model.cost and root not in EXCHANGE_FEES:
        raise ValueError(f"Declare a trade fee for unknown legacy root {root!r}")
    return PerOrderCommission(
        per_trade=_root_value(model.cost, root, fallback=EXCHANGE_FEES.get(root, 0.0))
    )
