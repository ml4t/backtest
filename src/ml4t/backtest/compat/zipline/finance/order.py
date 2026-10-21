"""Detached projections of native order state, with no independent order ledger."""

from dataclasses import dataclass
from enum import IntEnum

import pandas as pd

from ..assets import Asset


class ORDER_STATUS(IntEnum):
    OPEN = 0
    FILLED = 1
    CANCELLED = 2
    REJECTED = 3
    HELD = 4


@dataclass(frozen=True)
class OrderView:
    id: str
    native_order_id: str
    asset: Asset
    amount: float
    filled: float
    commission: float
    status: ORDER_STATUS
    created: pd.Timestamp | None
    dt: pd.Timestamp | None
    limit: float | None
    stop: float | None
    stop_reached: bool
    reason: str | None

    @property
    def sid(self) -> Asset:
        return self.asset
