"""Parameter conversion for native pre-submission validators."""

import logging
from dataclasses import dataclass

from ...controls import OrderControlContext, TradingControl, TradingControlViolation


@dataclass(frozen=True)
class _TradingRule:
    constraint: TradingControl
    on_error: str

    @property
    def counts_order_requests(self) -> bool:
        return getattr(self.constraint, "counts_order_requests", False)

    @property
    def timezone(self) -> str:
        return getattr(self.constraint, "timezone", "UTC")

    def validate(self, request: OrderControlContext) -> None:
        try:
            self.constraint.validate(request)
        except TradingControlViolation as error:
            if self.on_error == "fail":
                raise
            logging.getLogger(__name__).error("%s", error)
