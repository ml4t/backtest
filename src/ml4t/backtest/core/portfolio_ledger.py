"""Portfolio/account view helpers extracted from Broker."""

from __future__ import annotations


class PortfolioLedger:
    """Read-model helpers for account/portfolio state."""

    def __init__(self, broker):
        self.broker = broker

    def get_account_value(self) -> float:
        value = self.broker.cash
        for asset, pos in self.broker.positions.items():
            price = self.broker._current_prices.get(asset)
            if price is None:
                continue
            multiplier = self.broker.get_multiplier(asset)
            value += pos.quantity * price * multiplier
        return value

    def get_rejected_orders(self, asset: str | None = None):
        rejected = [o for o in self.broker.orders if o.status.value == "rejected"]
        if asset is not None:
            rejected = [o for o in rejected if o.asset == asset]
        return rejected

    @property
    def last_rejection_reason(self):
        rejected = [o for o in self.broker.orders if o.status.value == "rejected"]
        return rejected[-1].rejection_reason if rejected else None
