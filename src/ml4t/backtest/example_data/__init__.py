"""Small synthetic price panels used by the runnable documentation."""

from datetime import datetime
from importlib.resources import files
from typing import Any

import polars as pl

from ..strategy import Strategy

_PANELS = frozenset({"equity", "etf", "future", "fx", "crypto_perp"})


def load_example_prices(category: str) -> pl.DataFrame:
    """Load one bundled synthetic OHLCV panel for a documented asset category."""
    if category not in _PANELS:
        raise ValueError(f"Unknown example category {category!r}; choose from {sorted(_PANELS)}")
    path = files(__package__).joinpath(f"{category}.csv")
    return pl.read_csv(str(path), try_parse_dates=True)


class ExampleRoundTrip(Strategy):
    """Buy on the first asset bar and close after its fifth bar."""

    def __init__(self, asset: str, quantity: float) -> None:
        self.asset = asset
        self.quantity = quantity
        self.asset_bars = 0

    def on_data(
        self,
        timestamp: datetime,
        data: dict[str, dict[str, Any]],
        context: dict[str, Any],
        broker: Any,
    ) -> None:
        if self.asset not in data:
            return
        self.asset_bars += 1
        if self.asset_bars == 1:
            broker.submit_order(self.asset, self.quantity)
        elif self.asset_bars == 5:
            broker.close_position(self.asset)
