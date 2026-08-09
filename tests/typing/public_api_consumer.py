"""Representative typed use of the installed ml4t-backtest wheel."""

from datetime import datetime
from typing import Any

import polars as pl

from ml4t.backtest import (
    BacktestConfig,
    BacktestResult,
    Broker,
    DataFeed,
    Engine,
    OrderSide,
    Strategy,
)


class BuyFirstAsset(Strategy):
    def on_data(
        self,
        timestamp: datetime,
        data: dict[str, dict],
        context: dict[str, Any],
        broker: Broker,
    ) -> None:
        del timestamp, context
        if data and not broker.positions:
            broker.submit_order(next(iter(data)), 1.0, side=OrderSide.BUY)


def run_typed_consumer(prices: pl.DataFrame) -> BacktestResult:
    feed = DataFeed(prices_df=prices)
    engine = Engine(feed=feed, strategy=BuyFirstAsset(), config=BacktestConfig())
    return engine.run()
