"""End-to-end commission accounting for a one-order position flip."""

from datetime import datetime, timedelta

import polars as pl

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.types import OrderSide


def test_flat_ticket_fee_is_charged_once_for_one_flip_order():
    class FlatTicketFee:
        def __init__(self):
            self.calls = []

        def calculate(self, asset, quantity, price):
            self.calls.append((asset, quantity, price))
            return 1.0

    class Flip(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == datetime(2024, 1, 1):
                broker.submit_order("A", 100, OrderSide.BUY)
            elif timestamp == datetime(2024, 1, 3):
                broker.submit_order("A", 200, OrderSide.SELL)

    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)]
    prices = pl.DataFrame(
        {
            "timestamp": dates,
            "asset": ["A"] * len(dates),
            "open": [10.0] * len(dates),
            "high": [10.0] * len(dates),
            "low": [10.0] * len(dates),
            "close": [10.0] * len(dates),
            "volume": [10000] * len(dates),
        }
    )
    engine = Engine(
        feed=DataFeed(prices_df=prices),
        strategy=Flip(),
        config=BacktestConfig(
            initial_cash=10000,
            allow_short_selling=True,
            allow_leverage=True,
        ),
    )
    model = FlatTicketFee()
    engine.broker.commission_model = model
    result = engine.run()

    assert len(engine.broker.orders) == 2
    assert [fill.commission for fill in result.fills] == [1.0, 1.0]
    assert [quantity for _, quantity, _ in model.calls] == [100.0, 200.0]
    assert result.to_portfolio_state_dataframe()["cash"][-1] == 10998.0
    assert result.trades[0].fees + result.trades[1].fees == 2.0
