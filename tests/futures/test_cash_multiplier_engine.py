"""Cash checks for futures orders through the public Engine workflow."""

from datetime import datetime, timedelta

import polars as pl

from ml4t.backtest import BacktestConfig, DataFeed, Engine, ExecutionMode, Strategy
from ml4t.backtest.config import FillOrdering
from ml4t.backtest.types import AssetClass, ContractSpec, OrderSide


def _run_orders(price: float, count: int, **cash_settings):
    class BuyContracts(Strategy):
        def __init__(self):
            self.orders = []

        def on_data(self, timestamp, data, context, broker):
            if not self.orders:
                self.orders = [broker.submit_order("ES", 1, OrderSide.BUY) for _ in range(count)]

    start = datetime(2024, 1, 2)
    prices = pl.DataFrame(
        {
            "timestamp": [start + timedelta(days=i) for i in range(3)],
            "asset": ["ES"] * 3,
            "open": [price] * 3,
            "high": [price] * 3,
            "low": [price] * 3,
            "close": [price] * 3,
            "volume": [1_000_000] * 3,
        }
    )
    strategy = BuyContracts()
    result = Engine(
        DataFeed(prices_df=prices),
        strategy,
        BacktestConfig(
            initial_cash=100_000,
            execution_mode=ExecutionMode.NEXT_BAR,
            reject_on_insufficient_cash=True,
            **cash_settings,
        ),
        contract_specs={"ES": ContractSpec("ES", AssetClass.FUTURE, multiplier=50.0)},
    ).run()
    assert all(order is not None for order in strategy.orders)
    return [order.status.value for order in strategy.orders], len(result.fills)


def test_next_bar_cash_check_rejects_futures_notional_above_cash():
    statuses, fills = _run_orders(4_000.0, 1, next_bar_simple_cash_check=True)

    assert statuses == ["rejected"]
    assert fills == 0


def test_sequential_reservation_rejects_third_futures_contract():
    statuses, fills = _run_orders(
        1_000.0,
        3,
        buying_power_reservation=True,
        fill_ordering=FillOrdering.SEQUENTIAL,
    )

    assert statuses == ["filled", "filled", "rejected"]
    assert fills == 2
