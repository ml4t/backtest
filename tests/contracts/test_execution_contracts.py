from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from ml4t.backtest.engine import run_backtest
from ml4t.backtest.strategy import Strategy
from ml4t.backtest.types import ExecutionMode


def _prices() -> pl.DataFrame:
    start = datetime(2024, 1, 1)
    rows = [
        {
            "timestamp": start,
            "asset": "AAPL",
            "open": 90.0,
            "high": 105.0,
            "low": 89.0,
            "close": 100.0,
            "volume": 1_000_000.0,
        },
        {
            "timestamp": start + timedelta(days=1),
            "asset": "AAPL",
            "open": 110.0,
            "high": 112.0,
            "low": 109.0,
            "close": 111.0,
            "volume": 1_000_000.0,
        },
    ]
    return pl.DataFrame(rows)


class _BuyOnce(Strategy):
    def __init__(self) -> None:
        self.done = False

    def on_data(self, timestamp, data, context, broker) -> None:
        if not self.done:
            broker.submit_order("AAPL", 1.0)
            self.done = True


def _entry_price(mode: ExecutionMode) -> float:
    result = run_backtest(prices=_prices(), strategy=_BuyOnce(), execution_mode=mode)
    assert result.trades
    return result.trades[0].entry_price


def test_same_bar_fills_at_signal_bar_close() -> None:
    assert _entry_price(ExecutionMode.SAME_BAR) == 100.0


def test_next_bar_fills_at_following_bar_open() -> None:
    assert _entry_price(ExecutionMode.NEXT_BAR) == 110.0
