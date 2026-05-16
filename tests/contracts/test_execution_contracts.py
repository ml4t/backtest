from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from ml4t.backtest.config import (
    BacktestConfig,
    CommissionType,
    ExecutionPrice,
    SlippageType,
    SpreadConvention,
)
from ml4t.backtest.engine import run_backtest
from ml4t.backtest.strategy import Strategy
from ml4t.backtest.types import ExecutionMode
from ml4t.specs.market_data import FeedSpec


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


def _entry_price(mode: ExecutionMode, price: ExecutionPrice) -> float:
    config = BacktestConfig(
        execution_mode=mode,
        execution_price=price,
        commission_type=CommissionType.NONE,
        slippage_type=SlippageType.NONE,
    )
    result = run_backtest(prices=_prices(), strategy=_BuyOnce(), config=config)
    assert result.trades
    return result.trades[0].entry_price


def _quote_prices() -> pl.DataFrame:
    start = datetime(2024, 1, 1)
    rows = [
        {
            "timestamp": start,
            "asset": "AAPL",
            "open": 90.0,
            "high": 105.0,
            "low": 89.0,
            "close": 100.0,
            "mid_price": 100.0,
            "bid": 99.5,
            "ask": 100.5,
            "bid_size": 500.0,
            "ask_size": 750.0,
            "volume": 1_000_000.0,
        },
        {
            "timestamp": start + timedelta(days=1),
            "asset": "AAPL",
            "open": 110.0,
            "high": 112.0,
            "low": 109.0,
            "close": 111.0,
            "mid_price": 111.0,
            "bid": 110.75,
            "ask": 111.25,
            "bid_size": 800.0,
            "ask_size": 900.0,
            "volume": 1_000_000.0,
        },
    ]
    return pl.DataFrame(rows)


def test_same_bar_fills_at_signal_bar_close() -> None:
    assert _entry_price(ExecutionMode.SAME_BAR, ExecutionPrice.CLOSE) == 100.0


def test_next_bar_fills_at_following_bar_open() -> None:
    assert _entry_price(ExecutionMode.NEXT_BAR, ExecutionPrice.OPEN) == 110.0


def test_same_bar_quote_side_execution_uses_ask_for_buys() -> None:
    config = BacktestConfig(
        execution_mode=ExecutionMode.SAME_BAR,
        execution_price=ExecutionPrice.QUOTE_SIDE,
        mark_price=ExecutionPrice.QUOTE_SIDE,
        commission_type=CommissionType.NONE,
        slippage_type=SlippageType.NONE,
    )
    result = run_backtest(
        prices=_quote_prices(),
        strategy=_BuyOnce(),
        config=config,
        feed_spec=FeedSpec(
            price_col="mid_price",
            bid_col="bid",
            ask_col="ask",
            bid_size_col="bid_size",
            ask_size_col="ask_size",
        ),
    )

    assert result.trades
    assert result.trades[0].entry_price == 100.5
    assert result.trades[0].entry_ask_price == 100.5
    assert result.metrics["final_value"] == pytest.approx(100000.0 - 100.5 + 110.75)


def test_spread_slippage_full_spread_uses_half_spread_per_side() -> None:
    config = BacktestConfig(
        execution_mode=ExecutionMode.SAME_BAR,
        execution_price=ExecutionPrice.CLOSE,
        commission_type=CommissionType.NONE,
        slippage_type=SlippageType.SPREAD,
        slippage_spread=0.20,
        slippage_spread_convention=SpreadConvention.FULL_SPREAD,
    )
    result = run_backtest(prices=_prices(), strategy=_BuyOnce(), config=config)

    assert result.trades
    assert result.trades[0].entry_price == pytest.approx(100.10)
    assert result.trades[0].entry_slippage == pytest.approx(0.10)


def test_spread_slippage_asset_override_uses_configured_per_side_cost() -> None:
    config = BacktestConfig(
        execution_mode=ExecutionMode.SAME_BAR,
        execution_price=ExecutionPrice.CLOSE,
        commission_type=CommissionType.NONE,
        slippage_type=SlippageType.SPREAD,
        slippage_spread=0.20,
        slippage_spread_by_asset={"AAPL": 0.04},
        slippage_spread_convention=SpreadConvention.HALF_SPREAD,
    )
    result = run_backtest(prices=_prices(), strategy=_BuyOnce(), config=config)

    assert result.trades
    assert result.trades[0].entry_price == pytest.approx(100.04)
    assert result.trades[0].entry_slippage == pytest.approx(0.04)
