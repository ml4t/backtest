from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from ml4t.backtest.config import BacktestConfig
from ml4t.backtest.engine import run_backtest
from ml4t.backtest.profiles import list_profiles
from ml4t.backtest.strategy import Strategy
from ml4t.backtest.types import StopLevelBasis


def _prices() -> pl.DataFrame:
    start = datetime(2024, 1, 1)
    rows = []
    for i, (open_, close) in enumerate([(100.0, 101.0), (110.0, 111.0)]):
        ts = start + timedelta(days=i)
        rows.append(
            {
                "timestamp": ts,
                "asset": "AAPL",
                "open": open_,
                "high": max(open_, close),
                "low": min(open_, close),
                "close": close,
                "volume": 1_000_000.0,
            }
        )
    return pl.DataFrame(rows)


class _BuyOnce(Strategy):
    def __init__(self) -> None:
        self.done = False

    def on_data(self, timestamp, data, context, broker) -> None:
        if not self.done:
            broker.submit_order("AAPL", 1.0)
            self.done = True


def test_profile_registry_has_expected_core_profiles() -> None:
    assert list_profiles() == ["backtrader", "default", "realistic", "vectorbt", "zipline"]


def test_string_preset_and_explicit_preset_config_match() -> None:
    by_name = run_backtest(prices=_prices(), strategy=_BuyOnce(), config="vectorbt")
    by_config = run_backtest(
        prices=_prices(),
        strategy=_BuyOnce(),
        config=BacktestConfig.from_preset("vectorbt"),
    )

    assert by_name.metrics["final_value"] == by_config.metrics["final_value"]
    assert by_name.trades[0].entry_price == by_config.trades[0].entry_price


def test_profiles_enforce_expected_entry_timing_contract() -> None:
    vbt = run_backtest(prices=_prices(), strategy=_BuyOnce(), config="vectorbt")
    bt = run_backtest(prices=_prices(), strategy=_BuyOnce(), config="backtrader")
    zl = run_backtest(prices=_prices(), strategy=_BuyOnce(), config="zipline")

    assert vbt.trades[0].entry_price == 101.0  # same-bar close
    assert 110.0 < bt.trades[0].entry_price < 111.0  # next-bar open with default slippage
    assert zl.trades[0].entry_price == 110.0  # next-bar open


def test_backtrader_profile_uses_signal_price_stop_basis() -> None:
    cfg = BacktestConfig.from_preset("backtrader")
    assert cfg.stop_level_basis == StopLevelBasis.SIGNAL_PRICE
