"""``ExecutionPrice.VWAP`` fills at the feed's VWAP, or refuses.

Before this contract existed the enum resolved to the close, and under
``NEXT_BAR`` the ``use_open`` short-circuit returned before the VWAP branch was
consulted at all - so a VWAP fill returned the *open*. Both substitutions were
silent, and no test in the suite could tell the three prices apart. Every case
here therefore uses an open, a close and a VWAP that are mutually distinct: a
fixture where any two coincide cannot fail against the old behaviour.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from ml4t.backtest import Broker
from ml4t.backtest.config import ExecutionMode, ExecutionPrice
from ml4t.backtest.datafeed import DataFeed
from ml4t.backtest.models import NoCommission, NoSlippage
from ml4t.backtest.types import OrderSide

OPEN = 99.0
CLOSE = 102.0
VWAP = 100.5  # deliberately between the two, and equal to neither


def _broker(mode: ExecutionMode = ExecutionMode.SAME_BAR) -> Broker:
    return Broker(
        initial_cash=100_000.0,
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        execution_price=ExecutionPrice.VWAP,
        execution_mode=mode,
    )


def _advance(broker: Broker, *, vwaps: dict[str, float] | None = None) -> None:
    broker._update_time(
        timestamp=datetime(2024, 1, 2, 9, 31),
        prices={"AAPL": CLOSE},
        opens={"AAPL": OPEN},
        highs={"AAPL": 105.0},
        lows={"AAPL": 95.0},
        closes={"AAPL": CLOSE},
        volumes={"AAPL": 1_000_000.0},
        vwaps={"AAPL": VWAP} if vwaps is None else vwaps,
        signals={},
    )


class TestVwapIsItsOwnPrice:
    def test_same_bar_fill_takes_the_vwap_not_the_close(self):
        broker = _broker()
        _advance(broker)
        assert broker.get_price_for_source(ExecutionPrice.VWAP, "AAPL") == VWAP

    def test_next_bar_fill_takes_the_vwap_not_the_open(self):
        # The regression that motivated this file: use_open short-circuited ahead of
        # the VWAP branch, so this returned OPEN however the branch was written.
        broker = _broker(ExecutionMode.NEXT_BAR)
        _advance(broker)
        price = broker.get_price_for_source(ExecutionPrice.VWAP, "AAPL", use_open=True)
        assert price == VWAP
        assert price != OPEN

    def test_a_quote_source_still_ignores_the_vwap(self):
        broker = _broker(ExecutionMode.NEXT_BAR)
        _advance(broker)
        assert broker.get_price_for_source(ExecutionPrice.OPEN, "AAPL", use_open=True) == OPEN
        assert broker.get_price_for_source(ExecutionPrice.CLOSE, "AAPL") == CLOSE

    def test_an_executed_order_books_at_the_vwap(self):
        broker = _broker()
        _advance(broker)
        broker.submit_order("AAPL", 10.0, OrderSide.BUY)
        broker._process_orders()
        position = broker.get_position("AAPL")
        assert position is not None
        assert position.entry_price == VWAP


class TestANoTradeBarIsANonFill:
    """A bar with no prints has no VWAP, and that is not an error."""

    def test_a_missing_vwap_resolves_to_none_rather_than_raising(self):
        # Measured on the NASDAQ-100: 0.24% of production symbol-minutes inside the
        # exchange session carry no print, and every one of them would stop a run if
        # this raised. The engine's callers already read None as "cannot be priced on
        # this bar" and skip, which is the non-fill.
        broker = _broker()
        _advance(broker, vwaps={})
        assert broker.get_price_for_source(ExecutionPrice.VWAP, "AAPL") is None

    def test_it_does_not_fall_back_to_the_close_or_the_open(self):
        broker = _broker()
        _advance(broker, vwaps={})
        price = broker.get_price_for_source(ExecutionPrice.VWAP, "AAPL")
        assert price != CLOSE
        assert price != OPEN

    def test_one_asset_without_a_vwap_does_not_affect_another(self):
        broker = _broker()
        _advance(broker, vwaps={"MSFT": VWAP})
        assert broker.get_price_for_source(ExecutionPrice.VWAP, "AAPL") is None
        assert broker.get_price_for_source(ExecutionPrice.VWAP, "MSFT") == VWAP

    def test_an_order_on_a_no_trade_bar_does_not_fill(self):
        broker = _broker()
        _advance(broker, vwaps={})
        broker.submit_order("AAPL", 10.0, OrderSide.BUY)
        broker._process_orders()
        assert broker.get_position("AAPL") is None


class TestTheFeedCarriesItThrough:
    def test_a_declared_vwap_col_reaches_the_bar(self):
        prices = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 2, 9, 31)],
                "symbol": ["AAPL"],
                "open": [OPEN],
                "high": [105.0],
                "low": [95.0],
                "close": [CLOSE],
                "volume": [1_000_000.0],
                "vwap": [VWAP],
            }
        )
        feed = DataFeed(prices_df=prices, entity_col="symbol", vwap_col="vwap")
        _timestamp, assets, _context = next(iter(feed))
        assert assets["AAPL"]["vwap"] == VWAP
        assert assets._vwaps["AAPL"] == VWAP

    def test_an_undeclared_vwap_column_is_not_guessed_at(self):
        # The column is present and named exactly "vwap", but the feed does not declare
        # it. Picking it up anyway would make the contract depend on a naming
        # convention rather than on the spec.
        prices = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 2, 9, 31)],
                "symbol": ["AAPL"],
                "open": [OPEN],
                "high": [105.0],
                "low": [95.0],
                "close": [CLOSE],
                "volume": [1_000_000.0],
                "vwap": [VWAP],
            }
        )
        feed = DataFeed(prices_df=prices, entity_col="symbol")
        _timestamp, assets, _context = next(iter(feed))
        assert "vwap" not in assets["AAPL"]
        assert assets._vwaps == {}


class TestAnUndeclaredColumnIsAConfigurationError:
    """A feed with no VWAP column at all is rejected once, before the first bar."""

    @staticmethod
    def _engine(feed, config):
        from ml4t.backtest import Engine
        from ml4t.backtest.strategy import Strategy

        class _Noop(Strategy):
            def on_data(self, timestamp, data, broker):
                return None

        return Engine(feed, _Noop(), config)

    @staticmethod
    def _prices(with_vwap: bool) -> pl.DataFrame:
        frame = {
            "timestamp": [datetime(2024, 1, 2, 9, 31), datetime(2024, 1, 2, 9, 32)],
            "symbol": ["AAPL", "AAPL"],
            "open": [OPEN, OPEN],
            "high": [105.0, 105.0],
            "low": [95.0, 95.0],
            "close": [CLOSE, CLOSE],
            "volume": [1_000_000.0, 1_000_000.0],
        }
        if with_vwap:
            frame["vwap"] = [VWAP, VWAP]
        return pl.DataFrame(frame)

    def test_engine_rejects_vwap_execution_on_a_feed_without_the_column(self):
        from ml4t.backtest.config import BacktestConfig

        feed = DataFeed(prices_df=self._prices(with_vwap=False), entity_col="symbol")
        config = BacktestConfig(execution_price=ExecutionPrice.VWAP)
        with pytest.raises(ValueError, match="declares no VWAP column"):
            self._engine(feed, config)

    def test_engine_accepts_it_once_the_column_is_declared(self):
        from ml4t.backtest.config import BacktestConfig

        feed = DataFeed(
            prices_df=self._prices(with_vwap=True), entity_col="symbol", vwap_col="vwap"
        )
        config = BacktestConfig(execution_price=ExecutionPrice.VWAP)
        self._engine(feed, config)  # does not raise

    def test_a_feed_without_the_column_is_fine_under_another_execution_price(self):
        from ml4t.backtest.config import BacktestConfig

        feed = DataFeed(prices_df=self._prices(with_vwap=False), entity_col="symbol")
        self._engine(feed, BacktestConfig(execution_price=ExecutionPrice.CLOSE))
