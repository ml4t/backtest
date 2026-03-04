"""Direction x Order Type matrix tests.

Tests that limit/stop orders fill at correct prices in both long and short
directions, with explicit OHLC bars and round-trip PnL verification.
"""

from __future__ import annotations

import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine, ExecutionMode, OrderType
from ml4t.backtest.config import ExecutionPrice

from .helpers import OrderTypeStrategy, make_ohlcv_prices

# Use SAME_BAR + CLOSE execution for deterministic fill prices in tests.
_CONFIG = BacktestConfig(
    initial_cash=100_000.0,
    commission_rate=0.0,
    slippage_rate=0.0,
    execution_mode=ExecutionMode.SAME_BAR,
    execution_price=ExecutionPrice.CLOSE,
    allow_short_selling=True,
)


def _run(bars, strategy, config=None):
    """Helper to build Engine and run."""
    prices = make_ohlcv_prices(bars)
    if config is None:
        config = _CONFIG
    return Engine(DataFeed(prices_df=prices), strategy, config).run()


@pytest.fixture(params=["long", "short"])
def direction(request):
    return request.param


# ---------------------------------------------------------------------------
# Limit orders
# ---------------------------------------------------------------------------


class TestLimitFills:
    """Verify limit orders fill at exactly the limit price."""

    def test_limit_fills_at_limit_price(self, direction):
        """Bar touches limit -> fill at exactly limit price."""
        if direction == "long":
            bars = [
                (100.0, 101.0, 99.0, 100.0),  # bar 0: submit limit buy at 98
                (100.0, 100.0, 97.0, 99.0),  # bar 1: low=97 touches 98 -> fill
                (99.0, 102.0, 99.0, 101.0),  # bar 2: hold
                (101.0, 103.0, 101.0, 102.0),  # bar 3: hold
                (102.0, 103.0, 101.0, 102.0),  # bar 4: exit
            ]
            limit_price = 98.0
        else:
            bars = [
                (100.0, 101.0, 99.0, 100.0),  # bar 0: submit limit sell at 102
                (100.0, 103.0, 100.0, 101.0),  # bar 1: high=103 touches 102 -> fill
                (101.0, 101.0, 98.0, 99.0),  # bar 2: hold
                (99.0, 99.0, 97.0, 98.0),  # bar 3: hold
                (98.0, 99.0, 97.0, 98.0),  # bar 4: exit
            ]
            limit_price = 102.0

        strategy = OrderTypeStrategy(
            direction=direction,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            entry_bar=0,
            exit_bar=4,
        )
        result = _run(bars, strategy)

        assert len(result.trades) == 1
        assert result.trades[0].entry_price == limit_price
        assert result.trades[0].status == "closed"

    def test_limit_no_fill_if_not_touched(self, direction):
        """Bar doesn't touch limit -> no position opened."""
        if direction == "long":
            bars = [
                (100.0, 101.0, 99.0, 100.0),
                (100.0, 102.0, 98.0, 101.0),
                (101.0, 103.0, 100.0, 102.0),
                (102.0, 104.0, 101.0, 103.0),
            ]
            limit_price = 95.0
        else:
            bars = [
                (100.0, 101.0, 99.0, 100.0),
                (100.0, 102.0, 98.0, 101.0),
                (101.0, 103.0, 100.0, 102.0),
                (102.0, 104.0, 101.0, 103.0),
            ]
            limit_price = 105.0

        strategy = OrderTypeStrategy(
            direction=direction,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            entry_bar=0,
            exit_bar=3,
        )
        result = _run(bars, strategy)

        closed = [t for t in result.trades if t.status == "closed"]
        assert len(closed) == 0

    def test_limit_entry_market_exit_pnl(self, direction):
        """Full round-trip: limit entry, market exit, verify PnL."""
        if direction == "long":
            bars = [
                (100.0, 101.0, 99.0, 100.0),  # bar 0: submit limit buy at 98
                (100.0, 100.0, 97.0, 99.0),  # bar 1: fill at 98
                (99.0, 103.0, 99.0, 102.0),  # bar 2: hold
                (102.0, 104.0, 102.0, 103.0),  # bar 3: exit at close=103
            ]
            limit_price = 98.0
            expected_pnl = (103.0 - 98.0) * 100.0  # +500
        else:
            bars = [
                (100.0, 101.0, 99.0, 100.0),  # bar 0: submit limit sell at 102
                (100.0, 103.0, 100.0, 101.0),  # bar 1: fill at 102
                (101.0, 101.0, 97.0, 98.0),  # bar 2: hold
                (98.0, 98.0, 96.0, 97.0),  # bar 3: exit at close=97
            ]
            limit_price = 102.0
            expected_pnl = (102.0 - 97.0) * 100.0  # +500

        strategy = OrderTypeStrategy(
            direction=direction,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            entry_bar=0,
            exit_bar=3,
        )
        result = _run(bars, strategy)

        closed = [t for t in result.trades if t.status == "closed"]
        assert len(closed) == 1
        assert abs(closed[0].pnl - expected_pnl) < 0.01


# ---------------------------------------------------------------------------
# Stop orders
# ---------------------------------------------------------------------------


class TestStopFills:
    """Verify stop orders fill at correct prices."""

    def test_stop_fills_when_triggered(self, direction):
        """Bar breaches stop -> fill at stop price."""
        if direction == "long":
            bars = [
                (100.0, 101.0, 99.0, 100.0),
                (100.0, 103.0, 99.0, 101.0),  # high=103 triggers 102
                (101.0, 104.0, 101.0, 103.0),
                (103.0, 105.0, 103.0, 104.0),
                (104.0, 105.0, 103.0, 104.0),
            ]
            stop_price = 102.0
        else:
            bars = [
                (100.0, 101.0, 99.0, 100.0),
                (100.0, 101.0, 97.0, 99.0),  # low=97 triggers 98
                (99.0, 99.0, 96.0, 97.0),
                (97.0, 97.0, 95.0, 96.0),
                (96.0, 97.0, 95.0, 96.0),
            ]
            stop_price = 98.0

        strategy = OrderTypeStrategy(
            direction=direction,
            order_type=OrderType.STOP,
            stop_price=stop_price,
            entry_bar=0,
            exit_bar=4,
        )
        result = _run(bars, strategy)

        closed = [t for t in result.trades if t.status == "closed"]
        assert len(closed) == 1
        assert closed[0].entry_price == stop_price

    def test_stop_gap_through_fills_at_open(self, direction):
        """Gap through stop -> fill at bar open (not stop price)."""
        if direction == "long":
            bars = [
                (100.0, 101.0, 99.0, 100.0),
                (105.0, 107.0, 104.0, 106.0),  # gap open 105 > stop 102
                (106.0, 108.0, 106.0, 107.0),
                (107.0, 108.0, 106.0, 107.0),
            ]
            stop_price = 102.0
            expected_fill = 105.0
        else:
            bars = [
                (100.0, 101.0, 99.0, 100.0),
                (95.0, 96.0, 94.0, 95.0),  # gap open 95 < stop 98
                (95.0, 95.0, 93.0, 94.0),
                (94.0, 95.0, 93.0, 94.0),
            ]
            stop_price = 98.0
            expected_fill = 95.0

        strategy = OrderTypeStrategy(
            direction=direction,
            order_type=OrderType.STOP,
            stop_price=stop_price,
            entry_bar=0,
            exit_bar=3,
        )
        result = _run(bars, strategy)

        closed = [t for t in result.trades if t.status == "closed"]
        assert len(closed) == 1
        assert closed[0].entry_price == expected_fill

    def test_stop_no_fill_if_not_triggered(self, direction):
        """Stop level not breached -> no fill."""
        if direction == "long":
            bars = [
                (100.0, 101.0, 99.0, 100.0),
                (100.0, 103.0, 99.0, 102.0),
                (102.0, 104.0, 101.0, 103.0),
            ]
            stop_price = 105.0
        else:
            bars = [
                (100.0, 101.0, 99.0, 100.0),
                (100.0, 101.0, 97.0, 99.0),
                (99.0, 100.0, 96.0, 98.0),
            ]
            stop_price = 95.0

        strategy = OrderTypeStrategy(
            direction=direction,
            order_type=OrderType.STOP,
            stop_price=stop_price,
            entry_bar=0,
            exit_bar=2,
        )
        result = _run(bars, strategy)

        closed = [t for t in result.trades if t.status == "closed"]
        assert len(closed) == 0

    def test_stop_entry_market_exit_pnl(self, direction):
        """Full round-trip: stop entry, market exit, verify PnL."""
        if direction == "long":
            bars = [
                (100.0, 101.0, 99.0, 100.0),
                (100.0, 103.0, 100.0, 102.0),  # triggered at 102
                (102.0, 105.0, 102.0, 104.0),  # exit at close=104
            ]
            stop_price = 102.0
            expected_pnl = (104.0 - 102.0) * 100.0
        else:
            bars = [
                (100.0, 101.0, 99.0, 100.0),
                (100.0, 100.0, 97.0, 98.0),  # triggered at 98
                (98.0, 98.0, 95.0, 96.0),  # exit at close=96
            ]
            stop_price = 98.0
            expected_pnl = (98.0 - 96.0) * 100.0

        strategy = OrderTypeStrategy(
            direction=direction,
            order_type=OrderType.STOP,
            stop_price=stop_price,
            entry_bar=0,
            exit_bar=2,
        )
        result = _run(bars, strategy)

        closed = [t for t in result.trades if t.status == "closed"]
        assert len(closed) == 1
        assert abs(closed[0].pnl - expected_pnl) < 0.01


# ---------------------------------------------------------------------------
# Fill metadata verification
# ---------------------------------------------------------------------------


class TestFillMetadata:
    """Verify Fill dataclass carries order-type metadata."""

    def test_limit_fill_has_metadata(self):
        """Limit fill carries order_type and limit_price."""
        bars = [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 100.0, 97.0, 99.0),
            (99.0, 102.0, 99.0, 101.0),
        ]
        strategy = OrderTypeStrategy(
            direction="long",
            order_type=OrderType.LIMIT,
            limit_price=98.0,
            entry_bar=0,
            exit_bar=2,
        )
        result = _run(bars, strategy)

        limit_fills = [f for f in result.fills if f.order_type == "limit"]
        assert len(limit_fills) == 1
        assert limit_fills[0].limit_price == 98.0
        assert limit_fills[0].price == 98.0

    def test_stop_fill_has_metadata(self):
        """Stop fill carries order_type and stop_price."""
        bars = [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 103.0, 100.0, 102.0),
            (102.0, 104.0, 102.0, 103.0),
        ]
        strategy = OrderTypeStrategy(
            direction="long",
            order_type=OrderType.STOP,
            stop_price=102.0,
            entry_bar=0,
            exit_bar=2,
        )
        result = _run(bars, strategy)

        stop_fills = [f for f in result.fills if f.order_type == "stop"]
        assert len(stop_fills) == 1
        assert stop_fills[0].stop_price == 102.0
        assert stop_fills[0].price == 102.0

    def test_market_fill_has_metadata(self):
        """Market fill carries order_type='market'."""
        bars = [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 102.0, 99.0, 101.0),
        ]
        strategy = OrderTypeStrategy(
            direction="long",
            order_type=OrderType.MARKET,
            entry_bar=0,
            exit_bar=1,
        )
        result = _run(bars, strategy)

        market_fills = [f for f in result.fills if f.order_type == "market"]
        assert len(market_fills) >= 1
        assert market_fills[0].limit_price is None
        assert market_fills[0].stop_price is None
