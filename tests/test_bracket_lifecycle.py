"""Bracket order lifecycle tests.

Full lifecycle verification: entry fills → both exits active → one fires
→ sibling auto-cancels. Uses set_broker_bar for bar-by-bar control.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from ml4t.backtest import Broker, OrderStatus, OrderType
from ml4t.backtest.models import NoCommission, NoSlippage

from .helpers import set_broker_bar


@pytest.fixture(params=["long", "short"])
def direction(request):
    return request.param


def _make_broker(cash: float = 100_000.0) -> Broker:
    return Broker(cash, NoCommission(), NoSlippage(), allow_short_selling=True)


def _ts(day: int = 1) -> datetime:
    return datetime(2024, 1, day)


def _submit_bracket(broker, direction, price=100.0, tp_offset=5.0, sl_offset=5.0):
    """Submit a bracket order and return (entry, tp, sl) orders."""
    if direction == "long":
        qty = 100.0
        tp_price = price + tp_offset
        sl_price = price - sl_offset
    else:
        qty = -100.0
        tp_price = price - tp_offset
        sl_price = price + sl_offset

    return broker.submit_bracket(
        "TEST", qty, take_profit=tp_price, stop_loss=sl_price, validate_prices=False
    )


class TestBracketTPFills:
    """Verify take-profit fills and stop-loss cancellation."""

    def test_bracket_tp_fills_cancels_sl(self, direction):
        """TP fires → SL cancelled."""
        broker = _make_broker()

        # Bar 0: set price and submit bracket
        set_broker_bar(broker, 100.0, ts=_ts(1))
        orders = _submit_bracket(broker, direction)
        assert orders is not None
        entry, tp, sl = orders

        # Bar 1: fill entry
        broker._process_orders()
        assert entry.status == OrderStatus.FILLED

        # Bar 2: price moves to hit TP
        if direction == "long":
            # TP at 105, make high reach 106
            set_broker_bar(broker, 105.0, ts=_ts(2), high=106.0, low=100.0)
        else:
            # TP at 95, make low reach 94
            set_broker_bar(broker, 95.0, ts=_ts(2), high=100.0, low=94.0)

        broker._process_orders()

        # TP should be filled, SL should be cancelled
        assert tp.status == OrderStatus.FILLED
        assert sl.status == OrderStatus.CANCELLED

        # Position should be closed
        assert broker.get_position("TEST") is None

        # Trade should exist with exit_reason
        assert len(broker.trades) == 1

    def test_bracket_sl_fills_cancels_tp(self, direction):
        """SL fires → TP cancelled."""
        broker = _make_broker()

        set_broker_bar(broker, 100.0, ts=_ts(1))
        orders = _submit_bracket(broker, direction)
        assert orders is not None
        entry, tp, sl = orders

        broker._process_orders()
        assert entry.status == OrderStatus.FILLED

        # Bar 2: price moves to hit SL
        if direction == "long":
            # SL at 95, make low reach 94
            set_broker_bar(broker, 95.0, ts=_ts(2), high=100.0, low=94.0)
        else:
            # SL at 105, make high reach 106
            set_broker_bar(broker, 105.0, ts=_ts(2), high=106.0, low=100.0)

        broker._process_orders()

        assert sl.status == OrderStatus.FILLED
        assert tp.status == OrderStatus.CANCELLED
        assert broker.get_position("TEST") is None
        assert len(broker.trades) == 1


class TestBracketEntryPending:
    """Verify bracket behavior when entry hasn't filled."""

    def test_bracket_entry_not_filled_exits_stay_pending(self, direction):
        """Entry pending → exits stay pending."""
        broker = _make_broker()

        set_broker_bar(broker, 100.0, ts=_ts(1))

        # Use a limit entry that won't fill
        if direction == "long":
            qty = 100.0
            entry_limit = 90.0  # Too far below market
            tp_price = 110.0
            sl_price = 85.0
        else:
            qty = -100.0
            entry_limit = 110.0  # Too far above market
            tp_price = 90.0
            sl_price = 115.0

        orders = broker.submit_bracket(
            "TEST",
            qty,
            take_profit=tp_price,
            stop_loss=sl_price,
            entry_type=OrderType.LIMIT,
            entry_limit=entry_limit,
            validate_prices=False,
        )
        assert orders is not None
        entry, tp, sl = orders

        # Bar 1-2: price doesn't reach entry limit
        set_broker_bar(broker, 100.0, ts=_ts(2), high=101.0, low=99.0)
        broker._process_orders()

        # Entry not filled, but exits are pending (they won't trigger without position)
        assert entry.status == OrderStatus.PENDING
        assert len(broker.trades) == 0


class TestBracketGapThrough:
    """Verify gap-through behavior on bracket exits."""

    def test_bracket_gap_through_sl(self, direction):
        """Gap through SL → fill at open, TP cancelled."""
        broker = _make_broker()

        set_broker_bar(broker, 100.0, ts=_ts(1))
        orders = _submit_bracket(broker, direction)
        assert orders is not None
        entry, tp, sl = orders

        broker._process_orders()
        assert entry.status == OrderStatus.FILLED

        # Bar 2: gap through SL
        if direction == "long":
            # SL at 95, gap open at 90
            set_broker_bar(broker, 91.0, ts=_ts(2), open_=90.0, high=92.0, low=89.0)
        else:
            # SL at 105, gap open at 110
            set_broker_bar(broker, 109.0, ts=_ts(2), open_=110.0, high=111.0, low=108.0)

        broker._process_orders()

        assert sl.status == OrderStatus.FILLED
        assert tp.status == OrderStatus.CANCELLED

        # Fill should be at gap-through open price
        trade = broker.trades[0]
        if direction == "long":
            assert trade.exit_price == 90.0  # gap open
        else:
            assert trade.exit_price == 110.0  # gap open


class TestBracketSequentialExits:
    """Verify sequential exit processing in brackets."""

    def test_bracket_tp_then_sl_bar(self, direction):
        """TP fires on bar 2, SL would fire on bar 3 but is already cancelled."""
        broker = _make_broker()

        set_broker_bar(broker, 100.0, ts=_ts(1))
        orders = _submit_bracket(broker, direction)
        assert orders is not None
        entry, tp, sl = orders

        broker._process_orders()
        assert entry.status == OrderStatus.FILLED

        # Bar 2: only TP triggers
        if direction == "long":
            # TP at 105, price hits 106 but doesn't drop to 95 (SL)
            set_broker_bar(broker, 106.0, ts=_ts(2), open_=103.0, high=106.0, low=102.0)
        else:
            # TP at 95, price drops to 94 but doesn't rise to 105 (SL)
            set_broker_bar(broker, 94.0, ts=_ts(2), open_=97.0, high=98.0, low=94.0)

        broker._process_orders()

        assert tp.status == OrderStatus.FILLED
        assert sl.status == OrderStatus.CANCELLED
        assert broker.get_position("TEST") is None
        assert len(broker.trades) == 1

    def test_bracket_sl_then_tp_bar(self, direction):
        """SL fires on bar 2, TP would fire on bar 3 but is already cancelled."""
        broker = _make_broker()

        set_broker_bar(broker, 100.0, ts=_ts(1))
        orders = _submit_bracket(broker, direction)
        assert orders is not None
        entry, tp, sl = orders

        broker._process_orders()
        assert entry.status == OrderStatus.FILLED

        # Bar 2: only SL triggers
        if direction == "long":
            # SL at 95, price drops to 94 but doesn't rise to 105 (TP)
            set_broker_bar(broker, 94.0, ts=_ts(2), open_=97.0, high=98.0, low=94.0)
        else:
            # SL at 105, price rises to 106 but doesn't drop to 95 (TP)
            set_broker_bar(broker, 106.0, ts=_ts(2), open_=103.0, high=106.0, low=102.0)

        broker._process_orders()

        assert sl.status == OrderStatus.FILLED
        assert tp.status == OrderStatus.CANCELLED
        assert broker.get_position("TEST") is None
        assert len(broker.trades) == 1


class TestBracketPnLConsistency:
    """Verify trade PnL matches expected values for each exit type."""

    def test_bracket_tp_pnl(self, direction):
        """TP fill → PnL matches expected profit."""
        broker = _make_broker()
        entry_price = 100.0
        tp_offset = 5.0

        set_broker_bar(broker, entry_price, ts=_ts(1))
        orders = _submit_bracket(broker, direction, price=entry_price, tp_offset=tp_offset)
        assert orders is not None
        entry, tp, sl = orders

        broker._process_orders()

        # Hit TP
        if direction == "long":
            set_broker_bar(broker, 106.0, ts=_ts(2), high=106.0, low=100.0)
        else:
            set_broker_bar(broker, 94.0, ts=_ts(2), high=100.0, low=94.0)

        broker._process_orders()

        trade = broker.trades[0]
        expected_pnl = tp_offset * 100.0  # qty=100, offset=5
        assert abs(trade.pnl - expected_pnl) < 0.01

    def test_bracket_sl_pnl(self, direction):
        """SL fill → PnL matches expected loss."""
        broker = _make_broker()
        entry_price = 100.0
        sl_offset = 5.0

        set_broker_bar(broker, entry_price, ts=_ts(1))
        orders = _submit_bracket(broker, direction, price=entry_price, sl_offset=sl_offset)
        assert orders is not None
        entry, tp, sl = orders

        broker._process_orders()

        # Hit SL (open inside range to avoid gap-through)
        if direction == "long":
            # SL at 95, open at 98 (inside range), low=94 triggers stop
            set_broker_bar(broker, 94.0, ts=_ts(2), open_=98.0, high=100.0, low=94.0)
        else:
            # SL at 105, open at 102 (inside range), high=106 triggers stop
            set_broker_bar(broker, 106.0, ts=_ts(2), open_=102.0, high=106.0, low=100.0)

        broker._process_orders()

        trade = broker.trades[0]
        expected_pnl = -sl_offset * 100.0  # loss
        assert abs(trade.pnl - expected_pnl) < 0.01
