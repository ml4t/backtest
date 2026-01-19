"""Tests for bracket orders with short positions.

These tests verify:
1. Short entry bracket orders have correct BUY exit sides
2. Take-profit triggers on price decrease (profit for short)
3. Stop-loss triggers on price increase (loss for short)
4. Price validation warnings for inverted prices
"""

import warnings
from datetime import datetime

import pytest

from ml4t.backtest import Broker
from ml4t.backtest.models import NoCommission, NoSlippage
from ml4t.backtest.types import OrderSide, OrderStatus, OrderType


class TestBracketOrdersShortEntry:
    """Test bracket orders for short positions."""

    def test_short_bracket_exit_sides(self):
        """Short entry brackets should have BUY exits."""
        broker = Broker(100000.0, NoCommission(), NoSlippage(), account_type="margin")
        broker._update_time(
            timestamp=datetime(2024, 1, 1, 9, 30),
            prices={"AAPL": 150.0},
            opens={"AAPL": 150.0},
            highs={"AAPL": 151.0},
            lows={"AAPL": 149.0},
            volumes={"AAPL": 10000.0},
            signals={},
        )

        # Short entry with TP below and SL above
        result = broker.submit_bracket(
            "AAPL", -100.0, take_profit=140.0, stop_loss=160.0, validate_prices=False
        )
        assert result is not None

        entry, tp, sl = result
        assert entry.side == OrderSide.SELL  # Short entry
        assert tp.side == OrderSide.BUY  # Buy to cover at profit
        assert sl.side == OrderSide.BUY  # Buy to cover at stop

    def test_short_bracket_take_profit_triggers(self):
        """Short bracket take-profit should trigger when price drops."""
        broker = Broker(100000.0, NoCommission(), NoSlippage(), account_type="margin")

        # Entry bar
        broker._update_time(
            timestamp=datetime(2024, 1, 1, 9, 30),
            prices={"AAPL": 150.0},
            opens={"AAPL": 150.0},
            highs={"AAPL": 151.0},
            lows={"AAPL": 149.0},
            volumes={"AAPL": 10000.0},
            signals={},
        )

        result = broker.submit_bracket(
            "AAPL", -100.0, take_profit=140.0, stop_loss=160.0, validate_prices=False
        )
        assert result is not None
        entry, tp, sl = result

        # Process entry
        broker._process_orders()
        pos = broker.get_position("AAPL")
        assert pos is not None
        assert pos.quantity == -100.0  # Short position

        # Price drops to TP level
        broker._update_time(
            timestamp=datetime(2024, 1, 2, 9, 30),
            prices={"AAPL": 138.0},
            opens={"AAPL": 145.0},
            highs={"AAPL": 146.0},
            lows={"AAPL": 137.0},  # Low touches TP
            volumes={"AAPL": 10000.0},
            signals={},
        )

        broker._process_orders()

        # TP should have filled
        assert tp.status == OrderStatus.FILLED
        pos = broker.get_position("AAPL")
        assert pos is None or pos.quantity == 0

    def test_short_bracket_stop_loss_triggers(self):
        """Short bracket stop-loss should trigger when price rises."""
        broker = Broker(100000.0, NoCommission(), NoSlippage(), account_type="margin")

        broker._update_time(
            timestamp=datetime(2024, 1, 1, 9, 30),
            prices={"AAPL": 150.0},
            opens={"AAPL": 150.0},
            highs={"AAPL": 151.0},
            lows={"AAPL": 149.0},
            volumes={"AAPL": 10000.0},
            signals={},
        )

        result = broker.submit_bracket(
            "AAPL", -100.0, take_profit=140.0, stop_loss=160.0, validate_prices=False
        )
        assert result is not None
        entry, tp, sl = result

        broker._process_orders()

        # Price rises to SL level
        broker._update_time(
            timestamp=datetime(2024, 1, 2, 9, 30),
            prices={"AAPL": 162.0},
            opens={"AAPL": 155.0},
            highs={"AAPL": 163.0},  # High touches SL
            lows={"AAPL": 154.0},
            volumes={"AAPL": 10000.0},
            signals={},
        )

        broker._process_orders()

        # SL should have filled
        assert sl.status == OrderStatus.FILLED
        pos = broker.get_position("AAPL")
        assert pos is None or pos.quantity == 0


class TestBracketPriceValidation:
    """Test bracket order price validation warnings."""

    def test_long_bracket_inverted_tp_warning(self):
        """Long bracket with TP below entry should warn."""
        broker = Broker(100000.0, NoCommission(), NoSlippage())
        broker._update_time(
            timestamp=datetime(2024, 1, 1, 9, 30),
            prices={"AAPL": 150.0},
            opens={"AAPL": 150.0},
            highs={"AAPL": 151.0},
            lows={"AAPL": 149.0},
            volumes={"AAPL": 10000.0},
            signals={},
        )

        # TP below entry for long - wrong!
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            broker.submit_bracket("AAPL", 100.0, take_profit=140.0, stop_loss=145.0)

            assert len(w) >= 1
            assert "take_profit" in str(w[0].message).lower()
            assert "LONG" in str(w[0].message)

    def test_long_bracket_inverted_sl_warning(self):
        """Long bracket with SL above entry should warn."""
        broker = Broker(100000.0, NoCommission(), NoSlippage())
        broker._update_time(
            timestamp=datetime(2024, 1, 1, 9, 30),
            prices={"AAPL": 150.0},
            opens={"AAPL": 150.0},
            highs={"AAPL": 151.0},
            lows={"AAPL": 149.0},
            volumes={"AAPL": 10000.0},
            signals={},
        )

        # SL above entry for long - wrong!
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            broker.submit_bracket("AAPL", 100.0, take_profit=160.0, stop_loss=155.0)

            assert len(w) >= 1
            assert "stop_loss" in str(w[0].message).lower()
            assert "LONG" in str(w[0].message)

    def test_short_bracket_inverted_tp_warning(self):
        """Short bracket with TP above entry should warn."""
        broker = Broker(100000.0, NoCommission(), NoSlippage(), account_type="margin")
        broker._update_time(
            timestamp=datetime(2024, 1, 1, 9, 30),
            prices={"AAPL": 150.0},
            opens={"AAPL": 150.0},
            highs={"AAPL": 151.0},
            lows={"AAPL": 149.0},
            volumes={"AAPL": 10000.0},
            signals={},
        )

        # TP above entry for short - wrong!
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            broker.submit_bracket("AAPL", -100.0, take_profit=160.0, stop_loss=155.0)

            assert len(w) >= 1
            assert "take_profit" in str(w[0].message).lower()
            assert "SHORT" in str(w[0].message)

    def test_short_bracket_inverted_sl_warning(self):
        """Short bracket with SL below entry should warn."""
        broker = Broker(100000.0, NoCommission(), NoSlippage(), account_type="margin")
        broker._update_time(
            timestamp=datetime(2024, 1, 1, 9, 30),
            prices={"AAPL": 150.0},
            opens={"AAPL": 150.0},
            highs={"AAPL": 151.0},
            lows={"AAPL": 149.0},
            volumes={"AAPL": 10000.0},
            signals={},
        )

        # SL below entry for short - wrong!
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            broker.submit_bracket("AAPL", -100.0, take_profit=140.0, stop_loss=145.0)

            assert len(w) >= 1
            assert "stop_loss" in str(w[0].message).lower()
            assert "SHORT" in str(w[0].message)

    def test_valid_long_bracket_no_warning(self):
        """Valid long bracket should not warn."""
        broker = Broker(100000.0, NoCommission(), NoSlippage())
        broker._update_time(
            timestamp=datetime(2024, 1, 1, 9, 30),
            prices={"AAPL": 150.0},
            opens={"AAPL": 150.0},
            highs={"AAPL": 151.0},
            lows={"AAPL": 149.0},
            volumes={"AAPL": 10000.0},
            signals={},
        )

        # Correct: TP above, SL below for long
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = broker.submit_bracket("AAPL", 100.0, take_profit=160.0, stop_loss=140.0)

            assert result is not None
            assert len(w) == 0

    def test_valid_short_bracket_no_warning(self):
        """Valid short bracket should not warn."""
        broker = Broker(100000.0, NoCommission(), NoSlippage(), account_type="margin")
        broker._update_time(
            timestamp=datetime(2024, 1, 1, 9, 30),
            prices={"AAPL": 150.0},
            opens={"AAPL": 150.0},
            highs={"AAPL": 151.0},
            lows={"AAPL": 149.0},
            volumes={"AAPL": 10000.0},
            signals={},
        )

        # Correct: TP below, SL above for short
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = broker.submit_bracket("AAPL", -100.0, take_profit=140.0, stop_loss=160.0)

            assert result is not None
            assert len(w) == 0

    def test_validation_can_be_disabled(self):
        """Price validation can be disabled."""
        broker = Broker(100000.0, NoCommission(), NoSlippage())
        broker._update_time(
            timestamp=datetime(2024, 1, 1, 9, 30),
            prices={"AAPL": 150.0},
            opens={"AAPL": 150.0},
            highs={"AAPL": 151.0},
            lows={"AAPL": 149.0},
            volumes={"AAPL": 10000.0},
            signals={},
        )

        # Inverted prices but validation disabled
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = broker.submit_bracket(
                "AAPL", 100.0, take_profit=140.0, stop_loss=155.0, validate_prices=False
            )

            assert result is not None
            assert len(w) == 0


class TestBracketOrdersWithLimitEntry:
    """Test bracket orders with limit entry prices."""

    def test_limit_entry_uses_limit_price_for_validation(self):
        """Bracket validation should use limit price, not market price."""
        broker = Broker(100000.0, NoCommission(), NoSlippage())
        broker._update_time(
            timestamp=datetime(2024, 1, 1, 9, 30),
            prices={"AAPL": 150.0},
            opens={"AAPL": 150.0},
            highs={"AAPL": 151.0},
            lows={"AAPL": 149.0},
            volumes={"AAPL": 10000.0},
            signals={},
        )

        # Limit entry at 145, TP at 155, SL at 140 - all relative to limit
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = broker.submit_bracket(
                "AAPL",
                100.0,
                take_profit=155.0,
                stop_loss=140.0,
                entry_type=OrderType.LIMIT,
                entry_limit=145.0,
            )

            assert result is not None
            assert len(w) == 0  # Valid relative to limit price

    def test_limit_entry_short_validation(self):
        """Short bracket with limit entry should validate against limit."""
        broker = Broker(100000.0, NoCommission(), NoSlippage(), account_type="margin")
        broker._update_time(
            timestamp=datetime(2024, 1, 1, 9, 30),
            prices={"AAPL": 150.0},
            opens={"AAPL": 150.0},
            highs={"AAPL": 151.0},
            lows={"AAPL": 149.0},
            volumes={"AAPL": 10000.0},
            signals={},
        )

        # Short at 155, TP at 145, SL at 165 - all relative to limit
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = broker.submit_bracket(
                "AAPL",
                -100.0,
                take_profit=145.0,
                stop_loss=165.0,
                entry_type=OrderType.LIMIT,
                entry_limit=155.0,
            )

            assert result is not None
            assert len(w) == 0  # Valid relative to limit price
