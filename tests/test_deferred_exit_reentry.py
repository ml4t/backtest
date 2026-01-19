"""Tests for deferred exit and re-entry edge cases.

These tests verify behavior in NEXT_BAR execution mode, specifically:
1. Deferred exits are processed at next bar's open
2. Same-bar re-entry after stop exit is prevented (VBT Pro compatibility)
3. Re-entry is allowed on the following bar
4. Exit→Entry sequence is properly ordered
"""

from datetime import datetime

import pytest

from ml4t.backtest import Broker
from ml4t.backtest.models import NoCommission, NoSlippage
from ml4t.backtest.risk import StopLoss
from ml4t.backtest.types import ExecutionMode, OrderSide, StopFillMode


class TestDeferredExitReentry:
    """Test deferred exit with immediate re-entry attempt."""

    def test_same_bar_reentry_blocked_after_stop_exit(self):
        """After stop exit, new entry on same bar should be blocked (VBT Pro behavior)."""
        broker = Broker(
            100000.0,
            NoCommission(),
            NoSlippage(),
            execution_mode=ExecutionMode.SAME_BAR,
        )

        # Bar 1: Enter long position
        broker._update_time(
            timestamp=datetime(2024, 1, 1, 9, 30),
            prices={"AAPL": 100.0},
            opens={"AAPL": 100.0},
            highs={"AAPL": 101.0},
            lows={"AAPL": 99.0},
            volumes={"AAPL": 10000.0},
            signals={},
        )
        broker.submit_order("AAPL", 100.0, OrderSide.BUY)
        broker._process_orders()

        pos = broker.get_position("AAPL")
        assert pos is not None
        assert pos.quantity == 100.0

        # Set up stop loss rule
        broker.set_position_rules(StopLoss(pct=0.05))

        # Bar 2: Price drops, triggering stop loss
        broker._update_time(
            timestamp=datetime(2024, 1, 2, 9, 30),
            prices={"AAPL": 94.0},
            opens={"AAPL": 96.0},
            highs={"AAPL": 97.0},
            lows={"AAPL": 93.0},
            volumes={"AAPL": 10000.0},
            signals={},
        )

        # Evaluate rules - should trigger stop loss
        exit_orders = broker.evaluate_position_rules()
        assert len(exit_orders) == 1
        broker._process_orders()

        # Position should be closed
        pos = broker.get_position("AAPL")
        assert pos is None or pos.quantity == 0

        # Verify stop exit was recorded
        assert "AAPL" in broker._stop_exits_this_bar

        # Try to re-enter on same bar - should be blocked
        entry_order = broker.submit_order("AAPL", 100.0, OrderSide.BUY)
        assert entry_order is None  # Blocked due to same-bar stop exit

    def test_reentry_allowed_next_bar(self):
        """Re-entry should be allowed on the bar after stop exit."""
        broker = Broker(
            100000.0,
            NoCommission(),
            NoSlippage(),
            execution_mode=ExecutionMode.SAME_BAR,
        )

        # Bar 1: Enter long position
        broker._update_time(
            timestamp=datetime(2024, 1, 1, 9, 30),
            prices={"AAPL": 100.0},
            opens={"AAPL": 100.0},
            highs={"AAPL": 101.0},
            lows={"AAPL": 99.0},
            volumes={"AAPL": 10000.0},
            signals={},
        )
        broker.submit_order("AAPL", 100.0, OrderSide.BUY)
        broker._process_orders()

        broker.set_position_rules(StopLoss(pct=0.05))

        # Bar 2: Stop loss triggered
        broker._update_time(
            timestamp=datetime(2024, 1, 2, 9, 30),
            prices={"AAPL": 94.0},
            opens={"AAPL": 96.0},
            highs={"AAPL": 97.0},
            lows={"AAPL": 93.0},
            volumes={"AAPL": 10000.0},
            signals={},
        )
        broker.evaluate_position_rules()
        broker._process_orders()

        # Bar 3: New bar - re-entry should be allowed
        broker._update_time(
            timestamp=datetime(2024, 1, 3, 9, 30),
            prices={"AAPL": 95.0},
            opens={"AAPL": 94.0},
            highs={"AAPL": 96.0},
            lows={"AAPL": 93.0},
            volumes={"AAPL": 10000.0},
            signals={},
        )

        # Stop exits tracking cleared on new bar
        assert "AAPL" not in broker._stop_exits_this_bar

        entry_order = broker.submit_order("AAPL", 100.0, OrderSide.BUY)
        assert entry_order is not None  # Allowed on next bar

        broker._process_orders()
        pos = broker.get_position("AAPL")
        assert pos is not None
        assert pos.quantity == 100.0


class TestNextBarModeExitThenEntry:
    """Test NEXT_BAR mode specific edge cases."""

    def test_deferred_exit_fills_at_open(self):
        """Deferred exit should fill at next bar's open price."""
        broker = Broker(
            100000.0,
            NoCommission(),
            NoSlippage(),
            execution_mode=ExecutionMode.NEXT_BAR,
            stop_fill_mode=StopFillMode.NEXT_BAR_OPEN,
        )

        # Bar 1: Enter position
        broker._update_time(
            timestamp=datetime(2024, 1, 1, 9, 30),
            prices={"AAPL": 100.0},
            opens={"AAPL": 100.0},
            highs={"AAPL": 101.0},
            lows={"AAPL": 99.0},
            volumes={"AAPL": 10000.0},
            signals={},
        )
        broker.submit_order("AAPL", 100.0, OrderSide.BUY)

        # Bar 2: Process entry (NEXT_BAR mode)
        broker._update_time(
            timestamp=datetime(2024, 1, 2, 9, 30),
            prices={"AAPL": 101.0},
            opens={"AAPL": 100.5},
            highs={"AAPL": 102.0},
            lows={"AAPL": 100.0},
            volumes={"AAPL": 10000.0},
            signals={},
        )
        broker._process_orders(use_open=True)

        pos = broker.get_position("AAPL")
        assert pos is not None
        assert pos.entry_price == 100.5  # Filled at open

        broker.set_position_rules(StopLoss(pct=0.05))

        # Bar 3: Trigger stop loss (deferred)
        broker._update_time(
            timestamp=datetime(2024, 1, 3, 9, 30),
            prices={"AAPL": 94.0},
            opens={"AAPL": 97.0},
            highs={"AAPL": 98.0},
            lows={"AAPL": 93.0},
            volumes={"AAPL": 10000.0},
            signals={},
        )
        exit_orders = broker.evaluate_position_rules()

        # In NEXT_BAR_OPEN mode, exit is deferred
        assert len(exit_orders) == 0
        assert "AAPL" in broker._pending_exits

        # Bar 4: Deferred exit processes at this bar's open
        broker._update_time(
            timestamp=datetime(2024, 1, 4, 9, 30),
            prices={"AAPL": 92.0},
            opens={"AAPL": 93.5},  # This is the fill price
            highs={"AAPL": 94.0},
            lows={"AAPL": 91.0},
            volumes={"AAPL": 10000.0},
            signals={},
        )

        exit_orders = broker._process_pending_exits()
        assert len(exit_orders) == 1
        assert exit_orders[0]._risk_fill_price == 93.5  # Filled at open

    def test_entry_order_deferred_in_next_bar_mode(self):
        """Entry orders should be deferred to next bar in NEXT_BAR mode."""
        broker = Broker(
            100000.0,
            NoCommission(),
            NoSlippage(),
            execution_mode=ExecutionMode.NEXT_BAR,
        )

        # Bar 1: Submit order
        broker._update_time(
            timestamp=datetime(2024, 1, 1, 9, 30),
            prices={"AAPL": 100.0},
            opens={"AAPL": 100.0},
            highs={"AAPL": 101.0},
            lows={"AAPL": 99.0},
            volumes={"AAPL": 10000.0},
            signals={},
        )
        order = broker.submit_order("AAPL", 100.0, OrderSide.BUY)
        assert order is not None

        # Order tracked for next-bar mode
        assert order in broker._orders_this_bar

        # Process orders - entry should be skipped (it's a this-bar order)
        broker._process_orders(use_open=True)

        # No position yet
        pos = broker.get_position("AAPL")
        assert pos is None

        # Bar 2: Now the order executes
        broker._update_time(
            timestamp=datetime(2024, 1, 2, 9, 30),
            prices={"AAPL": 102.0},
            opens={"AAPL": 101.0},
            highs={"AAPL": 103.0},
            lows={"AAPL": 100.0},
            volumes={"AAPL": 10000.0},
            signals={},
        )

        broker._process_orders(use_open=True)

        pos = broker.get_position("AAPL")
        assert pos is not None
        assert pos.entry_price == 101.0  # Filled at bar 2 open


class TestExitBeforeEntryOrdering:
    """Test that exits are processed before entries each bar."""

    def test_exit_frees_capital_for_entry(self):
        """Exit should free capital that can be used for new entry same bar."""
        broker = Broker(
            10000.0,  # Limited cash
            NoCommission(),
            NoSlippage(),
            execution_mode=ExecutionMode.SAME_BAR,
        )

        # Bar 1: Enter position using most of capital
        broker._update_time(
            timestamp=datetime(2024, 1, 1, 9, 30),
            prices={"AAPL": 100.0, "GOOGL": 100.0},
            opens={"AAPL": 100.0, "GOOGL": 100.0},
            highs={"AAPL": 101.0, "GOOGL": 101.0},
            lows={"AAPL": 99.0, "GOOGL": 99.0},
            volumes={"AAPL": 10000.0, "GOOGL": 10000.0},
            signals={},
        )

        broker.submit_order("AAPL", 90.0, OrderSide.BUY)  # $9000
        broker._process_orders()

        assert broker.cash < 2000  # Mostly invested

        # Bar 2: Exit AAPL and enter GOOGL - should work due to exit-first
        broker._update_time(
            timestamp=datetime(2024, 1, 2, 9, 30),
            prices={"AAPL": 105.0, "GOOGL": 100.0},
            opens={"AAPL": 104.0, "GOOGL": 100.0},
            highs={"AAPL": 106.0, "GOOGL": 101.0},
            lows={"AAPL": 103.0, "GOOGL": 99.0},
            volumes={"AAPL": 10000.0, "GOOGL": 10000.0},
            signals={},
        )

        # Submit exit for AAPL and entry for GOOGL
        broker.close_position("AAPL")
        broker.submit_order("GOOGL", 90.0, OrderSide.BUY)

        broker._process_orders()

        # Both should have executed (exit frees capital for entry)
        aapl_pos = broker.get_position("AAPL")
        googl_pos = broker.get_position("GOOGL")

        assert aapl_pos is None or aapl_pos.quantity == 0
        assert googl_pos is not None
        assert googl_pos.quantity == 90.0


class TestMultiAssetDeferredExits:
    """Test deferred exits with multiple assets."""

    def test_multiple_deferred_exits_same_bar(self):
        """Multiple assets can have deferred exits on the same bar."""
        broker = Broker(
            100000.0,
            NoCommission(),
            NoSlippage(),
            stop_fill_mode=StopFillMode.NEXT_BAR_OPEN,
        )

        # Enter positions in both assets
        broker._update_time(
            timestamp=datetime(2024, 1, 1, 9, 30),
            prices={"AAPL": 100.0, "GOOGL": 200.0},
            opens={"AAPL": 100.0, "GOOGL": 200.0},
            highs={"AAPL": 101.0, "GOOGL": 201.0},
            lows={"AAPL": 99.0, "GOOGL": 199.0},
            volumes={"AAPL": 10000.0, "GOOGL": 10000.0},
            signals={},
        )
        broker.submit_order("AAPL", 100.0, OrderSide.BUY)
        broker.submit_order("GOOGL", 50.0, OrderSide.BUY)
        broker._process_orders()

        broker.set_position_rules(StopLoss(pct=0.05))

        # Both stop losses trigger
        broker._update_time(
            timestamp=datetime(2024, 1, 2, 9, 30),
            prices={"AAPL": 94.0, "GOOGL": 188.0},
            opens={"AAPL": 96.0, "GOOGL": 192.0},
            highs={"AAPL": 97.0, "GOOGL": 194.0},
            lows={"AAPL": 93.0, "GOOGL": 187.0},
            volumes={"AAPL": 10000.0, "GOOGL": 10000.0},
            signals={},
        )
        broker.evaluate_position_rules()

        # Both should be in pending exits
        assert "AAPL" in broker._pending_exits
        assert "GOOGL" in broker._pending_exits

        # Process on next bar
        broker._update_time(
            timestamp=datetime(2024, 1, 3, 9, 30),
            prices={"AAPL": 92.0, "GOOGL": 185.0},
            opens={"AAPL": 93.0, "GOOGL": 186.0},
            highs={"AAPL": 94.0, "GOOGL": 187.0},
            lows={"AAPL": 91.0, "GOOGL": 184.0},
            volumes={"AAPL": 10000.0, "GOOGL": 10000.0},
            signals={},
        )

        exit_orders = broker._process_pending_exits()
        assert len(exit_orders) == 2

        # All pending exits cleared
        assert "AAPL" not in broker._pending_exits
        assert "GOOGL" not in broker._pending_exits
