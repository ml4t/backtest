"""Tests for Broker class methods."""

from datetime import datetime

import pytest

from ml4t.backtest.broker import Broker
from ml4t.backtest.models import NoCommission, NoSlippage
from ml4t.backtest.types import Order, OrderSide, OrderStatus, OrderType, Position


@pytest.fixture
def broker():
    """Create a basic broker for testing."""
    return Broker(
        initial_cash=100000.0,
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
    )


@pytest.fixture
def broker_with_position(broker):
    """Create broker with an existing position."""
    # Simulate having a position by adding to positions dict
    pos = Position(
        asset="AAPL",
        quantity=100.0,
        entry_price=150.0,
        entry_time=datetime(2024, 1, 1, 9, 30),
    )
    broker.positions["AAPL"] = pos
    return broker


class TestBrokerBasics:
    """Test basic broker methods."""

    def test_get_cash(self, broker):
        """Test get_cash returns initial capital."""
        assert broker.get_cash() == 100000.0

    def test_get_account_value(self, broker):
        """Test get_account_value returns correct value."""
        value = broker.get_account_value()
        assert value == 100000.0

    def test_get_position_none(self, broker):
        """Test get_position returns None for no position."""
        assert broker.get_position("AAPL") is None

    def test_get_position_existing(self, broker_with_position):
        """Test get_position returns existing position from positions dict."""
        # Note: get_position checks positions dict
        pos = broker_with_position.positions.get("AAPL")
        assert pos is not None
        assert pos.quantity == 100.0
        assert pos.entry_price == 150.0


class TestOrderManagement:
    """Test order management methods."""

    def test_submit_market_order(self, broker):
        """Test submitting a market order."""
        order = broker.submit_order("AAPL", 100.0, OrderSide.BUY)
        assert order is not None
        assert order.asset == "AAPL"
        assert order.quantity == 100.0
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.PENDING

    def test_submit_limit_order(self, broker):
        """Test submitting a limit order."""
        order = broker.submit_order("AAPL", 50.0, OrderSide.BUY, OrderType.LIMIT, limit_price=145.0)
        assert order is not None
        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == 145.0

    def test_submit_stop_order(self, broker):
        """Test submitting a stop order."""
        order = broker.submit_order("AAPL", 100.0, OrderSide.SELL, OrderType.STOP, stop_price=140.0)
        assert order is not None
        assert order.order_type == OrderType.STOP
        assert order.stop_price == 140.0

    def test_get_order_existing(self, broker):
        """Test get_order finds submitted order."""
        submitted = broker.submit_order("AAPL", 100.0, OrderSide.BUY)
        found = broker.get_order(submitted.order_id)
        assert found is not None
        assert found.order_id == submitted.order_id

    def test_get_order_not_found(self, broker):
        """Test get_order returns None for unknown ID."""
        assert broker.get_order("nonexistent-id") is None

    def test_get_pending_orders_all(self, broker):
        """Test get_pending_orders returns all pending."""
        broker.submit_order("AAPL", 100.0, OrderSide.BUY)
        broker.submit_order("GOOG", 50.0, OrderSide.BUY)

        pending = broker.get_pending_orders()
        assert len(pending) == 2

    def test_get_pending_orders_by_asset(self, broker):
        """Test get_pending_orders filters by asset."""
        broker.submit_order("AAPL", 100.0, OrderSide.BUY)
        broker.submit_order("GOOG", 50.0, OrderSide.BUY)
        broker.submit_order("AAPL", 50.0, OrderSide.SELL, OrderType.LIMIT, limit_price=160.0)

        aapl_orders = broker.get_pending_orders("AAPL")
        assert len(aapl_orders) == 2
        assert all(o.asset == "AAPL" for o in aapl_orders)


class TestOrderUpdates:
    """Test order update and cancel methods."""

    def test_update_order_success(self, broker):
        """Test updating order parameters."""
        order = broker.submit_order(
            "AAPL", 100.0, OrderSide.BUY, OrderType.LIMIT, limit_price=145.0
        )
        result = broker.update_order(order.order_id, limit_price=143.0)
        assert result is True

        updated = broker.get_order(order.order_id)
        assert updated.limit_price == 143.0

    def test_update_order_quantity(self, broker):
        """Test updating order quantity."""
        order = broker.submit_order("AAPL", 100.0, OrderSide.BUY)
        result = broker.update_order(order.order_id, quantity=150.0)
        assert result is True

        updated = broker.get_order(order.order_id)
        assert updated.quantity == 150.0

    def test_update_order_not_found(self, broker):
        """Test updating nonexistent order."""
        result = broker.update_order("nonexistent-id", limit_price=100.0)
        assert result is False

    def test_cancel_order_success(self, broker):
        """Test cancelling an order."""
        order = broker.submit_order("AAPL", 100.0, OrderSide.BUY)
        result = broker.cancel_order(order.order_id)
        assert result is True

        # Order should be cancelled and removed from pending
        cancelled = broker.get_order(order.order_id)
        assert cancelled.status == OrderStatus.CANCELLED
        assert len(broker.get_pending_orders()) == 0

    def test_cancel_order_not_found(self, broker):
        """Test cancelling nonexistent order."""
        result = broker.cancel_order("nonexistent-id")
        assert result is False


class TestClosePosition:
    """Test close_position method."""

    def test_close_long_position(self, broker_with_position):
        """Test closing a long position."""
        order = broker_with_position.close_position("AAPL")
        assert order is not None
        assert order.side == OrderSide.SELL
        assert order.quantity == 100.0

    def test_close_no_position(self, broker):
        """Test close_position with no position."""
        order = broker.close_position("AAPL")
        assert order is None

    def test_close_short_position(self, broker):
        """Test closing a short position."""
        # Create short position in positions dict
        pos = Position(
            asset="AAPL",
            quantity=-50.0,
            entry_price=150.0,
            entry_time=datetime(2024, 1, 1, 9, 30),
        )
        broker.positions["AAPL"] = pos

        order = broker.close_position("AAPL")
        assert order is not None
        assert order.side == OrderSide.BUY  # Buy to cover short
        assert order.quantity == 50.0


class TestIsExitOrder:
    """Test _is_exit_order internal method."""

    def test_no_position_is_not_exit(self, broker):
        """Test order with no position is not exit."""
        order = Order(
            asset="AAPL",
            quantity=100.0,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
        )
        assert broker._is_exit_order(order) is False

    def test_sell_with_long_is_exit(self, broker_with_position):
        """Test sell order with long position is exit."""
        order = Order(
            asset="AAPL",
            quantity=50.0,  # Partial exit
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
        )
        assert broker_with_position._is_exit_order(order) is True

    def test_sell_full_position_is_exit(self, broker_with_position):
        """Test sell order that flattens is exit."""
        order = Order(
            asset="AAPL",
            quantity=100.0,  # Full exit
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
        )
        assert broker_with_position._is_exit_order(order) is True

    def test_sell_reversal_is_not_exit(self, broker_with_position):
        """Test sell that reverses position is not exit."""
        order = Order(
            asset="AAPL",
            quantity=150.0,  # Would go short
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
        )
        assert broker_with_position._is_exit_order(order) is False

    def test_buy_with_long_is_not_exit(self, broker_with_position):
        """Test buy with long position is not exit (adding)."""
        order = Order(
            asset="AAPL",
            quantity=50.0,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
        )
        assert broker_with_position._is_exit_order(order) is False

    def test_buy_with_short_is_exit(self, broker):
        """Test buy with short position is exit."""
        # Create short position in positions dict
        pos = Position(
            asset="AAPL",
            quantity=-100.0,
            entry_price=150.0,
            entry_time=datetime(2024, 1, 1, 9, 30),
        )
        broker.positions["AAPL"] = pos

        order = Order(
            asset="AAPL",
            quantity=50.0,  # Partial cover
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
        )
        assert broker._is_exit_order(order) is True


class TestContractSpec:
    """Test contract spec methods."""

    def test_get_contract_spec_none(self, broker):
        """Test get_contract_spec returns None without specs."""
        assert broker.get_contract_spec("AAPL") is None

    def test_get_multiplier_default(self, broker):
        """Test get_multiplier returns 1.0 for stocks."""
        assert broker.get_multiplier("AAPL") == 1.0


class TestStopOrderGapFill:
    """Test stop order fill prices when price gaps through stop level."""

    def test_gap_through_sell_stop(self, broker):
        """Test sell stop fills at open when price gaps down through stop.

        This test verifies Bug #2 fix: when price gaps through a stop,
        fill should occur at the open price, not at the stop price.

        Scenario:
        - Long 100 AAPL @ $150 with stop-loss at $145
        - Next bar: Open=$140 (gapped down), High=$142, Low=$138
        - Expected: Fill at $140 (open), not $142 (high)
        """
        # Setup: Create position and stop order
        broker.positions["AAPL"] = Position(
            asset="AAPL",
            quantity=100.0,
            entry_price=150.0,
            entry_time=datetime(2024, 1, 1, 9, 30),
        )

        stop_order = broker.submit_order(
            "AAPL", 100.0, OrderSide.SELL, OrderType.STOP, stop_price=145.0
        )

        # Simulate bar with gap through stop
        # Yesterday close was $150, today opens at $140 (gap down)
        broker._update_time(
            timestamp=datetime(2024, 1, 2, 9, 30),
            prices={"AAPL": 141.0},  # Close
            opens={"AAPL": 140.0},  # Open (gapped down)
            volumes={"AAPL": 1_000_000},
            highs={"AAPL": 142.0},  # High
            lows={"AAPL": 138.0},  # Low
            signals={},
        )

        # Process orders - stop should trigger and fill at open ($140)
        broker._process_orders()

        # Verify fill occurred
        assert stop_order.status == OrderStatus.FILLED
        assert len(broker.fills) == 1

        # Critical assertion: fill price should be open ($140), not high ($142) or stop ($145)
        fill = broker.fills[0]
        assert fill.price == 140.0, f"Expected fill at open $140, got ${fill.price}"
        assert fill.price < stop_order.stop_price, "Gap fill should be worse than stop price"

    def test_normal_sell_stop_no_gap(self, broker):
        """Test sell stop fills at stop price when no gap (normal case)."""
        # Setup: Create position and stop order
        broker.positions["AAPL"] = Position(
            asset="AAPL",
            quantity=100.0,
            entry_price=150.0,
            entry_time=datetime(2024, 1, 1, 9, 30),
        )

        stop_order = broker.submit_order(
            "AAPL", 100.0, OrderSide.SELL, OrderType.STOP, stop_price=145.0
        )

        # Simulate bar that hits stop normally (no gap)
        # Open above stop, but low reaches stop
        broker._update_time(
            timestamp=datetime(2024, 1, 2, 9, 30),
            prices={"AAPL": 146.0},  # Close
            opens={"AAPL": 148.0},  # Open (above stop)
            volumes={"AAPL": 1_000_000},
            highs={"AAPL": 149.0},
            lows={"AAPL": 144.0},  # Low (hits stop)
            signals={},
        )

        broker._process_orders()

        # Verify fill at stop price (normal case)
        assert stop_order.status == OrderStatus.FILLED
        fill = broker.fills[0]
        assert fill.price == 145.0, f"Expected fill at stop $145, got ${fill.price}"

    def test_gap_through_buy_stop(self, broker):
        """Test buy stop fills at open when price gaps up through stop."""
        # Setup: Short position with buy stop (stop-loss for short)
        broker.positions["AAPL"] = Position(
            asset="AAPL",
            quantity=-100.0,  # Short
            entry_price=100.0,
            entry_time=datetime(2024, 1, 1, 9, 30),
        )

        stop_order = broker.submit_order(
            "AAPL", 100.0, OrderSide.BUY, OrderType.STOP, stop_price=105.0
        )

        # Simulate bar with gap through stop
        # Yesterday close was $100, today opens at $110 (gap up)
        broker._update_time(
            timestamp=datetime(2024, 1, 2, 9, 30),
            prices={"AAPL": 109.0},  # Close
            opens={"AAPL": 110.0},  # Open (gapped up)
            volumes={"AAPL": 1_000_000},
            highs={"AAPL": 111.0},
            lows={"AAPL": 108.0},  # Low
            signals={},
        )

        broker._process_orders()

        # Verify fill at open (gap fill)
        assert stop_order.status == OrderStatus.FILLED
        fill = broker.fills[0]
        assert fill.price == 110.0, f"Expected fill at open $110, got ${fill.price}"
        assert fill.price > stop_order.stop_price, "Gap fill should be worse than stop price"


class TestCommissionSplitOnFlip:
    """Test commission is properly split when position flips (close + open)."""

    def test_commission_split_on_flip(self):
        """Test commission split between closing and opening when flipping position.

        This test verifies Bug #3 fix: when flipping a position (Long 100 → Short 100 via -200 order),
        commission should be calculated separately for:
        - Closing 100 shares (close commission)
        - Opening 100 shares short (open commission)

        Old behavior: All commission applied to closing trade only.
        New behavior: Commission split proportionally between close and open.
        """
        from ml4t.backtest.models import PerShareCommission

        # Create broker with per-share commission ($0.01/share) and margin account
        broker = Broker(
            initial_cash=100_000.0,
            commission_model=PerShareCommission(0.01),
            slippage_model=NoSlippage(),
            account_type="margin",  # Allow position flips
        )

        # Setup: Create long position (100 shares @ $100)
        broker.positions["AAPL"] = Position(
            asset="AAPL",
            quantity=100.0,
            entry_price=100.0,
            entry_time=datetime(2024, 1, 1, 9, 30),
        )
        broker.cash -= 100.0 * 100  # Deduct purchase cost

        # Flip position: Sell 200 shares @ $110
        # This closes 100 long and opens 100 short
        flip_order = broker.submit_order("AAPL", 200.0, OrderSide.SELL)

        # Simulate market update
        broker._update_time(
            timestamp=datetime(2024, 1, 2, 9, 30),
            prices={"AAPL": 110.0},
            opens={"AAPL": 110.0},
            volumes={"AAPL": 1_000_000},
            highs={"AAPL": 111.0},
            lows={"AAPL": 109.0},
            signals={},
        )

        # Process the flip
        broker._process_orders()

        # Verify order filled
        assert flip_order.status == OrderStatus.FILLED

        # Verify commission split
        # Expected: 100 shares closed × $0.01 = $1.00 (close commission)
        #           100 shares opened × $0.01 = $1.00 (open commission)
        #           Total = $2.00

        # Check closing trade has correct commission
        assert len(broker.trades) == 1
        closing_trade = broker.trades[0]
        assert closing_trade.quantity == 100.0  # Long 100 closed
        assert (
            closing_trade.commission == 1.0
        ), f"Expected close commission $1.00, got ${closing_trade.commission}"

        # Check PnL calculation includes only close commission
        expected_pnl = (110.0 - 100.0) * 100.0 - 1.0  # Profit minus close commission
        assert (
            closing_trade.pnl == expected_pnl
        ), f"Expected PnL ${expected_pnl}, got ${closing_trade.pnl}"

        # Verify new position created (short 100)
        new_pos = broker.positions.get("AAPL")
        assert new_pos is not None
        assert new_pos.quantity == -100.0  # Short position
        assert new_pos.entry_price == 110.0

        # Verify cash reflects both commissions ($1 close + $1 open = $2 total)
        # Initial: $100,000 - $10,000 (position cost) = $90,000
        # After flip:
        #   + $10,000 (close 100 @ $100)
        #   + $1,000 (profit on close: $110 - $100 × 100)
        #   - $1 (close commission)
        #   + $11,000 (short proceeds: 100 × $110)
        #   - $1 (open commission)
        # Total: $90,000 + $10,000 + $1,000 - $1 + $11,000 - $1 = $111,998
        expected_cash = 100_000.0 - 10_000.0 + 10_000.0 + 1_000.0 - 1.0 + 11_000.0 - 1.0
        assert (
            abs(broker.cash - expected_cash) < 0.01
        ), f"Expected cash ${expected_cash:.2f}, got ${broker.cash:.2f}"


class TestBracketCancellationOnFlip:
    """Test that bracket orders are cancelled when position flips.

    Bug #4: When position flips (Long → Short), old bracket orders (stop-loss,
    take-profit) must be cancelled. Otherwise they can trigger unexpectedly on
    the new position.

    Scenario:
    1. Long 100 AAPL with stop-loss @ $145 and take-profit @ $155
    2. Flip to short 100 via sell -200
    3. Old bracket orders should be cancelled
    4. Verify price moving to $145 doesn't trigger old stop-loss
    """

    def test_cancel_brackets_on_flip(self):
        """Test bracket orders cancelled when position flips Long → Short."""
        from ml4t.backtest.models import PerShareCommission

        # Create broker with margin account (allows flips)
        broker = Broker(
            initial_cash=100_000.0,
            commission_model=PerShareCommission(0.01),
            slippage_model=NoSlippage(),
            account_type="margin",
        )

        # Bar 1: $150 - Open long 100 with brackets
        broker._update_time(
            timestamp=datetime(2024, 1, 1, 9, 30),
            prices={"AAPL": 150.0},
            opens={"AAPL": 150.0},
            volumes={"AAPL": 1_000_000},
            highs={"AAPL": 152.0},
            lows={"AAPL": 148.0},
            signals={},
        )

        # Submit entry order (market buy 100)
        broker.submit_order("AAPL", 100.0, OrderSide.BUY)
        broker._process_orders()

        # Verify position opened
        pos = broker.get_position("AAPL")
        assert pos is not None
        assert pos.quantity == 100

        # Submit bracket orders (stop-loss and take-profit)
        # Use prices that won't trigger: stop at $140, take-profit at $165
        stop_loss = broker.submit_order(
            asset="AAPL",
            quantity=100.0,
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            stop_price=140.0,  # Below current $150
        )
        take_profit = broker.submit_order(
            asset="AAPL",
            quantity=100.0,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            limit_price=165.0,  # Above next bar's $160
        )

        # Verify 2 pending bracket orders
        assert len(broker.pending_orders) == 2

        # Bar 2: $160 - Flip to short 100 (sell 200 = close 100 long + open 100 short)
        broker._update_time(
            timestamp=datetime(2024, 1, 2, 9, 30),
            prices={"AAPL": 160.0},
            opens={"AAPL": 160.0},
            volumes={"AAPL": 1_000_000},
            highs={"AAPL": 162.0},
            lows={"AAPL": 158.0},
            signals={},
        )

        broker.submit_order("AAPL", 200.0, OrderSide.SELL)
        broker._process_orders()

        # Verify position flipped to short
        pos = broker.get_position("AAPL")
        assert pos is not None
        assert pos.quantity == -100, f"Expected short 100, got {pos.quantity}"

        # CRITICAL: Verify old bracket orders were cancelled
        assert (
            len(broker.pending_orders) == 0
        ), f"Expected 0 pending orders after flip, got {len(broker.pending_orders)}"

        # Verify bracket orders have CANCELLED status
        assert stop_loss.status == OrderStatus.CANCELLED
        assert take_profit.status == OrderStatus.CANCELLED

        # Bar 3: $135 - Price moves below old stop ($140), should NOT trigger
        broker._update_time(
            timestamp=datetime(2024, 1, 3, 9, 30),
            prices={"AAPL": 135.0},
            opens={"AAPL": 135.0},
            volumes={"AAPL": 1_000_000},
            highs={"AAPL": 137.0},
            lows={"AAPL": 133.0},
            signals={},
        )
        broker._process_orders()

        # Verify position unchanged (still short 100)
        pos = broker.get_position("AAPL")
        assert pos is not None
        assert pos.quantity == -100, "Old stop-loss triggered despite cancellation!"

        # Bar 4: $140 - Price at exact old stop price, still should NOT trigger
        broker._update_time(
            timestamp=datetime(2024, 1, 4, 9, 30),
            prices={"AAPL": 140.0},
            opens={"AAPL": 140.0},
            volumes={"AAPL": 1_000_000},
            highs={"AAPL": 142.0},
            lows={"AAPL": 138.0},
            signals={},
        )
        broker._process_orders()

        # Final verification: position still short 100
        pos = broker.get_position("AAPL")
        assert pos is not None
        assert pos.quantity == -100, "Old bracket orders affected new position!"


class TestPositionFlipValidation:
    """Test that Gatekeeper correctly validates position flips.

    Bug #5: When flipping position, Gatekeeper should simulate closing the old
    position first, then validate the new opposite position against post-close
    buying power.

    The bug was that it validated the flip as if both positions existed simultaneously,
    causing margin requirements to be calculated incorrectly.
    """

    def test_position_flip_validation_with_sufficient_margin(self):
        """Test position flip is allowed when post-close buying power is sufficient."""
        from ml4t.backtest.models import PerShareCommission

        # Setup: Margin account with initial $100k
        broker = Broker(
            initial_cash=100_000.0,
            commission_model=PerShareCommission(0.01),
            slippage_model=NoSlippage(),
            account_type="margin",
        )

        # Bar 1: $100 - Open long 100 shares (costs $10,000 + $1 commission)
        broker._update_time(
            timestamp=datetime(2024, 1, 1, 9, 30),
            prices={"AAPL": 100.0},
            opens={"AAPL": 100.0},
            volumes={"AAPL": 1_000_000},
            highs={"AAPL": 102.0},
            lows={"AAPL": 98.0},
            signals={},
        )

        broker.submit_order("AAPL", 100.0, OrderSide.BUY)
        broker._process_orders()

        # Verify position opened
        pos = broker.get_position("AAPL")
        assert pos is not None
        assert pos.quantity == 100

        # Cash after open: $100k - $10k - $1 = $89,999
        expected_cash = 100_000.0 - 10_000.0 - 1.0
        assert abs(broker.cash - expected_cash) < 0.01

        # Bar 2: $150 - Flip to short 100 (sell 200)
        # This should:
        # 1. Close long 100 @ $150 -> receive $15,000, pay $1 commission
        # 2. Open short 100 @ $150 -> receive $15,000, pay $1 commission
        # Net cash: $89,999 + $15,000 - $1 + $15,000 - $1 = $119,997
        broker._update_time(
            timestamp=datetime(2024, 1, 2, 9, 30),
            prices={"AAPL": 150.0},
            opens={"AAPL": 150.0},
            volumes={"AAPL": 1_000_000},
            highs={"AAPL": 152.0},
            lows={"AAPL": 148.0},
            signals={},
        )

        # Submit flip order - this should be ALLOWED
        flip_order = broker.submit_order("AAPL", 200.0, OrderSide.SELL)
        broker._process_orders()

        # Verify flip succeeded
        assert flip_order.status == OrderStatus.FILLED
        pos = broker.get_position("AAPL")
        assert pos is not None
        assert pos.quantity == -100, f"Expected short 100, got {pos.quantity}"

        # Verify cash is correct
        # Starting: $89,999
        # Close long: +$15,000 - $1 = +$14,999
        # Open short: +$15,000 - $1 = +$14,999
        # Final: $89,999 + $14,999 + $14,999 = $119,997
        expected_cash = 89_999.0 + 15_000.0 - 1.0 + 15_000.0 - 1.0
        assert (
            abs(broker.cash - expected_cash) < 0.01
        ), f"Expected cash ${expected_cash:.2f}, got ${broker.cash:.2f}"


class TestBrokerEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_account_type(self):
        """Test that invalid account_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown account_type"):
            Broker(initial_cash=100000.0, account_type="invalid")

    def test_position_scaling_up(self):
        """Test adding to an existing position (scaling up)."""
        broker = Broker(
            initial_cash=100000.0,
            commission_model=NoCommission(),
            slippage_model=NoSlippage(),
        )

        # Open initial long position: 100 shares @ $150
        broker._update_time(
            timestamp=datetime(2024, 1, 1, 9, 30),
            prices={"AAPL": 150.0},
            opens={"AAPL": 150.0},
            volumes={"AAPL": 1_000_000},
            highs={"AAPL": 152.0},
            lows={"AAPL": 148.0},
            signals={},
        )
        broker.submit_order("AAPL", 100.0, OrderSide.BUY)
        broker._process_orders()

        pos = broker.get_position("AAPL")
        assert pos.quantity == 100
        assert pos.entry_price == 150.0

        # Scale up: buy 50 more @ $160
        broker._update_time(
            timestamp=datetime(2024, 1, 2, 9, 30),
            prices={"AAPL": 160.0},
            opens={"AAPL": 160.0},
            volumes={"AAPL": 1_000_000},
            highs={"AAPL": 162.0},
            lows={"AAPL": 158.0},
            signals={},
        )
        broker.submit_order("AAPL", 50.0, OrderSide.BUY)
        broker._process_orders()

        # Verify position was scaled up
        pos = broker.get_position("AAPL")
        assert pos.quantity == 150

        # Entry price should be weighted average: (100*150 + 50*160) / 150 = 153.33
        expected_entry = (100 * 150.0 + 50 * 160.0) / 150
        assert abs(pos.entry_price - expected_entry) < 0.01

    def test_position_scaling_down_short(self):
        """Test adding to short position (scaling down)."""
        broker = Broker(
            initial_cash=100000.0,
            commission_model=NoCommission(),
            slippage_model=NoSlippage(),
            account_type="margin",
        )

        # Open initial short position: -100 shares @ $150
        broker._update_time(
            timestamp=datetime(2024, 1, 1, 9, 30),
            prices={"AAPL": 150.0},
            opens={"AAPL": 150.0},
            volumes={"AAPL": 1_000_000},
            highs={"AAPL": 152.0},
            lows={"AAPL": 148.0},
            signals={},
        )
        broker.submit_order("AAPL", 100.0, OrderSide.SELL)
        broker._process_orders()

        pos = broker.get_position("AAPL")
        assert pos.quantity == -100
        assert pos.entry_price == 150.0

        # Scale short: sell 50 more @ $140 (lower price, more profit potential)
        broker._update_time(
            timestamp=datetime(2024, 1, 2, 9, 30),
            prices={"AAPL": 140.0},
            opens={"AAPL": 140.0},
            volumes={"AAPL": 1_000_000},
            highs={"AAPL": 142.0},
            lows={"AAPL": 138.0},
            signals={},
        )
        broker.submit_order("AAPL", 50.0, OrderSide.SELL)
        broker._process_orders()

        # Verify position was scaled
        pos = broker.get_position("AAPL")
        assert pos.quantity == -150

        # Entry price should be weighted average: (100*150 + 50*140) / 150 = 146.67
        expected_entry = (100 * 150.0 + 50 * 140.0) / 150
        assert abs(pos.entry_price - expected_entry) < 0.01

    def test_limit_order_exact_price_fill(self):
        """Test limit order fills at exact limit price when touched."""
        broker = Broker(
            initial_cash=100000.0,
            commission_model=NoCommission(),
            slippage_model=NoSlippage(),
        )

        # Set up price data where low exactly touches limit price
        broker._update_time(
            timestamp=datetime(2024, 1, 1, 9, 30),
            prices={"AAPL": 150.0},
            opens={"AAPL": 152.0},
            volumes={"AAPL": 1_000_000},
            highs={"AAPL": 155.0},
            lows={"AAPL": 145.0},  # Low touches our limit
            signals={},
        )

        # Submit limit buy at 145 (which is the exact low)
        order = broker.submit_order(
            "AAPL", 100.0, OrderSide.BUY, OrderType.LIMIT, limit_price=145.0
        )
        broker._process_orders()

        # Order should be filled at limit price
        assert order.status == OrderStatus.FILLED
        pos = broker.get_position("AAPL")
        assert pos is not None
        assert pos.entry_price == 145.0

    def test_buy_stop_order(self):
        """Test buy stop order triggers above price."""
        broker = Broker(
            initial_cash=100000.0,
            commission_model=NoCommission(),
            slippage_model=NoSlippage(),
        )

        # Submit a buy stop at $155 (breakout entry)
        broker._update_time(
            timestamp=datetime(2024, 1, 1, 9, 30),
            prices={"AAPL": 150.0},
            opens={"AAPL": 148.0},
            volumes={"AAPL": 1_000_000},
            highs={"AAPL": 152.0},
            lows={"AAPL": 147.0},
            signals={},
        )
        order = broker.submit_order("AAPL", 100.0, OrderSide.BUY, OrderType.STOP, stop_price=155.0)
        broker._process_orders()

        # Stop not triggered yet
        assert order.status == OrderStatus.PENDING

        # Price breaks above stop
        broker._update_time(
            timestamp=datetime(2024, 1, 2, 9, 30),
            prices={"AAPL": 158.0},
            opens={"AAPL": 154.0},
            volumes={"AAPL": 1_000_000},
            highs={"AAPL": 160.0},
            lows={"AAPL": 153.0},
            signals={},
        )
        broker._process_orders()

        # Order should be filled at stop price
        assert order.status == OrderStatus.FILLED
        pos = broker.get_position("AAPL")
        assert pos is not None
        assert pos.entry_price == 155.0


class TestCommissionSlippageModels:
    """Test commission and slippage model calculations."""

    def test_tiered_commission_low_tier(self):
        """Test tiered commission in lowest tier."""
        from ml4t.backtest.models import TieredCommission

        model = TieredCommission(tiers=[(10000, 0.002), (50000, 0.001), (float("inf"), 0.0005)])
        # Trade value = 100 * 50 = 5000, should hit first tier (< 10000)
        commission = model.calculate("AAPL", 100, 50.0)
        assert commission == 5000 * 0.002  # 10.0

    def test_tiered_commission_mid_tier(self):
        """Test tiered commission in middle tier."""
        from ml4t.backtest.models import TieredCommission

        model = TieredCommission(tiers=[(10000, 0.002), (50000, 0.001), (float("inf"), 0.0005)])
        # Trade value = 200 * 100 = 20000, should hit second tier (> 10000, < 50000)
        commission = model.calculate("AAPL", 200, 100.0)
        assert commission == 20000 * 0.001  # 20.0

    def test_tiered_commission_top_tier(self):
        """Test tiered commission falls through to top tier."""
        from ml4t.backtest.models import TieredCommission

        model = TieredCommission(tiers=[(10000, 0.002), (50000, 0.001), (float("inf"), 0.0005)])
        # Trade value = 1000 * 100 = 100000, should hit final tier (> 50000)
        commission = model.calculate("AAPL", 1000, 100.0)
        assert commission == 100000 * 0.0005  # 50.0

    def test_combined_commission(self):
        """Test combined percentage + fixed commission."""
        from ml4t.backtest.models import CombinedCommission

        model = CombinedCommission(percentage=0.001, fixed=5.0)
        # Trade value = 100 * 150 = 15000
        commission = model.calculate("AAPL", 100, 150.0)
        assert commission == 15000 * 0.001 + 5.0  # 20.0

    def test_volume_share_slippage_no_volume(self):
        """Test volume share slippage with no volume returns zero."""
        from ml4t.backtest.models import VolumeShareSlippage

        model = VolumeShareSlippage(impact_factor=0.1)
        # No volume should return 0 slippage
        assert model.calculate("AAPL", 100, 150.0, None) == 0.0
        assert model.calculate("AAPL", 100, 150.0, 0) == 0.0


class TestEquityCurve:
    """Test EquityCurve class."""

    def test_append_and_len(self):
        """Test append and __len__."""
        from ml4t.backtest.analytics.equity import EquityCurve

        ec = EquityCurve()
        assert len(ec) == 0
        ec.append(datetime(2024, 1, 1), 100000.0)
        ec.append(datetime(2024, 1, 2), 101000.0)
        assert len(ec) == 2

    def test_returns_insufficient_data(self):
        """Test returns with insufficient data."""
        from ml4t.backtest.analytics.equity import EquityCurve

        ec = EquityCurve()
        assert len(ec.returns) == 0
        ec.append(datetime(2024, 1, 1), 100000.0)
        assert len(ec.returns) == 0  # Need at least 2 values

    def test_cumulative_returns(self):
        """Test cumulative returns calculation."""
        from ml4t.backtest.analytics.equity import EquityCurve

        ec = EquityCurve()
        ec.values = [100, 110, 115, 120]
        cr = ec.cumulative_returns
        assert len(cr) == 4
        assert cr[0] == 0.0  # First value is 0
        assert abs(cr[-1] - 0.2) < 0.001  # 20% cumulative return

    def test_total_return_zero_initial(self):
        """Test total return with zero initial value."""
        from ml4t.backtest.analytics.equity import EquityCurve

        ec = EquityCurve()
        ec.values = [0.0, 100.0]
        assert ec.total_return == 0.0  # Avoid division by zero

    def test_drawdown_series(self):
        """Test drawdown series calculation."""
        from ml4t.backtest.analytics.equity import EquityCurve

        ec = EquityCurve()
        ec.values = [100, 110, 105, 90, 100]
        dd = ec.drawdown_series()
        assert len(dd) == 5
        assert dd[0] == 0.0  # No drawdown at start
        assert dd[3] < 0  # Drawdown when value drops

    def test_to_dict(self):
        """Test to_dict export."""
        from ml4t.backtest.analytics.equity import EquityCurve

        ec = EquityCurve()
        ec.values = [100, 105, 110, 108]
        result = ec.to_dict()
        assert "initial_value" in result
        assert "final_value" in result
        assert "total_return" in result
        assert "sharpe" in result
        assert "sortino" in result
        assert "max_drawdown" in result
        assert result["initial_value"] == 100
        assert result["final_value"] == 108


class TestMetricsEdgeCases:
    """Test edge cases in metrics calculations."""

    def test_volatility_insufficient_data(self):
        """Test volatility with insufficient data returns 0."""
        from ml4t.backtest.analytics.metrics import volatility

        assert volatility([]) == 0.0
        assert volatility([0.01]) == 0.0

    def test_sharpe_insufficient_data(self):
        """Test Sharpe ratio with insufficient data returns 0."""
        from ml4t.backtest.analytics.metrics import sharpe_ratio

        assert sharpe_ratio([]) == 0.0
        assert sharpe_ratio([0.01]) == 0.0

    def test_sharpe_no_annualize(self):
        """Test Sharpe ratio without annualization."""
        from ml4t.backtest.analytics.metrics import sharpe_ratio

        returns = [0.01, 0.02, -0.01, 0.03, 0.01]
        result = sharpe_ratio(returns, annualize=False)
        assert result != 0.0  # Should compute non-zero value

    def test_sharpe_zero_volatility(self):
        """Test Sharpe ratio with zero volatility."""
        from ml4t.backtest.analytics.metrics import sharpe_ratio

        # All same returns = zero std
        returns = [0.01, 0.01, 0.01, 0.01]
        assert sharpe_ratio(returns) == 0.0

    def test_sortino_insufficient_data(self):
        """Test Sortino ratio with insufficient data."""
        from ml4t.backtest.analytics.metrics import sortino_ratio

        assert sortino_ratio([]) == 0.0
        assert sortino_ratio([0.01]) == 0.0

    def test_sortino_no_annualize(self):
        """Test Sortino ratio without annualization."""
        from ml4t.backtest.analytics.metrics import sortino_ratio

        returns = [0.01, 0.02, -0.01, 0.03, -0.02]
        result = sortino_ratio(returns, annualize=False)
        assert result != 0.0

    def test_sortino_no_downside(self):
        """Test Sortino ratio with no negative returns."""
        from ml4t.backtest.analytics.metrics import sortino_ratio

        # All positive returns - no downside
        returns = [0.01, 0.02, 0.03, 0.01]
        result = sortino_ratio(returns)
        assert result == float("inf")

    def test_max_drawdown_insufficient_data(self):
        """Test max drawdown with insufficient data."""
        from ml4t.backtest.analytics.metrics import max_drawdown

        dd, peak, trough = max_drawdown([])
        assert dd == 0.0
        dd, peak, trough = max_drawdown([100.0])
        assert dd == 0.0

    def test_cagr_edge_cases(self):
        """Test CAGR edge cases."""
        from ml4t.backtest.analytics.metrics import cagr

        # Zero initial value
        assert cagr(0, 100, 1.0) == 0.0
        # Zero years
        assert cagr(100, 200, 0.0) == 0.0
        # Negative initial value
        assert cagr(-100, 100, 1.0) == 0.0
        # Zero final value (total loss)
        assert cagr(100, 0, 1.0) == -1.0


class TestBacktestConfigMethods:
    """Test BacktestConfig methods for serialization."""

    def test_to_dict(self):
        """Test config to_dict serialization."""
        from ml4t.backtest.config import BacktestConfig

        config = BacktestConfig()
        result = config.to_dict()
        assert "execution" in result
        assert "commission" in result
        assert "slippage" in result
        assert "cash" in result

    def test_from_dict_round_trip(self):
        """Test from_dict restores config."""
        from ml4t.backtest.config import BacktestConfig

        original = BacktestConfig(initial_cash=50000.0, commission_rate=0.002)
        data = original.to_dict()
        restored = BacktestConfig.from_dict(data)
        assert restored.initial_cash == 50000.0
        assert restored.commission_rate == 0.002

    def test_to_yaml_from_yaml(self, tmp_path):
        """Test YAML serialization round trip."""
        from ml4t.backtest.config import BacktestConfig

        config = BacktestConfig(initial_cash=75000.0)
        yaml_path = tmp_path / "config.yaml"
        config.to_yaml(yaml_path)

        loaded = BacktestConfig.from_yaml(yaml_path)
        assert loaded.initial_cash == 75000.0

    def test_from_preset_invalid(self):
        """Test from_preset with invalid preset name."""
        from ml4t.backtest.config import BacktestConfig

        with pytest.raises(ValueError, match="Unknown preset"):
            BacktestConfig.from_preset("nonexistent_preset")


class TestBrokerPositionRules:
    """Test broker position rule functionality."""

    def test_set_position_rules_per_asset(self):
        """Test setting position rules for specific asset."""
        from ml4t.backtest.risk.position.static import StopLoss

        broker = Broker(100000.0, NoCommission(), NoSlippage())
        stop_rule = StopLoss(pct=0.05)

        # Set rules for specific asset
        broker.set_position_rules(stop_rule, asset="AAPL")

        # Verify it's stored per-asset
        assert "AAPL" in broker._position_rules_by_asset
        assert broker._get_position_rules("AAPL") == stop_rule
        # Global rules should be None
        assert broker._position_rules is None

    def test_set_position_rules_global(self):
        """Test setting global position rules."""
        from ml4t.backtest.risk.position.static import TakeProfit

        broker = Broker(100000.0, NoCommission(), NoSlippage())
        tp_rule = TakeProfit(pct=0.10)

        # Set global rules
        broker.set_position_rules(tp_rule)

        # Verify it's stored globally
        assert broker._position_rules == tp_rule
        # Should apply to any asset
        assert broker._get_position_rules("AAPL") == tp_rule

    def test_update_position_context(self):
        """Test updating position context."""
        broker = Broker(100000.0, NoCommission(), NoSlippage())

        # Create a position first
        broker._update_time(
            timestamp=datetime(2024, 1, 1, 9, 30),
            prices={"AAPL": 150.0},
            opens={"AAPL": 149.0},
            volumes={"AAPL": 1_000_000},
            highs={"AAPL": 151.0},
            lows={"AAPL": 148.0},
            signals={},
        )
        broker.submit_order("AAPL", 100.0, OrderSide.BUY)
        broker._process_orders()

        # Update context for the position
        broker.update_position_context("AAPL", {"exit_signal": -0.5, "atr": 2.5})

        pos = broker.get_position("AAPL")
        assert pos is not None
        assert pos.context.get("exit_signal") == -0.5
        assert pos.context.get("atr") == 2.5

    def test_update_position_context_no_position(self):
        """Test updating context for non-existent position."""
        broker = Broker(100000.0, NoCommission(), NoSlippage())

        # Should not raise error
        broker.update_position_context("AAPL", {"signal": 1.0})


class TestBrokerTrailingStopSell:
    """Test trailing stop for sell (protecting long position)."""

    def test_trailing_stop_sell_with_trail_amount(self):
        """Test trailing stop sell using trail_amount."""
        broker = Broker(100000.0, NoCommission(), NoSlippage())

        # Enter long position
        broker._update_time(
            timestamp=datetime(2024, 1, 1, 9, 30),
            prices={"AAPL": 150.0},
            opens={"AAPL": 149.0},
            volumes={"AAPL": 1_000_000},
            highs={"AAPL": 151.0},
            lows={"AAPL": 148.0},
            signals={},
        )
        broker.submit_order("AAPL", 100.0, OrderSide.BUY)
        broker._process_orders()

        # Submit trailing stop sell to protect position
        order = broker.submit_order(
            "AAPL",
            -100.0,
            OrderSide.SELL,
            OrderType.TRAILING_STOP,
            trail_amount=5.0,  # Trail $5 below high
        )
        assert order.status == OrderStatus.PENDING

        # Price goes up - stop should adjust
        broker._update_time(
            timestamp=datetime(2024, 1, 2, 9, 30),
            prices={"AAPL": 160.0},
            opens={"AAPL": 158.0},
            volumes={"AAPL": 1_000_000},
            highs={"AAPL": 162.0},  # New high
            lows={"AAPL": 159.0},  # Low stays above stop
            signals={},
        )
        broker._process_orders()

        # Stop should have moved up: 162 - 5 = 157
        assert order.stop_price == 157.0
        assert order.status == OrderStatus.PENDING  # Not triggered (low 159 > 157)

        # Price drops through stop
        broker._update_time(
            timestamp=datetime(2024, 1, 3, 9, 30),
            prices={"AAPL": 155.0},
            opens={"AAPL": 156.0},
            volumes={"AAPL": 1_000_000},
            highs={"AAPL": 158.0},
            lows={"AAPL": 154.0},  # Drops below 157 stop
            signals={},
        )
        broker._process_orders()

        # Stop should be triggered
        assert order.status == OrderStatus.FILLED


class TestBrokerMissingPriceHandling:
    """Test broker handling when price data is missing."""

    def test_order_skipped_when_no_price(self):
        """Test order is skipped when asset has no price data."""
        broker = Broker(100000.0, NoCommission(), NoSlippage())

        broker._update_time(
            timestamp=datetime(2024, 1, 1, 9, 30),
            prices={"AAPL": 150.0},  # Only AAPL has price
            opens={"AAPL": 149.0},
            volumes={"AAPL": 1_000_000},
            highs={"AAPL": 151.0},
            lows={"AAPL": 148.0},
            signals={},
        )

        # Submit order for asset without price
        order = broker.submit_order("MSFT", 100.0, OrderSide.BUY)
        broker._process_orders()

        # Order should remain pending (no price data)
        assert order.status == OrderStatus.PENDING
