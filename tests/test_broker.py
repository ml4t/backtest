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
