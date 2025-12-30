"""Unit tests for CashAccountPolicy."""

from datetime import datetime

from ml4t.backtest import Position
from src.ml4t.backtest.accounting.policy import CashAccountPolicy


class TestCashAccountPolicyBuyingPower:
    """Tests for buying power calculation."""

    def test_positive_cash_buying_power(self):
        """Test buying power equals cash when cash is positive."""
        policy = CashAccountPolicy()
        bp = policy.calculate_buying_power(cash=10000.0, positions={})
        assert bp == 10000.0

    def test_zero_cash_buying_power(self):
        """Test buying power is zero when cash is zero."""
        policy = CashAccountPolicy()
        bp = policy.calculate_buying_power(cash=0.0, positions={})
        assert bp == 0.0

    def test_negative_cash_buying_power_is_zero(self):
        """Test buying power is capped at zero when cash is negative."""
        policy = CashAccountPolicy()
        bp = policy.calculate_buying_power(cash=-5000.0, positions={})
        assert bp == 0.0

    def test_buying_power_ignores_positions(self):
        """Test buying power calculation ignores position values."""
        policy = CashAccountPolicy()
        positions = {
            "AAPL": Position(
                asset="AAPL",
                quantity=100.0,
                entry_price=150.0,
                current_price=160.0,
                entry_time=datetime.now(),
            )
        }
        bp = policy.calculate_buying_power(cash=5000.0, positions=positions)
        # Cash account buying power = cash only (no margin against positions)
        assert bp == 5000.0

    def test_buying_power_with_large_cash(self):
        """Test buying power with large cash amount."""
        policy = CashAccountPolicy()
        bp = policy.calculate_buying_power(cash=1_000_000.0, positions={})
        assert bp == 1_000_000.0


class TestCashAccountPolicyShortSelling:
    """Tests for short selling permissions."""

    def test_allows_short_selling_returns_false(self):
        """Test that cash accounts do not allow short selling."""
        policy = CashAccountPolicy()
        assert policy.allows_short_selling() is False


class TestCashAccountPolicyNewPositionValidation:
    """Tests for validate_new_position method."""

    def test_valid_long_position_with_sufficient_cash(self):
        """Test approving long position with sufficient cash."""
        policy = CashAccountPolicy()
        valid, reason = policy.validate_new_position(
            asset="AAPL",
            quantity=100.0,
            price=150.0,
            current_positions={},
            cash=20000.0,
        )
        assert valid is True
        assert reason == ""

    def test_valid_long_position_exact_cash(self):
        """Test approving long position when cash exactly equals cost."""
        policy = CashAccountPolicy()
        valid, reason = policy.validate_new_position(
            asset="AAPL",
            quantity=100.0,
            price=150.0,
            current_positions={},
            cash=15000.0,  # Exactly enough
        )
        assert valid is True
        assert reason == ""

    def test_reject_long_position_insufficient_cash(self):
        """Test rejecting long position with insufficient cash."""
        policy = CashAccountPolicy()
        valid, reason = policy.validate_new_position(
            asset="AAPL",
            quantity=100.0,
            price=150.0,
            current_positions={},
            cash=10000.0,  # Need $15,000
        )
        assert valid is False
        assert "Insufficient cash" in reason
        assert "15000.00" in reason
        assert "10000.00" in reason

    def test_reject_short_position(self):
        """Test rejecting short position (negative quantity)."""
        policy = CashAccountPolicy()
        valid, reason = policy.validate_new_position(
            asset="AAPL",
            quantity=-100.0,  # Short
            price=150.0,
            current_positions={},
            cash=50000.0,  # Plenty of cash, but shorts not allowed
        )
        assert valid is False
        assert "Short selling not allowed" in reason

    def test_reject_zero_quantity(self):
        """Test behavior with zero quantity (edge case)."""
        policy = CashAccountPolicy()
        valid, reason = policy.validate_new_position(
            asset="AAPL",
            quantity=0.0,
            price=150.0,
            current_positions={},
            cash=10000.0,
        )
        # Zero quantity is technically neither long nor short
        # Should be allowed (costs $0)
        assert valid is True

    def test_fractional_shares_validation(self):
        """Test validation with fractional shares."""
        policy = CashAccountPolicy()
        valid, reason = policy.validate_new_position(
            asset="BTC-USD",
            quantity=0.5,
            price=60000.0,
            current_positions={},
            cash=35000.0,  # Need $30,000
        )
        assert valid is True

    def test_fractional_shares_insufficient_cash(self):
        """Test rejection with fractional shares and insufficient cash."""
        policy = CashAccountPolicy()
        valid, reason = policy.validate_new_position(
            asset="BTC-USD",
            quantity=0.5,
            price=60000.0,
            current_positions={},
            cash=25000.0,  # Need $30,000
        )
        assert valid is False
        assert "Insufficient cash" in reason


class TestCashAccountPolicyPositionChangeValidation:
    """Tests for validate_position_change method."""

    def test_add_to_long_position_with_sufficient_cash(self):
        """Test adding to existing long position."""
        policy = CashAccountPolicy()
        valid, reason = policy.validate_position_change(
            asset="AAPL",
            current_quantity=100.0,
            quantity_delta=50.0,  # Add 50 more
            price=150.0,
            current_positions={},
            cash=10000.0,  # Need $7,500
        )
        assert valid is True
        assert reason == ""

    def test_add_to_position_insufficient_cash(self):
        """Test rejecting addition with insufficient cash."""
        policy = CashAccountPolicy()
        valid, reason = policy.validate_position_change(
            asset="AAPL",
            current_quantity=100.0,
            quantity_delta=100.0,  # Add 100 more
            price=150.0,
            current_positions={},
            cash=10000.0,  # Need $15,000
        )
        assert valid is False
        assert "Insufficient cash" in reason

    def test_close_long_position(self):
        """Test closing long position (always allowed)."""
        policy = CashAccountPolicy()
        valid, reason = policy.validate_position_change(
            asset="AAPL",
            current_quantity=100.0,
            quantity_delta=-100.0,  # Close entirely
            price=150.0,
            current_positions={},
            cash=0.0,  # No cash needed to sell
        )
        assert valid is True
        assert reason == ""

    def test_reduce_long_position(self):
        """Test reducing long position (partial close)."""
        policy = CashAccountPolicy()
        valid, reason = policy.validate_position_change(
            asset="AAPL",
            current_quantity=100.0,
            quantity_delta=-50.0,  # Sell half
            price=150.0,
            current_positions={},
            cash=0.0,
        )
        assert valid is True

    def test_reject_overselling(self):
        """Test rejecting sell order larger than position."""
        policy = CashAccountPolicy()
        valid, reason = policy.validate_position_change(
            asset="AAPL",
            current_quantity=100.0,
            quantity_delta=-150.0,  # Trying to sell 150 when only have 100
            price=150.0,
            current_positions={},
            cash=10000.0,
        )
        assert valid is False
        # Overselling = position reversal in cash account
        assert "reversal not allowed" in reason

    def test_reject_position_reversal_long_to_short(self):
        """Test rejecting position reversal from long to short."""
        policy = CashAccountPolicy()
        valid, reason = policy.validate_position_change(
            asset="AAPL",
            current_quantity=100.0,  # Long 100
            quantity_delta=-200.0,  # Sell 200 -> ends up short 100
            price=150.0,
            current_positions={},
            cash=50000.0,
        )
        assert valid is False
        assert "reversal not allowed" in reason
        assert "100.00" in reason  # Current quantity
        assert "-200.00" in reason  # Delta

    def test_reject_position_reversal_short_to_long(self):
        """Test rejecting position reversal from short to long."""
        policy = CashAccountPolicy()
        valid, reason = policy.validate_position_change(
            asset="AAPL",
            current_quantity=-100.0,  # Short 100
            quantity_delta=200.0,  # Buy 200 -> ends up long 100
            price=150.0,
            current_positions={},
            cash=50000.0,
        )
        assert valid is False
        assert "reversal not allowed" in reason

    def test_reject_opening_short_from_flat(self):
        """Test rejecting short position from no position."""
        policy = CashAccountPolicy()
        valid, reason = policy.validate_position_change(
            asset="AAPL",
            current_quantity=0.0,  # No position
            quantity_delta=-100.0,  # Trying to short
            price=150.0,
            current_positions={},
            cash=50000.0,
        )
        assert valid is False
        assert "Short positions not allowed" in reason

    def test_reject_adding_to_short_position(self):
        """Test rejecting addition to short position."""
        policy = CashAccountPolicy()
        valid, reason = policy.validate_position_change(
            asset="AAPL",
            current_quantity=-100.0,  # Already short
            quantity_delta=-50.0,  # Adding to short
            price=150.0,
            current_positions={},
            cash=50000.0,
        )
        assert valid is False
        assert "Short positions not allowed" in reason

    def test_closing_short_position_allowed(self):
        """Test that closing a short position is allowed."""
        policy = CashAccountPolicy()
        valid, reason = policy.validate_position_change(
            asset="AAPL",
            current_quantity=-100.0,  # Short 100
            quantity_delta=100.0,  # Buy 100 to close
            price=150.0,
            current_positions={},
            cash=15000.0,  # Need cash to cover
        )
        # Note: This closes to flat (0), which is allowed
        assert valid is True

    def test_zero_position_change(self):
        """Test edge case of zero position change."""
        policy = CashAccountPolicy()
        valid, reason = policy.validate_position_change(
            asset="AAPL",
            current_quantity=100.0,
            quantity_delta=0.0,  # No change
            price=150.0,
            current_positions={},
            cash=10000.0,
        )
        assert valid is True  # No-op is allowed


class TestCashAccountPolicyEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_fractional_order(self):
        """Test validation with very small fractional quantity."""
        policy = CashAccountPolicy()
        valid, reason = policy.validate_new_position(
            asset="ETH-USD",
            quantity=0.001,
            price=3000.0,
            current_positions={},
            cash=5.0,  # Need $3
        )
        assert valid is True

    def test_very_large_order(self):
        """Test validation with very large order."""
        policy = CashAccountPolicy()
        valid, reason = policy.validate_new_position(
            asset="AAPL",
            quantity=1_000_000.0,
            price=150.0,
            current_positions={},
            cash=200_000_000.0,  # $200M, need $150M
        )
        assert valid is True

    def test_very_large_order_insufficient_cash(self):
        """Test rejecting very large order with insufficient cash."""
        policy = CashAccountPolicy()
        valid, reason = policy.validate_new_position(
            asset="AAPL",
            quantity=1_000_000.0,
            price=150.0,
            current_positions={},
            cash=100_000_000.0,  # $100M, need $150M
        )
        assert valid is False

    def test_nearly_zero_cash_remaining(self):
        """Test order that leaves minimal cash remaining."""
        policy = CashAccountPolicy()
        valid, reason = policy.validate_new_position(
            asset="AAPL",
            quantity=100.0,
            price=150.0,
            current_positions={},
            cash=15000.01,  # Just barely enough
        )
        assert valid is True


class TestCashAccountPolicyRealWorldScenarios:
    """Tests using realistic trading scenarios."""

    def test_day_trader_multiple_trades(self):
        """Test scenario: Day trader executing multiple trades."""
        policy = CashAccountPolicy()

        # Trade 1: Buy 50 TSLA at $200 with $15,000 cash
        valid, _ = policy.validate_new_position(
            asset="TSLA", quantity=50.0, price=200.0, current_positions={}, cash=15000.0
        )
        assert valid is True

        # After trade 1: Cash = $5,000, position = 50 TSLA
        # Trade 2: Buy 20 more TSLA at $200
        valid, _ = policy.validate_position_change(
            asset="TSLA",
            current_quantity=50.0,
            quantity_delta=20.0,
            price=200.0,
            current_positions={},
            cash=5000.0,  # Need $4,000
        )
        assert valid is True

        # After trade 2: Cash = $1,000
        # Trade 3: Try to buy 10 more (need $2,000, only have $1,000)
        valid, reason = policy.validate_position_change(
            asset="TSLA",
            current_quantity=70.0,
            quantity_delta=10.0,
            price=200.0,
            current_positions={},
            cash=1000.0,
        )
        assert valid is False

    def test_retail_investor_ira_account(self):
        """Test scenario: Retail investor in IRA (cash account)."""
        policy = CashAccountPolicy()

        # Try to short in IRA (not allowed)
        valid, reason = policy.validate_new_position(
            asset="AAPL",
            quantity=-100.0,
            price=150.0,
            current_positions={},
            cash=50000.0,
        )
        assert valid is False
        assert "Short selling not allowed" in reason

    def test_penny_stock_trader(self):
        """Test scenario: Trading penny stocks with limited capital."""
        policy = CashAccountPolicy()

        # Buy 10,000 shares at $0.50
        valid, _ = policy.validate_new_position(
            asset="PENNY",
            quantity=10000.0,
            price=0.50,
            current_positions={},
            cash=6000.0,  # Need $5,000
        )
        assert valid is True

        # Price drops to $0.30, try to average down
        valid, _ = policy.validate_position_change(
            asset="PENNY",
            current_quantity=10000.0,
            quantity_delta=5000.0,
            price=0.30,
            current_positions={},
            cash=1000.0,  # Remaining cash, need $1,500
        )
        assert valid is False

    def test_crypto_fractional_trading(self):
        """Test scenario: Buying fractional crypto."""
        policy = CashAccountPolicy()

        # Buy 0.1 BTC at $60,000
        valid, _ = policy.validate_new_position(
            asset="BTC-USD",
            quantity=0.1,
            price=60000.0,
            current_positions={},
            cash=10000.0,  # Need $6,000
        )
        assert valid is True

        # Add 0.05 more BTC
        valid, _ = policy.validate_position_change(
            asset="BTC-USD",
            current_quantity=0.1,
            quantity_delta=0.05,
            price=60000.0,
            current_positions={},
            cash=4000.0,  # Remaining, need $3,000
        )
        assert valid is True
