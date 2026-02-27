"""Unit tests for CryptoAccountPolicy."""

from datetime import datetime

from ml4t.backtest import Position
from ml4t.backtest.accounting.policy import UnifiedAccountPolicy


class TestCryptoAccountPolicyBuyingPower:
    """Tests for buying power calculation."""

    def test_positive_cash_buying_power(self):
        """Test buying power equals cash when cash is positive."""
        policy = UnifiedAccountPolicy(allow_short_selling=True)
        bp = policy.calculate_buying_power(cash=10000.0, positions={})
        assert bp == 10000.0

    def test_zero_cash_buying_power(self):
        """Test buying power is zero when cash is zero."""
        policy = UnifiedAccountPolicy(allow_short_selling=True)
        bp = policy.calculate_buying_power(cash=0.0, positions={})
        assert bp == 0.0

    def test_negative_cash_buying_power_is_zero(self):
        """Test buying power is capped at zero when cash is negative."""
        policy = UnifiedAccountPolicy(allow_short_selling=True)
        bp = policy.calculate_buying_power(cash=-5000.0, positions={})
        assert bp == 0.0

    def test_buying_power_ignores_positions(self):
        """Test buying power calculation ignores position values."""
        policy = UnifiedAccountPolicy(allow_short_selling=True)
        positions = {
            "BTC": Position(
                asset="BTC",
                quantity=1.0,
                entry_price=50000.0,
                current_price=55000.0,
                entry_time=datetime.now(),
            )
        }
        bp = policy.calculate_buying_power(cash=5000.0, positions=positions)
        # Crypto account buying power = cash only (no leverage)
        assert bp == 5000.0


class TestCryptoAccountPolicyShortSelling:
    """Tests for short selling permissions."""

    def test_allows_short_selling_returns_true(self):
        """Test that crypto accounts allow short selling."""
        policy = UnifiedAccountPolicy(allow_short_selling=True)
        assert policy.allows_short_selling() is True


class TestCryptoAccountPolicyNewPositionValidation:
    """Tests for validate_new_position method."""

    def test_valid_long_position_with_sufficient_cash(self):
        """Test approving long position with sufficient cash."""
        policy = UnifiedAccountPolicy(allow_short_selling=True)
        valid, reason = policy.validate_new_position(
            asset="BTC",
            quantity=1.0,
            price=50000.0,
            current_positions={},
            cash=60000.0,
        )
        assert valid is True
        assert reason == ""

    def test_valid_short_position_with_sufficient_cash(self):
        """Test approving short position with sufficient cash (key difference from cash account)."""
        policy = UnifiedAccountPolicy(allow_short_selling=True)
        valid, reason = policy.validate_new_position(
            asset="BTC",
            quantity=-1.0,  # Short
            price=50000.0,
            current_positions={},
            cash=60000.0,  # Cash covers notional
        )
        assert valid is True
        assert reason == ""

    def test_reject_long_position_insufficient_cash(self):
        """Test rejecting long position with insufficient cash."""
        policy = UnifiedAccountPolicy(allow_short_selling=True)
        valid, reason = policy.validate_new_position(
            asset="BTC",
            quantity=1.0,
            price=50000.0,
            current_positions={},
            cash=40000.0,  # Need $50,000
        )
        assert valid is False
        assert "Insufficient cash" in reason
        assert "long" in reason

    def test_reject_short_position_insufficient_cash(self):
        """Test rejecting short position with insufficient cash."""
        policy = UnifiedAccountPolicy(allow_short_selling=True)
        valid, reason = policy.validate_new_position(
            asset="BTC",
            quantity=-1.0,  # Short
            price=50000.0,
            current_positions={},
            cash=40000.0,  # Need $50,000 to cover notional
        )
        assert valid is False
        assert "Insufficient cash" in reason
        assert "short" in reason

    def test_fractional_position_validation(self):
        """Test validation with fractional crypto position."""
        policy = UnifiedAccountPolicy(allow_short_selling=True)
        valid, reason = policy.validate_new_position(
            asset="BTC",
            quantity=0.5,
            price=60000.0,
            current_positions={},
            cash=35000.0,  # Need $30,000
        )
        assert valid is True


class TestCryptoAccountPolicyPositionChangeValidation:
    """Tests for validate_position_change method."""

    def test_add_to_long_position_with_sufficient_cash(self):
        """Test adding to existing long position."""
        policy = UnifiedAccountPolicy(allow_short_selling=True)
        valid, reason = policy.validate_position_change(
            asset="BTC",
            current_quantity=1.0,
            quantity_delta=0.5,  # Add 0.5 more
            price=50000.0,
            current_positions={},
            cash=30000.0,  # Need $25,000
        )
        assert valid is True
        assert reason == ""

    def test_add_to_short_position_with_sufficient_cash(self):
        """Test adding to existing short position (allowed in crypto account)."""
        policy = UnifiedAccountPolicy(allow_short_selling=True)
        valid, reason = policy.validate_position_change(
            asset="BTC",
            current_quantity=-1.0,  # Already short 1
            quantity_delta=-0.5,  # Add to short
            price=50000.0,
            current_positions={},
            cash=30000.0,  # Need $25,000
        )
        assert valid is True
        assert reason == ""

    def test_close_long_position(self):
        """Test closing long position (always allowed)."""
        policy = UnifiedAccountPolicy(allow_short_selling=True)
        valid, reason = policy.validate_position_change(
            asset="BTC",
            current_quantity=1.0,
            quantity_delta=-1.0,  # Close entirely
            price=50000.0,
            current_positions={},
            cash=0.0,  # No cash needed to sell
        )
        assert valid is True
        assert reason == ""

    def test_close_short_position(self):
        """Test closing short position (buy to cover)."""
        policy = UnifiedAccountPolicy(allow_short_selling=True)
        valid, reason = policy.validate_position_change(
            asset="BTC",
            current_quantity=-1.0,  # Short 1
            quantity_delta=1.0,  # Buy to close
            price=50000.0,
            current_positions={},
            cash=0.0,  # Closing reduces risk
        )
        assert valid is True
        assert reason == ""

    def test_position_reversal_long_to_short(self):
        """Test position reversal from long to short (allowed in crypto)."""
        policy = UnifiedAccountPolicy(allow_short_selling=True)
        valid, reason = policy.validate_position_change(
            asset="BTC",
            current_quantity=1.0,  # Long 1
            quantity_delta=-2.0,  # Sell 2 -> ends up short 1
            price=50000.0,
            current_positions={},
            cash=100000.0,  # Need cash for the new short
        )
        assert valid is True
        assert reason == ""

    def test_position_reversal_short_to_long(self):
        """Test position reversal from short to long (allowed in crypto)."""
        policy = UnifiedAccountPolicy(allow_short_selling=True)
        valid, reason = policy.validate_position_change(
            asset="BTC",
            current_quantity=-1.0,  # Short 1
            quantity_delta=2.0,  # Buy 2 -> ends up long 1
            price=50000.0,
            current_positions={},
            cash=100000.0,  # Need cash for the new long
        )
        assert valid is True
        assert reason == ""

    def test_reversal_insufficient_cash(self):
        """Test reversal rejected when insufficient cash for new position."""
        policy = UnifiedAccountPolicy(allow_short_selling=True)
        valid, reason = policy.validate_position_change(
            asset="BTC",
            current_quantity=1.0,  # Long 1 BTC
            quantity_delta=-2.0,  # Sell 2 -> short 1
            price=50000.0,
            current_positions={},
            cash=30000.0,  # Need $50k for short position
        )
        assert valid is False
        assert "Insufficient cash" in reason

    def test_opening_short_from_flat(self):
        """Test opening short position from no position."""
        policy = UnifiedAccountPolicy(allow_short_selling=True)
        valid, reason = policy.validate_position_change(
            asset="BTC",
            current_quantity=0.0,  # No position
            quantity_delta=-1.0,  # Short 1
            price=50000.0,
            current_positions={},
            cash=60000.0,  # Sufficient cash
        )
        assert valid is True


class TestCryptoAccountPolicyHandleReversal:
    """Tests for handle_reversal method."""

    def test_reversal_with_sufficient_cash_after_close(self):
        """Test reversal allowed when cash after close covers new position."""
        policy = UnifiedAccountPolicy(allow_short_selling=True)
        valid, reason = policy.handle_reversal(
            asset="BTC",
            current_quantity=1.0,  # Long 1 BTC @ $50k
            order_quantity_delta=-2.0,  # Sell 2 -> short 1
            price=50000.0,
            current_positions={},
            cash=10000.0,  # Have $10k + $50k from close = $60k
            commission=100.0,
        )
        assert valid is True

    def test_reversal_insufficient_cash_after_close(self):
        """Test reversal rejected when insufficient cash after close."""
        policy = UnifiedAccountPolicy(allow_short_selling=True)
        valid, reason = policy.handle_reversal(
            asset="BTC",
            current_quantity=0.5,  # Long 0.5 BTC @ $50k = $25k
            order_quantity_delta=-2.0,  # Sell 2 -> short 1.5 @ $50k = $75k needed
            price=50000.0,
            current_positions={},
            cash=10000.0,  # Have $10k + $25k from close - $100 = $34,900
            commission=100.0,
        )
        assert valid is False
        assert "Insufficient cash" in reason


class TestCryptoAccountPolicyRealWorldScenarios:
    """Tests using realistic crypto trading scenarios."""

    def test_crypto_futures_trader(self):
        """Test scenario: Crypto futures trader opening long and short."""
        policy = UnifiedAccountPolicy(allow_short_selling=True)

        # Open long 1 BTC at $50,000
        valid, _ = policy.validate_new_position(
            asset="BTC",
            quantity=1.0,
            price=50000.0,
            current_positions={},
            cash=100000.0,
        )
        assert valid is True

        # Close long and go short (reversal)
        # After buying 1 BTC at $50k, we have $50k left
        # Reversing to short 1 BTC at $51k needs $51k, so this fails
        # Increase cash to make it succeed
        valid, reason = policy.validate_position_change(
            asset="BTC",
            current_quantity=1.0,
            quantity_delta=-2.0,  # Long -> Short 1 BTC
            price=51000.0,  # Price moved up
            current_positions={},
            cash=55000.0,  # Need $51,000 for short position
        )
        assert valid is True

    def test_eth_perpetual_short(self):
        """Test scenario: Opening short ETH perpetual."""
        policy = UnifiedAccountPolicy(allow_short_selling=True)

        # Short 10 ETH at $3,000
        valid, _ = policy.validate_new_position(
            asset="ETH",
            quantity=-10.0,
            price=3000.0,
            current_positions={},
            cash=40000.0,  # Need $30,000
        )
        assert valid is True

    def test_no_leverage_enforcement(self):
        """Test that crypto account doesn't allow leverage."""
        policy = UnifiedAccountPolicy(allow_short_selling=True)

        # Try to buy $100k worth of BTC with only $50k
        valid, reason = policy.validate_new_position(
            asset="BTC",
            quantity=2.0,
            price=50000.0,  # $100k total
            current_positions={},
            cash=50000.0,  # Only $50k
        )
        assert valid is False
        assert "Insufficient cash" in reason

        # Same for short
        valid, reason = policy.validate_new_position(
            asset="BTC",
            quantity=-2.0,
            price=50000.0,
            current_positions={},
            cash=50000.0,
        )
        assert valid is False
        assert "Insufficient cash" in reason
