"""Unit tests for MarginAccountPolicy."""

from datetime import datetime

import pytest

from src.ml4t.backtest.accounting.models import Position
from src.ml4t.backtest.accounting.policy import MarginAccountPolicy


class TestMarginAccountPolicyInitialization:
    """Tests for MarginAccountPolicy initialization."""

    def test_default_initialization(self):
        """Test initialization with default Reg T parameters."""
        policy = MarginAccountPolicy()
        assert policy.initial_margin == 0.5  # 50% = Reg T standard
        assert policy.maintenance_margin == 0.25  # 25% = Reg T standard

    def test_custom_initialization(self):
        """Test initialization with custom margin parameters."""
        policy = MarginAccountPolicy(initial_margin=0.3, maintenance_margin=0.15)
        assert policy.initial_margin == 0.3
        assert policy.maintenance_margin == 0.15

    def test_conservative_margin(self):
        """Test initialization with conservative (no leverage) parameters."""
        policy = MarginAccountPolicy(initial_margin=1.0, maintenance_margin=0.5)
        assert policy.initial_margin == 1.0
        assert policy.maintenance_margin == 0.5

    def test_invalid_initial_margin_too_low(self):
        """Test that initial_margin must be > 0."""
        with pytest.raises(ValueError, match="Initial margin must be in"):
            MarginAccountPolicy(initial_margin=0.0)

    def test_invalid_initial_margin_too_high(self):
        """Test that initial_margin must be <= 1.0."""
        with pytest.raises(ValueError, match="Initial margin must be in"):
            MarginAccountPolicy(initial_margin=1.5)

    def test_invalid_maintenance_margin_too_low(self):
        """Test that maintenance_margin must be > 0."""
        with pytest.raises(ValueError, match="Maintenance margin must be in"):
            MarginAccountPolicy(maintenance_margin=0.0)

    def test_invalid_maintenance_margin_too_high(self):
        """Test that maintenance_margin must be <= 1.0."""
        with pytest.raises(ValueError, match="Maintenance margin must be in"):
            MarginAccountPolicy(maintenance_margin=1.5)

    def test_invalid_maintenance_greater_than_initial(self):
        """Test that maintenance_margin must be < initial_margin."""
        with pytest.raises(ValueError, match="Maintenance margin.*must be <"):
            MarginAccountPolicy(initial_margin=0.25, maintenance_margin=0.5)

    def test_invalid_maintenance_equal_to_initial(self):
        """Test that maintenance_margin cannot equal initial_margin."""
        with pytest.raises(ValueError, match="Maintenance margin.*must be <"):
            MarginAccountPolicy(initial_margin=0.5, maintenance_margin=0.5)


class TestMarginAccountPolicyBuyingPower:
    """Tests for buying power calculation using NLV/MM/BP formula."""

    def test_cash_only_no_positions(self):
        """Test buying power with cash only (no positions).

        Example from docstring:
        cash=$100k, positions={}
        NLV = $100k, MM = $0
        BP = ($100k - $0) / 0.5 = $200k (2x leverage)
        """
        policy = MarginAccountPolicy(initial_margin=0.5, maintenance_margin=0.25)
        bp = policy.calculate_buying_power(cash=100_000.0, positions={})
        assert bp == 200_000.0  # 2x leverage

    def test_long_position_with_cash(self):
        """Test buying power with long position.

        Example from docstring:
        cash=$50k, long 1000 shares @ $100 = $100k market value
        NLV = $50k + $100k = $150k
        MM = $100k × 0.25 = $25k
        BP = ($150k - $25k) / 0.5 = $250k
        """
        policy = MarginAccountPolicy(initial_margin=0.5, maintenance_margin=0.25)
        positions = {
            "AAPL": Position(
                asset="AAPL",
                quantity=1000.0,
                avg_entry_price=100.0,
                current_price=100.0,
                entry_time=datetime.now(),
            )
        }
        bp = policy.calculate_buying_power(cash=50_000.0, positions=positions)
        assert bp == 250_000.0

    def test_short_position_with_cash(self):
        """Test buying power with short position.

        Example from docstring:
        cash=$150k, short 1000 shares @ $100 = -$100k market value
        NLV = $150k + (-$100k) = $50k
        MM = |-$100k| × 0.25 = $25k
        BP = ($50k - $25k) / 0.5 = $50k
        """
        policy = MarginAccountPolicy(initial_margin=0.5, maintenance_margin=0.25)
        positions = {
            "AAPL": Position(
                asset="AAPL",
                quantity=-1000.0,  # Short position
                avg_entry_price=100.0,
                current_price=100.0,
                entry_time=datetime.now(),
            )
        }
        bp = policy.calculate_buying_power(cash=150_000.0, positions=positions)
        assert bp == 50_000.0

    def test_underwater_account_negative_nlv(self):
        """Test buying power when account is underwater (negative equity).

        cash=-$10k, long 1000 shares @ $50 = $50k market value
        NLV = -$10k + $50k = $40k
        MM = $50k × 0.25 = $12.5k
        BP = ($40k - $12.5k) / 0.5 = $55k
        """
        policy = MarginAccountPolicy(initial_margin=0.5, maintenance_margin=0.25)
        positions = {
            "AAPL": Position(
                asset="AAPL",
                quantity=1000.0,
                avg_entry_price=100.0,  # Bought at $100
                current_price=50.0,  # Now at $50 (down 50%)
                entry_time=datetime.now(),
            )
        }
        bp = policy.calculate_buying_power(cash=-10_000.0, positions=positions)
        assert bp == 55_000.0

    def test_multiple_positions_long_and_short(self):
        """Test buying power with multiple positions (long and short).

        cash=$100k
        Long AAPL 1000 @ $100 = +$100k market value
        Short MSFT 500 @ $200 = -$100k market value
        NLV = $100k + $100k + (-$100k) = $100k
        MM = (|$100k| + |-$100k|) × 0.25 = $200k × 0.25 = $50k
        BP = ($100k - $50k) / 0.5 = $100k
        """
        policy = MarginAccountPolicy(initial_margin=0.5, maintenance_margin=0.25)
        positions = {
            "AAPL": Position(
                asset="AAPL",
                quantity=1000.0,
                avg_entry_price=100.0,
                current_price=100.0,
                entry_time=datetime.now(),
            ),
            "MSFT": Position(
                asset="MSFT",
                quantity=-500.0,  # Short
                avg_entry_price=200.0,
                current_price=200.0,
                entry_time=datetime.now(),
            ),
        }
        bp = policy.calculate_buying_power(cash=100_000.0, positions=positions)
        assert bp == 100_000.0

    def test_no_leverage_margin_account(self):
        """Test margin account with no leverage (100% initial margin).

        cash=$100k, positions={}
        BP = ($100k - $0) / 1.0 = $100k (no leverage)
        """
        policy = MarginAccountPolicy(initial_margin=1.0, maintenance_margin=0.5)
        bp = policy.calculate_buying_power(cash=100_000.0, positions={})
        assert bp == 100_000.0  # No leverage

    def test_high_leverage_margin_account(self):
        """Test margin account with high leverage (25% initial margin).

        cash=$100k, positions={}
        BP = ($100k - $0) / 0.25 = $400k (4x leverage)
        """
        policy = MarginAccountPolicy(initial_margin=0.25, maintenance_margin=0.15)
        bp = policy.calculate_buying_power(cash=100_000.0, positions={})
        assert bp == 400_000.0  # 4x leverage

    def test_margin_call_scenario_negative_buying_power(self):
        """Test buying power when account is severely underwater.

        cash=-$50k, long 1000 shares @ $40 = $40k market value
        NLV = -$50k + $40k = -$10k (negative equity!)
        MM = $40k × 0.25 = $10k
        BP = (-$10k - $10k) / 0.5 = -$40k (negative buying power = margin call)
        """
        policy = MarginAccountPolicy(initial_margin=0.5, maintenance_margin=0.25)
        positions = {
            "AAPL": Position(
                asset="AAPL",
                quantity=1000.0,
                avg_entry_price=100.0,  # Bought at $100
                current_price=40.0,  # Now at $40 (down 60%)
                entry_time=datetime.now(),
            )
        }
        bp = policy.calculate_buying_power(cash=-50_000.0, positions=positions)
        assert bp == -40_000.0  # Negative = forced liquidation


class TestMarginAccountPolicyShortSelling:
    """Tests for short selling permissions."""

    def test_allows_short_selling_returns_true(self):
        """Test that margin accounts allow short selling."""
        policy = MarginAccountPolicy()
        assert policy.allows_short_selling() is True


class TestMarginAccountPolicyNewPositionValidation:
    """Tests for validate_new_position method."""

    def test_valid_long_position_with_sufficient_buying_power(self):
        """Test approving long position with sufficient buying power."""
        policy = MarginAccountPolicy(initial_margin=0.5, maintenance_margin=0.25)
        # BP = $100k / 0.5 = $200k, order cost = 100 × $150 = $15k
        valid, reason = policy.validate_new_position(
            asset="AAPL",
            quantity=100.0,
            price=150.0,
            current_positions={},
            cash=100_000.0,
        )
        assert valid is True
        assert reason == ""

    def test_valid_short_position(self):
        """Test approving short position (margin accounts allow shorts)."""
        policy = MarginAccountPolicy(initial_margin=0.5, maintenance_margin=0.25)
        # BP = $100k / 0.5 = $200k, order cost = 100 × $150 = $15k
        valid, reason = policy.validate_new_position(
            asset="AAPL",
            quantity=-100.0,  # Short position
            price=150.0,
            current_positions={},
            cash=100_000.0,
        )
        assert valid is True
        assert reason == ""

    def test_reject_position_insufficient_buying_power(self):
        """Test rejecting position with insufficient buying power."""
        policy = MarginAccountPolicy(initial_margin=0.5, maintenance_margin=0.25)
        # BP = $10k / 0.5 = $20k, order cost = 1000 × $100 = $100k (too much)
        valid, reason = policy.validate_new_position(
            asset="AAPL",
            quantity=1000.0,
            price=100.0,
            current_positions={},
            cash=10_000.0,
        )
        assert valid is False
        assert "Insufficient buying power" in reason
        assert "need $100000.00" in reason
        assert "have $20000.00" in reason

    def test_valid_position_with_existing_positions(self):
        """Test approving position with existing positions affecting BP."""
        policy = MarginAccountPolicy(initial_margin=0.5, maintenance_margin=0.25)
        existing = {
            "AAPL": Position(
                asset="AAPL",
                quantity=1000.0,
                avg_entry_price=100.0,
                current_price=100.0,
                entry_time=datetime.now(),
            )
        }
        # cash=$50k, long $100k AAPL
        # NLV = $150k, MM = $100k × 0.25 = $25k
        # BP = ($150k - $25k) / 0.5 = $250k
        # New order: 100 × $200 = $20k (OK)
        valid, reason = policy.validate_new_position(
            asset="MSFT",
            quantity=100.0,
            price=200.0,
            current_positions=existing,
            cash=50_000.0,
        )
        assert valid is True
        assert reason == ""


class TestMarginAccountPolicyPositionChange:
    """Tests for validate_position_change method."""

    def test_valid_add_to_long_position(self):
        """Test adding to existing long position."""
        policy = MarginAccountPolicy(initial_margin=0.5, maintenance_margin=0.25)
        # current=100, delta=+50, cash=$100k -> BP = $200k
        # Risk increase = 50 × $150 = $7.5k
        valid, reason = policy.validate_position_change(
            asset="AAPL",
            current_quantity=100.0,
            quantity_delta=50.0,
            price=150.0,
            current_positions={},
            cash=100_000.0,
        )
        assert valid is True
        assert reason == ""

    def test_valid_partial_close_long(self):
        """Test partial close of long position (always allowed)."""
        policy = MarginAccountPolicy(initial_margin=0.5, maintenance_margin=0.25)
        # current=100, delta=-50 (partial close)
        valid, reason = policy.validate_position_change(
            asset="AAPL",
            current_quantity=100.0,
            quantity_delta=-50.0,
            price=150.0,
            current_positions={},
            cash=10_000.0,  # Doesn't matter for closes
        )
        assert valid is True
        assert reason == ""

    def test_valid_full_close_long(self):
        """Test full close of long position (always allowed)."""
        policy = MarginAccountPolicy(initial_margin=0.5, maintenance_margin=0.25)
        # current=100, delta=-100 (full close)
        valid, reason = policy.validate_position_change(
            asset="AAPL",
            current_quantity=100.0,
            quantity_delta=-100.0,
            price=150.0,
            current_positions={},
            cash=10_000.0,  # Doesn't matter for closes
        )
        assert valid is True
        assert reason == ""

    def test_valid_position_reversal_long_to_short(self):
        """Test position reversal from long to short (allowed in margin accounts)."""
        policy = MarginAccountPolicy(initial_margin=0.5, maintenance_margin=0.25)
        # current=100, delta=-200 -> new=-100 (reversed to short)
        # BP = $100k / 0.5 = $200k
        # Risk = |-100| × $150 = $15k
        valid, reason = policy.validate_position_change(
            asset="AAPL",
            current_quantity=100.0,
            quantity_delta=-200.0,
            price=150.0,
            current_positions={},
            cash=100_000.0,
        )
        assert valid is True
        assert reason == ""

    def test_valid_position_reversal_short_to_long(self):
        """Test position reversal from short to long."""
        policy = MarginAccountPolicy(initial_margin=0.5, maintenance_margin=0.25)
        # current=-100, delta=+200 -> new=+100 (reversed to long)
        # BP = $100k / 0.5 = $200k
        # Risk = |100| × $150 = $15k
        valid, reason = policy.validate_position_change(
            asset="AAPL",
            current_quantity=-100.0,
            quantity_delta=200.0,
            price=150.0,
            current_positions={},
            cash=100_000.0,
        )
        assert valid is True
        assert reason == ""

    def test_valid_add_to_short_position(self):
        """Test adding to existing short position."""
        policy = MarginAccountPolicy(initial_margin=0.5, maintenance_margin=0.25)
        # current=-100, delta=-50 (adding to short)
        # BP = $100k / 0.5 = $200k
        # Risk = |-50| × $150 = $7.5k
        valid, reason = policy.validate_position_change(
            asset="AAPL",
            current_quantity=-100.0,
            quantity_delta=-50.0,
            price=150.0,
            current_positions={},
            cash=100_000.0,
        )
        assert valid is True
        assert reason == ""

    def test_reject_position_change_insufficient_buying_power(self):
        """Test rejecting position change with insufficient buying power."""
        policy = MarginAccountPolicy(initial_margin=0.5, maintenance_margin=0.25)
        # current=100, delta=+1000
        # BP = $10k / 0.5 = $20k
        # Risk = 1000 × $100 = $100k (too much)
        valid, reason = policy.validate_position_change(
            asset="AAPL",
            current_quantity=100.0,
            quantity_delta=1000.0,
            price=100.0,
            current_positions={},
            cash=10_000.0,
        )
        assert valid is False
        assert "Insufficient buying power" in reason

    def test_reject_reversal_insufficient_buying_power(self):
        """Test rejecting position reversal when insufficient BP for new side."""
        policy = MarginAccountPolicy(initial_margin=0.5, maintenance_margin=0.25)
        # current=100, delta=-1000 -> new=-900 (large short)
        # BP = $10k / 0.5 = $20k
        # Risk = |-900| × $100 = $90k (too much)
        valid, reason = policy.validate_position_change(
            asset="AAPL",
            current_quantity=100.0,
            quantity_delta=-1000.0,
            price=100.0,
            current_positions={},
            cash=10_000.0,
        )
        assert valid is False
        assert "Insufficient buying power" in reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
