"""Unit tests for Gatekeeper order validation."""

from datetime import datetime

from ml4t.backtest import (
    NoCommission,
    Order,
    OrderSide,
    PercentageCommission,
)
from ml4t.backtest.accounting import (
    AccountState,
    CashAccountPolicy,
    Gatekeeper,
    Position,
)


class TestGatekeeperInitialization:
    """Test Gatekeeper initialization."""

    def test_init_with_cash_account(self):
        """Test initialization with cash account."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=100000.0, policy=policy)
        commission = NoCommission()

        gatekeeper = Gatekeeper(account, commission)

        assert gatekeeper.account is account
        assert gatekeeper.commission_model is commission


class TestCalculateQuantityDelta:
    """Test _calculate_quantity_delta helper method."""

    def test_buy_order_positive_delta(self):
        """Buy orders produce positive delta."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=100000.0, policy=policy)
        gatekeeper = Gatekeeper(account, NoCommission())

        delta = gatekeeper._calculate_quantity_delta(OrderSide.BUY, 100)
        assert delta == 100.0

    def test_sell_order_negative_delta(self):
        """Sell orders produce negative delta."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=100000.0, policy=policy)
        gatekeeper = Gatekeeper(account, NoCommission())

        delta = gatekeeper._calculate_quantity_delta(OrderSide.SELL, 100)
        assert delta == -100.0


class TestIsReversal:
    """Test _is_reversal helper method."""

    def test_no_position_is_not_reversal(self):
        """Opening a position from flat is not a reversal."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=100000.0, policy=policy)
        gatekeeper = Gatekeeper(account, NoCommission())

        # No position, buy 100
        is_reversal = gatekeeper._is_reversal(0.0, 100.0)
        assert not is_reversal

        # No position, sell 100 (shorting)
        is_reversal = gatekeeper._is_reversal(0.0, -100.0)
        assert not is_reversal

    def test_long_to_short_is_reversal(self):
        """Selling more than long position creates reversal."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=100000.0, policy=policy)
        gatekeeper = Gatekeeper(account, NoCommission())

        # Long 100, sell 150 (reverse to short 50)
        is_reversal = gatekeeper._is_reversal(100.0, -150.0)
        assert is_reversal

    def test_short_to_long_is_reversal(self):
        """Buying more than short position creates reversal."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=100000.0, policy=policy)
        gatekeeper = Gatekeeper(account, NoCommission())

        # Short 100, buy 150 (reverse to long 50)
        is_reversal = gatekeeper._is_reversal(-100.0, 150.0)
        assert is_reversal

    def test_closing_position_is_not_reversal(self):
        """Closing a position completely is not a reversal."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=100000.0, policy=policy)
        gatekeeper = Gatekeeper(account, NoCommission())

        # Long 100, sell 100 (close to flat)
        is_reversal = gatekeeper._is_reversal(100.0, -100.0)
        assert not is_reversal

        # Short 100, buy 100 (close to flat)
        is_reversal = gatekeeper._is_reversal(-100.0, 100.0)
        assert not is_reversal

    def test_reducing_position_is_not_reversal(self):
        """Partial close is not a reversal."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=100000.0, policy=policy)
        gatekeeper = Gatekeeper(account, NoCommission())

        # Long 100, sell 50 (reduce to long 50)
        is_reversal = gatekeeper._is_reversal(100.0, -50.0)
        assert not is_reversal

        # Short 100, buy 50 (reduce to short 50)
        is_reversal = gatekeeper._is_reversal(-100.0, 50.0)
        assert not is_reversal

    def test_adding_to_position_is_not_reversal(self):
        """Adding to a position is not a reversal."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=100000.0, policy=policy)
        gatekeeper = Gatekeeper(account, NoCommission())

        # Long 100, buy 50 (increase to long 150)
        is_reversal = gatekeeper._is_reversal(100.0, 50.0)
        assert not is_reversal

        # Short 100, sell 50 (increase to short 150)
        is_reversal = gatekeeper._is_reversal(-100.0, -50.0)
        assert not is_reversal


class TestIsReducingOrder:
    """Test _is_reducing_order helper method."""

    def test_no_position_is_not_reducing(self):
        """Opening a position is not reducing."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=100000.0, policy=policy)
        gatekeeper = Gatekeeper(account, NoCommission())

        # No position, buy 100
        is_reducing = gatekeeper._is_reducing_order(0.0, 100.0)
        assert not is_reducing

        # No position, sell 100 (shorting)
        is_reducing = gatekeeper._is_reducing_order(0.0, -100.0)
        assert not is_reducing

    def test_long_position_sell_is_reducing(self):
        """Selling from long position is reducing."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=100000.0, policy=policy)
        gatekeeper = Gatekeeper(account, NoCommission())

        # Long 100, sell 50 (partial close)
        is_reducing = gatekeeper._is_reducing_order(100.0, -50.0)
        assert is_reducing

        # Long 100, sell 100 (full close)
        is_reducing = gatekeeper._is_reducing_order(100.0, -100.0)
        assert is_reducing

    def test_long_position_buy_is_not_reducing(self):
        """Buying more long is not reducing."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=100000.0, policy=policy)
        gatekeeper = Gatekeeper(account, NoCommission())

        # Long 100, buy 50 more (adding)
        is_reducing = gatekeeper._is_reducing_order(100.0, 50.0)
        assert not is_reducing

    def test_short_position_buy_is_reducing(self):
        """Buying to cover short is reducing."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=100000.0, policy=policy)
        gatekeeper = Gatekeeper(account, NoCommission())

        # Short 100, buy 50 (partial cover)
        is_reducing = gatekeeper._is_reducing_order(-100.0, 50.0)
        assert is_reducing

        # Short 100, buy 100 (full cover)
        is_reducing = gatekeeper._is_reducing_order(-100.0, 100.0)
        assert is_reducing

    def test_short_position_sell_is_not_reducing(self):
        """Selling more short is not reducing."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=100000.0, policy=policy)
        gatekeeper = Gatekeeper(account, NoCommission())

        # Short 100, sell 50 more (adding)
        is_reducing = gatekeeper._is_reducing_order(-100.0, -50.0)
        assert not is_reducing

    def test_position_reversal_is_not_reducing(self):
        """Position reversal (long->short or short->long) is not reducing."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=100000.0, policy=policy)
        gatekeeper = Gatekeeper(account, NoCommission())

        # Long 100, sell 150 (reverse to short 50)
        is_reducing = gatekeeper._is_reducing_order(100.0, -150.0)
        assert not is_reducing

        # Short 100, buy 150 (reverse to long 50)
        is_reducing = gatekeeper._is_reducing_order(-100.0, 150.0)
        assert not is_reducing


class TestValidateOrderReducing:
    """Test validate_order for reducing orders (exits)."""

    def test_reducing_order_always_approved(self):
        """Reducing orders always approved regardless of cash."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=1000.0, policy=policy)  # Low cash
        gatekeeper = Gatekeeper(account, NoCommission())

        # Add a position manually
        account.positions["AAPL"] = Position(
            asset="AAPL",
            quantity=100.0,
            entry_price=150.0,
            current_price=150.0,
            entry_time=datetime.now(),
            bars_held=0,
        )

        # Sell 50 (reducing)
        order = Order(asset="AAPL", side=OrderSide.SELL, quantity=50)
        valid, reason = gatekeeper.validate_order(order, price=150.0)

        assert valid
        assert reason == ""

    def test_closing_order_always_approved(self):
        """Closing full position always approved."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=0.0, policy=policy)  # No cash!
        gatekeeper = Gatekeeper(account, NoCommission())

        # Add a position
        account.positions["AAPL"] = Position(
            asset="AAPL",
            quantity=100.0,
            entry_price=150.0,
            current_price=150.0,
            entry_time=datetime.now(),
            bars_held=0,
        )

        # Sell 100 (closing)
        order = Order(asset="AAPL", side=OrderSide.SELL, quantity=100)
        valid, reason = gatekeeper.validate_order(order, price=150.0)

        assert valid
        assert reason == ""


class TestValidateOrderOpening:
    """Test validate_order for opening orders (entries)."""

    def test_new_long_position_approved_with_cash(self):
        """New long position approved if sufficient cash."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=100000.0, policy=policy)
        gatekeeper = Gatekeeper(account, NoCommission())

        # Buy 100 @ $150 = $15,000 (have $100k)
        order = Order(asset="AAPL", side=OrderSide.BUY, quantity=100)
        valid, reason = gatekeeper.validate_order(order, price=150.0)

        assert valid
        assert reason == ""

    def test_new_long_position_rejected_insufficient_cash(self):
        """New long position rejected if insufficient cash."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=10000.0, policy=policy)
        gatekeeper = Gatekeeper(account, NoCommission())

        # Buy 100 @ $150 = $15,000 (only have $10k)
        order = Order(asset="AAPL", side=OrderSide.BUY, quantity=100)
        valid, reason = gatekeeper.validate_order(order, price=150.0)

        assert not valid
        assert "Insufficient cash" in reason

    def test_new_short_position_rejected_cash_account(self):
        """Cash account rejects new short positions."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=100000.0, policy=policy)
        gatekeeper = Gatekeeper(account, NoCommission())

        # Sell 100 (short) - cash account doesn't allow
        order = Order(asset="AAPL", side=OrderSide.SELL, quantity=100)
        valid, reason = gatekeeper.validate_order(order, price=150.0)

        assert not valid
        assert "Short selling not allowed" in reason

    def test_adding_to_long_position_approved(self):
        """Adding to long position approved if sufficient cash."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=20000.0, policy=policy)
        gatekeeper = Gatekeeper(account, NoCommission())

        # Existing position
        account.positions["AAPL"] = Position(
            asset="AAPL",
            quantity=100.0,
            entry_price=150.0,
            current_price=150.0,
            entry_time=datetime.now(),
            bars_held=0,
        )

        # Buy 50 more @ $150 = $7,500 (have $20k)
        order = Order(asset="AAPL", side=OrderSide.BUY, quantity=50)
        valid, reason = gatekeeper.validate_order(order, price=150.0)

        assert valid
        assert reason == ""

    def test_adding_to_long_position_rejected_insufficient_cash(self):
        """Adding to long position rejected if insufficient cash."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=5000.0, policy=policy)
        gatekeeper = Gatekeeper(account, NoCommission())

        # Existing position
        account.positions["AAPL"] = Position(
            asset="AAPL",
            quantity=100.0,
            entry_price=150.0,
            current_price=150.0,
            entry_time=datetime.now(),
            bars_held=0,
        )

        # Buy 50 more @ $150 = $7,500 (only have $5k)
        order = Order(asset="AAPL", side=OrderSide.BUY, quantity=50)
        valid, reason = gatekeeper.validate_order(order, price=150.0)

        assert not valid
        assert "Insufficient cash" in reason


class TestValidateOrderPositionReversal:
    """Test validate_order for position reversals."""

    def test_position_reversal_rejected_cash_account(self):
        """Cash account rejects position reversals (long -> short)."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=100000.0, policy=policy)
        gatekeeper = Gatekeeper(account, NoCommission())

        # Existing long position
        account.positions["AAPL"] = Position(
            asset="AAPL",
            quantity=100.0,
            entry_price=150.0,
            current_price=150.0,
            entry_time=datetime.now(),
            bars_held=0,
        )

        # Sell 150 (would reverse to short 50)
        order = Order(asset="AAPL", side=OrderSide.SELL, quantity=150)
        valid, reason = gatekeeper.validate_order(order, price=150.0)

        assert not valid
        assert "Position reversal not allowed" in reason

    def test_position_reversal_approved_margin_account(self):
        """Margin account approves position reversals with sufficient buying power."""
        from ml4t.backtest.accounting import MarginAccountPolicy

        policy = MarginAccountPolicy(initial_margin=0.5)
        account = AccountState(initial_cash=100000.0, policy=policy)
        gatekeeper = Gatekeeper(account, NoCommission())

        # Existing long position
        account.positions["AAPL"] = Position(
            asset="AAPL",
            quantity=100.0,
            entry_price=150.0,
            current_price=150.0,
            entry_time=datetime.now(),
            bars_held=0,
        )

        # Sell 150 @ $150 (reverse to short 50)
        # Close 100: +$15,000 cash
        # Open short 50: $7,500 margin requirement
        # Buying power is high enough to approve
        order = Order(asset="AAPL", side=OrderSide.SELL, quantity=150)
        valid, reason = gatekeeper.validate_order(order, price=150.0)

        assert valid
        assert reason == ""

    def test_position_reversal_rejected_margin_account_insufficient_bp(self):
        """Margin account rejects reversals with insufficient buying power."""
        from ml4t.backtest.accounting import MarginAccountPolicy

        policy = MarginAccountPolicy(initial_margin=0.5)
        account = AccountState(initial_cash=1000.0, policy=policy)  # Low cash
        gatekeeper = Gatekeeper(account, NoCommission())

        # Existing long position
        account.positions["AAPL"] = Position(
            asset="AAPL",
            quantity=10.0,
            entry_price=150.0,
            current_price=150.0,
            entry_time=datetime.now(),
            bars_held=0,
        )

        # Try to sell 100 @ $150 (reverse to short 90)
        # Close 10: +$1,500 cash
        # Open short 90: $13,500 margin requirement
        # Buying power insufficient
        order = Order(asset="AAPL", side=OrderSide.SELL, quantity=100)
        valid, reason = gatekeeper.validate_order(order, price=150.0)

        assert not valid
        assert "Insufficient buying power" in reason


class TestValidateOrderWithCommission:
    """Test validate_order with commission included."""

    def test_commission_included_in_cost(self):
        """Commission is included when checking cash constraints."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=15100.0, policy=policy)
        commission = PercentageCommission(rate=0.01)  # 1% commission
        gatekeeper = Gatekeeper(account, commission)

        # Buy 100 @ $150 = $15,000
        # Commission = $15,000 × 0.01 = $150
        # Total cost = $15,150
        # Cash available = $15,100
        # Should be rejected (insufficient by $50)
        order = Order(asset="AAPL", side=OrderSide.BUY, quantity=100)
        valid, reason = gatekeeper.validate_order(order, price=150.0)

        assert not valid
        assert "Insufficient cash" in reason

    def test_order_approved_with_commission(self):
        """Order approved when cash covers price + commission."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=15200.0, policy=policy)
        commission = PercentageCommission(rate=0.01)  # 1% commission
        gatekeeper = Gatekeeper(account, commission)

        # Buy 100 @ $150 = $15,000
        # Commission = $15,000 × 0.01 = $150
        # Total cost = $15,150
        # Cash available = $15,200
        # Should be approved
        order = Order(asset="AAPL", side=OrderSide.BUY, quantity=100)
        valid, reason = gatekeeper.validate_order(order, price=150.0)

        assert valid
        assert reason == ""


class TestValidateOrderEdgeCases:
    """Test edge cases for validate_order."""

    def test_zero_cash_rejects_new_buy(self):
        """Zero cash rejects new buy orders."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=0.0, policy=policy)
        gatekeeper = Gatekeeper(account, NoCommission())

        order = Order(asset="AAPL", side=OrderSide.BUY, quantity=1)
        valid, reason = gatekeeper.validate_order(order, price=1.0)

        assert not valid
        assert "Insufficient cash" in reason

    def test_exact_cash_amount_approved(self):
        """Order approved when cash exactly covers cost."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=15000.0, policy=policy)
        gatekeeper = Gatekeeper(account, NoCommission())

        # Buy 100 @ $150 = exactly $15,000
        order = Order(asset="AAPL", side=OrderSide.BUY, quantity=100)
        valid, reason = gatekeeper.validate_order(order, price=150.0)

        assert valid
        assert reason == ""

    def test_fractional_quantities(self):
        """Test with fractional quantities (crypto, fractional shares)."""
        policy = CashAccountPolicy()
        account = AccountState(initial_cash=10000.0, policy=policy)
        gatekeeper = Gatekeeper(account, NoCommission())

        # Buy 0.5 shares @ $150 = $75
        order = Order(asset="AAPL", side=OrderSide.BUY, quantity=0.5)
        valid, reason = gatekeeper.validate_order(order, price=150.0)

        assert valid
        assert reason == ""
