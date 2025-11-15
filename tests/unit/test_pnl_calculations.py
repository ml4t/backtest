"""Test P&L calculations for all asset classes."""


import pytest

from ml4t.backtest.core.assets import AssetClass, AssetSpec


class TestEquityPnL:
    """Test P&L calculations for equities."""

    def test_long_equity_profit(self):
        """Test profitable long equity position."""
        spec = AssetSpec(
            asset_id="AAPL",
            asset_class=AssetClass.EQUITY,
            currency="USD",
            taker_fee=0.001,  # 0.1%
        )

        # Long 100 shares: buy at $150, sell at $160
        pnl = spec.calculate_pnl(
            entry_price=150.0,
            exit_price=160.0,
            quantity=100.0,
            include_costs=False,
        )
        assert pnl == 1000.0  # (160 - 150) * 100

        # With costs
        pnl_with_costs = spec.calculate_pnl(
            entry_price=150.0,
            exit_price=160.0,
            quantity=100.0,
            include_costs=True,
        )
        # Entry cost: 100 * 150 * 0.001 = $15
        # Exit cost: 100 * 160 * 0.001 = $16
        # Total costs: $31
        assert pnl_with_costs == 1000.0 - 31.0

    def test_short_equity_profit(self):
        """Test profitable short equity position."""
        spec = AssetSpec(
            asset_id="AAPL",
            asset_class=AssetClass.EQUITY,
            currency="USD",
            taker_fee=0.001,
        )

        # Short 100 shares: sell at $160, buy back at $150
        pnl = spec.calculate_pnl(
            entry_price=160.0,
            exit_price=150.0,
            quantity=-100.0,  # Negative for short
            include_costs=False,
        )
        assert pnl == 1000.0  # -100 * (150 - 160) = 1000


class TestFuturePnL:
    """Test P&L calculations for futures."""

    def test_long_future_profit(self):
        """Test profitable long futures position."""
        spec = AssetSpec(
            asset_id="ES",
            asset_class=AssetClass.FUTURE,
            currency="USD",
            contract_size=50,  # ES mini has multiplier of 50
            initial_margin=5000.0,
            taker_fee=0.0002,
        )

        # Long 1 contract: buy at 4500, sell at 4510
        pnl = spec.calculate_pnl(
            entry_price=4500.0,
            exit_price=4510.0,
            quantity=1.0,
            include_costs=False,
        )
        assert pnl == 500.0  # (4510 - 4500) * 1 * 50

    def test_short_future_loss(self):
        """Test losing short futures position."""
        spec = AssetSpec(
            asset_id="CL",
            asset_class=AssetClass.FUTURE,
            currency="USD",
            contract_size=1000,  # Crude oil: 1000 barrels
            initial_margin=3000.0,
        )

        # Short 2 contracts: sell at 80, buy back at 85
        pnl = spec.calculate_pnl(
            entry_price=80.0,
            exit_price=85.0,
            quantity=-2.0,
            include_costs=False,
        )
        assert pnl == -10000.0  # -2 * (85 - 80) * 1000


class TestOptionPnL:
    """Test P&L calculations for options."""

    def test_long_call_option_premium_based(self):
        """Test long call option P&L using premium change."""
        spec = AssetSpec(
            asset_id="AAPL_CALL_150",
            asset_class=AssetClass.OPTION,
            currency="USD",
            contract_size=100,  # Standard equity option
            strike=150.0,
            taker_fee=0.0,  # Simplified for testing
        )

        # Long 1 call: buy at $2.00 premium, sell at $3.50 premium
        pnl = spec.calculate_pnl(
            entry_price=2.00,  # Premium at entry
            exit_price=3.50,   # Premium at exit
            quantity=1.0,
            include_costs=False,
        )
        assert pnl == 150.0  # (3.50 - 2.00) * 1 * 100

    def test_short_put_option_premium_based(self):
        """Test short put option P&L using premium change."""
        spec = AssetSpec(
            asset_id="SPY_PUT_400",
            asset_class=AssetClass.OPTION,
            currency="USD",
            contract_size=100,
            strike=400.0,
        )

        # Short 2 puts: sell at $5.00 premium, buy back at $2.00 premium
        pnl = spec.calculate_pnl(
            entry_price=5.00,  # Premium received
            exit_price=2.00,   # Premium paid to close
            quantity=-2.0,     # Negative for short
            include_costs=False,
        )
        assert pnl == 600.0  # -2 * (2.00 - 5.00) * 100

    def test_option_premium_based_method(self):
        """Test the dedicated premium-based calculation method."""
        spec = AssetSpec(
            asset_id="AAPL_CALL_150",
            asset_class=AssetClass.OPTION,
            currency="USD",
            contract_size=100,
            strike=150.0,
        )

        # Long 1 call: buy at $2.00, sell at $1.50 (loss)
        pnl = spec.calculate_pnl_premium_based(
            entry_premium=2.00,
            exit_premium=1.50,
            quantity=1.0,
            include_costs=False,
        )
        assert pnl == -50.0  # (1.50 - 2.00) * 1 * 100

    def test_option_enhanced_method(self):
        """Test enhanced P&L calculation with premium support."""
        spec = AssetSpec(
            asset_id="AAPL_CALL_150",
            asset_class=AssetClass.OPTION,
            currency="USD",
            contract_size=100,
            strike=150.0,
        )

        # With premium data - should use premium-based calculation
        pnl = spec.calculate_pnl_enhanced(
            entry_price=155.0,  # Underlying price (ignored)
            exit_price=160.0,   # Underlying price (ignored)
            quantity=1.0,
            entry_premium=2.00,  # Premium used
            exit_premium=3.50,   # Premium used
            include_costs=False,
        )
        assert pnl == 150.0  # (3.50 - 2.00) * 1 * 100

        # Without premium data - falls back to regular calculation
        # This should use the prices as premiums, not underlying prices
        pnl_no_premium = spec.calculate_pnl_enhanced(
            entry_price=2.00,   # Should be treated as premium
            exit_price=3.50,    # Should be treated as premium
            quantity=1.0,
            include_costs=False,
        )
        assert pnl_no_premium == 150.0  # (3.50 - 2.00) * 1 * 100


class TestFXPnL:
    """Test P&L calculations for forex."""

    def test_long_fx_profit(self):
        """Test profitable long FX position."""
        spec = AssetSpec(
            asset_id="EUR/USD",
            asset_class=AssetClass.FX,
            currency="USD",
            pip_value=0.0001,  # Standard for EUR/USD
            leverage_available=50,
            taker_fee=0.00002,  # 2 pips spread
        )

        # Long 10,000 EUR: buy at 1.1000, sell at 1.1050
        pnl = spec.calculate_pnl(
            entry_price=1.1000,
            exit_price=1.1050,
            quantity=10000.0,
            include_costs=False,
        )
        assert pnl == pytest.approx(50.0)  # 10000 * (1.1050 - 1.1000)

    def test_short_fx_loss(self):
        """Test losing short FX position."""
        spec = AssetSpec(
            asset_id="GBP/USD",
            asset_class=AssetClass.FX,
            currency="USD",
            pip_value=0.0001,
            leverage_available=50,
        )

        # Short 5,000 GBP: sell at 1.2500, buy back at 1.2550
        pnl = spec.calculate_pnl(
            entry_price=1.2500,
            exit_price=1.2550,
            quantity=-5000.0,
            include_costs=False,
        )
        assert pnl == pytest.approx(-25.0)  # -5000 * (1.2550 - 1.2500)

    def test_fx_with_costs(self):
        """Test FX P&L with trading costs."""
        spec = AssetSpec(
            asset_id="USD/JPY",
            asset_class=AssetClass.FX,
            currency="JPY",
            pip_value=0.01,  # For JPY pairs
            taker_fee=0.00003,  # 3 pips spread
        )

        # Long 100,000 USD: buy at 110.00, sell at 110.50
        pnl = spec.calculate_pnl(
            entry_price=110.00,
            exit_price=110.50,
            quantity=100000.0,
            include_costs=True,
        )

        # Base P&L: 100000 * (110.50 - 110.00) = 50000 JPY
        # Entry cost: 100000 * 110.00 * 0.00003 = 330 JPY
        # Exit cost: 100000 * 110.50 * 0.00003 = 331.5 JPY
        # Net P&L: 50000 - 330 - 331.5 = 49338.5 JPY
        assert abs(pnl - 49338.5) < 0.01


class TestCryptoPnL:
    """Test P&L calculations for crypto."""

    def test_spot_crypto_profit(self):
        """Test profitable spot crypto position."""
        spec = AssetSpec(
            asset_id="BTC/USDT",
            asset_class=AssetClass.CRYPTO,
            currency="USDT",
            taker_fee=0.001,  # 0.1%
        )

        # Long 0.5 BTC: buy at 40,000, sell at 45,000
        pnl = spec.calculate_pnl(
            entry_price=40000.0,
            exit_price=45000.0,
            quantity=0.5,
            include_costs=False,
        )
        assert pnl == 2500.0  # 0.5 * (45000 - 40000)

    def test_leveraged_crypto_short(self):
        """Test leveraged short crypto position."""
        spec = AssetSpec(
            asset_id="ETH/USDT-PERP",
            asset_class=AssetClass.CRYPTO,
            currency="USDT",
            leverage_available=10,
            taker_fee=0.0005,
        )

        # Short 5 ETH: sell at 3000, buy back at 2800
        pnl = spec.calculate_pnl(
            entry_price=3000.0,
            exit_price=2800.0,
            quantity=-5.0,
            include_costs=False,
        )
        assert pnl == 1000.0  # -5 * (2800 - 3000)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_quantity(self):
        """Test P&L with zero quantity."""
        spec = AssetSpec(
            asset_id="TEST",
            asset_class=AssetClass.EQUITY,
        )

        pnl = spec.calculate_pnl(100.0, 110.0, 0.0)
        assert pnl == 0.0

    def test_zero_price_change(self):
        """Test P&L with no price change."""
        spec = AssetSpec(
            asset_id="TEST",
            asset_class=AssetClass.EQUITY,
            taker_fee=0.001,
        )

        pnl = spec.calculate_pnl(100.0, 100.0, 100.0, include_costs=False)
        assert pnl == 0.0

        # With costs, should be negative
        pnl_with_costs = spec.calculate_pnl(100.0, 100.0, 100.0, include_costs=True)
        assert pnl_with_costs < 0.0  # Lost money on fees

    def test_premium_method_on_non_option(self):
        """Test that premium-based method raises error for non-options."""
        spec = AssetSpec(
            asset_id="AAPL",
            asset_class=AssetClass.EQUITY,
        )

        with pytest.raises(ValueError, match="Premium-based P&L calculation is only for options"):
            spec.calculate_pnl_premium_based(2.00, 3.00, 1.0)


class TestOptionExpiryPnL:
    """Test P&L calculations for options held to expiry."""

    def test_long_call_expires_in_the_money(self):
        """Test long call option that expires in the money."""
        spec = AssetSpec(
            asset_id="AAPL_CALL_150",
            asset_class=AssetClass.OPTION,
            currency="USD",
            contract_size=100,
            strike=150.0,
        )

        # Long 1 call: paid $2.00 premium, underlying at $155 at expiry
        # Intrinsic value = max(0, 155 - 150) = $5.00
        # P&L = (5.00 - 2.00) * 1 * 100 = $300
        pnl = spec.calculate_option_pnl_at_expiry(
            entry_premium=2.00,
            underlying_price_at_expiry=155.0,
            quantity=1.0,
            option_type="call",
            include_costs=False,
        )
        assert pnl == 300.0

    def test_long_put_expires_worthless(self):
        """Test long put option that expires worthless."""
        spec = AssetSpec(
            asset_id="SPY_PUT_400",
            asset_class=AssetClass.OPTION,
            currency="USD",
            contract_size=100,
            strike=400.0,
        )

        # Long 2 puts: paid $3.00 premium, underlying at $410 at expiry
        # Intrinsic value = max(0, 400 - 410) = $0
        # P&L = (0 - 3.00) * 2 * 100 = -$600
        pnl = spec.calculate_option_pnl_at_expiry(
            entry_premium=3.00,
            underlying_price_at_expiry=410.0,
            quantity=2.0,
            option_type="put",
            include_costs=False,
        )
        assert pnl == -600.0

    def test_short_call_expires_worthless(self):
        """Test short call option that expires worthless (profitable)."""
        spec = AssetSpec(
            asset_id="AAPL_CALL_160",
            asset_class=AssetClass.OPTION,
            currency="USD",
            contract_size=100,
            strike=160.0,
        )

        # Short 1 call: received $1.50 premium, underlying at $155 at expiry
        # Intrinsic value = max(0, 155 - 160) = $0
        # P&L = (1.50 - 0) * 1 * 100 = $150 (keep full premium)
        pnl = spec.calculate_option_pnl_at_expiry(
            entry_premium=1.50,
            underlying_price_at_expiry=155.0,
            quantity=-1.0,
            option_type="call",
            include_costs=False,
        )
        assert pnl == 150.0

    def test_short_put_expires_in_the_money(self):
        """Test short put option that expires in the money (loss)."""
        spec = AssetSpec(
            asset_id="SPY_PUT_400",
            asset_class=AssetClass.OPTION,
            currency="USD",
            contract_size=100,
            strike=400.0,
        )

        # Short 1 put: received $2.00 premium, underlying at $395 at expiry
        # Intrinsic value = max(0, 400 - 395) = $5.00
        # P&L = (2.00 - 5.00) * 1 * 100 = -$300
        pnl = spec.calculate_option_pnl_at_expiry(
            entry_premium=2.00,
            underlying_price_at_expiry=395.0,
            quantity=-1.0,
            option_type="put",
            include_costs=False,
        )
        assert pnl == -300.0

    def test_expiry_method_on_non_option(self):
        """Test that expiry P&L method raises error for non-options."""
        spec = AssetSpec(
            asset_id="AAPL",
            asset_class=AssetClass.EQUITY,
        )

        with pytest.raises(ValueError, match="Expiry P&L calculation is only for options"):
            spec.calculate_option_pnl_at_expiry(2.00, 155.0, 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
