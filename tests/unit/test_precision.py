"""Tests for PrecisionManager - numerical rounding for asset classes."""

import pytest
from ml4t.backtest.core.assets import AssetSpec, AssetClass
from ml4t.backtest.core.precision import PrecisionManager, PRECISION_DEFAULTS


class TestPrecisionManager:
    """Test PrecisionManager rounding behavior."""

    def test_equity_whole_shares_only(self):
        """Equities should truncate to whole shares."""
        pm = PrecisionManager(position_decimals=0, price_decimals=2)

        assert pm.round_quantity(100.0) == 100.0
        assert pm.round_quantity(100.4) == 100.0  # Truncates down
        assert pm.round_quantity(100.9) == 100.0  # Truncates down
        assert pm.round_quantity(101.0) == 101.0

    def test_equity_penny_prices(self):
        """Equity prices should round to cents."""
        pm = PrecisionManager(position_decimals=0, price_decimals=2)

        assert pm.round_price(123.45) == 123.45
        assert pm.round_price(123.456) == 123.46  # Rounds up
        assert pm.round_price(123.454) == 123.45  # Rounds down

    def test_crypto_fractional_shares(self):
        """Crypto should allow 8 decimal places (satoshi precision)."""
        pm = PrecisionManager(position_decimals=8, price_decimals=8)

        assert pm.round_quantity(3.12345678) == 3.12345678
        assert pm.round_quantity(3.123456789) == 3.12345679  # Rounds up
        assert pm.round_quantity(3.123456781) == 3.12345678  # Rounds down

    def test_crypto_high_precision_prices(self):
        """Crypto prices should support 8 decimals."""
        pm = PrecisionManager(position_decimals=8, price_decimals=8)

        assert pm.round_price(28894.68123456) == 28894.68123456
        assert pm.round_price(28894.681234567) == 28894.68123457

    def test_cash_always_cents_for_usd(self):
        """Commission and cash should always round to cents."""
        pm_equity = PrecisionManager(position_decimals=0, price_decimals=2)
        pm_crypto = PrecisionManager(position_decimals=8, price_decimals=8)

        # Both should round commission to cents
        assert pm_equity.round_cash(10.999) == 11.0
        assert pm_crypto.round_cash(10.999) == 11.0

        assert pm_equity.round_cash(123.456) == 123.46
        assert pm_crypto.round_cash(123.456) == 123.46

    def test_from_asset_spec_equity(self):
        """Create PrecisionManager from equity AssetSpec."""
        spec = AssetSpec(asset_id="AAPL", asset_class=AssetClass.EQUITY)
        pm = PrecisionManager.from_asset_spec(spec)

        assert pm.position_decimals == 0  # Whole shares
        assert pm.price_decimals == 2  # Penny prices
        assert pm.cash_decimals == 2  # Cent commission

        # Test behavior
        assert pm.round_quantity(100.7) == 100.0
        assert pm.round_price(123.456) == 123.46

    def test_from_asset_spec_crypto(self):
        """Create PrecisionManager from crypto AssetSpec."""
        spec = AssetSpec(asset_id="BTC", asset_class=AssetClass.CRYPTO)
        pm = PrecisionManager.from_asset_spec(spec)

        assert pm.position_decimals == 8  # Satoshi
        assert pm.price_decimals == 8  # 8 decimals
        assert pm.cash_decimals == 2  # Commission still in USD cents

        # Test behavior
        assert pm.round_quantity(3.123456789) == 3.12345679
        assert pm.round_price(28894.681234567) == 28894.68123457

    def test_from_asset_spec_with_overrides(self):
        """AssetSpec can override default precision."""
        spec = AssetSpec(
            asset_id="CUSTOM",
            asset_class=AssetClass.EQUITY,
            position_decimals=2,  # Allow fractional shares
            price_decimals=4,  # High precision prices
        )
        pm = PrecisionManager.from_asset_spec(spec)

        assert pm.position_decimals == 2  # Override
        assert pm.price_decimals == 4  # Override
        assert pm.cash_decimals == 2  # Default

        # Test behavior
        assert pm.round_quantity(100.777) == 100.78  # 2 decimals
        assert pm.round_price(123.45678) == 123.4568  # 4 decimals

    def test_from_asset_class_string(self):
        """Create PrecisionManager from asset class name."""
        pm_equity = PrecisionManager.from_asset_class("EQUITY")
        assert pm_equity.position_decimals == 0
        assert pm_equity.price_decimals == 2

        pm_crypto = PrecisionManager.from_asset_class("CRYPTO")
        assert pm_crypto.position_decimals == 8
        assert pm_crypto.price_decimals == 8

    def test_is_position_zero_equity(self):
        """Test zero detection for equities (integer positions)."""
        pm = PrecisionManager(position_decimals=0, price_decimals=2)

        # Below 0.5 rounds to zero
        assert pm.is_position_zero(0.0) is True
        assert pm.is_position_zero(0.1) is True
        assert pm.is_position_zero(0.4) is True
        assert pm.is_position_zero(-0.4) is True

        # 0.5 and above rounds to 1
        assert pm.is_position_zero(0.5) is False
        assert pm.is_position_zero(1.0) is False

    def test_is_position_zero_crypto(self):
        """Test zero detection for crypto (8 decimal precision)."""
        pm = PrecisionManager(position_decimals=8, price_decimals=8)

        # Below precision threshold
        assert pm.is_position_zero(0.0) is True
        assert pm.is_position_zero(0.000000001) is True
        assert pm.is_position_zero(-0.000000001) is True

        # Above precision threshold
        assert pm.is_position_zero(0.00000001) is False
        assert pm.is_position_zero(0.00001) is False

    def test_precision_defaults_exist_for_all_classes(self):
        """Verify defaults are defined for all asset classes."""
        expected_classes = ["EQUITY", "CRYPTO", "FUTURE", "OPTION", "FX", "BOND", "COMMODITY"]

        for asset_class in expected_classes:
            assert asset_class in PRECISION_DEFAULTS, f"Missing defaults for {asset_class}"
            defaults = PRECISION_DEFAULTS[asset_class]
            assert defaults.position_decimals >= 0
            assert defaults.price_decimals >= 0
            assert defaults.cash_decimals == 2  # USD always 2 decimals

    def test_futures_whole_contracts(self):
        """Futures should use whole contracts (integer)."""
        spec = AssetSpec(asset_id="ES", asset_class=AssetClass.FUTURE)
        pm = PrecisionManager.from_asset_spec(spec)

        assert pm.position_decimals == 0  # Whole contracts
        assert pm.round_quantity(5.7) == 5.0

    def test_options_whole_contracts(self):
        """Options should use whole contracts (integer)."""
        spec = AssetSpec(asset_id="SPY_CALL_450", asset_class=AssetClass.OPTION)
        pm = PrecisionManager.from_asset_spec(spec)

        assert pm.position_decimals == 0  # Whole contracts
        assert pm.round_quantity(10.3) == 10.0

    def test_consistent_rounding_eliminates_mismatches(self):
        """Demonstrate that consistent rounding prevents position tracking bugs.

        This is the core issue that PrecisionManager solves.
        """
        pm = PrecisionManager(position_decimals=8, price_decimals=2)

        # Strategy calculates entry size
        cash = 100000.0
        price = 28894.68
        commission_pct = 0.001
        fixed_fee = 2.0

        # WITHOUT rounding (old behavior - causes bugs):
        size_raw = (cash - fixed_fee) / (price * (1 + commission_pct))
        # Result: 3.45731816023233084... (many decimals)

        # WITH rounding (new behavior - consistent):
        size = pm.round_quantity(size_raw)
        # Result: 3.45731816 (exactly 8 decimals)

        # Both strategy AND broker now use: 3.45731816
        # Exit request: 3.45731816
        # Broker has: 3.45731816
        # ✅ Perfect match! No leftover positions.

        assert size == 3.45731816
        assert pm.round_quantity(size) == size  # Idempotent


class TestPrecisionIntegrationWithAssetSpec:
    """Test PrecisionManager integration with AssetSpec."""

    def test_asset_spec_creates_precision_manager(self):
        """AssetSpec.get_precision_manager() creates correct manager."""
        spec = AssetSpec(asset_id="AAPL", asset_class=AssetClass.EQUITY)
        pm = spec.get_precision_manager()

        assert isinstance(pm, PrecisionManager)
        assert pm.position_decimals == 0
        assert pm.price_decimals == 2

    def test_different_assets_get_different_precision(self):
        """Different asset classes get different precision rules."""
        equity_spec = AssetSpec(asset_id="AAPL", asset_class=AssetClass.EQUITY)
        crypto_spec = AssetSpec(asset_id="BTC", asset_class=AssetClass.CRYPTO)

        pm_equity = equity_spec.get_precision_manager()
        pm_crypto = crypto_spec.get_precision_manager()

        # Equities: whole shares
        assert pm_equity.round_quantity(100.7) == 100.0

        # Crypto: fractional
        assert pm_crypto.round_quantity(3.123456789) == 3.12345679

    def test_custom_precision_overrides(self):
        """Test that AssetSpec overrides work correctly."""
        # Broker that allows fractional equity shares
        spec = AssetSpec(
            asset_id="AAPL",
            asset_class=AssetClass.EQUITY,
            position_decimals=6,  # Allow fractional shares
        )
        pm = spec.get_precision_manager()

        # Should use override, not default (0)
        assert pm.position_decimals == 6
        assert pm.round_quantity(100.123456789) == 100.123457
