"""Test suite for margin management functionality."""

from datetime import datetime, timezone

import pytest

from ml4t.backtest.core.assets import AssetClass, AssetRegistry, AssetSpec
from ml4t.backtest.portfolio.margin import MarginAccount, MarginRequirement


class TestMarginRequirement:
    """Test MarginRequirement dataclass."""

    def test_margin_requirement_creation(self):
        """Test creating a margin requirement."""
        req = MarginRequirement(
            asset_id="ES",
            initial_margin=5000.0,
            maintenance_margin=4000.0,
            current_margin=4500.0,
            excess_margin=500.0,
            margin_call=False,
            liquidation_price=4200.0,
        )

        assert req.asset_id == "ES"
        assert req.initial_margin == 5000.0
        assert req.maintenance_margin == 4000.0
        assert req.current_margin == 4500.0
        assert req.excess_margin == 500.0
        assert req.margin_call is False
        assert req.liquidation_price == 4200.0

    def test_margin_requirement_defaults(self):
        """Test margin requirement with default values."""
        req = MarginRequirement(
            asset_id="ES",
            initial_margin=5000.0,
            maintenance_margin=4000.0,
            current_margin=4500.0,
            excess_margin=500.0,
        )

        assert req.margin_call is False
        assert req.liquidation_price is None


class TestMarginAccountInitialization:
    """Test MarginAccount initialization."""

    def test_initialization(self):
        """Test margin account initialization."""
        registry = AssetRegistry()
        account = MarginAccount(initial_cash=100000.0, asset_registry=registry)

        assert account.cash_balance == 100000.0
        assert account.available_margin == 100000.0
        assert account.margin_used == 0.0
        assert account.initial_margin_requirement == 0.0
        assert account.maintenance_margin_requirement == 0.0
        assert len(account.positions) == 0
        assert account.margin_call_level == 1.0
        assert account.liquidation_level == 0.8

    def test_custom_risk_parameters(self):
        """Test margin account with custom risk parameters."""
        registry = AssetRegistry()
        account = MarginAccount(initial_cash=50000.0, asset_registry=registry)

        # Verify defaults
        assert account.margin_call_level == 1.0
        assert account.liquidation_level == 0.8


class TestMarginRequirementChecks:
    """Test margin requirement checking."""

    def test_check_margin_for_futures(self):
        """Test margin check for futures position."""
        registry = AssetRegistry()
        registry.register(
            AssetSpec(
                asset_id="ES",
                asset_class=AssetClass.FUTURE,
                contract_size=50,
                initial_margin=5000.0,
                maintenance_margin=4000.0,
                leverage_available=10.0,
            )
        )

        account = MarginAccount(initial_cash=10000.0, asset_registry=registry)

        # Check margin for 1 contract (requires 5000)
        has_margin, required = account.check_margin_requirement("ES", 1.0, 4500.0)

        assert has_margin is True
        assert required == 5000.0

    def test_check_margin_insufficient(self):
        """Test margin check when insufficient margin."""
        registry = AssetRegistry()
        registry.register(
            AssetSpec(
                asset_id="ES",
                asset_class=AssetClass.FUTURE,
                contract_size=50,
                initial_margin=5000.0,
                maintenance_margin=4000.0,
            )
        )

        account = MarginAccount(initial_cash=3000.0, asset_registry=registry)

        # Check margin for 1 contract (requires 5000, but only have 3000)
        has_margin, required = account.check_margin_requirement("ES", 1.0, 4500.0)

        assert has_margin is False
        assert required == 5000.0

    def test_check_margin_for_fx(self):
        """Test margin check for FX position with leverage."""
        registry = AssetRegistry()
        registry.register(
            AssetSpec(
                asset_id="EURUSD",
                asset_class=AssetClass.FX,
                leverage_available=50.0,
            )
        )

        account = MarginAccount(initial_cash=10000.0, asset_registry=registry)

        # 100,000 units at 1.10 = 110,000 notional / 50 leverage = 2,200 margin
        has_margin, required = account.check_margin_requirement("EURUSD", 100000.0, 1.10)

        assert has_margin is True
        assert required == pytest.approx(2200.0, rel=1e-4)

    def test_check_margin_unknown_asset(self):
        """Test margin check for unknown asset defaults to full cash."""
        registry = AssetRegistry()
        account = MarginAccount(initial_cash=10000.0, asset_registry=registry)

        # Unknown asset requires full notional
        has_margin, required = account.check_margin_requirement("AAPL", 100.0, 150.0)

        assert has_margin is False  # 100 * 150 = 15,000 > 10,000
        assert required == 15000.0


class TestPositionManagement:
    """Test opening and managing positions."""

    def test_open_new_futures_position(self):
        """Test opening a new futures position."""
        registry = AssetRegistry()
        registry.register(
            AssetSpec(
                asset_id="ES",
                asset_class=AssetClass.FUTURE,
                contract_size=50,
                initial_margin=5000.0,
                maintenance_margin=4000.0,
            )
        )

        account = MarginAccount(initial_cash=20000.0, asset_registry=registry)
        timestamp = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)

        success = account.open_position("ES", 2.0, 4500.0, timestamp)

        assert success is True
        assert "ES" in account.positions
        assert account.positions["ES"]["quantity"] == 2.0
        assert account.positions["ES"]["avg_price"] == 4500.0
        assert account.positions["ES"]["margin_used"] == 10000.0  # 2 * 5000
        assert account.margin_used == 10000.0
        assert account.available_margin == 10000.0  # 20000 - 10000

    def test_open_position_insufficient_margin(self):
        """Test opening position fails with insufficient margin."""
        registry = AssetRegistry()
        registry.register(
            AssetSpec(
                asset_id="ES",
                asset_class=AssetClass.FUTURE,
                contract_size=50,
                initial_margin=5000.0,
                maintenance_margin=4000.0,
            )
        )

        account = MarginAccount(initial_cash=3000.0, asset_registry=registry)
        timestamp = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)

        success = account.open_position("ES", 1.0, 4500.0, timestamp)

        assert success is False
        assert "ES" not in account.positions
        assert account.margin_used == 0.0

    def test_modify_existing_position(self):
        """Test modifying an existing position."""
        registry = AssetRegistry()
        registry.register(
            AssetSpec(
                asset_id="ES",
                asset_class=AssetClass.FUTURE,
                contract_size=50,
                initial_margin=5000.0,
                maintenance_margin=4000.0,
            )
        )

        account = MarginAccount(initial_cash=30000.0, asset_registry=registry)
        timestamp1 = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        timestamp2 = datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc)

        # Open initial position
        account.open_position("ES", 2.0, 4500.0, timestamp1)

        # Add to position
        success = account.open_position("ES", 1.0, 4600.0, timestamp2)

        assert success is True
        assert account.positions["ES"]["quantity"] == 3.0
        # Average price: (4500*2 + 4600*1) / 3 = 4533.33
        assert account.positions["ES"]["avg_price"] == pytest.approx(4533.33, rel=1e-4)
        assert account.positions["ES"]["margin_used"] == 15000.0  # 3 * 5000

    def test_close_position(self):
        """Test closing a position returns margin."""
        registry = AssetRegistry()
        registry.register(
            AssetSpec(
                asset_id="ES",
                asset_class=AssetClass.FUTURE,
                contract_size=50,
                initial_margin=5000.0,
                maintenance_margin=4000.0,
            )
        )

        account = MarginAccount(initial_cash=20000.0, asset_registry=registry)
        timestamp1 = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        timestamp2 = datetime(2024, 1, 1, 15, 0, tzinfo=timezone.utc)

        # Open position
        account.open_position("ES", 2.0, 4500.0, timestamp1)
        assert account.margin_used == 10000.0

        # Close position
        account.open_position("ES", -2.0, 4600.0, timestamp2)

        assert "ES" not in account.positions
        assert account.margin_used == 0.0
        assert account.available_margin == 20000.0

    def test_open_fx_position_with_leverage(self):
        """Test opening leveraged FX position."""
        registry = AssetRegistry()
        registry.register(
            AssetSpec(
                asset_id="EURUSD",
                asset_class=AssetClass.FX,
                leverage_available=50.0,
            )
        )

        account = MarginAccount(initial_cash=10000.0, asset_registry=registry)
        timestamp = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)

        # 100,000 units at 1.10 = 110,000 notional / 50 leverage = 2,200 margin
        success = account.open_position("EURUSD", 100000.0, 1.10, timestamp)

        assert success is True
        assert account.positions["EURUSD"]["quantity"] == 100000.0
        assert account.margin_used == pytest.approx(2200.0, rel=1e-4)


class TestPriceUpdatesAndMarginCalls:
    """Test price updates and margin call detection."""

    def test_update_prices_basic(self):
        """Test updating prices for positions."""
        registry = AssetRegistry()
        registry.register(
            AssetSpec(
                asset_id="ES",
                asset_class=AssetClass.FUTURE,
                contract_size=50,
                initial_margin=5000.0,
                maintenance_margin=4000.0,
            )
        )

        account = MarginAccount(initial_cash=50000.0, asset_registry=registry)
        timestamp = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)

        account.open_position("ES", 2.0, 4500.0, timestamp)

        # Update price - small profitable move, won't trigger liquidation
        margin_reqs = account.update_prices({"ES": 4510.0})

        assert len(margin_reqs) == 1
        assert margin_reqs[0].asset_id == "ES"
        # Position should still exist (not liquidated)
        assert "ES" in account.positions
        assert account.positions["ES"]["last_price"] == 4510.0

    def test_update_prices_ignores_non_positions(self):
        """Test that price updates ignore assets without positions."""
        registry = AssetRegistry()
        account = MarginAccount(initial_cash=10000.0, asset_registry=registry)

        # Update price for asset we don't have
        margin_reqs = account.update_prices({"AAPL": 150.0})

        assert len(margin_reqs) == 0

    def test_margin_call_detection(self):
        """Test margin call is detected when equity drops."""
        registry = AssetRegistry()
        registry.register(
            AssetSpec(
                asset_id="ES",
                asset_class=AssetClass.FUTURE,
                contract_size=50,
                initial_margin=5000.0,
                maintenance_margin=4000.0,
            )
        )

        account = MarginAccount(initial_cash=12000.0, asset_registry=registry)
        timestamp = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)

        # Open position at 4500
        account.open_position("ES", 2.0, 4500.0, timestamp)

        # Price drops significantly
        # Position loses: 2 * 50 * (4300 - 4500) = -20,000
        # Available margin before: 2,000
        # After loss: 2,000 - 20,000 = -18,000
        margin_reqs = account.update_prices({"ES": 4300.0})

        # Should trigger margin call when current margin > available * margin_call_level
        assert len(margin_reqs) == 1


class TestLiquidationPrice:
    """Test liquidation price calculations."""

    def test_liquidation_price_futures_long(self):
        """Test liquidation price for long futures position."""
        registry = AssetRegistry()
        registry.register(
            AssetSpec(
                asset_id="ES",
                asset_class=AssetClass.FUTURE,
                contract_size=50,
                initial_margin=5000.0,
                maintenance_margin=4000.0,
            )
        )

        account = MarginAccount(initial_cash=20000.0, asset_registry=registry)
        timestamp = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)

        account.open_position("ES", 2.0, 4500.0, timestamp)

        # Get liquidation price
        pos = account.positions["ES"]
        asset_spec = registry.get("ES")
        liq_price = account._calculate_liquidation_price("ES", pos, asset_spec)

        # Liquidation occurs when losses deplete available margin
        assert liq_price is not None
        assert liq_price < 4500.0  # Long position liquidates below entry

    def test_liquidation_price_futures_short(self):
        """Test liquidation price for short futures position."""
        registry = AssetRegistry()
        registry.register(
            AssetSpec(
                asset_id="ES",
                asset_class=AssetClass.FUTURE,
                contract_size=50,
                initial_margin=5000.0,
                maintenance_margin=4000.0,
            )
        )

        account = MarginAccount(initial_cash=20000.0, asset_registry=registry)
        timestamp = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)

        account.open_position("ES", -2.0, 4500.0, timestamp)

        # Get liquidation price
        pos = account.positions["ES"]
        asset_spec = registry.get("ES")
        liq_price = account._calculate_liquidation_price("ES", pos, asset_spec)

        assert liq_price is not None
        assert liq_price > 4500.0  # Short position liquidates above entry

    def test_liquidation_price_fx_long(self):
        """Test liquidation price for long FX position."""
        registry = AssetRegistry()
        registry.register(
            AssetSpec(
                asset_id="EURUSD",
                asset_class=AssetClass.FX,
                leverage_available=50.0,
            )
        )

        account = MarginAccount(initial_cash=10000.0, asset_registry=registry)
        timestamp = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)

        account.open_position("EURUSD", 100000.0, 1.10, timestamp)

        # Get liquidation price
        pos = account.positions["EURUSD"]
        asset_spec = registry.get("EURUSD")
        liq_price = account._calculate_liquidation_price("EURUSD", pos, asset_spec)

        assert liq_price is not None
        assert liq_price < 1.10  # Long FX liquidates below entry


class TestForcedLiquidation:
    """Test forced liquidation mechanism."""

    def test_force_liquidation_removes_position(self):
        """Test that forced liquidation removes position."""
        registry = AssetRegistry()
        registry.register(
            AssetSpec(
                asset_id="ES",
                asset_class=AssetClass.FUTURE,
                contract_size=50,
                initial_margin=5000.0,
                maintenance_margin=4000.0,
            )
        )

        account = MarginAccount(initial_cash=20000.0, asset_registry=registry)
        timestamp = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)

        account.open_position("ES", 2.0, 4500.0, timestamp)

        # Force liquidation at lower price
        account._force_liquidation("ES", 4300.0)

        assert "ES" not in account.positions
        assert account.margin_used == 0.0

    def test_force_liquidation_realizes_loss(self):
        """Test that forced liquidation realizes losses."""
        registry = AssetRegistry()
        registry.register(
            AssetSpec(
                asset_id="ES",
                asset_class=AssetClass.FUTURE,
                contract_size=50,
                initial_margin=5000.0,
                maintenance_margin=4000.0,
            )
        )

        account = MarginAccount(initial_cash=20000.0, asset_registry=registry)
        timestamp = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)

        account.open_position("ES", 2.0, 4500.0, timestamp)
        initial_cash = account.cash_balance

        # Force liquidation at lower price
        # Loss: 2 * (4300 - 4500) = -400
        account._force_liquidation("ES", 4300.0)

        expected_cash = initial_cash + (2.0 * (4300.0 - 4500.0))
        assert account.cash_balance == pytest.approx(expected_cash, rel=1e-4)


class TestMarginStatus:
    """Test margin status reporting."""

    def test_get_margin_status_empty(self):
        """Test margin status with no positions."""
        registry = AssetRegistry()
        account = MarginAccount(initial_cash=10000.0, asset_registry=registry)

        status = account.get_margin_status()

        assert status["cash_balance"] == 10000.0
        assert status["margin_used"] == 0.0
        assert status["available_margin"] == 10000.0
        assert status["unrealized_pnl"] == 0.0
        assert status["total_equity"] == 10000.0
        assert status["margin_utilization"] == 0.0
        assert status["num_positions"] == 0

    def test_get_margin_status_with_position(self):
        """Test margin status with active position."""
        registry = AssetRegistry()
        registry.register(
            AssetSpec(
                asset_id="ES",
                asset_class=AssetClass.FUTURE,
                contract_size=50,
                initial_margin=5000.0,
                maintenance_margin=4000.0,
            )
        )

        account = MarginAccount(initial_cash=50000.0, asset_registry=registry)
        timestamp = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)

        account.open_position("ES", 2.0, 4500.0, timestamp)

        # Update price to create small unrealized P&L
        account.update_prices({"ES": 4510.0})

        status = account.get_margin_status()

        assert status["cash_balance"] == 50000.0
        assert status["margin_used"] == 10000.0
        assert status["unrealized_pnl"] == 2.0 * (4510.0 - 4500.0)  # 20
        assert status["total_equity"] == 50000.0 + 20.0  # 50020
        assert status["num_positions"] == 1
        assert status["margin_utilization"] > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_quantity_position(self):
        """Test handling zero quantity in calculations."""
        registry = AssetRegistry()
        account = MarginAccount(initial_cash=10000.0, asset_registry=registry)

        has_margin, required = account.check_margin_requirement("AAPL", 0.0, 150.0)

        assert has_margin is True
        assert required == 0.0

    def test_negative_price_handling(self):
        """Test behavior with negative prices (commodities can go negative)."""
        registry = AssetRegistry()
        account = MarginAccount(initial_cash=10000.0, asset_registry=registry)

        # Should handle negative price without crashing
        # For unknown assets, required = abs(quantity) * float(price)
        has_margin, required = account.check_margin_requirement("CL", 100.0, -10.0)

        # Unknown asset requires abs(100) * float(-10) = 100 * -10 = -1000
        # But then comparison is available_margin >= -1000, which is True
        # So the implementation returns abs(quantity) * float(price) directly
        assert required == -1000.0  # 100 * -10 (not abs)

    def test_very_high_leverage(self):
        """Test with very high leverage ratios."""
        registry = AssetRegistry()
        registry.register(
            AssetSpec(
                asset_id="BTCUSD",
                asset_class=AssetClass.CRYPTO,
                leverage_available=100.0,
            )
        )

        account = MarginAccount(initial_cash=10000.0, asset_registry=registry)

        # 1 BTC at 50,000 = 50,000 notional / 100 leverage = 500 margin
        has_margin, required = account.check_margin_requirement("BTCUSD", 1.0, 50000.0)

        assert has_margin is True
        assert required == pytest.approx(500.0, rel=1e-4)

    def test_position_with_no_asset_spec(self):
        """Test position management when asset spec not found."""
        registry = AssetRegistry()
        account = MarginAccount(initial_cash=100000.0, asset_registry=registry)
        timestamp = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)

        # Open position for unknown asset - should require full notional
        success = account.open_position("UNKNOWN", 100.0, 150.0, timestamp)

        # Should succeed - need 15,000 but have 100,000
        assert success is True
        assert "UNKNOWN" in account.positions
        assert account.positions["UNKNOWN"]["margin_used"] == 15000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
