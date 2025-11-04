"""Tests for slippage models."""

from datetime import datetime

import pytest

from qengine.core.event import MarketDataType, MarketEvent
from qengine.core.types import OrderSide, OrderType
from qengine.execution.broker import SimulationBroker
from qengine.execution.order import Order
from qengine.execution.slippage import (
    AssetClassSlippage,
    FixedSlippage,
    LinearImpactSlippage,
    NoSlippage,
    PercentageSlippage,
    SquareRootImpactSlippage,
    VolumeShareSlippage,
)


class TestNoSlippage:
    """Test no slippage model."""

    def test_no_slippage_fill_price(self):
        """Test that fill price equals market price."""
        model = NoSlippage()

        buy_order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )

        fill_price = model.calculate_fill_price(buy_order, 100.0)
        assert fill_price == 100.0

        sell_order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            quantity=100,
        )

        fill_price = model.calculate_fill_price(sell_order, 100.0)
        assert fill_price == 100.0

    def test_no_slippage_cost(self):
        """Test that slippage cost is zero."""
        model = NoSlippage()

        order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )

        cost = model.calculate_slippage_cost(order, 100, 100.0, 100.0)
        assert cost == 0.0


class TestFixedSlippage:
    """Test fixed spread slippage model."""

    def test_fixed_slippage_default(self):
        """Test fixed slippage with default spread."""
        model = FixedSlippage()  # Default 0.01 spread

        buy_order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )

        fill_price = model.calculate_fill_price(buy_order, 100.0)
        assert fill_price == 100.005  # 100 + 0.01/2

        sell_order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            quantity=100,
        )

        fill_price = model.calculate_fill_price(sell_order, 100.0)
        assert fill_price == 99.995  # 100 - 0.01/2

    def test_fixed_slippage_custom_spread(self):
        """Test fixed slippage with custom spread."""
        model = FixedSlippage(spread=0.02)

        order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )

        fill_price = model.calculate_fill_price(order, 100.0)
        assert fill_price == 100.01  # 100 + 0.02/2

    def test_fixed_slippage_cost(self):
        """Test slippage cost calculation."""
        model = FixedSlippage(spread=0.01)

        order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )

        cost = model.calculate_slippage_cost(order, 100, 100.0, 100.005)
        assert cost == pytest.approx(0.5, rel=1e-6)  # 0.005 * 100

    def test_invalid_spread(self):
        """Test that negative spread raises error."""
        with pytest.raises(ValueError, match="Spread must be non-negative"):
            FixedSlippage(spread=-0.01)


class TestPercentageSlippage:
    """Test percentage-based slippage model."""

    def test_percentage_slippage_default(self):
        """Test percentage slippage with defaults."""
        model = PercentageSlippage()  # Default 0.1%

        buy_order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )

        fill_price = model.calculate_fill_price(buy_order, 100.0)
        assert fill_price == pytest.approx(100.1, rel=1e-6)  # 100 * 1.001

        sell_order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            quantity=100,
        )

        fill_price = model.calculate_fill_price(sell_order, 100.0)
        assert fill_price == pytest.approx(99.9, rel=1e-6)  # 100 * 0.999

    def test_percentage_slippage_custom(self):
        """Test percentage slippage with custom percentage."""
        model = PercentageSlippage(slippage_pct=0.002)  # 0.2%

        order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )

        fill_price = model.calculate_fill_price(order, 100.0)
        assert fill_price == pytest.approx(100.2, rel=1e-6)

    def test_minimum_slippage(self):
        """Test minimum slippage enforcement."""
        model = PercentageSlippage(slippage_pct=0.001, min_slippage=0.05)

        order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )

        # For a $1 stock, 0.1% would be $0.001, but min is $0.05
        fill_price = model.calculate_fill_price(order, 1.0)
        assert fill_price == 1.05  # 1 + 0.05 (minimum)

    def test_invalid_percentage(self):
        """Test that negative percentage raises error."""
        with pytest.raises(ValueError, match="Slippage percentage must be non-negative"):
            PercentageSlippage(slippage_pct=-0.001)


class TestLinearImpactSlippage:
    """Test linear market impact slippage model."""

    def test_linear_impact_small_order(self):
        """Test linear impact for small order."""
        model = LinearImpactSlippage(base_slippage=0.0001, impact_coefficient=0.00001)

        order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )

        fill_price = model.calculate_fill_price(order, 100.0)
        # Base: 0.0001, Impact: 0.00001 * 100 = 0.001, Total: 0.0011
        expected = 100.0 * (1 + 0.0011)
        assert fill_price == pytest.approx(expected, rel=1e-6)

    def test_linear_impact_large_order(self):
        """Test linear impact for large order."""
        model = LinearImpactSlippage(base_slippage=0.0001, impact_coefficient=0.00001)

        order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=10000,
        )

        fill_price = model.calculate_fill_price(order, 100.0)
        # Base: 0.0001, Impact: 0.00001 * 10000 = 0.1, Total: 0.1001
        expected = 100.0 * (1 + 0.1001)
        assert fill_price == pytest.approx(expected, rel=1e-6)

    def test_linear_impact_sell_order(self):
        """Test linear impact for sell order."""
        model = LinearImpactSlippage()

        order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            quantity=100,
        )

        fill_price = model.calculate_fill_price(order, 100.0)
        assert fill_price < 100.0  # Sell at worse price


class TestSquareRootImpactSlippage:
    """Test square root market impact model."""

    def test_sqrt_impact_small_order(self):
        """Test square root impact for small order."""
        model = SquareRootImpactSlippage(temporary_impact=0.1, permanent_impact=0.05)

        order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )

        fill_price = model.calculate_fill_price(order, 100.0)
        assert fill_price > 100.0  # Should have positive slippage for buy

    def test_sqrt_impact_large_order(self):
        """Test square root impact for large order."""
        model = SquareRootImpactSlippage(temporary_impact=0.1, permanent_impact=0.05)

        small_order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )

        large_order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=10000,
        )

        small_fill = model.calculate_fill_price(small_order, 100.0)
        large_fill = model.calculate_fill_price(large_order, 100.0)

        # Larger order should have more slippage
        assert large_fill > small_fill

        # But not proportionally more (square root relationship)
        small_impact = small_fill - 100.0
        large_impact = large_fill - 100.0

        # Impact ratio should be sqrt(10000/100) = 10, not 100
        impact_ratio = large_impact / small_impact
        assert impact_ratio < 20  # Much less than linear


class TestVolumeShareSlippage:
    """Test volume-based slippage model."""

    def test_volume_slippage_with_volume(self):
        """Test volume slippage with volume data."""
        model = VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)
        model.set_daily_volume(100000)

        order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=1000,  # 1% of daily volume
        )

        fill_price = model.calculate_fill_price(order, 100.0)
        # Volume share = 0.01, impact = 0.01^2 * 0.1 = 0.00001
        expected = 100.0 * (1 + 0.00001)
        assert fill_price == pytest.approx(expected, rel=1e-6)

    def test_volume_slippage_no_volume(self):
        """Test volume slippage without volume data."""
        model = VolumeShareSlippage()

        order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )

        fill_price = model.calculate_fill_price(order, 100.0)
        # Should use minimal slippage
        assert fill_price == pytest.approx(100.01, rel=1e-3)

    def test_volume_limit(self):
        """Test volume limit enforcement."""
        model = VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)
        model.set_daily_volume(1000)

        order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=500,  # 50% of daily volume, but limited to 2.5%
        )

        fill_price = model.calculate_fill_price(order, 100.0)
        # Volume share capped at 0.025, impact = 0.025^2 * 0.1 = 0.0000625
        expected = 100.0 * (1 + 0.0000625)
        assert fill_price == pytest.approx(expected, rel=1e-6)

    def test_invalid_volume_limit(self):
        """Test that invalid volume limit raises error."""
        with pytest.raises(ValueError, match="Volume limit must be between 0 and 1"):
            VolumeShareSlippage(volume_limit=1.5)


class TestAssetClassSlippage:
    """Test asset class specific slippage model."""

    def test_equity_slippage(self):
        """Test equity slippage."""
        model = AssetClassSlippage()

        order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
            metadata={"asset_class": "equity"},
        )

        fill_price = model.calculate_fill_price(order, 100.0)
        assert fill_price == pytest.approx(100.01, rel=1e-3)  # 0.01% slippage

    def test_crypto_slippage(self):
        """Test crypto slippage."""
        model = AssetClassSlippage()

        order = Order(
            asset_id="BTC",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=1,
            metadata={"asset_class": "crypto"},
        )

        fill_price = model.calculate_fill_price(order, 50000.0)
        assert fill_price == pytest.approx(50050.0, rel=1e-3)  # 0.1% slippage

    def test_fx_slippage(self):
        """Test forex slippage."""
        model = AssetClassSlippage()

        order = Order(
            asset_id="EURUSD",
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            quantity=100000,
            metadata={"asset_class": "fx"},
        )

        fill_price = model.calculate_fill_price(order, 1.1000)
        assert fill_price == pytest.approx(1.099945, rel=1e-6)  # 0.005% slippage

    def test_default_asset_class(self):
        """Test default asset class handling."""
        model = AssetClassSlippage()

        order = Order(
            asset_id="UNKNOWN",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
            # No asset_class in metadata
        )

        fill_price = model.calculate_fill_price(order, 100.0)
        # Should use equity default
        assert fill_price == pytest.approx(100.01, rel=1e-3)


class TestBrokerIntegration:
    """Test slippage models with broker integration."""

    def test_broker_with_no_slippage(self):
        """Test broker with NoSlippage model."""
        broker = SimulationBroker(
            initial_cash=100000.0,
            slippage_model=NoSlippage(),
            execution_delay=False,
        )

        order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )

        broker.submit_order(order)

        market_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 10, 0),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            close=100.0,
        )

        fills = broker.on_market_event(market_event)

        assert len(fills) == 1
        assert fills[0].fill_price == 100.0  # No slippage
        assert fills[0].slippage == 0.0

    def test_broker_with_fixed_slippage(self):
        """Test broker with FixedSlippage model."""
        broker = SimulationBroker(
            initial_cash=100000.0,
            slippage_model=FixedSlippage(spread=0.02),
            execution_delay=False,
        )

        order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )

        broker.submit_order(order)

        market_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 10, 0),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            close=100.0,
        )

        fills = broker.on_market_event(market_event)

        assert len(fills) == 1
        assert fills[0].fill_price == 100.01  # Fixed spread/2
        assert fills[0].slippage == pytest.approx(1.0, rel=1e-3)  # 0.01 * 100

    def test_broker_with_percentage_slippage(self):
        """Test broker with PercentageSlippage model."""
        broker = SimulationBroker(
            initial_cash=100000.0,
            slippage_model=PercentageSlippage(slippage_pct=0.002),  # 0.2%
            execution_delay=False,
        )

        order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            quantity=100,
        )

        broker.submit_order(order)

        # Need to have some shares to sell
        broker._positions["AAPL"] = 100

        market_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 10, 0),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            close=100.0,
        )

        fills = broker.on_market_event(market_event)

        assert len(fills) == 1
        assert fills[0].fill_price == pytest.approx(99.8, rel=1e-6)  # 100 * (1 - 0.002)
        assert fills[0].slippage == pytest.approx(20.0, rel=1e-3)  # 0.2 * 100
