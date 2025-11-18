"""Unit tests for enhanced slippage models (TASK-INT-039).

Tests SpreadAwareSlippage, VolumeAwareSlippage, and OrderTypeDependentSlippage models.
"""

from datetime import datetime, timezone

import pytest

from ml4t.backtest.core.event import MarketEvent
from ml4t.backtest.core.types import OrderSide, OrderType
from ml4t.backtest.execution.order import Order
from ml4t.backtest.execution.slippage import (
    OrderTypeDependentSlippage,
    SpreadAwareSlippage,
    VolumeAwareSlippage,
)


@pytest.fixture
def asset_id() -> str:
    """Test asset ID."""
    return "AAPL"


@pytest.fixture
def timestamp() -> datetime:
    """Test timestamp."""
    return datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)


@pytest.fixture
def market_event_with_spread(asset_id: str, timestamp: datetime) -> MarketEvent:
    """Create MarketEvent with bid/ask spread."""
    return MarketEvent(
        timestamp=timestamp,
        asset_id=asset_id,
        data_type="bar",
        price=100.0,
        open=99.5,
        high=100.5,
        low=99.0,
        close=100.0,
        volume=1_000_000,
        bid_price=99.98,  # Bid
        ask_price=100.02,  # Ask (spread = 0.04)
    )


@pytest.fixture
def market_event_with_volume(asset_id: str, timestamp: datetime) -> MarketEvent:
    """Create MarketEvent with volume data."""
    return MarketEvent(
        timestamp=timestamp,
        asset_id=asset_id,
        data_type="bar",
        price=100.0,
        open=99.5,
        high=100.5,
        low=99.0,
        close=100.0,
        volume=100_000,  # 100K shares volume
    )


@pytest.fixture
def market_event_minimal(asset_id: str, timestamp: datetime) -> MarketEvent:
    """Create minimal MarketEvent without bid/ask or volume."""
    return MarketEvent(
        timestamp=timestamp,
        asset_id=asset_id,
        data_type="trade",
        price=100.0,
        close=100.0,
    )


class TestSpreadAwareSlippage:
    """Test SpreadAwareSlippage model."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        model = SpreadAwareSlippage()
        assert model.spread_factor == 0.5
        assert model.fallback_slippage_pct == 0.001

    def test_initialization_custom_params(self):
        """Test custom parameter initialization."""
        model = SpreadAwareSlippage(spread_factor=0.3, fallback_slippage_pct=0.0005)
        assert model.spread_factor == 0.3
        assert model.fallback_slippage_pct == 0.0005

    def test_initialization_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError, match="Spread factor must be non-negative"):
            SpreadAwareSlippage(spread_factor=-0.1)

        with pytest.raises(ValueError, match="Fallback slippage percentage must be non-negative"):
            SpreadAwareSlippage(fallback_slippage_pct=-0.001)

    def test_buy_with_spread(self, asset_id: str, market_event_with_spread: MarketEvent):
        """Test buy order with bid/ask spread available."""
        model = SpreadAwareSlippage(spread_factor=0.5)
        order = Order(asset_id=asset_id, side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)

        fill_price = model.calculate_fill_price(order, 100.0, market_event_with_spread)

        # mid = (99.98 + 100.02) / 2 = 100.00
        # spread = 100.02 - 99.98 = 0.04
        # half_spread = 0.02
        # slippage = 0.5 × 0.02 = 0.01
        # Buy: mid + slippage = 100.00 + 0.01 = 100.01
        assert fill_price == pytest.approx(100.01, abs=1e-6)

    def test_sell_with_spread(self, asset_id: str, market_event_with_spread: MarketEvent):
        """Test sell order with bid/ask spread available."""
        model = SpreadAwareSlippage(spread_factor=0.5)
        order = Order(
            asset_id=asset_id, side=OrderSide.SELL, quantity=100, order_type=OrderType.MARKET
        )

        fill_price = model.calculate_fill_price(order, 100.0, market_event_with_spread)

        # Sell: mid - slippage = 100.00 - 0.01 = 99.99
        assert fill_price == pytest.approx(99.99, abs=1e-6)

    def test_fallback_no_market_event(self, asset_id: str):
        """Test fallback when no MarketEvent provided."""
        model = SpreadAwareSlippage(fallback_slippage_pct=0.001)
        order = Order(asset_id=asset_id, side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)

        fill_price = model.calculate_fill_price(order, 100.0, None)

        # Fallback: 100.0 × 0.001 = 0.1, buy: 100.0 + 0.1 = 100.1
        assert fill_price == pytest.approx(100.1, abs=1e-6)

    def test_fallback_no_bid_ask(self, asset_id: str, market_event_minimal: MarketEvent):
        """Test fallback when bid/ask not available in MarketEvent."""
        model = SpreadAwareSlippage(fallback_slippage_pct=0.001)
        order = Order(asset_id=asset_id, side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)

        fill_price = model.calculate_fill_price(order, 100.0, market_event_minimal)

        # Fallback: 100.0 × 0.001 = 0.1, buy: 100.0 + 0.1 = 100.1
        assert fill_price == pytest.approx(100.1, abs=1e-6)

    def test_different_spread_factors(self, asset_id: str, market_event_with_spread: MarketEvent):
        """Test different spread factors."""
        order = Order(asset_id=asset_id, side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)

        # k = 0.0 (fill at mid)
        model_0 = SpreadAwareSlippage(spread_factor=0.0)
        fill_0 = model_0.calculate_fill_price(order, 100.0, market_event_with_spread)
        assert fill_0 == pytest.approx(100.0, abs=1e-6)  # Mid

        # k = 1.0 (fill at ask for buy)
        model_1 = SpreadAwareSlippage(spread_factor=1.0)
        fill_1 = model_1.calculate_fill_price(order, 100.0, market_event_with_spread)
        assert fill_1 == pytest.approx(100.02, abs=1e-6)  # Mid + full half_spread

    def test_slippage_cost(self, asset_id: str):
        """Test slippage cost calculation."""
        model = SpreadAwareSlippage()
        order = Order(asset_id=asset_id, side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)

        cost = model.calculate_slippage_cost(order, 100, 100.0, 100.1)

        # |100.1 - 100.0| × 100 = 10.0
        assert cost == pytest.approx(10.0, abs=1e-6)


class TestVolumeAwareSlippage:
    """Test VolumeAwareSlippage model."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        model = VolumeAwareSlippage()
        assert model.base_slippage_pct == 0.0001
        assert model.linear_impact_coeff == 0.01
        assert model.sqrt_impact_coeff == 0.0
        assert model.fallback_slippage_pct == 0.001
        assert model.max_participation_rate == 0.1

    def test_initialization_custom_params(self):
        """Test custom parameter initialization."""
        model = VolumeAwareSlippage(
            base_slippage_pct=0.0002,
            linear_impact_coeff=0.02,
            sqrt_impact_coeff=0.01,
            fallback_slippage_pct=0.0005,
            max_participation_rate=0.05,
        )
        assert model.base_slippage_pct == 0.0002
        assert model.linear_impact_coeff == 0.02
        assert model.sqrt_impact_coeff == 0.01
        assert model.fallback_slippage_pct == 0.0005
        assert model.max_participation_rate == 0.05

    def test_initialization_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError, match="Base slippage must be non-negative"):
            VolumeAwareSlippage(base_slippage_pct=-0.1)

        with pytest.raises(ValueError, match="Linear impact coefficient must be non-negative"):
            VolumeAwareSlippage(linear_impact_coeff=-0.01)

        with pytest.raises(ValueError, match="Max participation rate must be in"):
            VolumeAwareSlippage(max_participation_rate=0.0)

        with pytest.raises(ValueError, match="Max participation rate must be in"):
            VolumeAwareSlippage(max_participation_rate=1.5)

    def test_linear_impact_only(self, asset_id: str, market_event_with_volume: MarketEvent):
        """Test linear impact model (no sqrt component)."""
        model = VolumeAwareSlippage(
            base_slippage_pct=0.0001, linear_impact_coeff=0.01, sqrt_impact_coeff=0.0
        )
        order = Order(asset_id=asset_id, side=OrderSide.BUY, quantity=1000, order_type=OrderType.MARKET)

        fill_price = model.calculate_fill_price(order, 100.0, market_event_with_volume)

        # participation_rate = 1000 / 100000 = 0.01 = 1%
        # impact_pct = 0.0001 + 0.01 × 0.01 = 0.0001 + 0.0001 = 0.0002 = 0.02%
        # slippage_amount = 100.0 × 0.0002 = 0.02
        # Buy: 100.0 + 0.02 = 100.02
        assert fill_price == pytest.approx(100.02, abs=1e-6)

    def test_sqrt_impact_only(self, asset_id: str, market_event_with_volume: MarketEvent):
        """Test square-root impact model (no linear component)."""
        import math

        model = VolumeAwareSlippage(
            base_slippage_pct=0.0001, linear_impact_coeff=0.0, sqrt_impact_coeff=0.01
        )
        order = Order(asset_id=asset_id, side=OrderSide.BUY, quantity=1000, order_type=OrderType.MARKET)

        fill_price = model.calculate_fill_price(order, 100.0, market_event_with_volume)

        # participation_rate = 1000 / 100000 = 0.01
        # impact_pct = 0.0001 + 0.01 × sqrt(0.01) = 0.0001 + 0.01 × 0.1 = 0.0001 + 0.001 = 0.0011
        # slippage_amount = 100.0 × 0.0011 = 0.11
        # Buy: 100.0 + 0.11 = 100.11
        expected_impact = 0.0001 + 0.01 * math.sqrt(0.01)
        expected_fill = 100.0 + 100.0 * expected_impact
        assert fill_price == pytest.approx(expected_fill, abs=1e-6)

    def test_max_participation_capped(self, asset_id: str, market_event_with_volume: MarketEvent):
        """Test participation rate is capped at max."""
        model = VolumeAwareSlippage(
            base_slippage_pct=0.0001,
            linear_impact_coeff=0.01,
            sqrt_impact_coeff=0.0,
            max_participation_rate=0.05,  # Cap at 5%
        )
        # Order size 20% of volume (will be capped)
        order = Order(
            asset_id=asset_id, side=OrderSide.BUY, quantity=20_000, order_type=OrderType.MARKET
        )

        fill_price = model.calculate_fill_price(order, 100.0, market_event_with_volume)

        # participation_rate = min(20000 / 100000, 0.05) = min(0.2, 0.05) = 0.05
        # impact_pct = 0.0001 + 0.01 × 0.05 = 0.0001 + 0.0005 = 0.0006
        # slippage_amount = 100.0 × 0.0006 = 0.06
        # Buy: 100.0 + 0.06 = 100.06
        assert fill_price == pytest.approx(100.06, abs=1e-6)

    def test_sell_order(self, asset_id: str, market_event_with_volume: MarketEvent):
        """Test sell order pays slippage in opposite direction."""
        model = VolumeAwareSlippage(
            base_slippage_pct=0.0001, linear_impact_coeff=0.01, sqrt_impact_coeff=0.0
        )
        order = Order(
            asset_id=asset_id, side=OrderSide.SELL, quantity=1000, order_type=OrderType.MARKET
        )

        fill_price = model.calculate_fill_price(order, 100.0, market_event_with_volume)

        # Sell: 100.0 - 0.02 = 99.98
        assert fill_price == pytest.approx(99.98, abs=1e-6)

    def test_fallback_no_market_event(self, asset_id: str):
        """Test fallback when no MarketEvent provided."""
        model = VolumeAwareSlippage(fallback_slippage_pct=0.001)
        order = Order(asset_id=asset_id, side=OrderSide.BUY, quantity=1000, order_type=OrderType.MARKET)

        fill_price = model.calculate_fill_price(order, 100.0, None)

        # Fallback: 100.0 × 0.001 = 0.1, buy: 100.0 + 0.1 = 100.1
        assert fill_price == pytest.approx(100.1, abs=1e-6)

    def test_fallback_no_volume(self, asset_id: str, market_event_minimal: MarketEvent):
        """Test fallback when volume not available in MarketEvent."""
        model = VolumeAwareSlippage(fallback_slippage_pct=0.001)
        order = Order(asset_id=asset_id, side=OrderSide.BUY, quantity=1000, order_type=OrderType.MARKET)

        fill_price = model.calculate_fill_price(order, 100.0, market_event_minimal)

        # Fallback: 100.0 × 0.001 = 0.1, buy: 100.0 + 0.1 = 100.1
        assert fill_price == pytest.approx(100.1, abs=1e-6)

    def test_slippage_cost(self, asset_id: str):
        """Test slippage cost calculation."""
        model = VolumeAwareSlippage()
        order = Order(asset_id=asset_id, side=OrderSide.BUY, quantity=1000, order_type=OrderType.MARKET)

        cost = model.calculate_slippage_cost(order, 1000, 100.0, 100.02)

        # |100.02 - 100.0| × 1000 = 20.0
        assert cost == pytest.approx(20.0, abs=1e-6)


class TestOrderTypeDependentSlippage:
    """Test OrderTypeDependentSlippage model."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        model = OrderTypeDependentSlippage()
        assert model.slippage_rates[OrderType.MARKET] == 0.001
        assert model.slippage_rates[OrderType.LIMIT] == 0.0001
        assert model.slippage_rates[OrderType.STOP] == 0.0005
        assert model.slippage_rates[OrderType.STOP_LIMIT] == 0.0005
        assert model.slippage_rates[OrderType.TRAILING_STOP] == 0.0005
        assert model.default_slippage_pct == 0.001

    def test_initialization_custom_params(self):
        """Test custom parameter initialization."""
        model = OrderTypeDependentSlippage(
            market_slippage_pct=0.002,
            limit_slippage_pct=0.00005,
            stop_slippage_pct=0.001,
            default_slippage_pct=0.0005,
        )
        assert model.slippage_rates[OrderType.MARKET] == 0.002
        assert model.slippage_rates[OrderType.LIMIT] == 0.00005
        assert model.slippage_rates[OrderType.STOP] == 0.001
        assert model.default_slippage_pct == 0.0005

    def test_initialization_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError, match="Market slippage must be non-negative"):
            OrderTypeDependentSlippage(market_slippage_pct=-0.1)

        with pytest.raises(ValueError, match="Limit slippage must be non-negative"):
            OrderTypeDependentSlippage(limit_slippage_pct=-0.0001)

        with pytest.raises(ValueError, match="Stop slippage must be non-negative"):
            OrderTypeDependentSlippage(stop_slippage_pct=-0.0005)

        with pytest.raises(ValueError, match="Default slippage must be non-negative"):
            OrderTypeDependentSlippage(default_slippage_pct=-0.001)

    def test_market_order_buy(self, asset_id: str):
        """Test MARKET order pays highest slippage."""
        model = OrderTypeDependentSlippage(market_slippage_pct=0.001)
        order = Order(asset_id=asset_id, side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)

        fill_price = model.calculate_fill_price(order, 100.0, None)

        # Market: 100.0 × 0.001 = 0.1, buy: 100.0 + 0.1 = 100.1
        assert fill_price == pytest.approx(100.1, abs=1e-6)

    def test_market_order_sell(self, asset_id: str):
        """Test MARKET sell order."""
        model = OrderTypeDependentSlippage(market_slippage_pct=0.001)
        order = Order(
            asset_id=asset_id, side=OrderSide.SELL, quantity=100, order_type=OrderType.MARKET
        )

        fill_price = model.calculate_fill_price(order, 100.0, None)

        # Sell: 100.0 - 0.1 = 99.9
        assert fill_price == pytest.approx(99.9, abs=1e-6)

    def test_limit_order_buy(self, asset_id: str):
        """Test LIMIT order pays lowest slippage."""
        model = OrderTypeDependentSlippage(limit_slippage_pct=0.0001)
        order = Order(
            asset_id=asset_id,
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=99.0,  # Required for LIMIT orders
        )

        fill_price = model.calculate_fill_price(order, 100.0, None)

        # Limit: 100.0 × 0.0001 = 0.01, buy: 100.0 + 0.01 = 100.01
        assert fill_price == pytest.approx(100.01, abs=1e-6)

    def test_stop_order_buy(self, asset_id: str):
        """Test STOP order pays medium slippage."""
        model = OrderTypeDependentSlippage(stop_slippage_pct=0.0005)
        order = Order(
            asset_id=asset_id,
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.STOP,
            stop_price=101.0,  # Required for STOP orders
        )

        fill_price = model.calculate_fill_price(order, 100.0, None)

        # Stop: 100.0 × 0.0005 = 0.05, buy: 100.0 + 0.05 = 100.05
        assert fill_price == pytest.approx(100.05, abs=1e-6)

    def test_stop_limit_order(self, asset_id: str):
        """Test STOP_LIMIT uses stop slippage."""
        model = OrderTypeDependentSlippage(stop_slippage_pct=0.0005)
        order = Order(
            asset_id=asset_id,
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.STOP_LIMIT,
            stop_price=101.0,  # Required for STOP_LIMIT
            limit_price=101.5,  # Required for STOP_LIMIT
        )

        fill_price = model.calculate_fill_price(order, 100.0, None)

        # Stop_Limit: same as STOP
        assert fill_price == pytest.approx(100.05, abs=1e-6)

    def test_trailing_stop_order(self, asset_id: str):
        """Test TRAILING_STOP uses stop slippage."""
        model = OrderTypeDependentSlippage(stop_slippage_pct=0.0005)
        order = Order(
            asset_id=asset_id,
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.TRAILING_STOP,
            trail_amount=1.0,  # Required for TRAILING_STOP
        )

        fill_price = model.calculate_fill_price(order, 100.0, None)

        # Trailing_Stop: same as STOP
        assert fill_price == pytest.approx(100.05, abs=1e-6)

    def test_bracket_order_uses_default(self, asset_id: str):
        """Test BRACKET order uses default slippage."""
        model = OrderTypeDependentSlippage(default_slippage_pct=0.001)
        order = Order(
            asset_id=asset_id,
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.BRACKET,
            profit_target=105.0,  # Required for BRACKET orders
            stop_loss=95.0,  # Required for BRACKET orders
        )

        fill_price = model.calculate_fill_price(order, 100.0, None)

        # Bracket: default 0.001
        assert fill_price == pytest.approx(100.1, abs=1e-6)

    def test_slippage_cost(self, asset_id: str):
        """Test slippage cost calculation."""
        model = OrderTypeDependentSlippage()
        order = Order(asset_id=asset_id, side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)

        cost = model.calculate_slippage_cost(order, 100, 100.0, 100.1)

        # |100.1 - 100.0| × 100 = 10.0
        assert cost == pytest.approx(10.0, abs=1e-6)


class TestIntegrationComparison:
    """Integration tests comparing different slippage models."""

    def test_spread_vs_percentage_comparison(
        self, asset_id: str, market_event_with_spread: MarketEvent
    ):
        """Compare SpreadAwareSlippage with and without spread data."""
        spread_model = SpreadAwareSlippage(spread_factor=0.5, fallback_slippage_pct=0.001)
        order = Order(asset_id=asset_id, side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)

        # With spread data
        fill_with_spread = spread_model.calculate_fill_price(order, 100.0, market_event_with_spread)

        # Without spread data (fallback)
        fill_fallback = spread_model.calculate_fill_price(order, 100.0, None)

        # With spread should be tighter (less slippage)
        # With spread: 100.01 vs fallback: 100.1
        assert fill_with_spread < fill_fallback
        assert fill_with_spread == pytest.approx(100.01, abs=1e-6)
        assert fill_fallback == pytest.approx(100.1, abs=1e-6)

    def test_volume_scaling(self, asset_id: str, market_event_with_volume: MarketEvent):
        """Test volume impact scales with order size."""
        model = VolumeAwareSlippage(
            base_slippage_pct=0.0001, linear_impact_coeff=0.01, sqrt_impact_coeff=0.0
        )

        # Small order (1% of volume)
        small_order = Order(
            asset_id=asset_id, side=OrderSide.BUY, quantity=1000, order_type=OrderType.MARKET
        )
        small_fill = model.calculate_fill_price(small_order, 100.0, market_event_with_volume)

        # Large order (10% of volume)
        large_order = Order(
            asset_id=asset_id, side=OrderSide.BUY, quantity=10_000, order_type=OrderType.MARKET
        )
        large_fill = model.calculate_fill_price(large_order, 100.0, market_event_with_volume)

        # Large order should have more slippage
        assert large_fill > small_fill
        assert small_fill == pytest.approx(100.02, abs=1e-6)  # 1% participation
        assert large_fill == pytest.approx(100.11, abs=1e-6)  # 10% participation

    def test_order_type_comparison(self, asset_id: str):
        """Compare slippage across different order types."""
        model = OrderTypeDependentSlippage(
            market_slippage_pct=0.001, limit_slippage_pct=0.0001, stop_slippage_pct=0.0005
        )

        market_order = Order(
            asset_id=asset_id, side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET
        )
        limit_order = Order(
            asset_id=asset_id,
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=99.0,
        )
        stop_order = Order(
            asset_id=asset_id,
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.STOP,
            stop_price=101.0,
        )

        market_fill = model.calculate_fill_price(market_order, 100.0, None)
        limit_fill = model.calculate_fill_price(limit_order, 100.0, None)
        stop_fill = model.calculate_fill_price(stop_order, 100.0, None)

        # Market > Stop > Limit (slippage hierarchy)
        assert market_fill > stop_fill > limit_fill
        assert market_fill == pytest.approx(100.1, abs=1e-6)
        assert stop_fill == pytest.approx(100.05, abs=1e-6)
        assert limit_fill == pytest.approx(100.01, abs=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
