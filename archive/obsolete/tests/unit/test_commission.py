"""Tests for commission models."""

from datetime import datetime

import pytest

from ml4t.backtest.execution.commission import (
    AssetClassCommission,
    FlatCommission,
    InteractiveBrokersCommission,
    MakerTakerCommission,
    NoCommission,
    PercentageCommission,
    PerShareCommission,
    TieredCommission,
)
from ml4t.backtest.execution.order import Order, OrderSide, OrderType


@pytest.fixture
def sample_order():
    """Create a sample order for testing."""
    return Order(
        order_id="TEST001",
        asset_id="AAPL",

        quantity=100,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        created_time=datetime.now(),
    )


class TestNoCommission:
    """Test NoCommission model."""

    def test_zero_commission(self, sample_order):
        """Test that no commission returns zero."""
        model = NoCommission()
        commission = model.calculate(sample_order, 100, 150.0)
        assert commission == 0.0

    def test_repr(self):
        """Test string representation."""
        model = NoCommission()
        assert repr(model) == "NoCommission()"


class TestFlatCommission:
    """Test FlatCommission model."""

    def test_default_commission(self, sample_order):
        """Test default flat commission."""
        model = FlatCommission()
        commission = model.calculate(sample_order, 100, 150.0)
        assert commission == 1.0

    def test_custom_commission(self, sample_order):
        """Test custom flat commission."""
        model = FlatCommission(commission=5.0)
        commission = model.calculate(sample_order, 100, 150.0)
        assert commission == 5.0

    def test_negative_commission_raises(self):
        """Test that negative commission raises error."""
        with pytest.raises(ValueError, match="Commission cannot be negative"):
            FlatCommission(commission=-1.0)

    def test_repr(self):
        """Test string representation."""
        model = FlatCommission(commission=2.5)
        assert repr(model) == "FlatCommission(commission=2.5)"


class TestPercentageCommission:
    """Test PercentageCommission model."""

    def test_default_rate(self, sample_order):
        """Test default percentage rate."""
        model = PercentageCommission()
        commission = model.calculate(sample_order, 100, 150.0)
        assert commission == 15.0  # 100 * 150 * 0.001

    def test_custom_rate(self, sample_order):
        """Test custom percentage rate."""
        model = PercentageCommission(rate=0.002)
        commission = model.calculate(sample_order, 100, 150.0)
        assert commission == 30.0  # 100 * 150 * 0.002

    def test_negative_rate_raises(self):
        """Test that negative rate raises error."""
        with pytest.raises(ValueError, match="Commission rate cannot be negative"):
            PercentageCommission(rate=-0.001)

    def test_excessive_rate_raises(self):
        """Test that excessive rate raises error."""
        with pytest.raises(ValueError, match="Commission rate too high"):
            PercentageCommission(rate=0.15)

    def test_repr(self):
        """Test string representation."""
        model = PercentageCommission(rate=0.0015)
        assert repr(model) == "PercentageCommission(rate=0.0015)"


class TestPerShareCommission:
    """Test PerShareCommission model."""

    def test_default_rate(self, sample_order):
        """Test default per-share commission."""
        model = PerShareCommission()
        commission = model.calculate(sample_order, 100, 150.0)
        assert commission == 0.5  # 100 * 0.005

    def test_custom_rate(self, sample_order):
        """Test custom per-share commission."""
        model = PerShareCommission(commission_per_share=0.01)
        commission = model.calculate(sample_order, 200, 150.0)
        assert commission == 2.0  # 200 * 0.01

    def test_negative_rate_raises(self):
        """Test that negative rate raises error."""
        with pytest.raises(ValueError, match="Per-share commission cannot be negative"):
            PerShareCommission(commission_per_share=-0.005)

    def test_repr(self):
        """Test string representation."""
        model = PerShareCommission(commission_per_share=0.007)
        assert repr(model) == "PerShareCommission(commission_per_share=0.007)"


class TestTieredCommission:
    """Test TieredCommission model."""

    def test_default_tiers(self, sample_order):
        """Test default tier structure."""
        model = TieredCommission()

        # Small trade (<$10k)
        commission = model.calculate(sample_order, 50, 100.0)  # $5k notional
        assert commission == 5.0  # 5000 * 0.001

        # Medium trade ($10k-$50k)
        commission = model.calculate(sample_order, 200, 100.0)  # $20k notional
        assert commission == 16.0  # 20000 * 0.0008

        # Large trade ($50k-$100k)
        commission = model.calculate(sample_order, 600, 100.0)  # $60k notional
        assert commission == 30.0  # 60000 * 0.0005

        # Very large trade (>$100k)
        commission = model.calculate(sample_order, 2000, 100.0)  # $200k notional
        assert pytest.approx(commission) == 60.0  # 200000 * 0.0003

    def test_custom_tiers(self, sample_order):
        """Test custom tier structure."""
        tiers = [
            (5000, 0.002),
            (20000, 0.001),
            (float("inf"), 0.0005),
        ]
        model = TieredCommission(tiers=tiers, minimum=0.5)

        # Below minimum
        commission = model.calculate(sample_order, 1, 100.0)  # $100 notional
        assert commission == 0.5  # Minimum applies

        # First tier
        commission = model.calculate(sample_order, 30, 100.0)  # $3k notional
        assert commission == 6.0  # 3000 * 0.002

    def test_invalid_tiers_raises(self):
        """Test that invalid tiers raise errors."""
        # Non-ascending tiers
        with pytest.raises(ValueError, match="Tiers must be in ascending order"):
            TieredCommission(tiers=[(5000, 0.001), (3000, 0.002)])

        # Negative rate
        with pytest.raises(ValueError, match="Commission rates cannot be negative"):
            TieredCommission(tiers=[(5000, -0.001)])

    def test_negative_minimum_raises(self):
        """Test that negative minimum raises error."""
        with pytest.raises(ValueError, match="Minimum commission cannot be negative"):
            TieredCommission(minimum=-1.0)

    def test_repr(self):
        """Test string representation."""
        model = TieredCommission()
        assert "TieredCommission" in repr(model)


class TestMakerTakerCommission:
    """Test MakerTakerCommission model."""

    def test_taker_fee(self, sample_order):
        """Test taker fee for market orders."""
        model = MakerTakerCommission()
        # Market order takes liquidity
        commission = model.calculate(sample_order, 100, 150.0)
        assert commission == 4.5  # 15000 * 0.0003

    def test_maker_rebate(self):
        """Test maker rebate for limit orders."""
        limit_order = Order(
            order_id="TEST002",
            asset_id="AAPL",

            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=150.0,
            created_time=datetime.now(),
        )
        model = MakerTakerCommission()
        commission = model.calculate(limit_order, 100, 150.0)
        assert commission == -3.0  # 15000 * -0.0002 (rebate)

    def test_custom_rates(self, sample_order):
        """Test custom maker-taker rates."""
        model = MakerTakerCommission(maker_rate=0.0001, taker_rate=0.0005)
        commission = model.calculate(sample_order, 100, 150.0)
        assert commission == 7.5  # 15000 * 0.0005

    def test_invalid_rates_raise(self):
        """Test that invalid rates raise errors."""
        # Negative taker rate
        with pytest.raises(ValueError, match="Taker rate should be positive"):
            MakerTakerCommission(taker_rate=-0.0003)

        # Maker rate exceeds taker rate
        with pytest.raises(ValueError, match="Maker rate should not exceed taker rate"):
            MakerTakerCommission(maker_rate=0.001, taker_rate=0.0005)

    def test_rebate_cap(self):
        """Test that rebates are capped."""
        limit_order = Order(
            order_id="TEST003",
            asset_id="AAPL",

            quantity=1000,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=10.0,
            created_time=datetime.now(),
        )
        model = MakerTakerCommission(maker_rate=-0.002)  # Large rebate
        commission = model.calculate(limit_order, 1000, 10.0)
        assert commission == -10.0  # Capped at 10bps

    def test_repr(self):
        """Test string representation."""
        model = MakerTakerCommission()
        assert "MakerTakerCommission" in repr(model)


class TestAssetClassCommission:
    """Test AssetClassCommission model."""

    def test_equity_commission(self, sample_order):
        """Test equity commission."""
        model = AssetClassCommission()
        sample_order.metadata["asset_class"] = "equity"
        commission = model.calculate(sample_order, 100, 150.0)
        assert commission == 15.0  # 15000 * 0.001

    def test_futures_commission(self, sample_order):
        """Test futures commission (per contract)."""
        model = AssetClassCommission()
        sample_order.metadata["asset_class"] = "futures"
        commission = model.calculate(sample_order, 5, 4000.0)  # 5 contracts
        assert commission == 12.5  # 5 * 2.50

    def test_options_commission(self, sample_order):
        """Test options commission (per contract)."""
        model = AssetClassCommission()
        sample_order.metadata["asset_class"] = "options"
        commission = model.calculate(sample_order, 500, 2.5)  # 5 contracts (100 shares each)
        assert commission == 3.25  # (500/100) * 0.65

    def test_forex_commission(self, sample_order):
        """Test forex commission."""
        model = AssetClassCommission()
        sample_order.metadata["asset_class"] = "forex"
        commission = model.calculate(sample_order, 10000, 1.2)  # 10k units
        assert commission == 2.4  # 12000 * 0.0002

    def test_crypto_commission(self, sample_order):
        """Test crypto commission."""
        model = AssetClassCommission()
        sample_order.metadata["asset_class"] = "crypto"
        commission = model.calculate(sample_order, 0.5, 40000.0)  # 0.5 BTC
        assert commission == 40.0  # 20000 * 0.002

    def test_default_commission(self, sample_order):
        """Test default commission for unknown asset class."""
        model = AssetClassCommission()
        sample_order.metadata["asset_class"] = "unknown"
        commission = model.calculate(sample_order, 100, 150.0)
        assert commission == 15.0  # 15000 * 0.001

    def test_no_metadata_defaults_to_equity(self, sample_order):
        """Test that missing metadata defaults to equity."""
        model = AssetClassCommission()
        commission = model.calculate(sample_order, 100, 150.0)
        assert commission == 15.0  # 15000 * 0.001

    def test_repr(self):
        """Test string representation."""
        model = AssetClassCommission()
        assert "AssetClassCommission" in repr(model)


class TestInteractiveBrokersCommission:
    """Test InteractiveBrokersCommission model."""

    def test_fixed_tier_minimum(self, sample_order):
        """Test fixed tier with minimum commission."""
        model = InteractiveBrokersCommission(tier="fixed")
        commission = model.calculate(sample_order, 100, 10.0)
        assert commission == 1.0  # Minimum $1

    def test_fixed_tier_normal(self, sample_order):
        """Test fixed tier normal calculation."""
        model = InteractiveBrokersCommission(tier="fixed")
        commission = model.calculate(sample_order, 500, 50.0)
        assert commission == 2.5  # 500 * 0.005

    def test_fixed_tier_maximum(self, sample_order):
        """Test fixed tier with maximum commission."""
        model = InteractiveBrokersCommission(tier="fixed")
        commission = model.calculate(sample_order, 1000, 200.0)
        # Per share would be 1000 * 0.005 = 5.0
        # Maximum is 1000 * 200 * 0.01 = 2000.0
        # Actual is min(5.0, 2000.0) = 5.0
        assert commission == 5.0

    def test_tiered_small_quantity(self, sample_order):
        """Test tiered pricing for small quantity."""
        model = InteractiveBrokersCommission(tier="tiered")
        commission = model.calculate(sample_order, 100, 50.0)
        assert commission == max(100 * 0.0035, 0.35)

    def test_tiered_medium_quantity(self, sample_order):
        """Test tiered pricing for medium quantity."""
        model = InteractiveBrokersCommission(tier="tiered")
        commission = model.calculate(sample_order, 1000, 50.0)
        assert commission == 1000 * 0.0025

    def test_tiered_large_quantity(self, sample_order):
        """Test tiered pricing for large quantity."""
        model = InteractiveBrokersCommission(tier="tiered")
        commission = model.calculate(sample_order, 5000, 50.0)
        assert commission == 5000 * 0.0015

    def test_tiered_minimum(self, sample_order):
        """Test tiered pricing minimum commission."""
        model = InteractiveBrokersCommission(tier="tiered")
        commission = model.calculate(sample_order, 10, 50.0)
        assert commission == 0.35  # Minimum

    def test_invalid_tier_raises(self):
        """Test that invalid tier raises error."""
        with pytest.raises(ValueError, match="Tier must be 'fixed' or 'tiered'"):
            InteractiveBrokersCommission(tier="invalid")

    def test_repr(self):
        """Test string representation."""
        model = InteractiveBrokersCommission(tier="fixed")
        assert repr(model) == "InteractiveBrokersCommission(tier='fixed')"


class TestBrokerIntegration:
    """Test commission model integration with broker."""

    def test_broker_with_commission_model(self):
        """Test that broker can use commission models."""
        from datetime import datetime

        from ml4t.backtest.core.event import MarketEvent
        from ml4t.backtest.core.types import MarketDataType, OrderSide, OrderType
        from ml4t.backtest.execution.broker import SimulationBroker
        from ml4t.backtest.execution.order import Order

        broker = SimulationBroker(
            initial_cash=100000.0,
            commission_model=FlatCommission(commission=2.0),
            execution_delay=False,
        )

        order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100.0,
        )
        broker.submit_order(order)

        market_event = MarketEvent(
            timestamp=datetime.now(),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            close=150.0,
        )
        fills = broker.on_market_event(market_event)

        assert len(fills) == 1
        assert fills[0].commission == 2.0

    def test_broker_fallback_commission(self):
        """Test broker fallback when no model specified."""
        from datetime import datetime

        from ml4t.backtest.core.event import MarketEvent
        from ml4t.backtest.core.types import MarketDataType, OrderSide, OrderType
        from ml4t.backtest.execution.broker import SimulationBroker
        from ml4t.backtest.execution.order import Order

        broker = SimulationBroker(initial_cash=100000.0, execution_delay=False)

        order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100.0,
        )
        broker.submit_order(order)

        market_event = MarketEvent(
            timestamp=datetime.now(),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            close=150.0,
        )
        fills = broker.on_market_event(market_event)

        assert len(fills) == 1
        assert fills[0].commission == 1.0  # Default FlatCommission
