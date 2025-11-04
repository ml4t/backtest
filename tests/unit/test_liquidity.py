"""Test suite for liquidity modeling functionality."""

from datetime import datetime

from qengine.core.types import OrderSide, OrderType
from qengine.execution.broker import SimulationBroker
from qengine.execution.liquidity import (
    ConstantLiquidityModel,
    LiquidityInfo,
    RealisticLiquidityModel,
    VolumeLimitedLiquidityModel,
)
from qengine.execution.order import Order


class TestLiquidityModels:
    """Test different liquidity model implementations."""

    def test_constant_liquidity_model(self):
        """Test constant liquidity model."""
        model = ConstantLiquidityModel(default_volume=10000.0)

        # Default volume
        volume = model.get_available_volume("AAPL", 150.0, "buy")
        assert volume == 10000.0

        # Set specific liquidity
        liquidity_info = LiquidityInfo(asset_id="AAPL", available_volume=5000.0)
        model.set_liquidity("AAPL", liquidity_info)

        volume = model.get_available_volume("AAPL", 150.0, "buy")
        assert volume == 5000.0

        # Update doesn't change constant model
        model.update_volume("AAPL", 150.0, "buy", 1000.0)
        volume = model.get_available_volume("AAPL", 150.0, "buy")
        assert volume == 5000.0

    def test_volume_limited_liquidity_model(self):
        """Test volume-limited liquidity model."""
        model = VolumeLimitedLiquidityModel()

        # Set volume limit
        model.set_volume_limit("AAPL", 5000.0)

        # Initial volume should equal limit
        volume = model.get_available_volume("AAPL", 150.0, "buy")
        assert volume == 5000.0

        # Consume some volume
        model.update_volume("AAPL", 150.0, "buy", 2000.0)

        # Should have less volume available
        volume = model.get_available_volume("AAPL", 150.0, "buy")
        assert volume == 3000.0

        # Reset volume
        model.reset_volume("AAPL")
        volume = model.get_available_volume("AAPL", 150.0, "buy")
        assert volume == 5000.0

    def test_realistic_liquidity_model(self):
        """Test realistic liquidity model with regeneration."""
        model = RealisticLiquidityModel(
            default_volume=10000.0,
            regeneration_rate=0.0,  # Disable for testing
        )

        # Initial volume
        volume = model.get_available_volume("AAPL", 150.0, "buy")
        assert volume == 10000.0

        # Consume volume
        model.update_volume("AAPL", 150.0, "buy", 3000.0)

        # Should have reduced volume
        volume = model.get_available_volume("AAPL", 150.0, "buy")
        assert volume == 7000.0

    def test_max_fill_quantity(self):
        """Test max fill quantity calculation."""
        model = ConstantLiquidityModel(default_volume=1000.0)

        # Create order larger than available liquidity
        order = Order(
            order_id="TEST001",
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=5000.0,
        )

        # Should be limited by liquidity
        max_fill = model.get_max_fill_quantity(order, 150.0)
        assert max_fill == 1000.0

        # Create smaller order
        small_order = Order(
            order_id="TEST002",
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=500.0,
        )

        # Should be limited by order size
        max_fill = model.get_max_fill_quantity(small_order, 150.0)
        assert max_fill == 500.0


class TestBrokerLiquidityIntegration:
    """Test liquidity model integration with broker."""

    def test_broker_without_liquidity_model(self):
        """Test broker behavior without liquidity model."""
        broker = SimulationBroker(initial_cash=100000.0)

        # Create large order
        order = Order(
            order_id="TEST001",
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=5000.0,
        )

        # Should be limited by cash constraints (commission reduces available cash)
        # At $100/share with $1 commission: (100000 - 1) / 100 ≈ 999.99 shares
        timestamp = datetime.now()
        fill_result = broker.fill_simulator.try_fill_order(
            order,
            100.0,
            broker.position_tracker.get_cash(),
            broker.position_tracker.get_position("AAPL"),
            timestamp,
        )

        assert fill_result is not None
        # Cash constraint limits to ~998.9 shares (accounting for commission and slippage)
        # With default 0.01% slippage: fill_price = 100.01
        # Max shares: 100000 / (100.01 * 1.001) ≈ 998.9
        assert 998.0 < fill_result.fill_quantity < 1000.0

    def test_broker_with_constant_liquidity(self):
        """Test broker with constant liquidity model."""
        liquidity_model = ConstantLiquidityModel(default_volume=1000.0)
        broker = SimulationBroker(
            initial_cash=100000.0,
            liquidity_model=liquidity_model,
        )

        # Create order larger than available liquidity
        order = Order(
            order_id="TEST001",
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=5000.0,
        )

        # Both liquidity (1000 shares) and cash constraints apply
        # Cash constraint is more restrictive: (100000 - 1) / 100 ≈ 999.99 shares
        timestamp = datetime.now()
        fill_result = broker.fill_simulator.try_fill_order(
            order,
            100.0,
            broker.position_tracker.get_cash(),
            broker.position_tracker.get_position("AAPL"),
            timestamp,
        )

        assert fill_result is not None
        # Cash constraint is more restrictive than liquidity constraint
        # With slippage, should be around 998.9 shares
        assert 998.0 < fill_result.fill_quantity < 1000.0

    def test_broker_liquidity_constraint_dominates(self):
        """Test broker behavior when liquidity constraint is more restrictive than cash."""
        liquidity_model = ConstantLiquidityModel(default_volume=500.0)  # Small liquidity
        broker = SimulationBroker(
            initial_cash=100000.0,  # Plenty of cash
            liquidity_model=liquidity_model,
        )

        # Create order where liquidity is the constraint
        order = Order(
            order_id="TEST001",
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=2000.0,
        )

        # Should be limited by liquidity (500 shares), not cash
        timestamp = datetime.now()
        fill_result = broker.fill_simulator.try_fill_order(
            order,
            10.0,
            broker.position_tracker.get_cash(),
            broker.position_tracker.get_position("AAPL"),
            timestamp,
        )

        assert fill_result is not None
        assert fill_result.fill_quantity == 500.0  # Limited by liquidity

    def test_broker_with_volume_limited_liquidity(self):
        """Test broker with volume-limited liquidity model."""
        liquidity_model = VolumeLimitedLiquidityModel()
        liquidity_model.set_volume_limit("AAPL", 1500.0)  # Reduced limit for single order test

        broker = SimulationBroker(
            initial_cash=500000.0,  # Increase cash to avoid cash constraints
            liquidity_model=liquidity_model,
        )

        # Test 1: Order within liquidity limit should fill completely
        order1 = Order(
            order_id="TEST001",
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=1000.0,
        )

        timestamp = datetime.now()
        fill_result1 = broker.fill_simulator.try_fill_order(
            order1,
            100.0,
            broker.position_tracker.get_cash(),
            broker.position_tracker.get_position("AAPL"),
            timestamp,
        )
        assert fill_result1 is not None
        assert fill_result1.fill_quantity == 1000.0  # Within 1500 limit

        # Test 2: Order exceeding liquidity limit should be constrained
        liquidity_model2 = VolumeLimitedLiquidityModel()
        liquidity_model2.set_volume_limit("AAPL", 800.0)  # Smaller limit

        broker2 = SimulationBroker(
            initial_cash=500000.0,
            liquidity_model=liquidity_model2,
        )

        order2 = Order(
            order_id="TEST002",
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=1000.0,
        )

        fill_result2 = broker2.fill_simulator.try_fill_order(
            order2,
            100.0,
            broker2.position_tracker.get_cash(),
            broker2.position_tracker.get_position("AAPL"),
            timestamp,
        )
        assert fill_result2 is not None
        assert fill_result2.fill_quantity == 800.0  # Limited by liquidity

    def test_broker_liquidity_too_small(self):
        """Test broker rejects order when liquidity is too small."""
        liquidity_model = ConstantLiquidityModel(default_volume=0.005)  # Very small
        broker = SimulationBroker(
            initial_cash=100000.0,
            liquidity_model=liquidity_model,
        )

        # Create order
        order = Order(
            order_id="TEST001",
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100.0,
        )

        # Should be rejected (too small to fill)
        timestamp = datetime.now()
        fill_result = broker.fill_simulator.try_fill_order(
            order,
            100.0,
            broker.position_tracker.get_cash(),
            broker.position_tracker.get_position("AAPL"),
            timestamp,
        )
        assert fill_result is None

    def test_partial_fill_with_cash_and_liquidity_constraints(self):
        """Test that both cash and liquidity constraints work together."""
        liquidity_model = ConstantLiquidityModel(default_volume=100.0)
        broker = SimulationBroker(
            initial_cash=5000.0,  # Limited cash
            liquidity_model=liquidity_model,
        )

        # Create large order that would be limited by both cash and liquidity
        order = Order(
            order_id="TEST001",
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=1000.0,
        )

        # At $100/share:
        # - Liquidity limit: 100 shares
        # - Cash limit: ~50 shares (5000/100)
        # Should be limited by cash (more restrictive)
        timestamp = datetime.now()
        fill_result = broker.fill_simulator.try_fill_order(
            order,
            100.0,
            broker.position_tracker.get_cash(),
            broker.position_tracker.get_position("AAPL"),
            timestamp,
        )

        assert fill_result is not None
        # Should be limited by cash constraint (around 49 shares after commission)
        assert fill_result.fill_quantity < 100.0  # Less than liquidity limit
        assert fill_result.fill_quantity < 50.0  # Less than naive cash limit
