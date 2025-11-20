"""Unit tests for FillSimulator component."""

from datetime import datetime

import pytest

from ml4t.backtest.core.constants import MIN_FILL_SIZE
from ml4t.backtest.core.types import OrderSide, OrderType
from ml4t.backtest.data.asset_registry import AssetRegistry
from ml4t.backtest.execution.commission import PercentageCommission
from ml4t.backtest.execution.fill_simulator import FillResult, FillSimulator
from ml4t.backtest.execution.order import Order, OrderState


class TestFillSimulator:
    """Test suite for FillSimulator."""

    @pytest.fixture
    def asset_registry(self):
        """Create asset registry with test assets."""
        return AssetRegistry()

    @pytest.fixture
    def fill_simulator(self, asset_registry):
        """Create FillSimulator with default settings (no leverage)."""
        return FillSimulator(
            asset_registry=asset_registry,
            max_leverage=1.0,
        )

    @pytest.fixture
    def fill_simulator_2x_leverage(self, asset_registry):
        """Create FillSimulator with 2x leverage."""
        return FillSimulator(
            asset_registry=asset_registry,
            max_leverage=2.0,
        )

    @pytest.fixture
    def fill_simulator_with_commission(self, asset_registry):
        """Create FillSimulator with commission model."""
        commission_model = PercentageCommission(rate=0.001)  # 0.1%
        return FillSimulator(
            asset_registry=asset_registry,
            commission_model=commission_model,
            max_leverage=1.0,
        )

    @pytest.fixture
    def market_order(self):
        """Create sample market buy order."""
        return Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100.0,
        )

    @pytest.fixture
    def limit_order(self):
        """Create sample limit buy order."""
        return Order(
            asset_id="AAPL",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=100.0,
            limit_price=150.0,
        )

    @pytest.fixture
    def sell_order(self):
        """Create sample market sell order."""
        return Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            quantity=50.0,
        )

    # ==================== INITIALIZATION TESTS ====================

    def test_fill_simulator_initialization(self, asset_registry):
        """Test FillSimulator initialization with default parameters."""
        simulator = FillSimulator(
            asset_registry=asset_registry,
            max_leverage=1.0,
        )

        assert simulator.asset_registry == asset_registry
        assert simulator.max_leverage == 1.0
        assert simulator.commission_model is None
        assert simulator.slippage_model is None
        assert simulator.market_impact_model is None
        assert simulator.liquidity_model is None
        assert simulator.margin_account is None
        assert simulator._fill_count == 0

    def test_fill_simulator_with_custom_leverage(self, asset_registry):
        """Test FillSimulator initialization with custom leverage."""
        simulator = FillSimulator(
            asset_registry=asset_registry,
            max_leverage=2.5,
        )

        assert simulator.max_leverage == 2.5

    # ==================== BASIC FUNCTIONALITY TESTS ====================

    def test_simple_market_order_fill(self, fill_simulator, market_order):
        """Test basic market order fill with sufficient cash."""
        result = fill_simulator.try_fill_order(
            order=market_order,
            market_price=100.0,
            current_cash=20000.0,
            current_position=0.0,
            timestamp=datetime.now(),
        )

        assert result is not None
        assert isinstance(result, FillResult)
        assert result.fill_quantity == 100.0
        assert result.fill_price > 100.0  # Should include default slippage
        assert result.commission > 0  # Default $1 commission
        assert market_order.state == OrderState.FILLED
        assert result.fill_event is not None
        assert result.fill_event.asset_id == "AAPL"
        assert result.fill_event.side == OrderSide.BUY
        assert result.fill_event.fill_quantity == 100.0

    def test_limit_order_fill_at_limit_price(self, fill_simulator, limit_order):
        """Test limit order fills at limit price when market price is favorable."""
        result = fill_simulator.try_fill_order(
            order=limit_order,
            market_price=145.0,  # Market price below limit
            current_cash=20000.0,
            current_position=0.0,
            timestamp=datetime.now(),
        )

        assert result is not None
        assert result.fill_price == 145.0  # Should fill at market, which is better
        assert result.fill_quantity == 100.0

    def test_order_cannot_fill_when_price_unfavorable(self, fill_simulator, limit_order):
        """Test that limit order doesn't fill when price is unfavorable."""
        result = fill_simulator.try_fill_order(
            order=limit_order,
            market_price=155.0,  # Market price above limit for BUY
            current_cash=20000.0,
            current_position=0.0,
            timestamp=datetime.now(),
        )

        assert result is None
        assert limit_order.state == OrderState.PENDING  # State unchanged

    def test_fill_count_increments(self, fill_simulator, market_order):
        """Test that fill counter increments correctly."""
        assert fill_simulator._fill_count == 0

        fill_simulator.try_fill_order(
            order=market_order,
            market_price=100.0,
            current_cash=20000.0,
            current_position=0.0,
            timestamp=datetime.now(),
        )

        assert fill_simulator._fill_count == 1

        # Create another order and fill it
        order2 = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=50.0,
        )
        fill_simulator.try_fill_order(
            order=order2,
            market_price=100.0,
            current_cash=20000.0,
            current_position=0.0,
            timestamp=datetime.now(),
        )

        assert fill_simulator._fill_count == 2

    def test_reset_clears_fill_count(self, fill_simulator, market_order):
        """Test that reset clears fill count."""
        fill_simulator.try_fill_order(
            order=market_order,
            market_price=100.0,
            current_cash=20000.0,
            current_position=0.0,
            timestamp=datetime.now(),
        )

        assert fill_simulator._fill_count == 1

        fill_simulator.reset()

        assert fill_simulator._fill_count == 0

    # ==================== MAX_LEVERAGE TESTS (CRITICAL) ====================

    def test_max_leverage_default_prevents_overleveraging(self, fill_simulator, market_order):
        """Test that default max_leverage=1.0 prevents overleveraging."""
        # With $10,000 cash and max_leverage=1.0, should only be able to buy ~$10,000 worth
        result = fill_simulator.try_fill_order(
            order=market_order,
            market_price=100.0,
            current_cash=10000.0,
            current_position=0.0,
            timestamp=datetime.now(),
        )

        assert result is not None
        # Should fill partially - can only afford ~99 shares with commission
        # Formula: cash / (price * (1 + commission_rate))
        # With default $1 commission, commission_rate = 1/(100*100) = 0.0001
        # Max: 10000 / (100 * 1.0001) = ~99.99 shares
        assert result.fill_quantity < 100.0
        assert result.fill_quantity > 90.0  # But close to 100

        # Verify total cost doesn't exceed cash
        total_cost = result.fill_quantity * result.fill_price + result.commission
        assert total_cost <= 10000.0

    def test_max_leverage_2x_allows_double_position(self, fill_simulator_2x_leverage, market_order):
        """Test that max_leverage=2.0 allows buying 2x cash position."""
        # With $10,000 cash and max_leverage=2.0, should be able to buy ~$20,000 worth
        result = fill_simulator_2x_leverage.try_fill_order(
            order=market_order,
            market_price=100.0,
            current_cash=10000.0,
            current_position=0.0,
            timestamp=datetime.now(),
        )

        assert result is not None
        # Should fill completely - can afford 100 shares with 2x leverage
        # Max: (10000 * 2.0) / (100 * 1.0001) = ~199.98 shares
        assert result.fill_quantity == 100.0

    def test_max_leverage_enforced_as_capital_depletes(self, fill_simulator):
        """Test that position size reduces as cash depletes with max_leverage=1.0."""
        # Start with $10,000 cash
        # After losses, cash drops to $5,000
        # With max_leverage=1.0, should only be able to buy ~$5,000 worth

        large_order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100.0,  # Want 100 shares @ $100 = $10,000
        )

        result = fill_simulator.try_fill_order(
            order=large_order,
            market_price=100.0,
            current_cash=5000.0,  # Depleted capital
            current_position=0.0,
            timestamp=datetime.now(),
        )

        assert result is not None
        # Should only fill ~50 shares (partial fill)
        # Max: 5000 / (100 * 1.0001) = ~49.995 shares
        assert result.fill_quantity < 51.0
        assert result.fill_quantity > 45.0

        # Verify this prevents unlimited leverage
        total_cost = result.fill_quantity * result.fill_price + result.commission
        assert total_cost <= 5000.0

    def test_max_leverage_with_large_order(self, fill_simulator_2x_leverage):
        """Test max_leverage constrains very large orders."""
        # Try to buy $100,000 worth with only $10,000 cash and 2x leverage
        huge_order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=1000.0,  # 1000 shares @ $100 = $100,000
        )

        result = fill_simulator_2x_leverage.try_fill_order(
            order=huge_order,
            market_price=100.0,
            current_cash=10000.0,
            current_position=0.0,
            timestamp=datetime.now(),
        )

        assert result is not None
        # Should only fill ~200 shares (2x leverage on $10K)
        # Max: (10000 * 2.0) / (100 * 1.0001) = ~199.98 shares
        assert result.fill_quantity < 201.0
        assert result.fill_quantity > 195.0

    # ==================== CASH CONSTRAINT TESTS ====================

    def test_insufficient_cash_rejects_order(self, fill_simulator):
        """Test that order is rejected when cash is too low for MIN_FILL_SIZE."""
        tiny_order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100.0,
        )

        # Only $0.50 cash - can't even afford MIN_FILL_SIZE (0.01 shares)
        result = fill_simulator.try_fill_order(
            order=tiny_order,
            market_price=100.0,
            current_cash=0.50,
            current_position=0.0,
            timestamp=datetime.now(),
        )

        assert result is None

    def test_insufficient_cash_partial_fill(self, fill_simulator):
        """Test that order is partially filled when cash is limited."""
        order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100.0,
        )

        result = fill_simulator.try_fill_order(
            order=order,
            market_price=100.0,
            current_cash=5000.0,
            current_position=0.0,
            timestamp=datetime.now(),
        )

        assert result is not None
        assert result.fill_quantity < 100.0
        assert result.fill_quantity > MIN_FILL_SIZE

        # Verify cost doesn't exceed cash
        total_cost = result.fill_quantity * result.fill_price + result.commission
        assert total_cost <= 5000.0

    def test_binary_search_with_tiered_commission(self, asset_registry):
        """Test binary search for complex commission models."""
        # Create simulator with percentage commission model
        commission_model = PercentageCommission(rate=0.01)  # 1% commission
        simulator = FillSimulator(
            asset_registry=asset_registry,
            commission_model=commission_model,
            max_leverage=1.0,
        )

        order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100.0,
        )

        result = simulator.try_fill_order(
            order=order,
            market_price=100.0,
            current_cash=5000.0,
            current_position=0.0,
            timestamp=datetime.now(),
        )

        assert result is not None
        # With 1% commission: cost = qty * price * 1.01
        # Max qty: 5000 / (100 * 1.01) = ~49.50 shares
        assert result.fill_quantity < 50.0
        assert result.fill_quantity > 48.0

        # Verify exact cost
        total_cost = result.fill_quantity * result.fill_price + result.commission
        assert total_cost <= 5000.0

    def test_binary_search_converges_below_min_fill_size(self, fill_simulator):
        """Test that None is returned when binary search converges to <MIN_FILL_SIZE."""
        order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=10.0,
        )

        # Very little cash - less than needed for MIN_FILL_SIZE
        result = fill_simulator.try_fill_order(
            order=order,
            market_price=100.0,
            current_cash=0.50,  # Less than MIN_FILL_SIZE * price
            current_position=0.0,
            timestamp=datetime.now(),
        )

        assert result is None

    # ==================== SELL ORDER TESTS ====================

    def test_sell_order_with_position(self, fill_simulator, sell_order):
        """Test sell order fills when position exists."""
        result = fill_simulator.try_fill_order(
            order=sell_order,
            market_price=100.0,
            current_cash=10000.0,
            current_position=100.0,  # Own 100 shares
            timestamp=datetime.now(),
        )

        assert result is not None
        assert result.fill_quantity == 50.0
        assert sell_order.state == OrderState.FILLED

    def test_sell_order_without_position_rejected(self, fill_simulator, sell_order):
        """Test sell order is rejected when no position exists (no short)."""
        result = fill_simulator.try_fill_order(
            order=sell_order,
            market_price=100.0,
            current_cash=10000.0,
            current_position=0.0,  # No shares to sell
            timestamp=datetime.now(),
        )

        assert result is None

    def test_sell_order_with_insufficient_shares_partial(self, fill_simulator):
        """Test sell order partially fills with available shares."""
        sell_order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            quantity=100.0,  # Want to sell 100
        )

        result = fill_simulator.try_fill_order(
            order=sell_order,
            market_price=100.0,
            current_cash=10000.0,
            current_position=50.0,  # Only have 50 shares
            timestamp=datetime.now(),
        )

        assert result is not None
        assert result.fill_quantity == 50.0  # Partial fill

    # ==================== ORDER STATE TESTS ====================

    def test_order_state_updated_on_fill(self, fill_simulator, market_order):
        """Test that order state is updated correctly on fill."""
        assert market_order.state == OrderState.PENDING

        result = fill_simulator.try_fill_order(
            order=market_order,
            market_price=100.0,
            current_cash=20000.0,
            current_position=0.0,
            timestamp=datetime.now(),
        )

        assert result is not None
        assert market_order.state == OrderState.FILLED
        assert market_order.filled_quantity == result.fill_quantity
        assert market_order.average_fill_price == result.fill_price

    def test_order_state_unchanged_when_cannot_fill(self, fill_simulator, limit_order):
        """Test that order state remains unchanged when fill fails."""
        original_state = limit_order.state

        result = fill_simulator.try_fill_order(
            order=limit_order,
            market_price=155.0,  # Price too high for limit buy
            current_cash=20000.0,
            current_position=0.0,
            timestamp=datetime.now(),
        )

        assert result is None
        assert limit_order.state == original_state
        assert limit_order.filled_quantity == 0.0

    # ==================== EDGE CASES ====================

    def test_zero_cash(self, fill_simulator, market_order):
        """Test handling of zero cash."""
        result = fill_simulator.try_fill_order(
            order=market_order,
            market_price=100.0,
            current_cash=0.0,
            current_position=0.0,
            timestamp=datetime.now(),
        )

        assert result is None

    def test_negative_cash(self, fill_simulator, market_order):
        """Test handling of negative cash (shouldn't happen, but defensive)."""
        result = fill_simulator.try_fill_order(
            order=market_order,
            market_price=100.0,
            current_cash=-1000.0,
            current_position=0.0,
            timestamp=datetime.now(),
        )

        assert result is None

    def test_very_small_order(self, fill_simulator):
        """Test handling of very small order quantities."""
        tiny_order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=0.5,  # Very small
        )

        result = fill_simulator.try_fill_order(
            order=tiny_order,
            market_price=100.0,
            current_cash=10000.0,
            current_position=0.0,
            timestamp=datetime.now(),
        )

        assert result is not None
        assert result.fill_quantity == 0.5

    def test_fill_result_contains_all_required_fields(self, fill_simulator, market_order):
        """Test that FillResult contains all expected fields."""
        result = fill_simulator.try_fill_order(
            order=market_order,
            market_price=100.0,
            current_cash=20000.0,
            current_position=0.0,
            timestamp=datetime.now(),
        )

        assert result is not None
        assert hasattr(result, "fill_event")
        assert hasattr(result, "commission")
        assert hasattr(result, "slippage")
        assert hasattr(result, "fill_quantity")
        assert hasattr(result, "fill_price")

        # Verify types
        assert isinstance(result.commission, float)
        assert isinstance(result.slippage, float)
        assert result.fill_quantity > 0
        assert result.fill_price > 0

    def test_timestamp_propagates_to_fill_event(self, fill_simulator, market_order):
        """Test that timestamp is correctly propagated to fill event."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0)

        result = fill_simulator.try_fill_order(
            order=market_order,
            market_price=100.0,
            current_cash=20000.0,
            current_position=0.0,
            timestamp=timestamp,
        )

        assert result is not None
        assert result.fill_event.timestamp == timestamp

    # ==================== COMMISSION CALCULATION TESTS ====================

    def test_default_commission(self, fill_simulator, market_order):
        """Test default commission based on asset spec taker_fee (0.1%)."""
        result = fill_simulator.try_fill_order(
            order=market_order,
            market_price=100.0,
            current_cash=20000.0,
            current_position=0.0,
            timestamp=datetime.now(),
        )

        assert result is not None
        # Default commission is 0.1% of notional (100 shares * ~100 price = ~10,000 * 0.001 = ~10)
        assert result.commission > 9.0
        assert result.commission < 11.0

    def test_percentage_commission(self, fill_simulator_with_commission, market_order):
        """Test percentage-based commission calculation."""
        result = fill_simulator_with_commission.try_fill_order(
            order=market_order,
            market_price=100.0,
            current_cash=20000.0,
            current_position=0.0,
            timestamp=datetime.now(),
        )

        assert result is not None
        # Commission should be 0.1% of notional
        # Notional = 100 shares * ~100 price = ~10,000
        # Commission = 10,000 * 0.001 = 10.0
        assert result.commission > 9.0
        assert result.commission < 11.0

    # ==================== SLIPPAGE CALCULATION TESTS ====================

    def test_default_slippage_on_market_buy(self, fill_simulator, market_order):
        """Test default slippage (0.01%) on market buy orders."""
        result = fill_simulator.try_fill_order(
            order=market_order,
            market_price=100.0,
            current_cash=20000.0,
            current_position=0.0,
            timestamp=datetime.now(),
        )

        assert result is not None
        # Buy should have positive slippage (worse price)
        assert result.fill_price > 100.0
        # Default is 0.01% = 100 * 1.0001 = 100.01
        assert result.fill_price == pytest.approx(100.01, rel=0.001)

    def test_default_slippage_on_market_sell(self, fill_simulator, sell_order):
        """Test default slippage (0.01%) on market sell orders."""
        result = fill_simulator.try_fill_order(
            order=sell_order,
            market_price=100.0,
            current_cash=10000.0,
            current_position=100.0,
            timestamp=datetime.now(),
        )

        assert result is not None
        # Sell should have negative slippage (worse price)
        assert result.fill_price < 100.0
        # Default is 0.01% = 100 * 0.9999 = 99.99
        assert result.fill_price == pytest.approx(99.99, rel=0.001)

    # ==================== LIQUIDITY MODEL INTEGRATION TESTS ====================

    def test_liquidity_model_reduces_fill_quantity(self, asset_registry):
        """Test that liquidity model can reduce fill quantity."""
        from unittest.mock import Mock

        # Mock liquidity model that limits to 500 shares
        liquidity_model = Mock()
        liquidity_model.get_max_fill_quantity.return_value = 500.0

        simulator = FillSimulator(
            asset_registry=asset_registry,
            liquidity_model=liquidity_model,
            max_leverage=1.0,
        )

        # Create order for 1000 shares
        large_order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=1000.0,
        )

        result = simulator.try_fill_order(
            order=large_order,
            market_price=100.0,
            current_cash=150000.0,
            current_position=0.0,
            timestamp=datetime.now(),
        )

        # Should be reduced to 500 by liquidity model
        assert result is not None
        assert result.fill_quantity == 500.0

    def test_liquidity_below_min_fill_size_rejects(self, asset_registry):
        """Test that order is rejected when liquidity < MIN_FILL_SIZE."""
        from unittest.mock import Mock

        # Mock liquidity model that returns very low liquidity
        liquidity_model = Mock()
        liquidity_model.get_max_fill_quantity.return_value = MIN_FILL_SIZE / 2

        simulator = FillSimulator(
            asset_registry=asset_registry,
            liquidity_model=liquidity_model,
            max_leverage=1.0,
        )

        order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100.0,
        )

        result = simulator.try_fill_order(
            order=order,
            market_price=100.0,
            current_cash=20000.0,
            current_position=0.0,
            timestamp=datetime.now(),
        )

        assert result is None

    # ==================== MARKET IMPACT MODEL INTEGRATION TESTS ====================

    def test_market_impact_applied_to_price(self, asset_registry):
        """Test that market impact affects fill price."""
        from unittest.mock import Mock

        # Mock market impact model
        impact_model = Mock()
        impact_model.get_current_impact.return_value = 2.0  # $2 impact
        impact_model.calculate_impact.return_value = (1.0, 1.0)  # Permanent, temporary

        simulator = FillSimulator(
            asset_registry=asset_registry,
            market_impact_model=impact_model,
            max_leverage=1.0,
        )

        order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100.0,
        )

        result = simulator.try_fill_order(
            order=order,
            market_price=100.0,
            current_cash=20000.0,
            current_position=0.0,
            timestamp=datetime.now(),
        )

        assert result is not None
        # Verify impact model was called
        impact_model.get_current_impact.assert_called_once()
        impact_model.calculate_impact.assert_called_once()
        impact_model.update_market_state.assert_called_once()

    # ==================== MARGIN ACCOUNT INTEGRATION TESTS ====================

    def test_margin_sufficient_fills_order(self, asset_registry):
        """Test that order fills when sufficient margin is available."""
        from unittest.mock import Mock

        margin_account = Mock()
        margin_account.check_margin_requirement.return_value = (True, 1000.0)

        # Create asset spec that requires margin
        from ml4t.backtest.core.assets import AssetClass, AssetSpec, ContractType

        future_spec = AssetSpec(
            asset_id="ES",
            asset_class=AssetClass.FUTURE,
            contract_type=ContractType.FUTURE,
            contract_size=50.0,
        )
        asset_registry.register(future_spec)

        simulator = FillSimulator(
            asset_registry=asset_registry,
            margin_account=margin_account,
            max_leverage=1.0,
        )

        order = Order(
            asset_id="ES",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=1.0,
        )

        result = simulator.try_fill_order(
            order=order,
            market_price=4000.0,
            current_cash=50000.0,
            current_position=0.0,
            timestamp=datetime.now(),
        )

        assert result is not None
        assert result.fill_quantity == 1.0

    def test_margin_insufficient_rejects_order(self, asset_registry):
        """Test that order is rejected when insufficient margin."""
        from unittest.mock import Mock

        margin_account = Mock()
        margin_account.check_margin_requirement.return_value = (False, 50000.0)
        margin_account.available_margin = 0.0

        # Create asset spec that requires margin
        from ml4t.backtest.core.assets import AssetClass, AssetSpec, ContractType

        future_spec = AssetSpec(
            asset_id="ES",
            asset_class=AssetClass.FUTURE,
            contract_type=ContractType.FUTURE,
            contract_size=50.0,
            min_quantity=1.0,
        )
        asset_registry.register(future_spec)

        simulator = FillSimulator(
            asset_registry=asset_registry,
            margin_account=margin_account,
            max_leverage=1.0,
        )

        order = Order(
            asset_id="ES",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=1.0,
        )

        result = simulator.try_fill_order(
            order=order,
            market_price=4000.0,
            current_cash=50000.0,
            current_position=0.0,
            timestamp=datetime.now(),
        )

        assert result is None

    # ==================== LIMIT ORDER SELL TESTS ====================

    def test_limit_sell_order_fill_at_market_price(self, fill_simulator):
        """Test limit sell order fills at market price when favorable."""
        limit_sell = Order(
            asset_id="AAPL",
            order_type=OrderType.LIMIT,
            side=OrderSide.SELL,
            quantity=50.0,
            limit_price=95.0,  # Limit is $95
        )

        result = fill_simulator.try_fill_order(
            order=limit_sell,
            market_price=100.0,  # Market is higher - favorable for sell
            current_cash=10000.0,
            current_position=100.0,
            timestamp=datetime.now(),
        )

        assert result is not None
        # Should fill at market price (better than limit)
        assert result.fill_price == 100.0
        assert result.fill_quantity == 50.0

    def test_limit_sell_order_no_fill_when_price_unfavorable(self, fill_simulator):
        """Test limit sell order doesn't fill when market price below limit."""
        limit_sell = Order(
            asset_id="AAPL",
            order_type=OrderType.LIMIT,
            side=OrderSide.SELL,
            quantity=50.0,
            limit_price=100.0,
        )

        result = fill_simulator.try_fill_order(
            order=limit_sell,
            market_price=95.0,  # Market below limit - unfavorable for sell
            current_cash=10000.0,
            current_position=100.0,
            timestamp=datetime.now(),
        )

        assert result is None
