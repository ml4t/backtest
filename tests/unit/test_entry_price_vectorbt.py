"""Unit tests for VectorBT-compatible entry price logic.

Tests verify that qengine's entry price calculation matches VectorBT Pro exactly:
1. Slippage applied FIRST to base price
2. Size calculated using slippage-adjusted price
3. Fees calculated on slippage-adjusted order value
4. Order of operations: Slippage → Size → Fees → Cash deduction

Reference: TASK-003 - VectorBT Entry Price Analysis
"""

import pytest
import sys
from pathlib import Path

# Add tests directory to path for validation models
tests_path = Path(__file__).parent.parent
if str(tests_path) not in sys.path:
    sys.path.insert(0, str(tests_path))

from qengine.core.event import MarketEvent
from qengine.core.types import MarketDataType
from qengine.data.asset_registry import AssetRegistry, AssetSpec
from qengine.execution.broker import SimulationBroker
from qengine.execution.fill_simulator import FillSimulator
from qengine.execution.order import Order, OrderSide, OrderType
from validation.models import VectorBTCommission, VectorBTSlippage


class TestVectorBTEntryPriceLogic:
    """Test VectorBT-compatible entry price calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.asset_registry = AssetRegistry()

        # Register BTC with specs
        self.asset_registry.register(
            spec=AssetSpec(
                asset_id="BTC",
                asset_type="crypto",
                tick_size=0.01,
                lot_size=1,
                multiplier=1.0,
                margin_requirement=1.0,
            )
        )

    def test_slippage_applied_first(self):
        """Test that slippage is applied before size and fee calculations."""
        # Create FillSimulator with VectorBT models
        slippage_model = VectorBTSlippage(slippage=0.0002)  # 0.02%
        commission_model = VectorBTCommission(fee_rate=0.0002, fixed_fee=0.0)

        fill_simulator = FillSimulator(
            asset_registry=self.asset_registry,
            commission_model=commission_model,
            slippage_model=slippage_model,
        )

        # Create buy order
        order = Order(
            order_type=OrderType.MARKET,
            order_id="test_001",
            asset_id="BTC",
            side=OrderSide.BUY,
            quantity=0.1,
        )

        # Market event with base price = $50,000
        market_event = MarketEvent(
            timestamp="2024-01-01T00:00:00Z",
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0,
        )

        # Try to fill the order
        fill_result = fill_simulator.try_fill_order(
            order=order,
            timestamp=market_event.timestamp,
            market_price=market_event.open,  # $50,000 base price
            high=market_event.high,
            low=market_event.low,
            current_cash=10000.0,
            current_position=0.0,
        )

        assert fill_result is not None, "Order should fill"

        # Expected: base_price * (1 + slippage) = 50000 * 1.0002 = 50010
        expected_fill_price = 50000.0 * 1.0002
        assert abs(fill_result.fill_price - expected_fill_price) < 0.01

        # Verify fees calculated on slippage-adjusted price
        order_value = fill_result.fill_quantity * fill_result.fill_price
        expected_commission = order_value * 0.0002
        assert abs(fill_result.commission - expected_commission) < 0.01

    def test_fees_on_slippage_adjusted_price(self):
        """Test that fees are calculated on slippage-adjusted order value."""
        commission_model = VectorBTCommission(fee_rate=0.0002, fixed_fee=0.0)
        slippage_model = VectorBTSlippage(slippage=0.0002)

        fill_simulator = FillSimulator(
            asset_registry=self.asset_registry,
            commission_model=commission_model,
            slippage_model=slippage_model,
        )

        order = Order(
            order_type=OrderType.MARKET,
            order_id="test_002",
            asset_id="BTC",
            side=OrderSide.BUY,
            quantity=0.1,
        )

        fill_result = fill_simulator.try_fill_order(
            order=order,
            timestamp="2024-01-01T00:00:00Z",
            market_price=50000.0,
            high=51000.0,
            low=49000.0,
            current_cash=10000.0,
            current_position=0.0,
        )

        # VectorBT formula:
        # adj_price = 50000 * 1.0002 = 50010
        # order_value = 0.1 * 50010 = 5001
        # fees = 5001 * 0.0002 = 1.0002

        expected_adj_price = 50000.0 * 1.0002
        expected_order_value = 0.1 * expected_adj_price
        expected_fees = expected_order_value * 0.0002

        assert abs(fill_result.fill_price - expected_adj_price) < 0.01
        assert abs(fill_result.commission - expected_fees) < 0.01

    def test_size_calculation_with_slippage(self):
        """Test that size=np.inf uses slippage-adjusted price."""
        commission_model = VectorBTCommission(fee_rate=0.0002, fixed_fee=0.0)
        slippage_model = VectorBTSlippage(slippage=0.0002)

        fill_simulator = FillSimulator(
            asset_registry=self.asset_registry,
            commission_model=commission_model,
            slippage_model=slippage_model,
        )

        # Large order that will be constrained by cash
        order = Order(
            order_type=OrderType.MARKET,
            order_id="test_003",
            asset_id="BTC",
            side=OrderSide.BUY,
            quantity=10.0,  # Request large quantity
        )

        cash_limit = 10000.0

        fill_result = fill_simulator.try_fill_order(
            order=order,
            timestamp="2024-01-01T00:00:00Z",
            market_price=50000.0,
            high=51000.0,
            low=49000.0,
            current_cash=cash_limit,
            current_position=0.0,
        )

        # VectorBT formula for size=np.inf:
        # adj_price = 50000 * 1.0002 = 50010
        # max_req_cash = 10000 / 1.0002 = 9998.0004
        # max_size = 9998.0004 / 50010 = 0.19988

        expected_adj_price = 50000.0 * 1.0002
        expected_max_req_cash = cash_limit / 1.0002
        expected_max_size = expected_max_req_cash / expected_adj_price

        # Size should be close to expected (may be slightly less due to rounding/granularity)
        assert fill_result.fill_quantity <= expected_max_size
        assert fill_result.fill_quantity > expected_max_size * 0.99  # Within 1%

        # Total cost should not exceed cash limit
        total_cost = fill_result.fill_quantity * fill_result.fill_price + fill_result.commission
        assert total_cost <= cash_limit

    def test_fixed_fees_in_size_calculation(self):
        """Test that fixed fees are accounted for in size calculation."""
        commission_model = VectorBTCommission(fee_rate=0.0002, fixed_fee=10.0)
        slippage_model = VectorBTSlippage(slippage=0.0002)

        fill_simulator = FillSimulator(
            asset_registry=self.asset_registry,
            commission_model=commission_model,
            slippage_model=slippage_model,
        )

        order = Order(
            order_type=OrderType.MARKET,
            order_id="test_004",
            asset_id="BTC",
            side=OrderSide.BUY,
            quantity=10.0,
        )

        cash_limit = 10000.0

        fill_result = fill_simulator.try_fill_order(
            order=order,
            timestamp="2024-01-01T00:00:00Z",
            market_price=50000.0,
            high=51000.0,
            low=49000.0,
            current_cash=cash_limit,
            current_position=0.0,
        )

        # VectorBT formula with fixed fees:
        # adj_price = 50000 * 1.0002 = 50010
        # max_req_cash = (10000 - 10) / 1.0002 = 9988.0024
        # max_size = 9988.0024 / 50010 = 0.19968

        # Fixed fees should reduce available size
        # Total cost = order_value + percentage_fees + fixed_fees
        total_cost = (
            fill_result.fill_quantity * fill_result.fill_price +
            fill_result.commission
        )

        assert total_cost <= cash_limit

        # Commission should include fixed fee
        expected_percentage_fees = fill_result.fill_quantity * fill_result.fill_price * 0.0002
        expected_total_fees = expected_percentage_fees + 10.0
        assert abs(fill_result.commission - expected_total_fees) < 0.01

    def test_order_of_operations_example_1(self):
        """Test Example 1 from TASK-003: Finite size entry."""
        # Given (from TASK-003 Example 1):
        # base_price = $50,000
        # size = 0.1 BTC
        # slippage = 0.02% = 0.0002
        # fees = 0.02% = 0.0002
        # fixed_fees = $0

        commission_model = VectorBTCommission(fee_rate=0.0002, fixed_fee=0.0)
        slippage_model = VectorBTSlippage(slippage=0.0002)

        fill_simulator = FillSimulator(
            asset_registry=self.asset_registry,
            commission_model=commission_model,
            slippage_model=slippage_model,
        )

        order = Order(
            order_type=OrderType.MARKET,
            order_id="test_ex1",
            asset_id="BTC",
            side=OrderSide.BUY,
            quantity=0.1,
        )

        fill_result = fill_simulator.try_fill_order(
            order=order,
            timestamp="2024-01-01T00:00:00Z",
            market_price=50000.0,
            high=51000.0,
            low=49000.0,
            current_cash=10000.0,
            current_position=0.0,
        )

        # Expected (from TASK-003):
        # adj_price = 50000 * 1.0002 = 50010.00
        # order_value = 0.1 * 50010.00 = 5001.00
        # fees_paid = 5001.00 * 0.0002 = 1.0002
        # req_cash = 5001.00 + 1.0002 = 5002.0002

        assert abs(fill_result.fill_price - 50010.0) < 0.01
        assert abs(fill_result.fill_quantity - 0.1) < 0.001
        assert abs(fill_result.commission - 1.0002) < 0.01

        total_cost = fill_result.fill_quantity * fill_result.fill_price + fill_result.commission
        assert abs(total_cost - 5002.0002) < 0.01

    def test_order_of_operations_example_2(self):
        """Test Example 2 from TASK-003: size=np.inf entry."""
        # Given (from TASK-003 Example 2):
        # base_price = $50,000
        # cash_limit = $10,000
        # slippage = 0.02% = 0.0002
        # fees = 0.02% = 0.0002
        # fixed_fees = $0

        commission_model = VectorBTCommission(fee_rate=0.0002, fixed_fee=0.0)
        slippage_model = VectorBTSlippage(slippage=0.0002)

        fill_simulator = FillSimulator(
            asset_registry=self.asset_registry,
            commission_model=commission_model,
            slippage_model=slippage_model,
        )

        order = Order(
            order_type=OrderType.MARKET,
            order_id="test_ex2",
            asset_id="BTC",
            side=OrderSide.BUY,
            quantity=10.0,  # Large quantity (will be limited by cash)
        )

        fill_result = fill_simulator.try_fill_order(
            order=order,
            timestamp="2024-01-01T00:00:00Z",
            market_price=50000.0,
            high=51000.0,
            low=49000.0,
            current_cash=10000.0,
            current_position=0.0,
        )

        # Expected (from TASK-003):
        # adj_price = 50000 * 1.0002 = 50010.00
        # max_req_cash = 10000 / 1.0002 = 9998.0004
        # max_acq_size = 9998.0004 / 50010.00 = 0.19988

        assert abs(fill_result.fill_price - 50010.0) < 0.01

        # Size may be slightly less due to granularity, but should be close
        expected_size = 0.19988
        assert fill_result.fill_quantity <= expected_size
        assert fill_result.fill_quantity > expected_size * 0.99

        # Total cost should not exceed cash limit
        total_cost = fill_result.fill_quantity * fill_result.fill_price + fill_result.commission
        assert total_cost <= 10000.0

    def test_order_of_operations_example_3(self):
        """Test Example 3 from TASK-003: Entry with fixed fees."""
        # Given (from TASK-003 Example 3):
        # base_price = $50,000
        # cash_limit = $10,000
        # slippage = 0.02% = 0.0002
        # fees = 0.02% = 0.0002
        # fixed_fees = $10

        commission_model = VectorBTCommission(fee_rate=0.0002, fixed_fee=10.0)
        slippage_model = VectorBTSlippage(slippage=0.0002)

        fill_simulator = FillSimulator(
            asset_registry=self.asset_registry,
            commission_model=commission_model,
            slippage_model=slippage_model,
        )

        order = Order(
            order_type=OrderType.MARKET,
            order_id="test_ex3",
            asset_id="BTC",
            side=OrderSide.BUY,
            quantity=10.0,
        )

        fill_result = fill_simulator.try_fill_order(
            order=order,
            timestamp="2024-01-01T00:00:00Z",
            market_price=50000.0,
            high=51000.0,
            low=49000.0,
            current_cash=10000.0,
            current_position=0.0,
        )

        # Expected (from TASK-003):
        # adj_price = 50000 * 1.0002 = 50010.00
        # max_req_cash = (10000 - 10) / 1.0002 = 9988.0024
        # max_acq_size = 9988.0024 / 50010.00 = 0.19968

        assert abs(fill_result.fill_price - 50010.0) < 0.01

        # Size should be close to expected (allow small floating point tolerance)
        expected_size = 0.19968
        assert abs(fill_result.fill_quantity - expected_size) < 0.0001  # Within 0.01% tolerance
        assert fill_result.fill_quantity > expected_size * 0.99

        # Commission should include fixed fee
        order_value = fill_result.fill_quantity * fill_result.fill_price
        expected_percentage_fees = order_value * 0.0002
        expected_total_fees = expected_percentage_fees + 10.0
        assert abs(fill_result.commission - expected_total_fees) < 0.01

        # Total cost should not exceed cash limit
        total_cost = fill_result.fill_quantity * fill_result.fill_price + fill_result.commission
        assert total_cost <= 10000.0

    def test_sell_order_negative_slippage(self):
        """Test that sell orders use negative slippage (price * (1 - slippage))."""
        commission_model = VectorBTCommission(fee_rate=0.0002, fixed_fee=0.0)
        slippage_model = VectorBTSlippage(slippage=0.0002)

        fill_simulator = FillSimulator(
            asset_registry=self.asset_registry,
            commission_model=commission_model,
            slippage_model=slippage_model,
        )

        # Sell order (exit)
        order = Order(
            order_type=OrderType.MARKET,
            order_id="test_sell",
            asset_id="BTC",
            side=OrderSide.SELL,
            quantity=0.1,
        )

        fill_result = fill_simulator.try_fill_order(
            order=order,
            timestamp="2024-01-01T00:00:00Z",
            market_price=51000.0,  # Exit at higher price
            high=51500.0,
            low=50500.0,
            current_cash=10000.0,
            current_position=0.1,  # Have position to sell
        )

        # Sell orders get WORSE price: base_price * (1 - slippage)
        # Expected: 51000 * (1 - 0.0002) = 51000 * 0.9998 = 50989.80
        expected_fill_price = 51000.0 * 0.9998
        assert abs(fill_result.fill_price - expected_fill_price) < 0.01

        # Fees still calculated on order value
        order_value = 0.1 * expected_fill_price
        expected_fees = order_value * 0.0002
        assert abs(fill_result.commission - expected_fees) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
