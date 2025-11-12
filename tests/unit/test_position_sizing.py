"""Unit tests for position sizing models.

Tests cover VectorBT's size=np.inf behavior with various scenarios:
- Basic position sizing
- Fee impact on buying power
- Slippage reducing position size
- Granularity creating unused cash
- Insufficient cash scenarios

Test cases derived from TASK-011 numerical examples.
"""

import pytest
from qengine.execution.position_sizer import (
    FixedQuantitySizer,
    VectorBTInfiniteSizer,
    PercentageOfEquitySizer,
)
from qengine.execution.commission import (
    NoCommission,
    PercentageCommission,
    VectorBTCommission,
)
from qengine.execution.slippage import (
    NoSlippage,
    VectorBTSlippage,
)
from qengine.execution.order import Order
from qengine.core.types import OrderType, OrderSide


def create_test_order():
    """Create a test order for position sizing."""
    return Order(
        asset_id="BTC",
        order_type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=1.0,  # Will be overridden by sizer
    )


class TestFixedQuantitySizer:
    """Test fixed quantity position sizer."""

    def test_returns_fixed_quantity(self):
        """Fixed sizer always returns same quantity."""
        sizer = FixedQuantitySizer(quantity=0.5)
        order = create_test_order()

        quantity = sizer.calculate_quantity(
            price=50000.0,
            available_cash=10000.0,
            commission_model=NoCommission(),
            slippage_model=NoSlippage(),
            order=order,
        )

        assert quantity == 0.5

    def test_rejects_negative_quantity(self):
        """Cannot create sizer with negative quantity."""
        with pytest.raises(ValueError, match="must be positive"):
            FixedQuantitySizer(quantity=-1.0)

    def test_rejects_zero_quantity(self):
        """Cannot create sizer with zero quantity."""
        with pytest.raises(ValueError, match="must be positive"):
            FixedQuantitySizer(quantity=0.0)


class TestVectorBTInfiniteSizer:
    """Test VectorBT size=np.inf position sizer.

    Test cases match TASK-011 numerical examples.
    """

    def test_basic_size_inf_example_1(self):
        """Example 1: Basic size=np.inf with slippage and fees.

        Configuration:
        - Cash: $10,000
        - Price: $50,000
        - Slippage: 0.02% (0.0002)
        - Fees: 0.02% (0.0002)
        - Granularity: 0.001 BTC

        Expected:
        - Adjusted price: $50,010
        - Max order value: $9,998.00
        - Max size: 0.19988 BTC
        - Final size (rounded): 0.199 BTC
        - Total cost: $9,953.98
        - Unused cash: $46.02
        """
        sizer = VectorBTInfiniteSizer(granularity=0.001)
        order = create_test_order()

        commission = PercentageCommission(rate=0.0002)
        slippage = VectorBTSlippage(slippage=0.0002)

        quantity = sizer.calculate_quantity(
            price=50000.0,
            available_cash=10000.0,
            commission_model=commission,
            slippage_model=slippage,
            order=order,
        )

        # Verify quantity
        assert quantity == pytest.approx(0.199, abs=1e-6)

        # Verify total cost doesn't exceed cash
        adj_price = 50000.0 * 1.0002
        order_value = quantity * adj_price
        fees = order_value * 0.0002
        total_cost = order_value + fees
        unused_cash = 10000.0 - total_cost

        assert total_cost <= 10000.0
        assert unused_cash >= 0
        assert unused_cash == pytest.approx(46.02, abs=0.01)

    def test_with_fixed_fees_example_2(self):
        """Example 2: size=np.inf with fixed fees.

        Configuration:
        - Cash: $10,000
        - Price: $50,000
        - Slippage: 0.02%
        - Fixed fees: $10
        - Percentage fees: 0.02%
        - Granularity: 0.001 BTC

        Expected:
        - Same size: 0.199 BTC (granularity dominates)
        - Total cost: $9,963.98
        - Unused cash: $36.02 (less due to fixed fee)
        """
        sizer = VectorBTInfiniteSizer(granularity=0.001)
        order = create_test_order()

        commission = VectorBTCommission(fee_rate=0.0002, fixed_fee=10.0)
        slippage = VectorBTSlippage(slippage=0.0002)

        quantity = sizer.calculate_quantity(
            price=50000.0,
            available_cash=10000.0,
            commission_model=commission,
            slippage_model=slippage,
            order=order,
        )

        # Quantity should be same (granularity rounds down similarly)
        assert quantity == pytest.approx(0.199, abs=1e-6)

        # Verify cost calculation
        adj_price = 50000.0 * 1.0002
        order_value = quantity * adj_price
        fees = order_value * 0.0002 + 10.0
        total_cost = order_value + fees

        assert total_cost <= 10000.0
        assert total_cost == pytest.approx(9963.98, abs=0.01)

    def test_insufficient_cash_example_3(self):
        """Example 3: Insufficient cash to cover fees.

        Configuration:
        - Cash: $5
        - Fixed fees: $10
        - Result: Order rejected
        """
        sizer = VectorBTInfiniteSizer(granularity=0.001)
        order = create_test_order()

        commission = VectorBTCommission(fee_rate=0.0002, fixed_fee=10.0)
        slippage = NoSlippage()

        with pytest.raises(ValueError, match="Insufficient cash to cover fees"):
            sizer.calculate_quantity(
                price=50000.0,
                available_cash=5.0,
                commission_model=commission,
                slippage_model=slippage,
                order=order,
            )

    def test_large_cash_small_granularity_example_4(self):
        """Example 4: Large cash, precise granularity.

        Configuration:
        - Cash: $100,000
        - Price: $50,000
        - Granularity: 0.00000001 BTC (1 satoshi)
        - Fees: 0.02%
        - Slippage: 0.02%

        Expected:
        - Max size: ~1.9988 BTC
        - Very small unused cash (~$0.01)
        """
        sizer = VectorBTInfiniteSizer(granularity=0.00000001)
        order = create_test_order()

        commission = PercentageCommission(rate=0.0002)
        slippage = VectorBTSlippage(slippage=0.0002)

        quantity = sizer.calculate_quantity(
            price=50000.0,
            available_cash=100000.0,
            commission_model=commission,
            slippage_model=slippage,
            order=order,
        )

        # Should be very close to theoretical max (1.9988)
        expected_max = 100000.0 / (50000.0 * 1.0002 * 1.0002)
        assert quantity == pytest.approx(expected_max, abs=0.001)

        # Verify minimal unused cash
        adj_price = 50000.0 * 1.0002
        total_cost = quantity * adj_price * 1.0002
        unused_cash = 100000.0 - total_cost
        assert unused_cash >= 0
        assert unused_cash < 1.0  # Less than $1 unused

    def test_no_costs_example_5(self):
        """Example 5: Perfect allocation with no costs.

        Configuration:
        - Cash: $10,000
        - Price: $50,000
        - No slippage, no fees, fine granularity
        - Expected: Exactly 0.2 BTC
        """
        sizer = VectorBTInfiniteSizer(granularity=0.00000001)
        order = create_test_order()

        commission = NoCommission()
        slippage = NoSlippage()

        quantity = sizer.calculate_quantity(
            price=50000.0,
            available_cash=10000.0,
            commission_model=commission,
            slippage_model=slippage,
            order=order,
        )

        # Should be exactly 0.2 BTC (or very close due to rounding)
        assert quantity == pytest.approx(0.2, abs=1e-6)

        # Verify perfect allocation
        total_cost = quantity * 50000.0
        assert total_cost == pytest.approx(10000.0, abs=0.01)

    def test_granularity_rounds_down(self):
        """Granularity always rounds down, never up."""
        sizer = VectorBTInfiniteSizer(granularity=0.01)
        order = create_test_order()

        commission = NoCommission()
        slippage = NoSlippage()

        # Price chosen so max_size = 0.2049 (should round to 0.20, not 0.21)
        quantity = sizer.calculate_quantity(
            price=4878.0,
            available_cash=1000.0,
            commission_model=commission,
            slippage_model=slippage,
            order=order,
        )

        # Should round down to 0.20
        expected = 0.20
        assert quantity == pytest.approx(expected, abs=1e-6)

        # Verify it's not rounded up
        assert quantity < 0.21

    def test_rejects_zero_granularity(self):
        """Cannot create sizer with zero granularity."""
        with pytest.raises(ValueError, match="must be positive"):
            VectorBTInfiniteSizer(granularity=0.0)

    def test_rejects_negative_granularity(self):
        """Cannot create sizer with negative granularity."""
        with pytest.raises(ValueError, match="must be positive"):
            VectorBTInfiniteSizer(granularity=-0.001)

    def test_slippage_reduces_position_size(self):
        """Higher slippage reduces affordable position size."""
        order = create_test_order()
        commission = NoCommission()

        # No slippage case
        sizer_no_slip = VectorBTInfiniteSizer(granularity=0.001)
        qty_no_slip = sizer_no_slip.calculate_quantity(
            price=50000.0,
            available_cash=10000.0,
            commission_model=commission,
            slippage_model=NoSlippage(),
            order=order,
        )

        # With 1% slippage
        sizer_with_slip = VectorBTInfiniteSizer(granularity=0.001)
        slippage = VectorBTSlippage(slippage=0.01)
        qty_with_slip = sizer_with_slip.calculate_quantity(
            price=50000.0,
            available_cash=10000.0,
            commission_model=commission,
            slippage_model=slippage,
            order=order,
        )

        # With slippage should be smaller
        assert qty_with_slip < qty_no_slip

    def test_fees_reduce_position_size(self):
        """Higher fees reduce affordable position size."""
        order = create_test_order()
        slippage = NoSlippage()

        # No fees
        sizer_no_fees = VectorBTInfiniteSizer(granularity=0.001)
        qty_no_fees = sizer_no_fees.calculate_quantity(
            price=50000.0,
            available_cash=10000.0,
            commission_model=NoCommission(),
            slippage_model=slippage,
            order=order,
        )

        # With 0.5% fees
        sizer_with_fees = VectorBTInfiniteSizer(granularity=0.001)
        commission = PercentageCommission(rate=0.005)
        qty_with_fees = sizer_with_fees.calculate_quantity(
            price=50000.0,
            available_cash=10000.0,
            commission_model=commission,
            slippage_model=slippage,
            order=order,
        )

        # With fees should be smaller
        assert qty_with_fees < qty_no_fees


class TestPercentageOfEquitySizer:
    """Test percentage of equity position sizer."""

    def test_calculates_percentage_correctly(self):
        """Sizer allocates correct percentage of equity."""
        sizer = PercentageOfEquitySizer(percentage=0.1)  # 10%
        order = create_test_order()

        quantity = sizer.calculate_quantity(
            price=50000.0,
            available_cash=100000.0,
            commission_model=NoCommission(),
            slippage_model=NoSlippage(),
            order=order,
        )

        # 10% of $100k = $10k, at $50k/BTC = 0.2 BTC
        assert quantity == pytest.approx(0.2, abs=1e-6)

    def test_accounts_for_fees(self):
        """Percentage sizer accounts for fees."""
        sizer = PercentageOfEquitySizer(percentage=0.1)
        order = create_test_order()

        quantity = sizer.calculate_quantity(
            price=50000.0,
            available_cash=100000.0,
            commission_model=PercentageCommission(rate=0.001),  # 0.1%
            slippage_model=NoSlippage(),
            order=order,
        )

        # Should be slightly less than 0.2 due to fees
        assert quantity < 0.2
        assert quantity == pytest.approx(0.1998, abs=0.001)

    def test_rejects_invalid_percentage(self):
        """Cannot create sizer with invalid percentage."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            PercentageOfEquitySizer(percentage=0.0)

        with pytest.raises(ValueError, match="between 0 and 1"):
            PercentageOfEquitySizer(percentage=-0.1)

        with pytest.raises(ValueError, match="between 0 and 1"):
            PercentageOfEquitySizer(percentage=1.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
