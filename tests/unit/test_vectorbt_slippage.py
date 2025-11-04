"""Unit tests for VectorBT-compatible slippage model.

Tests verify that qengine's VectorBTSlippage class implements VectorBT's
multiplicative slippage formula exactly, matching the specification in TASK-010.
"""

import pytest

from qengine.execution.order import Order, OrderSide, OrderType
from qengine.execution.slippage import VectorBTSlippage


class TestVectorBTSlippage:
    """Test VectorBTSlippage implementation matches VectorBT specification."""

    def test_initialization_valid(self):
        """Test VectorBTSlippage initializes with valid slippage values."""
        # Zero slippage (valid)
        model = VectorBTSlippage(slippage=0.0)
        assert model.slippage == 0.0

        # Typical slippage values
        model = VectorBTSlippage(slippage=0.0002)  # 0.02% = 2 bps
        assert model.slippage == 0.0002

        model = VectorBTSlippage(slippage=0.001)  # 0.1% = 10 bps
        assert model.slippage == 0.001

    def test_initialization_negative_raises(self):
        """Test VectorBTSlippage rejects negative slippage."""
        with pytest.raises(ValueError, match="Slippage must be non-negative"):
            VectorBTSlippage(slippage=-0.001)

    def test_buy_order_multiplicative_formula(self):
        """Test buy orders use multiplicative formula: price * (1 + slippage)."""
        slippage_model = VectorBTSlippage(slippage=0.0002)  # 0.02%

        # Create buy order
        order = Order(
            order_id="BUY001",
            asset_id="BTC-USDT",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=0.1,
        )

        # Test at $50,000
        market_price = 50000.0
        fill_price = slippage_model.calculate_fill_price(order, market_price)

        # Expected: 50000 * (1 + 0.0002) = 50000 * 1.0002 = 50010
        expected_price = 50000.0 * 1.0002
        assert fill_price == expected_price
        assert fill_price == 50010.0

    def test_sell_order_multiplicative_formula(self):
        """Test sell orders use multiplicative formula: price * (1 - slippage)."""
        slippage_model = VectorBTSlippage(slippage=0.0002)  # 0.02%

        # Create sell order
        order = Order(
            order_id="SELL001",
            asset_id="BTC-USDT",
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            quantity=0.1,
        )

        # Test at $50,000
        market_price = 50000.0
        fill_price = slippage_model.calculate_fill_price(order, market_price)

        # Expected: 50000 * (1 - 0.0002) = 50000 * 0.9998 = 49990
        expected_price = 50000.0 * 0.9998
        assert fill_price == expected_price
        assert fill_price == 49990.0

    def test_zero_slippage(self):
        """Test zero slippage returns market price unchanged."""
        slippage_model = VectorBTSlippage(slippage=0.0)

        # Buy order
        buy_order = Order(
            order_id="BUY001",
            asset_id="BTC-USDT",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=0.1,
        )

        # Sell order
        sell_order = Order(
            order_id="SELL001",
            asset_id="BTC-USDT",
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            quantity=0.1,
        )

        market_price = 50000.0

        # Both should return market price unchanged
        buy_fill = slippage_model.calculate_fill_price(buy_order, market_price)
        sell_fill = slippage_model.calculate_fill_price(sell_order, market_price)

        assert buy_fill == market_price
        assert sell_fill == market_price

    def test_slippage_cost_calculation(self):
        """Test slippage cost calculation returns absolute difference * quantity."""
        slippage_model = VectorBTSlippage(slippage=0.0002)

        # Buy order
        order = Order(
            order_id="BUY001",
            asset_id="BTC-USDT",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=0.1,
        )

        market_price = 50000.0
        fill_price = 50010.0  # After slippage
        fill_quantity = 0.1

        cost = slippage_model.calculate_slippage_cost(
            order, fill_quantity, market_price, fill_price
        )

        # Expected: abs(50010 - 50000) * 0.1 = 10 * 0.1 = 1.0
        assert cost == 1.0

    def test_task010_example1_long_entry(self):
        """Test TASK-010 Example 1: Long entry with slippage.

        From TASK-010 documentation:
        - 0.1 BTC at $50,000
        - 0.02% slippage
        - Result: Pay $50,010 (slippage adds $10)
        """
        slippage_model = VectorBTSlippage(slippage=0.0002)

        order = Order(
            order_id="ENTRY001",
            asset_id="BTC-USDT",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=0.1,
        )

        market_price = 50000.0
        fill_price = slippage_model.calculate_fill_price(order, market_price)

        # Verify fill price
        assert fill_price == 50010.0

        # Verify slippage cost
        cost = slippage_model.calculate_slippage_cost(
            order, 0.1, market_price, fill_price
        )
        assert cost == 1.0  # $10 slippage / 0.1 BTC = $1 per BTC... wait

        # Actually: abs(50010 - 50000) * 0.1 = 10 * 0.1 = 1.0
        # Total cost is $1.00, which is correct for 0.1 BTC
        # But the total notional slippage is 10 * 0.1 = 1.0
        # Let me recalculate: price diff is $10, quantity is 0.1 BTC
        # Total cost = $10 (price diff) * 0.1 (qty) = $1.00
        # Hmm, that seems wrong. Let me check the math.

        # Actually the slippage cost should be:
        # (fill_price - market_price) * quantity
        # = (50010 - 50000) * 0.1
        # = 10 * 0.1 = 1.0

        # But intuitively, buying 0.1 BTC with $10 slippage should cost $1.00 extra?
        # No wait: 0.1 BTC * $50,010 = $5,001 total cost
        # 0.1 BTC * $50,000 = $5,000 base cost
        # Slippage cost = $5,001 - $5,000 = $1.00

        # Yes! That's correct. The cost is $1.00 total.
        assert cost == 1.0

    def test_task010_example2_long_exit(self):
        """Test TASK-010 Example 2: Long exit with slippage.

        From TASK-010 documentation:
        - 0.1 BTC at $51,000
        - 0.02% slippage
        - Result: Receive $50,989.80 (slippage reduces by $10.20)
        """
        slippage_model = VectorBTSlippage(slippage=0.0002)

        order = Order(
            order_id="EXIT001",
            asset_id="BTC-USDT",
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            quantity=0.1,
        )

        market_price = 51000.0
        fill_price = slippage_model.calculate_fill_price(order, market_price)

        # Expected: 51000 * (1 - 0.0002) = 51000 * 0.9998 = 50989.80
        assert fill_price == pytest.approx(50989.80, rel=1e-9)

        # Verify slippage cost
        cost = slippage_model.calculate_slippage_cost(
            order, 0.1, market_price, fill_price
        )

        # Expected: abs(50989.80 - 51000) * 0.1 = 10.20 * 0.1 = 1.02
        assert cost == pytest.approx(1.02, rel=1e-9)

    def test_task010_example3_round_trip(self):
        """Test TASK-010 Example 3: Round-trip slippage cost.

        From TASK-010 documentation:
        - Entry at $50,000, exit at $51,000
        - Gross PnL: $100
        - Net PnL: $97.98
        - Slippage cost: $2.02 (2% of profit!)
        """
        slippage_model = VectorBTSlippage(slippage=0.0002)

        # Entry (buy)
        entry_order = Order(
            order_id="ENTRY001",
            asset_id="BTC-USDT",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=0.1,
        )

        entry_market_price = 50000.0
        entry_fill_price = slippage_model.calculate_fill_price(entry_order, entry_market_price)
        assert entry_fill_price == 50010.0

        entry_cost = slippage_model.calculate_slippage_cost(
            entry_order, 0.1, entry_market_price, entry_fill_price
        )
        assert entry_cost == 1.0

        # Exit (sell)
        exit_order = Order(
            order_id="EXIT001",
            asset_id="BTC-USDT",
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            quantity=0.1,
        )

        exit_market_price = 51000.0
        exit_fill_price = slippage_model.calculate_fill_price(exit_order, exit_market_price)
        assert exit_fill_price == pytest.approx(50989.80, rel=1e-9)

        exit_cost = slippage_model.calculate_slippage_cost(
            exit_order, 0.1, exit_market_price, exit_fill_price
        )
        assert exit_cost == pytest.approx(1.02, rel=1e-9)

        # Total slippage cost
        total_slippage_cost = entry_cost + exit_cost
        assert total_slippage_cost == pytest.approx(2.02, rel=1e-9)

        # Verify PnL calculation
        # Buy at 50010, sell at 50989.80, quantity 0.1
        # PnL = (50989.80 - 50010) * 0.1 = 979.80 * 0.1 = 97.98
        pnl = (exit_fill_price - entry_fill_price) * 0.1
        assert pnl == pytest.approx(97.98, rel=1e-9)

        # Gross PnL (without slippage): (51000 - 50000) * 0.1 = 100
        gross_pnl = (exit_market_price - entry_market_price) * 0.1
        assert gross_pnl == 100.0

        # Net PnL should be gross - slippage
        net_pnl = gross_pnl - total_slippage_cost
        assert net_pnl == pytest.approx(97.98, rel=1e-9)

    def test_large_slippage_example(self):
        """Test TASK-010 Example 4: Large slippage in illiquid market.

        From TASK-010 documentation:
        - 0.5% slippage
        - Round-trip cost: $50 on flat market
        - Loss: 1% even with no price movement
        """
        slippage_model = VectorBTSlippage(slippage=0.005)  # 0.5%

        market_price = 50000.0

        # Entry (buy)
        entry_order = Order(
            order_id="ENTRY001",
            asset_id="BTC-USDT",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=0.1,
        )

        entry_fill_price = slippage_model.calculate_fill_price(entry_order, market_price)
        # Expected: 50000 * (1 + 0.005) = 50000 * 1.005 = 50250
        assert entry_fill_price == pytest.approx(50250.0, rel=1e-9)

        entry_cost = slippage_model.calculate_slippage_cost(
            entry_order, 0.1, market_price, entry_fill_price
        )
        # Expected: abs(50250 - 50000) * 0.1 = 250 * 0.1 = 25.0
        assert entry_cost == pytest.approx(25.0, rel=1e-9)

        # Exit (sell) at same price
        exit_order = Order(
            order_id="EXIT001",
            asset_id="BTC-USDT",
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            quantity=0.1,
        )

        exit_fill_price = slippage_model.calculate_fill_price(exit_order, market_price)
        # Expected: 50000 * (1 - 0.005) = 50000 * 0.995 = 49750
        assert exit_fill_price == 49750.0

        exit_cost = slippage_model.calculate_slippage_cost(
            exit_order, 0.1, market_price, exit_fill_price
        )
        # Expected: abs(49750 - 50000) * 0.1 = 250 * 0.1 = 25.0
        assert exit_cost == pytest.approx(25.0, rel=1e-9)

        # Total slippage cost
        total_slippage_cost = entry_cost + exit_cost
        assert total_slippage_cost == pytest.approx(50.0, rel=1e-9)

        # Verify 1% loss even with flat market
        # Buy at 50250, sell at 49750, quantity 0.1
        # Loss = (49750 - 50250) * 0.1 = -500 * 0.1 = -50
        pnl = (exit_fill_price - entry_fill_price) * 0.1
        assert pnl == pytest.approx(-50.0, rel=1e-9)

        # Loss percentage: -50 / (50000 * 0.1) = -50 / 5000 = -1%
        loss_pct = pnl / (market_price * 0.1)
        assert loss_pct == pytest.approx(-0.01, rel=1e-9)

    def test_multiplicative_vs_additive_difference(self):
        """Test that multiplicative formula differs from additive at high prices.

        This test demonstrates why VectorBT uses multiplicative slippage.
        """
        slippage_model = VectorBTSlippage(slippage=0.0002)

        order = Order(
            order_id="BUY001",
            asset_id="BTC-USDT",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=0.1,
        )

        # Low price
        low_price = 1000.0
        low_fill = slippage_model.calculate_fill_price(order, low_price)
        # Multiplicative: 1000 * (1 + 0.0002) = 1000.20
        assert low_fill == pytest.approx(1000.20, rel=1e-9)

        # High price
        high_price = 100000.0
        high_fill = slippage_model.calculate_fill_price(order, high_price)
        # Multiplicative: 100000 * (1 + 0.0002) = 100020.0
        assert high_fill == pytest.approx(100020.0, rel=1e-9)

        # Verify slippage scales with price (proportional)
        low_slippage_amount = low_fill - low_price
        high_slippage_amount = high_fill - high_price

        # Ratio should match price ratio
        price_ratio = high_price / low_price
        slippage_ratio = high_slippage_amount / low_slippage_amount

        assert price_ratio == pytest.approx(slippage_ratio, rel=1e-9)
        assert slippage_ratio == pytest.approx(100.0, rel=1e-9)

    def test_different_slippage_rates(self):
        """Test various slippage rates match expected values."""
        test_cases = [
            (0.0001, 50000.0, OrderSide.BUY, 50005.0),  # 1 bp
            (0.0002, 50000.0, OrderSide.BUY, 50010.0),  # 2 bp
            (0.001, 50000.0, OrderSide.BUY, 50050.0),  # 10 bp
            (0.0001, 50000.0, OrderSide.SELL, 49995.0),  # 1 bp sell
            (0.0002, 50000.0, OrderSide.SELL, 49990.0),  # 2 bp sell
            (0.001, 50000.0, OrderSide.SELL, 49950.0),  # 10 bp sell
        ]

        for slippage_rate, market_price, side, expected_fill in test_cases:
            slippage_model = VectorBTSlippage(slippage=slippage_rate)
            order = Order(
                order_id="TEST001",
                asset_id="BTC-USDT",
                order_type=OrderType.MARKET,
                side=side,
                quantity=1.0,
                )

            fill_price = slippage_model.calculate_fill_price(order, market_price)
            assert fill_price == pytest.approx(expected_fill, rel=1e-9), (
                f"Slippage {slippage_rate}, price {market_price}, "
                f"side {side}: expected {expected_fill}, got {fill_price}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
