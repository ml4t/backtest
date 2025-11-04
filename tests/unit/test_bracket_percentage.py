"""Test percentage-based bracket orders (VectorBT compatibility)."""

import pytest
from datetime import datetime

from qengine.core.event import FillEvent
from qengine.core.types import OrderSide, OrderType
from qengine.execution.bracket_manager import BracketOrderManager
from qengine.execution.order import Order


class TestPercentageBracketOrders:
    """Test percentage-based TP/SL/TSL bracket orders."""

    @pytest.fixture
    def bracket_manager(self):
        """Create bracket manager with mock submit callback."""
        submitted_orders = []

        def submit_callback(order):
            submitted_orders.append(order)

        manager = BracketOrderManager(submit_callback)
        manager.submitted_orders = submitted_orders  # For testing
        return manager

    def test_percentage_tp_and_tsl_long(self, bracket_manager):
        """Test TP and TSL percentage-based bracket (LONG position).

        CRITICAL: VectorBT calculates TP/TSL from BASE PRICE (close before slippage),
        not from fill_price. This test verifies correct implementation.
        """
        # Base price (market close): $50,000
        # Entry slippage: +0.02% → Fill price: $50,010
        base_price = 50000.0
        fill_price = base_price * (1 + 0.0002)  # $50,010

        # Create parent order: BUY with TP=2.5%, TSL=1%
        parent = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=1.0,
            tp_pct=0.025,  # 2.5% take profit
            tsl_pct=0.01,   # 1% trailing stop
            metadata={"base_price": base_price},  # ← CRITICAL for VectorBT compatibility
        )

        # Simulate fill with slippage
        fill_event = FillEvent(
            timestamp=datetime(2024, 1, 1),
            order_id=parent.order_id,
            trade_id="trade_001",
            asset_id="BTC",
            side=OrderSide.BUY,
            fill_quantity=1.0,
            fill_price=fill_price,  # $50,010 (includes slippage)
            commission=10.0,
            slippage=fill_price - base_price,  # $10
        )

        # Create bracket legs
        parent.filled_quantity = 1.0
        legs = bracket_manager.handle_bracket_fill(parent, fill_event)

        # Should create 2 legs: TP + TSL
        assert len(legs) == 2

        # Find TP and TSL orders
        tp_order = next((o for o in legs if o.metadata.get("bracket_type") == "take_profit"), None)
        tsl_order = next((o for o in legs if o.metadata.get("bracket_type") == "trailing_stop"), None)

        assert tp_order is not None, "TP order should be created"
        assert tsl_order is not None, "TSL order should be created"

        # Verify TP order (calculated from BASE PRICE, not fill_price)
        assert tp_order.order_type == OrderType.LIMIT
        assert tp_order.side == OrderSide.SELL  # Exit side for long
        assert tp_order.quantity == 1.0
        expected_tp = base_price * 1.025  # $51,250 (from base, NOT from $50,010)
        assert tp_order.limit_price == pytest.approx(expected_tp), \
            f"TP should be {expected_tp} (from base_price {base_price}), not {fill_price * 1.025}"

        # Verify TSL order
        assert tsl_order.order_type == OrderType.TRAILING_STOP
        assert tsl_order.side == OrderSide.SELL  # Exit side for long
        assert tsl_order.quantity == 1.0
        assert tsl_order.trail_percent == 0.01  # 1%

        # Verify OCO linking
        assert tsl_order.order_id in tp_order.child_order_ids
        assert tp_order.order_id in tsl_order.child_order_ids

    def test_percentage_tp_and_sl_long(self, bracket_manager):
        """Test TP and SL percentage-based bracket (LONG position)."""
        # Create parent order: BUY with TP=2%, SL=1.5%
        parent = Order(
            asset_id="ETH",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=10.0,
            tp_pct=0.02,   # 2% take profit
            sl_pct=0.015,  # 1.5% stop loss
        )

        # Simulate fill at $3,000
        fill_event = FillEvent(
            timestamp=datetime(2024, 1, 1),
            order_id=parent.order_id,
            trade_id="trade_002",
            asset_id="ETH",
            side=OrderSide.BUY,
            fill_quantity=10.0,
            fill_price=3000.0,
            commission=6.0,
        )

        # Create bracket legs
        parent.filled_quantity = 10.0
        legs = bracket_manager.handle_bracket_fill(parent, fill_event)

        # Should create 2 legs: TP + SL
        assert len(legs) == 2

        # Find TP and SL orders
        tp_order = next((o for o in legs if o.metadata.get("bracket_type") == "take_profit"), None)
        sl_order = next((o for o in legs if o.metadata.get("bracket_type") == "stop_loss"), None)

        assert tp_order is not None
        assert sl_order is not None

        # Verify TP order
        assert tp_order.limit_price == pytest.approx(3000.0 * 1.02)  # $3,060

        # Verify SL order
        assert sl_order.order_type == OrderType.STOP
        assert sl_order.stop_price == pytest.approx(3000.0 * 0.985)  # $2,955

    def test_percentage_short_position(self, bracket_manager):
        """Test percentage-based bracket for SHORT position."""
        # Create parent order: SELL (short) with TP=3%, SL=2%
        parent = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.SELL,
            quantity=0.5,
            tp_pct=0.03,  # 3% take profit
            sl_pct=0.02,  # 2% stop loss
        )

        # Simulate fill at $50,000
        fill_event = FillEvent(
            timestamp=datetime(2024, 1, 1),
            order_id=parent.order_id,
            trade_id="trade_003",
            asset_id="BTC",
            side=OrderSide.SELL,
            fill_quantity=0.5,
            fill_price=50000.0,
            commission=5.0,
        )

        # Create bracket legs
        parent.filled_quantity = 0.5
        legs = bracket_manager.handle_bracket_fill(parent, fill_event)

        assert len(legs) == 2

        tp_order = next((o for o in legs if o.metadata.get("bracket_type") == "take_profit"), None)
        sl_order = next((o for o in legs if o.metadata.get("bracket_type") == "stop_loss"), None)

        # For SHORT: TP is below entry, SL is above entry
        assert tp_order.limit_price == pytest.approx(50000.0 * 0.97)  # $48,500 (3% below)
        assert sl_order.stop_price == pytest.approx(50000.0 * 1.02)   # $51,000 (2% above)

        # Both should be BUY orders (covering the short)
        assert tp_order.side == OrderSide.BUY
        assert sl_order.side == OrderSide.BUY

    def test_absolute_prices_still_work(self, bracket_manager):
        """Test that absolute price brackets still work (backward compatibility)."""
        # Create parent order with absolute prices
        parent = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=1.0,
            profit_target=52000.0,  # Absolute TP
            stop_loss=48000.0,       # Absolute SL
        )

        fill_event = FillEvent(
            timestamp=datetime(2024, 1, 1),
            order_id=parent.order_id,
            trade_id="trade_004",
            asset_id="BTC",
            side=OrderSide.BUY,
            fill_quantity=1.0,
            fill_price=50000.0,
            commission=10.0,
        )

        parent.filled_quantity = 1.0
        legs = bracket_manager.handle_bracket_fill(parent, fill_event)

        assert len(legs) == 2

        tp_order = next((o for o in legs if o.metadata.get("bracket_type") == "take_profit"), None)
        sl_order = next((o for o in legs if o.metadata.get("bracket_type") == "stop_loss"), None)

        # Should use absolute prices, not calculate from fill
        assert tp_order.limit_price == 52000.0
        assert sl_order.stop_price == 48000.0
