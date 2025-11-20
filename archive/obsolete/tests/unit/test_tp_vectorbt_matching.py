"""Test TP (Take Profit) exit logic for VectorBT exact matching.

This test suite validates TASK-016: TP exit logic implementation matching VectorBT Pro.

Key Requirements from TASK-005:
1. TP triggers when high >= tp_level (intra-bar)
2. TP level calculated from BASE PRICE (close before slippage), NOT fill_price
3. TP exit price = tp_level * (1 - exit_slippage)
4. Exit slippage applied opposite to entry slippage

References:
- TASK-005: TP triggering logic documentation
- TASK-016: Implementation requirements
- docs/vectorbt_tp_logic.md: Detailed behavior specification
"""

import pytest
from datetime import datetime

from ml4t.backtest.core.event import FillEvent
from ml4t.backtest.core.types import OrderSide, OrderType
from ml4t.backtest.execution.bracket_manager import BracketOrderManager
from ml4t.backtest.execution.order import Order


class TestTPVectorBTMatching:
    """Test TP exit logic matches VectorBT Pro behavior exactly."""

    @pytest.fixture
    def bracket_manager(self):
        """Create bracket manager with mock submit callback."""
        submitted_orders = []

        def submit_callback(order):
            submitted_orders.append(order)

        manager = BracketOrderManager(submit_callback)
        manager.submitted_orders = submitted_orders  # For testing
        return manager

    def test_tp_from_base_price_not_fill_price(self, bracket_manager):
        """CRITICAL: TP calculated from base price, not fill_price.

        From TASK-005 empirical test:
        Close at entry:     $43,885.00  ← BASE PRICE
        Entry slippage:     +0.02%
        Entry price:        $43,893.78  (= $43,885 * 1.0002)

        TP = 2.5%:
        TP level:           $44,982.12  (= $43,885 * 1.025)  ← From CLOSE, not entry_price
        """
        base_price = 43885.0
        slippage_rate = 0.0002
        fill_price = base_price * (1 + slippage_rate)  # $43,893.78

        parent = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=1.0,
            tp_pct=0.025,  # 2.5% TP
            metadata={"base_price": base_price},  # ← CRITICAL
        )

        fill_event = FillEvent(
            timestamp=datetime(2024, 1, 8, 2, 40),
            order_id=parent.order_id,
            trade_id="trade_001",
            asset_id="BTC",
            side=OrderSide.BUY,
            fill_quantity=1.0,
            fill_price=fill_price,
            slippage=fill_price - base_price,
        )

        parent.filled_quantity = 1.0
        legs = bracket_manager.handle_bracket_fill(parent, fill_event)

        tp_order = next((o for o in legs if o.metadata.get("bracket_type") == "take_profit"), None)
        assert tp_order is not None

        # TP must be from base price, not fill price
        expected_tp = base_price * 1.025  # $44,982.12
        wrong_tp = fill_price * 1.025     # $45,090.62 (WRONG!)

        assert tp_order.limit_price == pytest.approx(expected_tp, abs=0.01), \
            f"TP level must be {expected_tp} (from base {base_price}), not {wrong_tp} (from fill {fill_price})"

    def test_tp_exact_values_from_task005(self, bracket_manager):
        """Test exact values from TASK-005 empirical verification.

        Test #1:
        Entry price:        $43,893.78
        Base price:         $43,885.00
        TP level:           $44,982.12  (= $43,885 * 1.025)
        TP exit price:      $44,973.13  (= $44,982.12 * 0.9998)
        """
        base_price = 43885.0
        fill_price = 43893.78
        tp_stop = 0.025
        exit_slippage_rate = 0.0002

        parent = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=1.0,
            tp_pct=tp_stop,
            metadata={"base_price": base_price},
        )

        fill_event = FillEvent(
            timestamp=datetime(2024, 1, 8, 2, 40),
            order_id=parent.order_id,
            trade_id="test_001",
            asset_id="BTC",
            side=OrderSide.BUY,
            fill_quantity=1.0,
            fill_price=fill_price,
            slippage=fill_price - base_price,
        )

        parent.filled_quantity = 1.0
        legs = bracket_manager.handle_bracket_fill(parent, fill_event)

        tp_order = next((o for o in legs if o.metadata.get("bracket_type") == "take_profit"), None)

        # Verify TP level (before exit slippage)
        expected_tp_level = base_price * (1 + tp_stop)  # $44,982.12
        assert tp_order.limit_price == pytest.approx(expected_tp_level, abs=0.01)

        # Note: Exit slippage applied by fill_simulator, not bracket_manager
        # Expected exit price after slippage: $44,982.12 * 0.9998 = $44,973.13

    def test_tp_test2_from_task005(self, bracket_manager):
        """Test #2 from TASK-005.

        Entry price:        $45,484.10
        Base price:         $45,475.00  (= $45,484.10 / 1.0002)
        TP level:           $46,611.88  (= $45,475 * 1.025)
        TP exit price:      $46,602.56  (= $46,611.88 * 0.9998)
        """
        base_price = 45475.0
        fill_price = 45484.10
        tp_stop = 0.025

        parent = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=1.0,
            tp_pct=tp_stop,
            metadata={"base_price": base_price},
        )

        fill_event = FillEvent(
            timestamp=datetime(2024, 1, 8, 8, 30),
            order_id=parent.order_id,
            trade_id="test_002",
            asset_id="BTC",
            side=OrderSide.BUY,
            fill_quantity=1.0,
            fill_price=fill_price,
            slippage=fill_price - base_price,
        )

        parent.filled_quantity = 1.0
        legs = bracket_manager.handle_bracket_fill(parent, fill_event)

        tp_order = next((o for o in legs if o.metadata.get("bracket_type") == "take_profit"), None)

        expected_tp = base_price * (1 + tp_stop)  # $46,611.88
        assert tp_order.limit_price == pytest.approx(expected_tp, abs=0.01)

    def test_tp_fallback_without_base_price(self, bracket_manager):
        """Test fallback when base_price not in metadata.

        BracketOrderManager should estimate base price by reversing slippage.
        This is less accurate but provides graceful degradation.
        """
        fill_price = 50010.0
        slippage_amount = 10.0  # $10 slippage
        expected_base = fill_price - slippage_amount  # $50,000

        parent = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=1.0,
            tp_pct=0.025,
            metadata={},  # No base_price provided
        )

        fill_event = FillEvent(
            timestamp=datetime(2024, 1, 1),
            order_id=parent.order_id,
            trade_id="test_fallback",
            asset_id="BTC",
            side=OrderSide.BUY,
            fill_quantity=1.0,
            fill_price=fill_price,
            slippage=slippage_amount,
        )

        parent.filled_quantity = 1.0
        legs = bracket_manager.handle_bracket_fill(parent, fill_event)

        tp_order = next((o for o in legs if o.metadata.get("bracket_type") == "take_profit"), None)

        # Should estimate base price and calculate TP from it
        expected_tp = expected_base * 1.025  # $51,250
        assert tp_order.limit_price == pytest.approx(expected_tp, abs=0.01)

    def test_tp_short_position(self, bracket_manager):
        """Test TP for SHORT position (TP is below entry).

        For SHORT:
        - TP level = base_price * (1 - tp_pct)
        - Exit is BUY to cover
        """
        base_price = 50000.0
        fill_price = base_price * (1 - 0.0002)  # Short sells slightly lower

        parent = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.SELL,  # Short
            quantity=1.0,
            tp_pct=0.03,  # 3% TP (3% below entry)
            metadata={"base_price": base_price},
        )

        fill_event = FillEvent(
            timestamp=datetime(2024, 1, 1),
            order_id=parent.order_id,
            trade_id="test_short",
            asset_id="BTC",
            side=OrderSide.SELL,
            fill_quantity=1.0,
            fill_price=fill_price,
            slippage=base_price - fill_price,
        )

        parent.filled_quantity = 1.0
        legs = bracket_manager.handle_bracket_fill(parent, fill_event)

        tp_order = next((o for o in legs if o.metadata.get("bracket_type") == "take_profit"), None)

        # For short: TP is BELOW entry
        expected_tp = base_price * (1 - 0.03)  # $48,500
        assert tp_order.limit_price == pytest.approx(expected_tp, abs=0.01)
        assert tp_order.side == OrderSide.BUY  # Cover the short

    def test_tp_with_zero_slippage(self, bracket_manager):
        """Test TP calculation when slippage is zero.

        When slippage = 0, base_price == fill_price.
        """
        price = 50000.0  # Same for both base and fill

        parent = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=1.0,
            tp_pct=0.025,
            metadata={"base_price": price},
        )

        fill_event = FillEvent(
            timestamp=datetime(2024, 1, 1),
            order_id=parent.order_id,
            trade_id="test_zero_slip",
            asset_id="BTC",
            side=OrderSide.BUY,
            fill_quantity=1.0,
            fill_price=price,
            slippage=0.0,
        )

        parent.filled_quantity = 1.0
        legs = bracket_manager.handle_bracket_fill(parent, fill_event)

        tp_order = next((o for o in legs if o.metadata.get("bracket_type") == "take_profit"), None)

        expected_tp = price * 1.025  # $51,250
        assert tp_order.limit_price == pytest.approx(expected_tp, abs=0.01)

    def test_tp_creates_limit_order(self, bracket_manager):
        """Test that TP creates a LIMIT order with correct parameters."""
        base_price = 50000.0

        parent = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=1.5,
            tp_pct=0.02,
            metadata={"base_price": base_price},
        )

        fill_event = FillEvent(
            timestamp=datetime(2024, 1, 1),
            order_id=parent.order_id,
            trade_id="test_limit",
            asset_id="BTC",
            side=OrderSide.BUY,
            fill_quantity=1.5,
            fill_price=base_price,
        )

        parent.filled_quantity = 1.5
        legs = bracket_manager.handle_bracket_fill(parent, fill_event)

        tp_order = next((o for o in legs if o.metadata.get("bracket_type") == "take_profit"), None)

        # Verify order properties
        assert tp_order.order_type == OrderType.LIMIT
        assert tp_order.side == OrderSide.SELL  # Exit side
        assert tp_order.quantity == 1.5  # Match parent quantity
        assert tp_order.asset_id == "BTC"
        assert tp_order.parent_order_id == parent.order_id
        assert tp_order.metadata.get("bracket_type") == "take_profit"

    def test_tp_precision(self, bracket_manager):
        """Test TP calculation maintains high precision.

        Critical for trades where fees > 50% of PnL (30 trades in Q1 2024).
        """
        base_price = 43885.0  # From TASK-005
        tp_pct = 0.025

        parent = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=1.0,
            tp_pct=tp_pct,
            metadata={"base_price": base_price},
        )

        fill_event = FillEvent(
            timestamp=datetime(2024, 1, 1),
            order_id=parent.order_id,
            trade_id="test_precision",
            asset_id="BTC",
            side=OrderSide.BUY,
            fill_quantity=1.0,
            fill_price=base_price * 1.0002,
        )

        parent.filled_quantity = 1.0
        legs = bracket_manager.handle_bracket_fill(parent, fill_event)

        tp_order = next((o for o in legs if o.metadata.get("bracket_type") == "take_profit"), None)

        # Calculate with high precision
        expected_tp = base_price * (1 + tp_pct)

        # Should match to at least 2 decimal places ($0.01)
        assert tp_order.limit_price == pytest.approx(expected_tp, abs=0.01)

        # For crypto, ideally match to 8 decimal places
        assert abs(tp_order.limit_price - expected_tp) < 1e-6

    def test_tp_oco_linking(self, bracket_manager):
        """Test that TP order is properly linked as OCO with other exits."""
        base_price = 50000.0

        parent = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=1.0,
            tp_pct=0.025,  # TP
            tsl_pct=0.01,  # TSL
            metadata={"base_price": base_price},
        )

        fill_event = FillEvent(
            timestamp=datetime(2024, 1, 1),
            order_id=parent.order_id,
            trade_id="test_oco",
            asset_id="BTC",
            side=OrderSide.BUY,
            fill_quantity=1.0,
            fill_price=base_price,
        )

        parent.filled_quantity = 1.0
        legs = bracket_manager.handle_bracket_fill(parent, fill_event)

        # Should create both TP and TSL
        assert len(legs) == 2

        tp_order = next((o for o in legs if o.metadata.get("bracket_type") == "take_profit"), None)
        tsl_order = next((o for o in legs if o.metadata.get("bracket_type") == "trailing_stop"), None)

        # Verify OCO linking (each order cancels the other)
        assert tsl_order.order_id in tp_order.child_order_ids
        assert tp_order.order_id in tsl_order.child_order_ids


class TestTPEdgeCases:
    """Test edge cases for TP logic."""

    @pytest.fixture
    def bracket_manager(self):
        """Create bracket manager."""
        submitted_orders = []
        manager = BracketOrderManager(lambda o: submitted_orders.append(o))
        manager.submitted_orders = submitted_orders
        return manager

    def test_tp_very_small_percentage(self, bracket_manager):
        """Test TP with very small percentage (0.1%)."""
        base_price = 50000.0

        parent = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=1.0,
            tp_pct=0.001,  # 0.1% (10 basis points)
            metadata={"base_price": base_price},
        )

        fill_event = FillEvent(
            timestamp=datetime(2024, 1, 1),
            order_id=parent.order_id,
            trade_id="small_tp",
            asset_id="BTC",
            side=OrderSide.BUY,
            fill_quantity=1.0,
            fill_price=base_price,
        )

        parent.filled_quantity = 1.0
        legs = bracket_manager.handle_bracket_fill(parent, fill_event)

        tp_order = next((o for o in legs if o.metadata.get("bracket_type") == "take_profit"), None)

        expected_tp = base_price * 1.001  # $50,050
        assert tp_order.limit_price == pytest.approx(expected_tp, abs=0.01)

    def test_tp_large_percentage(self, bracket_manager):
        """Test TP with large percentage (50%)."""
        base_price = 50000.0

        parent = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=1.0,
            tp_pct=0.50,  # 50%!
            metadata={"base_price": base_price},
        )

        fill_event = FillEvent(
            timestamp=datetime(2024, 1, 1),
            order_id=parent.order_id,
            trade_id="large_tp",
            asset_id="BTC",
            side=OrderSide.BUY,
            fill_quantity=1.0,
            fill_price=base_price,
        )

        parent.filled_quantity = 1.0
        legs = bracket_manager.handle_bracket_fill(parent, fill_event)

        tp_order = next((o for o in legs if o.metadata.get("bracket_type") == "take_profit"), None)

        expected_tp = base_price * 1.5  # $75,000
        assert tp_order.limit_price == pytest.approx(expected_tp, abs=0.01)
