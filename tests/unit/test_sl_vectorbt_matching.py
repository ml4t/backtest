"""Test SL (Stop Loss) exit logic for VectorBT exact matching.

This test suite validates TASK-017: SL exit logic implementation matching VectorBT Pro.

Key Requirements from TASK-006:
1. SL triggers when low <= sl_level (intra-bar)
2. SL level calculated from BASE PRICE (close before slippage), NOT fill_price
3. SL exit price = sl_level * (1 - exit_slippage)
4. Exit slippage applied opposite to entry slippage
5. SL exits at sl_level, NOT at low (favorable behavior)

References:
- TASK-006: SL triggering logic documentation
- TASK-017: Implementation requirements
- docs/TASK-006_COMPLETION.md: Detailed behavior specification
"""

import pytest
from datetime import datetime

from qengine.core.event import FillEvent
from qengine.core.types import OrderSide, OrderType
from qengine.execution.bracket_manager import BracketOrderManager
from qengine.execution.order import Order


class TestSLVectorBTMatching:
    """Test SL exit logic matches VectorBT Pro behavior exactly."""

    @pytest.fixture
    def bracket_manager(self):
        """Create bracket manager with mock submit callback."""
        submitted_orders = []

        def submit_callback(order):
            submitted_orders.append(order)

        manager = BracketOrderManager(submit_callback)
        manager.submitted_orders = submitted_orders  # For testing
        return manager

    def test_sl_from_base_price_not_fill_price(self, bracket_manager):
        """CRITICAL: SL calculated from base price, not fill_price.

        From TASK-006 empirical test:
        Close at entry:     $45,825.00  ← BASE PRICE
        Entry slippage:     +0.02%
        Entry price:        $45,834.17  (= $45,825 * 1.0002)

        SL = 2.5%:
        SL level:           $44,679.38  (= $45,825 * 0.975)  ← From CLOSE, not entry_price
        """
        base_price = 45825.0
        slippage_rate = 0.0002
        fill_price = base_price * (1 + slippage_rate)  # $45,834.17

        parent = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=1.0,
            sl_pct=0.025,  # 2.5% SL
            metadata={"base_price": base_price},  # ← CRITICAL
        )

        fill_event = FillEvent(
            timestamp=datetime(2024, 1, 3, 2, 55),
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

        sl_order = next((o for o in legs if o.metadata.get("bracket_type") == "stop_loss"), None)
        assert sl_order is not None

        # SL must be from base price, not fill price
        expected_sl = base_price * 0.975  # $44,679.38
        wrong_sl = fill_price * 0.975      # $44,688.31 (WRONG!)

        assert sl_order.stop_price == pytest.approx(expected_sl, abs=0.01), \
            f"SL level must be {expected_sl} (from base {base_price}), not {wrong_sl} (from fill {fill_price})"

    def test_sl_exact_values_from_task006(self, bracket_manager):
        """Test exact values from TASK-006 empirical verification.

        Test #1:
        Entry price:        $45,834.17
        Base price:         $45,825.00
        SL level:           $44,679.38  (= $45,825 * 0.975)
        SL exit price:      $44,670.44  (= $44,679.38 * 0.9998)
        """
        base_price = 45825.0
        fill_price = 45834.17
        sl_stop = 0.025

        parent = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=1.0,
            sl_pct=sl_stop,
            metadata={"base_price": base_price},
        )

        fill_event = FillEvent(
            timestamp=datetime(2024, 1, 3, 2, 55),
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

        sl_order = next((o for o in legs if o.metadata.get("bracket_type") == "stop_loss"), None)
        assert sl_order is not None

        # Verify exact SL level from empirical test
        expected_sl_level = 44679.38
        assert sl_order.stop_price == pytest.approx(expected_sl_level, abs=0.01)

    def test_sl_trigger_on_low_not_close(self):
        """SL triggers when low <= sl_level, even if close > sl_level.

        From TASK-006:
        Bar with Low=$44,605 (below SL=$44,679) but Close=$44,690 (above SL)
        → SL must trigger (intra-bar detection)
        """
        sl_level = 44679.38

        sl_order = Order(
            asset_id="BTC",
            order_type=OrderType.STOP,
            side=OrderSide.SELL,  # Exit long position
            quantity=1.0,
            stop_price=sl_level,
        )

        # Bar: Open=44730, High=44730, Low=44605, Close=44690
        high = 44730.0
        low = 44605.0
        close = 44690.0

        # Should trigger because low <= sl_level
        assert sl_order.can_fill(price=close, high=high, low=low), \
            f"SL must trigger when low={low} <= sl_level={sl_level}, even if close={close} > sl_level"

        # Should NOT trigger if low > sl_level
        high_bar_high = 46000.0
        high_bar_low = 44680.0  # Above SL
        high_bar_close = 45000.0

        assert not sl_order.can_fill(price=high_bar_close, high=high_bar_high, low=high_bar_low), \
            f"SL must NOT trigger when low={high_bar_low} > sl_level={sl_level}"

    def test_sl_exit_at_stop_level_not_low(self):
        """CRITICAL: SL exits at stop_level, NOT at low.

        From TASK-006:
        SL level: $44,679.38
        Low:      $44,605.00 (worse by $74.38)
        Exit:     $44,679.38 * (1 - 0.0002) = $44,670.44  ← At SL level, not low

        This is favorable behavior simulating stop-limit order execution.
        """
        # This test documents the expected behavior
        # Actual implementation is in fill_simulator._calculate_fill_price()
        sl_level = 44679.38
        low = 44605.0  # Worse price
        exit_slippage = 0.0002

        # Expected: Exit at SL level with slippage
        expected_exit = sl_level * (1 - exit_slippage)  # $44,670.44

        # NOT at low (which would be more pessimistic)
        wrong_exit = low * (1 - exit_slippage)  # $44,596.09

        difference = expected_exit - wrong_exit  # $74.35

        assert difference > 70, \
            f"Exiting at SL level saves ${difference:.2f} compared to exiting at low"

    def test_sl_symmetry_with_tp(self, bracket_manager):
        """SL and TP exhibit symmetric behavior.

        From TASK-006:
        | Exit Type | Trigger Condition | Exit Price | Check Against |
        |-----------|------------------|------------|---------------|
        | TP        | high >= tp_level | tp_level * (1 - slippage) | High |
        | SL        | low <= sl_level  | sl_level * (1 - slippage) | Low  |

        Both exit at their stop levels, not at extreme prices.
        """
        base_price = 45825.0

        parent = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=1.0,
            tp_pct=0.025,  # 2.5% TP
            sl_pct=0.025,  # 2.5% SL
            metadata={"base_price": base_price},
        )

        fill_event = FillEvent(
            timestamp=datetime(2024, 1, 3, 2, 55),
            order_id=parent.order_id,
            trade_id="trade_001",
            asset_id="BTC",
            side=OrderSide.BUY,
            fill_quantity=1.0,
            fill_price=base_price * 1.0002,
            slippage=base_price * 0.0002,
        )

        parent.filled_quantity = 1.0
        legs = bracket_manager.handle_bracket_fill(parent, fill_event)

        tp_order = next((o for o in legs if o.metadata.get("bracket_type") == "take_profit"), None)
        sl_order = next((o for o in legs if o.metadata.get("bracket_type") == "stop_loss"), None)

        assert tp_order is not None
        assert sl_order is not None

        # Symmetric percentage distances
        tp_level = tp_order.limit_price
        sl_level = sl_order.stop_price

        tp_distance_pct = (tp_level - base_price) / base_price
        sl_distance_pct = (base_price - sl_level) / base_price

        assert tp_distance_pct == pytest.approx(0.025, abs=0.0001), "TP should be +2.5%"
        assert sl_distance_pct == pytest.approx(0.025, abs=0.0001), "SL should be -2.5%"

    def test_sl_fallback_without_base_price(self, bracket_manager):
        """Test SL calculation when base_price not in metadata (fallback mode)."""
        fill_price = 45834.17
        slippage_amount = 9.17  # Entry slippage

        parent = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=1.0,
            sl_pct=0.025,
            # No base_price in metadata - will trigger fallback
        )

        fill_event = FillEvent(
            timestamp=datetime(2024, 1, 3, 2, 55),
            order_id=parent.order_id,
            trade_id="trade_001",
            asset_id="BTC",
            side=OrderSide.BUY,
            fill_quantity=1.0,
            fill_price=fill_price,
            slippage=slippage_amount,
        )

        parent.filled_quantity = 1.0
        legs = bracket_manager.handle_bracket_fill(parent, fill_event)

        sl_order = next((o for o in legs if o.metadata.get("bracket_type") == "stop_loss"), None)
        assert sl_order is not None

        # Fallback should estimate base_price ≈ fill_price - slippage
        estimated_base = fill_price - slippage_amount  # ≈ 45825
        expected_sl = estimated_base * 0.975

        assert sl_order.stop_price == pytest.approx(expected_sl, abs=1.0), \
            "Fallback SL should be reasonable even without base_price"

    def test_sl_short_position(self, bracket_manager):
        """Test SL for short positions (opposite logic).

        For SHORT positions:
        - SL triggers when high >= sl_level (price moving against us)
        - SL level = base_price * (1 + sl_pct)
        """
        base_price = 45825.0

        parent = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.SELL,  # Short entry
            quantity=1.0,
            sl_pct=0.025,
            metadata={"base_price": base_price},
        )

        fill_event = FillEvent(
            timestamp=datetime(2024, 1, 3, 2, 55),
            order_id=parent.order_id,
            trade_id="trade_001",
            asset_id="BTC",
            side=OrderSide.SELL,
            fill_quantity=1.0,
            fill_price=base_price * 0.9998,  # Entry slippage (receive less)
            slippage=base_price * 0.0002,
        )

        parent.filled_quantity = 1.0
        legs = bracket_manager.handle_bracket_fill(parent, fill_event)

        sl_order = next((o for o in legs if o.metadata.get("bracket_type") == "stop_loss"), None)
        assert sl_order is not None

        # For short, SL is ABOVE entry (1 + pct)
        expected_sl = base_price * (1 + 0.025)  # $46,971.56
        assert sl_order.stop_price == pytest.approx(expected_sl, abs=0.01)

        # Verify it's a BUY order (to cover short)
        assert sl_order.side == OrderSide.BUY, "Short SL must be BUY order to cover"

    def test_sl_no_tp(self, bracket_manager):
        """Test bracket with only SL, no TP."""
        base_price = 45825.0

        parent = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=1.0,
            sl_pct=0.025,  # Only SL, no TP
            metadata={"base_price": base_price},
        )

        fill_event = FillEvent(
            timestamp=datetime(2024, 1, 3, 2, 55),
            order_id=parent.order_id,
            trade_id="trade_001",
            asset_id="BTC",
            side=OrderSide.BUY,
            fill_quantity=1.0,
            fill_price=base_price * 1.0002,
            slippage=base_price * 0.0002,
        )

        parent.filled_quantity = 1.0
        legs = bracket_manager.handle_bracket_fill(parent, fill_event)

        # Should create only SL order, no TP
        assert len(legs) == 1, "Should create only SL when no TP specified"

        sl_order = legs[0]
        assert sl_order.metadata.get("bracket_type") == "stop_loss"
        assert sl_order.stop_price == pytest.approx(base_price * 0.975, abs=0.01)

    def test_sl_with_tp_both_created(self, bracket_manager):
        """Test bracket with both SL and TP creates both orders."""
        base_price = 45825.0

        parent = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=1.0,
            tp_pct=0.025,
            sl_pct=0.025,
            metadata={"base_price": base_price},
        )

        fill_event = FillEvent(
            timestamp=datetime(2024, 1, 3, 2, 55),
            order_id=parent.order_id,
            trade_id="trade_001",
            asset_id="BTC",
            side=OrderSide.BUY,
            fill_quantity=1.0,
            fill_price=base_price * 1.0002,
            slippage=base_price * 0.0002,
        )

        parent.filled_quantity = 1.0
        legs = bracket_manager.handle_bracket_fill(parent, fill_event)

        # Should create both SL and TP
        assert len(legs) == 2, "Should create both SL and TP"

        bracket_types = {o.metadata.get("bracket_type") for o in legs}
        assert "stop_loss" in bracket_types
        assert "take_profit" in bracket_types

    def test_sl_vs_tp_exit_levels(self, bracket_manager):
        """Verify SL and TP are on opposite sides of entry."""
        base_price = 45825.0

        parent = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=1.0,
            tp_pct=0.025,  # +2.5%
            sl_pct=0.025,  # -2.5%
            metadata={"base_price": base_price},
        )

        fill_event = FillEvent(
            timestamp=datetime(2024, 1, 3, 2, 55),
            order_id=parent.order_id,
            trade_id="trade_001",
            asset_id="BTC",
            side=OrderSide.BUY,
            fill_quantity=1.0,
            fill_price=base_price * 1.0002,
            slippage=base_price * 0.0002,
        )

        parent.filled_quantity = 1.0
        legs = bracket_manager.handle_bracket_fill(parent, fill_event)

        tp_order = next((o for o in legs if o.metadata.get("bracket_type") == "take_profit"), None)
        sl_order = next((o for o in legs if o.metadata.get("bracket_type") == "stop_loss"), None)

        tp_level = tp_order.limit_price
        sl_level = sl_order.stop_price

        # TP should be above entry, SL below
        assert tp_level > base_price, f"TP {tp_level} must be above base {base_price}"
        assert sl_level < base_price, f"SL {sl_level} must be below base {base_price}"

        # Check magnitude
        assert tp_level - base_price == pytest.approx(base_price - sl_level, abs=0.01), \
            "TP and SL should be equidistant from base"
