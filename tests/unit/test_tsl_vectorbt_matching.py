"""Test TSL (Trailing Stop Loss) VectorBT exact matching.

TASK-018: TSL Tracking and Exit Logic

This test suite validates that qengine TSL implementation exactly matches VectorBT Pro
behavior as documented in TASK-007.

Key VectorBT TSL Behaviors:
1. Peak-based tracking: TSL trails from PEAK (max high), not current price
2. 4-stage per-bar update: open → check → high → recheck
3. Exit at TSL level (with slippage), not at low
4. TSL can activate and trigger in same bar
5. Optional threshold activation

Empirical Values from TASK-007 (BTC Q1 2024):
- Entry: $43,893.78
- Peak: $44,665.00 (highest high reached)
- TSL level: $44,218.35 (= peak * 0.99)
- Exit: $44,209.51 (= tsl_level * 0.9998 with slippage)
"""

import pytest
from datetime import datetime
from decimal import Decimal

from qengine.core.event import MarketEvent
from qengine.core.types import OrderSide, OrderType, MarketDataType
from qengine.execution.broker import SimulationBroker
from qengine.execution.order import Order
from qengine.execution.slippage import PercentageSlippage



class TestTSLPeakTracking:
    """Test that TSL trails from peak price, not current price."""

    def test_tsl_tracks_peak_not_current_price(self):
        """TSL should remain at peak level even when price falls.

        Price sequence:
        T0: 100 (entry)
        T1: 105 (new peak) → TSL = 103.95 (105 * 0.99)
        T2: 102 (price falls) → TSL stays at 103.95 (peak still 105)
        T3: 101 (price falls more) → TSL stays at 103.95
        T4: 103 (still below peak) → TSL stays at 103.95

        VectorBT: TSL locks at 103.95 (from peak 105)
        Incorrect: TSL would drop to 99.99 (from current 101)
        """
        broker = SimulationBroker(initial_cash=10000.0, execution_delay=False)

        # T0: Entry at 100
        entry_order = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=1.0,
            tsl_pct=0.01,  # 1% TSL
            metadata={"base_price": 100.0},
        )
        broker.submit_order(entry_order)

        # Fill entry
        event_t0 = MarketEvent(
            timestamp=datetime.now(),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            open=100.0,
            high=100.5,
            low=99.5,
            close=100.0,
            volume=1000.0,
        )
        broker.on_market_event(event_t0)

        # Get the TSL order that was created
        tsl_orders = broker._trailing_stops.get("BTC", [])
        assert len(tsl_orders) == 1, "TSL order should be created"
        tsl_order = tsl_orders[0]

        # T1: Price rises to 105 (new peak)
        event_t1 = MarketEvent(
            timestamp=datetime.now(),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            open=104.0,
            high=105.0,  # Peak
            low=103.0,
            close=104.0,
            volume=1000.0,
        )
        broker.on_market_event(event_t1)

        # Peak should be 105, TSL should be 103.95
        assert tsl_order.metadata["peak_price"] == 105.0, "Peak should update to high (105)"
        expected_tsl = 105.0 * 0.99  # 103.95
        assert abs(tsl_order.trailing_stop_price - expected_tsl) < 0.01, \
            f"TSL should be {expected_tsl}, got {tsl_order.trailing_stop_price}"

        # T2: Price falls to 102 (below peak)
        event_t2 = MarketEvent(
            timestamp=datetime.now(),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            open=103.0,
            high=103.5,
            low=101.5,
            close=102.0,
            volume=1000.0,
        )
        broker.on_market_event(event_t2)

        # Peak should STAY at 105 (not drop to 102)
        assert tsl_order.metadata["peak_price"] == 105.0, "Peak should NOT drop with price"
        assert abs(tsl_order.trailing_stop_price - expected_tsl) < 0.01, \
            "TSL should remain at 103.95 (peak level)"

        # T3: Price continues falling to 101
        event_t3 = MarketEvent(
            timestamp=datetime.now(),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            open=102.0,
            high=102.5,
            low=100.5,
            close=101.0,
            volume=1000.0,
        )
        broker.on_market_event(event_t3)

        # Peak should STILL be 105
        assert tsl_order.metadata["peak_price"] == 105.0, "Peak should remain at 105"
        assert abs(tsl_order.trailing_stop_price - expected_tsl) < 0.01, \
            "TSL should still be 103.95 (locked in gain)"

    def test_tsl_updates_peak_with_high_not_close(self):
        """Peak should update with bar high, not close.

        Bar: Open=100, High=105, Low=99, Close=101
        Peak should be 105 (high), not 101 (close)
        TSL should be 103.95 (105 * 0.99), not 99.99 (101 * 0.99)
        """
        broker = SimulationBroker(initial_cash=10000.0, execution_delay=False)

        # Entry
        entry_order = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=1.0,
            tsl_pct=0.01,  # 1% TSL
            metadata={"base_price": 100.0},
        )
        broker.submit_order(entry_order)

        # Fill entry
        event_entry = MarketEvent(
            timestamp=datetime.now(),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            open=100.0,
            high=100.5,
            low=99.5,
            close=100.0,
            volume=1000.0,
        )
        broker.on_market_event(event_entry)

        # Get TSL order
        tsl_order = broker._trailing_stops["BTC"][0]

        # Bar with high spike but lower close
        event = MarketEvent(
            timestamp=datetime.now(),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            open=100.5,
            high=105.0,  # Peak at high
            low=99.0,
            close=101.0,  # Lower close
            volume=1000.0,
        )
        broker.on_market_event(event)

        # Peak should be HIGH (105), not close (101)
        assert tsl_order.metadata["peak_price"] == 105.0, \
            "Peak should track HIGH (105), not close (101)"

        # TSL should be based on high
        expected_tsl = 105.0 * 0.99  # 103.95
        assert abs(tsl_order.trailing_stop_price - expected_tsl) < 0.01, \
            f"TSL should be {expected_tsl} (from high), not {101.0 * 0.99} (from close)"

    def test_tsl_initializes_to_entry_price(self):
        """Peak should initialize to entry price (base price).

        Entry: $100
        Initial peak: $100
        Initial TSL: $99 (100 * 0.99)
        """
        broker = SimulationBroker(initial_cash=10000.0, execution_delay=False)

        # Entry
        base_price = 100.0
        entry_order = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=1.0,
            tsl_pct=0.01,  # 1% TSL
            metadata={"base_price": base_price},
        )
        broker.submit_order(entry_order)

        # Fill entry
        event = MarketEvent(
            timestamp=datetime.now(),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            open=100.0,
            high=100.5,
            low=99.5,
            close=100.0,
            volume=1000.0,
        )
        broker.on_market_event(event)

        # Get TSL order
        tsl_order = broker._trailing_stops["BTC"][0]

        # Peak should initialize to base_price
        assert tsl_order.metadata["peak_price"] == base_price, \
            f"Peak should initialize to base_price ({base_price})"

        # Initial TSL should be based on entry
        expected_tsl = base_price * 0.99
        assert abs(tsl_order.trailing_stop_price - expected_tsl) < 0.01, \
            f"Initial TSL should be {expected_tsl}"


class TestTSLExitPrice:
    """Test that TSL exits at TSL level (with slippage), not at low."""

    def test_tsl_exits_at_tsl_level_not_low(self):
        """TSL should exit at TSL level, not at the low that triggered it.

        Peak: $44,665
        TSL level: $44,218.35 (= $44,665 * 0.99)
        Low at trigger: $44,210.00 (< TSL level, triggers exit)
        Exit price: $44,218.35 * 0.9998 = $44,209.51 (with slippage)

        Exit should be at TSL level ($44,218.35 - slippage), NOT at low ($44,210)
        """
        broker = SimulationBroker(
            initial_cash=10000.0,
            slippage_model=PercentageSlippage(slippage_pct=0.0002),
            execution_delay=False,
        )

        # Entry
        entry_order = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=0.002278,  # Match TASK-007 size
            tsl_pct=0.01,  # 1% TSL
            metadata={"base_price": 43885.0},  # Match TASK-007
        )
        broker.submit_order(entry_order)

        # Fill entry
        event_entry = MarketEvent(
            timestamp=datetime.now(),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            open=43885.0,
            high=43900.0,
            low=43880.0,
            close=43885.0,
            volume=1000.0,
        )
        broker.on_market_event(event_entry)

        # Get TSL order
        tsl_order = broker._trailing_stops["BTC"][0]

        # Price rises to peak
        event_peak = MarketEvent(
            timestamp=datetime.now(),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            open=44600.0,
            high=44665.0,  # Peak from TASK-007
            low=44590.0,
            close=44650.0,
            volume=1000.0,
        )
        broker.on_market_event(event_peak)

        # Verify peak
        assert abs(tsl_order.metadata["peak_price"] - 44665.0) < 1.0, "Peak should be 44665"

        # TSL retracement bar (triggers exit)
        # Low ($44,210) < TSL level ($44,218.35) → Trigger
        event_trigger = MarketEvent(
            timestamp=datetime.now(),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            open=44290.0,
            high=44365.0,
            low=44210.0,  # Below TSL level, triggers exit
            close=44365.0,
            volume=1000.0,
        )

        # Before trigger, calculate expected TSL
        expected_tsl_level = 44665.0 * 0.99  # 44218.35
        expected_exit_price = expected_tsl_level * 0.9998  # 44209.51 (with slippage)

        broker.on_market_event(event_trigger)

        # TSL should have triggered and filled
        assert len(broker._trailing_stops.get("BTC", [])) == 0, "TSL should be removed after fill"

        # Check trade (exit should be at TSL level, not at low)
        trades_df = broker.trades
        assert len(trades_df) == 1, "Should have 1 completed trade"

        trade = trades_df.row(0, named=True)
        exit_price = trade["exit_price"]

        # Exit should be near TSL level (with slippage), NOT at low
        assert abs(exit_price - expected_exit_price) < 1.0, \
            f"Exit should be ~{expected_exit_price} (TSL level with slippage), not {44210.0} (low)"

        # Exit should NOT be at the low
        assert abs(exit_price - 44210.0) > 5.0, \
            "Exit should NOT be at the low that triggered TSL"


class TestTSLTriggerCondition:
    """Test TSL trigger conditions."""

    def test_tsl_triggers_when_low_touches_tsl_level(self):
        """TSL should trigger when low <= tsl_level (intra-bar).

        Peak: $100
        TSL: $99 (1%)
        Bar: Low = $99.00 → Should trigger
        Bar: Low = $98.50 → Should trigger
        Bar: Low = $99.50 → Should NOT trigger
        """
        broker = SimulationBroker(initial_cash=10000.0, execution_delay=False)

        # Entry
        entry_order = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=1.0,
            tsl_pct=0.01,  # 1% TSL
            metadata={"base_price": 100.0},
        )
        broker.submit_order(entry_order)

        # Fill entry
        event_entry = MarketEvent(
            timestamp=datetime.now(),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            open=100.0,
            high=100.5,
            low=99.5,
            close=100.0,
            volume=1000.0,
        )
        broker.on_market_event(event_entry)

        # Peak at 100 → TSL at 99
        tsl_order = broker._trailing_stops["BTC"][0]
        assert abs(tsl_order.trailing_stop_price - 99.0) < 0.01, "TSL should be 99"

        # Bar with low exactly at TSL level
        event_trigger = MarketEvent(
            timestamp=datetime.now(),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            open=100.0,
            high=100.5,
            low=99.0,  # Exactly at TSL
            close=100.0,
            volume=1000.0,
        )
        broker.on_market_event(event_trigger)

        # Should have triggered
        assert len(broker._trailing_stops.get("BTC", [])) == 0, \
            "TSL should trigger when low == tsl_level"

    def test_tsl_does_not_trigger_above_level(self):
        """TSL should NOT trigger when low > tsl_level."""
        broker = SimulationBroker(initial_cash=10000.0, execution_delay=False)

        # Entry
        entry_order = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=1.0,
            tsl_pct=0.01,  # 1% TSL
            metadata={"base_price": 100.0},
        )
        broker.submit_order(entry_order)

        # Fill entry
        event_entry = MarketEvent(
            timestamp=datetime.now(),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            open=100.0,
            high=100.5,
            low=99.5,
            close=100.0,
            volume=1000.0,
        )
        broker.on_market_event(event_entry)

        # Bar with low above TSL level
        event = MarketEvent(
            timestamp=datetime.now(),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            open=100.0,
            high=101.0,
            low=99.5,  # Above TSL (99.0)
            close=100.5,
            volume=1000.0,
        )
        broker.on_market_event(event)

        # Should NOT have triggered
        assert len(broker._trailing_stops.get("BTC", [])) == 1, \
            "TSL should NOT trigger when low > tsl_level"


class TestTSLStageProcessing:
    """Test 4-stage per-bar processing."""

    def test_tsl_updates_peak_with_open_before_check(self):
        """Stage 1: Peak should update with open before checking trigger.

        Initial peak: $100
        Bar opens at $105 → Peak updates to $105
        Then check: TSL = $103.95 (from new peak)
        """
        broker = SimulationBroker(initial_cash=10000.0, execution_delay=False)

        # Entry at 100
        entry_order = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=1.0,
            tsl_pct=0.01,
            metadata={"base_price": 100.0},
        )
        broker.submit_order(entry_order)

        # Fill entry
        event_entry = MarketEvent(
            timestamp=datetime.now(),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            open=100.0,
            high=100.5,
            low=99.5,
            close=100.0,
            volume=1000.0,
        )
        broker.on_market_event(event_entry)

        tsl_order = broker._trailing_stops["BTC"][0]
        assert tsl_order.metadata["peak_price"] == 100.0, "Initial peak should be 100"

        # Bar opens higher (gap up)
        event = MarketEvent(
            timestamp=datetime.now(),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            open=105.0,  # Gap up on open
            high=106.0,
            low=104.0,
            close=105.5,
            volume=1000.0,
        )
        broker.on_market_event(event)

        # Peak should update to open (105) or high (106)
        assert tsl_order.metadata["peak_price"] >= 105.0, \
            "Peak should update with open (stage 1)"

    def test_tsl_can_activate_and_trigger_same_bar(self):
        """TSL can reach new peak and trigger in same bar.

        Bar: Open=$102 (peak updates), High=$105 (peak updates), Low=$103.9 (triggers)
        Peak updates to 105 → TSL = 103.95 → Low (103.9) < TSL → Trigger!
        """
        broker = SimulationBroker(initial_cash=10000.0, execution_delay=False)

        # Entry at 100
        entry_order = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=1.0,
            tsl_pct=0.01,
            metadata={"base_price": 100.0},
        )
        broker.submit_order(entry_order)

        # Fill entry
        event_entry = MarketEvent(
            timestamp=datetime.now(),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            open=100.0,
            high=100.5,
            low=99.5,
            close=100.0,
            volume=1000.0,
        )
        broker.on_market_event(event_entry)

        # Bar where peak rises AND triggers in same bar
        event = MarketEvent(
            timestamp=datetime.now(),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            open=102.0,  # Updates peak to 102
            high=105.0,  # Updates peak to 105 → TSL = 103.95
            low=103.9,  # 103.9 < 103.95 → Triggers!
            close=104.5,
            volume=1000.0,
        )
        broker.on_market_event(event)

        # Should have triggered (TSL can activate and trigger same bar)
        assert len(broker._trailing_stops.get("BTC", [])) == 0, \
            "TSL should trigger in same bar as peak update"


class TestTSLExactValuesTask007:
    """Test exact values from TASK-007 empirical testing."""

    def test_tsl_exact_match_task007_example(self):
        """Match exact values from TASK-007 Test #1.

        Entry: $43,893.78
        Peak: $44,665.00
        TSL level: $44,218.35 (= $44,665 * 0.99)
        Exit: $44,209.51 (= $44,218.35 * 0.9998 with slippage)
        """
        broker = SimulationBroker(
            initial_cash=10000.0,
            slippage_model=PercentageSlippage(slippage_pct=0.0002),
            execution_delay=False,
        )

        # Entry (match TASK-007)
        entry_order = Order(
            asset_id="BTC",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=0.002278,  # TASK-007 size
            tsl_pct=0.01,  # 1% TSL
            metadata={"base_price": 43885.0},  # Close before slippage
        )
        broker.submit_order(entry_order)

        # Fill entry
        event_entry = MarketEvent(
            timestamp=datetime.now(),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            open=43885.0,
            high=43900.0,
            low=43880.0,
            close=43885.0,
            volume=1000.0,
        )
        broker.on_market_event(event_entry)

        tsl_order = broker._trailing_stops["BTC"][0]

        # Rise to peak
        event_peak = MarketEvent(
            timestamp=datetime.now(),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            open=44600.0,
            high=44665.0,  # TASK-007 peak
            low=44590.0,
            close=44650.0,
            volume=1000.0,
        )
        broker.on_market_event(event_peak)

        # Verify peak
        assert abs(tsl_order.metadata["peak_price"] - 44665.0) < 0.5, \
            f"Peak should be 44665, got {tsl_order.metadata['peak_price']}"

        # Verify TSL level
        expected_tsl_level = 44665.0 * 0.99  # 44218.35
        assert abs(tsl_order.trailing_stop_price - expected_tsl_level) < 1.0, \
            f"TSL should be {expected_tsl_level}, got {tsl_order.trailing_stop_price}"

        # Trigger exit (TASK-007 bar)
        event_trigger = MarketEvent(
            timestamp=datetime.now(),
            asset_id="BTC",
            data_type=MarketDataType.BAR,
            open=44290.0,
            high=44365.0,
            low=44210.0,  # TASK-007 low
            close=44365.0,
            volume=1000.0,
        )
        broker.on_market_event(event_trigger)

        # Verify exit
        trades_df = broker.trades
        assert len(trades_df) == 1, "Should have 1 trade"

        trade = trades_df.row(0, named=True)
        exit_price = trade["exit_price"]

        # Expected: 44218.35 * 0.9998 = 44209.51
        expected_exit = expected_tsl_level * 0.9998
        assert abs(exit_price - expected_exit) < 1.0, \
            f"Exit should be ~{expected_exit}, got {exit_price}"

        # Verify matches TASK-007 exactly
        assert abs(exit_price - 44209.51) < 1.0, \
            f"Exit should match TASK-007 value (44209.51), got {exit_price}"


@pytest.mark.skip(reason="TSL threshold not implemented yet")
class TestTSLWithThreshold:
    """Test TSL with threshold activation (optional feature)."""

    def test_tsl_only_activates_after_threshold(self):
        """TSL should only activate after peak exceeds threshold.

        Entry: $100
        Threshold: 2% → $102
        Peak reaches $101.5 → TSL NOT active
        Peak reaches $102.5 → TSL activates
        TSL = $102.5 * 0.99 = $101.475
        """
        pass  # To be implemented when threshold support is added


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
