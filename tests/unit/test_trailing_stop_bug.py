"""Test to demonstrate the trailing stop bug.

This test shows that qengine's TSL implementation incorrectly calculates
the trailing distance from the current price instead of from the peak price.
"""

from qengine.core.types import OrderSide, OrderType
from qengine.execution.order import Order


def test_trailing_stop_should_lock_profit():
    """Demonstrate that TSL should lock in profit at 1% below peak."""

    # Create a trailing stop order for a LONG position (SELL stop)
    # Entry was at $44,460, now protecting with 1% TSL
    order = Order(
        asset_id="BTC",
        order_type=OrderType.TRAILING_STOP,
        side=OrderSide.SELL,  # Exit side for long position
        quantity=1.0,
        trail_percent=1.0,  # 1% (converted from 0.01 by bracket_manager)
    )

    # Scenario: Price rises to $45,075 (peak), then drops to $44,950

    # Bar 1: Price at $45,075 (peak)
    order.update_trailing_stop(45075.0)
    stop_at_peak = order.trailing_stop_price
    print(f"\nPeak price: $45,075")
    print(f"Stop price at peak: ${stop_at_peak:,.2f}")
    print(f"Expected: ${45075 * 0.99:,.2f} (1% below peak)")

    # Bar 2: Price drops to $44,950
    order.update_trailing_stop(44950.0)
    stop_after_drop = order.trailing_stop_price
    print(f"\nPrice dropped to: $44,950")
    print(f"Stop price after drop: ${stop_after_drop:,.2f}")
    print(f"Stop should NOT change: ${stop_at_peak:,.2f}")

    # THE BUG: Current implementation recalculates from new lower price
    # Expected: Stop stays at $44,624.25 (1% below $45,075 peak)
    # Actual: Stop might recalculate from $44,950

    # Check if stop is correctly locked
    expected_stop = 45075.0 * 0.99  # $44,624.25

    print(f"\n{'='*60}")
    print(f"Expected locked stop: ${expected_stop:,.2f}")
    print(f"Actual stop price:    ${stop_after_drop:,.2f}")
    print(f"Match: {abs(stop_after_drop - expected_stop) < 0.01}")
    print(f"{'='*60}")

    # The test currently FAILS because of the bug
    # After fix, this should pass
    assert abs(stop_after_drop - expected_stop) < 0.01, (
        f"TSL should lock at ${expected_stop:,.2f} (1% below peak), "
        f"but got ${stop_after_drop:,.2f}"
    )


def test_trailing_stop_current_behavior():
    """Document the CURRENT (buggy) behavior for comparison."""

    order = Order(
        asset_id="BTC",
        order_type=OrderType.TRAILING_STOP,
        side=OrderSide.SELL,
        quantity=1.0,
        trail_percent=1.0,
    )

    # Bar 1: Price at $45,075
    order.update_trailing_stop(45075.0)
    stop1 = order.trailing_stop_price

    # Calculate what CURRENT code does:
    # trail_amount = 45075 * 0.01 = 450.75
    # new_stop = 45075 - 450.75 = 44624.25
    expected1 = 45075 - (45075 * 0.01)
    print(f"\nBar 1: Price=$45,075, Stop=${stop1:,.2f}, Expected=${expected1:,.2f}")
    assert abs(stop1 - expected1) < 0.01

    # Bar 2: Price drops to $44,950
    order.update_trailing_stop(44950.0)
    stop2 = order.trailing_stop_price

    # What CURRENT code does:
    # trail_amount = 44950 * 0.01 = 449.50
    # new_stop = 44950 - 449.50 = 44500.50
    # if 44500.50 > 44624.25: NO! (price fell, stop would go DOWN)
    # So stop stays at 44624.25 (by accident, works correctly!)

    print(f"Bar 2: Price=$44,950, Stop=${stop2:,.2f}")
    print(f"Stop stayed at ${stop2:,.2f} (correctly didn't move down)")

    # Actually, the current logic DOES work for SELL stops!
    # It only updates when new_stop > current_stop (line 353)
    # This prevents lowering the stop, which is correct.

    # Let me test if it breaks on a different scenario...


def test_trailing_stop_with_lower_new_peak():
    """Test scenario where TSL might behave unexpectedly."""

    order = Order(
        asset_id="BTC",
        order_type=OrderType.TRAILING_STOP,
        side=OrderSide.SELL,
        quantity=1.0,
        trail_percent=1.0,  # 1%
    )

    # Scenario: Entry at $44,460
    # Price sequence: $44,500 → $45,075 (peak) → $44,950 → $45,050 (new local peak)

    prices = [44500, 45075, 44950, 45050, 44900, 44624, 44623]

    for i, price in enumerate(prices):
        order.update_trailing_stop(price)
        stop = order.trailing_stop_price
        expected_stop = price * 0.99

        print(f"Bar {i+1}: Price=${price:,.2f}, Stop=${stop:,.2f}, "
              f"Trail=${price - stop:,.2f}")

        # Check if it should trigger at $44,623
        can_fill = order.can_fill(price)
        if can_fill:
            print(f"  → TSL TRIGGERED at ${price:,.2f}!")


if __name__ == "__main__":
    print("="*60)
    print("TEST 1: Expected behavior (will likely FAIL due to bug)")
    print("="*60)
    try:
        test_trailing_stop_should_lock_profit()
        print("\n✅ PASS: TSL correctly locks profit")
    except AssertionError as e:
        print(f"\n❌ FAIL: {e}")

    print("\n" + "="*60)
    print("TEST 2: Current behavior documentation")
    print("="*60)
    test_trailing_stop_current_behavior()

    print("\n" + "="*60)
    print("TEST 3: Price sequence test")
    print("="*60)
    test_trailing_stop_with_lower_new_peak()
