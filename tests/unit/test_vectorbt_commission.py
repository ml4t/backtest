"""Test VectorBT Commission Model (TASK-020)."""

from qengine.execution.commission import VectorBTCommission
from qengine.execution.order import Order
from qengine.core.types import OrderType, OrderSide


def test_vectorbt_commission_percentage_only():
    """Test percentage-only fees (matching TASK-009 Example 1)."""
    # 0.1 BTC at $50,010 (post-slippage)
    # Order value = 0.1 * 50010 = $5,001
    # Fees = 5001 * 0.0002 = $1.0002
    
    commission_model = VectorBTCommission(fee_rate=0.0002, fixed_fee=0.0)
    
    order = Order(
        asset_id="BTC",
        order_type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=0.1,
    )
    
    fill_quantity = 0.1
    fill_price = 50010.0  # Post-slippage price
    
    fees = commission_model.calculate(order, fill_quantity, fill_price)
    
    order_value = fill_quantity * fill_price
    expected_fees = order_value * 0.0002
    
    print(f"Order value: ${order_value:.2f}")
    print(f"Calculated fees: ${fees:.4f}")
    print(f"Expected fees: ${expected_fees:.4f}")
    
    assert abs(fees - expected_fees) < 0.001, f"Expected ${expected_fees}, got ${fees}"
    print("✅ Percentage-only fees correct!")


def test_vectorbt_commission_with_fixed_fee():
    """Test percentage + fixed fees (matching TASK-009 Example 2)."""
    # 0.1 BTC at $50,010 (post-slippage)
    # Order value = 0.1 * 50010 = $5,001
    # Percentage fees = 5001 * 0.0002 = $1.0002
    # Fixed fees = $10.00
    # Total = $11.0002
    
    commission_model = VectorBTCommission(fee_rate=0.0002, fixed_fee=10.0)
    
    order = Order(
        asset_id="BTC",
        order_type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=0.1,
    )
    
    fill_quantity = 0.1
    fill_price = 50010.0  # Post-slippage price
    
    fees = commission_model.calculate(order, fill_quantity, fill_price)
    
    order_value = fill_quantity * fill_price
    percentage_fees = order_value * 0.0002
    expected_total = percentage_fees + 10.0
    
    print(f"\nOrder value: ${order_value:.2f}")
    print(f"Percentage fees: ${percentage_fees:.4f}")
    print(f"Fixed fees: $10.00")
    print(f"Total fees: ${fees:.4f}")
    print(f"Expected fees: ${expected_total:.4f}")
    
    assert abs(fees - expected_total) < 0.001, f"Expected ${expected_total}, got ${fees}"
    print("✅ Percentage + fixed fees correct!")


def test_vectorbt_commission_exit_fees():
    """Test exit fees (matching TASK-009 Example 3)."""
    # Exit 0.1 BTC at $50,989.80 (post-slippage)
    # Order value = 0.1 * 50989.80 = $5,098.98
    # Fees = 5098.98 * 0.0002 = $1.01980
    
    commission_model = VectorBTCommission(fee_rate=0.0002, fixed_fee=0.0)
    
    order = Order(
        asset_id="BTC",
        order_type=OrderType.MARKET,
        side=OrderSide.SELL,  # Exit order
        quantity=0.1,
    )
    
    fill_quantity = 0.1
    fill_price = 50989.80  # Post-slippage price
    
    fees = commission_model.calculate(order, fill_quantity, fill_price)
    
    order_value = fill_quantity * fill_price
    expected_fees = order_value * 0.0002
    
    print(f"\nExit order value: ${order_value:.2f}")
    print(f"Calculated exit fees: ${fees:.5f}")
    print(f"Expected fees: ${expected_fees:.5f}")
    
    assert abs(fees - expected_fees) < 0.001, f"Expected ${expected_fees}, got ${fees}"
    print("✅ Exit fees correct!")


def test_vectorbt_commission_two_component_formula():
    """Verify the two-component formula: total = (value * rate) + fixed."""
    commission_model = VectorBTCommission(fee_rate=0.0002, fixed_fee=5.0)
    
    order = Order(
        asset_id="BTC",
        order_type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=1.0,
    )
    
    # Test various order sizes
    test_cases = [
        (1.0, 1000.0),    # Small: $1000
        (1.0, 10000.0),   # Medium: $10,000
        (10.0, 10000.0),  # Large: $100,000
    ]
    
    print("\n" + "="*60)
    print("Two-Component Formula Verification")
    print("="*60)
    
    for quantity, price in test_cases:
        fees = commission_model.calculate(order, quantity, price)
        
        order_value = quantity * price
        percentage_part = order_value * 0.0002
        fixed_part = 5.0
        expected = percentage_part + fixed_part
        
        print(f"\nOrder: {quantity} @ ${price}")
        print(f"  Order value: ${order_value:,.2f}")
        print(f"  Percentage fees: ${percentage_part:.4f}")
        print(f"  Fixed fees: $5.00")
        print(f"  Total: ${fees:.4f} (expected: ${expected:.4f})")
        
        assert abs(fees - expected) < 0.001, f"Expected ${expected}, got ${fees}"
    
    print("\n✅ Two-component formula verified for all test cases!")


if __name__ == "__main__":
    test_vectorbt_commission_percentage_only()
    test_vectorbt_commission_with_fixed_fee()
    test_vectorbt_commission_exit_fees()
    test_vectorbt_commission_two_component_formula()
    
    print("\n" + "="*60)
    print("🎉 All VectorBT Commission tests passed!")
    print("="*60)
