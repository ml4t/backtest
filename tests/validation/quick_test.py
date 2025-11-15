"""Quick test of validation framework components."""
import sys
from pathlib import Path

# Add validation directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing validation framework components...")
print("=" * 60)

# Test 1: Import signal generators
print("\n1. Testing signal generator imports...")
try:
    from signals import MACrossoverSignals, MeanReversionSignals, RandomSignals, Signal
    print("   ✓ Signal generators imported successfully")
except Exception as e:
    print(f"   ✗ Error importing signals: {e}")
    sys.exit(1)

# Test 2: Import data loaders
print("\n2. Testing data loader imports...")
try:
    from data import DataLoader, load_test_data
    print("   ✓ Data loaders imported successfully")
except Exception as e:
    print(f"   ✗ Error importing data: {e}")
    sys.exit(1)

# Test 3: Import adapters
print("\n3. Testing adapter imports...")
try:
    from adapters import (
        ml4t.backtestAdapter,
        VectorBTProAdapter,
        VectorBTFreeAdapter,
        BacktraderAdapter,
        ZiplineAdapter,
    )
    print("   ✓ Adapters imported successfully")
except Exception as e:
    print(f"   ✗ Error importing adapters: {e}")
    sys.exit(1)

# Test 4: Import validators
print("\n4. Testing validator imports...")
try:
    from validators import TradeValidator
    print("   ✓ Validators imported successfully")
except Exception as e:
    print(f"   ✗ Error importing validators: {e}")
    sys.exit(1)

# Test 5: Generate test signals
print("\n5. Testing signal generation...")
try:
    import polars as pl
    from datetime import datetime, timedelta

    # Create simple test data
    start = datetime(2020, 1, 1)
    dates = [start + timedelta(days=i) for i in range(100)]
    test_data = pl.DataFrame({
        'timestamp': dates,
        'symbol': ['AAPL'] * 100,
        'open': [100.0 + i * 0.5 for i in range(100)],
        'high': [101.0 + i * 0.5 for i in range(100)],
        'low': [99.0 + i * 0.5 for i in range(100)],
        'close': [100.5 + i * 0.5 for i in range(100)],
        'volume': [1000000] * 100,
    })

    generator = MACrossoverSignals(fast_period=5, slow_period=10)
    signals = generator.generate_signals(test_data)

    print(f"   ✓ Generated {len(signals)} signals from 100 bars")

except Exception as e:
    print(f"   ✗ Error generating signals: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test ml4t.backtest adapter initialization
print("\n6. Testing ml4t.backtest adapter...")
try:
    adapter = ml4t.backtestAdapter()
    print(f"   ✓ ml4t.backtest adapter created: {adapter.name}")
    print(f"   Stop Loss: {adapter.supports_stop_loss()}")
    print(f"   Take Profit: {adapter.supports_take_profit()}")
    print(f"   Trailing Stop: {adapter.supports_trailing_stop()}")
except Exception as e:
    print(f"   ✗ Error creating ml4t.backtest adapter: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test VectorBT adapter initialization
print("\n7. Testing VectorBT adapters...")
try:
    # Try Pro first
    try:
        adapter_pro = VectorBTProAdapter()
        print(f"   ✓ VectorBT Pro adapter available: {adapter_pro.name}")
    except ImportError:
        print("   ⚠ VectorBT Pro not available (expected)")

    # Free version
    adapter_free = VectorBTFreeAdapter()
    print(f"   ✓ VectorBT Free adapter created: {adapter_free.name}")

except Exception as e:
    print(f"   ✗ Error with VectorBT adapters: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ All component tests passed!")
print("=" * 60)
print("\nNext steps:")
print("1. Run: python tests/validation/run_validation.py --help")
print("2. Try: python tests/validation/run_validation.py --strategy ma_cross --platforms ml4t.backtest")
print("3. Compare: python tests/validation/run_validation.py --strategy ma_cross --platforms ml4t.backtest,vectorbt_free")
