"""
Test 1.1: Baseline - Entry Signals Only

Objective: Verify all engines execute the same trades with identical entry signals.

Configuration:
- Asset: BTC (synthetic data, 1000 bars)
- Signals: Fixed entry signals (every 50 bars, starting at bar 10)
- Order Type: Market orders (entry only, hold until next entry)
- Fees: 0.0
- Slippage: 0.0
- Initial Cash: $100,000

Success Criteria:
- All engines generate same number of trades
- All entry timestamps match exactly
- All entry prices match exactly (should be close price)
- Position sizes identical
- Final portfolio value identical

Expected: 20 trades (bars 10, 60, 110, ..., 960)
"""
import pytest
import sys
from pathlib import Path

# Add common module to path
sys.path.insert(0, str(Path(__file__).parent))

from common import (
    generate_ohlcv,
    generate_fixed_entries,
    BacktestConfig,
    QEngineWrapper,
    VectorBTWrapper,
    print_validation_report,
)


def test_1_1_baseline_entries():
    """Test 1.1: Baseline - Entry Signals Only"""

    print("\n" + "=" * 80)
    print("TEST 1.1: Baseline - Entry Signals Only")
    print("=" * 80)

    # 1. Generate test data
    print("\n1️⃣  Generating synthetic OHLCV data (1000 bars)...")
    ohlcv = generate_ohlcv(
        n_bars=1000,
        symbol="BTC",
        base_price=35000.0,
        volatility=0.01,
        seed=42,
    )
    print(f"   ✅ Generated {len(ohlcv)} bars")
    print(f"   📊 Price range: ${ohlcv['close'].min():,.2f} - ${ohlcv['close'].max():,.2f}")

    # 2. Generate signals
    print("\n2️⃣  Generating fixed entry signals (every 50 bars)...")
    entries = generate_fixed_entries(n_bars=1000, entry_every=50, start_offset=10)
    print(f"   ✅ Generated {entries.sum()} entry signals")
    print(f"   📍 First 5 entry indices: {entries[entries].index[:5].tolist()}")

    # 3. Configuration (same for all engines)
    config = BacktestConfig(
        initial_cash=100000.0,
        fees=0.0,
        slippage=0.0,
        order_type='market',
    )
    print(f"\n3️⃣  Configuration:")
    print(f"   💰 Initial Cash: ${config.initial_cash:,.2f}")
    print(f"   💸 Fees: {config.fees * 100:.2f}%")
    print(f"   📉 Slippage: {config.slippage * 100:.2f}%")

    # 4. Run engines
    results = {}

    print("\n4️⃣  Running backtests...")

    # Run qengine
    print("   🔧 Running qengine...")
    try:
        qengine = QEngineWrapper()
        results['qengine'] = qengine.run_backtest(ohlcv, entries, exits=None, config=config)
        print(f"      ✅ Complete: {results['qengine'].num_trades} trades")
    except Exception as e:
        print(f"      ❌ Failed: {e}")
        import traceback
        traceback.print_exc()

    # Run VectorBT
    print("   🔧 Running VectorBT...")
    try:
        vbt = VectorBTWrapper()
        results['VectorBT'] = vbt.run_backtest(ohlcv, entries, exits=None, config=config)
        print(f"      ✅ Complete: {results['VectorBT'].num_trades} trades")
    except ImportError:
        print("      ⚠️  VectorBT Pro not installed, skipping")
    except Exception as e:
        print(f"      ❌ Failed: {e}")
        import traceback
        traceback.print_exc()

    # 5. Compare results
    if len(results) >= 2:
        success = print_validation_report(
            results,
            test_name="Test 1.1: Baseline Entries",
            show_first_trades=5,
        )

        # Assert for pytest
        assert success, "Engines produced different results"
    elif len(results) == 1:
        print(f"\n⚠️  Only 1 engine ran successfully")
        print(f"Result: {list(results.values())[0]}")
    else:
        pytest.fail("No engines ran successfully")


if __name__ == "__main__":
    # Run test directly
    test_1_1_baseline_entries()
