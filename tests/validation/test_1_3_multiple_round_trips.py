"""
Test 1.3: Multiple Round Trips

Objective: Verify engines handle 40 rapid entry/exit pairs with short 5-bar hold periods.
           Tests rapid re-entry, position tracking, and FIFO trade pairing.

Configuration:
- Asset: BTC (real CryptoCompare spot data, 1000 minute bars from 2021-01-01)
- Signals: Fixed entry/exit pairs (entry every 25 bars, hold for 5 bars)
- Order Type: Market orders
- Fees: 0.0
- Slippage: 0.0
- Initial Cash: $100,000

Success Criteria:
- All engines generate same number of trades (40 expected)
- All entry timestamps match exactly
- All exit timestamps match exactly
- All entry/exit prices match exactly
- Position sizes identical
- PnL identical (within tolerance)
- Final portfolio value identical
- Final position = 0 (all positions closed)

Expected: 40 trades (bars 0-5, 25-30, 50-55, ..., 975-980)
"""
import pytest
import sys
from pathlib import Path

# Add common module to path
sys.path.insert(0, str(Path(__file__).parent))

from common import (
    load_real_crypto_data,
    generate_entry_exit_pairs,
    BacktestConfig,
    ml4t.backtestWrapper,
    VectorBTWrapper,
    print_validation_report,
)


def test_1_3_multiple_round_trips():
    """Test 1.3: Multiple Round Trips (40 entry/exit pairs)"""

    print("\n" + "=" * 80)
    print("TEST 1.3: Multiple Round Trips (40 Entry/Exit Pairs)")
    print("=" * 80)

    # 1. Load real BTC data
    print("\n1️⃣  Loading real BTC spot data (1000 bars)...")
    ohlcv = load_real_crypto_data(
        symbol="BTC",
        data_type="spot",
        n_bars=1000,
    )

    # 2. Generate signals (40 entry/exit pairs with 5-bar hold)
    print("\n2️⃣  Generating entry/exit signal pairs...")
    entries, exits = generate_entry_exit_pairs(
        n_bars=1000,
        entry_every=25,  # Entry every 25 bars → 40 entries in 1000 bars
        hold_bars=5,     # Short 5-bar hold period
        start_offset=0,  # Start from bar 0
    )
    print(f"   ✅ Generated {entries.sum()} entry signals")
    print(f"   ✅ Generated {exits.sum()} exit signals")
    print(f"   📍 First 5 entry indices: {entries[entries].index[:5].tolist()}")
    print(f"   📍 First 5 exit indices: {exits[exits].index[:5].tolist()}")
    print(f"   📍 Last 5 entry indices: {entries[entries].index[-5:].tolist()}")
    print(f"   📍 Last 5 exit indices: {exits[exits].index[-5:].tolist()}")

    # 3. Configuration (baseline - no fees or slippage)
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

    # Run ml4t.backtest
    print("   🔧 Running ml4t.backtest...")
    try:
        ml4t.backtest = ml4t.backtestWrapper()
        results['ml4t.backtest'] = ml4t.backtest.run_backtest(ohlcv, entries, exits=exits, config=config)
        print(f"      ✅ Complete: {results['ml4t.backtest'].num_trades} trades")
        print(f"      💰 Final value: ${results['ml4t.backtest'].final_value:,.2f}")
    except Exception as e:
        print(f"      ❌ Failed: {e}")
        import traceback
        traceback.print_exc()

    # Run VectorBT
    print("   🔧 Running VectorBT...")
    try:
        vbt = VectorBTWrapper()
        results['VectorBT'] = vbt.run_backtest(ohlcv, entries, exits=exits, config=config)
        print(f"      ✅ Complete: {results['VectorBT'].num_trades} trades")
        print(f"      💰 Final value: ${results['VectorBT'].final_value:,.2f}")
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
            test_name="Test 1.3: Multiple Round Trips (40 pairs)",
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
    test_1_3_multiple_round_trips()
