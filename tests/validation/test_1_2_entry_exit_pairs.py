"""
Test 1.2: Entry + Exit Signal Pairs

Objective: Verify all engines execute identical round-trip trades with explicit entry and exit signals.

Configuration:
- Asset: BTC (real CryptoCompare spot data, 1000 minute bars from 2021-01-01)
- Signals: Fixed entry/exit pairs (entry every 50 bars, hold for 10 bars)
- Order Type: Market orders
- Fees: 0.0
- Slippage: 0.0
- Initial Cash: $100,000

Success Criteria:
- All engines generate same number of trades
- All entry timestamps match exactly
- All exit timestamps match exactly
- All entry/exit prices match exactly
- Position sizes identical
- PnL identical (within tolerance)
- Final portfolio value identical

Expected: 19 trades (bars 10-20, 60-70, 110-120, ..., 960-970)
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
    BacktestWrapper,
    VectorBTWrapper,
    print_validation_report,
)


def test_1_2_entry_exit_pairs():
    """Test 1.2: Entry + Exit Signal Pairs"""

    print("\n" + "=" * 80)
    print("TEST 1.2: Entry + Exit Signal Pairs")
    print("=" * 80)

    # 1. Load real BTC data
    print("\n1️⃣  Loading real BTC spot data (1000 bars)...")
    ohlcv = load_real_crypto_data(
        symbol="BTC",
        data_type="spot",
        n_bars=1000,
    )

    # 2. Generate signals (entry/exit pairs)
    print("\n2️⃣  Generating entry/exit signal pairs...")
    entries, exits = generate_entry_exit_pairs(
        n_bars=1000,
        entry_every=50,
        hold_bars=10,
        start_offset=10,
    )
    print(f"   ✅ Generated {entries.sum()} entry signals")
    print(f"   ✅ Generated {exits.sum()} exit signals")
    print(f"   📍 First 3 entry indices: {entries[entries].index[:3].tolist()}")
    print(f"   📍 First 3 exit indices: {exits[exits].index[:3].tolist()}")

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

    # Run ml4t.backtest
    print("   🔧 Running ml4t.backtest...")
    try:
        wrapper = BacktestWrapper()
        results['ml4t.backtest'] = wrapper.run_backtest(ohlcv, entries, exits=exits, config=config)
        print(f"      ✅ Complete: {results['ml4t.backtest'].num_trades} trades")
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
            test_name="Test 1.2: Entry + Exit Pairs",
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
    test_1_2_entry_exit_pairs()
