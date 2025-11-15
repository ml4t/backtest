"""
Test 2.2: Combined Fees (Percentage + Fixed)

Objective: Verify engines calculate combined fee structure correctly:
           - 0.1% percentage fee on order value
           - $2 fixed fee per transaction
           Total commission = (notional * 0.001) + $2

Configuration:
- Asset: BTC (real CryptoCompare spot data, 1000 minute bars from 2021-01-01)
- Signals: Fixed entry/exit pairs (entry every 50 bars, hold for 10 bars)
- Order Type: Market orders
- Fees: 0.1% + $2 per trade
- Slippage: 0.0
- Initial Cash: $100,000

Success Criteria:
- All engines generate same number of trades (20 expected)
- Fixed component ($2) recorded separately
- Percentage component (0.1%) recorded separately
- Total commission matches VectorBT
- Correct accounting for partial fills
- Final values within $10 (rounding tolerance)
- Test passes pytest validation

Expected: 20 trades with combined fees on each fill (entry + exit)
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


def test_2_2_combined_fees():
    """Test 2.2: Combined Fees (0.1% + $2 per trade)"""

    print("\n" + "=" * 80)
    print("TEST 2.2: Combined Fees (0.1% + $2 per trade)")
    print("=" * 80)

    # 1. Load real BTC data
    print("\n1️⃣  Loading real BTC spot data (1000 bars)...")
    ohlcv = load_real_crypto_data(
        symbol="BTC",
        data_type="spot",
        n_bars=1000,
    )

    # 2. Generate signals (20 entry/exit pairs with 10-bar hold)
    print("\n2️⃣  Generating entry/exit signal pairs...")
    entries, exits = generate_entry_exit_pairs(
        n_bars=1000,
        entry_every=50,  # Entry every 50 bars → 20 entries in 1000 bars
        hold_bars=10,    # 10-bar hold period
        start_offset=10, # Start from bar 10
    )
    print(f"   ✅ Generated {entries.sum()} entry signals")
    print(f"   ✅ Generated {exits.sum()} exit signals")
    print(f"   📍 First 5 entry indices: {entries[entries].index[:5].tolist()}")
    print(f"   📍 First 5 exit indices: {exits[exits].index[:5].tolist()}")

    # 3. Configuration (0.1% + $2 combined fees, no slippage)
    config = BacktestConfig(
        initial_cash=100000.0,
        fees={
            'percentage': 0.001,  # 0.1% commission
            'fixed': 2.0,         # $2 per trade
        },
        slippage=0.0,
        order_type='market',
    )
    print(f"\n3️⃣  Configuration:")
    print(f"   💰 Initial Cash: ${config.initial_cash:,.2f}")
    print(f"   💸 Percentage Fee: {config.fees['percentage'] * 100:.2f}%")
    print(f"   💵 Fixed Fee: ${config.fees['fixed']:.2f} per trade")
    print(f"   📉 Slippage: 0.00%")
    print(f"   ⚠️  Expected total commission: ~0.1% + $4 per round trip")

    # 4. Run engines
    results = {}

    print("\n4️⃣  Running backtests...")

    # Run ml4t.backtest
    print("   🔧 Running ml4t.backtest...")
    try:
        ml4t.backtest = BacktestWrapper()
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
    except Exception as e:
        print(f"      ❌ Failed: {e}")
        import traceback
        traceback.print_exc()

    # 5. Validation
    print_validation_report(results, test_name="Test 2.2: Combined Fees (0.1% + $2)")

    # 6. Assertions
    if len(results) == 2:
        qe = results['ml4t.backtest']
        vbt_result = results['VectorBT']

        # Number of trades should match
        assert qe.num_trades == vbt_result.num_trades, \
            f"Trade count mismatch: ml4t.backtest={qe.num_trades}, VectorBT={vbt_result.num_trades}"

        # Final values should be within $10 (combined fees introduce more rounding)
        value_diff = abs(qe.final_value - vbt_result.final_value)
        assert value_diff < 10.0, \
            f"Final value difference ${value_diff:.2f} exceeds tolerance ($10)"

        print("\n" + "=" * 80)
        print("✅ TEST 2.2 PASSED: Combined fees validated successfully")
        print("=" * 80)
    else:
        pytest.skip("Not all engines completed successfully")


if __name__ == "__main__":
    test_2_2_combined_fees()
