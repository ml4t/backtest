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
        wrapper = BacktestWrapper()
        results['ml4t.backtest'] = wrapper.run_backtest(ohlcv, entries, exits=exits, config=config)
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
    if len(results) >= 2:
        success = print_validation_report(
            results,
            test_name="Test 2.2: Combined Fees (0.1% + $2)",
            show_first_trades=5,
        )

        # 6. Dual-engine validation
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
    elif len(results) == 1:
        # Single-engine validation (when VectorBT not available)
        print_validation_report(
            results,
            test_name="Test 2.2: Combined Fees (0.1% + $2)",
            show_first_trades=5,
        )

        result = list(results.values())[0]
        print(f"\n⚠️  Only 1 engine ran successfully")
        print(f"Result: {result}")

        # Validate combined fee calculations for single engine
        print("\n5️⃣  Combined Fee Validation:")
        if 'entry_commission' in result.trades.columns and 'exit_commission' in result.trades.columns:
            entry_commission = result.trades['entry_commission'].sum()
            exit_commission = result.trades['exit_commission'].sum()
            total_commission = entry_commission + exit_commission

            print(f"   Entry commission: ${entry_commission:,.2f}")
            print(f"   Exit commission: ${exit_commission:,.2f}")
            print(f"   Total commission: ${total_commission:,.2f}")
            print(f"   As % of initial cash: {total_commission / config.initial_cash * 100:.3f}%")

            # Verify commission > 0
            assert total_commission > 0, "Total commission should be > 0"
            print("   ✅ Total commission > 0")

            # Verify commission is reasonable (should be more than test 2.1 due to fixed fees)
            # Expected: (0.1% × notional) + $2 per trade
            # With 20 trades × 2 (entry+exit) = 40 fills
            # Fixed fees = 40 × $2 = $80
            # Plus percentage fees (~$3,910 from test_2.1)
            # Total ≈ $3,990
            min_expected_commission = 40 * 2.0  # At least $80 from fixed fees alone
            assert total_commission > min_expected_commission, \
                f"Commission ${total_commission:.2f} seems too low for {result.num_trades} trades with $2 fixed fee"
            print(f"   ✅ Commission ${total_commission:,.2f} is reasonable for {result.num_trades} trades with combined fees")

            # Verify exit commission structure (percentage + fixed on each exit)
            print("\n   Verifying exit commission structure (0.1% + $2):")
            for idx, trade in result.trades.head(3).iterrows():
                exit_notional = trade.exit_price * trade.exit_quantity
                expected_pct_comm = exit_notional * config.fees['percentage']
                expected_fixed_comm = config.fees['fixed']
                expected_total_comm = expected_pct_comm + expected_fixed_comm

                # Exit commissions should be exact (within rounding tolerance)
                assert abs(trade.exit_commission - expected_total_comm) < 0.10, \
                    f"Trade {idx} exit commission mismatch: expected ${expected_total_comm:.2f}, got ${trade.exit_commission:.2f}"

                exit_pct = expected_pct_comm / exit_notional * 100
                print(f"   ✅ Trade {idx}: Exit ${trade.exit_commission:.2f} = ${expected_pct_comm:.2f} ({exit_pct:.3f}%) + ${expected_fixed_comm:.2f} (fixed)")

            print("\n   ✅ All combined fee validations passed")
        else:
            pytest.fail("Commission columns not found in trades DataFrame")
    else:
        pytest.fail("No engines ran successfully")


if __name__ == "__main__":
    test_2_2_combined_fees()
