"""
Diagnostic Test: Signal Timing Investigation

Objective: Isolate the signal timing discrepancy between ml4t.backtest and VectorBT.

This test uses a minimal signal set to verify engines execute at identical times.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import (
    load_real_crypto_data,
    BacktestConfig,
    ml4t.backtestWrapper,
    VectorBTWrapper,
)
import pandas as pd


def test_diagnostic_signal_timing():
    """Diagnostic: Verify engines execute at identical signal times"""

    print("\n" + "=" * 80)
    print("DIAGNOSTIC TEST: Signal Timing Verification")
    print("=" * 80)

    # 1. Load minimal data
    print("\n1️⃣  Loading 200 bars of BTC data...")
    ohlcv = load_real_crypto_data(symbol="BTC", data_type="spot", n_bars=200)
    print(f"   ✅ Loaded {len(ohlcv)} bars")

    # 2. Create EXPLICIT signals at specific bars
    print("\n2️⃣  Creating explicit signals...")
    entries = pd.Series([False] * 200, index=ohlcv.index)
    exits = pd.Series([False] * 200, index=ohlcv.index)

    # Explicit entry/exit pairs:
    # Trade 1: Enter at bar 10, exit at bar 20
    # Trade 2: Enter at bar 60, exit at bar 70
    # Trade 3: Enter at bar 110, exit at bar 120
    entries.iloc[10] = True
    exits.iloc[20] = True
    entries.iloc[60] = True
    exits.iloc[70] = True
    entries.iloc[110] = True
    exits.iloc[120] = True

    print(f"   ✅ Entry signals at bars: {entries[entries].index.tolist()}")
    print(f"   ✅ Exit signals at bars: {exits[exits].index.tolist()}")
    print(f"   Expected: 3 trades at (10-20, 60-70, 110-120)")

    # 3. Test WITHOUT fees first (baseline)
    print("\n3️⃣  Test A: NO FEES (should match)")
    config_no_fees = BacktestConfig(
        initial_cash=100000.0,
        fees=0.0,
        slippage=0.0,
        order_type='market',
    )

    ml4t.backtest = ml4t.backtestWrapper()
    result_qe_no_fees = ml4t.backtest.run_backtest(ohlcv, entries, exits=exits, config=config_no_fees)

    vbt = VectorBTWrapper()
    result_vbt_no_fees = vbt.run_backtest(ohlcv, entries, exits=exits, config=config_no_fees)

    print(f"\n   Results (NO FEES):")
    print(f"   ml4t.backtest: {result_qe_no_fees.num_trades} trades, final: ${result_qe_no_fees.final_value:,.2f}")
    print(f"   VectorBT: {result_vbt_no_fees.num_trades} trades, final: ${result_vbt_no_fees.final_value:,.2f}")

    # Check entry prices
    print(f"\n   Entry prices:")
    for i, (qe_price, vbt_price) in enumerate(zip(
        result_qe_no_fees.trades['entry_price'].head(3),
        result_vbt_no_fees.trades['entry_price'].head(3)
    ), 1):
        match = "✅" if abs(qe_price - vbt_price) < 0.01 else "❌"
        print(f"     Trade {i}: ml4t.backtest ${qe_price:,.2f} vs VectorBT ${vbt_price:,.2f} {match}")

    # 4. Test WITH fees (where discrepancy occurs)
    print("\n4️⃣  Test B: WITH FEES (0.1%)")
    config_with_fees = BacktestConfig(
        initial_cash=100000.0,
        fees=0.001,  # 0.1%
        slippage=0.0,
        order_type='market',
    )

    result_qe_fees = ml4t.backtest.run_backtest(ohlcv, entries, exits=exits, config=config_with_fees)
    result_vbt_fees = vbt.run_backtest(ohlcv, entries, exits=exits, config=config_with_fees)

    print(f"\n   Results (WITH FEES):")
    print(f"   ml4t.backtest: {result_qe_fees.num_trades} trades, final: ${result_qe_fees.final_value:,.2f}")
    print(f"   VectorBT: {result_vbt_fees.num_trades} trades, final: ${result_vbt_fees.final_value:,.2f}")

    # Check entry prices
    print(f"\n   Entry prices:")
    for i, (qe_price, vbt_price) in enumerate(zip(
        result_qe_fees.trades['entry_price'].head(3),
        result_vbt_fees.trades['entry_price'].head(3)
    ), 1):
        match = "✅" if abs(qe_price - vbt_price) < 0.01 else "❌"
        print(f"     Trade {i}: ml4t.backtest ${qe_price:,.2f} vs VectorBT ${vbt_price:,.2f} {match}")

    # Check if number of trades differs
    if result_qe_fees.num_trades != result_vbt_fees.num_trades:
        print(f"\n   ⚠️  TRADE COUNT MISMATCH!")
        print(f"   ml4t.backtest has {result_qe_fees.num_trades} trades")
        print(f"   VectorBT has {result_vbt_fees.num_trades} trades")
        print(f"\n   This suggests phantom trades being generated!")

    # 5. Detailed trade comparison
    print("\n5️⃣  Detailed Trade Comparison (WITH FEES):")
    print(f"\n   ml4t.backtest trades:")
    for i, row in result_qe_fees.trades.head(5).iterrows():
        print(f"     Trade {i+1}: entry ${row['entry_price']:,.2f}, exit ${row['exit_price']:,.2f}, pnl ${row['pnl']:,.2f}")

    print(f"\n   VectorBT trades:")
    for i, row in result_vbt_fees.trades.head(5).iterrows():
        print(f"     Trade {i+1}: entry ${row['entry_price']:,.2f}, exit ${row['exit_price']:,.2f}, pnl ${row['pnl']:,.2f}")

    # Assertions
    print("\n6️⃣  Validation:")
    no_fees_match = abs(result_qe_no_fees.final_value - result_vbt_no_fees.final_value) < 1.0
    with_fees_match = abs(result_qe_fees.final_value - result_vbt_fees.final_value) < 5.0

    print(f"   NO FEES: {'✅ PASS' if no_fees_match else '❌ FAIL'}")
    print(f"   WITH FEES: {'✅ PASS' if with_fees_match else '❌ FAIL - DISCREPANCY CONFIRMED'}")

    if not with_fees_match:
        diff = abs(result_qe_fees.final_value - result_vbt_fees.final_value)
        print(f"\n   💡 Difference: ${diff:,.2f}")
        print(f"   This confirms the signal timing issue when fees are enabled.")


if __name__ == "__main__":
    test_diagnostic_signal_timing()
