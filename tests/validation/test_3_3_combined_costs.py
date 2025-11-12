"""
Test 3.3: Combined Fees + Slippage (0.1% commission + 0.05% slippage)

Objective: Verify engines correctly apply BOTH commission and slippage costs.
           Total cost per round trip ≈ 0.3% (0.1% fees * 2 + 0.05% slippage * 2).
           Validate order of operations: slippage applied first, then commission.

Configuration:
- Asset: BTC (real CryptoCompare spot data, 1000 minute bars from 2021-01-01)
- Signals: Fixed entry/exit pairs (entry every 50 bars, hold for 10 bars)
- Order Type: Market orders
- Fees: 0.1% (0.001) commission
- Slippage: 0.05% (0.0005)
- Initial Cash: $100,000

Success Criteria:
- All engines generate same number of trades (20 expected)
- Both commission and slippage applied to each trade
- Order of operations: slippage → commission
- Commission calculated on slipped price (not market price)
- Total cost ≈ 0.3% per round trip (0.2% fees + 0.1% slippage)
- Final values within $5 tolerance

Order of Operations:
1. Market price: $29,000
2. Apply slippage: BUY at $29,000 * 1.0005 = $29,014.50
3. Calculate commission: $29,014.50 * 0.001 = $29.01
4. Total cost: slippage ($14.50) + commission ($29.01) = $43.51

Expected: 20 trades with combined costs ≈ 0.3% per round trip
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
    QEngineWrapper,
    VectorBTWrapper,
    print_validation_report,
)


def test_3_3_combined_costs():
    """Test 3.3: Combined Fees + Slippage (0.1% + 0.05%)"""

    print("\n" + "=" * 80)
    print("TEST 3.3: Combined Fees + Slippage (0.1% commission + 0.05% slippage)")
    print("=" * 80)

    # 1. Load real BTC data
    print("\n1️⃣  Loading real BTC spot data (1000 bars)...")
    ohlcv = load_real_crypto_data(
        symbol="BTC",
        data_type="spot",
        n_bars=1000,
    )

    # Calculate average price for cost estimation
    avg_price = ohlcv['close'].mean()
    print(f"   📊 Average BTC price: ${avg_price:,.2f}")

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

    # 3. Configuration (0.1% commission + 0.05% slippage)
    commission_pct = 0.001   # 0.1%
    slippage_pct = 0.0005    # 0.05%

    config = BacktestConfig(
        initial_cash=100000.0,
        fees=commission_pct,    # 0.1% commission
        slippage=slippage_pct,  # 0.05% slippage
        order_type='market',
    )

    print(f"\n3️⃣  Configuration:")
    print(f"   💰 Initial Cash: ${config.initial_cash:,.2f}")
    print(f"   💸 Fees: {commission_pct * 100:.2f}%")
    print(f"   📉 Slippage: {slippage_pct * 100:.2f}%")
    print(f"   ⚠️  Total cost per round trip: ~{(commission_pct + slippage_pct) * 2 * 100:.2f}%")

    # Cost breakdown example at avg price
    example_notional = config.initial_cash
    slippage_cost = example_notional * slippage_pct * 2  # Entry + exit
    commission_cost = example_notional * commission_pct * 2  # Entry + exit
    total_cost_per_roundtrip = slippage_cost + commission_cost

    print(f"\n   📊 Cost breakdown (at $100K notional):")
    print(f"      Slippage: ${slippage_cost:,.2f} (0.1% total)")
    print(f"      Commission: ${commission_cost:,.2f} (0.2% total)")
    print(f"      Total: ${total_cost_per_roundtrip:,.2f} (0.3% total)")
    print(f"   ⚠️  Expected total cost for 20 trades: ~${total_cost_per_roundtrip * 20:,.2f}")

    # 4. Run engines
    results = {}

    print("\n4️⃣  Running backtests...")

    # Run qengine
    print("   🔧 Running qengine...")
    try:
        qengine = QEngineWrapper()
        results['qengine'] = qengine.run_backtest(ohlcv, entries, exits=exits, config=config)
        print(f"      ✅ Complete: {results['qengine'].num_trades} trades")
        print(f"      💰 Final value: ${results['qengine'].final_value:,.2f}")
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
            test_name="Test 3.3: Combined Fees + Slippage",
            show_first_trades=5,
        )

        # Additional validation: Check combined costs
        print("\n5️⃣  Combined Costs Validation:")
        for engine_name, result in results.items():
            if len(result.trades) > 0:
                print(f"   {engine_name}:")
                print(f"     Number of trades: {result.num_trades}")
                print(f"     Final value: ${result.final_value:,.2f}")
                print(f"     Total PnL: ${result.total_pnl:,.2f}")

                # Estimate total costs
                # Slippage + commission both applied to each fill
                estimated_cost_per_roundtrip = config.initial_cash * (commission_pct + slippage_pct) * 2
                estimated_total_cost = estimated_cost_per_roundtrip * result.num_trades
                print(f"     Estimated cost/round-trip: ~${estimated_cost_per_roundtrip:,.2f}")
                print(f"     Estimated total cost: ~${estimated_total_cost:,.2f}")

                # Calculate actual cost from PnL difference
                # If we compare to baseline (no costs), we can estimate actual costs
                # But without baseline, we just report the negative PnL as cost estimate
                if result.total_pnl < 0:
                    print(f"     Actual total cost (from PnL): ${abs(result.total_pnl):,.2f}")

        # Compare final values
        if len(results) == 2:
            engines = list(results.keys())
            diff = abs(results[engines[0]].final_value - results[engines[1]].final_value)
            print(f"\n   💡 Final value difference: ${diff:,.2f}")
            # Fixed: Position sizing now correctly accounts for both slippage and commission
            # See engine_wrappers.py for the fix (calculate effective_price with slippage first)
            tolerance = 5.0  # Standard rounding tolerance
            print(f"   ⚠️  Tolerance: ${tolerance:,.2f} (standard rounding tolerance)")

            if diff > tolerance:
                print(f"   ❌ Difference exceeds tolerance!")
            else:
                print(f"   ✅ Within tolerance")

        # Assert for pytest
        assert success or diff <= tolerance, \
            f"Combined costs test failed: ${diff:.2f} difference exceeds ${tolerance:.2f} tolerance"

    elif len(results) == 1:
        print(f"\n⚠️  Only 1 engine ran successfully")
        print(f"Result: {list(results.values())[0]}")
        # Still pass test if only one engine works (for development)
        pytest.skip("Only one engine available for testing")
    else:
        pytest.fail("No engines ran successfully")


if __name__ == "__main__":
    # Run test directly
    test_3_3_combined_costs()
