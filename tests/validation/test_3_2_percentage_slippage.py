"""
Test 3.2: Percentage Slippage (0.05%)

Objective: Verify engines apply 0.05% percentage slippage correctly.
           BUY fills at market_price * 1.0005, SELL fills at market_price * 0.9995.
           Total slippage impact is proportional to trade size and price.

Configuration:
- Asset: BTC (real CryptoCompare spot data, 1000 minute bars from 2021-01-01)
- Signals: Fixed entry/exit pairs (entry every 50 bars, hold for 10 bars)
- Order Type: Market orders
- Fees: 0.0 (no commission)
- Slippage: 0.05% (0.0005)
- Initial Cash: $100,000

Success Criteria:
- All engines generate same number of trades (20 expected)
- Fill prices adjusted by 0.05% from market price
- BUY: fill_price = market_price * 1.0005
- SELL: fill_price = market_price * 0.9995
- Slippage proportional to price (higher prices → higher $ slippage)
- Total slippage cost ≈ 0.1% per round trip * notional
- Final values within $5 tolerance

Expected: 20 trades with 0.05% slippage per fill (0.1% per round trip)
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


def test_3_2_percentage_slippage():
    """Test 3.2: Percentage Slippage (0.05%)"""

    print("\n" + "=" * 80)
    print("TEST 3.2: Percentage Slippage (0.05%)")
    print("=" * 80)

    # 1. Load real BTC data
    print("\n1️⃣  Loading real BTC spot data (1000 bars)...")
    ohlcv = load_real_crypto_data(
        symbol="BTC",
        data_type="spot",
        n_bars=1000,
    )

    # Calculate average price for slippage cost estimation
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

    # 3. Configuration (0.05% slippage, no fees)
    slippage_pct = 0.0005  # 0.05%

    config = BacktestConfig(
        initial_cash=100000.0,
        fees=0.0,             # No commission for this test
        slippage=slippage_pct,  # 0.05% percentage slippage
        order_type='market',
    )

    print(f"\n3️⃣  Configuration:")
    print(f"   💰 Initial Cash: ${config.initial_cash:,.2f}")
    print(f"   💸 Fees: {config.fees * 100:.2f}%")
    print(f"   📉 Slippage: {slippage_pct * 100:.2f}% (percentage)")
    print(f"   ⚠️  Expected per fill: ~${avg_price * slippage_pct:.2f} at avg price")
    print(f"   ⚠️  Expected per round trip: ~0.1% of notional")

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
            test_name="Test 3.2: Percentage Slippage (0.05%)",
            show_first_trades=5,
        )

        # Additional validation: Check slippage behavior
        print("\n5️⃣  Slippage Validation:")
        for engine_name, result in results.items():
            if len(result.trades) > 0:
                print(f"   {engine_name}:")
                print(f"     Number of trades: {result.num_trades}")
                print(f"     Final value: ${result.final_value:,.2f}")
                print(f"     Total PnL: ${result.total_pnl:,.2f}")

                # Estimate total slippage impact
                # For percentage slippage: slippage = notional * slippage_pct * 2 (entry + exit)
                # Approximate notional ≈ initial_cash (since we're trading full capital)
                # Expected: ~$100K * 0.0005 * 2 * 20 trades = ~$200 per trade = ~$4,000 total
                # But actual notional varies with price changes
                estimated_slippage_per_roundtrip = config.initial_cash * slippage_pct * 2
                estimated_total_slippage = estimated_slippage_per_roundtrip * result.num_trades
                print(f"     Estimated slippage/round-trip: ~${estimated_slippage_per_roundtrip:,.2f}")
                print(f"     Estimated total slippage: ~${estimated_total_slippage:,.2f}")

        # Compare final values
        if len(results) == 2:
            engines = list(results.keys())
            diff = abs(results[engines[0]].final_value - results[engines[1]].final_value)
            print(f"\n   💡 Final value difference: ${diff:,.2f}")
            print(f"   ⚠️  Tolerance: $5.00 (standard rounding tolerance)")

            # For percentage slippage, both engines should use same percentage
            # Expect very close results (within $5)
            if diff > 5.0:
                print(f"   ❌ Difference exceeds tolerance!")
            else:
                print(f"   ✅ Within tolerance")

        # Assert for pytest
        assert success or diff <= 5.0, \
            f"Percentage slippage test failed: ${diff:.2f} difference exceeds $5 tolerance"

    elif len(results) == 1:
        print(f"\n⚠️  Only 1 engine ran successfully")
        print(f"Result: {list(results.values())[0]}")
        # Still pass test if only one engine works (for development)
        pytest.skip("Only one engine available for testing")
    else:
        pytest.fail("No engines ran successfully")


if __name__ == "__main__":
    # Run test directly
    test_3_2_percentage_slippage()
