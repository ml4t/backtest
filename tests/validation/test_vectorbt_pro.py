"""VectorBT Pro validation test (separate from OSS)."""
import time
import sys
sys.path.insert(0, '/home/stefan/ml4t/software/backtest/tests/validation')

from test_validation import (
    generate_test_data, generate_signals,
    run_engine, run_vectorbt_pro, compare_results
)

if __name__ == "__main__":
    print("="*80)
    print("VECTORBT PRO VALIDATION TEST")
    print("250 assets, 1 year, 25 long + 25 short daily")
    print("="*80)

    # Parameters
    INITIAL_CASH = 1_000_000.0
    COMMISSION_RATE = 0.001
    SLIPPAGE_RATE = 0.0005

    # Generate data
    print("\n[1/3] Generating test data...")
    prices_pl, prices_pd = generate_test_data(n_assets=250, n_days=252)
    signals_pl = generate_signals(prices_pl, n_long=25, n_short=25)
    print(f"  Prices: {len(prices_pl)} rows")
    print(f"  Signals: {len(signals_pl)} rows")

    # Run engine (same-bar to match VectorBT)
    print("\n[2/3] Running engine (same-bar mode)...")
    engine_result = run_engine(
        prices_pl, signals_pl,
        INITIAL_CASH, COMMISSION_RATE, SLIPPAGE_RATE,
        use_next_bar=False,  # Match VectorBT's same-bar execution
    )
    print(f"  engine: PnL = ${engine_result.total_pnl:,.2f}, Trades = {engine_result.num_trades}, Runtime = {engine_result.runtime_seconds:.3f}s")

    # Run VectorBT Pro
    print("\n[3/3] Running VectorBT Pro...")
    vbt_pro_result = run_vectorbt_pro(
        prices_pd, signals_pl,
        INITIAL_CASH, COMMISSION_RATE, SLIPPAGE_RATE,
    )

    if vbt_pro_result:
        print(f"  VectorBT Pro: PnL = ${vbt_pro_result.total_pnl:,.2f}, Trades = {vbt_pro_result.num_trades}, Runtime = {vbt_pro_result.runtime_seconds:.3f}s")
        print(f"  Final Value: ${vbt_pro_result.final_value:,.2f}")

        # Sample trades for debugging
        if vbt_pro_result.trades:
            print(f"  Sample trades (VBT Pro):")
            for t in sorted(vbt_pro_result.trades, key=lambda x: x.entry_time)[:5]:
                print(f"    {t.asset}: {t.entry_time} -> {t.exit_time}, entry=${t.entry_price:.2f}, exit=${t.exit_price:.2f}, qty={t.quantity:.2f}, pnl=${t.pnl:.2f}")

        # Compare with engine trades
        if engine_result.trades:
            print(f"  Sample trades (engine):")
            for t in sorted(engine_result.trades, key=lambda x: x.entry_time)[:5]:
                print(f"    {t.asset}: {t.entry_time} -> {t.exit_time}, entry=${t.entry_price:.2f}, exit=${t.exit_price:.2f}, qty={t.quantity:.2f}, pnl=${t.pnl:.2f}")

        # Compare
        results = [vbt_pro_result, engine_result]
        comparison = compare_results(results)

        print("\n" + "="*80)
        print("RESULTS COMPARISON")
        print("="*80)

        pnl_diff = abs(vbt_pro_result.total_pnl - engine_result.total_pnl) / abs(engine_result.total_pnl) * 100
        print(f"\nPnL Difference:  {pnl_diff:.4f}%")
        print(f"Trade Count:     VectorBT Pro = {vbt_pro_result.num_trades}, engine = {engine_result.num_trades}")

        speedup = vbt_pro_result.runtime_seconds / engine_result.runtime_seconds
        if speedup > 1:
            print(f"Performance:     engine is {speedup:.2f}x FASTER than VectorBT Pro")
        else:
            print(f"Performance:     engine is {1/speedup:.2f}x SLOWER than VectorBT Pro")

        if pnl_diff < 0.1:
            print(f"\n✓ PASS: PnL matches within 0.1% (diff = {pnl_diff:.4f}%)")
        elif pnl_diff < 1.0:
            print(f"\n⚠ ACCEPTABLE: PnL within 1% (diff = {pnl_diff:.4f}%)")
        else:
            print(f"\n✗ FAIL: PnL difference too large (diff = {pnl_diff:.4f}%)")

    else:
        print("  VectorBT Pro not available")
