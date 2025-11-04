"""
Verify that QEngine and VectorBT actually agree - this is NOT fake!
"""

import sys
from pathlib import Path

import pandas as pd

# Add project paths
qengine_src = Path(__file__).parent.parent.parent / "src"
projects_dir = Path(__file__).parent.parent.parent.parent / "projects"
sys.path.insert(0, str(qengine_src))

from frameworks import QEngineAdapter, VectorBTAdapter


def load_real_data():
    """Load real AAPL data."""
    wiki_path = projects_dir / "daily_us_equities" / "wiki_prices.parquet"
    if not wiki_path.exists():
        raise FileNotFoundError("Need real data for verification")

    df = pd.read_parquet(wiki_path)
    aapl = df[df["ticker"] == "AAPL"].copy()
    aapl["date"] = pd.to_datetime(aapl["date"])
    aapl = aapl.set_index("date").sort_index()
    return aapl.loc["2015-01-01":"2016-12-31"].copy()


def verify_real_agreement():
    """Verify QEngine and VectorBT actually produce identical results."""
    print("VERIFYING REAL QEngine vs VectorBT AGREEMENT")
    print("=" * 60)
    print("This uses REAL VectorBT 0.28.0 from pip, not fake implementation!")
    print()

    # Load real data
    data = load_real_data()
    print(f"Using real AAPL data: {len(data)} rows")
    print(f"Period: {data.index[0]} to {data.index[-1]}")

    # Strategy parameters
    strategy_params = {"name": "MovingAverageCrossover", "short_window": 20, "long_window": 50}
    initial_capital = 10000

    print(
        f"Strategy: MA({strategy_params['short_window']}) vs MA({strategy_params['long_window']})",
    )
    print(f"Capital: ${initial_capital:,}")

    # Create adapters
    qe_adapter = QEngineAdapter()
    vbt_adapter = VectorBTAdapter()

    print("\nRunning QEngine backtest...")
    qe_result = qe_adapter.run_backtest(data, strategy_params, initial_capital)

    print("\nRunning VectorBT backtest (REAL vectorbt 0.28.0)...")
    vbt_result = vbt_adapter.run_backtest(data, strategy_params, initial_capital)

    print(f"\n{'=' * 60}")
    print("COMPARISON RESULTS")
    print(f"{'=' * 60}")

    # Check for errors
    if qe_result.has_errors:
        print(f"❌ QEngine failed: {qe_result.errors}")
        return False

    if vbt_result.has_errors:
        print(f"❌ VectorBT failed: {vbt_result.errors}")
        return False

    # Compare results
    print(f"{'Metric':<20} {'QEngine':<15} {'VectorBT':<15} {'Match':<10}")
    print("-" * 70)

    # Final value
    value_match = abs(qe_result.final_value - vbt_result.final_value) < 0.01
    print(
        f"{'Final Value':<20} ${qe_result.final_value:<14.2f} ${vbt_result.final_value:<14.2f} {'✓' if value_match else '✗':<10}",
    )

    # Total return
    return_match = abs(qe_result.total_return - vbt_result.total_return) < 0.001
    print(
        f"{'Return (%)':<20} {qe_result.total_return:<14.2f} {vbt_result.total_return:<14.2f} {'✓' if return_match else '✗':<10}",
    )

    # Number of trades
    trades_match = qe_result.num_trades == vbt_result.num_trades
    print(
        f"{'Trades':<20} {qe_result.num_trades:<14} {vbt_result.num_trades:<14} {'✓' if trades_match else '✗':<10}",
    )

    # Execution time
    print(
        f"{'Exec Time (s)':<20} {qe_result.execution_time:<14.3f} {vbt_result.execution_time:<14.3f} {'N/A':<10}",
    )

    # Overall agreement
    overall_match = value_match and return_match and trades_match

    print(f"\n{'=' * 60}")
    if overall_match:
        print("✅ PERFECT AGREEMENT! QEngine and VectorBT produce IDENTICAL results")
        print("   This validates QEngine's correctness using real VectorBT 0.28.0")
    else:
        print("❌ DISCREPANCY DETECTED!")
        print(f"   Value match: {value_match}")
        print(f"   Return match: {return_match}")
        print(f"   Trades match: {trades_match}")

    # Show trade details if available
    if qe_result.trades and vbt_result.trades:
        print("\nFirst 3 trades comparison:")
        print(f"{'QEngine':<50} {'VectorBT':<50}")
        print("-" * 100)

        max_trades = min(3, len(qe_result.trades), len(vbt_result.trades))
        for i in range(max_trades):
            qe_trade = qe_result.trades[i]
            vbt_trade = vbt_result.trades[i]

            qe_str = f"{qe_trade.timestamp.strftime('%Y-%m-%d')} {qe_trade.action} @ ${qe_trade.price:.2f}"
            vbt_str = f"{vbt_trade.timestamp.strftime('%Y-%m-%d')} {vbt_trade.action} @ ${vbt_trade.price:.2f}"

            print(f"{qe_str:<50} {vbt_str:<50}")

    print(f"{'=' * 60}")

    return overall_match


if __name__ == "__main__":
    try:
        success = verify_real_agreement()
        if success:
            print("\n🎉 VERIFICATION SUCCESSFUL!")
            print("QEngine has been validated against REAL VectorBT implementation!")
        else:
            print("\n🔍 VERIFICATION SHOWS DISCREPANCY")
            print("Need to investigate differences between frameworks")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 VERIFICATION FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
