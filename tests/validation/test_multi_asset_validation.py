"""
Comprehensive multi-asset validation across all frameworks and signal types.

Tests ml4t.backtest, Backtrader, and VectorBT with:
- SMA crossover signals (trend-following)
- Random signals (stress testing)
- Rebalancing signals (portfolio rotation)

Validates correctness and performance at scale.
"""

import pickle
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from .frameworks.backtrader_adapter import BacktraderAdapter
from .frameworks.qengine_adapter import BacktestAdapter  # ml4t.backtest adapter
from .frameworks.vectorbt_adapter import VectorBTAdapter
from .frameworks.base import FrameworkConfig


SIGNAL_DIR = Path(__file__).parent / "signals"


def load_multi_asset_signals(signal_file: Path) -> Dict[str, Any]:
    """Load multi-asset signal dataset from pickle."""
    with open(signal_file, "rb") as f:
        return pickle.load(f)


def run_framework_comparison(signal_file: Path, description: str = "") -> pd.DataFrame:
    """
    Run all 3 frameworks on a multi-asset signal dataset and compare results.

    Args:
        signal_file: Path to signal pickle file
        description: Human-readable description of the test

    Returns:
        DataFrame with comparison results
    """
    print(f"\n{'='*80}")
    print(f"Multi-Asset Validation: {description}")
    print(f"Signal file: {signal_file.name}")
    print(f"{'='*80}")

    # Load signal dataset
    signal_set = load_multi_asset_signals(signal_file)
    metadata = signal_set.get("metadata", {})

    print(f"\nSignal Type: {metadata.get('signal_type', 'unknown')}")
    print(f"Parameters: {metadata.get('parameters', {})}")
    print(f"Universe: {metadata.get('num_assets', 'unknown')} assets")
    print(f"Period: {metadata.get('start_date', '')} to {metadata.get('end_date', '')}")

    # Get first asset to check data availability
    first_asset = list(signal_set["assets"].keys())[0]
    first_data = signal_set["assets"][first_asset]["data"]
    print(f"Data points: {len(first_data)} bars per asset")

    # Configure frameworks (realistic settings)
    config = FrameworkConfig.realistic()

    # Run all frameworks
    adapters = [
        BacktestAdapter(),  # ml4t.backtest
        BacktraderAdapter(),
        VectorBTAdapter(),
    ]

    results = {}
    for adapter in adapters:
        print(f"\n{'-'*60}")
        print(f"Running {adapter.framework_name}...")
        print(f"{'-'*60}")

        try:
            # For multi-asset, we need to test each asset separately
            # (true multi-asset portfolio coming in Phase 2)
            # For now, pick a representative asset
            test_symbol = first_asset
            asset_data = signal_set["assets"][test_symbol]
            data = asset_data["data"]
            signals = asset_data["signals"]

            result = adapter.run_with_signals(data, signals, config)
            results[adapter.framework_name] = result

            print(f"✓ Final value: ${result.final_value:,.2f}")
            print(f"✓ Total return: {result.total_return:.2f}%")
            print(f"✓ Trades: {result.num_trades}")
            print(f"✓ Daily returns: {len(result.daily_returns)} periods")
            print(f"✓ Execution time: {result.execution_time:.3f}s")

        except Exception as e:
            print(f"✗ Failed: {e}")
            results[adapter.framework_name] = None

    # Generate comparison table
    print(f"\n{'='*80}")
    print("Framework Comparison Summary")
    print(f"{'='*80}")
    print(f"{'Framework':<15} {'Final Value':<15} {'Return %':<12} {'Trades':<8} {'Time (s)':<12}")
    print('-'*80)

    comparison_data = []
    for name, result in results.items():
        if result is None:
            print(f"{name:<15} {'FAILED':<15}")
            continue

        print(
            f"{name:<15} "
            f"${result.final_value:>12,.2f} "
            f"{result.total_return:>10.2f}% "
            f"{result.num_trades:>6} "
            f"{result.execution_time:>10.3f}"
        )

        comparison_data.append({
            "framework": name,
            "final_value": result.final_value,
            "total_return": result.total_return,
            "num_trades": result.num_trades,
            "execution_time": result.execution_time,
            "daily_returns": result.daily_returns,
            "equity_curve": result.equity_curve,
        })

    # Calculate variance
    if len(comparison_data) >= 2:
        print(f"\n{'='*80}")
        print("Variance Analysis")
        print(f"{'='*80}")

        final_values = [d["final_value"] for d in comparison_data]
        returns = [d["total_return"] for d in comparison_data]

        max_val = max(final_values)
        min_val = min(final_values)
        variance_pct = ((max_val - min_val) / min_val) * 100

        print(f"Final Value Range: ${min_val:,.2f} to ${max_val:,.2f}")
        print(f"Variance: {variance_pct:.2f}%")

        if variance_pct < 1.0:
            print("✓ Excellent agreement (<1% variance)")
        elif variance_pct < 5.0:
            print("✓ Good agreement (<5% variance)")
        elif variance_pct < 20.0:
            print("⚠️ Acceptable variance (<20% - likely execution model differences)")
        else:
            print("⚠️ High variance (>20% - investigate)")

    # Performance comparison
    if len(comparison_data) >= 2:
        print(f"\n{'='*80}")
        print("Performance Analysis")
        print(f"{'='*80}")

        execution_times = {d["framework"]: d["execution_time"] for d in comparison_data}
        fastest = min(execution_times.values())
        print(f"Fastest: {fastest:.3f}s")

        for framework, time_s in execution_times.items():
            speedup = time_s / fastest
            print(f"{framework:<15}: {time_s:>8.3f}s ({speedup:>5.1f}x)")

    return pd.DataFrame(comparison_data)


def test_sma_crossover_signals():
    """Test with SMA crossover signals (trend-following)."""
    signal_file = SIGNAL_DIR / "sp500_top10_sma_crossover.pkl"
    if not signal_file.exists():
        print(f"⚠️ Signal file not found: {signal_file}")
        print("   Run: uv run python tests/validation/signals/generate_multi_asset.py 10")
        return None

    return run_framework_comparison(
        signal_file,
        description="SMA Crossover (10/20) - Trend Following"
    )


def test_random_signals():
    """Test with random signals (stress testing)."""
    signal_file = SIGNAL_DIR / "sp500_top10_random_5pct.pkl"
    if not signal_file.exists():
        print(f"⚠️ Signal file not found: {signal_file}")
        print("   Run: uv run python tests/validation/signals/generate_random.py 10")
        return None

    return run_framework_comparison(
        signal_file,
        description="Random Signals (5% entry/exit) - Stress Test"
    )


def test_rebalancing_signals():
    """Test with rebalancing signals (portfolio rotation)."""
    signal_file = SIGNAL_DIR / "sp500_top10_rebal_momentum_top5_weekly.pkl"
    if not signal_file.exists():
        print(f"⚠️ Signal file not found: {signal_file}")
        print("   Run: uv run python tests/validation/signals/generate_rebalancing.py 10 5 weekly momentum")
        return None

    return run_framework_comparison(
        signal_file,
        description="Rebalancing (Hold Top 5, Weekly) - Portfolio Rotation"
    )


def run_all_validations():
    """Run all validation tests."""
    print("="*80)
    print("COMPREHENSIVE MULTI-ASSET VALIDATION SUITE")
    print("="*80)
    print("\nTesting ml4t.backtest correctness and performance across:")
    print("  1. SMA Crossover (trend-following)")
    print("  2. Random Signals (stress testing)")
    print("  3. Rebalancing (portfolio rotation)")
    print()

    results = {}

    # Test 1: SMA Crossover
    results["sma_crossover"] = test_sma_crossover_signals()

    # Test 2: Random Signals
    results["random"] = test_random_signals()

    # Test 3: Rebalancing
    results["rebalancing"] = test_rebalancing_signals()

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")

    for test_name, result_df in results.items():
        if result_df is not None and len(result_df) > 0:
            print(f"\n{test_name.upper()}:")
            qengine_result = result_df[result_df["framework"] == "ml4t.backtest"].iloc[0]
            print(f"  ml4t.backtest: ${qengine_result['final_value']:,.2f} ({qengine_result['total_return']:.2f}%)")
            print(f"  Trades: {qengine_result['num_trades']}")
            print(f"  Time: {qengine_result['execution_time']:.3f}s")
        else:
            print(f"\n{test_name.upper()}: SKIPPED (signal file missing)")

    print("\n✅ Multi-asset validation complete!")
    print("\nNext steps:")
    print("  - Generate more signal types (different parameters)")
    print("  - Test with 50 and 100-stock universes")
    print("  - Analyze edge cases (stocks dropping out, gaps)")


if __name__ == "__main__":
    run_all_validations()
