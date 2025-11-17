"""Cross-framework validation runner.

Loads pre-calculated signals and runs them through multiple backtesting frameworks
to validate execution fidelity.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# Add necessary paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.validation.signals.generate import load_signal_set, list_available_signals
from tests.validation.frameworks.qengine_adapter import BacktestAdapter
from tests.validation.frameworks.backtrader_adapter import BacktraderAdapter
from tests.validation.frameworks.vectorbt_adapter import VectorBTAdapter
from tests.validation.frameworks.base import ValidationResult


@dataclass
class ComparisonResult:
    """Results from comparing multiple frameworks."""

    signal_set_name: str
    frameworks: list[str]
    results: dict[str, ValidationResult]
    matches_within_tolerance: bool
    max_variance_pct: float
    comparison_details: dict[str, Any]


class CrossFrameworkValidator:
    """Validates backtesting frameworks against each other using pre-calculated signals."""

    def __init__(
        self,
        tolerance_pct: float = 0.01,  # 0.01% = 1 basis point
        initial_capital: float = 10000.0,
        commission_rate: float = 0.001,
    ):
        """
        Initialize validator.

        Args:
            tolerance_pct: Maximum acceptable variance in final P&L (as percentage)
            initial_capital: Starting capital for all backtests
            commission_rate: Commission rate (0.001 = 0.1%)
        """
        self.tolerance_pct = tolerance_pct
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate

        # Initialize adapters
        self.adapters = {
            "ml4t.backtest": BacktestAdapter(),
            "Backtrader": BacktraderAdapter(),
            "VectorBT": VectorBTAdapter(),
        }

    def run_validation(self, signal_set_name: str) -> ComparisonResult:
        """
        Run cross-framework validation for a signal set.

        Args:
            signal_set_name: Name of signal set to validate (without .pkl extension)

        Returns:
            ComparisonResult with framework comparison
        """
        print("=" * 80)
        print(f"Cross-Framework Validation: {signal_set_name}")
        print("=" * 80)
        print()

        # Load signal set
        print("1. Loading signal set...")
        signal_set = load_signal_set(signal_set_name)

        data = signal_set["data"]
        signals = signal_set["signals"]
        metadata = signal_set["metadata"]

        print(f"   Asset: {metadata.get('asset', 'Unknown')}")
        print(f"   Signal Type: {metadata.get('signal_type', 'Unknown')}")
        print(f"   Parameters: {metadata.get('parameters', {})}")
        print(f"   Data: {len(data)} bars from {data.index[0]} to {data.index[-1]}")
        print(f"   Signals: {signals['entry'].sum()} entries, {signals['exit'].sum()} exits")
        print()

        # Run through each framework
        print("2. Running backtests...")
        results = {}

        for framework_name, adapter in self.adapters.items():
            print(f"\n   Testing {framework_name}...")
            try:
                result = adapter.run_with_signals(
                    data=data,
                    signals=signals,
                    initial_capital=self.initial_capital,
                    commission_rate=self.commission_rate,
                )
                results[framework_name] = result

                if result.has_errors:
                    print(f"   ⚠️  {framework_name} completed with errors: {result.errors}")

            except Exception as e:
                print(f"   ✗ {framework_name} FAILED: {e}")
                import traceback
                traceback.print_exc()

        print()

        # Compare results
        print("3. Comparing results...")
        comparison = self._compare_results(results)

        print()
        self._print_comparison_table(results, comparison)

        # Create comparison result
        return ComparisonResult(
            signal_set_name=signal_set_name,
            frameworks=list(results.keys()),
            results=results,
            matches_within_tolerance=comparison["matches_within_tolerance"],
            max_variance_pct=comparison["max_variance_pct"],
            comparison_details=comparison,
        )

    def _compare_results(self, results: dict[str, ValidationResult]) -> dict[str, Any]:
        """Compare results across frameworks."""
        if len(results) < 2:
            return {
                "matches_within_tolerance": False,
                "max_variance_pct": 0.0,
                "error": "Need at least 2 frameworks to compare",
            }

        # Use first framework as baseline
        baseline_name = list(results.keys())[0]
        baseline = results[baseline_name]

        comparison = {
            "baseline": baseline_name,
            "final_values": {},
            "returns": {},
            "num_trades": {},
            "variances": {},
        }

        # Collect metrics
        for name, result in results.items():
            comparison["final_values"][name] = result.final_value
            comparison["returns"][name] = result.total_return
            comparison["num_trades"][name] = result.num_trades

        # Calculate variances from baseline
        max_variance = 0.0
        for name, result in results.items():
            if name == baseline_name:
                comparison["variances"][name] = 0.0
            else:
                # Calculate percentage variance
                variance_pct = abs(
                    (result.final_value - baseline.final_value) / baseline.final_value * 100
                )
                comparison["variances"][name] = variance_pct
                max_variance = max(max_variance, variance_pct)

        comparison["max_variance_pct"] = max_variance
        comparison["matches_within_tolerance"] = max_variance <= self.tolerance_pct

        return comparison

    def _print_comparison_table(
        self, results: dict[str, ValidationResult], comparison: dict[str, Any]
    ):
        """Print comparison table."""
        print("┌" + "─" * 78 + "┐")
        print("│" + " " * 25 + "COMPARISON RESULTS" + " " * 35 + "│")
        print("├" + "─" * 78 + "┤")

        # Header
        print(f"│ {'Framework':<20} │ {'Final Value':>12} │ {'Return':>10} │ {'Trades':>7} │ {'Var %':>8} │")
        print("├" + "─" * 78 + "┤")

        # Data rows
        for name, result in results.items():
            variance = comparison["variances"].get(name, 0.0)
            variance_str = f"{variance:>7.3f}%" if variance > 0 else "baseline"

            print(
                f"│ {name:<20} │ ${result.final_value:>11,.2f} │ {result.total_return:>9.2f}% │ {result.num_trades:>7} │ {variance_str:>8} │"
            )

        print("└" + "─" * 78 + "┘")
        print()

        # Summary
        if comparison["matches_within_tolerance"]:
            print(
                f"✅ VALIDATION PASSED - Max variance {comparison['max_variance_pct']:.3f}% (tolerance: {self.tolerance_pct}%)"
            )
        else:
            print(
                f"❌ VALIDATION FAILED - Max variance {comparison['max_variance_pct']:.3f}% exceeds tolerance {self.tolerance_pct}%"
            )


def main():
    """Main entry point."""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "CROSS-FRAMEWORK VALIDATION SUITE" + " " * 26 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    # List available signals
    available_signals = list_available_signals()

    if not available_signals:
        print("❌ No signal sets found. Run signal generation first:")
        print("   python tests/validation/signals/generate.py")
        sys.exit(1)

    print(f"Found {len(available_signals)} signal set(s):")
    for signal_name in available_signals:
        print(f"  • {signal_name}")
    print()

    # Create validator
    validator = CrossFrameworkValidator(
        tolerance_pct=0.5,  # 0.5% = 50 basis points (realistic tolerance)
        initial_capital=10000.0,
        commission_rate=0.001,
    )

    # Run validation for each signal set
    all_passed = True
    results = []

    for signal_name in available_signals:
        result = validator.run_validation(signal_name)
        results.append(result)

        if not result.matches_within_tolerance:
            all_passed = False

        print()

    # Final summary
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print()

    for result in results:
        status = "✅ PASSED" if result.matches_within_tolerance else "❌ FAILED"
        print(f"{result.signal_set_name:.<50} {status}")

    print()

    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
        print()
        print("Conclusion: ml4t.backtest execution matches Backtrader and VectorBT")
        print("            within acceptable tolerance. Execution fidelity validated.")
        sys.exit(0)
    else:
        print("⚠️  SOME VALIDATIONS FAILED")
        print()
        print("Review results above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
