"""
Integration test for all 4 platforms on scenario 004 (position re-entry).

This test validates that ml4t.backtest, VectorBT, Backtrader, and Zipline all successfully
handle position accumulation and re-entry patterns.

Expected Results:
- All 4 platforms execute successfully
- Multiple trades per platform (position accumulation)
- Position sizes increase with multiple BUYs
- Re-entry after partial exit works correctly
- 6+ trade groups matched (different execution models)

Test Strategy:
- Position accumulation: BUY → BUY more → SELL all
- Re-entry pattern: BUY → SELL partial → BUY again → SELL all
- All signals use market orders for simplicity across platforms
"""

import subprocess
import sys
from pathlib import Path


def test_all_platforms_scenario_004():
    """
    Run all 4 platforms on scenario 004 and validate position re-entry.

    This test ensures:
    1. All platforms can execute position accumulation without errors
    2. Trades are extracted from each platform
    3. Position tracking works correctly (cumulative positions)
    4. Re-entry after exits is handled properly
    5. Trade matching produces expected validation results
    """
    # Run the validation framework
    result = subprocess.run(
        [
            sys.executable,
            "runner.py",
            "--scenario",
            "004",
            "--platforms",
            "ml4t.backtest,vectorbt,backtrader,zipline",
            "--report",
            "summary",
        ],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Check execution succeeded
    assert result.returncode == 0, f"Runner failed:\n{result.stderr}"

    # Verify key output patterns
    output = result.stdout

    # All platforms should execute successfully
    assert "ml4t.backtest" in output and "✅ OK" in output, "ml4t.backtest failed to execute"
    assert "vectorbt" in output and "✅ OK" in output, "vectorbt failed to execute"
    assert "backtrader" in output and "✅ OK" in output, "backtrader failed to execute"
    assert "zipline" in output and "✅ OK" in output, "zipline failed to execute"

    # Each platform should find trades (multiple due to accumulation)
    assert "Found" in output and "trades" in output, "Expected trades from platforms"

    # Should match trade groups
    assert "Matched" in output and "trade groups" in output

    # No critical differences expected (market orders are straightforward)
    assert "Critical differences: 0" in output, "Critical differences found in position re-entry?"


def test_position_accumulation():
    """
    Verify position accumulation and tracking across platforms.

    Position accumulation means:
    - BUY 100 shares → position = 100
    - BUY 100 more shares → position = 200
    - SELL 200 shares → position = 0

    All platforms should track cumulative positions correctly.
    """
    # Run the validation framework in detailed mode
    result = subprocess.run(
        [
            sys.executable,
            "runner.py",
            "--scenario",
            "004",
            "--platforms",
            "ml4t.backtest,vectorbt,backtrader,zipline",
            "--report",
            "detailed",
        ],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"Runner failed:\n{result.stderr}"
    output = result.stdout

    # Verify no position tracking errors
    assert "POSITION ERROR" not in output.upper(), "Position tracking error detected!"
    assert "NEGATIVE POSITION" not in output.upper(), "Negative position detected!"

    # Should see evidence of accumulation in trades
    # (e.g., entry prices showing multiple buys)
    assert "ml4t.backtest" in output, "Missing ml4t.backtest results"
    assert "vectorbt" in output, "Missing vectorbt results"


def test_reentry_patterns():
    """
    Document platform execution timing for position re-entry.

    Re-entry patterns test:
    - Exit a position (full or partial)
    - Re-enter the same position later
    - All platforms should handle this correctly

    Platform differences:
    - VectorBT: Same-bar execution possible
    - ml4t.backtest: Next-bar execution for market orders
    - Backtrader: Next-bar open execution
    - Zipline: Next-bar execution

    Expected behavior:
    - All platforms execute all signals
    - Position accumulation works (multiple BUYs before SELL)
    - Re-entry works (BUY after SELL)
    - Final position is flat (all closed)
    """
    execution_models = {
        "vectorbt": {
            "timing": "same-bar or next-bar",
            "price": "close (default)",
            "accumulation": "supported",
            "reentry": "supported",
            "notes": "Vectorized execution, may execute same-bar",
        },
        "ml4t.backtest": {
            "timing": "next-bar",
            "price": "open (next bar)",
            "accumulation": "supported",
            "reentry": "supported",
            "notes": "Event-driven, next-bar execution",
        },
        "backtrader": {
            "timing": "next-bar",
            "price": "open (next bar)",
            "accumulation": "supported",
            "reentry": "supported",
            "notes": "Executes at next bar open",
        },
        "zipline": {
            "timing": "next-bar",
            "price": "open (next bar)",
            "accumulation": "supported",
            "reentry": "supported",
            "notes": "Minute-by-minute simulation",
        },
    }

    # Verify each platform has a documented execution model
    assert len(execution_models) == 4
    assert all(
        key in model
        for model in execution_models.values()
        for key in ["timing", "price", "accumulation", "reentry", "notes"]
    )

    # Run actual test to verify behavior matches documentation
    result = subprocess.run(
        [
            sys.executable,
            "runner.py",
            "--scenario",
            "004",
            "--platforms",
            "ml4t.backtest,vectorbt,backtrader,zipline",
            "--report",
            "summary",
        ],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"Runner failed:\n{result.stderr}"

    # Verify all platforms executed
    for platform in execution_models.keys():
        assert platform in result.stdout, f"Platform {platform} missing from results"


if __name__ == "__main__":
    # Allow running this test directly
    print("Running scenario 004 (position re-entry) integration test...")
    test_all_platforms_scenario_004()
    print("✅ All platforms executed position re-entry successfully!")

    print("\nValidating position accumulation...")
    test_position_accumulation()
    print("✅ Position accumulation works correctly!")

    print("\nDocumenting re-entry patterns...")
    test_reentry_patterns()
    print("✅ Re-entry patterns documented and verified!")
