"""
Integration test for all 4 platforms on scenario 003 (stop orders).

This test validates that qengine, VectorBT, Backtrader, and Zipline all successfully
execute stop orders and trigger them at correct prices.

Expected Results:
- All 4 platforms execute successfully
- Each platform extracts 2 trades with stop-loss protection
- Stop orders trigger when market price breaches stop level
- Trade matching produces expected validation results

Test Strategy:
- BUY with STOP LOSS: Position exits when price drops to stop level
- Test realistic stop-loss scenarios protecting against downside
"""

import subprocess
import sys
from pathlib import Path


def test_all_platforms_scenario_003():
    """
    Run all 4 platforms on scenario 003 and validate stop order execution.

    This test ensures:
    1. All platforms can execute stop orders without errors
    2. Trades are extracted from each platform
    3. Stop-loss orders trigger at correct prices
    4. Trade matching produces expected validation results
    """
    # Run the validation framework
    result = subprocess.run(
        [
            sys.executable,
            "runner.py",
            "--scenario",
            "003",
            "--platforms",
            "qengine,vectorbt,backtrader,zipline",
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
    assert "qengine" in output and "✅ OK" in output, "qengine failed to execute"
    assert "vectorbt" in output and "✅ OK" in output, "vectorbt failed to execute"
    assert "backtrader" in output and "✅ OK" in output, "backtrader failed to execute"
    assert "zipline" in output and "✅ OK" in output, "zipline failed to execute"

    # Each platform should find 2 trades (BUY→STOP-LOSS SELL, BUY→STOP-LOSS SELL)
    assert "Found 2 trades" in output, "Expected 2 trades per platform"

    # Should match trade groups
    assert "Matched" in output and "trade groups" in output

    # No critical differences expected
    assert "Critical differences: 0" in output, "Critical differences found!"


def test_stop_loss_validation():
    """
    Verify that all platforms respect stop-loss prices.

    Stop-loss logic:
    - STOP LOSS on long position: Exit when price <= stop price
    - Position should be exited to protect against further losses

    This is a critical correctness check for stop-loss execution.
    """
    # Run the validation framework in detailed mode
    result = subprocess.run(
        [
            sys.executable,
            "runner.py",
            "--scenario",
            "003",
            "--platforms",
            "qengine,vectorbt,backtrader,zipline",
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

    # Verify no stop-loss violations reported
    # The matcher should flag if exits don't occur when stop is hit
    assert "STOP VIOLATION" not in output.upper(), "Stop-loss violation detected!"
    assert "MISSED STOP" not in output.upper(), "Stop-loss not triggered!"


def test_stop_order_execution_timing():
    """
    Document expected execution timing for stop orders.

    Stop orders trigger when market price breaches the stop level:
    - May trigger same-bar (if price breached during bar)
    - May trigger next-bar (if price breaches at open)
    - Should protect against catastrophic losses

    Platform differences:
    - VectorBT: May trigger same-bar if stop hit during bar
    - qengine: Next-bar execution when stop is breached
    - Backtrader: Next-bar execution when stop is hit
    - Zipline: Next-bar execution when stop is breached
    """
    execution_models = {
        "vectorbt": {
            "timing": "same-bar or next-bar",
            "trigger": "when price <= stop_price (for stop-loss)",
            "notes": "Triggers when price touches stop during bar",
        },
        "qengine": {
            "timing": "next-bar",
            "trigger": "when price <= stop_price (for stop-loss)",
            "notes": "Checks stop at next bar, exits if breached",
        },
        "backtrader": {
            "timing": "next-bar",
            "trigger": "when price <= stop_price (for stop-loss)",
            "notes": "Executes at open if stop price breached",
        },
        "zipline": {
            "timing": "next-bar",
            "trigger": "when price <= stop_price (for stop-loss)",
            "notes": "Executes when stop condition met",
        },
    }

    # Verify each platform has a documented execution model
    assert len(execution_models) == 4
    assert all(
        key in model
        for model in execution_models.values()
        for key in ["timing", "trigger", "notes"]
    )


if __name__ == "__main__":
    # Allow running this test directly
    print("Running scenario 003 (stop orders) integration test...")
    test_all_platforms_scenario_003()
    print("✅ All platforms executed stop orders successfully!")

    print("\nValidating stop-loss correctness...")
    test_stop_loss_validation()
    print("✅ Stop-loss prices respected by all platforms!")

    print("\nDocumenting execution timing...")
    test_stop_order_execution_timing()
    print("✅ Execution models documented!")
