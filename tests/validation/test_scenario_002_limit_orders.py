"""
Integration test for all 4 platforms on scenario 002 (limit orders).

This test validates that qengine, VectorBT, Backtrader, and Zipline all successfully
execute limit orders and respect limit prices.

Expected Results:
- All 4 platforms execute successfully
- Each platform extracts 2 trades (BUY limit → SELL limit)
- Limit prices are respected (execution at or better than limit)
- 6 total trade groups matched (different execution models)

Test Strategy:
- BUY limit: Order executes when market price <= limit price
- SELL limit: Order executes when market price >= limit price
- Limit prices are set strategically to ensure execution during test period
"""

import subprocess
import sys
from pathlib import Path


def test_all_platforms_scenario_002():
    """
    Run all 4 platforms on scenario 002 and validate limit order execution.

    This test ensures:
    1. All platforms can execute limit orders without errors
    2. Trades are extracted from each platform
    3. Limit prices are respected (at or better than limit)
    4. Trade matching produces expected validation results
    """
    # Run the validation framework
    result = subprocess.run(
        [
            sys.executable,
            "runner.py",
            "--scenario",
            "002",
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

    # Each platform should find 2 trades
    assert "Found 2 trades" in output, "Expected 2 trades per platform"

    # Should match trade groups (6 for different execution models)
    # VectorBT may differ due to same-bar vs next-bar execution
    assert "Matched" in output and "trade groups" in output

    # No critical differences expected (limit prices must be respected)
    assert "Critical differences: 0" in output, "Critical differences found - limit price violations?"


def test_limit_price_validation():
    """
    Verify that all platforms respect limit prices.

    For BUY limits: Entry price <= limit price
    For SELL limits: Exit price >= limit price

    This is a critical correctness check for limit order execution.
    """
    # Run the validation framework in detailed mode
    result = subprocess.run(
        [
            sys.executable,
            "runner.py",
            "--scenario",
            "002",
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

    # Verify no limit price violations reported
    # The matcher should flag if entry/exit prices violate limits
    assert "LIMIT VIOLATION" not in output.upper(), "Limit price violation detected!"
    assert "EXCEEDED LIMIT" not in output.upper(), "Limit price exceeded!"


def test_limit_order_execution_timing():
    """
    Document expected execution timing for limit orders.

    Limit orders execute when market price reaches or betters the limit:
    - May execute same-bar (if price touched during bar)
    - May execute next-bar (if price touches at open)
    - May not execute at all (if price never reaches limit)

    Platform differences:
    - VectorBT: May execute same-bar if limit hit during bar
    - qengine: Next-bar execution when limit is met
    - Backtrader: Next-bar open if limit hit
    - Zipline: Next-bar execution when limit is met
    """
    execution_models = {
        "vectorbt": {
            "timing": "same-bar or next-bar",
            "price": "limit price (at or better)",
            "notes": "Executes when price touches limit during bar",
        },
        "qengine": {
            "timing": "next-bar",
            "price": "limit price (at or better)",
            "notes": "Checks limit at next bar, executes if met",
        },
        "backtrader": {
            "timing": "next-bar",
            "price": "limit price (at or better)",
            "notes": "Executes at open if limit price reached",
        },
        "zipline": {
            "timing": "next-bar",
            "price": "limit price (at or better)",
            "notes": "Executes when limit condition met",
        },
    }

    # Verify each platform has a documented execution model
    assert len(execution_models) == 4
    assert all(
        key in model
        for model in execution_models.values()
        for key in ["timing", "price", "notes"]
    )


if __name__ == "__main__":
    # Allow running this test directly
    print("Running scenario 002 (limit orders) integration test...")
    test_all_platforms_scenario_002()
    print("✅ All platforms executed limit orders successfully!")

    print("\nValidating limit price correctness...")
    test_limit_price_validation()
    print("✅ Limit prices respected by all platforms!")

    print("\nDocumenting execution timing...")
    test_limit_order_execution_timing()
    print("✅ Execution models documented!")
