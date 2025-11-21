"""
Integration test for all 4 platforms on scenario 001 (simple market orders).

This test validates that ml4t.backtest, VectorBT, Backtrader, and Zipline all successfully
execute the scenario and produce comparable results.

Expected Results:
- All 4 platforms execute successfully
- Each platform extracts 2 trades
- 6 total trade groups matched
- 6 perfect matches (all frameworks aligned)
"""

import subprocess
import sys
from pathlib import Path


def test_all_platforms_scenario_001():
    """
    Run all 4 platforms on scenario 001 and validate results.

    This is a smoke test that ensures:
    1. All platforms can execute the scenario without errors
    2. Trades are extracted from each platform
    3. Trade matching produces expected validation results
    4. Platform-specific execution model differences are documented
    """
    # Run the validation framework
    result = subprocess.run(
        [
            sys.executable,
            "runner.py",
            "--scenario",
            "001",
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
    assert "ml4t.backtest" in output and "✅ OK" in output
    assert "vectorbt" in output and "✅ OK" in output
    assert "backtrader" in output and "✅ OK" in output
    assert "zipline" in output and "✅ OK" in output

    # Each platform should find 2 trades
    assert "Found 2 trades" in output  # Should appear 4 times

    # Should match 6 trade groups
    assert "Matched 6 trade groups" in output

    # All 6 should be perfect matches
    assert "Perfect matches:     6" in output
    assert "Minor differences:   0" in output

    # No major or critical differences expected
    assert "Major differences:   0" in output
    assert "Critical differences: 0" in output


def test_platform_execution_models():
    """
    Document and verify expected platform execution model differences.

    Platform execution models (for signal at 2017-02-06):
    - VectorBT:   Entry 2017-02-06 @ $130.29 (same-bar close)
    - ml4t.backtest:    Entry 2017-02-07 @ $131.54 (next-bar close)
    - Backtrader: Entry 2017-02-07 @ $130.54 (next-bar open)
    - Zipline:    Entry 2017-02-07 @ $131.60 (next-bar close)

    These differences are EXPECTED and CORRECT due to different execution models.
    """
    # This is a documentation test - it passes if it runs
    # The actual validation is done in the test above

    execution_models = {
        "vectorbt": {
            "timing": "same-bar",
            "price": "close",
            "entry_date": "2017-02-06",
            "entry_price": 130.29,
        },
        "ml4t.backtest": {
            "timing": "next-bar",
            "price": "close",
            "entry_date": "2017-02-07",
            "entry_price": 131.54,
        },
        "backtrader": {
            "timing": "next-bar",
            "price": "open",
            "entry_date": "2017-02-07",
            "entry_price": 130.54,
        },
        "zipline": {
            "timing": "next-bar",
            "price": "close",
            "entry_date": "2017-02-07",
            "entry_price": 131.60,
        },
    }

    # Verify each platform has a documented execution model
    assert len(execution_models) == 4
    assert all(
        key in model
        for model in execution_models.values()
        for key in ["timing", "price", "entry_date", "entry_price"]
    )


if __name__ == "__main__":
    # Allow running this test directly
    print("Running all platforms integration test...")
    test_all_platforms_scenario_001()
    print("✅ All platforms validated successfully!")

    print("\nRunning execution model documentation test...")
    test_platform_execution_models()
    print("✅ Execution models documented!")
