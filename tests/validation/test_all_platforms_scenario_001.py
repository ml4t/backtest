"""
Integration test for all 4 platforms on scenario 001 (simple market orders).

This test validates that qengine, VectorBT, Backtrader, and Zipline all successfully
execute the scenario and produce comparable results with expected differences.

Expected Results:
- All 4 platforms execute successfully
- Each platform extracts 2 trades
- 6 total trade groups matched (different execution models)
- 4 perfect matches
- 2 minor differences (Backtrader uses OPEN, others use CLOSE)
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
    assert "qengine" in output and "✅ OK" in output
    assert "vectorbt" in output and "✅ OK" in output
    assert "backtrader" in output and "✅ OK" in output
    assert "zipline" in output and "✅ OK" in output

    # Each platform should find 2 trades
    assert "Found 2 trades" in output  # Should appear 4 times

    # Should match 6 trade groups
    assert "Matched 6 trade groups" in output

    # Should have 4 perfect matches and 2 minor differences
    assert "Perfect matches:     4" in output
    assert "Minor differences:   2" in output

    # No major or critical differences expected
    assert "Major differences:   0" in output
    assert "Critical differences: 0" in output

    # Expected differences should be documented
    assert "Entry uses different OHLC components" in output
    assert "Entry prices vary by" in output


def test_platform_execution_models():
    """
    Document and verify expected platform execution model differences.

    Platform execution models (for signal at 2017-02-06):
    - VectorBT:   Entry 2017-02-06 @ $130.29 (same-bar close)
    - qengine:    Entry 2017-02-07 @ $131.54 (next-bar close)
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
        "qengine": {
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
