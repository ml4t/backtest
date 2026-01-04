#!/usr/bin/env python3
"""Unified Correctness Validation Runner.

Runs all validation scenarios across all frameworks and generates a summary report.

Usage:
    # Run all frameworks
    python validation/run_all_correctness.py

    # Run specific framework
    python validation/run_all_correctness.py --framework vectorbt_pro

    # Run specific scenarios
    python validation/run_all_correctness.py --scenarios 01,02,03,04

Output:
    - Console summary
    - validation/CORRECTNESS_RESULTS.md
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Directory structure
VALIDATION_DIR = Path(__file__).parent
PROJECT_ROOT = VALIDATION_DIR.parent

# Framework configurations
FRAMEWORKS = {
    "vectorbt_pro": {
        "venv": ".venv-vectorbt-pro",
        "scenarios": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
        "display_name": "VectorBT Pro",
    },
    "vectorbt_oss": {
        "venv": ".venv",  # Can also use .venv-validation
        "scenarios": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
        "display_name": "VectorBT OSS",
    },
    "backtrader": {
        "venv": ".venv",  # Can also use .venv-validation or .venv-backtrader
        "scenarios": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
        "display_name": "Backtrader",
    },
    "zipline": {
        "venv": ".venv",  # Can also use .venv-validation or .venv-zipline
        "scenarios": ["01", "02", "03", "04", "05", "06", "07", "08", "09"],  # No scenario 10
        "display_name": "Zipline",
    },
    "lean": {
        "venv": None,  # Uses Docker
        "scenarios": ["01"],  # Start with basic scenarios
        "display_name": "LEAN CLI",
    },
}

# Scenario names
SCENARIO_NAMES = {
    "01": "Long Only",
    "02": "Long/Short",
    "03": "Stop Loss",
    "04": "Take Profit",
    "05": "Commission (Pct)",
    "06": "Commission (Per-Share)",
    "07": "Slippage (Fixed)",
    "08": "Slippage (Pct)",
    "09": "Trailing Stop",
    "10": "Bracket Order",
}


def run_scenario(framework: str, scenario: str) -> dict:
    """Run a single validation scenario.

    Returns dict with keys: passed, error, output
    """
    config = FRAMEWORKS[framework]
    scenario_file = VALIDATION_DIR / framework / f"scenario_{scenario}_*.py"

    # Find the actual file
    matches = list(VALIDATION_DIR.glob(f"{framework}/scenario_{scenario}_*.py"))
    if not matches:
        return {"passed": None, "error": f"Scenario file not found: {scenario_file}", "output": ""}

    script_path = matches[0]

    if framework == "lean":
        # LEAN uses lean backtest command
        return run_lean_scenario(scenario)

    # Build command with venv activation
    venv_path = PROJECT_ROOT / config["venv"]
    python_path = venv_path / "bin" / "python"

    if not python_path.exists():
        return {"passed": None, "error": f"venv not found: {venv_path}", "output": ""}

    try:
        result = subprocess.run(
            [str(python_path), str(script_path)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT),
        )

        output = result.stdout + result.stderr

        # Check for PASS/FAIL in output
        if "PASS" in output.upper() and "FAIL" not in output.upper():
            passed = True
        elif "FAIL" in output.upper() or result.returncode != 0:
            passed = False
        else:
            # Check for error indicators
            passed = result.returncode == 0

        return {"passed": passed, "error": None, "output": output}

    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "Timeout (120s)", "output": ""}
    except Exception as e:
        return {"passed": False, "error": str(e), "output": ""}


def run_lean_scenario(scenario: str) -> dict:
    """Run a LEAN validation scenario using Docker."""
    lean_dir = VALIDATION_DIR / "lean" / "workspace"
    scenario_dir = lean_dir / f"scenario_{scenario}_long_only"  # Adjust as needed

    if not scenario_dir.exists():
        return {"passed": None, "error": f"LEAN scenario not found: {scenario_dir}", "output": ""}

    try:
        # Activate venv for lean CLI
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"

        result = subprocess.run(
            ["lean", "backtest", str(scenario_dir)],
            capture_output=True,
            text=True,
            timeout=300,  # LEAN can be slow
            cwd=str(lean_dir),
            env={"PATH": f"{PROJECT_ROOT}/.venv/bin:{subprocess.os.environ.get('PATH', '')}"},
        )

        output = result.stdout + result.stderr
        passed = result.returncode == 0

        return {"passed": passed, "error": None, "output": output}

    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "Timeout (300s)", "output": ""}
    except Exception as e:
        return {"passed": False, "error": str(e), "output": ""}


def run_all_validations(frameworks: list = None, scenarios: list = None) -> dict:
    """Run all validations and return results."""
    if frameworks is None:
        frameworks = list(FRAMEWORKS.keys())

    results = {}

    for framework in frameworks:
        config = FRAMEWORKS.get(framework)
        if not config:
            print(f"Unknown framework: {framework}")
            continue

        print(f"\n{'='*60}")
        print(f"Framework: {config['display_name']}")
        print(f"{'='*60}")

        framework_results = {}
        available_scenarios = scenarios if scenarios else config["scenarios"]

        for scenario in available_scenarios:
            if scenario not in config["scenarios"]:
                print(f"  Scenario {scenario}: SKIPPED (not available)")
                continue

            scenario_name = SCENARIO_NAMES.get(scenario, f"Scenario {scenario}")
            print(f"  Running {scenario}: {scenario_name}...", end=" ", flush=True)

            result = run_scenario(framework, scenario)
            framework_results[scenario] = result

            if result["passed"] is True:
                print("PASS")
            elif result["passed"] is False:
                print(f"FAIL - {result.get('error', 'Unknown error')}")
            else:
                print(f"SKIP - {result.get('error', 'Not found')}")

        results[framework] = framework_results

    return results


def generate_report(results: dict) -> str:
    """Generate markdown report from results."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Correctness Validation Results",
        "",
        f"**Generated**: {now}",
        "",
        "## Summary",
        "",
        "| Framework | Scenario | Status |",
        "|-----------|----------|--------|",
    ]

    pass_count = 0
    fail_count = 0
    skip_count = 0

    for framework, scenarios in results.items():
        display_name = FRAMEWORKS[framework]["display_name"]
        for scenario, result in scenarios.items():
            scenario_name = SCENARIO_NAMES.get(scenario, f"Scenario {scenario}")

            if result["passed"] is True:
                status = "✅ PASS"
                pass_count += 1
            elif result["passed"] is False:
                status = "❌ FAIL"
                fail_count += 1
            else:
                status = "⏭️ SKIP"
                skip_count += 1

            lines.append(f"| {display_name} | {scenario}: {scenario_name} | {status} |")

    lines.extend([
        "",
        "## Statistics",
        "",
        f"- **Passed**: {pass_count}",
        f"- **Failed**: {fail_count}",
        f"- **Skipped**: {skip_count}",
        f"- **Total**: {pass_count + fail_count + skip_count}",
        "",
    ])

    # Add failure details
    failures = []
    for framework, scenarios in results.items():
        for scenario, result in scenarios.items():
            if result["passed"] is False:
                failures.append((framework, scenario, result))

    if failures:
        lines.extend([
            "## Failures",
            "",
        ])
        for framework, scenario, result in failures:
            display_name = FRAMEWORKS[framework]["display_name"]
            scenario_name = SCENARIO_NAMES.get(scenario, f"Scenario {scenario}")
            lines.extend([
                f"### {display_name} - {scenario}: {scenario_name}",
                "",
                f"**Error**: {result.get('error', 'Unknown')}",
                "",
                "```",
                result.get("output", "No output")[:500],
                "```",
                "",
            ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run correctness validations")
    parser.add_argument(
        "--framework",
        type=str,
        help="Specific framework to test (vectorbt_pro, vectorbt_oss, backtrader, zipline, lean)",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        help="Comma-separated list of scenarios to run (e.g., 01,02,03,04)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="validation/CORRECTNESS_RESULTS.md",
        help="Output file path",
    )

    args = parser.parse_args()

    frameworks = [args.framework] if args.framework else None
    scenarios = args.scenarios.split(",") if args.scenarios else None

    print("=" * 60)
    print("Correctness Validation Runner")
    print("=" * 60)

    results = run_all_validations(frameworks=frameworks, scenarios=scenarios)

    # Generate and save report
    report = generate_report(results)
    output_path = PROJECT_ROOT / args.output
    output_path.write_text(report)

    print(f"\n{'='*60}")
    print(f"Report saved to: {output_path}")
    print("=" * 60)

    # Return exit code based on results
    has_failures = any(
        r["passed"] is False
        for scenarios in results.values()
        for r in scenarios.values()
    )

    return 1 if has_failures else 0


if __name__ == "__main__":
    sys.exit(main())
