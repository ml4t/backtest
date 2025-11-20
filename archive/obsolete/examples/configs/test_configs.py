#!/usr/bin/env python3
"""Configuration Validation Script

This script validates all example configuration files:
1. Loads each config with Pydantic validation
2. Checks data files exist
3. Validates config structure
4. Reports results

Usage:
    python test_configs.py

Exit codes:
    0: All configs valid
    1: One or more configs invalid
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ml4t.backtest.config import BacktestConfig, ConfigError


def test_config(config_path: Path) -> Tuple[bool, str]:
    """Test a single configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Tuple of (success, message)
    """
    try:
        # Load and validate config
        config = BacktestConfig.from_yaml(config_path)

        # Additional validation checks
        checks = []

        # Check 1: Data sources present
        if config.data_sources.prices:
            checks.append("✓ Price data configured")
        else:
            return False, "✗ Price data missing"

        # Check 2: Execution parameters
        if config.execution.initial_capital > 0:
            checks.append(
                f"✓ Initial capital: ${config.execution.initial_capital:,.0f}"
            )
        else:
            return False, "✗ Invalid initial capital"

        # Check 3: Commission configured (if present)
        if config.execution.commission:
            checks.append(
                f"✓ Commission: {config.execution.commission.type.value} "
                f"@ {config.execution.commission.rate}"
            )

        # Check 4: Slippage configured (if present)
        if config.execution.slippage:
            checks.append(
                f"✓ Slippage: {config.execution.slippage.type.value} "
                f"@ {config.execution.slippage.rate}"
            )

        # Check 5: Risk rules (if present)
        if config.risk_rules:
            risk_checks = []
            if config.risk_rules.max_position_size:
                risk_checks.append(
                    f"max_pos={config.risk_rules.max_position_size * 100:.0f}%"
                )
            if config.risk_rules.stop_loss:
                risk_checks.append(
                    f"stop={config.risk_rules.stop_loss * 100:.0f}%"
                )
            if config.risk_rules.take_profit:
                risk_checks.append(
                    f"tp={config.risk_rules.take_profit * 100:.0f}%"
                )
            if config.risk_rules.max_vix:
                risk_checks.append(f"max_vix={config.risk_rules.max_vix}")
            if risk_checks:
                checks.append(f"✓ Risk rules: {', '.join(risk_checks)}")

        # Check 6: Features (if present)
        if config.features:
            if config.features.type == "precomputed":
                checks.append(
                    f"✓ Features: precomputed from {Path(config.features.path).name}"
                )
            elif config.features.type == "callable":
                checks.append(
                    f"✓ Features: callable {config.features.module}.{config.features.function}"
                )

        # Check 7: Data files exist
        data_file_checks = []
        if config.data_sources.prices:
            prices_path = Path(config.data_sources.prices.path)
            if prices_path.exists():
                data_file_checks.append(f"prices ({prices_path.name})")
            else:
                return False, f"✗ Price data file not found: {prices_path}"

        if config.data_sources.signals:
            signals_path = Path(config.data_sources.signals.path)
            if signals_path.exists():
                data_file_checks.append(f"signals ({signals_path.name})")
            else:
                return False, f"✗ Signals file not found: {signals_path}"

        if config.data_sources.context:
            context_path = Path(config.data_sources.context.path)
            if context_path.exists():
                data_file_checks.append(f"context ({context_path.name})")
            else:
                return False, f"✗ Context file not found: {context_path}"

        if data_file_checks:
            checks.append(f"✓ Data files exist: {', '.join(data_file_checks)}")

        # Success
        message = "\n    ".join(checks)
        return True, message

    except ConfigError as e:
        return False, f"✗ Configuration error: {e}"
    except Exception as e:
        return False, f"✗ Unexpected error: {e}"


def main():
    """Run validation on all example configs."""
    print("=" * 80)
    print("ML4T.BACKTEST CONFIGURATION VALIDATION")
    print("=" * 80)
    print()

    # Find all YAML configs in this directory
    config_dir = Path(__file__).parent
    config_files = sorted(config_dir.glob("*.yaml"))

    if not config_files:
        print("No configuration files found in", config_dir)
        return 1

    print(f"Found {len(config_files)} configuration file(s)")
    print()

    # Test each config
    results = []
    for config_path in config_files:
        print(f"Testing: {config_path.name}")
        print("-" * 80)

        success, message = test_config(config_path)
        results.append((config_path.name, success))

        if success:
            print("  STATUS: ✓ VALID")
            print(f"  {message}")
        else:
            print("  STATUS: ✗ INVALID")
            print(f"  {message}")

        print()

    # Summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print()

    valid_count = sum(1 for _, success in results if success)
    total_count = len(results)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")

    print()
    print(f"Results: {valid_count}/{total_count} configs valid")
    print()

    # Detailed validation checks
    if valid_count == total_count:
        print("=" * 80)
        print("DETAILED VALIDATION CHECKS")
        print("=" * 80)
        print()

        for config_path in config_files:
            print(f"Config: {config_path.name}")
            print("-" * 80)

            config = BacktestConfig.from_yaml(config_path)

            # Show key parameters
            print(f"  Name: {config.name or 'N/A'}")
            print(f"  Description: {config.description or 'N/A'}")
            print(f"  Initial capital: ${config.execution.initial_capital:,.0f}")

            if config.execution.commission:
                print(
                    f"  Commission: {config.execution.commission.type.value} "
                    f"@ {config.execution.commission.rate}"
                )

            if config.execution.slippage:
                print(
                    f"  Slippage: {config.execution.slippage.type.value} "
                    f"@ {config.execution.slippage.rate}"
                )

            print(f"  Execution delay: {config.execution.execution_delay}")

            if config.risk_rules:
                print("  Risk rules:")
                if config.risk_rules.max_position_size:
                    print(
                        f"    - Max position: {config.risk_rules.max_position_size * 100:.0f}%"
                    )
                if config.risk_rules.stop_loss:
                    print(
                        f"    - Stop loss: {config.risk_rules.stop_loss * 100:.0f}%"
                    )
                if config.risk_rules.take_profit:
                    print(
                        f"    - Take profit: {config.risk_rules.take_profit * 100:.0f}%"
                    )
                if config.risk_rules.max_vix:
                    print(f"    - Max VIX: {config.risk_rules.max_vix}")

            # Data sources
            print("  Data sources:")
            print(f"    - Prices: {Path(config.data_sources.prices.path).name}")
            if config.data_sources.signals:
                print(
                    f"    - Signals: {Path(config.data_sources.signals.path).name}"
                )
            if config.data_sources.context:
                print(
                    f"    - Context: {Path(config.data_sources.context.path).name}"
                )

            # Features
            if config.features:
                if config.features.type == "precomputed":
                    print(
                        f"  Features: precomputed from {Path(config.features.path).name}"
                    )
                    if config.features.columns:
                        print(f"    Columns: {', '.join(config.features.columns)}")
                elif config.features.type == "callable":
                    print(
                        f"  Features: {config.features.module}.{config.features.function}"
                    )

            print()

        print("=" * 80)
        print("✓ ALL CONFIGS VALID - READY FOR USE")
        print("=" * 80)
        return 0
    else:
        print("=" * 80)
        print("✗ SOME CONFIGS INVALID - SEE ERRORS ABOVE")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
