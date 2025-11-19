#!/usr/bin/env python3
"""Demonstration: How to Use Configuration Files

This script demonstrates how to load and use YAML configurations
with ml4t.backtest.

Shows:
1. Loading configs from YAML
2. Accessing config parameters
3. Validating configurations
4. Converting to/from JSON
5. Using configs in practice

Run:
    python demo_config_usage.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ml4t.backtest.config import BacktestConfig, ConfigError


def demo_loading():
    """Demonstrate loading configurations."""
    print("=" * 80)
    print("DEMO 1: Loading Configurations")
    print("=" * 80)
    print()

    # Load simple strategy config
    config_path = Path(__file__).parent / "simple_strategy.yaml"
    print(f"Loading: {config_path.name}")

    try:
        config = BacktestConfig.from_yaml(config_path)
        print("✓ Config loaded successfully")
        print()

        # Show what's in the config
        print(f"  Name: {config.name}")
        print(f"  Description: {config.description}")
        print()

        return config
    except ConfigError as e:
        print(f"✗ Failed to load config: {e}")
        return None


def demo_accessing_parameters(config: BacktestConfig):
    """Demonstrate accessing config parameters."""
    print("=" * 80)
    print("DEMO 2: Accessing Parameters")
    print("=" * 80)
    print()

    # Execution parameters
    print("Execution Parameters:")
    print(f"  Initial capital: ${config.execution.initial_capital:,.2f}")
    print(f"  Execution delay: {config.execution.execution_delay}")
    print(f"  Allow re-entry: {config.execution.allow_immediate_reentry}")
    print()

    # Commission
    if config.execution.commission:
        print("Commission Model:")
        print(f"  Type: {config.execution.commission.type.value}")
        print(f"  Rate: {config.execution.commission.rate}")
        if config.execution.commission.minimum:
            print(f"  Minimum: ${config.execution.commission.minimum}")
        print()

    # Slippage
    if config.execution.slippage:
        print("Slippage Model:")
        print(f"  Type: {config.execution.slippage.type.value}")
        print(f"  Rate: {config.execution.slippage.rate}")
        print()

    # Risk rules
    if config.risk_rules:
        print("Risk Rules:")
        if config.risk_rules.max_position_size:
            print(
                f"  Max position size: {config.risk_rules.max_position_size * 100:.1f}%"
            )
        if config.risk_rules.stop_loss:
            print(f"  Stop loss: {config.risk_rules.stop_loss * 100:.1f}%")
        if config.risk_rules.take_profit:
            print(f"  Take profit: {config.risk_rules.take_profit * 100:.1f}%")
        if config.risk_rules.max_vix:
            print(f"  Max VIX: {config.risk_rules.max_vix}")
        print()

    # Data sources
    print("Data Sources:")
    print(f"  Prices: {config.data_sources.prices.path}")
    if config.data_sources.signals:
        print(f"  Signals: {config.data_sources.signals.path}")
    if config.data_sources.context:
        print(f"  Context: {config.data_sources.context.path}")
    print()


def demo_validation():
    """Demonstrate config validation."""
    print("=" * 80)
    print("DEMO 3: Validation")
    print("=" * 80)
    print()

    # Try loading all configs
    config_dir = Path(__file__).parent
    for config_file in sorted(config_dir.glob("*.yaml")):
        print(f"Validating: {config_file.name}")
        try:
            config = BacktestConfig.from_yaml(config_file)
            print(f"  ✓ Valid - Initial capital: ${config.execution.initial_capital:,.0f}")
        except ConfigError as e:
            print(f"  ✗ Invalid - {e}")
        print()


def demo_conversion(config: BacktestConfig):
    """Demonstrate converting configs to/from JSON."""
    print("=" * 80)
    print("DEMO 4: Format Conversion")
    print("=" * 80)
    print()

    # Save to JSON
    json_path = Path(__file__).parent / "temp_config.json"
    print(f"Saving config to JSON: {json_path.name}")
    config.to_json(json_path)
    print("✓ Saved")
    print()

    # Load from JSON
    print(f"Loading config from JSON: {json_path.name}")
    loaded_config = BacktestConfig.from_json(json_path)
    print("✓ Loaded")
    print()

    # Verify they match
    print("Verification:")
    print(
        f"  Original capital: ${config.execution.initial_capital:,.2f}"
    )
    print(
        f"  Loaded capital: ${loaded_config.execution.initial_capital:,.2f}"
    )
    print(
        f"  Match: {config.execution.initial_capital == loaded_config.execution.initial_capital}"
    )
    print()

    # Clean up
    json_path.unlink()
    print("✓ Cleaned up temp file")
    print()


def demo_practical_usage():
    """Demonstrate practical usage in a backtest."""
    print("=" * 80)
    print("DEMO 5: Practical Usage")
    print("=" * 80)
    print()

    # Load ML strategy config
    config_path = Path(__file__).parent / "ml_strategy.yaml"
    config = BacktestConfig.from_yaml(config_path)

    print("Strategy: ML Signal Strategy")
    print()

    # Show how you'd use this in a strategy class
    print("In your strategy code, you would:")
    print()

    print("1. Access execution parameters:")
    print(f"   capital = {config.execution.initial_capital}")
    print()

    print("2. Check VIX filtering:")
    print(f"   if vix > {config.risk_rules.max_vix}:")
    print("       return  # Don't trade")
    print()

    print("3. Size positions:")
    print(f"   max_position = nav * {config.risk_rules.max_position_size}")
    print()

    print("4. Apply stop loss:")
    print(f"   stop_price = entry_price * (1 - {config.risk_rules.stop_loss})")
    print()

    print("5. Load data:")
    print(f"   prices = pl.read_parquet('{config.data_sources.prices.path}')")
    if config.data_sources.signals:
        print(f"   signals = pl.read_parquet('{config.data_sources.signals.path}')")
    print()


def demo_all_configs():
    """Show summary of all available configs."""
    print("=" * 80)
    print("DEMO 6: Available Configurations")
    print("=" * 80)
    print()

    config_dir = Path(__file__).parent
    configs = []

    for config_file in sorted(config_dir.glob("*.yaml")):
        try:
            config = BacktestConfig.from_yaml(config_file)
            configs.append((config_file.name, config))
        except ConfigError:
            continue

    print(f"Found {len(configs)} valid configuration(s):\n")

    for name, config in configs:
        print(f"📄 {name}")
        print(f"   Name: {config.name or 'N/A'}")
        print(f"   Capital: ${config.execution.initial_capital:,.0f}")

        # Count features
        features = []
        if config.risk_rules:
            if config.risk_rules.stop_loss:
                features.append("stop loss")
            if config.risk_rules.take_profit:
                features.append("take profit")
            if config.risk_rules.max_vix:
                features.append("VIX filter")

        if config.features:
            features.append(f"{config.features.type} features")

        if features:
            print(f"   Features: {', '.join(features)}")

        print()


def main():
    """Run all demos."""
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  ML4T.BACKTEST CONFIGURATION DEMONSTRATION".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # Run demos
    config = demo_loading()

    if config:
        demo_accessing_parameters(config)

    demo_validation()

    if config:
        demo_conversion(config)

    demo_practical_usage()
    demo_all_configs()

    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("  1. Review the example configs in examples/configs/")
    print("  2. Copy and customize one for your use case")
    print("  3. Validate with: python examples/configs/test_configs.py")
    print("  4. Use in your backtest with BacktestConfig.from_yaml()")
    print()


if __name__ == "__main__":
    main()
