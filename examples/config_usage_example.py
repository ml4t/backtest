#!/usr/bin/env python3
"""Example demonstrating configuration system usage.

This script shows how to:
1. Load configuration from YAML/JSON
2. Access configuration values
3. Handle configuration errors
4. Use environment variables
"""

import os
from pathlib import Path

from ml4t.backtest.config import BacktestConfig, ConfigError


def example_basic_loading():
    """Example 1: Basic configuration loading."""
    print("=" * 60)
    print("Example 1: Basic Configuration Loading")
    print("=" * 60)

    # Set up environment variables for example configs
    os.environ["DATA_PATH"] = str(Path.cwd() / "tests" / "unit" / "test_data")

    try:
        # Load configuration from YAML
        config_path = Path("examples/configs/simple_ma_strategy.yaml")

        # Note: This will fail because test data doesn't exist at those paths
        # In real usage, ensure data files exist at specified paths
        print(f"\nLoading config from: {config_path}")
        print("Note: This example shows the loading process.")
        print("In production, ensure all data files exist at configured paths.\n")

        # Show config structure without loading (data files don't exist)
        import yaml

        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        print("Configuration structure:")
        print(f"  Name: {config_data.get('name')}")
        print(f"  Description: {config_data.get('description', 'N/A')}")
        print(f"  Data sources: {list(config_data.get('data_sources', {}).keys())}")
        print(f"  Features: {config_data.get('features', {}).get('type', 'None')}")
        print(f"  Initial capital: ${config_data.get('execution', {}).get('initial_capital', 0):,}")

    except ConfigError as e:
        print(f"\n✗ Configuration error: {e}")
        print("\nHint: Ensure data files exist and paths are correct.")


def example_accessing_values():
    """Example 2: Accessing configuration values."""
    print("\n" + "=" * 60)
    print("Example 2: Accessing Configuration Values")
    print("=" * 60)

    print("\nConfiguration is a Pydantic model with type-safe access:")
    print("""
# After loading:
config = BacktestConfig.from_yaml(Path("config.yaml"))

# Access values:
print(config.execution.initial_capital)  # 100000.0
print(config.execution.commission.type)  # CommissionType.PER_SHARE
print(config.execution.commission.rate)  # 0.005

# Data sources:
print(config.data_sources.prices.path)   # /path/to/prices.parquet
print(config.data_sources.signals.path)  # /path/to/signals.parquet (if configured)

# Risk rules (optional):
if config.risk_rules:
    print(config.risk_rules.max_position_size)  # 0.5
    print(config.risk_rules.stop_loss)          # 0.05
""")


def example_error_handling():
    """Example 3: Error handling."""
    print("\n" + "=" * 60)
    print("Example 3: Error Handling")
    print("=" * 60)

    print("\nThe configuration system provides clear error messages:\n")

    print("1. File not found:")
    print("   ConfigError: Configuration file not found: config.yaml")
    print("   Hint: Check that the path is correct.\n")

    print("2. Invalid YAML syntax:")
    print("   ConfigError: Invalid YAML in config.yaml:")
    print("   Hint: Check YAML syntax (indentation, colons, quotes).\n")

    print("3. Missing required fields:")
    print("   ConfigError: Configuration validation failed:")
    print("   Field required [type=missing, input_value=...]\n")

    print("4. Invalid values:")
    print("   ConfigError: Configuration validation failed:")
    print("   Field 'initial_capital' must be greater than 0\n")

    print("5. Undefined environment variables:")
    print("   ValueError: Undefined environment variable(s): DATA_PATH")


def example_environment_variables():
    """Example 4: Environment variable usage."""
    print("\n" + "=" * 60)
    print("Example 4: Environment Variables")
    print("=" * 60)

    print("\nEnvironment variables enable deployment flexibility:\n")

    print("Config file (config.yaml):")
    print("""
data_sources:
  prices:
    path: ${DATA_PATH}/prices.parquet
  signals:
    path: ${MODEL_PATH}/predictions.parquet

execution:
  initial_capital: ${INITIAL_CAPITAL}
""")

    print("\nDevelopment environment:")
    print("""
export DATA_PATH=/local/dev/data
export MODEL_PATH=/local/dev/models
export INITIAL_CAPITAL=100000
""")

    print("\nProduction environment:")
    print("""
export DATA_PATH=/mnt/production/data
export MODEL_PATH=/mnt/production/models
export INITIAL_CAPITAL=10000000
""")

    print("\nSame config file works in both environments!")


def example_best_practices():
    """Example 5: Best practices."""
    print("\n" + "=" * 60)
    print("Example 5: Best Practices")
    print("=" * 60)

    print("\n1. Validate config early:")
    print("""
try:
    config = BacktestConfig.from_yaml(Path("config.yaml"))
    print("✓ Configuration valid")
except ConfigError as e:
    print(f"✗ Configuration error: {e}")
    sys.exit(1)
""")

    print("\n2. Use Parquet for data (fastest, smallest):")
    print("""
data_sources:
  prices:
    path: ${DATA_PATH}/prices.parquet
    format: parquet  # Recommended
""")

    print("\n3. Separate concerns (multiple data sources):")
    print("""
data_sources:
  prices: {...}      # OHLCV data
  signals: {...}     # ML predictions
  features: {...}    # Technical indicators
  context: {...}     # Market-wide data
""")

    print("\n4. Version your configs (store in git):")
    print("""
configs/
├── prod_strategy_v1.yaml
├── prod_strategy_v2.yaml
└── dev_strategy.yaml
""")

    print("\n5. Start simple, add complexity incrementally:")
    print("   - Start: Just prices + basic execution")
    print("   - Add: Signals")
    print("   - Add: Features for risk management")
    print("   - Add: Commission and slippage models")
    print("   - Add: Risk rules")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ML4T Backtest Configuration System - Examples")
    print("=" * 60)

    example_basic_loading()
    example_accessing_values()
    example_error_handling()
    example_environment_variables()
    example_best_practices()

    print("\n" + "=" * 60)
    print("For more information, see:")
    print("  - docs/configuration_guide.md")
    print("  - examples/configs/*.yaml")
    print("  - tests/unit/test_config.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
