# Backtest Configuration Guide

This guide explains how to use the declarative configuration system for ml4t.backtest to specify backtesting setups using YAML or JSON files.

## Table of Contents

- [Quick Start](#quick-start)
- [Configuration Structure](#configuration-structure)
- [Data Sources](#data-sources)
- [Feature Providers](#feature-providers)
- [Risk Rules](#risk-rules)
- [Execution Parameters](#execution-parameters)
- [Environment Variables](#environment-variables)
- [Best Practices](#best-practices)
- [Common Patterns](#common-patterns)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Basic YAML Configuration

```yaml
# minimal_config.yaml
data_sources:
  prices:
    path: /path/to/prices.parquet
    format: parquet

execution:
  initial_capital: 100000
```

### Loading Configuration in Python

```python
from pathlib import Path
from ml4t.backtest.config import BacktestConfig

# Load from YAML
config = BacktestConfig.from_yaml(Path("config.yaml"))

# Load from JSON
config = BacktestConfig.from_json(Path("config.json"))

# Access configuration values
print(config.execution.initial_capital)
print(config.data_sources.prices.path)
```

## Configuration Structure

A backtest configuration consists of four main sections:

1. **data_sources**: Paths to price data, signals, features, and market context
2. **features**: Optional feature provider configuration for indicators and context
3. **risk_rules**: Optional risk management rules (position limits, stops, filters)
4. **execution**: Execution parameters (capital, commission, slippage, margin)

### Complete Structure

```yaml
name: my_strategy                    # Optional strategy name
description: Strategy description    # Optional description

data_sources:
  prices: {...}                      # Required: OHLCV price data
  signals: {...}                     # Optional: ML signals or trading signals
  features: {...}                    # Optional: Precomputed features
  context: {...}                     # Optional: Market-wide context (VIX, SPY)

features:                            # Optional: Feature provider config
  type: precomputed                  # or 'callable'
  path: /path/to/features.parquet
  columns: [atr, rsi, volatility]

execution:
  initial_capital: 100000
  commission: {...}
  slippage: {...}
  enable_margin: false
  max_leverage: 1.0

risk_rules:                          # Optional: Risk management
  max_position_size: 0.5
  stop_loss: 0.05
  take_profit: 0.15
```

## Data Sources

### Price Data (Required)

Price data is the only required data source. It must contain OHLCV (Open, High, Low, Close, Volume) data.

```yaml
data_sources:
  prices:
    path: ${DATA_PATH}/prices.parquet    # Supports env vars
    format: parquet                       # parquet, csv, or hdf5
    timestamp_column: timestamp           # Default: 'timestamp'
    asset_column: asset_id                # Default: 'asset_id'
    columns:                              # Optional: load only these columns
      - timestamp
      - asset_id
      - open
      - high
      - low
      - close
      - volume
```

**Supported Formats:**
- `parquet` (recommended for performance)
- `csv`
- `hdf5`

### Signals (Optional)

Signals represent trading signals from ML models or technical indicators.

```yaml
data_sources:
  signals:
    path: ${DATA_PATH}/ml_predictions.parquet
    format: parquet
    columns:
      - timestamp
      - asset_id
      - signal           # -1.0 to 1.0 (short to long)
      - confidence       # 0.0 to 1.0
      - expected_return  # Optional: predicted return
```

**Signal Convention:**
- `1.0`: Maximum long position
- `0.0`: Neutral / flat
- `-1.0`: Maximum short position

### Features (Optional)

Features are technical indicators or derived metrics used by the strategy for decision-making or risk management.

```yaml
data_sources:
  features:
    path: ${DATA_PATH}/features.parquet
    format: parquet
    columns:
      - timestamp
      - asset_id
      - atr            # Average True Range
      - rsi            # Relative Strength Index
      - volatility     # Historical volatility
      - volume_ma      # Volume moving average
```

### Context (Optional)

Context data contains market-wide information used for regime filtering or risk management.

```yaml
data_sources:
  context:
    path: ${DATA_PATH}/market_context.parquet
    format: parquet
    columns:
      - timestamp
      - vix                # VIX volatility index
      - spy_return         # SPY daily return
      - treasury_yield     # Risk-free rate
      - market_regime      # 0=bear, 1=normal, 2=bull
```

## Feature Providers

Feature providers offer an alternative way to specify features, with two types available:

### Precomputed Features

Use precomputed features from a DataFrame for fast backtesting.

```yaml
features:
  type: precomputed
  path: ${DATA_PATH}/features.parquet
  columns:                    # Optional: select specific features
    - atr
    - rsi
    - volatility
  timestamp_column: timestamp # Default: 'timestamp'
  asset_column: asset_id     # Default: 'asset_id'
```

### Callable Features

Use callable features for on-the-fly computation (useful for live trading).

```yaml
features:
  type: callable
  module: my_module.features    # Python module
  function: compute_features    # Function name
  kwargs:                       # Optional kwargs for function
    lookback: 14
    method: exponential
```

**Callable Function Signature:**
```python
def compute_features(asset_id: str, timestamp: datetime, **kwargs) -> dict[str, float]:
    """Compute features on-the-fly.

    Returns:
        Dictionary of feature_name -> value
    """
    return {
        'atr': 2.5,
        'rsi': 65.0,
        'volatility': 0.02
    }
```

## Risk Rules

Risk rules define position limits, stops, and market filters. This is a basic structure for Phase 1.

```yaml
risk_rules:
  # Position sizing
  max_position_size: 0.5        # Max 50% of portfolio per position
  max_portfolio_heat: 0.25      # Max 25% total portfolio risk

  # Stop loss and profit targets
  stop_loss: 0.05               # 5% stop loss (fraction of entry)
  take_profit: 0.15             # 15% take profit (fraction of entry)

  # Market regime filters
  min_vix: 10.0                 # Don't trade if VIX < 10 (complacency)
  max_vix: 40.0                 # Don't trade if VIX > 40 (crisis)
```

**Important Notes:**
- All size limits are fractions (0.0 to 1.0)
- Stop loss and take profit are fractions of entry price
- VIX filters apply to market-wide trading (requires VIX in context data)
- Advanced risk rules (correlations, sector limits) will be added in Phase 2

## Execution Parameters

Execution parameters control capital, transaction costs, and execution realism.

### Basic Setup

```yaml
execution:
  initial_capital: 100000       # Starting cash
  enable_margin: false          # Allow margin trading?
  max_leverage: 1.0             # Max leverage (1.0 = no leverage)
  execution_delay: true         # Delay fills to prevent lookahead
  allow_immediate_reentry: true # Allow same-bar re-entry
```

### Commission Models

**Per-Share Commission:**
```yaml
execution:
  commission:
    type: per_share
    rate: 0.005      # $0.005 per share
    minimum: 1.0     # $1 minimum per trade
```

**Percentage Commission:**
```yaml
execution:
  commission:
    type: percentage
    rate: 0.001      # 0.1% of trade value
```

**Fixed Commission:**
```yaml
execution:
  commission:
    type: fixed
    rate: 7.0        # $7 per trade
```

**Tiered Commission:**
```yaml
execution:
  commission:
    type: tiered
    rate: 0.0005     # 5 bps for large trades
    minimum: 2.0     # $2 minimum
```

### Slippage Models

**Fixed Slippage:**
```yaml
execution:
  slippage:
    type: fixed
    rate: 0.01       # $0.01 per share
```

**Percentage Slippage:**
```yaml
execution:
  slippage:
    type: percentage
    rate: 0.001      # 0.1% slippage
```

**Volume-Share Slippage:**
```yaml
execution:
  slippage:
    type: volume_share
    rate: 0.1        # 10% of volume share impact
```

## Environment Variables

Use environment variables for sensitive data or deployment-specific paths.

### Syntax

Use `${VARIABLE_NAME}` in any string field:

```yaml
data_sources:
  prices:
    path: ${DATA_PATH}/prices.parquet
  signals:
    path: ${MODEL_OUTPUT_PATH}/predictions.parquet

execution:
  initial_capital: ${INITIAL_CAPITAL}
```

### Setting Environment Variables

**In Shell:**
```bash
export DATA_PATH=/mnt/data/trading
export MODEL_OUTPUT_PATH=/mnt/models/output
export INITIAL_CAPITAL=100000
```

**In Python:**
```python
import os
os.environ['DATA_PATH'] = '/mnt/data/trading'
os.environ['INITIAL_CAPITAL'] = '100000'
```

**In Docker:**
```dockerfile
ENV DATA_PATH=/mnt/data/trading
ENV INITIAL_CAPITAL=100000
```

### Validation

The configuration system validates that all environment variables are defined. If a variable is undefined, you'll get a clear error:

```
ValueError: Undefined environment variable(s): DATA_PATH
Hint: Set the variable with: export DATA_PATH=/path/to/data
```

## Best Practices

### 1. Use Parquet for Data

Parquet is the recommended format for performance and storage efficiency:

```yaml
data_sources:
  prices:
    path: ${DATA_PATH}/prices.parquet
    format: parquet  # Fastest, smallest
```

### 2. Separate Concerns

Keep data sources separate rather than embedding everything in one file:

```yaml
data_sources:
  prices: {...}      # OHLCV data
  signals: {...}     # ML predictions
  features: {...}    # Technical indicators
  context: {...}     # Market-wide data
```

This allows:
- Easier debugging (isolate data issues)
- Parallel data generation (compute signals independently)
- Faster iteration (change signals without reprocessing prices)

### 3. Version Your Configs

Store configs in version control alongside code:

```
project/
├── configs/
│   ├── prod_strategy_v1.yaml
│   ├── prod_strategy_v2.yaml
│   └── dev_strategy.yaml
├── data/
└── src/
```

### 4. Use Environment Variables for Deployment

Different environments should use different data paths:

```yaml
# config.yaml (same for all environments)
data_sources:
  prices:
    path: ${DATA_PATH}/prices.parquet  # Env-specific
```

```bash
# Development
export DATA_PATH=/local/dev/data

# Production
export DATA_PATH=/mnt/production/data
```

### 5. Validate Configs Early

Load and validate configs at the start of your script:

```python
from pathlib import Path
from ml4t.backtest.config import BacktestConfig, ConfigError

try:
    config = BacktestConfig.from_yaml(Path("config.yaml"))
    print("✓ Configuration valid")
except ConfigError as e:
    print(f"✗ Configuration error: {e}")
    sys.exit(1)
```

### 6. Start Simple

Begin with minimal config and add complexity incrementally:

1. Start: Just prices + basic execution
2. Add: Signals
3. Add: Features for risk management
4. Add: Commission and slippage models
5. Add: Risk rules

### 7. Document Your Configs

Add comments explaining strategy logic:

```yaml
name: momentum_top10
description: |
  Weekly rebalancing momentum strategy.
  Ranks universe by 12-month momentum.
  Holds top 10 stocks equally weighted.

data_sources:
  signals:
    path: ${DATA_PATH}/momentum_signals.parquet
    # Signals generated by scripts/generate_momentum_signals.py
    # Recomputed every Sunday for next week
```

## Common Patterns

### Pattern 1: Single-Asset Strategy

Simple strategy for one asset (e.g., AAPL):

```yaml
name: aapl_ma_crossover

data_sources:
  prices:
    path: ${DATA_PATH}/aapl_prices.parquet
    format: parquet
  signals:
    path: ${DATA_PATH}/aapl_ma_signals.parquet
    format: parquet

execution:
  initial_capital: 100000
  commission:
    type: per_share
    rate: 0.005
```

### Pattern 2: Multi-Asset Portfolio

Portfolio strategy with multiple assets:

```yaml
name: sp500_top10_momentum

data_sources:
  prices:
    path: ${DATA_PATH}/sp500_top10_prices.parquet
    format: parquet
  signals:
    path: ${DATA_PATH}/momentum_signals.parquet
    format: parquet
  context:
    path: ${DATA_PATH}/market_context.parquet
    format: parquet

risk_rules:
  max_position_size: 0.15    # Max 15% per position
  max_vix: 40.0              # Don't trade if VIX > 40

execution:
  initial_capital: 1000000
  commission:
    type: percentage
    rate: 0.001
```

### Pattern 3: ML Strategy with Risk Management

ML-based strategy with comprehensive risk controls:

```yaml
name: ml_long_short

data_sources:
  prices:
    path: ${DATA_PATH}/universe_prices.parquet
    format: parquet
  signals:
    path: ${MODEL_PATH}/predictions.parquet
    format: parquet
    columns:
      - timestamp
      - asset_id
      - signal
      - confidence
  context:
    path: ${DATA_PATH}/market_context.parquet
    format: parquet

features:
  type: precomputed
  path: ${DATA_PATH}/risk_features.parquet
  columns:
    - atr
    - volatility
    - volume_ma

execution:
  initial_capital: 10000000
  commission:
    type: tiered
    rate: 0.0005
  slippage:
    type: volume_share
    rate: 0.1
  enable_margin: true
  max_leverage: 2.0

risk_rules:
  max_position_size: 0.05
  max_portfolio_heat: 0.20
  stop_loss: 0.15
  min_vix: 9.0
  max_vix: 50.0
```

### Pattern 4: Research Backtest

Quick research backtest with minimal friction:

```yaml
name: research_test

data_sources:
  prices:
    path: ./data/test_prices.parquet
    format: parquet
  signals:
    path: ./data/test_signals.parquet
    format: parquet

execution:
  initial_capital: 100000
  # No commission/slippage for initial testing
```

## Troubleshooting

### Error: "Configuration file not found"

**Problem:**
```
ConfigError: Configuration file not found: config.yaml
Hint: Check that the path is correct.
```

**Solution:**
- Verify file exists: `ls -la config.yaml`
- Check working directory: `pwd`
- Use absolute path: `/full/path/to/config.yaml`

### Error: "Data file not found"

**Problem:**
```
ValueError: Data file not found: /path/to/prices.parquet
Hint: Check that the path is correct and environment variables are set.
```

**Solution:**
- Verify data file exists: `ls -la /path/to/prices.parquet`
- Check environment variables: `echo $DATA_PATH`
- Validate expanded path after env var substitution

### Error: "Undefined environment variable"

**Problem:**
```
ValueError: Undefined environment variable(s): DATA_PATH
```

**Solution:**
```bash
# Set the variable before loading config
export DATA_PATH=/path/to/data

# Or in Python
import os
os.environ['DATA_PATH'] = '/path/to/data'
```

### Error: "Invalid YAML"

**Problem:**
```
ConfigError: Invalid YAML in config.yaml:
...
Hint: Check YAML syntax (indentation, colons, quotes).
```

**Solution:**
- YAML is indentation-sensitive (use 2 or 4 spaces, not tabs)
- Quote strings with special characters: `path: "C:\\Data\\prices.parquet"`
- Check colons have space after: `key: value` (not `key:value`)
- Use YAML validator: https://www.yamllint.com/

### Error: "Configuration validation failed"

**Problem:**
```
ConfigError: Configuration validation failed for config.yaml:
Field 'initial_capital' must be greater than 0
```

**Solution:**
- Read error message carefully - it tells you what's wrong
- Check field types match expected types (number, string, boolean)
- Verify required fields are present
- Check value constraints (e.g., position size 0.0-1.0)

### Config Loads But Backtest Fails

**Symptom:** Config loads successfully but backtest crashes

**Common Causes:**
1. **Empty data files**: Config validates file exists, not that it has data
2. **Schema mismatch**: File has different columns than expected
3. **Timestamp format**: Timestamps not in expected format
4. **Asset ID mismatch**: Asset IDs in signals don't match prices

**Debug Steps:**
```python
import polars as pl

# Check data schema
prices = pl.read_parquet(config.data_sources.prices.path)
print(prices.schema)
print(prices.head())

# Verify expected columns exist
assert 'timestamp' in prices.columns
assert 'close' in prices.columns

# Check for data
assert prices.height > 0, "Price data is empty"
```

## Examples

See `examples/configs/` for complete working examples:

- `simple_ma_strategy.yaml` - Basic moving average crossover
- `multi_asset_portfolio.yaml` - Multi-asset momentum strategy
- `ml_with_risk.yaml` - ML strategy with comprehensive risk management

## Additional Resources

- **API Documentation**: See docstrings in `ml4t.backtest.config`
- **Test Examples**: See `tests/unit/test_config.py` for usage patterns
- **Schema Reference**: All Pydantic models have detailed field documentation

## Summary

The configuration system enables:

✅ **Declarative backtesting** - Specify what you want, not how to implement it
✅ **Type safety** - Pydantic validates all inputs with clear error messages
✅ **Environment flexibility** - Use env vars for deployment-specific settings
✅ **Versioning** - Track configuration changes alongside code
✅ **Reusability** - Share configs across projects and teams

Start simple, iterate quickly, and scale to production with confidence.
