# ml4t.backtest Configuration Examples

This directory contains example YAML configuration files demonstrating common backtesting scenarios with ml4t.backtest.

## Overview

Configuration files enable **declarative backtesting** - specify your strategy setup in YAML/JSON instead of Python code. Benefits:

- **Type-safe**: Pydantic validation catches errors before runtime
- **Portable**: Share configs across teams, version control strategy parameters
- **Reproducible**: Same config → same results (given same data)
- **Clear**: Separate strategy logic (Python) from parameters (YAML)

## Available Examples

### 1. `simple_strategy.yaml` - Basic Single-Asset Strategy

**What it demonstrates:**
- Simple time-based exit (60 bars)
- Fixed stop loss (2%)
- No ML signals, just price-based rules
- Minimal configuration

**Use cases:**
- Learning the config system
- Testing basic execution logic
- Validating data pipeline

**Expected runtime:** <1 second

**How to use:**
```python
from pathlib import Path
from ml4t.backtest.config import BacktestConfig

# Load config
config = BacktestConfig.from_yaml(Path("examples/configs/simple_strategy.yaml"))

# Inspect parameters
print(f"Initial capital: ${config.execution.initial_capital:,.0f}")
print(f"Commission: ${config.execution.commission.rate} per share")
print(f"Stop loss: {config.risk_rules.stop_loss * 100}%")

# Use with engine (when implemented)
# from ml4t.backtest.engine import BacktestEngine
# engine = BacktestEngine.from_config(config)
# results = engine.run()
```

---

### 2. `ml_strategy.yaml` - ML-Driven Strategy

**What it demonstrates:**
- Precomputed ML signals (ml_score from trained model)
- Precomputed features (ATR for volatility-based stops)
- VIX-based regime filtering
- Single-asset or small-universe application

**Use cases:**
- ML model deployment
- Signal-based trading
- Feature integration

**Expected runtime:** <5 seconds

**Data requirements:**
- `ml_score` column in signals/features data
- `atr` column for volatility-scaled stops
- `vix` column in context data for regime filtering

**Strategy implementation example:**
```python
from ml4t.backtest.strategy.base import Strategy
from ml4t.backtest.core.event import MarketEvent

class MLStrategy(Strategy):
    def on_market_data(self, event: MarketEvent):
        # Get ML signal from event
        ml_score = event.signals.get('ml_score', 0.0)

        # Get VIX from context
        vix = event.context.get('vix', 0.0)

        # Apply VIX filter (from config)
        if vix > self.config.risk_rules.max_vix:
            return  # Don't trade during high volatility

        # Trade on ML signal
        if ml_score > 0.6 and not self.has_position(event.asset_id):
            self.buy_percent(event.asset_id, 0.10)  # 10% of NAV

        elif ml_score < 0.4 and self.has_position(event.asset_id):
            self.close_position(event.asset_id)
```

---

### 3. `ml_risk_multiasset.yaml` - Production Multi-Asset Strategy

**What it demonstrates:**
- 500-stock universe, select top 25 by ML scores
- Precomputed ML signals and features (ATR)
- VIX-based regime filtering
- Multiple integrated risk rules (volatility-scaled, trailing, time-based)
- Position sizing and rebalancing

**Use cases:**
- Production ML trading systems
- Portfolio optimization
- Multi-asset research

**Expected runtime:** 30-60 seconds for 126,000 events

**This is the reference implementation** - maps directly to `examples/integrated/top25_ml_strategy_complete.py`

**Key features:**
- **Universe**: 500 stocks loaded, strategy selects top 25
- **Ranking**: Sort by `ml_score` descending
- **Position sizing**: Equal weight (4% each)
- **Risk management**: 3 integrated rules
  - VolatilityScaledStopLoss: Stop at entry - 2.0 × ATR
  - DynamicTrailingStop: Starts at 5%, tightens 0.1%/bar to min 0.5%
  - TimeBasedExit: Force exit after 60 bars
- **Regime filter**: Don't trade if VIX > 30

**Strategy implementation pattern:**
```python
class Top25MLStrategy(Strategy):
    def on_market_data(self, events: List[MarketEvent]):
        # Get VIX from context
        vix = events[0].context.get('vix', 0.0)

        # VIX filter
        if vix > 30.0:
            return  # Don't rebalance during high volatility

        # Rank all stocks by ML score
        scores = [(e.asset_id, e.signals['ml_score'], e.close)
                  for e in events if e.signals.get('atr') is not None]
        scores.sort(key=lambda x: x[1], reverse=True)

        # Select top 25
        top_25 = scores[:25]

        # Target weights: 4% each
        target_weights = {asset_id: 0.04 for asset_id, _, _ in top_25}

        # Rebalance portfolio
        self.rebalance(target_weights)
```

---

## Configuration Schema

### Top-Level Structure

```yaml
name: "Strategy Name"  # Optional
description: "Strategy description"  # Optional

data_sources:
  prices: {...}    # REQUIRED: OHLCV bars
  signals: {...}   # OPTIONAL: ML signals
  features: {...}  # OPTIONAL: Precomputed features
  context: {...}   # OPTIONAL: Market context (VIX, SPY)

features: {...}    # OPTIONAL: Feature provider config

risk_rules: {...}  # OPTIONAL: Risk management rules

execution: {...}   # REQUIRED: Execution parameters
```

### Data Sources

Each data source requires:
- `path`: File path (supports `${ENV_VAR}` expansion)
- `format`: File format (`parquet`, `csv`, `hdf5`)
- `timestamp_column`: Name of timestamp column (default: `timestamp`)
- `asset_column`: Name of asset ID column (default: `asset_id`)

Optional:
- `columns`: List of columns to load (default: load all)
- `filters`: Filter expressions (e.g., `["asset_id == 'AAPL'"]`)

### Feature Provider

Two types:

**Precomputed** (load from file):
```yaml
features:
  type: precomputed
  path: "data/features.parquet"
  columns: [atr, rsi, volatility]
  timestamp_column: timestamp
  asset_column: asset_id
```

**Callable** (compute on-the-fly):
```yaml
features:
  type: callable
  module: my_features  # Python module
  function: compute_features  # Function name
  kwargs:
    lookback: 20
    indicators: [rsi, macd]
```

### Risk Rules

All optional:
```yaml
risk_rules:
  max_position_size: 0.1      # Max 10% of NAV per position
  stop_loss: 0.02             # 2% stop loss
  take_profit: 0.05           # 5% take profit
  max_portfolio_heat: 0.15    # Max 15% portfolio risk
  min_vix: 10.0               # Min VIX to trade
  max_vix: 30.0               # Max VIX to trade
```

### Execution Parameters

```yaml
execution:
  initial_capital: 100000.0

  commission:
    type: per_share  # per_share, percentage, fixed, tiered
    rate: 0.005      # $0.005 per share
    minimum: 1.0     # $1 minimum

  slippage:
    type: percentage  # fixed, percentage, volume_share
    rate: 0.001       # 0.1% slippage

  enable_margin: false
  max_leverage: 1.0
  execution_delay: true  # Prevent lookahead bias
  allow_immediate_reentry: true
```

---

## Environment Variables

Configs support environment variable expansion for paths:

```yaml
data_sources:
  prices:
    path: "${DATA_PATH}/prices.parquet"
```

Set before loading:
```bash
export DATA_PATH=/path/to/data
```

Or in Python:
```python
import os
os.environ['DATA_PATH'] = '/path/to/data'

config = BacktestConfig.from_yaml(Path("config.yaml"))
```

---

## Validation

All configs are validated on load:

```python
from ml4t.backtest.config import BacktestConfig, ConfigError

try:
    config = BacktestConfig.from_yaml(Path("config.yaml"))
except ConfigError as e:
    print(f"Configuration error: {e}")
```

**Common validation errors:**
- `Data file not found` - Check path, ensure file exists
- `Undefined environment variable` - Set required `${ENV_VAR}`
- `min_vix > max_vix` - Fix VIX range
- `Invalid YAML syntax` - Check indentation, colons, quotes

---

## Testing Configurations

Run the validation script to test all configs:

```bash
cd examples/configs
python test_configs.py
```

This script:
1. Loads each config
2. Validates against Pydantic schema
3. Checks data files exist
4. Runs a minimal backtest
5. Reports results

---

## Creating Your Own Configs

### Step 1: Start with a template

Copy the example that best matches your use case:
- Simple strategy → `simple_strategy.yaml`
- ML signals → `ml_strategy.yaml`
- Multi-asset portfolio → `ml_risk_multiasset.yaml`

### Step 2: Customize paths

Update `data_sources.prices.path` to point to your data:
```yaml
data_sources:
  prices:
    path: "/absolute/path/to/your/data.parquet"
    # Or use environment variable:
    # path: "${MY_DATA_DIR}/prices.parquet"
```

### Step 3: Adjust parameters

Modify execution and risk parameters:
```yaml
execution:
  initial_capital: 50000.0  # Your starting capital
  commission:
    rate: 0.001  # Your broker's commission
  slippage:
    rate: 0.002  # Your estimated slippage

risk_rules:
  stop_loss: 0.015  # Tighter stop (1.5%)
  max_position_size: 0.05  # Smaller positions (5%)
```

### Step 4: Validate

```python
from pathlib import Path
from ml4t.backtest.config import BacktestConfig

config = BacktestConfig.from_yaml(Path("my_config.yaml"))
print("✓ Config valid!")
```

### Step 5: Test

Run a backtest with your config to ensure everything works.

---

## Common Patterns

### Pattern 1: Single Asset Trading

```yaml
data_sources:
  prices:
    path: "data/prices.parquet"
    filters: ["asset_id == 'AAPL'"]  # Trade only AAPL

risk_rules:
  max_position_size: 1.0  # 100% (single asset)
```

### Pattern 2: ML Signal Integration

```yaml
data_sources:
  prices:
    path: "data/prices.parquet"
  signals:
    path: "data/ml_predictions.parquet"
    columns: [timestamp, asset_id, signal, confidence]

features:
  type: precomputed
  path: "data/features.parquet"
  columns: [atr, rsi, regime]
```

### Pattern 3: VIX Filtering

```yaml
data_sources:
  context:
    path: "data/vix.parquet"
    columns: [timestamp, vix]

risk_rules:
  min_vix: 12.0
  max_vix: 25.0
```

### Pattern 4: Margin Trading

```yaml
execution:
  enable_margin: true
  max_leverage: 2.0  # 2× leverage

risk_rules:
  max_position_size: 0.5  # 50% per position
  max_portfolio_heat: 0.25  # 25% max risk
```

---

## Best Practices

### 1. Start Simple
Begin with `simple_strategy.yaml`, validate your data pipeline, then add complexity.

### 2. Use Environment Variables
Don't hardcode paths. Use `${DATA_PATH}` for portability.

### 3. Version Control Configs
Commit configs to git, track parameter changes over time.

### 4. Document Custom Configs
Add comments explaining non-obvious parameter choices.

### 5. Test Thoroughly
Validate configs with `test_configs.py` before production use.

### 6. Prevent Lookahead Bias
Always use `execution_delay: true` for realistic backtests.

### 7. Use Realistic Costs
Set commission and slippage to match your broker (or higher for safety margin).

### 8. Validate ML Signals
Before trusting ML scores, backtest with and without to measure alpha.

---

## Troubleshooting

### Issue: "Data file not found"

**Cause:** Incorrect path or file doesn't exist

**Fix:**
- Check path is correct
- Use absolute path or set environment variable
- Verify file exists: `ls -l /path/to/file.parquet`

### Issue: "Undefined environment variable"

**Cause:** `${VAR}` referenced but not set

**Fix:**
```bash
export VAR=/path/to/data
```
Or in Python:
```python
os.environ['VAR'] = '/path/to/data'
```

### Issue: "Invalid YAML syntax"

**Cause:** Indentation, colons, or quotes incorrect

**Fix:**
- Check indentation (use spaces, not tabs)
- Ensure colons have space after: `key: value`
- Quote strings with special chars: `path: "my/path"`

### Issue: "ATR feature not available"

**Cause:** Feature provider doesn't have `atr` column

**Fix:**
- Add `atr` to `features.columns`
- Verify your data has `atr` column
- Or remove volatility-scaled stop rule

### Issue: "VIX filtering not working"

**Cause:** Context data not loaded or wrong column name

**Fix:**
- Verify `data_sources.context.path` is correct
- Check column name is `vix` (case-sensitive)
- Inspect event.context in strategy code

---

## Next Steps

1. **Run validation script**: `python test_configs.py`
2. **Try simple example**: Load and inspect `simple_strategy.yaml`
3. **Customize for your data**: Copy and modify an example
4. **Integrate with strategy**: Write strategy code that uses config
5. **Backtest**: Run full backtest with your config

---

## Additional Resources

- **Config schema**: `src/ml4t/backtest/config.py` (Pydantic models)
- **Example strategy**: `examples/integrated/top25_ml_strategy_complete.py`
- **API docs**: `docs/api/config.md` (when available)
- **Framework docs**: `README.md` at project root

---

**Questions or issues?** Check the main project README or open an issue.
