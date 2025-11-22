# ml4t.backtest Validation Strategy

## Overview

Framework validation is performed **per-framework** in **isolated environments**, NOT through a unified pytest suite.

This approach was adopted after two days of struggling with dependency conflicts between VectorBT Pro, Backtrader, and Zipline-Reloaded in a single environment.

## Key Principles

1. **Separate venvs per framework** - Each framework has its own virtual environment
2. **VectorBT Pro is internal only** - Cannot be distributed to users
3. **Validation scripts, not pytest** - Manual verification with clear outputs
4. **Identical signals** - Test with pre-computed signals to eliminate strategy variance
5. **Configuration-based matching** - Document what config produces matching results

## Framework-Specific Validation

### VectorBT Pro

**Location**: `validation/vectorbt_pro/`
**Environment**: `.venv-vectorbt-pro`

VectorBT Pro validation is:
- Internal only (commercial license)
- Run manually, not in CI
- Used to verify our implementation matches their behavior

**Scenarios to validate**:
1. Single asset, long only
2. Single asset, long/short
3. Multi-asset equal weighted
4. Commission (percentage, per-share)
5. Slippage (fixed, percentage)
6. Stop-loss and take-profit
7. Trailing stops

### Backtrader

**Location**: `validation/backtrader/`
**Environment**: `.venv-backtrader`

Backtrader validation:
- Can be integrated into CI (open source)
- Test with COO/COC flags
- Document which ml4t.backtest config matches Backtrader defaults

### Zipline

**Location**: `validation/zipline/` (optional)
**Environment**: `.venv-zipline`

Zipline validation:
- Known issues with bundle data (uses its own data, not test data)
- Lower priority than VectorBT Pro and Backtrader
- May skip for now

## Test Coverage Matrix

| Feature | VectorBT Pro | Backtrader | Zipline |
|---------|--------------|------------|---------|
| Long only | ⬜ | ⬜ | ⬜ |
| Long/Short | ⬜ | ⬜ | ⬜ |
| Multi-asset | ⬜ | ⬜ | ⬜ |
| % Commission | ⬜ | ⬜ | ⬜ |
| Per-share commission | ⬜ | ⬜ | ⬜ |
| Fixed slippage | ⬜ | ⬜ | ⬜ |
| % Slippage | ⬜ | ⬜ | ⬜ |
| Stop-loss | ⬜ | ⬜ | ⬜ |
| Take-profit | ⬜ | ⬜ | ⬜ |
| Trailing stop | ⬜ | ⬜ | ⬜ |

## Configuration Presets

Once validation is complete, we document configuration presets that reproduce each framework's behavior:

```python
from ml4t.backtest import BacktestConfig

# Reproduce VectorBT behavior
config = BacktestConfig.from_preset("vectorbt")

# Reproduce Backtrader behavior
config = BacktestConfig.from_preset("backtrader")

# Reproduce Zipline behavior
config = BacktestConfig.from_preset("zipline")
```

## How to Run Validation

### VectorBT Pro
```bash
cd validation/vectorbt_pro
source ../../.venv-vectorbt-pro/bin/activate
python run_scenarios.py
```

### Backtrader
```bash
cd validation/backtrader
source ../../.venv-backtrader/bin/activate
python run_scenarios.py
```

## Success Criteria

For each framework, we aim for:
- **Trade count**: Exact match
- **Trade timestamps**: Match within 1 bar (due to execution timing differences)
- **Final P&L**: < 0.1% variance
- **Individual fills**: Price within OHLC bounds

When discrepancies exist:
1. Document the difference
2. Identify the root cause (execution model, fill price, etc.)
3. Add configuration option to replicate if desired
4. Document in preset that reproduces that behavior

## Future: CI Integration

Once per-framework validation is stable:
1. Add optional CI jobs for Backtrader (open source only)
2. Run on PR/release, not every commit
3. Keep VectorBT Pro validation manual/internal
