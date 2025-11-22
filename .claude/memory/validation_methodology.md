# Validation Methodology

**Updated**: 2025-11-22
**Status**: Per-framework validation in isolated environments

## Core Principle

ml4t.backtest should be able to replicate the behavior of established frameworks through configuration options, allowing users migrating from other frameworks to get comparable results.

## Validation Strategy (Post-Cleanup)

**Per-framework in isolated venvs** - NOT unified pytest.

This approach was adopted after two days of dependency conflicts between VectorBT Pro, Backtrader, and Zipline in a single environment.

### Virtual Environments

| Environment | Purpose | Contents |
|-------------|---------|----------|
| `.venv` | Main development | Core dependencies only |
| `.venv-vectorbt-pro` | VectorBT Pro validation | VectorBT Pro 2025.x |
| `.venv-backtrader` | Backtrader validation | Backtrader only |
| `.venv-zipline` | Zipline validation | Zipline-reloaded (low priority) |
| `.venv-validation` | DEPRECATED | Has OSS/Pro conflict |

### Critical: VectorBT OSS/Pro Conflict

VectorBT OSS and Pro CANNOT coexist - both register pandas `.vbt` accessor which conflicts:
```
AttributeError: 'OHLCVDFAccessor' object has no attribute 'has_ohlc'
```

**Solution**: Separate venvs for OSS and Pro.

## Validation Process

### Step 1: Create Validation Script

For each framework, create standalone Python script in `validation/<framework>/`:

```python
# validation/vectorbt_pro/scenario_long_only.py
"""Compare single asset long-only behavior."""

import vectorbtpro as vbt
from ml4t.backtest import Engine, Strategy, DataFeed

# Same test data and signals for both
# Compare: trades, P&L, positions
```

### Step 2: Run in Isolated Environment

```bash
# VectorBT Pro
source .venv-vectorbt-pro/bin/activate
cd validation/vectorbt_pro
python scenario_long_only.py

# Backtrader
source .venv-backtrader/bin/activate
cd validation/backtrader
python scenario_long_only.py
```

### Step 3: Document Configuration

When differences exist, document the ml4t.backtest configuration that replicates each framework's behavior.

## Success Criteria

For each framework comparison:
- **Trade count**: Exact match
- **Trade timestamps**: Match within 1 bar
- **Final P&L**: < 0.1% variance
- **Fill prices**: Within OHLC bounds

## Known Framework Behaviors

### VectorBT
- Vectorized execution (all signals processed)
- `accumulate=False` prevents same-bar re-entry
- Uses close price for fills by default

### Backtrader
- Event-driven, bar-by-bar
- COO (Cheat-on-Open) / COC (Cheat-on-Close) flags
- Integer shares by default

### Zipline
- **EXCLUDED** - uses bundle data, not test DataFrame
- Known ~4.3x difference from other frameworks

## Test Coverage Matrix

| Scenario | VectorBT Pro | Backtrader | Zipline |
|----------|--------------|------------|---------|
| Long only | ⬜ TODO | ⬜ TODO | ❌ N/A |
| Long/Short | ⬜ TODO | ⬜ TODO | ❌ N/A |
| Commission | ⬜ TODO | ⬜ TODO | ❌ N/A |
| Slippage | ⬜ TODO | ⬜ TODO | ❌ N/A |
| Stop-loss | ⬜ TODO | ⬜ TODO | ❌ N/A |

## Files

- `validation/README.md` - Validation strategy overview
- `validation/vectorbt_pro/` - VectorBT Pro scripts (internal only)
- `validation/backtrader/` - Backtrader scripts (open source)

## Historical Note

The old `tests/validation/` with 32,643 lines was deleted in Nov 2025 cleanup due to:
- Persistent dependency conflicts
- Chaotic test organization
- Unreliable test results

The new approach is simpler and more reliable.
