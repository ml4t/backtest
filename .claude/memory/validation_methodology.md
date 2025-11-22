# Validation Methodology

**Updated**: 2025-11-22
**Status**: Per-framework validation in isolated environments

## Core Principle

ml4t.backtest should be able to replicate the behavior of established frameworks through **configuration options**, not defaults. Users should be able to match any framework's exact behavior when needed.

## Validation Strategy

**Scenario-based, per-framework validation** in isolated virtual environments.

### Why This Approach

1. **Dependency conflicts**: VectorBT OSS/Pro, Backtrader, and Zipline cannot coexist
2. **Different semantics**: Each framework has unique execution models
3. **Clear comparisons**: Isolated scripts show exact behavior matching

### Virtual Environments (All Pre-Created)

| Environment | Contents | Status |
|-------------|----------|--------|
| `.venv` | Main development | Active |
| `.venv-vectorbt-pro` | VectorBT Pro 2025.x | **Installed** |
| `.venv-backtrader` | Backtrader only | Available |
| `.venv-zipline` | Zipline-reloaded | Low priority |
| `.venv-validation` | DEPRECATED | OSS/Pro conflict |

### Critical: VectorBT OSS/Pro Conflict

VectorBT OSS and Pro CANNOT coexist - both register pandas `.vbt` accessor:
```
AttributeError: 'OHLCVDFAccessor' object has no attribute 'has_ohlc'
```

## Validation Scenarios

Each scenario tests a specific behavior dimension:

| Scenario | Tests |
|----------|-------|
| `scenario_01_long_only` | Basic long entries and exits |
| `scenario_02_long_short` | Position flipping, shorts |
| `scenario_03_with_commission` | Commission impact on P&L |
| `scenario_04_with_slippage` | Slippage impact on fills |
| `scenario_05_stop_orders` | Stop-loss execution timing |
| `scenario_06_multi_asset` | Cross-asset behavior |

## Validation Process

### Step 1: Create Scenario Script

```python
# validation/vectorbt_pro/scenario_01_long_only.py
"""Compare single asset long-only behavior."""

import vectorbtpro as vbt
from ml4t.backtest import Engine, Strategy, DataFeed

# Identical data and signals for both frameworks
# Run both, compare trade-by-trade
```

### Step 2: Run in Isolated Environment

```bash
source .venv-vectorbt-pro/bin/activate
python validation/vectorbt_pro/scenario_01_long_only.py
```

### Step 3: Document Configuration

When ml4t.backtest differs from external framework, document:
1. What configuration option matches the external behavior
2. Why the default differs (if it does)

## Success Criteria

For each scenario:
- **Trade count**: Exact match
- **Trade timestamps**: Exact match (within execution mode semantics)
- **Fill prices**: Match (or within slippage model bounds)
- **Final P&L**: Exact match (accounting for floating point)

## Framework Behaviors to Match

### VectorBT Pro
- Vectorized: processes all signals at once
- Same-bar fills at close price (default)
- Fractional shares allowed
- `accumulate=False` prevents re-entry on same bar

### Backtrader
- Event-driven: bar-by-bar processing
- Next-bar fills at open (default)
- Integer shares only
- COO/COC flags modify execution timing

### Zipline (Excluded)
- Uses bundle data system (incompatible with custom DataFrames)
- Different price universe causes ~4.3x P&L difference
- Not practical to validate against

## Files

```
validation/
├── README.md                    # Validation overview
├── vectorbt_pro/                # VectorBT Pro scenarios
│   ├── scenario_01_long_only.py
│   └── ...
└── backtrader/                  # Backtrader scenarios
    ├── scenario_01_long_only.py
    └── ...
```

## Historical Note

The old `tests/validation/` (32,643 lines) was deleted due to:
- Dependency conflicts making tests unreliable
- Chaotic organization spanning 152 files
- No clear framework isolation

The new approach is simpler, isolated, and reproducible.
