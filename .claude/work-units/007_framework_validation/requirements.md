# Work Unit 007: Framework Validation

**Created**: 2025-11-22
**Status**: Active
**Priority**: High

## Objective

Validate ml4t.backtest against VectorBT Pro and Backtrader using scenario-based testing in isolated environments.

## Success Criteria

For each validation scenario, ml4t.backtest must produce **100% matching results** when configured appropriately:
- Trade count: Exact match
- Trade timestamps: Exact match
- Fill prices: Exact match
- Final P&L: Exact match (within floating point tolerance)

## Scope

### In Scope

1. **VectorBT Pro validation** (`.venv-vectorbt-pro`)
   - Long-only scenarios
   - Long/short scenarios
   - Commission handling
   - Multi-asset behavior

2. **Backtrader validation** (`.venv-backtrader`)
   - Long-only scenarios
   - Long/short scenarios
   - Commission handling
   - Next-bar execution semantics

3. **Documentation**
   - Configuration options needed to match each framework
   - Behavioral differences and their causes
   - Example usage for users migrating from other frameworks

### Out of Scope

- Zipline validation (bundle data incompatibility, AD-001)
- Performance benchmarking (separate effort)
- New feature development

## Validation Scenarios

| ID | Scenario | Description |
|----|----------|-------------|
| 01 | Long-only | Single asset, buy/sell signals, no shorts |
| 02 | Long/short | Position flipping, short selling (margin) |
| 03 | Commission | Per-share and percentage commission |
| 04 | Slippage | Fixed and percentage slippage |
| 05 | Stop orders | Stop-loss execution timing |
| 06 | Multi-asset | Multiple assets with unified portfolio |

## Technical Requirements

### Environment Setup

```bash
# VectorBT Pro (already installed)
source .venv-vectorbt-pro/bin/activate
pip list | grep vectorbt  # Verify Pro installed

# Backtrader
source .venv-backtrader/bin/activate
pip install backtrader
```

### Script Structure

Each scenario creates:
1. Identical test data and signals
2. Runs ml4t.backtest with specific config
3. Runs external framework
4. Compares results trade-by-trade
5. Reports match/mismatch with details

## Deliverables

1. `validation/vectorbt_pro/scenario_*.py` - VectorBT Pro validation scripts
2. `validation/backtrader/scenario_*.py` - Backtrader validation scripts
3. `validation/README.md` - How to run validation
4. Updated `BacktestConfig` presets if needed

## Dependencies

- VectorBT Pro license (internal only)
- Backtrader (open source)
- ml4t.backtest core engine (complete)

## Estimated Effort

- Scenario 01-02: 1-2 hours each
- Scenario 03-04: 1 hour each
- Scenario 05-06: 2 hours each
- Documentation: 2 hours

Total: ~10-12 hours
