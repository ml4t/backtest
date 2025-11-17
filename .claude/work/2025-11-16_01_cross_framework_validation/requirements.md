# Requirements: Cross-Framework Validation

## Source
- Type: User requirement (from previous handoff)
- Reference: `.claude/transitions/2025-11-16/121417.md`
- Date: 2025-11-16T12:30:00Z

## Context

### What Was Just Completed
✅ **Robust Performance Reporting** (previous work session)
- Trade tracking with entry/exit metadata
- Daily/session-based returns resampling
- BacktestResults export API
- Signal metadata flow (Order → Fill → TradeRecord)

This provides the **foundation** for validation - we can now export standardized data from ml4t.backtest for comparison.

### User Requirement (Previous Session)
**Key insight**: Signals must be generated INDEPENDENTLY, once, outside any framework.

**Validation Approach**:
```
Step 1: Generate signals (ONCE)
  - Calculate SMA crossover, momentum, etc.
  - Save boolean entry/exit to .pkl files
  - NO framework involvement

Step 2: Feed SAME signals to ALL frameworks
  - Each framework receives identical boolean decisions
  - NO indicator calculation in frameworks
  - Only execution/fill logic tested

Step 3: Compare execution results
  - Same trades (timestamp, price, quantity)
  - Same P&L
  - Proves execution equivalence
```

**Why signal-based validation**:
- Different frameworks calculate indicators differently (rounding, edge cases)
- Can't compare if signal generation differs
- Signal-based approach tests execution fidelity ONLY

## Functional Requirements

### 1. Signal Generation Module
- **Location**: `tests/validation/signals/`
- **Purpose**: Pre-calculate trading signals independently of any framework
- **Outputs**: Pickle files with `{data: DataFrame, signals: DataFrame}` format
- **Signal Types**: Boolean entry/exit decisions
- **Data Sources**: Small datasets (crypto from `projects/crypto_futures/data/`)

### 2. Framework Adapter Updates
- **Files**: `tests/validation/frameworks/{qengine,backtrader,vectorbt,zipline}_adapter.py`
- **New Method**: `run_with_signals(data, signals, initial_capital, commission_rate)`
- **Requirement**: NO indicator calculation, only signal execution
- **Interface**: Extend `BaseFrameworkAdapter` with signal-based method

### 3. Validation Comparison
- **Compare**: Trades (timestamp, price, quantity, side)
- **Compare**: Final portfolio value and P&L
- **Tolerance**: Within acceptable precision (0.002% variance from previous tests)
- **Frameworks**: ml4t.backtest, Backtrader, VectorBT, (Zipline deferred)

### 4. Test Runner
- **File**: `tests/validation/run_signal_validation.py`
- **Purpose**: Run same signals through all frameworks and compare results
- **Output**: Validation report showing match/mismatch across frameworks
- **Success Criteria**: Identical trades within tolerance

## Non-Functional Requirements

### Performance
- Small datasets (BTC, ETH, SOL daily) - testing correctness, not scale
- Validation should complete in <1 minute for single asset

### Maintainability
- Use existing `tests/validation/` infrastructure (no duplication)
- Follow existing adapter pattern
- Document signal format clearly

### Compatibility
- Backward compatible with existing adapters (keep `run_backtest()` method)
- New `run_with_signals()` method is addition, not replacement

## Existing Infrastructure

### Located in `tests/validation/`
✅ Framework adapters:
- `frameworks/qengine_adapter.py` (ml4t.backtest wrapper)
- `frameworks/backtrader_adapter.py` (Backtrader wrapper)
- `frameworks/vectorbt_adapter.py` (VectorBT wrapper)
- `frameworks/zipline_adapter.py` (Zipline wrapper)

✅ Base classes:
- `frameworks/base.py` - `BaseFrameworkAdapter`, `ValidationResult`, `TradeRecord`

✅ Zipline infrastructure:
- `bundles/validation_ingest.py` - Data bundle creation
- `bundles/extension.py` - Zipline extension registration

⚠️ **Current limitation**: Adapters use `strategy_params` (each calculates indicators)

## Acceptance Criteria

### Phase 1: Signal Generation (Complete when)
- [ ] `tests/validation/signals/generate.py` exists
- [ ] BTC SMA(10,20) crossover signals generated
- [ ] Signals saved to `signals/btc_sma_crossover_daily.pkl`
- [ ] Signal format documented and validated

### Phase 2: Adapter Updates (Complete when)
- [ ] `BaseFrameworkAdapter.run_with_signals()` method added
- [ ] ml4t.backtest adapter implements signal execution
- [ ] Backtrader adapter implements signal execution
- [ ] VectorBT adapter implements signal execution
- [ ] All adapters pass smoke test with same signals

### Phase 3: Validation (Complete when)
- [ ] Test runner compares results across frameworks
- [ ] ml4t.backtest matches Backtrader within 0.002%
- [ ] ml4t.backtest matches VectorBT within 0.002%
- [ ] Trade count matches exactly
- [ ] Trade timestamps match exactly
- [ ] Validation report generated

### Phase 4: Documentation (Complete when)
- [ ] Signal generation process documented
- [ ] Adapter API documented
- [ ] Validation results documented
- [ ] Known limitations documented (Zipline deferred)

## Out of Scope

### Explicitly NOT Included
- ❌ Indicator calculation in frameworks (defeats purpose)
- ❌ Zipline validation (deferred - bundle data issues)
- ❌ Multi-asset validation (focus on single asset first)
- ❌ Minute-bar data (daily only)
- ❌ Complex strategies (SMA crossover sufficient for validation)

### Future Work (Not This Work Unit)
- Multi-asset signal validation
- Multiple signal types (momentum, mean reversion)
- Minute/hour frequency testing
- Zipline integration (after bundle resolved)

## Dependencies

### External Dependencies
- **Backtrader**: `>=1.9.78.123` (already installed)
- **VectorBT Pro**: `>=2.0.5` (already installed)
- **Zipline-reloaded**: `>=3.0.4` (installed but deferred)
- **Crypto data**: `/home/stefan/ml4t/projects/crypto_futures/data/`

### Internal Dependencies
- **New BacktestResults API** (just implemented) - provides standardized export
- **Existing validation infrastructure** - `tests/validation/frameworks/`
- **Trade metadata tracking** (just implemented) - enables signal reason tracking

## Risks and Assumptions

### Risks
1. **Framework API differences**: Each framework has unique order submission patterns
   - **Mitigation**: Use adapters to normalize interfaces

2. **Fill model differences**: Frameworks may handle fills differently
   - **Mitigation**: Use simple market orders only for validation

3. **Precision differences**: Floating point rounding across frameworks
   - **Mitigation**: Define acceptable tolerance (0.002% from previous tests)

4. **VectorBT complexity**: NumPy-based, different paradigm from event-driven
   - **Mitigation**: Simple signals reduce complexity

### Assumptions
1. **Crypto data exists**: `/home/stefan/ml4t/projects/crypto_futures/data/` has BTC/ETH/SOL
2. **Daily frequency sufficient**: Don't need minute bars for validation
3. **SMA crossover sufficient**: Don't need complex strategies
4. **Exact timestamp matching**: All frameworks use same OHLC bars
5. **Commission models align**: All frameworks support percentage commission

## Technical Notes

### Signal Format Specification
```python
{
    'data': pd.DataFrame(  # OHLCV data
        index=DatetimeIndex,
        columns=['open', 'high', 'low', 'close', 'volume']
    ),
    'signals': pd.DataFrame(  # Boolean entry/exit
        index=DatetimeIndex,  # MUST match data index
        columns=['entry', 'exit']  # Both boolean
    ),
    'metadata': {
        'signal_type': 'sma_crossover',
        'parameters': {'fast': 10, 'slow': 20},
        'generated_at': '2025-11-16T12:30:00Z'
    }
}
```

### Validation Result Schema
```python
@dataclass
class ValidationResult:
    framework: str
    trades: list[TradeRecord]
    final_value: float
    total_return: float
    num_trades: int
    execution_time: float
    metadata: dict[str, Any]
```

### Comparison Tolerance
- **Trade count**: Must match exactly
- **Trade timestamps**: Must match exactly
- **Trade prices**: Within 0.01% (fill model tolerance)
- **Final P&L**: Within 0.002% (from previous cross-framework tests)
- **Total return**: Within 0.002%

## Reference Materials

### Previous Work
- **Handoff doc**: `.claude/transitions/2025-11-16/121417.md`
- **Signal plan**: `tests/validation/SIGNAL_BASED_VALIDATION_PLAN.md`
- **Architecture**: `tests/validation/SIGNAL_VALIDATION_ARCHITECTURE.md`
- **Reporting infrastructure**: `.claude/memory/reporting_infrastructure.md` (NEW)

### Example Code
- **ML3T Zipline**: `/home/stefan/ml3t/ch11_strategy_backtesting/code/04_ml4t_workflow_with_zipline`
- **ML3T Backtrader**: `/home/stefan/ml3t/ch11_strategy_backtesting/code/03_backtesting_with_backtrader.py`

### Data Sources
- **Crypto**: `/home/stefan/ml3t/projects/crypto_futures/data/`
- **Test scenarios**: `tests/validation/scenarios/`

## Next Action

**Recommended**: Run `/plan` to create detailed task breakdown with dependencies.

The requirements are well-defined from previous session, but implementation needs:
1. Detailed task dependencies (signal gen before adapter updates)
2. Testing strategy for each phase
3. Validation criteria for each task
4. Time estimates for each phase

The `/plan` command will create a structured implementation plan with all tasks properly sequenced.
