# Exploration Summary: Cross-Framework Validation

**Date**: 2025-11-16
**Work Unit**: 2025-11-16_01_cross_framework_validation
**Status**: Requirements analyzed, ready for planning

## Executive Summary

Cross-framework validation is **well-defined** and **ready for implementation**. The previous session laid excellent groundwork with:
1. Clear signal-based validation architecture
2. Existing adapter infrastructure in `tests/validation/`
3. Small, well-scoped datasets identified

**New context**: Just completed robust performance reporting infrastructure, which provides standardized export format for comparison.

**Recommendation**: Proceed to `/plan` for detailed task breakdown.

---

## Codebase Analysis

### Existing Infrastructure (Very Solid Foundation)

**Location**: `tests/validation/`

#### 1. Framework Adapters (`frameworks/`)
```
frameworks/
├── base.py                    # BaseFrameworkAdapter, ValidationResult, TradeRecord
├── qengine_adapter.py        # ml4t.backtest wrapper
├── backtrader_adapter.py     # Backtrader wrapper
├── vectorbt_adapter.py       # VectorBT wrapper
├── zipline_adapter.py        # Zipline wrapper
└── __init__.py
```

**Current State**:
- ✅ Adapters exist for all 4 frameworks
- ✅ Common interface (`BaseFrameworkAdapter`)
- ✅ Standardized result format (`ValidationResult`)
- ⚠️ **Limitation**: Current `run_backtest()` method uses `strategy_params` (each framework calculates indicators)
- 🎯 **Need**: Add `run_with_signals()` method for signal-based validation

#### 2. Zipline Bundle Infrastructure (`bundles/`)
```
bundles/
├── validation_ingest.py      # Data bundle creation
├── extension.py              # Zipline extension registration
└── __init__.py
```

**Status**: Exists but deferred (bundle data incompatibility issues)

#### 3. Test Scenarios (`scenarios/`)
```
scenarios/
└── scenario_001_simple_market_orders.py  # Example test scenario
```

**Observation**: Shows pattern for test organization

### New Infrastructure (Just Implemented)

#### Performance Reporting (Session Earlier Today)

**Files Modified**:
- `src/ml4t/backtest/execution/trade_tracker.py` - Entry/exit metadata capture
- `src/ml4t/backtest/portfolio/analytics.py` - Daily returns resampling
- `src/ml4t/backtest/strategy/base.py` - Metadata support in `rebalance_to_weights()`
- `src/ml4t/backtest/engine.py` - `get_results()` method
- `src/ml4t/backtest/results.py` - **NEW** - BacktestResults export API

**Relevance to Validation**:
- ✅ Standardized trade export format
- ✅ Metadata tracking (entry/exit reasons)
- ✅ Daily returns resampling
- ✅ Polars DataFrames for efficient comparison

This provides **identical export format** for ml4t.backtest that we can compare against other frameworks.

### Data Sources

#### Available Data
```
/home/stefan/ml3t/projects/crypto_futures/data/
├── BTC daily bars
├── ETH daily bars
└── SOL daily bars
```

**Characteristics**:
- Daily frequency (suitable for validation)
- Small datasets (fast execution)
- Clean, reliable data

#### Reference Implementations
```
/home/stefan/ml3t/ch11_strategy_backtesting/code/
├── 04_ml4t_workflow_with_zipline/       # Zipline example
└── 03_backtesting_with_backtrader.py    # Backtrader example
```

**Value**: Real-world examples of framework usage patterns

---

## Implementation Approach

### Overall Strategy

**4-Phase Implementation**:
1. **Signal Generation** - Pre-calculate signals independently
2. **Adapter Updates** - Add signal-based execution to each adapter
3. **Test Runner** - Automated comparison across frameworks
4. **Validation** - Verify results match within tolerance

**Key Principle**: **No framework calculates indicators** - all receive identical boolean signals

### Phase 1: Signal Generation (Est: 2 hours)

**Create**: `tests/validation/signals/generate.py`

**Purpose**: Generate trading signals ONCE, save to disk

**Signal Format**:
```python
{
    'data': pd.DataFrame(index=DatetimeIndex, columns=['open','high','low','close','volume']),
    'signals': pd.DataFrame(index=DatetimeIndex, columns=['entry','exit']),  # Boolean
    'metadata': {'signal_type': 'sma_crossover', 'parameters': {'fast': 10, 'slow': 20}}
}
```

**Signals to Generate**:
1. **BTC SMA(10,20) crossover** - Simple, well-understood
2. Save to: `signals/btc_sma_crossover_daily.pkl`

**Why SMA Crossover**:
- Simple logic (reduces framework-specific complexity)
- Well-defined entry/exit points
- Easy to verify manually

### Phase 2: Adapter Updates (Est: 4 hours)

**Modify**: `tests/validation/frameworks/base.py`

Add abstract method:
```python
@abstractmethod
def run_with_signals(
    self,
    data: pd.DataFrame,
    signals: pd.DataFrame,  # Boolean entry/exit
    initial_capital: float = 10000,
    commission_rate: float = 0.001
) -> ValidationResult:
    """Execute pre-calculated signals (NO indicator calculation)."""
```

**Implement in Each Adapter**:

1. **ml4t.backtest Adapter** (`qengine_adapter.py`)
   - Create signal-only strategy class
   - Strategy reads signals from dict
   - Submit orders based on boolean values
   - Export via new `BacktestResults` API

2. **Backtrader Adapter** (`backtrader_adapter.py`)
   - Create Backtrader strategy that reads signals
   - Use Backtrader's data feed for synchronization
   - Map boolean signals to BUY/SELL orders

3. **VectorBT Adapter** (`vectorbt_adapter.py`)
   - Use VectorBT's `entries`/`exits` parameters directly
   - Boolean signals map naturally to VectorBT API
   - Simplest adapter (VectorBT designed for signal-based)

4. **Zipline Adapter** (`zipline_adapter.py`)
   - **DEFERRED** (bundle data issues)
   - Document why deferred
   - Provide placeholder implementation

**Backward Compatibility**:
- Keep existing `run_backtest()` method
- `run_with_signals()` is **addition**, not replacement

### Phase 3: Test Runner (Est: 2 hours)

**Create**: `tests/validation/run_signal_validation.py`

**Functionality**:
```python
def run_signal_validation(signal_file: Path) -> ComparisonReport:
    """
    Load signal file, run through all frameworks, compare results.

    Returns:
        ComparisonReport with:
        - Trade comparison (timestamp, price, quantity)
        - P&L comparison (final value, total return)
        - Execution metrics (time, memory)
        - Match/mismatch details
    """
```

**Comparison Logic**:
1. Load signals from pickle
2. Run through ml4t.backtest → get trades
3. Run through Backtrader → get trades
4. Run through VectorBT → get trades
5. Compare trades:
   - Count must match exactly
   - Timestamps must match exactly
   - Prices within 0.01% (fill model tolerance)
   - P&L within 0.002% (from previous validation)

**Output**: Markdown report with pass/fail for each framework

### Phase 4: Documentation & Validation (Est: 2 hours)

**Documents to Create**:
1. **Signal generation guide** - How to create new signal sets
2. **Adapter implementation guide** - How `run_with_signals()` works
3. **Validation results** - Comparison report
4. **Known limitations** - Zipline deferral, tolerance explanations

**Validation Criteria**:
- ✅ All 3 frameworks (ml4t.backtest, Backtrader, VectorBT) produce identical trades
- ✅ Final P&L matches within 0.002%
- ✅ Trade count matches exactly
- ✅ Validation completes in <1 minute

---

## Technical Considerations

### 1. Framework-Specific Challenges

#### ml4t.backtest
- **Strength**: Just added metadata tracking, clean export API
- **Challenge**: None (home framework)
- **Approach**: Straightforward signal-only strategy

#### Backtrader
- **Strength**: Well-documented, event-driven like ml4t.backtest
- **Challenge**: Different order submission patterns
- **Approach**: Create Backtrader strategy that reads signals from data feed
- **Reference**: `/home/stefan/ml3t/ch11_strategy_backtesting/code/03_backtesting_with_backtrader.py`

#### VectorBT
- **Strength**: Designed for signal-based backtesting (entries/exits)
- **Challenge**: NumPy-based, different paradigm
- **Approach**: Direct mapping of boolean signals to `entries`/`exits` arrays
- **Note**: Should be easiest adapter (VectorBT's native mode)

#### Zipline
- **Strength**: Institutional-grade, well-tested
- **Challenge**: Bundle data format incompatibility (4.3x price difference in previous test)
- **Approach**: **DEFER** - Document limitation, provide placeholder

### 2. Precision & Tolerance

**Sources of Variance**:
1. **Fill prices**: Frameworks may handle OHLC fills differently
2. **Commission rounding**: Different precision in commission calculation
3. **Position quantity**: Floating point accumulation

**Mitigation**:
- Use simple market orders only (no limit/stop complexity)
- Percentage commission (easier to replicate)
- Define acceptable tolerance levels:
  - Trade count: Exact match
  - Trade timestamps: Exact match
  - Trade prices: ±0.01%
  - Final P&L: ±0.002%

### 3. Signal Synchronization

**Critical Requirement**: All frameworks must process signals at same timestamps

**Approach**:
- Signals use same DatetimeIndex as OHLC data
- Frameworks iterate chronologically
- Boolean `True` at timestamp T triggers action at timestamp T

**Validation**:
- First test: Verify all frameworks see same number of events
- Second test: Verify all frameworks trigger at same timestamps

---

## Integration Points

### With Newly Implemented Reporting

**ml4t.backtest Adapter Can Now**:
```python
# In qengine_adapter.py
def run_with_signals(...) -> ValidationResult:
    engine = BacktestEngine(...)
    engine.run()

    # Use NEW export API
    results = engine.get_results()
    trades_df = results.get_trades()  # Polars DataFrame

    # Convert to ValidationResult format
    return ValidationResult(
        framework="ml4t.backtest",
        trades=[...],  # From trades_df
        final_value=results.summary()['final_equity'],
        ...
    )
```

**Benefits**:
- Standardized export format
- Metadata preserved (entry/exit reasons)
- Daily returns available for additional validation

### With Existing Test Infrastructure

**Leverages**:
- `BaseFrameworkAdapter` interface
- `ValidationResult` data class
- `TradeRecord` format

**Extends**:
- Adds `run_with_signals()` method
- Keeps `run_backtest()` for backward compatibility

---

## Risk Assessment

### Low Risk ✅
1. **Signal generation** - Straightforward Pandas operations
2. **ml4t.backtest adapter** - Home framework, just implemented reporting
3. **Test infrastructure** - Well-established patterns

### Medium Risk ⚠️
4. **Backtrader adapter** - Different API, but well-documented
5. **VectorBT adapter** - Different paradigm, but signals fit naturally
6. **Precision tolerance** - May need iteration to find right thresholds

### Deferred 🔄
7. **Zipline adapter** - Bundle data incompatibility (previous test showed 4.3x price difference)

### Mitigation Strategies

**For framework-specific challenges**:
- Start with simplest signals (SMA crossover)
- Test each framework independently before comparison
- Use reference implementations from ML3T book

**For precision issues**:
- Define tolerance levels based on previous cross-framework test (0.002%)
- Document expected variance sources
- Provide detailed comparison reports

---

## Success Metrics

### Quantitative
- ✅ 3/4 frameworks validate (ml4t.backtest, Backtrader, VectorBT)
- ✅ Trade count matches exactly across all 3
- ✅ Final P&L within 0.002% across all 3
- ✅ Validation runs in <1 minute

### Qualitative
- ✅ Clear documentation of signal-based approach
- ✅ Reproducible validation process
- ✅ Foundation for future multi-asset validation
- ✅ Confidence in ml4t.backtest execution fidelity

---

## Next Steps

### Immediate Action Required
**Run `/plan`** to create detailed task breakdown with:
1. Task dependencies (signal gen → adapters → validation)
2. Estimated effort per task
3. Testing criteria for each phase
4. Success verification steps

### Why /plan Is Needed
While requirements are clear, implementation needs:
- Detailed subtasks for each phase
- Proper sequencing (can't test adapters without signals)
- Test strategy for each component
- Time allocation across 10-hour estimate

The `/plan` command will create `plan.md` and `state.json` for systematic execution with `/next`.

---

## Reference Files

### Documentation (Read Before Starting)
- `.claude/transitions/2025-11-16/121417.md` - Previous handoff
- `tests/validation/SIGNAL_BASED_VALIDATION_PLAN.md` - Original plan
- `tests/validation/SIGNAL_VALIDATION_ARCHITECTURE.md` - Architecture rationale
- `.claude/memory/reporting_infrastructure.md` - NEW reporting features

### Code to Study
- `tests/validation/frameworks/base.py` - Adapter interface
- `src/ml4t/backtest/results.py` - NEW export API
- `/home/stefan/ml3t/ch11_strategy_backtesting/code/` - Framework examples

### Data
- `/home/stefan/ml3t/projects/crypto_futures/data/` - BTC/ETH/SOL data

---

**Status**: ✅ Exploration complete, requirements clear, ready for planning

**Complexity**: Medium (4 phases, 3 frameworks, ~10 hours estimated)

**Confidence**: High (clear requirements, existing infrastructure, small scope)

**Recommendation**: Proceed to `/plan` for structured implementation
