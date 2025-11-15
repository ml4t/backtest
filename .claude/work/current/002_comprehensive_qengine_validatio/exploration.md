# Exploration Summary: ml4t.backtest Cross-Framework Validation

## Codebase Analysis

### Existing Infrastructure
**Strong foundation already in place:**

1. **Validation Framework** (`tests/validation/frameworks/`)
   - `base.py:74` - `BaseFrameworkAdapter` abstract class with `run_backtest()` interface
   - `base.py:16` - `ValidationResult` dataclass for standardized results
   - `base.py:17` - `TradeRecord` for cross-framework trade comparison
   - Existing adapters need extension for VectorBT Pro, Zipline, Backtrader

2. **Strategy Specifications** (`tests/validation/strategy_specifications.py`)
   - `RSI_MEAN_REVERSION` (line 60)
   - `DUAL_MA_CROSSOVER` (line 81)
   - `BOLLINGER_BREAKOUT` (line 105)
   - `ML_MOMENTUM_STRATEGY` (line 133)
   - `HIGH_FREQUENCY_SCALPING` (line 163)
   - Generic `StrategySpec` allows framework-agnostic definitions

3. **Previous Validation Results**
   - `tests/validation/VALIDATION_SUMMARY.md` - 100% agreement with VectorBT
   - `tests/validation/COMPREHENSIVE_VALIDATION_REPORT.md` - Multi-asset validation
   - MA crossover: $1,507.06 final value, 14 trades (exact match)
   - 30-stock portfolio: 5,000 trades, perfect agreement

4. **ml4t.backtest Architecture**
   - Event-driven design with Clock, EventBus, Strategy, Broker, Portfolio
   - Polars-based for performance (10-100x faster than pandas)
   - Comprehensive execution models: 7 slippage, 9 commission, 6 market impact
   - ML-first design for qfeatures integration

### Data Resources Available

**Comprehensive data sources in `~/ml4t/projects/`:**

1. **Daily US Equities**
   - Location: `daily_us_equities/equity_prices_enhanced_1962_2025.parquet`
   - Coverage: 1962-2025 (63 years)
   - Use: Long-term strategies, fundamental analysis

2. **NASDAQ100 Minute Bars**
   - Location: `nasdaq100_minute_bars/{2021,2022}.parquet`
   - Coverage: 2021-2022 intraday data
   - Use: Scalping, intraday strategies

3. **Crypto Data**
   - Futures: `crypto_futures/data/futures/{BTC,ETH}.parquet`
   - Spot: `crypto_futures/data/spot/{BTC,ETH}.parquet`
   - Use: 24/7 strategies, crypto-specific patterns

4. **SPY Order Flow**
   - Location: `spy_order_flow/spy_features.parquet`
   - Features: Order imbalance, flow toxicity, VPIN
   - Use: Microstructure strategies

5. **Tick Data**
   - Location: `tick_data/` (high-frequency)
   - Use: Sub-second strategies, market making

### Integration Points

**Key areas where validation work connects:**

1. **Framework Adapters** (NEW - to be implemented)
   - Extend `BaseFrameworkAdapter` for each framework
   - Isolated subprocess execution (separate venvs)
   - Standardized result conversion to `ValidationResult`

2. **Data Loading** (NEW - to be created)
   - Universal data loader for all formats (Parquet, CSV)
   - Framework-specific format conversion
   - Handle Polars ↔ Pandas conversions

3. **Signal Generation** (THREE-TIER APPROACH)
   - Tier 1: Deterministic (hardcoded indicators)
   - Tier 2: qfeatures technical indicators
   - Tier 3: ML models (qfeatures → qeval → signals)

4. **ML Pipeline** (`~/ml4t/features/` and `~/ml4t/evaluation/`)
   - qfeatures: Feature engineering library
   - qeval: Model validation library
   - End-to-end: features → model → signals → backtest

## Implementation Approach

### Strategic Decisions

**1. Isolated Virtual Environments** (USER RECOMMENDED)
```bash
.venv-vectorbt     # VectorBT Pro environment
.venv-zipline      # Zipline-Reloaded environment
.venv-backtrader   # Backtrader environment
.venv-ml4t.backtest      # ml4t.backtest main environment
```
**Rationale**: Clean dependency management, no conflicts, easier debugging

**2. VectorBT Pro Documentation Strategy** (USER SPECIFIED)
- Primary: Use quantgpt.chat for VectorBT Pro questions
- Fallback: Manual user assistance if needed
- Resources: `resources/vectorbt.pro-main/` (320K lines, 271 files)

**3. Quality-First Approach** (USER EMPHASIZED)
- No artificial deadlines ("your timelines don't make sense you are not a person")
- Each phase completes when quality criteria met
- Investigate thoroughly, document findings
- "Need to do this well before we release anything"

**4. Phased Validation Sequence** (TIER 1→4)
```
Phase 0: Infrastructure Setup
  ↓
Phase 1: Tier 1 Core Validation (95%+ agreement)
  ↓
Phase 2: Tier 2 Advanced Execution (order types, sizing)
  ↓
Phase 3: Tier 3 ML Integration (qfeatures → ml4t.backtest)
  ↓
Phase 4: Tier 4 Performance & Edge Cases
  ↓
Phase 5: Documentation & Production Guide
```

### Technical Approach

**Framework Adapter Pattern:**
```python
class VectorBTProAdapter(BaseFrameworkAdapter):
    def __init__(self):
        self.venv_path = ".venv-vectorbt"

    def run_backtest(self, data, strategy_params, initial_capital):
        # Serialize inputs
        # Execute in isolated venv via subprocess
        # Parse and return ValidationResult
        pass
```

**Universal Data Loader:**
```python
class UniversalDataLoader:
    @staticmethod
    def load_daily_equities(symbol, start, end):
        # Load from ~/ml4t/projects/daily_us_equities/
        # Convert to framework-specific format

    @staticmethod
    def to_framework_format(df, framework):
        # VectorBT: specific column names
        # Zipline: bundle format
        # Backtrader: feed format
        # ml4t.backtest: Polars DataFrame
```

**Signal Generation Strategy:**
```python
# Tier 1: Deterministic
signals = generate_ma_crossover(data, fast=20, slow=50)

# Tier 2: qfeatures
from qfeatures import TechnicalFeatures
features = TechnicalFeatures(data).compute(['rsi', 'macd'])
signals = generate_signals_from_features(features)

# Tier 3: ML Pipeline
from qfeatures import FeatureEngineer
from qeval import ModelValidator
features = FeatureEngineer(data).create_feature_set()
model = ModelValidator(features).train_model('xgboost')
signals = model.predict(features)
```

### Performance Benchmarking Methodology

**Metrics to Measure:**
1. **Speed**: Events/sec, trades/sec, completion time
2. **Resources**: Peak memory, average memory, CPU %
3. **Accuracy**: Final value (±0.01%), trade count, timing
4. **Scalability**: Linear scaling 1→500 assets

**Expected Performance Profiles:**
- ml4t.backtest: 300-500 trades/sec (baseline)
- VectorBT Pro: 189K trades/sec (20-25x faster)
- Backtrader: 50-100 trades/sec (0.1-0.2x)
- Zipline: 2-5 trades/sec (0.01x)

**When ml4t.backtest Wins:**
- Complex ML strategies (custom execution logic)
- Realistic execution simulation (slippage, impact models)
- Integration with qfeatures/qeval pipeline

## Key Findings

### Strengths of Current Setup
✅ **Solid Foundation**: BaseFrameworkAdapter pattern well-designed
✅ **Proven Correctness**: 100% agreement with VectorBT already achieved
✅ **Rich Data**: Comprehensive datasets spanning frequencies and assets
✅ **ML Ready**: qfeatures and qeval libraries available
✅ **Performance Proven**: 9-265x faster than alternatives already shown

### Known Issues & Workarounds
⚠️ **Backtrader**: Signal execution bug (missing trades) - document only, don't rely on for correctness
⚠️ **Zipline**: Timezone and data feed complexity - use simple daily data only
⚠️ **Frameworks Not Installed**: All 3 need installation in isolated venvs

### Risks & Mitigation
1. **Framework installation complexity** → Start Phase 0 early, separate venvs
2. **VectorBT Pro documentation** → Use quantgpt.chat, fallback to user
3. **Data format issues** → Universal loader with explicit conversion
4. **Signal reproducibility** → Deterministic seeds, save signals to disk

## Next Steps

### Immediate Action Required
**Run `/plan --from-requirements .claude/planning/COMPREHENSIVE_VALIDATION_PLAN.md`**

This will create:
1. **state.json** - Ordered task list with dependencies
2. **Task breakdown** - Each task 2-4 hours, single responsibility
3. **Acceptance criteria** - Clear success criteria per task
4. **Effort estimates** - Total hours and critical path

### Key Planning Areas
1. **Phase 0 Tasks**: Framework installation sequence, verification tests
2. **Adapter Implementation**: VectorBT Pro, Zipline, Backtrader adapters
3. **Data Pipeline**: Universal loaders for all data sources
4. **Validation Sequence**: Tier 1→4 test execution order
5. **Documentation**: Report templates and delivery schedule

### Validation Strategy Summary
- **Correctness First**: Establish 95%+ agreement baseline
- **Performance Second**: Systematic benchmarking across scales
- **ML Integration Third**: Prove qfeatures → ml4t.backtest pipeline
- **Edge Cases Last**: Stress test and production readiness

### Expected Deliverables
1. `TIER1_CORE_VALIDATION.md` - Correctness proof
2. `TIER2_EXECUTION_VALIDATION.md` - Order execution accuracy
3. `TIER3_ML_PIPELINE_VALIDATION.md` - ML workflow validation
4. `TIER4_PERFORMANCE_BENCHMARKS.md` - Performance analysis
5. `FRAMEWORK_SELECTION_GUIDE.md` - When to use which framework
6. `PRODUCTION_READINESS_CHECKLIST.md` - Deployment guide

---

**Exploration Complete** ✅

The requirements are well-understood, the codebase is analyzed, the approach is clear. Ready for detailed task planning.
