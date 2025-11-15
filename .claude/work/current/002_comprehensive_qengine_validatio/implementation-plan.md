# Implementation Plan: ml4t.backtest Cross-Framework Validation

## Overview

Comprehensive validation of ml4t.backtest backtesting framework against VectorBT Pro, Zipline-Reloaded, and Backtrader. Goal: Prove correctness (95%+ agreement), measure performance, validate ML integration, and establish framework selection guidance.

**Total Effort**: 76-152 hours across 38 tasks
**Approach**: Quality-first, no artificial deadlines - tasks complete when correct

## Phase Structure

### Phase 0: Infrastructure Setup (9 tasks, 18-36 hours)

**Goal**: Install all frameworks in isolated environments, implement adapters, create universal data loader

**Tasks**:
- TASK-001: Install VectorBT Pro (.venv-vectorbt) - 3h
- TASK-002: Install Zipline-Reloaded (.venv-zipline) - 4h
- TASK-003: Install Backtrader (.venv-backtrader) - 2h
- TASK-004: Implement VectorBT Pro adapter - 4h
- TASK-005: Implement Zipline adapter - 4h
- TASK-006: Implement Backtrader adapter - 3h
- TASK-007: Create universal data loader - 4h
- TASK-008: Baseline verification test - 3h
- TASK-009: Document infrastructure setup - 2h

**Dependencies**: TASK-001/002/003/007 can run in parallel. Adapters depend on installations. Baseline test depends on all adapters + data loader.

**Success Criteria**: All 3 frameworks installed and verified with hello-world test passing on all 4 frameworks.

---

### Phase 1: Tier 1 Core Validation (6 tasks, 12-24 hours)

**Goal**: Establish 95%+ agreement baseline across fundamental strategies (correctness proof)

**Strategies**:
- TASK-010: RSI Mean Reversion - 3h
- TASK-011: Bollinger Band Breakout with stop-loss - 3h
- TASK-012: Multi-indicator (MACD + RSI) - 3h
- TASK-013: Multi-asset momentum ranking (30+ stocks) - 4h
- TASK-014: Investigate discrepancies - 4h
- TASK-015: Tier 1 validation report - 3h

**Dependencies**: All require Phase 0 complete. Can run validation tasks in parallel, then investigate, then report.

**Success Criteria**: 95%+ agreement with VectorBT Pro on final portfolio value, ±5% on trade count, zero lookahead bias proven.

---

### Phase 2: Tier 2 Advanced Execution (6 tasks, 12-24 hours)

**Goal**: Validate all order types and execution timing accuracy

**Order Types**:
- TASK-016: Limit orders - 3h
- TASK-017: Stop-loss orders - 3h
- TASK-018: Bracket orders (entry + stop + target) - 4h
- TASK-019: Trailing stops - 3h
- TASK-020: Minute bar intraday strategies - 4h
- TASK-021: Tier 2 validation report - 2h

**Dependencies**: Requires TASK-015 complete. TASK-018 depends on TASK-016 + TASK-017. TASK-019 depends on TASK-017. Others can run in parallel.

**Success Criteria**: All order types execute correctly, fill timing accurate, 95%+ agreement maintained.

---

### Phase 3: Tier 3 ML Integration (6 tasks, 12-24 hours)

**Goal**: Prove end-to-end ML pipeline (qfeatures → qeval → ml4t.backtest)

**ML Strategies**:
- TASK-022: qfeatures technical indicators - 3h
- TASK-023: Binary classification ML signals - 4h
- TASK-024: Return forecasting position sizing - 4h
- TASK-025: Multi-asset ML ranking - 4h
- TASK-026: SPY order flow microstructure - 3h
- TASK-027: Tier 3 ML pipeline report - 3h

**Dependencies**: Requires TASK-015 complete. TASK-023 depends on TASK-022. TASK-024 depends on TASK-023. TASK-025 depends on TASK-024. TASK-026 can run in parallel with others.

**Success Criteria**: Full ML pipeline proven, qfeatures integration working, model signals execute correctly, 95%+ agreement.

---

### Phase 4: Tier 4 Performance & Edge Cases (7 tasks, 14-28 hours)

**Goal**: Performance benchmarks and edge case handling

**Tests**:
- TASK-028: High-frequency tick data - 4h
- TASK-029: Irregular bars (volume/dollar) - 3h
- TASK-030: Corporate actions (dividends, splits) - 3h
- TASK-031: Extreme market conditions (crashes) - 3h
- TASK-032: 1000-asset scalability - 4h
- TASK-033: Performance benchmarking suite - 4h
- TASK-034: Tier 4 performance report - 3h

**Dependencies**: Requires TASK-015 complete. TASK-028 through TASK-032 can run in parallel. TASK-033 depends on all validation tasks. TASK-034 depends on TASK-033.

**Success Criteria**: Performance profiles documented (events/sec, memory, scalability), edge cases handled, ml4t.backtest advantages identified.

---

### Phase 5: Documentation (4 tasks, 8-16 hours)

**Goal**: Comprehensive documentation for production deployment

**Deliverables**:
- TASK-035: Validation summary report - 3h
- TASK-036: Framework selection guide - 2h
- TASK-037: Production readiness checklist - 2h
- TASK-038: Known limitations documentation - 2h

**Dependencies**: All require completion of Tier 1-4 reports (TASK-015, TASK-021, TASK-027, TASK-034).

**Success Criteria**: Complete documentation package ready for production deployment and stakeholder review.

---

## Critical Path

```
Phase 0: Infrastructure Setup (sequential, 18-36h)
  ↓
Phase 1: Tier 1 Core Validation (sequential, 12-24h)
  ↓
Phase 2-4: Parallel Validation (can run simultaneously, 38-76h)
  ├── Phase 2: Advanced Execution
  ├── Phase 3: ML Integration
  └── Phase 4: Performance & Edge Cases
  ↓
Phase 5: Documentation (sequential, 8-16h)
```

**Total Critical Path**: 76-152 hours (quality-first, no rush)

## Parallelization Strategy

### Immediate Parallelization (Phase 0)
- TASK-001, TASK-002, TASK-003, TASK-007 can run in parallel (framework installations + data loader)
- This reduces Phase 0 from potential 29h sequential to ~12h with parallelization

### Major Parallelization Point (After Phase 1)
Once TASK-015 completes:
- **Phase 2 tasks** (TASK-016 to TASK-020) can run in parallel
- **Phase 3 tasks** (TASK-022 to TASK-026) can run in parallel (with internal dependencies)
- **Phase 4 tasks** (TASK-028 to TASK-032) can run in parallel

This allows 3 validation tiers to progress simultaneously, significantly reducing wall-clock time.

## Quality Gates

### Phase 0 Gate
- [ ] All 3 frameworks installed successfully
- [ ] All 3 adapters implement BaseFrameworkAdapter correctly
- [ ] Universal data loader handles all data sources
- [ ] Hello-world test passes on all 4 frameworks
- [ ] Documentation complete

### Phase 1 Gate (CRITICAL)
- [ ] 95%+ agreement across all 5 strategies
- [ ] Trade count within ±5%
- [ ] All discrepancies investigated and explained
- [ ] Correctness proof established
- [ ] Tier 1 report published

### Phase 2 Gate
- [ ] All order types execute correctly
- [ ] Fill timing accurate across frameworks
- [ ] Partial fills and edge cases handled
- [ ] 95%+ agreement maintained
- [ ] Tier 2 report published

### Phase 3 Gate
- [ ] qfeatures → qeval → ml4t.backtest pipeline proven
- [ ] ML model signals execute correctly
- [ ] Feature engineering working
- [ ] 95%+ agreement maintained
- [ ] Tier 3 report published

### Phase 4 Gate
- [ ] Performance profiles documented
- [ ] Scalability verified (1-1000 assets)
- [ ] Edge cases handled
- [ ] ml4t.backtest advantages identified
- [ ] Tier 4 report published

### Phase 5 Gate (FINAL)
- [ ] Validation summary complete
- [ ] Framework selection guide published
- [ ] Production readiness checklist finalized
- [ ] Known limitations documented
- [ ] All deliverables ready for stakeholder review

## Key Technical Decisions

### 1. Isolated Virtual Environments (USER RECOMMENDED)
```bash
.venv-vectorbt     # VectorBT Pro environment
.venv-zipline      # Zipline-Reloaded environment
.venv-backtrader   # Backtrader environment
.venv-ml4t.backtest      # ml4t.backtest main environment (current)
```
**Rationale**: Clean dependency management, no conflicts, easier debugging

### 2. VectorBT Pro Documentation Strategy (USER SPECIFIED)
- Primary: Use quantgpt.chat for VectorBT Pro questions
- Fallback: Manual user assistance if needed
- Resources: `resources/vectorbt.pro-main/` (320K lines, 271 files)

### 3. Quality-First Approach (USER EMPHASIZED)
- No artificial deadlines ("your timelines don't make sense you are not a person")
- Each phase completes when quality criteria met
- Investigate thoroughly, document findings
- "Need to do this well before we release anything"

### 4. Data Sources (USER SPECIFIED)
All data in `~/ml4t/projects/`:
- Daily equities: `daily_us_equities/equity_prices_enhanced_1962_2025.parquet`
- Minute bars: `nasdaq100_minute_bars/{2021,2022}.parquet`
- Crypto: `crypto_futures/data/{futures,spot}/{BTC,ETH}.parquet`
- Order flow: `spy_order_flow/spy_features.parquet`
- Tick data: `tick_data/` (high-frequency)

## Implementation Patterns

### Framework Adapter Pattern
```python
class VectorBTProAdapter(BaseFrameworkAdapter):
    def __init__(self):
        self.venv_path = ".venv-vectorbt"

    def run_backtest(self, data, strategy_params, initial_capital):
        # 1. Serialize inputs to JSON
        # 2. Execute in isolated venv via subprocess
        # 3. Parse results from JSON
        # 4. Return ValidationResult
        pass
```

### Universal Data Loader Pattern
```python
class UniversalDataLoader:
    @staticmethod
    def load_daily_equities(symbol, start, end):
        # Load from ~/ml4t/projects/daily_us_equities/
        df = pl.read_parquet(path)
        return df

    @staticmethod
    def to_framework_format(df, framework):
        if framework == "vectorbt":
            # VectorBT-specific column names
        elif framework == "zipline":
            # Zipline bundle format
        elif framework == "backtrader":
            # Backtrader feed format
        elif framework == "ml4t.backtest":
            # Already Polars, return as-is
        return formatted_df
```

### Signal Generation Strategy (Three-Tier)
```python
# Tier 1: Deterministic (hardcoded indicators)
signals = generate_ma_crossover(data, fast=20, slow=50)

# Tier 2: qfeatures technical indicators
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

## Performance Benchmarking Methodology

### Metrics to Measure
1. **Speed**: Events/sec, trades/sec, completion time
2. **Resources**: Peak memory, average memory, CPU %
3. **Accuracy**: Final value (±0.01%), trade count, timing
4. **Scalability**: Linear scaling test (1→500 assets)

### Expected Performance Profiles
- **ml4t.backtest**: 300-500 trades/sec (baseline)
- **VectorBT Pro**: 189K trades/sec (20-25x faster, vectorized)
- **Backtrader**: 50-100 trades/sec (0.1-0.2x, event-driven overhead)
- **Zipline**: 2-5 trades/sec (0.01x, data bundle overhead)

### When ml4t.backtest Wins
- Complex ML strategies (custom execution logic)
- Realistic execution simulation (7 slippage, 9 commission, 6 market impact models)
- Integration with qfeatures/qeval pipeline
- Type safety and code quality requirements

## Risk Mitigation

### Known Risks and Mitigations

1. **Framework Installation Complexity** (High)
   - Mitigation: Start Phase 0 early, use isolated venvs, document setup thoroughly

2. **VectorBT Pro Documentation Access** (Medium)
   - Mitigation: Use quantgpt.chat, fallback to manual user assistance

3. **Data Format Incompatibility** (Medium)
   - Mitigation: Universal loader with explicit format conversion, test all sources

4. **Framework Bugs** (Known)
   - Backtrader: Signal execution bugs → document limitations, don't rely for correctness
   - Zipline: Timezone issues → use simple daily data only
   - Mitigation: Document known issues, work around or exclude affected tests

5. **Performance Regression** (Low)
   - Mitigation: Continuous benchmarking, performance tests in CI/CD

## Success Metrics

### Correctness (PRIMARY)
- ✅ 95%+ agreement with VectorBT Pro across all Tier 1-2 tests
- ✅ Trade count within ±5%
- ✅ Zero lookahead bias (architectural guarantee via Clock)
- ✅ Correct order execution timing

### Performance (SECONDARY)
- ✅ Within 2x of VectorBT Pro for simple strategies
- ✅ Proven advantages for complex ML strategies
- ✅ Linear scaling verified (1-500 assets)
- ✅ Sub-second backtest for daily strategies (1 year, single asset)

### ML Integration (TERTIARY)
- ✅ qfeatures → qeval → ml4t.backtest pipeline proven
- ✅ Feature engineering working correctly
- ✅ Model signals execute with same accuracy as deterministic

### Documentation (REQUIRED)
- ✅ Comprehensive validation reports for each tier
- ✅ Framework selection guide published
- ✅ Known limitations documented
- ✅ Production readiness checklist finalized

## Expected Deliverables

### Tier Reports
1. `TIER1_CORE_VALIDATION.md` - Correctness proof (95%+ agreement)
2. `TIER2_EXECUTION_VALIDATION.md` - Order execution accuracy
3. `TIER3_ML_PIPELINE_VALIDATION.md` - ML workflow validation
4. `TIER4_PERFORMANCE_BENCHMARKS.md` - Performance analysis

### Strategic Documentation
5. `VALIDATION_SUMMARY.md` - Executive summary of all findings
6. `FRAMEWORK_SELECTION_GUIDE.md` - When to use which framework
7. `PRODUCTION_READINESS_CHECKLIST.md` - Deployment guide
8. `KNOWN_LIMITATIONS.md` - Edge cases and workarounds

## Next Steps

### To Begin Execution
Run `/next` to start with first available tasks:
- TASK-001: Install VectorBT Pro
- TASK-002: Install Zipline-Reloaded
- TASK-003: Install Backtrader
- TASK-007: Create universal data loader

These 4 tasks can run in parallel to minimize Phase 0 wall-clock time.

### Progress Tracking
- Use `/status` to view current work state
- Use `/next` to execute next available task
- Each task completion updates state.json and unlocks dependencies
- Quality gates enforce phase completion criteria

---

**Planning Complete** ✅

Ready for execution. Quality over speed. No artificial deadlines. Done when correct.
