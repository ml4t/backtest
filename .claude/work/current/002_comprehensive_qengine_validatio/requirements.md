# Requirements: QEngine Cross-Framework Validation

## Source
- Type: description
- Reference: Comprehensive QEngine validation plan
- Date: 2025-10-04T19:59:00-05:00
- Detailed Plan: `.claude/planning/COMPREHENSIVE_VALIDATION_PLAN.md`

## Overview

Comprehensive validation of QEngine backtesting framework against three major industry frameworks:
- **VectorBT Pro** (commercial, vectorized, 20-25x faster)
- **Zipline-Reloaded** (Quantopian legacy, event-driven)
- **Backtrader** (popular, event-driven)

Goals: Prove correctness, measure performance, validate ML integration, establish framework selection guidance.

## Functional Requirements

### 1. Framework Infrastructure Setup
- Install VectorBT Pro, Zipline-Reloaded, Backtrader in isolated virtual environments
- Implement framework adapters extending `tests/validation/frameworks/base.py`
- Create universal data loaders for all data sources in `~/ml4t/projects/`
- Establish baseline "hello world" test on all frameworks

### 2. Tier 1: Core Correctness Validation (95%+ agreement)
- **T1.1**: MA Crossover validation (✅ already done - 100% agreement)
- **T1.2**: RSI Mean Reversion strategy
- **T1.3**: Bollinger Band Breakout with stop-loss
- **T1.4**: Multi-indicator combination (MACD + RSI)
- **T1.5**: Multi-asset momentum ranking (30+ stocks)

### 3. Tier 2: Advanced Execution Validation
- **T2.1**: Limit order execution and fill timing
- **T2.2**: Stop-loss order triggering accuracy
- **T2.3**: Bracket orders (entry + stop + target)
- **T2.4**: Trailing stop dynamic adjustment
- **T2.5**: Minute bar intraday strategies

### 4. Tier 3: ML Pipeline Integration
- **T3.1**: qfeatures technical indicator signals
- **T3.2**: Binary classification ML signals (buy/sell)
- **T3.3**: Return forecasting for position sizing
- **T3.4**: Multi-asset ranking with ML scoring
- **T3.5**: Order flow microstructure signals (SPY)

### 5. Tier 4: Performance & Edge Cases
- **T4.1**: High-frequency tick data validation
- **T4.2**: Irregular bars (volume/dollar bars)
- **T4.3**: Corporate actions (dividends, splits)
- **T4.4**: Extreme market conditions (crashes)
- **T4.5**: Scalability (1000-asset portfolios)

### 6. Performance Benchmarking
- Speed metrics: Events/sec, trades/sec, completion time
- Resource metrics: Memory usage, CPU utilization
- Scalability tests: 1→500 assets, 1 month→10 years
- Accuracy metrics: Final value agreement, trade timing

## Non-Functional Requirements

### Performance
- QEngine within 2x of VectorBT Pro for simple strategies
- Linear scaling with asset count (validated up to 500 assets)
- Sub-second backtest for daily strategies (1 year, single asset)

### Correctness
- 95%+ agreement with VectorBT Pro on final portfolio value
- 99%+ agreement on trade count (±5%)
- Zero lookahead bias (validated by timeline analysis)

### Compatibility
- Support daily, minute, tick, and irregular bar frequencies
- Handle equities, crypto futures/spot, multi-asset portfolios
- Work with qfeatures → qeval → qengine ML pipeline

### Documentation
- Comprehensive validation reports for each tier
- Framework selection guide (when to use which)
- Known limitations and edge cases documented
- Production readiness checklist

## Acceptance Criteria

- [ ] All 3 frameworks installed and verified in isolated environments
- [ ] Framework adapters implemented and tested
- [ ] Tier 1 validation: 95%+ agreement on 5 strategies
- [ ] Tier 2 validation: All order types executing correctly
- [ ] Tier 3 validation: ML pipeline end-to-end proven
- [ ] Tier 4 validation: Performance profiles documented
- [ ] Validation summary report complete
- [ ] Framework selection guide published
- [ ] Production readiness checklist finalized

## Out of Scope

- Live trading integration (future work)
- Options and derivatives strategies (future work)
- Multi-currency portfolios (future work)
- Real-time data feed integration (future work)
- Web dashboard for results visualization (future work)

## Dependencies

### External Dependencies
- VectorBT Pro source code (`resources/vectorbt.pro-main/`)
- Zipline-Reloaded framework (pip installable)
- Backtrader framework (pip installable)
- quantgpt.chat for VectorBT Pro documentation

### Internal Dependencies
- qfeatures library (`~/ml4t/features/qfeatures`) for feature engineering
- qeval library (`~/ml4t/evaluation/qeval`) for model validation
- Data sources in `~/ml4t/projects/` (daily equities, minute bars, crypto, tick data)

### Existing Infrastructure
- `tests/validation/frameworks/base.py` - BaseFrameworkAdapter
- `tests/validation/frameworks/qengine_adapter.py` - QEngine implementation
- `tests/validation/strategy_specifications.py` - Generic strategy specs
- Previous validation results (MA crossover, multi-asset portfolio)

## Risks and Assumptions

### Risks
1. **Framework installation complexity** (High) - VectorBT Pro may have specific requirements
2. **VectorBT Pro documentation access** (Medium) - Rely on quantgpt.chat, manual help if needed
3. **Data format incompatibility** (Medium) - Different frameworks expect different formats
4. **Framework bugs** (Known) - Backtrader has signal execution bugs, Zipline has timezone issues
5. **Performance regression** (Low) - Continuous benchmarking mitigates this

### Assumptions
1. Separated virtual environments will prevent dependency conflicts
2. VectorBT Pro is the correctness baseline (proven reliable)
3. Existing test infrastructure (base.py) is solid foundation
4. Data in ~/ml4t/projects/ is accessible and well-formatted
5. Quality over speed - no artificial deadlines, done when correct

## Implementation Strategy

### Phase-Based Approach
**Phase 0**: Infrastructure setup and verification
**Phase 1**: Tier 1 core validation (correctness proof)
**Phase 2**: Tier 2 execution validation (order types, risk)
**Phase 3**: Tier 3 ML integration (qfeatures pipeline)
**Phase 4**: Tier 4 performance & edge cases (scalability)
**Phase 5**: Documentation and production guide

### Quality Gates
Each tier completes when:
- All tests pass with required agreement thresholds
- Documentation complete and reviewed
- Edge cases identified and handled
- Code reviewed and refactored
- Ready for next tier

### Success Metrics
- ✅ 95%+ agreement with VectorBT Pro across all Tier 1-2 tests
- ✅ Performance within 2x of VectorBT Pro for simple strategies
- ✅ Proven advantages for complex ML strategies
- ✅ Comprehensive documentation of findings
- ✅ Clear guidance on framework selection

## Next Steps

Run `/plan --from-requirements .claude/planning/COMPREHENSIVE_VALIDATION_PLAN.md` to create detailed task breakdown with:
- Dependency-ordered task sequence
- Acceptance criteria per task
- Effort estimates
- Parallel work opportunities
