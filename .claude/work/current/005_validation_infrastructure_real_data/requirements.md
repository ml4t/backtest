# Work Unit 005: Validation Infrastructure with Real Data

**Created**: 2025-11-04
**Status**: In Progress
**Priority**: High

## Objective

Build production-quality validation infrastructure that tests qengine against VectorBT, Backtrader, and Zipline using real market data.

## Success Criteria

1. ✅ **Real Data Integration**
   - Real market data from Quandl Wiki prices (not synthetic)
   - Custom Zipline bundle for validation testing
   - Reusable fixtures for all platforms

2. 🔄 **Multi-Platform Validation** (In Progress)
   - All 4 platforms execute with real data
   - Trade-by-trade comparison working
   - Clear reporting of differences

3. ⏳ **Scenario Library** (Pending)
   - Complete Tier 1 scenarios (001-005)
   - Cover market orders, limit orders, stops
   - Test multi-asset and re-entry patterns

4. ⏳ **Test-Driven Development** (Pending)
   - Write tests before implementation
   - Red → Green → Refactor cycle
   - 80%+ coverage of validation framework

## Context

This work builds on:
- **Work Unit 003**: Validation framework architecture and design
- **Work Unit 004**: VectorBT exact matching work
- **Recent Sessions**: Built complete trade-by-trade comparison framework

We moved from synthetic data to real market data to:
- Validate realistic price movements
- Test with actual splits/dividends
- Match production trading conditions

## Key Deliverables

### Phase 1: Infrastructure (DONE ✅)

1. **Market Data Fixtures** (`tests/validation/fixtures/market_data.py`)
   - ✅ Load wiki_prices dataset
   - ✅ Filter by ticker and date range
   - ✅ Prepare Zipline bundle data with splits/dividends

2. **Custom Zipline Bundle** (`tests/validation/bundles/`)
   - ✅ Bundle ingest function
   - ✅ Extension registration
   - ✅ Successfully ingested (4 tickers, 249 days)
   - ✅ Setup scripts and documentation

3. **Updated Scenario 001** with Real Data
   - ✅ Uses AAPL 2017 data (249 trading days)
   - ✅ 4 signals → 2 complete trades
   - ✅ All signal dates validated in dataset

### Phase 2: Multi-Platform Testing (IN PROGRESS 🔄)

**Current Status**:
- ✅ Backtrader: 2 trades extracted successfully
- ❌ qengine: 0 trades (not executing signals)
- ❌ VectorBT: 0 trades (not executing signals)
- ⏸️ Zipline: Bundle ready, not tested yet

**Issues to Resolve**:
1. Why are qengine and VectorBT not executing trades?
   - Signal dates are valid (confirmed in dataset)
   - Data is properly formatted
   - Likely signal processing or platform setup issue

2. Need to test Zipline platform integration

### Phase 3: Test-Driven Scenario Expansion (PENDING ⏳)

Build 25-scenario test suite following TDD:
- **Tier 1 (001-005)**: Basic market orders, limits, stops
- **Tier 2 (006-010)**: Advanced orders, brackets, trailing stops
- **Tier 3 (011-015)**: Complex patterns, re-entry, multi-timeframe
- **Tier 4 (016-020)**: Edge cases, constraints, failures
- **Tier 5 (021-025)**: Stress tests, performance, large portfolios

## Technical Requirements

### Functional

1. **Data Management**
   - Real OHLCV data from Quandl Wiki (1962-2018)
   - Support for daily and minute frequencies
   - Splits and dividends included
   - Timezone-aware timestamps (UTC)

2. **Platform Integration**
   - qengine: Event-driven execution
   - VectorBT: Vectorized backtesting
   - Backtrader: Strategy-based execution
   - Zipline: Algorithm API with bundles

3. **Validation Framework**
   - Platform-agnostic signal specification
   - Trade extraction from platform outputs
   - Trade matching with configurable tolerance
   - Detailed and summary reports

### Non-Functional

1. **Performance**
   - Scenario execution < 5 seconds
   - Support 100+ tickers
   - Handle 10+ years of daily data

2. **Reliability**
   - Deterministic results
   - Comprehensive error handling
   - Clear failure diagnostics

3. **Maintainability**
   - TDD approach (tests first)
   - Modular architecture
   - Well-documented fixtures

## Dependencies

- Python 3.9+
- qengine (local package)
- VectorBT (installed)
- Backtrader (installed)
- Zipline-reloaded (installed)
- Polars, Pandas, NumPy

## Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Platform execution differences | HIGH | Document differences, standardize where possible |
| Data quality issues | MEDIUM | Use well-tested Quandl dataset |
| Timezone handling complexity | MEDIUM | Standardize on UTC everywhere |
| Test development time | MEDIUM | Prioritize Tier 1 scenarios first |

## Out of Scope

- Live trading validation
- Options/futures complex instruments
- High-frequency trading (< 1 minute bars)
- Custom indicators (focus on execution)

## Timeline Estimate

- **Phase 1** (Infrastructure): ✅ Complete (1 session)
- **Phase 2** (Multi-platform): 🔄 Current (1-2 sessions)
- **Phase 3** (Scenarios 001-005): ⏳ 2-3 sessions
- **Phase 4** (Scenarios 006-025): ⏳ 8-10 sessions
- **Phase 5** (Production polish): ⏳ 2 sessions

**Total**: ~15-20 development sessions

## Acceptance Criteria

✅ = Complete, 🔄 = In Progress, ⏳ = Pending

- ✅ Real data fixtures working
- ✅ Zipline bundle ingested successfully
- 🔄 All 4 platforms execute with real data
- ⏳ Scenario 001 passes on all platforms
- ⏳ Tier 1 scenarios (001-005) complete
- ⏳ 80%+ test coverage of validation framework
- ⏳ Documentation complete (README, guides)
- ⏳ CI/CD integration (optional)
