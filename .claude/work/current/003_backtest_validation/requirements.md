# Requirements: Cross-Platform Backtest Validation

## Source
- Type: Natural language description
- Reference: User request for validation framework
- Date: 2025-10-08
- Work Unit: 003_backtest_validation

## Overview

Create a comprehensive validation framework to compare ml4t.backtest backtesting results against three established platforms (zipline-reloaded, vectorbt-pro, backtrader) using identical trading signals. The goal is to verify that the same signals produce the same trades and P&L across all platforms.

## Functional Requirements

### 1. Signal Generation Framework (Platform-Independent)
- **FR-1.1**: Implement signal generators that are completely independent of any backtesting platform
- **FR-1.2**: Signal output format must be standardized (timestamp, symbol, action, quantity)
- **FR-1.3**: Signals must be deterministic and reproducible across runs
- **FR-1.4**: Support multiple signal types:
  - Simple: Moving average crossover
  - Medium: Mean reversion (Bollinger/RSI)
  - Complex: Multi-factor momentum
  - Edge cases: Entry-only, exit-only, position sizing variations

### 2. Platform Adapters
- **FR-2.1**: Implement adapter for ml4t.backtest
- **FR-2.2**: Implement adapter for zipline-reloaded
- **FR-2.3**: Implement adapter for vectorbt-pro (or fallback to free vectorbt)
- **FR-2.4**: Implement adapter for backtrader
- **FR-2.5**: Each adapter translates signals to platform-specific orders
- **FR-2.6**: Adapters must configure platforms with identical execution rules

### 3. Test Cases (Using ../projects/ Data)
- **FR-3.1**: Buy-and-hold baseline test
- **FR-3.2**: MA crossover with multiple entries/exits
- **FR-3.3**: Mean reversion with frequent trading
- **FR-3.4**: Multiple symbols (portfolio of 5-10 assets)
- **FR-3.5**: Long/short positions
- **FR-3.6**: Stop loss handling
- **FR-3.7**: High frequency rebalancing
- **FR-3.8**: Price gap scenarios
- **FR-3.9**: Position sizing variations
- **FR-3.10**: Edge case handling

### 4. Validation Framework
- **FR-4.1**: Signal consistency validation (100% match expected)
- **FR-4.2**: Order generation comparison
- **FR-4.3**: Trade execution comparison (count, timing, fills)
- **FR-4.4**: P&L comparison (trade-by-trade and total)
- **FR-4.5**: Performance metrics (Sharpe, drawdown, etc.)
- **FR-4.6**: Automated HTML/Markdown report generation
- **FR-4.7**: Difference highlighting and explanation

## Non-Functional Requirements

### Performance
- **NFR-1**: Validation suite should complete in <5 minutes for daily data
- **NFR-2**: Support parallel execution across platforms
- **NFR-3**: Efficient data loading (reuse across platforms)

### Data Quality
- **NFR-4**: Use real market data from ../projects/
- **NFR-5**: Support multiple data formats (parquet, CSV)
- **NFR-6**: Handle missing data gracefully

### Reproducibility
- **NFR-7**: Version pin all platform dependencies
- **NFR-8**: Document all configuration parameters
- **NFR-9**: Results must be reproducible across runs
- **NFR-10**: Consider Docker container for environment isolation

### Maintainability
- **NFR-11**: Modular architecture (easy to add signals/platforms)
- **NFR-12**: Comprehensive documentation
- **NFR-13**: Type hints and docstrings throughout
- **NFR-14**: Follow ml4t.backtest code standards (ruff, mypy)

## Acceptance Criteria

- ✅ Signal generator produces identical signals for all platforms
- ✅ Trade count matches exactly across all platforms (for same signals)
- ✅ P&L differences are within tolerance or explainable
- ✅ At least 5 test cases pass validation
- ✅ Validation report clearly shows:
  - Signal consistency
  - Trade-level comparison
  - P&L comparison
  - Platform-specific differences documented
- ✅ Code has 80%+ test coverage
- ✅ All quality checks pass (make check)

## Out of Scope

- **High-frequency tick data**: Start with daily/minute bars
- **Live trading validation**: Focus on backtesting only
- **Performance optimization**: Correctness over speed initially
- **GUI/Interactive reports**: Command-line and static reports only
- **Additional platforms**: Only the 4 specified platforms

## Data Availability

Available datasets in ../projects/:
- **Daily US Equities**: `equity_prices_enhanced_1962_2025.parquet` (comprehensive)
- **NASDAQ-100 Minute**: `2021.parquet`, `2022.parquet` (intraday)
- **Crypto Futures**: BTC/ETH futures and spot data
- **SPY Order Flow**: Microstructure data
- **Crypto ML Strategy**: Example strategy results

## Dependencies

### Platform Dependencies
1. **zipline-reloaded**:
   - Install: `pip install zipline-reloaded`
   - Status: Open source, actively maintained
   - Risk: Complex dependency tree

2. **vectorbt-pro**:
   - Install: `pip install vectorbt-pro` (license required)
   - Status: Commercial, may need license
   - Fallback: Use free `vectorbt` if license unavailable
   - Risk: HIGH - License availability unknown

3. **backtrader**:
   - Install: `pip install backtrader`
   - Status: Open source, stable
   - Risk: LOW - Lightweight dependencies

4. **ml4t.backtest**:
   - Status: Under active development
   - Version: Production-ready (September 2025)
   - Features: Event-driven, execution delay, multi-feed sync

### ml4t.backtest State
- ✅ Event flow working (data → portfolio)
- ✅ Temporal accuracy (execution delay)
- ✅ Multi-feed synchronization
- ✅ P&L calculations verified
- ✅ 159 unit tests passing
- ✅ 100% agreement with VectorBT on basic tests

## Risks and Assumptions

### Technical Risks
1. **Platform Installation** (HIGH): Zipline may have complex dependencies
2. **Vectorbt-Pro License** (HIGH): Commercial license may not be available
3. **Execution Model Differences** (HIGH): Platforms may have inherent differences
4. **ml4t.backtest Maturity** (MEDIUM): May discover missing features during testing

### Process Risks
5. **Scope Creep** (MEDIUM): Too many test cases initially
6. **Configuration Complexity** (MEDIUM): Getting identical settings across platforms

### Assumptions
- Same execution rules will produce same results
- Platforms support comparable order types
- Data quality is sufficient across all datasets
- Differences are explainable by platform design

## Success Metrics

1. **Signal Consistency**: 100% match across platforms
2. **Trade Count**: 100% match (same signals = same trades)
3. **Fill Prices**: <0.1% difference (with same rules)
4. **Final P&L**: <1% difference or explained by platform design
5. **Coverage**: 80%+ code coverage
6. **Documentation**: All differences documented and explained
