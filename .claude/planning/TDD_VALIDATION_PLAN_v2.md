# ml4t.backtest TDD Validation Plan: 100-Scenario Roadmap
**Platform Parity Validation Against VectorBT Pro, Zipline, and Backtrader**

**Version**: 2.0  
**Date**: 2025-11-04  
**Timeline**: 5-6 months (~100 scenarios)  
**Approach**: Hybrid - Validate existing + Flag gaps for future implementation

---

## Executive Summary

### Current Implementation Reality (From Code Exploration)

**✅ WELL IMPLEMENTED (70-80% Complete):**
- **Order Types**: All 7 types (MARKET, LIMIT, STOP, STOP_LIMIT, TRAILING_STOP, BRACKET, OCO)
- **Execution Logic**: Event-driven with intrabar (OHLC) execution, lookahead prevention
- **Slippage Models**: 8 models including VectorBTSlippage
- **Commission Models**: 9 models including VectorBTCommission
- **Position Tracking**: Multi-asset with dual-tracker (PositionTracker + MarginAccount)
- **Trade Tracking**: Efficient entry/exit pairing with TradeTracker
- **Bracket Orders**: TP/SL/TSL with VectorBT-compatible priority (SL > TSL > TP)
- **Position Sizing**: VectorBTInfiniteSizer (size=np.inf) + 2 others

**⚠️ PARTIALLY IMPLEMENTED (Need Enhancement):**
- **Portfolio Optimization**: No mean-variance or Kelly criterion
- **Multi-Timeframe**: Basic support, needs validation
- **Risk Controls**: Basic leverage limits, no position limits
- **Data Handling**: Single-feed optimized, multi-feed exists but untested
- **Performance Benchmarks**: Ad-hoc, not systematic

**❌ MISSING / GAPS:**
- **Live Market Conditions**: No realistic volume/liquidity simulation by default
- **Corporate Actions**: Stub implementation, not validated
- **Advanced Risk**: No drawdown limits, position concentration limits
- **Framework Adapters**: VectorBT exists, Zipline/Backtrader need work

**🎯 Strategic Approach:**
1. **Months 1-2**: Fix critical gaps + Validate core implemented features
2. **Months 2-3**: Advanced order validation + Portfolio basics
3. **Months 3-4**: Platform-specific parity (VBT/Zipline/Backtrader unique features)
4. **Months 4-5**: Performance benchmarking (1000+ assets, 20+ years)
5. **Month 6**: Gap documentation + Migration guides

---

## Phase 1: Critical Path (Months 1-2, ~40 scenarios)

**Goal**: Fix current issues + Validate core event-driven engine against all 3 platforms

### Week 1-2: Current Platform Issues (TASKS 001-004)

**EXISTING KNOWN ISSUES FROM PLANNING DOCS:**

#### TASK-001: Backtrader Missing Trades Bug
**Priority**: P0 (Critical)  
**Status**: Implemented/Partial  
**RED**: Write test reproducing Backtrader missing 5 of 14 trades  
**GREEN**: Fix Backtrader adapter signal→order translation  
**REFACTOR**: Add comprehensive logging for order lifecycle  
**Acceptance**: 14/14 trades executed, matches ml4t.backtest exactly  
**Files**: `tests/validation/frameworks/backtrader_adapter.py`

#### TASK-002: Zipline Data Feed Complexity
**Priority**: P0 (Critical)  
**Status**: Implemented/Partial  
**RED**: Test Zipline timezone handling with crypto 24/7 data  
**GREEN**: Fix timezone conversion in Zipline adapter  
**REFACTOR**: Standardize data feed interface across adapters  
**Acceptance**: Zipline processes crypto data without TZ errors  
**Files**: `tests/validation/frameworks/zipline_adapter.py`

#### TASK-003: VectorBT Exit Priority Validation
**Priority**: P0 (Critical)  
**Status**: Implemented (lines 379-400 in broker.py)  
**RED**: Test all 6 combinations of (SL, TSL, TP) triggering simultaneously  
**GREEN**: Verify ml4t.backtest matches VectorBT priority (SL > TSL > TP)  
**REFACTOR**: Document priority logic with inline comments  
**Acceptance**: 100% match on exit priority across 20 scenarios  
**Files**: `tests/unit/test_exit_priority.py` (EXISTS), add multi-exit scenarios

#### TASK-004: Position Sync After Fills
**Priority**: P0 (Critical)  
**Status**: Fixed (documented in CLAUDE.md)  
**RED**: Test position query after fill (broker.get_position vs portfolio.positions.get)  
**GREEN**: Verify strategies use broker.get_position() consistently  
**REFACTOR**: Deprecate portfolio.positions.get() or sync automatically  
**Acceptance**: No stale position data after fills  
**Files**: Strategy adapters in `src/ml4t.backtest/strategy/`

### Week 3-4: Core Order Type Validation (TASKS 005-011)

**All order types exist - need comprehensive validation against all 3 platforms**

#### TASK-005: Market Orders - Cross-Platform Validation
**Priority**: P1 (High)  
**Status**: Implemented, needs validation  
**RED**: Test 100 market orders across ml4t.backtest/VectorBT/Zipline/Backtrader  
**GREEN**: Verify fill prices within 0.01% across platforms  
**REFACTOR**: Standardize market order test harness  
**Acceptance**: 99%+ agreement on fill price/quantity  
**Files**: `tests/validation/test_market_orders_cross_platform.py` (NEW)

#### TASK-006: Limit Orders - Intrabar Execution
**Priority**: P1 (High)  
**Status**: Implemented (lines 210-230 in order.py)  
**RED**: Test limit order triggering with OHLC: high touches limit for SELL, low for BUY  
**GREEN**: Verify intrabar vs end-of-bar execution matches VectorBT  
**REFACTOR**: Add visual test output showing bar OHLC + limit touch  
**Acceptance**: 100% match on limit trigger timing  
**Files**: `tests/validation/test_limit_orders_intrabar.py` (NEW)

#### TASK-007: Stop-Loss Orders - VectorBT Matching
**Priority**: P1 (High)  
**Status**: Implemented (TASK-013 completed per docs)  
**RED**: Test SL triggering with low touching stop for SELL (long exit)  
**GREEN**: Verify ml4t.backtest matches VectorBT SL fill prices exactly  
**REFACTOR**: Consolidate SL tests into single parameterized suite  
**Acceptance**: Zero discrepancies across 50 SL scenarios  
**Files**: `tests/unit/test_sl_vectorbt_matching.py` (EXISTS), extend coverage

#### TASK-008: Trailing Stop Loss - 4-Stage VectorBT Process
**Priority**: P1 (High)  
**Status**: Implemented (lines 450-505 in broker.py, TASK-018 completed)  
**RED**: Test TSL 4-stage: open→peak→check→high→recheck with varying peaks  
**GREEN**: Verify peak tracking matches VectorBT (metadata-based approach)  
**REFACTOR**: Add docstring explaining 4-stage process with example  
**Acceptance**: 100% TSL exit agreement with VectorBT across 30 scenarios  
**Files**: `tests/unit/test_tsl_vectorbt_matching.py` (EXISTS), add edge cases

#### TASK-009: Take Profit Orders - High Touch Detection
**Priority**: P1 (High)  
**Status**: Implemented (TASK-014 completed)  
**RED**: Test TP triggering when high >= limit_price for SELL (long TP)  
**GREEN**: Verify intrabar detection matches VectorBT exactly  
**REFACTOR**: Unified TP test suite with visual bar inspection  
**Acceptance**: 100% TP agreement on 40 scenarios  
**Files**: `tests/unit/test_tp_vectorbt_matching.py` (EXISTS), expand

#### TASK-010: Stop-Limit Orders - Two-Stage Trigger
**Priority**: P2 (Medium)  
**Status**: Implemented (lines 251-266 in order.py)  
**RED**: Test stop triggers → converts to limit → limit fills  
**GREEN**: Verify both stages work with intrabar execution  
**REFACTOR**: Clear logging for stop trigger + limit fill events  
**Acceptance**: 95%+ match with VectorBT (has limited stop-limit support)  
**Files**: `tests/validation/test_stop_limit_two_stage.py` (NEW)

#### TASK-011: Bracket Orders - VectorBT % Parameters
**Priority**: P1 (High)  
**Status**: Implemented (lines 56-63, 108-117 in order.py, TASK-016 completed)  
**RED**: Test bracket with tp_pct/sl_pct/tsl_pct matching VectorBT format  
**GREEN**: Verify percentage-based brackets create correct child orders  
**REFACTOR**: Add validation for both absolute and percentage bracket styles  
**Acceptance**: 100% match on bracket child order creation  
**Files**: `tests/unit/test_bracket_percentage.py` (EXISTS), full scenarios

### Week 5-6: Slippage & Commission Validation (TASKS 012-015)

#### TASK-012: VectorBT Slippage Replication
**Priority**: P1 (High)  
**Status**: Implemented (VectorBTSlippage lines 369-454 in slippage.py, TASK-010 completed)  
**RED**: Test multiplicative slippage: buy at price*(1+slip), sell at price*(1-slip)  
**GREEN**: Verify matches VectorBT exactly on 100 trades  
**REFACTOR**: Document formula with examples in docstring  
**Acceptance**: <0.01% deviation from VectorBT on 100 scenarios  
**Files**: `tests/unit/test_vectorbt_slippage.py` (EXISTS), comprehensive

#### TASK-013: VectorBT Commission Replication
**Priority**: P1 (High)  
**Status**: Implemented (VectorBTCommission lines 358-438 in commission.py, TASK-009 completed)  
**RED**: Test two-component: (order_value * fee_rate) + fixed_fee  
**GREEN**: Verify fees applied to slippage-adjusted prices  
**REFACTOR**: Add edge cases (zero fees, high fixed fees)  
**Acceptance**: Exact commission match with VectorBT  
**Files**: `tests/unit/test_vectorbt_commission.py` (EXISTS), edge cases

#### TASK-014: Asset Class Slippage Models
**Priority**: P2 (Medium)  
**Status**: Implemented (AssetClassSlippage lines 314-367 in slippage.py)  
**RED**: Test slippage differences: crypto (10bp) vs FX (0.5bp) vs futures (2bp)  
**GREEN**: Verify asset-specific rates applied correctly  
**REFACTOR**: Make rates configurable per asset  
**Acceptance**: Correct slippage per asset class  
**Files**: `tests/unit/test_asset_class_slippage.py` (NEW)

#### TASK-015: Commission Model Comparison
**Priority**: P2 (Medium)  
**Status**: Implemented (8 commission models exist)  
**RED**: Test all 8 commission models on same 100 trades  
**GREEN**: Verify commission impacts on final PnL  
**REFACTOR**: Create commission model comparison report generator  
**Acceptance**: Commission model guide with use case recommendations  
**Files**: `tests/validation/test_commission_comparison.py` (NEW)

### Week 7-8: Position Sizing & Cash Constraints (TASKS 016-020)

#### TASK-016: VectorBT Infinite Sizer (size=np.inf)
**Priority**: P1 (High)  
**Status**: Implemented (VectorBTInfiniteSizer lines 80-231 in position_sizer.py, TASK-011 completed)  
**RED**: Test max capital allocation with fees/slippage/granularity  
**GREEN**: Verify matches VectorBT size=np.inf exactly  
**REFACTOR**: Add visual output showing cash reservation for fees  
**Acceptance**: <0.001 BTC deviation from VectorBT on 20 scenarios  
**Files**: `tests/unit/test_position_sizing.py` (EXISTS), comprehensive

#### TASK-017: Cash Constraint Enforcement
**Priority**: P1 (High)  
**Status**: Implemented (lines 339-428 in fill_simulator.py)  
**RED**: Test order rejection when insufficient cash (with leverage=1.0)  
**GREEN**: Verify partial fills when close to cash limit  
**REFACTOR**: Clear error messages for insufficient funds  
**Acceptance**: No overdrafts, correct partial fill quantities  
**Files**: `tests/unit/test_cash_constraints.py` (EXISTS), edge cases

#### TASK-018: Leverage Limits (max_leverage)
**Priority**: P2 (Medium)  
**Status**: Implemented (max_leverage in broker.py line 85, fill_simulator.py line 82)  
**RED**: Test leverage limits: 1x (no leverage), 2x, 5x  
**GREEN**: Verify position sizing respects leverage constraint  
**REFACTOR**: Add leverage utilization reporting  
**Acceptance**: Positions capped at max_leverage * cash  
**Files**: `tests/validation/test_leverage_limits.py` (NEW)

#### TASK-019: Position Sizing Comparison
**Priority**: P2 (Medium)  
**Status**: Implemented (3 sizers exist)  
**RED**: Test FixedQuantity vs VectorBTInfinite vs PercentageOfEquity on same strategy  
**GREEN**: Generate comparative PnL report  
**REFACTOR**: Position sizer selection guide  
**Acceptance**: Clear guidance on when to use each sizer  
**Files**: `tests/validation/test_position_sizer_comparison.py` (NEW)

#### TASK-020: Granularity Rounding (Crypto)
**Priority**: P2 (Medium)  
**Status**: Implemented (VectorBTInfiniteSizer lines 192 in position_sizer.py)  
**RED**: Test BTC quantities rounded to 0.001 granularity  
**GREEN**: Verify matches VectorBT rounding behavior  
**REFACTOR**: Make granularity asset-specific  
**Acceptance**: Correct rounding, no over-allocation  
**Files**: `tests/validation/test_granularity_rounding.py` (NEW)

---

## Phase 2: Core Feature Validation (Months 2-3, ~30 scenarios)

### Week 9-10: Advanced Order Features (TASKS 021-025)

#### TASK-021: OCO (One-Cancels-Other) Orders
**Priority**: P2 (Medium)  
**Status**: Implemented (BracketOrderManager handles OCO, lines 597 in broker.py)  
**RED**: Test OCO: when SL fills, TP cancels (and vice versa)  
**GREEN**: Verify sibling order cancellation logic  
**REFACTOR**: Support standalone OCO (not just bracket legs)  
**Acceptance**: Correct OCO behavior across 20 scenarios  
**Files**: `tests/unit/test_oco_orders.py` (NEW)

#### TASK-022: Newly-Created Bracket Skip (VectorBT)
**Priority**: P1 (High)  
**Status**: Implemented (_newly_created_brackets set, lines 150, 428, 881 in broker.py)  
**RED**: Test bracket legs don't trigger on same bar they're created  
**GREEN**: Verify VectorBT-compatible skip behavior  
**REFACTOR**: Document why this prevents premature exits  
**Acceptance**: Bracket legs active starting next bar  
**Files**: `tests/validation/test_bracket_creation_skip.py` (NEW)

#### TASK-023: Execution Delay Toggle
**Priority**: P1 (High)  
**Status**: Implemented (execution_delay flag, line 84 in broker.py)  
**RED**: Test execution_delay=True (realistic) vs False (immediate)  
**GREEN**: Verify lookahead prevention with delay=True  
**REFACTOR**: Add test showing lookahead bias when delay=False  
**Acceptance**: Clear documentation of when to use each mode  
**Files**: `tests/unit/test_lookahead_prevention.py` (EXISTS), expand

#### TASK-024: Same-Bar Re-Entry Control
**Priority**: P2 (Medium)  
**Status**: Implemented (allow_immediate_reentry flag, line 86 in broker.py)  
**RED**: Test allow_immediate_reentry=True (realistic) vs False (VectorBT mode)  
**GREEN**: Verify exit → entry on same bar when True  
**REFACTOR**: Document VectorBT compatibility reason for False  
**Acceptance**: Correct re-entry behavior in both modes  
**Files**: `tests/validation/test_same_bar_reentry.py` (NEW)

#### TASK-025: Fill Price Correctness (Intrabar)
**Priority**: P1 (High)  
**Status**: Implemented (can_fill logic lines 173-304 in order.py)  
**RED**: Test fill prices: TP at high for long, SL at low for long, etc.  
**GREEN**: Verify fill prices within OHLC bounds  
**REFACTOR**: Visualize OHLC + fill price on test failures  
**Acceptance**: 100% fill prices within bar range  
**Files**: `tests/unit/test_fill_price_ohlc_bounds.py` (NEW)

### Week 11-12: Portfolio & Risk (TASKS 026-030)

#### TASK-026: Multi-Asset Position Tracking
**Priority**: P1 (High)  
**Status**: Implemented (PositionTracker, lines 122 in broker.py)  
**RED**: Test 10-asset portfolio with simultaneous trades  
**GREEN**: Verify position sync after fills  
**REFACTOR**: Add position summary reporting  
**Acceptance**: Correct positions for all assets  
**Files**: `tests/validation/test_multi_asset_tracking.py` (NEW)

#### TASK-027: Margin Account for Derivatives
**Priority**: P2 (Medium)  
**Status**: Implemented (MarginAccount, lines 24 in margin.py)  
**RED**: Test futures margin requirements (initial vs maintenance)  
**GREEN**: Verify margin calls and liquidations  
**REFACTOR**: Add margin utilization metrics  
**Acceptance**: Realistic margin enforcement  
**Files**: `tests/unit/test_margin.py` (EXISTS), comprehensive

#### TASK-028: Trade Tracking Efficiency
**Priority**: P2 (Medium)  
**Status**: Implemented (TradeTracker, lines 153 in broker.py)  
**RED**: Benchmark trade tracking overhead on 10,000 trades  
**GREEN**: Verify <1% performance impact  
**REFACTOR**: Optimize entry/exit pairing if needed  
**Acceptance**: Minimal performance cost  
**Files**: `tests/performance/test_trade_tracker_overhead.py` (NEW)

#### TASK-029: PnL Calculation Validation
**Priority**: P1 (High)  
**Status**: Implemented (TradeTracker.get_trades_df(), line 208 in broker.py)  
**RED**: Test PnL calculation with fees/slippage on 100 round-trip trades  
**GREEN**: Verify matches manual calculation  
**REFACTOR**: Add PnL breakdown (gross vs net)  
**Acceptance**: <0.01% deviation from manual calc  
**Files**: `tests/unit/test_pnl_calculations.py` (EXISTS), expand

#### TASK-030: Risk Metrics (Sharpe, Max DD)
**Priority**: P3 (Low)  
**Status**: Missing (qeval integration needed)  
**RED**: Test risk metric calculation on backtest results  
**GREEN**: Implement basic Sharpe/MaxDD calculation  
**REFACTOR**: Integrate with qeval for comprehensive metrics  
**Acceptance**: Correct risk metrics vs manual calculation  
**Implementation Need**: 4-8 hours  
**Files**: `tests/validation/test_risk_metrics.py` (NEW)

### Week 13-14: Data Handling (TASKS 031-035)

#### TASK-031: Daily Bars - Long Backtest
**Priority**: P1 (High)  
**Status**: Tested (wiki_prices validation exists)  
**RED**: Test 1962-2025 (60+ years) daily data processing  
**GREEN**: Verify memory efficiency and speed  
**REFACTOR**: Add progress reporting for long backtests  
**Acceptance**: Completes in <5 minutes  
**Files**: `tests/validation/test_daily_long_backtest.py` (NEW)

#### TASK-032: Minute Bars - NASDAQ100
**Priority**: P2 (Medium)  
**Status**: Untested  
**RED**: Test 2021-2022 minute bars (NASDAQ100 data)  
**GREEN**: Verify intraday strategy execution  
**REFACTOR**: Optimize minute bar processing  
**Acceptance**: Correct intraday signals and execution  
**Files**: `tests/validation/test_minute_bars_nasdaq100.py` (NEW)

#### TASK-033: Crypto 24/7 Markets
**Priority**: P2 (Medium)  
**Status**: Partially tested (crypto_futures project)  
**RED**: Test continuous trading (no market close)  
**GREEN**: Verify no weekend gaps in execution  
**REFACTOR**: Add crypto-specific asset specs  
**Acceptance**: Trades execute on weekends/holidays  
**Files**: `tests/validation/test_crypto_24_7.py` (NEW)

#### TASK-034: Irregular Timestamps
**Priority**: P3 (Low)  
**Status**: Supported (Clock handles irregular)  
**RED**: Test non-uniform bar spacing (tick data → irregular bars)  
**GREEN**: Verify Clock.sync() handles correctly  
**REFACTOR**: Document irregular timestamp handling  
**Acceptance**: No timestamp errors on irregular data  
**Files**: `tests/validation/test_irregular_timestamps.py` (NEW)

#### TASK-035: Multi-Timeframe Data
**Priority**: P3 (Low)  
**Status**: Implemented (multi-feed Clock support)  
**RED**: Test daily signals + minute execution  
**GREEN**: Verify timeframe synchronization  
**REFACTOR**: Add multi-timeframe example strategy  
**Acceptance**: Correct signal alignment across timeframes  
**Implementation Need**: 8-16 hours (validation only, core exists)  
**Files**: `tests/validation/test_multi_timeframe.py` (NEW)

---

## Phase 3: Platform Parity (Months 3-4, ~20 scenarios)

### Week 15-16: VectorBT-Specific Features (TASKS 036-040)

#### TASK-036: VectorBT Adapter Comprehensive Test
**Priority**: P1 (High)  
**Status**: Implemented, needs expansion  
**RED**: Test adapter on 20 diverse strategies  
**GREEN**: Verify 95%+ agreement on all metrics  
**REFACTOR**: Add adapter compatibility matrix  
**Acceptance**: Adapter reliable for validation work  
**Files**: `tests/validation/frameworks/vectorbt_adapter.py`, expand tests

#### TASK-037: VectorBT Pro Features (if available)
**Priority**: P2 (Medium)  
**Status**: Partial (Pro features in resources/)  
**RED**: Identify Pro-only features we can replicate  
**GREEN**: Implement subset of Pro features  
**REFACTOR**: Document Pro vs Open feature split  
**Acceptance**: Key Pro features available in ml4t.backtest  
**Implementation Need**: 16-40 hours (depends on features)  
**Files**: Research `resources/vectorbt.pro-main/`, implement in ml4t.backtest

#### TASK-038: VectorBT Performance Benchmarks
**Priority**: P1 (High)  
**Status**: Ad-hoc tests exist  
**RED**: Systematic benchmark: ml4t.backtest vs VectorBT on 10 strategies  
**GREEN**: Document speed tradeoffs (correctness vs speed)  
**REFACTOR**: Create benchmark suite runner  
**Acceptance**: Clear performance comparison report  
**Files**: `tests/performance/test_vectorbt_benchmark.py` (NEW)

#### TASK-039: VectorBT Signal Generation
**Priority**: P2 (Medium)  
**Status**: Implemented (adapters use signals)  
**RED**: Test complex signal logic (vectorized vs event-driven)  
**GREEN**: Verify signal timing matches VectorBT  
**REFACTOR**: Add signal visualization tools  
**Acceptance**: Signals match VectorBT exactly  
**Files**: `tests/validation/test_signal_generation.py` (NEW)

#### TASK-040: VectorBT Portfolio Optimization
**Priority**: P3 (Low)  
**Status**: Missing  
**RED**: Test mean-variance optimization  
**GREEN**: Implement basic portfolio optimizer  
**REFACTOR**: Integrate with qeval for advanced metrics  
**Acceptance**: Basic optimization working  
**Implementation Need**: 40-80 hours  
**Files**: `src/ml4t.backtest/portfolio/optimizer.py` (NEW)

### Week 17-18: Zipline-Specific Features (TASKS 041-045)

#### TASK-041: Zipline Adapter Fix & Test
**Priority**: P1 (High)  
**Status**: Implemented, has timezone bugs  
**RED**: Test Zipline adapter on 10 strategies  
**GREEN**: Fix timezone and data feed issues  
**REFACTOR**: Simplify Zipline data pipeline  
**Acceptance**: Adapter works reliably  
**Files**: `tests/validation/frameworks/zipline_adapter.py`, fix bugs

#### TASK-042: Zipline Pipeline API Compatibility
**Priority**: P2 (Medium)  
**Status**: Not implemented  
**RED**: Test ml4t.backtest with Zipline Pipeline signals  
**GREEN**: Create Zipline Pipeline → ml4t.backtest bridge  
**REFACTOR**: Document pipeline integration  
**Acceptance**: Can use Pipeline factors in ml4t.backtest  
**Implementation Need**: 24-40 hours  
**Files**: `src/ml4t.backtest/data/zipline_pipeline.py` (NEW)

#### TASK-043: Zipline Benchmark Comparison
**Priority**: P1 (High)  
**Status**: Untested  
**RED**: Benchmark ml4t.backtest vs Zipline on same 5 strategies  
**GREEN**: Document speed difference (ml4t.backtest ~126x faster)  
**REFACTOR**: Identify Zipline's bottlenecks  
**Acceptance**: Speed comparison report  
**Files**: `tests/performance/test_zipline_benchmark.py` (NEW)

#### TASK-044: Zipline Commission Models
**Priority**: P2 (Medium)  
**Status**: Partial (generic models exist)  
**RED**: Test Zipline-specific commission models  
**GREEN**: Implement missing Zipline commission types  
**REFACTOR**: Commission model compatibility guide  
**Acceptance**: Zipline commission parity  
**Implementation Need**: 8-16 hours  
**Files**: `src/ml4t.backtest/execution/zipline_commission.py` (NEW)

#### TASK-045: Zipline Corporate Actions
**Priority**: P3 (Low)  
**Status**: Stub implementation  
**RED**: Test stock splits and dividends  
**GREEN**: Implement basic corporate action handling  
**REFACTOR**: Corporate action event processing  
**Acceptance**: Basic splits/dividends work  
**Implementation Need**: 24-40 hours  
**Files**: `src/ml4t.backtest/execution/corporate_actions.py` (EXISTS, needs implementation)

### Week 19-20: Backtrader-Specific Features (TASKS 046-050)

#### TASK-046: Backtrader Adapter Fix & Test
**Priority**: P1 (High)  
**Status**: Has missing trades bug  
**RED**: Test adapter on 20 strategies  
**GREEN**: Fix missing trades issue  
**REFACTOR**: Improve Backtrader signal translation  
**Acceptance**: All trades execute correctly  
**Files**: `tests/validation/frameworks/backtrader_adapter.py`, fix bugs

#### TASK-047: Backtrader Indicator Library
**Priority**: P2 (Medium)  
**Status**: Not integrated  
**RED**: Test using Backtrader indicators with ml4t.backtest  
**GREEN**: Create Backtrader indicator bridge  
**REFACTOR**: Indicator compatibility layer  
**Acceptance**: Can use BT indicators  
**Implementation Need**: 16-32 hours  
**Files**: `src/ml4t.backtest/strategy/backtrader_indicators.py` (NEW)

#### TASK-048: Backtrader Benchmark Comparison
**Priority**: P1 (High)  
**Status**: Untested  
**RED**: Benchmark ml4t.backtest vs Backtrader on 5 strategies  
**GREEN**: Document speed difference (ml4t.backtest ~6x faster)  
**REFACTOR**: Identify Backtrader's strengths  
**Acceptance**: Speed comparison report  
**Files**: `tests/performance/test_backtrader_benchmark.py` (NEW)

#### TASK-049: Backtrader Sizer Compatibility
**Priority**: P2 (Medium)  
**Status**: Not implemented  
**RED**: Test Backtrader sizer logic  
**GREEN**: Implement Backtrader-compatible sizers  
**REFACTOR**: Sizer compatibility guide  
**Acceptance**: Backtrader sizer parity  
**Implementation Need**: 8-16 hours  
**Files**: `src/ml4t.backtest/execution/backtrader_sizers.py` (NEW)

#### TASK-050: Backtrader Analyzer Compatibility
**Priority**: P3 (Low)  
**Status**: Not implemented  
**RED**: Test Backtrader analyzers on ml4t.backtest results  
**GREEN**: Create analyzer compatibility layer  
**REFACTOR**: Result export to Backtrader format  
**Acceptance**: Can use BT analyzers  
**Implementation Need**: 16-24 hours  
**Files**: `src/ml4t.backtest/reporting/backtrader_analyzers.py` (NEW)

---

## Phase 4: Performance & Scale (Months 4-5, ~10 scenarios)

### Week 21-22: Large Universe Testing (TASKS 051-055)

#### TASK-051: 100-Asset Portfolio
**Priority**: P1 (High)  
**Status**: Untested at scale  
**RED**: Test 100-stock universe with daily rebalancing  
**GREEN**: Verify memory and speed acceptable  
**REFACTOR**: Optimize multi-asset processing  
**Acceptance**: Completes in <10 minutes  
**Files**: `tests/performance/test_100_asset_portfolio.py` (NEW)

#### TASK-052: 1000-Asset Universe
**Priority**: P2 (Medium)  
**Status**: Untested  
**RED**: Test 1000-stock universe (full market scan)  
**GREEN**: Benchmark memory usage and speed  
**REFACTOR**: Parallel processing for large universes  
**Acceptance**: Completes in <30 minutes  
**Implementation Need**: 16-32 hours (optimization)  
**Files**: `tests/performance/test_1000_asset_universe.py` (NEW)

#### TASK-053: Memory Profiling
**Priority**: P1 (High)  
**Status**: Ad-hoc  
**RED**: Profile memory usage on 100-asset, 20-year backtest  
**GREEN**: Identify memory bottlenecks  
**REFACTOR**: Optimize data structures  
**Acceptance**: <2GB memory for 100 assets, 20 years  
**Files**: `tests/performance/test_memory_profiling.py` (NEW)

#### TASK-054: Speed Benchmarks vs Alternatives
**Priority**: P1 (High)  
**Status**: Ad-hoc (existing data: ml4t.backtest 9-265x faster)  
**RED**: Systematic benchmark on 10 strategies across all platforms  
**GREEN**: Generate performance comparison matrix  
**REFACTOR**: Automated benchmark runner  
**Acceptance**: Comprehensive speed report  
**Files**: `tests/performance/test_comprehensive_benchmark.py` (NEW)

#### TASK-055: Parallel Strategy Execution
**Priority**: P3 (Low)  
**Status**: Not implemented  
**RED**: Test running 10 strategies in parallel  
**GREEN**: Implement basic parallel execution  
**REFACTOR**: Thread-safe data structures  
**Acceptance**: Linear speedup with cores  
**Implementation Need**: 24-40 hours  
**Files**: `src/ml4t.backtest/engine_parallel.py` (NEW)

### Week 23-24: Historical Depth (TASKS 056-060)

#### TASK-056: 20-Year Backtest
**Priority**: P1 (High)  
**Status**: Partially tested (wiki_prices 1962-2025)  
**RED**: Test 20-year daily backtest with realistic data  
**GREEN**: Verify memory efficiency  
**REFACTOR**: Streaming data processing if needed  
**Acceptance**: Completes with <2GB memory  
**Files**: `tests/validation/test_20_year_backtest.py` (NEW)

#### TASK-057: 60-Year Backtest (Maximum Depth)
**Priority**: P2 (Medium)  
**Status**: Untested  
**RED**: Test full wiki_prices 1962-2025 dataset  
**GREEN**: Benchmark speed and memory  
**REFACTOR**: Chunked processing for extreme depth  
**Acceptance**: Completes in <15 minutes  
**Files**: `tests/validation/test_60_year_backtest.py` (NEW)

#### TASK-058: High-Frequency 1-Year (Minute Bars)
**Priority**: P2 (Medium)  
**Status**: Untested  
**RED**: Test 1 year of minute bars (~100k bars)  
**GREEN**: Verify intraday execution correctness  
**REFACTOR**: Optimize minute bar processing  
**Acceptance**: Completes in <5 minutes  
**Files**: `tests/performance/test_hf_1year_minute.py` (NEW)

#### TASK-059: Tick Data Processing
**Priority**: P3 (Low)  
**Status**: Untested  
**RED**: Test tick-level data ingestion  
**GREEN**: Implement tick aggregation to bars  
**REFACTOR**: Streaming tick processor  
**Acceptance**: Can process 1M ticks in <2 minutes  
**Implementation Need**: 16-32 hours  
**Files**: `tests/performance/test_tick_processing.py` (NEW)

#### TASK-060: Data Feed Benchmarks
**Priority**: P2 (Medium)  
**Status**: Untested  
**RED**: Benchmark Polars vs Pandas data feeds  
**GREEN**: Document data feed performance  
**REFACTOR**: Optimize data feed pipeline  
**Acceptance**: Polars 2-5x faster than Pandas  
**Files**: `tests/performance/test_data_feed_benchmark.py` (NEW)

---

## Phase 5: Polish & Documentation (Month 6, ~10 scenarios)

### Week 25-26: Gap Documentation (TASKS 061-070)

#### TASK-061: Feature Comparison Matrix
**Priority**: P1 (High)  
**Status**: Partial (exists in planning docs)  
**RED**: Test all features across 4 platforms  
**GREEN**: Generate comprehensive comparison matrix  
**REFACTOR**: HTML report with feature coverage  
**Acceptance**: Clear feature parity status  
**Files**: `docs/FEATURE_COMPARISON_MATRIX.md` (NEW)

#### TASK-062: Migration Guide - VectorBT to ml4t.backtest
**Priority**: P1 (High)  
**Status**: Missing  
**RED**: Document API differences  
**GREEN**: Create migration guide with examples  
**REFACTOR**: Side-by-side code comparisons  
**Acceptance**: Users can migrate strategies  
**Files**: `docs/guides/MIGRATION_VECTORBT.md` (NEW)

#### TASK-063: Migration Guide - Zipline to ml4t.backtest
**Priority**: P2 (Medium)  
**Status**: Missing  
**RED**: Document Pipeline API differences  
**GREEN**: Create migration guide  
**REFACTOR**: Example conversions  
**Acceptance**: Users can migrate strategies  
**Files**: `docs/guides/MIGRATION_ZIPLINE.md` (NEW)

#### TASK-064: Migration Guide - Backtrader to ml4t.backtest
**Priority**: P2 (Medium)  
**Status**: Missing  
**RED**: Document indicator/sizer differences  
**GREEN**: Create migration guide  
**REFACTOR**: Example conversions  
**Acceptance**: Users can migrate strategies  
**Files**: `docs/guides/MIGRATION_BACKTRADER.md` (NEW)

#### TASK-065: Performance Comparison Report
**Priority**: P1 (High)  
**Status**: Missing  
**RED**: Aggregate all benchmark data  
**GREEN**: Generate professional report  
**REFACTOR**: Charts and tables  
**Acceptance**: Clear performance guidance  
**Files**: `docs/PERFORMANCE_COMPARISON.md` (NEW)

#### TASK-066: Known Limitations Document
**Priority**: P1 (High)  
**Status**: Informal  
**RED**: List all known gaps and workarounds  
**GREEN**: Create limitations doc  
**REFACTOR**: Prioritize roadmap items  
**Acceptance**: Users know what's not supported  
**Files**: `docs/KNOWN_LIMITATIONS.md` (NEW)

#### TASK-067: Use Case Selection Guide
**Priority**: P1 (High)  
**Status**: Missing  
**RED**: Analyze when to use each platform  
**GREEN**: Create decision tree guide  
**REFACTOR**: Flowchart and examples  
**Acceptance**: Clear framework selection guidance  
**Files**: `docs/FRAMEWORK_SELECTION_GUIDE.md` (NEW)

#### TASK-068: API Documentation Completion
**Priority**: P2 (Medium)  
**Status**: Partial  
**RED**: Audit all public APIs for docstrings  
**GREEN**: Complete missing docstrings  
**REFACTOR**: Generate Sphinx docs  
**Acceptance**: 100% API documentation  
**Files**: All `src/ml4t.backtest/` modules

#### TASK-069: Example Strategy Library
**Priority**: P2 (Medium)  
**Status**: Minimal  
**RED**: Create 10 reference strategies  
**GREEN**: Document each strategy  
**REFACTOR**: Add to examples/ directory  
**Acceptance**: Users have starting templates  
**Files**: `examples/strategies/` (expand)

#### TASK-070: CI/CD Integration
**Priority**: P1 (High)  
**Status**: Missing  
**RED**: Set up GitHub Actions for validation tests  
**GREEN**: Automate benchmark runs  
**REFACTOR**: Nightly cross-platform validation  
**Acceptance**: Automated testing on PRs  
**Files**: `.github/workflows/validation.yml` (NEW)

---

## Additional Scenarios (TASKS 071-100)

### Advanced Features (TASKS 071-080)

#### TASK-071: Kelly Criterion Position Sizing
**Implementation Need**: 16-24 hours  
**Files**: `src/ml4t.backtest/execution/kelly_sizer.py` (NEW)

#### TASK-072: Mean-Variance Portfolio Optimization
**Implementation Need**: 40-60 hours  
**Files**: `src/ml4t.backtest/portfolio/mvo.py` (NEW)

#### TASK-073: Realistic Volume/Liquidity Simulation
**Implementation Need**: 24-40 hours  
**Files**: `src/ml4t.backtest/execution/realistic_liquidity.py` (NEW)

#### TASK-074: Position Concentration Limits
**Implementation Need**: 8-16 hours  
**Files**: `src/ml4t.backtest/execution/risk_controls.py` (NEW)

#### TASK-075: Drawdown-Based Position Sizing
**Implementation Need**: 8-16 hours  
**Files**: `src/ml4t.backtest/execution/drawdown_sizer.py` (NEW)

#### TASK-076: Multi-Timeframe Strategy Validation
**Implementation Need**: 8-16 hours (validation only)  
**Files**: `tests/validation/test_multi_timeframe_strategy.py` (NEW)

#### TASK-077: Order Routing Logic
**Implementation Need**: 16-24 hours  
**Files**: `src/ml4t.backtest/execution/advanced_routing.py` (NEW)

#### TASK-078: Market Impact Decay Models
**Implementation Need**: 16-24 hours  
**Files**: `src/ml4t.backtest/execution/impact_decay.py` (NEW)

#### TASK-079: Walk-Forward Optimization
**Implementation Need**: 40-60 hours  
**Files**: `src/ml4t.backtest/optimization/walk_forward.py` (NEW)

#### TASK-080: Monte Carlo Simulation
**Implementation Need**: 24-40 hours  
**Files**: `src/ml4t.backtest/analysis/monte_carlo.py` (NEW)

### Integration & Ecosystem (TASKS 081-090)

#### TASK-081: qfeatures Integration Tests
**Files**: `tests/integration/test_qfeatures_ml4t.backtest.py` (NEW)

#### TASK-082: qeval Integration Tests
**Files**: `tests/integration/test_qeval_ml4t.backtest.py` (NEW)

#### TASK-083: End-to-End ML Pipeline Test
**Files**: `tests/integration/test_ml_pipeline_e2e.py` (NEW)

#### TASK-084: Real Data Ingestion Pipeline
**Files**: `tests/validation/test_real_data_pipeline.py` (NEW)

#### TASK-085: Multi-Source Data Merge
**Files**: `tests/validation/test_multi_source_merge.py` (NEW)

#### TASK-086: Data Quality Checks
**Files**: `tests/validation/test_data_quality.py` (NEW)

#### TASK-087: Reporting Format Compatibility
**Files**: `tests/validation/test_reporting_formats.py` (NEW)

#### TASK-088: Web Dashboard Integration
**Implementation Need**: 60-100 hours  
**Files**: `src/ml4t.backtest/reporting/dashboard.py` (NEW)

#### TASK-089: Live Trading Bridge (Paper)
**Implementation Need**: 80-120 hours  
**Files**: `src/ml4t.backtest/live/paper_trading.py` (NEW)

#### TASK-090: Broker Integration Framework
**Implementation Need**: 100-200 hours  
**Files**: `src/ml4t.backtest/live/broker_api.py` (NEW)

### Edge Cases & Robustness (TASKS 091-100)

#### TASK-091: Zero-Price Handling
**Files**: `tests/validation/test_zero_price_edge_case.py` (NEW)

#### TASK-092: Gap-Up/Gap-Down Scenarios
**Files**: `tests/validation/test_gap_scenarios.py` (NEW)

#### TASK-093: Extreme Volatility (Circuit Breakers)
**Files**: `tests/validation/test_extreme_volatility.py` (NEW)

#### TASK-094: Order Rejection Scenarios
**Files**: `tests/validation/test_order_rejections.py` (NEW)

#### TASK-095: Partial Fill Handling
**Files**: `tests/validation/test_partial_fills.py` (NEW)

#### TASK-096: Data Missing/NaN Handling
**Files**: `tests/validation/test_missing_data.py` (NEW)

#### TASK-097: Timestamp Precision Issues
**Files**: `tests/validation/test_timestamp_precision.py` (NEW)

#### TASK-098: Floating Point Edge Cases
**Files**: `tests/validation/test_float_precision.py` (NEW)

#### TASK-099: Concurrent Order Submission
**Files**: `tests/validation/test_concurrent_orders.py` (NEW)

#### TASK-100: Memory Leak Detection
**Files**: `tests/performance/test_memory_leaks.py` (NEW)

---

## Dependency Graph & Critical Path

### Critical Path (Fastest Route to Working Validation)

```
Month 1: TASKS 001-020 (Fix platforms → Validate core orders → Position sizing)
  └─→ Month 2: TASKS 021-035 (Advanced orders → Portfolio → Data handling)
       └─→ Month 3: TASKS 036-050 (Platform parity)
            └─→ Month 4: TASKS 051-060 (Performance benchmarks)
                 └─→ Month 5-6: TASKS 061-100 (Polish + Documentation)
```

### Parallel Work Streams

**Stream 1: VectorBT Validation** (Highest Priority)
- TASKS 001, 003, 005-009, 012-013, 016, 022, 036-040

**Stream 2: Zipline Validation** (Medium Priority)
- TASKS 002, 041-045

**Stream 3: Backtrader Validation** (Medium Priority)
- TASKS 001 (shared), 046-050

**Stream 4: Performance & Scale** (Can run independently)
- TASKS 051-060

**Stream 5: Documentation & Polish** (Final phase)
- TASKS 061-070

---

## Resource Estimation

### By Phase

| Phase | Duration | Task Count | Test Hours | Implementation Hours | Total Hours |
|-------|----------|------------|------------|---------------------|-------------|
| Phase 1 | 2 months | 40 | 120 | 60 | 180 |
| Phase 2 | 1 month | 30 | 80 | 100 | 180 |
| Phase 3 | 1 month | 20 | 60 | 120 | 180 |
| Phase 4 | 1 month | 10 | 40 | 80 | 120 |
| Phase 5 | 1 month | 10 | 30 | 50 | 80 |
| Additional | Ongoing | 30 | 100 | 400 | 500 |
| **Total** | **6 months** | **140** | **430** | **810** | **1240** |

### By Priority

| Priority | Task Count | Description |
|----------|------------|-------------|
| P0 (Critical) | 4 | Blocking bugs (TASKS 001-004) |
| P1 (High) | 35 | Core validation (Orders, Execution, Data) |
| P2 (Medium) | 40 | Platform features, Advanced validation |
| P3 (Low) | 31 | Nice-to-have, Future enhancements |
| **Total** | **110** | (30 tasks are future work) |

---

## Risk Assessment & Mitigations

### High-Risk Items

1. **VectorBT Pro Access**: Some Pro features may be inaccessible
   - **Mitigation**: Focus on open-source VectorBT features, document Pro gaps

2. **Zipline Complexity**: Data pipeline and timezone issues
   - **Mitigation**: Allocate extra time for Zipline adapter fixes

3. **Performance at Scale**: 1000+ asset universe may reveal bottlenecks
   - **Mitigation**: Profile early, optimize incrementally

4. **Missing Features**: Some advanced features not yet implemented
   - **Mitigation**: Hybrid approach - validate what exists, flag gaps

5. **Platform Differences**: Exact replication may be impossible
   - **Mitigation**: Document acceptable deviation thresholds (e.g., <0.1%)

### Medium-Risk Items

1. **Test Data Availability**: Some data sources may have gaps
   - **Mitigation**: Use multiple data sources, synthetic data for edge cases

2. **Framework Version Drift**: Platforms may update during 6-month timeline
   - **Mitigation**: Pin framework versions, document tested versions

3. **Resource Availability**: 1240 hours over 6 months = ~50 hours/week
   - **Mitigation**: Prioritize critical path, defer low-priority tasks

---

## Success Metrics

### Quantitative Targets

- **Correctness**: 95%+ agreement with VectorBT across 100 scenarios
- **Speed**: Maintain 10-100x speed advantage for event-driven execution
- **Coverage**: 80%+ test coverage on ml4t.backtest core
- **Scalability**: Handle 1000-asset universe, 20-year backtests
- **Documentation**: 100% API documentation, 4 migration guides

### Qualitative Targets

- **Production Readiness**: Clear guidance on when to use ml4t.backtest vs alternatives
- **Community Confidence**: Comprehensive validation builds trust
- **Developer Experience**: Easy migration from other platforms
- **Maintenance**: CI/CD automation reduces manual validation burden

---

## Deliverables

### Month 2 Checkpoint
- ✅ All critical platform bugs fixed (TASKS 001-004)
- ✅ Core order types validated against all 3 platforms (TASKS 005-011)
- ✅ Slippage/commission models match VectorBT (TASKS 012-015)
- ✅ Position sizing validated (TASKS 016-020)
- 📄 Phase 1 Report: Core Validation Status

### Month 3 Checkpoint
- ✅ Advanced order features validated (TASKS 021-025)
- ✅ Multi-asset portfolio tracking validated (TASKS 026-030)
- ✅ Data handling validated (TASKS 031-035)
- 📄 Phase 2 Report: Feature Validation Status

### Month 4 Checkpoint
- ✅ VectorBT parity achieved (TASKS 036-040)
- ✅ Zipline adapter working (TASKS 041-045)
- ✅ Backtrader adapter working (TASKS 046-050)
- 📄 Phase 3 Report: Platform Parity Status

### Month 5 Checkpoint
- ✅ Large universe testing complete (TASKS 051-055)
- ✅ Historical depth testing complete (TASKS 056-060)
- 📄 Phase 4 Report: Performance & Scale

### Month 6 Final Deliverables
- ✅ Feature comparison matrix (TASK 061)
- ✅ 4 migration guides (TASKS 062-064)
- ✅ Performance comparison report (TASK 065)
- ✅ Known limitations doc (TASK 066)
- ✅ Framework selection guide (TASK 067)
- ✅ Complete API documentation (TASK 068)
- ✅ Example strategy library (TASK 069)
- ✅ CI/CD automation (TASK 070)
- 📄 **Final Report: ml4t.backtest Production Readiness Assessment**

---

## TDD Methodology Template

For every task, apply this structure:

### RED Phase
```python
def test_feature_that_should_work():
    """Test description explaining what we're validating."""
    # Arrange: Set up test data
    # Act: Execute the feature
    # Assert: Verify expected behavior
    assert actual == expected, "Descriptive failure message"
```

### GREEN Phase
```python
# Implement minimal code to make test pass
# Focus on correctness, not optimization
```

### REFACTOR Phase
```python
# Improve code quality without changing behavior
# Add documentation, optimize performance
# Ensure tests still pass
```

### Acceptance Criteria Checklist
- [ ] Test passes consistently (run 10 times)
- [ ] Code quality checks pass (ruff, mypy)
- [ ] Test coverage >80% for new code
- [ ] Documentation updated
- [ ] Performance acceptable (if relevant)

---

## Next Steps

1. **Review & Approve** this plan with stakeholders
2. **Set up tracking** (GitHub Projects, Jira, or simple spreadsheet)
3. **Begin Phase 1 Week 1** with TASKS 001-004 (critical bugs)
4. **Weekly sync** to review progress and adjust priorities
5. **Monthly checkpoints** to assess timeline and scope

---

**Questions for User Before Starting:**

1. Do you want to prioritize any specific platform (VectorBT/Zipline/Backtrader)?
2. Are there specific features you need validated urgently?
3. Should we defer any low-priority tasks to focus on core validation?
4. Do you have access to VectorBT Pro license for Pro feature validation?

