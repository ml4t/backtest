# Comprehensive TDD Implementation Plan: ml4t.backtest Feature Parity Validation

**Work Unit**: 005_validation_infrastructure_real_data
**Version**: 2.0 (Approved 2025-11-04)
**Timeline**: 5-6 months, ~480 hours
**Scope**: 100 validation scenarios across 5 phases

---

## Executive Summary

### Approach
**Hybrid Validation Strategy**:
- ✅ Validate 70% existing ml4t.backtest features (primary focus)
- 🔍 Document 30% missing features for future implementation

### Key Findings from Code Exploration

**ml4t.backtest Implementation Status: 70-80% Complete**

| Category | Status | Details |
|----------|--------|---------|
| **Order Types** | ✅ 100% | All 7 types implemented (MARKET, LIMIT, STOP, STOP_LIMIT, TRAILING_STOP, BRACKET, OCO) |
| **Execution Engine** | ✅ 95% | Event-driven with intrabar precision, needs multi-timeframe validation |
| **Slippage Models** | ✅ 100% | 8 models including VectorBT-compatible |
| **Commission Models** | ✅ 100% | 9 models including tiered and VectorBT-compatible |
| **Position Management** | ✅ 90% | Multi-asset tracking, margin support, needs edge case validation |
| **Portfolio Optimization** | ❌ 0% | Missing: mean-variance, Kelly, risk parity (flag for future) |
| **Pipeline API** | ❌ 0% | Missing: Zipline-style factors (flag for future) |
| **Performance** | ⚠️ 30% | Ad-hoc tests exist, needs systematization |

**Critical Discovery**: ml4t.backtest has **excellent feature coverage** - most work is validation, not implementation!

---

## Timeline Overview

```
Month 1-2 (160h): Critical Path
├─ Week 1-2: Platform debugging (20h)
├─ Week 3-6: Core order validation (70h)
├─ Week 7-8: Cost models (40h)
└─ Week 9-10: Position management (30h)

Month 3 (120h): Core Features
├─ Advanced orders (50h)
├─ Portfolio & risk (40h)
└─ Data handling (30h)

Month 4 (80h): Platform Parity
├─ VectorBT-specific (30h)
├─ Zipline-specific (30h)
└─ Backtrader-specific (20h)

Month 5 (80h): Performance & Scale
├─ Large universe (40h)
└─ Historical depth (40h)

Month 6 (40h): Polish & Documentation
├─ Documentation (20h)
└─ CI/CD automation (20h)

Total: 480 hours over 6 months
```

---

## Phase 1: Critical Path (Months 1-2, 160 hours)

### Week 1-2: Platform Debugging (20 hours)

#### TASK-001: Debug ml4t.backtest Signal Processing (6h)
**Priority**: P0 (Critical)
**Status**: 🔴 BLOCKED - 0 trades extracted

**TDD Cycle**:
```python
# RED
def test_ml4t.backtest_executes_market_order():
    """ml4t.backtest should execute BUY market order and fill at next bar"""
    assert len(orders_placed) >= 1
    assert orders_filled[0].status == OrderStatus.FILLED
    # Currently: ❌ Fails (0 orders)

# GREEN
- Add verbose logging to runner.py
- Fix signal timestamp timezone (UTC)
- Verify data feed alignment
- Ensure broker receives orders
# Expected: ✅ Passes (1+ orders)

# REFACTOR
- Extract signal validation helper
- Add clear error messages for signal format
- Document timezone requirements
```

**Files**:
- `tests/validation/runner.py` (fix signal processing)
- `tests/validation/test_ml4t.backtest_integration.py` (NEW - diagnostic tests)

**Acceptance Criteria**:
- ✅ ml4t.backtest executes 4 signals from Scenario 001
- ✅ 2 complete trades extracted
- ✅ Prices match expected OHLC data
- ✅ No timezone errors

---

#### TASK-002: Debug VectorBT Signal Processing (6h)
**Priority**: P0 (Critical)
**Status**: 🔴 BLOCKED - 0 trades extracted

**TDD Cycle**:
```python
# RED
def test_vectorbt_executes_market_order():
    """VectorBT should execute entries and exits correctly"""
    assert portfolio.orders.count() >= 2
    assert len(portfolio.trades.records) == 1
    # Currently: ❌ Fails (0 trades)

# GREEN
- Fix entries/exits DataFrame alignment
- Ensure index matches data timestamps
- Verify signal format (True/False array)
- Test with VectorBT same-bar execution
# Expected: ✅ Passes (1+ trades)

# REFACTOR
- Create VectorBT signal converter utility
- Document execution timing (same-bar vs next-bar)
- Add helper for signal DataFrame creation
```

**Files**:
- `tests/validation/runner.py` (fix VectorBT integration)
- `tests/validation/test_vectorbt_integration.py` (NEW)

**Acceptance Criteria**:
- ✅ VectorBT executes 4 signals
- ✅ 2 complete trades extracted
- ✅ Same-bar execution validated
- ✅ Commission properly calculated

---

#### TASK-003: Test Zipline Integration (4h)
**Priority**: P1 (High)
**Status**: ⏸️ READY - Bundle ingested, untested

**TDD Cycle**:
```python
# RED
def test_zipline_executes_with_validation_bundle():
    """Zipline should execute using custom validation bundle"""
    assert len(perf['transactions']) >= 4
    assert perf['returns'].notna().all()
    # Expected: May pass or fail depending on setup

# GREEN
- Test bundle access in algorithm
- Verify symbol lookup works
- Check calendar alignment
- Validate commission model
# Expected: ✅ Passes

# REFACTOR
- Simplify bundle registration
- Add bundle validation check
- Improve error messages
```

**Files**:
- `tests/validation/runner.py` (test Zipline execution)
- `tests/validation/bundles/validation_ingest.py` (verify)
- `tests/validation/test_zipline_integration.py` (NEW)

**Acceptance Criteria**:
- ✅ Zipline executes without errors
- ✅ Trades extracted from transactions
- ✅ Bundle data accessed correctly
- ✅ Results comparable to other platforms

---

#### TASK-004: Cross-Platform Validation (4h)
**Priority**: P0 (Critical)
**Status**: ⏳ BLOCKED by TASKS 001-003

**TDD Cycle**:
```python
# RED
def test_all_platforms_scenario_001():
    """All 4 platforms should execute Scenario 001 successfully"""
    for platform in ['ml4t.backtest', 'vectorbt', 'backtrader', 'zipline']:
        assert len(trades[platform]) == 2
        assert trades[platform][0].entry_price > 0
    # Currently: ❌ Fails (ml4t.backtest, vectorbt have 0 trades)

# GREEN
- Complete TASKS 001-003
- Run all platforms
- Verify trade extraction
# Expected: ✅ Passes (all platforms working)

# REFACTOR
- Document platform execution differences
- Create comparison table (timing, prices)
- Add platform-specific notes
```

**Files**:
- `tests/validation/test_cross_platform.py` (NEW - integration test)
- `tests/validation/STATUS.md` (update with results)

**Acceptance Criteria**:
- ✅ All 4 platforms execute Scenario 001
- ✅ Each extracts 2 trades
- ✅ Validation report generated
- ✅ Differences documented

---

### Week 3-6: Core Order Validation (70 hours)

#### TASK-005: Market Orders (Scenarios 001-010, 12h)
**Priority**: P1 (High)
**Status**: ⏳ PENDING - Blocked by TASK-004

**Scope**: 100 market order scenarios across all platforms

**Scenarios**:
- 001: ✅ Simple market orders (existing)
- 002: Large quantity market orders (10,000 shares)
- 003: Small quantity market orders (1 share)
- 004: Fractional shares (0.5 shares, if supported)
- 005: Multiple simultaneous market orders (same bar)
- 006: Market orders with gaps in data
- 007: Market orders at market open
- 008: Market orders at market close
- 009: Market orders with very high volume
- 010: Market orders with very low volume

**TDD Cycle**:
```python
# RED
def test_market_orders_cross_platform():
    """100 market order scenarios should agree within 0.01%"""
    for scenario in market_scenarios:
        trades = run_all_platforms(scenario)
        agreement = calculate_agreement(trades)
        assert agreement['fill_price_pct'] >= 99.99
        assert agreement['fill_quantity'] == 100.0
```

**Acceptance Criteria**:
- ✅ 99%+ agreement on fill prices
- ✅ 100% agreement on fill quantities
- ✅ All platforms use correct OHLC component
- ✅ Execution timing documented

**Files**:
- `tests/validation/scenarios/scenarios_002_010_market_orders.py` (NEW)
- `tests/validation/test_market_orders_validation.py` (NEW)

---

#### TASK-006: Limit Orders (Scenarios 011-020, 14h)
**Priority**: P1 (High)
**Status**: ⏳ PENDING

**Scope**: 50 limit order scenarios

**Scenarios**:
- 011: BUY limit below market (should fill)
- 012: BUY limit above market (should not fill immediately)
- 013: SELL limit above market (should fill)
- 014: SELL limit below market (should not fill immediately)
- 015: Limit order fills partially
- 016: Limit order never triggers (expired)
- 017: Limit order triggers on gap
- 018: Limit order with high intrabar volatility
- 019: Limit order fills at exact limit price
- 020: Multiple limit orders at same price

**TDD Cycle**:
```python
# RED
def test_limit_orders_trigger_correctly():
    """Limit orders should only fill when price touches limit"""
    for scenario in limit_scenarios:
        trades = run_all_platforms(scenario)
        for trade in trades.values():
            # BUY limit: fill price ≤ limit price
            if trade.direction == 'BUY':
                assert trade.entry_price <= scenario.limit_price + tolerance
            # SELL limit: fill price ≥ limit price
            else:
                assert trade.entry_price >= scenario.limit_price - tolerance
```

**Acceptance Criteria**:
- ✅ 97%+ agreement on fill logic (trigger vs no trigger)
- ✅ 99%+ agreement on fill prices when triggered
- ✅ Partial fills handled consistently
- ✅ Expiration logic validated

**Files**:
- `tests/validation/scenarios/scenarios_011_020_limit_orders.py` (NEW)
- `tests/validation/test_limit_orders_validation.py` (NEW)

---

#### TASK-007: Stop Orders (Scenarios 021-030, 12h)
**Priority**: P1 (High)
**Status**: ⏳ PENDING

**Scope**: 40 stop order scenarios

**Scenarios**:
- 021: STOP buy above market (breakout)
- 022: STOP sell below market (stop-loss)
- 023: Stop triggers on gap
- 024: Stop triggers intrabar
- 025: Stop never triggers
- 026: Multiple stops at same level
- 027: Stop with high volatility
- 028: Stop at market open
- 029: Stop at market close
- 030: Stop order priority vs limit order

**TDD Cycle**:
```python
# RED
def test_stop_orders_trigger_at_correct_price():
    """Stop orders should trigger when price crosses stop level"""
    for scenario in stop_scenarios:
        trades = run_all_platforms(scenario)
        for trade in trades.values():
            # Verify stop was triggered
            assert trade.triggered == True
            # Verify trigger price ≥ stop price (for stop buy)
            assert trade.trigger_price >= scenario.stop_price - tolerance
```

**Acceptance Criteria**:
- ✅ 97%+ agreement on trigger logic
- ✅ 98%+ agreement on fill prices after trigger
- ✅ Gap handling consistent
- ✅ Priority rules validated

**Files**:
- `tests/validation/scenarios/scenarios_021_030_stop_orders.py` (NEW)
- `tests/validation/test_stop_orders_validation.py` (NEW)

---

#### TASK-008: Trailing Stops (Scenarios 031-040, 12h)
**Priority**: P1 (High)
**Status**: ⏳ PENDING

**Scope**: 30 trailing stop scenarios

**Scenarios**:
- 031: Trailing stop absolute ($10 trail)
- 032: Trailing stop percentage (5% trail)
- 033: Trailing stop dynamic (ATR-based)
- 034: Trailing stop never triggers (profit)
- 035: Trailing stop triggers on reversal
- 036: Trailing stop with high volatility
- 037: Trailing stop with gaps
- 038: Multiple trailing stops
- 039: Trailing stop priority
- 040: Trailing stop reset logic

**TDD Cycle**:
```python
# RED
def test_trailing_stop_tracks_peak_correctly():
    """Trailing stop should track highest price and trigger on reversal"""
    for scenario in trailing_scenarios:
        trades = run_all_platforms(scenario)
        for trade in trades.values():
            # Verify peak tracking
            assert trade.peak_price == scenario.expected_peak
            # Verify trigger price = peak - trail_amount
            expected_trigger = trade.peak_price - scenario.trail_amount
            assert abs(trade.trigger_price - expected_trigger) < tolerance
```

**Acceptance Criteria**:
- ✅ 95%+ agreement on peak tracking
- ✅ 97%+ agreement on trigger prices
- ✅ Absolute vs percentage modes validated
- ✅ Dynamic trailing logic tested

**Files**:
- `tests/validation/scenarios/scenarios_031_040_trailing_stops.py` (NEW)
- `tests/validation/test_trailing_stops_validation.py` (NEW)

---

#### TASK-009: Bracket Orders (Scenarios 041-050, 10h)
**Priority**: P1 (High)
**Status**: ⏳ PENDING

**Scope**: 40 bracket order scenarios

**Scenarios**:
- 041: Bracket with TP and SL
- 042: Bracket with TSL (trailing stop-loss)
- 043: Bracket with TP only
- 044: Bracket with SL only
- 045: Bracket priority (TP vs TSL)
- 046: Bracket with multiple TPs
- 047: Bracket with gaps
- 048: Bracket never triggers (flat market)
- 049: Bracket triggers intrabar
- 050: VectorBT bracket compatibility

**TDD Cycle**:
```python
# RED
def test_bracket_orders_execute_correctly():
    """Bracket orders should manage TP/SL/TSL with correct priority"""
    for scenario in bracket_scenarios:
        trades = run_all_platforms(scenario)
        for trade in trades.values():
            # Verify correct exit type
            assert trade.exit_type in ['TP', 'SL', 'TSL']
            # Verify price matches exit type
            if trade.exit_type == 'TP':
                assert trade.exit_price >= scenario.tp_price - tolerance
            elif trade.exit_type == 'SL':
                assert trade.exit_price <= scenario.sl_price + tolerance
```

**Acceptance Criteria**:
- ✅ 95%+ agreement on exit type (TP/SL/TSL)
- ✅ 97%+ agreement on exit prices
- ✅ Priority rules validated (TP > TSL > SL)
- ✅ VectorBT bracket compatibility confirmed

**Files**:
- `tests/validation/scenarios/scenarios_041_050_bracket_orders.py` (NEW)
- `tests/validation/test_bracket_orders_validation.py` (NEW)

---

#### TASK-010: OCO Orders (Scenarios 051-060, 6h)
**Priority**: P2 (Medium)
**Status**: ⏳ PENDING

**Scope**: 25 OCO (One-Cancels-Other) scenarios

**Scenarios**:
- 051: OCO with 2 limit orders
- 052: OCO with 2 stop orders
- 053: OCO with limit + stop
- 054: OCO cancellation logic
- 055: OCO with partial fills
- 056: OCO priority rules
- 057: OCO with multiple assets
- 058: OCO expiration
- 059: OCO with gaps
- 060: OCO modification

**TDD Cycle**:
```python
# RED
def test_oco_orders_cancel_correctly():
    """When one OCO order fills, the other should be cancelled"""
    for scenario in oco_scenarios:
        trades = run_all_platforms(scenario)
        for result in trades.values():
            # Verify one order filled
            assert len(result.filled_orders) == 1
            # Verify other order cancelled
            assert len(result.cancelled_orders) == 1
            # Verify correct order filled
            assert result.filled_orders[0].order_id == scenario.expected_fill_id
```

**Acceptance Criteria**:
- ✅ 95%+ agreement on which order fills
- ✅ 98%+ agreement on cancellation timing
- ✅ Priority rules validated
- ✅ Edge cases handled

**Files**:
- `tests/validation/scenarios/scenarios_051_060_oco_orders.py` (NEW)
- `tests/validation/test_oco_orders_validation.py` (NEW)

---

#### TASK-011: Stop-Limit Orders (Scenarios 061-070, 4h)
**Priority**: P2 (Medium)
**Status**: ⏳ PENDING

**Scope**: 20 stop-limit scenarios

**Scenarios**:
- 061: Stop-limit buy (both trigger)
- 062: Stop-limit buy (stop triggers, limit doesn't)
- 063: Stop-limit sell (both trigger)
- 064: Stop-limit sell (stop triggers, limit doesn't)
- 065: Stop-limit with gap
- 066: Stop-limit priority
- 067: Stop-limit expiration
- 068: Stop-limit modification
- 069: Stop-limit partial fill
- 070: Stop-limit vs pure stop

**TDD Cycle**:
```python
# RED
def test_stop_limit_dual_trigger():
    """Stop-limit should require both stop and limit conditions"""
    for scenario in stop_limit_scenarios:
        trades = run_all_platforms(scenario)
        for trade in trades.values():
            # Verify stop triggered
            assert trade.stop_triggered == True
            # Verify limit respected
            if trade.direction == 'BUY':
                assert trade.fill_price <= scenario.limit_price + tolerance
            else:
                assert trade.fill_price >= scenario.limit_price - tolerance
```

**Acceptance Criteria**:
- ✅ 95%+ agreement on trigger logic
- ✅ 97%+ agreement on fill prices
- ✅ Gap handling validated
- ✅ Stop vs stop-limit differences clear

**Files**:
- `tests/validation/scenarios/scenarios_061_070_stop_limit.py` (NEW)
- `tests/validation/test_stop_limit_validation.py` (NEW)

---

### Week 7-8: Cost Models (40 hours)

#### TASK-012: Slippage Model Validation (Scenarios 071-080, 20h)
**Priority**: P1 (High)
**Status**: ⏳ PENDING

**Scope**: 30 slippage scenarios across 8 models

**Models to Validate** (from ml4t.backtest):
1. NoSlippage
2. FixedSlippage (absolute $)
3. PercentageSlippage (% of price)
4. VolumeShareSlippage (sqrt impact)
5. BidAskSlippage (spread-based)
6. RandomSlippage (noise)
7. TieredSlippage (volume-dependent)
8. VectorBTSlippage (VBT-compatible)

**Scenarios**:
- 071-073: Fixed slippage (small/medium/large orders)
- 074-076: Percentage slippage (low/medium/high volatility)
- 077-079: Volume share slippage (varying volume)
- 080: VectorBT slippage compatibility

**TDD Cycle**:
```python
# RED
def test_slippage_models_apply_correctly():
    """Each slippage model should apply cost according to its logic"""
    for model in slippage_models:
        for scenario in scenarios:
            trades = run_with_slippage(scenario, model)
            expected_slippage = calculate_expected_slippage(scenario, model)
            actual_slippage = trades['actual_price'] - trades['mid_price']
            assert abs(actual_slippage - expected_slippage) < tolerance
```

**Acceptance Criteria**:
- ✅ All 8 models validated
- ✅ 98%+ agreement with theoretical calculations
- ✅ VectorBT compatibility confirmed
- ✅ Edge cases tested (zero volume, gaps, etc.)

**Files**:
- `tests/validation/scenarios/scenarios_071_080_slippage.py` (NEW)
- `tests/validation/test_slippage_models.py` (NEW)

---

#### TASK-013: Commission Model Validation (Scenarios 081-090, 15h)
**Priority**: P1 (High)
**Status**: ⏳ PENDING

**Scope**: 25 commission scenarios across 9 models

**Models to Validate** (from ml4t.backtest):
1. NoCommission
2. PerShareCommission (fixed per share)
3. PerTradeCommission (flat fee)
4. PerDollarCommission (% of notional)
5. TieredCommission (volume brackets)
6. MinimumCommission (min + per share)
7. MakerTakerCommission (rebates)
8. PercentageCommission (% wrapper)
9. VectorBTCommission (VBT-compatible)

**Scenarios**:
- 081-083: Per-share commission (small/medium/large orders)
- 084-085: Per-trade commission (multiple trades)
- 086-087: Per-dollar commission (varying notionals)
- 088: Tiered commission (volume brackets)
- 089: Maker-taker (rebates vs fees)
- 090: VectorBT commission compatibility

**TDD Cycle**:
```python
# RED
def test_commission_models_calculate_correctly():
    """Each commission model should calculate fees according to its logic"""
    for model in commission_models:
        for scenario in scenarios:
            trades = run_with_commission(scenario, model)
            expected_commission = calculate_expected_commission(scenario, model)
            assert abs(trades['commission'] - expected_commission) < 0.01
```

**Acceptance Criteria**:
- ✅ All 9 models validated
- ✅ 99.9%+ agreement with theoretical calculations
- ✅ VectorBT compatibility confirmed
- ✅ Rebate logic tested (maker-taker)

**Files**:
- `tests/validation/scenarios/scenarios_081_090_commission.py` (NEW)
- `tests/validation/test_commission_models.py` (NEW)

---

#### TASK-014: VectorBT Cost Compatibility (Scenarios 091-095, 5h)
**Priority**: P1 (High)
**Status**: ⏳ PENDING

**Scope**: 20 VectorBT compatibility scenarios

**Scenarios**:
- 091: VectorBT slippage exact match
- 092: VectorBT commission exact match
- 093: VectorBT bracket order costs
- 094: VectorBT infinite sizing with costs
- 095: VectorBT complex order costs

**TDD Cycle**:
```python
# RED
def test_vectorbt_cost_compatibility():
    """ml4t.backtest costs should exactly match VectorBT Pro when configured"""
    for scenario in vbt_scenarios:
        ml4t.backtest_trades = run_ml4t.backtest_with_vbt_costs(scenario)
        vbt_trades = run_vectorbt(scenario)
        # Exact match on net P&L
        assert abs(ml4t.backtest_trades['net_pnl'] - vbt_trades['net_pnl']) < 0.01
        # Exact match on commission
        assert abs(ml4t.backtest_trades['commission'] - vbt_trades['commission']) < 0.01
```

**Acceptance Criteria**:
- ✅ 100% P&L agreement with VectorBT (when configured identically)
- ✅ Commission calculations match exactly
- ✅ Slippage calculations match exactly
- ✅ Bracket order costs match

**Files**:
- `tests/validation/scenarios/scenarios_091_095_vbt_compatibility.py` (NEW)
- `tests/validation/test_vbt_cost_compatibility.py` (NEW)

---

### Week 9-10: Position Management (30 hours)

#### TASK-015: Position Sizing (Scenarios 096-100, 8h)
**Priority**: P1 (High)
**Status**: ⏳ PENDING

**Scope**: 20 position sizing scenarios

**Scenarios**:
- 096: Fixed share sizing (100 shares)
- 097: Fixed dollar sizing ($10,000)
- 098: Percentage of capital sizing (10%)
- 099: VectorBT infinite sizing (size=np.inf)
- 100: Risk-based sizing (1% risk per trade)

**TDD Cycle**:
```python
# RED
def test_position_sizing_calculates_correctly():
    """Position sizer should calculate correct share quantity"""
    for scenario in sizing_scenarios:
        trades = run_with_sizer(scenario)
        expected_size = calculate_expected_size(scenario)
        assert abs(trades['size'] - expected_size) < 1  # Within 1 share
```

**Acceptance Criteria**:
- ✅ All sizing modes validated
- ✅ VectorBT infinite sizing works
- ✅ Edge cases tested (insufficient capital, fractional shares)
- ✅ 99%+ agreement on calculated sizes

**Files**:
- `tests/validation/test_position_sizing.py` (NEW)

---

*(Continuing with remaining tasks...)*

**Due to length, I'll create separate detailed task files for Phases 2-5. The structure continues with the same TDD approach for all 100 scenarios.**

---

## Summary of Remaining Phases

### Phase 2: Core Features (Month 3, 120 hours)
- **TASK-016 to TASK-035**: Advanced orders, portfolio features, data handling
- **30 scenarios** covering time-in-force, leverage, risk controls, multi-timeframe

### Phase 3: Platform Parity (Month 4, 80 hours)
- **TASK-036 to TASK-050**: Platform-specific features and compatibility
- **20 scenarios** for VectorBT/Zipline/Backtrader unique capabilities

### Phase 4: Performance & Scale (Month 5, 80 hours)
- **TASK-051 to TASK-060**: Large universe and historical depth testing
- **10 scenarios** with 100-1000 assets, 5-60 years data

### Phase 5: Polish & Documentation (Month 6, 40 hours)
- **TASK-061 to TASK-070**: Documentation, CI/CD, reporting
- **10 deliverables** including feature matrix, migration guides, benchmarks

---

## Success Metrics

### Quantitative Targets
- ✅ **95%+ cross-platform agreement** on trade execution
- ✅ **99%+ agreement on fill prices** (market orders)
- ✅ **97%+ agreement on complex orders** (limit/stop/bracket)
- ✅ **<0.01% P&L variance** for identical strategies
- ✅ **80%+ test coverage** of validation framework
- ✅ **<2 seconds per scenario** (100 scenarios in <4 minutes)
- ✅ **<100MB memory** for 1000-asset universe

### Qualitative Targets
- ✅ All documented ml4t.backtest features validated
- ✅ Platform differences clearly explained
- ✅ Migration guides complete
- ✅ Performance characteristics benchmarked
- ✅ Known limitations documented

---

## Resource Requirements

**Total**: ~480 hours over 6 months

**Breakdown**:
- Testing development: 190 hours
- Implementation/fixes: 190 hours
- Documentation: 60 hours
- Performance/benchmarking: 40 hours

**Dependencies**:
- VectorBT (free version sufficient for most features)
- Zipline-reloaded (✅ installed)
- Backtrader (✅ installed)
- Real market data (✅ Quandl Wiki 1962-2018)
- Compute resources for performance tests

---

## Risk Assessment

### High Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| Platform execution differences | Could block comparisons | Document differences, standardize where possible |
| Performance bottlenecks | Could fail scale tests | Profile early, optimize incrementally |
| Test development time | Could extend timeline | Use templates, automate scenario generation |

### Medium Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| VectorBT Pro features | Some tests might need Pro | Start with free version, upgrade if needed |
| Data quality | Bad data = bad tests | Use vetted Quandl dataset, validate inputs |
| Missing implementations | Might need to implement features | Flag for future, document gaps clearly |

---

## Next Steps

### Immediate (This Week)
1. ✅ **Plan Approved** - Ready to execute
2. 🔄 **Start TASK-001** - Debug ml4t.backtest signal processing
3. 🔄 **Start TASK-002** - Debug VectorBT signal processing
4. ⏳ **Prepare TASK-003** - Zipline integration test

### Week 2
5. ⏳ **Complete TASK-004** - Cross-platform validation
6. ⏳ **Begin TASK-005** - Market order scenarios

### Month 1
7. ⏳ **Complete Phase 1 Weeks 1-2** - Platform debugging
8. ⏳ **Begin Phase 1 Week 3** - Core order validation

---

**End of Comprehensive Plan**

*For detailed task specifications, see individual task documents in `/planning/tasks/`*
