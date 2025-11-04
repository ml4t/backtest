# Implementation Plan: Test-Driven Validation Infrastructure

**Work Unit**: 005_validation_infrastructure_real_data
**Created**: 2025-11-04
**Approach**: Test-Driven Development (TDD)

## TDD Principles for This Project

### Red-Green-Refactor Cycle

For each task:
1. **🔴 RED**: Write a failing test first
   - Define expected behavior
   - Test should fail (code doesn't exist yet)
   - Verify test can actually fail

2. **🟢 GREEN**: Make the test pass
   - Write minimal code to pass the test
   - Don't worry about perfection
   - Focus on making it work

3. **🔵 REFACTOR**: Improve the code
   - Clean up implementation
   - Remove duplication
   - Improve names and structure
   - Tests should still pass

### Testing Strategy

**Unit Tests**: Test individual components in isolation
- Fixtures (data loading)
- Extractors (trade parsing)
- Matcher (trade comparison)
- Reporter (output generation)

**Integration Tests**: Test components working together
- Platform execution end-to-end
- Data flow through full pipeline
- Error handling and edge cases

**Validation Tests**: Test against known results
- Compare with manual calculations
- Verify against external tools
- Check mathematical correctness

---

## Phase 1: Fix Current Platform Issues (PRIORITY 🔴)

### TASK-001: Debug qengine Signal Processing

**Status**: 🔴 BLOCKED (0 trades extracted)

**TDD Approach**:

#### 1. Write Diagnostic Test (RED)
```python
# tests/validation/test_qengine_signal_processing.py

def test_qengine_executes_simple_buy_signal():
    """qengine should execute a simple BUY market order"""
    # Given: AAPL data and one BUY signal
    data = get_ticker_data('AAPL', '2017-02-01', '2017-02-28')
    signal = Signal(
        timestamp=datetime(2017, 2, 6, tzinfo=timezone.utc),
        asset='AAPL',
        action='BUY',
        quantity=100,
    )

    # When: Run qengine backtest
    runner = PlatformRunner(Scenario001)
    result = runner.run_qengine()

    # Then: Should have 1 order placed
    assert len(result.raw_results['orders']) >= 1

    # Then: Order should be filled
    filled_orders = [o for o in result.raw_results['orders'] if o.status == OrderStatus.FILLED]
    assert len(filled_orders) == 1

    # Then: Position should be 100 shares
    assert result.raw_results['final_position'] == 100
```

**Expected**: ❌ Test fails (no orders placed)

#### 2. Add Logging (GREEN step 1)
```python
# runner.py - Add verbose logging

def run_qengine(self):
    print(f"📊 Signals to process: {len(signals)}")
    for sig in signals:
        print(f"  - {sig.timestamp}: {sig.action} {sig.quantity} {sig.asset}")

    # ... execution ...

    print(f"📈 Orders placed: {len(broker.orders)}")
    print(f"📊 Final position: {portfolio.get_position('AAPL')}")
```

**Expected**: Identify where signal processing breaks

#### 3. Fix Signal Processing (GREEN step 2)

**Possible fixes**:
- Add timezone to signals: `datetime(..., tzinfo=timezone.utc)`
- Fix data feed timestamp alignment
- Correct signal-to-order translation
- Fix broker order placement

#### 4. Refactor (REFACTOR)
- Extract signal validation helper
- Add clear error messages
- Document signal format requirements

**Acceptance Criteria**:
- ✅ Test passes (qengine executes signal)
- ✅ At least 1 order placed
- ✅ Position correctly updated
- ✅ Trade extracted successfully

**Estimated Time**: 2-3 hours

---

### TASK-002: Debug VectorBT Signal Processing

**Status**: 🔴 BLOCKED (0 trades extracted)

**TDD Approach**:

#### 1. Write Diagnostic Test (RED)
```python
# tests/validation/test_vectorbt_signal_processing.py

def test_vectorbt_executes_simple_buy_signal():
    """VectorBT should execute a simple BUY signal"""
    # Given: Price data and entry signals
    data = get_ticker_data('AAPL', '2017-02-01', '2017-02-28')
    entries = pd.Series(False, index=data['timestamp'])
    entries.loc[datetime(2017, 2, 6)] = True

    # When: Run VectorBT portfolio
    portfolio = vbt.Portfolio.from_signals(
        close=data['close'],
        entries=entries,
        exits=pd.Series(False, index=data['timestamp']),
        init_cash=100_000,
    )

    # Then: Should have executed 1 trade entry
    assert portfolio.orders.count() >= 1
    assert portfolio.positions.count() >= 1
```

**Expected**: ❌ Test fails (identify issue)

#### 2. Fix Signal Alignment (GREEN)

**Possible fixes**:
- Convert polars to pandas correctly
- Align timestamps with index
- Set proper timezone
- Configure VectorBT execution timing

#### 3. Refactor (REFACTOR)
- Create VectorBT signal converter helper
- Document execution timing model
- Add validation for signal format

**Acceptance Criteria**:
- ✅ Test passes (VectorBT executes)
- ✅ Trades extracted from portfolio
- ✅ Entry/exit prices captured
- ✅ Commission calculated

**Estimated Time**: 2-3 hours

---

### TASK-003: Test Zipline Integration

**Status**: ⏸️ NOT TESTED

**TDD Approach**:

#### 1. Write Integration Test (RED)
```python
# tests/validation/test_zipline_integration.py

def test_zipline_executes_with_validation_bundle():
    """Zipline should execute using custom validation bundle"""
    # Given: Bundle is ingested
    # (Assume already done)

    # When: Run scenario 001 with Zipline
    runner = PlatformRunner(Scenario001)
    result = runner.run_zipline()

    # Then: Should complete without errors
    assert result.errors == []

    # Then: Should have performance data
    assert result.raw_results is not None
    assert 'perf' in result.raw_results

    # Then: Should have executed trades
    perf = result.raw_results['perf']
    transactions = perf['transactions']
    assert len(transactions) >= 2  # At least 1 buy + 1 sell
```

**Expected**: Could pass or fail depending on implementation

#### 2. Fix Any Issues (GREEN)

**Potential issues**:
- Bundle not loaded correctly
- Symbol lookup fails
- Signal timing misalignment
- Commission model incorrect

#### 3. Refactor (REFACTOR)
- Simplify bundle setup
- Add bundle validation check
- Improve error messages

**Acceptance Criteria**:
- ✅ Zipline executes without errors
- ✅ Trades extracted successfully
- ✅ Results comparable to other platforms
- ✅ Bundle data accessed correctly

**Estimated Time**: 1-2 hours

---

### TASK-004: Validate All 4 Platforms with Scenario 001

**Status**: ⏳ BLOCKED BY TASKS 001-003

**TDD Approach**:

#### 1. Write Cross-Platform Test (RED)
```python
# tests/validation/test_all_platforms_scenario_001.py

def test_all_platforms_execute_scenario_001():
    """All 4 platforms should execute scenario 001 successfully"""
    # When: Run all platforms
    runner = PlatformRunner(Scenario001)
    results = runner.run_all_platforms(['qengine', 'vectorbt', 'backtrader', 'zipline'])

    # Then: All should complete
    for platform, result in results.items():
        assert result.errors == [], f"{platform} had errors"

    # Then: All should extract trades
    for platform, result in results.items():
        trades = extract_trades(platform, result)
        assert len(trades) > 0, f"{platform} extracted no trades"
```

**Expected**: ❌ Fails until TASKS 001-003 complete

#### 2. Ensure All Pass (GREEN)
- Complete TASKS 001-003
- Run integration test
- Verify all 4 platforms working

#### 3. Document Differences (REFACTOR)
- Create execution model comparison table
- Document expected price differences
- Add platform-specific notes

**Acceptance Criteria**:
- ✅ All 4 platforms execute scenario 001
- ✅ Trades extracted from each platform
- ✅ Validation report generated
- ✅ Differences understood and documented

**Estimated Time**: 1 hour (after dependencies)

---

## Phase 2: Build Test Infrastructure (TDD Foundation)

### TASK-005: Unit Tests for Market Data Fixtures

**Status**: ⏳ PENDING

**TDD Approach**:

#### 1. Write Fixture Tests (RED)
```python
# tests/validation/test_fixtures_market_data.py

def test_load_wiki_prices_returns_dataframe():
    """load_wiki_prices should return polars DataFrame"""
    df = load_wiki_prices()
    assert isinstance(df, pl.DataFrame)
    assert 'ticker' in df.columns
    assert 'date' in df.columns
    assert len(df) > 1_000_000  # Should have lots of data


def test_get_ticker_data_filters_correctly():
    """get_ticker_data should filter by ticker and date"""
    df = get_ticker_data('AAPL', '2017-01-01', '2017-12-31')

    # Should have AAPL only
    assert df['symbol'].unique().to_list() == ['AAPL']

    # Should be in date range
    assert df['timestamp'].min() >= datetime(2017, 1, 1, tzinfo=timezone.utc)
    assert df['timestamp'].max() <= datetime(2017, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    # Should have ~250 trading days
    assert 240 <= len(df) <= 260


def test_get_ticker_data_has_ohlcv_columns():
    """get_ticker_data should return complete OHLCV data"""
    df = get_ticker_data('AAPL', '2017-01-01', '2017-12-31')

    required_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    assert all(col in df.columns for col in required_cols)

    # No nulls
    assert df.null_count().sum().sum() == 0


def test_get_ticker_data_timezone_aware():
    """Timestamps should be timezone-aware (UTC)"""
    df = get_ticker_data('AAPL', '2017-01-01', '2017-01-31')

    # Should be datetime with timezone
    assert df['timestamp'].dtype == pl.Datetime('us', 'UTC')
```

**Expected**: ✅ Should all pass (fixtures already working)

#### 2. Add Edge Case Tests (RED)
```python
def test_get_ticker_data_invalid_ticker():
    """Should handle invalid ticker gracefully"""
    df = get_ticker_data('INVALID_TICKER_XYZ', '2017-01-01', '2017-12-31')
    assert len(df) == 0  # Empty DataFrame


def test_get_ticker_data_future_dates():
    """Should handle future dates gracefully"""
    df = get_ticker_data('AAPL', '2030-01-01', '2030-12-31')
    assert len(df) == 0  # No data yet


def test_prepare_zipline_bundle_creates_hdf5():
    """prepare_zipline_bundle_data should create HDF5 file"""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = prepare_zipline_bundle_data(
            tickers=['AAPL'],
            start_date='2017-01-01',
            end_date='2017-01-31',
            output_dir=Path(tmpdir),
        )

        # Should create bundle file
        assert result['bundle_file'].exists()

        # Should have metadata
        assert len(result['metadata']) == 1
        assert result['metadata'][0]['symbol'] == 'AAPL'
```

**Expected**: May need to add error handling

#### 3. Implement Error Handling (GREEN)
- Add try/except in fixtures
- Return empty DataFrames for invalid inputs
- Add clear error messages

#### 4. Refactor (REFACTOR)
- Extract validation helpers
- Add type hints
- Improve documentation

**Acceptance Criteria**:
- ✅ All fixture functions have unit tests
- ✅ Edge cases covered
- ✅ 90%+ coverage of market_data.py
- ✅ Clear error messages for invalid inputs

**Estimated Time**: 2 hours

---

### TASK-006: Unit Tests for Trade Extractors

**Status**: ⏳ PENDING

**TDD Approach**:

#### 1. Write Extractor Tests (RED)
```python
# tests/validation/test_extractors.py

def test_qengine_extractor_basic_trade():
    """QEngine extractor should parse basic BUY→SELL trade"""
    # Given: Mock qengine orders
    orders = [
        MockOrder(
            order_id=1,
            timestamp=datetime(2017, 2, 7),
            symbol='AAPL',
            direction=Direction.BUY,
            quantity=100,
            filled_price=130.0,
            status=OrderStatus.FILLED,
        ),
        MockOrder(
            order_id=2,
            timestamp=datetime(2017, 4, 18),
            symbol='AAPL',
            direction=Direction.SELL,
            quantity=100,
            filled_price=140.0,
            status=OrderStatus.FILLED,
        ),
    ]

    # When: Extract trades
    trades = extract_qengine_trades(orders, data)

    # Then: Should have 1 complete trade
    assert len(trades) == 1

    # Then: Trade details correct
    trade = trades[0]
    assert trade.entry_timestamp == datetime(2017, 2, 7)
    assert trade.entry_price == 130.0
    assert trade.exit_timestamp == datetime(2017, 4, 18)
    assert trade.exit_price == 140.0
    assert trade.pnl_net == 1000.0  # (140-130) * 100


def test_vectorbt_extractor_from_portfolio():
    """VectorBT extractor should parse portfolio trades"""
    # Given: Mock VectorBT portfolio
    portfolio = MockVectorBTPortfolio(
        trades_df=pd.DataFrame({
            'Entry Timestamp': [datetime(2017, 2, 7)],
            'Entry Price': [130.0],
            'Exit Timestamp': [datetime(2017, 4, 18)],
            'Exit Price': [140.0],
            'Size': [100],
            'PnL': [1000.0],
        })
    )

    # When: Extract trades
    trades = extract_vectorbt_trades(portfolio, data)

    # Then: Should match expected
    assert len(trades) == 1
    assert trades[0].entry_price == 130.0


def test_backtrader_extractor_from_notifications():
    """Backtrader extractor should parse notify_trade() data"""
    # Test similar to above
    ...


def test_zipline_extractor_from_transactions():
    """Zipline extractor should match transactions into trades"""
    # Test similar to above
    ...
```

**Expected**: ❌ May fail if extractors have bugs

#### 2. Fix Extractor Bugs (GREEN)
- Fix any parsing errors
- Handle edge cases
- Add error logging

#### 3. Refactor (REFACTOR)
- Extract common parsing logic
- Add helper functions
- Improve code clarity

**Acceptance Criteria**:
- ✅ All 4 extractors have unit tests
- ✅ Basic trade parsing works
- ✅ Edge cases handled
- ✅ 80%+ coverage of extractor files

**Estimated Time**: 3 hours

---

### TASK-007: Unit Tests for Trade Matcher

**Status**: ⏳ PENDING

**TDD Approach**:

#### 1. Write Matcher Tests (RED)
```python
# tests/validation/test_matcher.py

def test_matcher_matches_identical_trades():
    """Matcher should match trades with same timestamp"""
    # Given: Two identical trades from different platforms
    trade1 = StandardTrade(
        platform='qengine',
        entry_timestamp=datetime(2017, 2, 7),
        entry_price=130.0,
        # ...
    )
    trade2 = StandardTrade(
        platform='vectorbt',
        entry_timestamp=datetime(2017, 2, 7),
        entry_price=130.0,
        # ...
    )

    # When: Match trades
    matches = match_trades([trade1, trade2])

    # Then: Should have 1 match group
    assert len(matches) == 1
    assert len(matches[0].trades) == 2


def test_matcher_allows_timing_tolerance():
    """Matcher should allow configurable time tolerance"""
    # Given: Trades 1 day apart (next-bar vs same-bar)
    trade1 = StandardTrade(
        entry_timestamp=datetime(2017, 2, 6),  # Same-bar
        # ...
    )
    trade2 = StandardTrade(
        entry_timestamp=datetime(2017, 2, 7),  # Next-bar
        # ...
    )

    # When: Match with 24h tolerance
    matches = match_trades([trade1, trade2], time_tolerance=timedelta(hours=24))

    # Then: Should match
    assert len(matches) == 1


def test_matcher_calculates_price_differences():
    """Matcher should calculate price variance"""
    # Given: Trades with different prices
    trade1 = StandardTrade(entry_price=130.0, ...)
    trade2 = StandardTrade(entry_price=131.0, ...)

    # When: Match
    matches = match_trades([trade1, trade2])

    # Then: Should calculate difference
    assert matches[0].entry_price_diff_pct == pytest.approx(0.77, abs=0.01)


def test_matcher_identifies_severity():
    """Matcher should classify differences by severity"""
    # Test NONE, MINOR, MAJOR, CRITICAL severity levels
    ...
```

**Expected**: ✅ Should mostly pass (matcher working)

#### 2. Add Edge Cases (RED)
```python
def test_matcher_handles_unmatched_trades():
    """Matcher should handle trades with no match"""
    ...

def test_matcher_handles_empty_input():
    """Matcher should handle empty trade list"""
    ...
```

#### 3. Refactor (REFACTOR)
- Simplify matching logic
- Extract comparison helpers
- Add clear comments

**Acceptance Criteria**:
- ✅ All matching scenarios tested
- ✅ Tolerance configuration works
- ✅ Severity classification correct
- ✅ 90%+ coverage of matcher.py

**Estimated Time**: 2 hours

---

## Phase 3: Scenario Library Expansion (TDD for Each Scenario)

### TASK-008: Scenario 002 - Limit Orders (TDD)

**Status**: ⏳ PENDING
**Priority**: HIGH (Next after Phase 1)

**TDD Approach**:

#### 1. Write Scenario Test (RED)
```python
# tests/validation/scenarios/test_scenario_002_limit_orders.py

def test_scenario_002_all_platforms():
    """All platforms should execute limit orders correctly"""
    # Given: Scenario 002
    from scenarios.scenario_002_limit_orders import Scenario002

    # When: Run all platforms
    runner = PlatformRunner(Scenario002)
    results = runner.run_all_platforms()

    # Then: All should have 2 trades (2 limit orders filled)
    for platform, result in results.items():
        trades = extract_trades(platform, result)
        assert len(trades) == 2, f"{platform} should have 2 trades"

        # Then: Entry prices should be at or better than limit
        for trade in trades:
            assert trade.entry_price <= trade.limit_price  # For BUY
```

**Expected**: ❌ Fails (scenario doesn't exist yet)

#### 2. Create Scenario (GREEN)
```python
# scenarios/scenario_002_limit_orders.py

class Scenario002:
    """Limit order execution validation"""

    name = "002_limit_orders"
    description = "Test limit order triggering and fill prices"

    @staticmethod
    def get_data():
        return get_ticker_data('AAPL', '2017-01-01', '2017-12-31')

    signals = [
        Signal(
            timestamp=datetime(2017, 2, 6),
            asset='AAPL',
            action='BUY',
            quantity=100,
            order_type='LIMIT',
            limit_price=129.0,  # Below market, should fill
        ),
        Signal(
            timestamp=datetime(2017, 4, 17),
            asset='AAPL',
            action='SELL',
            quantity=100,
            order_type='LIMIT',
            limit_price=145.0,  # Above market, should fill
        ),
    ]
```

#### 3. Implement Platform Support (GREEN)
- Ensure all platforms handle LIMIT orders
- Test fill price logic
- Verify partial fill handling

#### 4. Refactor (REFACTOR)
- Extract limit order test patterns
- Create limit order validator helper
- Document platform differences

**Acceptance Criteria**:
- ✅ Scenario implemented
- ✅ All 4 platforms execute
- ✅ Limit prices respected
- ✅ Test passes

**Estimated Time**: 3-4 hours

---

### TASK-009: Scenario 003 - Stop Orders (TDD)

**Similar TDD approach to TASK-008**

**Estimated Time**: 3-4 hours

---

### TASK-010: Scenario 004 - Position Re-Entry (TDD)

**Similar TDD approach to TASK-008**

**Estimated Time**: 3-4 hours

---

### TASK-011: Scenario 005 - Multi-Asset (TDD)

**Similar TDD approach to TASK-008**

**Estimated Time**: 4 hours

---

## Phase 4: Production Polish

### TASK-012: Documentation

**Deliverables**:
- README with quick start
- Platform execution model comparison
- Troubleshooting guide
- Contribution guide

**Estimated Time**: 2 hours

---

### TASK-013: CI/CD Integration (Optional)

**Deliverables**:
- GitHub Actions workflow
- Automated test runs
- Coverage reporting

**Estimated Time**: 2 hours

---

## Summary

### Task Dependencies

```
Phase 1 (Critical Path):
  TASK-001 (qengine fix) ─┐
  TASK-002 (VectorBT fix) ├─→ TASK-004 (All platforms validated)
  TASK-003 (Zipline test) ─┘

Phase 2 (Parallel to Phase 1):
  TASK-005 (Fixture tests) ─┐
  TASK-006 (Extractor tests) ├─→ Strong test foundation
  TASK-007 (Matcher tests) ──┘

Phase 3 (After Phase 1):
  TASK-004 complete ─→ TASK-008 (Scenario 002) ─→ TASK-009, 010, 011...

Phase 4 (Final):
  All scenarios ─→ TASK-012 (Docs) ─→ TASK-013 (CI/CD)
```

### Time Estimates

| Phase | Tasks | Total Time |
|-------|-------|-----------|
| Phase 1 | TASK-001 to TASK-004 | 6-9 hours |
| Phase 2 | TASK-005 to TASK-007 | 7 hours |
| Phase 3 | TASK-008 to TASK-011 | 13-16 hours |
| Phase 4 | TASK-012 to TASK-013 | 4 hours |
| **Total** | | **30-36 hours** |

### Deliverables Checklist

**By End of Phase 1**:
- ✅ All 4 platforms execute scenario 001
- ✅ Trade-by-trade comparison working
- ✅ Real data integration complete

**By End of Phase 2**:
- ✅ 80%+ test coverage of validation framework
- ✅ Unit tests for all core components
- ✅ TDD discipline established

**By End of Phase 3**:
- ✅ Tier 1 scenarios complete (001-005)
- ✅ All scenario patterns documented
- ✅ Fixtures library built

**By End of Phase 4**:
- ✅ Production-ready validation suite
- ✅ Comprehensive documentation
- ✅ CI/CD integration (optional)

---

## Next Actions

**Immediate** (This Session):
1. ✅ Create work unit structure
2. ✅ Write requirements and exploration docs
3. ✅ Create this implementation plan
4. ⏳ Create state.json with task tracking
5. ⏳ Start TASK-001 (Debug qengine)

**Next Session**:
1. Complete TASK-001, 002, 003
2. Validate all 4 platforms
3. Start Phase 2 testing

**This Week**:
1. Complete Phase 1 and Phase 2
2. Begin scenario expansion

**This Month**:
1. Complete 25-scenario roadmap
2. Production polish
