# ml4t.backtest Testing Environment - Comprehensive Exploration Report

**Date**: November 4, 2025
**Scope**: Design and architecture for an incremental test suite for ml4t.backtest backtesting library
**Status**: Complete analysis with actionable recommendations

---

## Executive Summary

The ml4t.backtest backtesting library has a solid foundation with:
- **34 existing unit tests** covering broker operations, execution logic, and edge cases
- **1 validation scenario** (scenario_001: simple market orders) demonstrating cross-platform comparison
- **Cross-platform framework** supporting ml4t.backtest, VectorBT, Backtrader, and Zipline
- **Rich order type support**: Market, Limit, Stop, Stop-Limit, Trailing Stop, Bracket (OCA)

**Key Finding**: The existing testing infrastructure is well-designed and extensible. We need to leverage this foundation and build 10-20 additional validation scenarios that progress from simple to complex.

---

## Part 1: Current Testing Infrastructure Analysis

### 1.1 Directory Structure

```
tests/
├── unit/                          # 34 unit test files
│   ├── test_broker.py             # 380 lines - broker submission/execution
│   ├── test_advanced_orders.py     # 473 lines - complex order types
│   ├── test_bracket_percentage.py  # 213 lines - bracket order parameters
│   ├── test_trailing_stop_bug.py   # Edge case handling
│   ├── test_pnl_calculations.py    # Economics validation
│   └── 29 more specific tests      # Position tracking, margin, cash, etc.
│
└── validation/                     # Cross-platform validation
    ├── scenarios/                  # Test scenario specifications
    │   └── scenario_001_simple_market_orders.py  # Single scenario example
    ├── extractors/                 # Platform-specific trade extractors
    │   ├── ml4t.backtest.py              # Converts ml4t.backtest broker results to StandardTrade
    │   ├── vectorbt.py             # VectorBT Portfolio → StandardTrade
    │   ├── backtrader.py           # Backtrader trade records → StandardTrade
    │   └── zipline.py              # Zipline performance → StandardTrade
    ├── comparison/                 # Trade matching and reporting
    │   ├── matcher.py              # Match trades across platforms
    │   └── reporter.py             # Generate comparison reports
    ├── core/                       # Shared validation infrastructure
    │   └── trade.py                # StandardTrade dataclass + utilities
    ├── runner.py                   # Main scenario runner (CLI + programmatic)
    └── adapters/                   # Framework adapter stubs (development phase)
```

### 1.2 Key Infrastructure Components

#### StandardTrade Format (core/trade.py)
Platform-independent representation with:
- **Identity**: trade_id, platform
- **Entry**: timestamp, price, OHLC bar, inferred component (open/close/high/low)
- **Exit**: timestamp, price, reason (signal/stop_loss/take_profit/trailing_stop)
- **Economics**: gross_pnl, entry/exit commission, slippage, net_pnl
- **Metadata**: signal_id, notes, links to original data

**Purpose**: Allows precise comparison of how different platforms execute the same signals.

#### Scenario Specification Pattern (scenario_001)
```python
class Scenario:
    name = "scenario_id"
    description = "Human-readable description"
    
    @staticmethod
    def get_data() -> pl.DataFrame:
        """Return OHLCV data"""
    
    signals = [...]  # List of Signal dataclass instances
    config = {...}   # Backtest parameters
    expected = {...} # Execution expectations
    comparison = {...} # Tolerance levels for matching
```

**Benefits**:
- Self-contained: Data, signals, expectations in one file
- Platform-agnostic: No platform-specific imports
- Testable: Can run analyze_data_at_signals() to debug
- Extensible: Can add expected_pnl() method

#### Runner Architecture (runner.py)
1. **ScenarioRunner class**: Executes scenario on multiple platforms
2. **Platform executors**: run_ml4t.backtest(), run_vectorbt(), run_backtrader(), run_zipline()
3. **Trade extraction**: Platform-specific extractors convert native trade format
4. **Trade matching**: match_trades() groups trades by entry timestamp
5. **Reporting**: generate_summary_report() and generate_trade_report()

**Current CLI**:
```bash
python runner.py --scenario 001 --platforms ml4t.backtest,vectorbt --report detailed
```

---

## Part 2: ml4t.backtest Order Type Analysis

### 2.1 Supported Order Types

From `/src/ml4t.backtest/core/types.py` and `/src/ml4t.backtest/execution/order.py`:

| Order Type | Implementation | Status | Key Parameters |
|---|---|---|---|
| **MARKET** | ✅ Full | Shipped | None (fills immediately) |
| **LIMIT** | ✅ Full | Shipped | `limit_price` |
| **STOP** | ✅ Full | Shipped | `stop_price` |
| **STOP_LIMIT** | ✅ Full | Shipped | `stop_price`, `limit_price` |
| **TRAILING_STOP** | ✅ Full | Shipped | `trail_amount` OR `trail_percent` |
| **BRACKET** | ✅ Full | Shipped | Absolute: `profit_target`, `stop_loss`<br/>Percentage: `tp_pct`, `sl_pct`, `tsl_pct` |
| **OCO** (One-Cancels-All) | ⚠️ Defined | Partial | Listed in enum but needs validation |

### 2.2 Execution Timing Configuration

From `/src/ml4t.backtest/execution/order.py` (can_fill method):

**Intrabar Execution Logic**:
```python
use_intrabar = high is not None and low is not None

# LIMIT orders check if bar HIGH/LOW reached limit
if order_type == LIMIT:
    if is_buy:
        return low <= limit_price  # Check if we could buy
    else:
        return high >= limit_price  # Check if we could sell

# STOP orders check if bar HIGH/LOW reached stop
if order_type == STOP:
    if is_buy:
        return high >= stop_price   # Short cover
    else:
        return low <= stop_price    # Long stop-loss
```

**Time-In-Force Support** (from types.py):
- DAY: Valid for the day
- GTC: Good till canceled
- IOC: Immediate or cancel
- FOK: Fill or kill
- GTD: Good till date
- MOC: Market on close
- MOO: Market on open

**Current Execution Model**: Event-driven, fills on bar events with OHLC range checks.

### 2.3 Known Issues from CLAUDE.md

**Position Sync Issue (Fixed)**:
- ❌ Dual tracking: `broker.position_tracker` vs `broker.portfolio.positions`
- ✅ **Solution**: Use `broker.get_position()` method

**Implications for Testing**:
- Must validate that position tracking is correct after fills
- Bracket orders need special handling (entry + simultaneous exits)

---

## Part 3: Competitor Execution Models

### 3.1 VectorBT Execution Timing

**Key Findings**:
- **Default**: Same-bar execution possible but requires careful signal design
- **Timing Assumption**: Events occur at unknown time between open and close
- **Stop Order Priority**: Stop signals assumed to come before user signals
- **Prevention**: VectorBT assumes conservative timing to prevent look-ahead bias
- **Intrabar Execution**: Can execute on OHLC if price reached during bar

**For Testing**: Need scenario validating same-bar vs next-bar execution difference.

### 3.2 Backtrader Execution Timing

**Key Findings**:
- **Default**: Next-bar execution with opening price
- **Reason**: Closed bars are already known; execution must wait for next price
- **Cheat-on-Close**: Optional mode for same-bar execution at close price
- **Typical Use**: Market orders use next bar open, limit/stop use matched OHLC
- **Data Feed Dependency**: Behavior varies with feed (daily vs intraday)

**For Testing**: Need scenario testing next-bar open execution with limit/stop orders.

### 3.3 Zipline Execution Timing

**Key Findings**:
- **Default**: Next-bar execution, NOT immediate fill
- **Volume-Based**: Can take multiple bars to fill large orders (max 2.5% volume)
- **Realistic Simulation**: Slippage and order book impact modeled
- **instant_fill Parameter**: Can force immediate fill if configured
- **Order Cancellation**: EOD cancellation (prevents multi-day carries)

**For Testing**: Need scenario testing volume-limited fills and partial fills.

### 3.4 Summary: Execution Timing Comparison

| Framework | Market Orders | Limit Orders | Execution Timing | Multi-Bar Fills |
|---|---|---|---|---|
| **VectorBT** | Same-bar possible | Same-bar if touched | OHLC range check | No (single bar) |
| **Backtrader** | Next-bar open | Next-bar matched | Event-driven | Yes |
| **Zipline** | Next-bar open | Volume-limited | Event-driven | Yes (default) |
| **ml4t.backtest** | Flexible | OHLC range check | Event-driven | Can support |

---

## Part 4: Existing Validation Scenario Analysis

### 4.1 Scenario 001: Simple Market Orders

**Specification** (scenario_001_simple_market_orders.py):

```python
signals = [
    Signal(2020-02-03, AAPL, BUY, 100),      # Expected: next bar open
    Signal(2020-04-15, AAPL, SELL, 100),     # Trade 1 complete
    Signal(2020-07-20, AAPL, BUY, 100),      # Trade 2
    Signal(2020-12-15, AAPL, SELL, 100),     # Trade 2 complete
]

config = {
    'initial_capital': 100,000,
    'commission': 0.1%,  # Per trade (entry + exit)
    'slippage': 0.0,
}

expected = {
    'trade_count': 2,
    'execution_timing': 'next_bar',
    'price_used': 'open',
    'final_position': 0,  # Flat
}
```

**Data**: Synthetic 252-day daily data, price trending $70→$120.

**Purpose**: Baseline validation that all platforms:
1. Execute all 4 signals
2. Match trades correctly (BUY→SELL)
3. Calculate P&L consistently
4. Handle commissions

**Test Coverage**:
- ✅ Basic execution
- ✅ Trade counting
- ✅ Commission calculation
- ❌ Repeated signals (re-entry with existing position)
- ❌ Limit/stop orders
- ❌ Same-bar execution
- ❌ Multi-asset trading
- ❌ Partial fills

---

## Part 5: Test Scenario Roadmap

### 5.1 Scenario Progression Framework

**Complexity Levels**:
- **Basic** (001-005): Single asset, market orders, no stops
- **Intermediate** (006-010): Limit/stop orders, multi-asset
- **Advanced** (011-015): Bracket orders, OCA, trailing stops
- **Complex** (016-020): Volume-limited fills, re-entry patterns
- **Stress** (021+): Large portfolios, margin, corporate actions

### 5.2 Detailed Scenario Roadmap

#### **BASIC TIER (Scenarios 001-005)**

**Scenario 001: Simple Market Orders** [✅ EXISTS]
- 2 trades, 4 signals, next-bar execution
- Validates basic execution and P&L

**Scenario 002: Market Orders - Same-Bar Execution**
- Test alternate execution timing (if ml4t.backtest supports)
- Same 4 signals, expect same-bar close execution
- **Key Insight**: Compare with Backtrader's cheat-on-close
- **Complexity**: Low (2 trades)
- **Platforms**: ml4t.backtest (validate support), VectorBT (baseline)

**Scenario 003: Market Orders - No Position Changes**
- 4 signals but all in same direction (4 BUYs in sequence)
- Validates position accumulation and no closing
- **Expected**: 0 closed trades, 1 open position
- **Complexity**: Low (position tracking)
- **Platforms**: All

**Scenario 004: Market Orders - Multiple Assets (2 stocks)**
- AAPL: 2 trades, GOOGL: 2 trades (4 total trades)
- Interleaved signals: AAPL→GOOGL→AAPL→GOOGL
- **Key Test**: Portfolio-level tracking across assets
- **Complexity**: Low (asset isolation)
- **Platforms**: All

**Scenario 005: Market Orders - High-Frequency Signals**
- 10 signals in 10 consecutive bars (every day)
- Rapid entry/exit cycles
- **Key Test**: Order queuing and fill timing
- **Complexity**: Medium (rapid execution)
- **Platforms**: All

---

#### **INTERMEDIATE TIER (Scenarios 006-010)**

**Scenario 006: Limit Orders - Entry Only**
- 2 BUY LIMIT signals at different prices
- Test limit order triggering and partial fills
- **Expected**: Both trigger, match entry prices
- **Complexity**: Medium (limit execution logic)
- **Key Variables**:
  - Scenario 006a: Limits that trigger in same bar
  - Scenario 006b: Limits that trigger in next bar
  - Scenario 006c: Limits that never trigger (cancel)

**Scenario 007: Stop Orders - Exit Only**
- 1 BUY signal (market), then SELL STOP at lower price
- Validates stop-loss functionality
- **Expected**: Stop triggers if price dips below level
- **Complexity**: Medium (stop triggering)
- **Platforms**: All (critical for risk management)

**Scenario 008: Stop-Limit Orders**
- Entry: BUY STOP-LIMIT at entry level
- Exit: SELL STOP-LIMIT for risk management
- **Key Test**: Both components (stop + limit) interact correctly
- **Complexity**: High (dual-trigger logic)
- **Platforms**: All

**Scenario 009: Trailing Stop - Performance**
- 1 BUY signal, then trailing stop at 5% below peak
- Bar sequence: Up 10%, down 3%, down 2%, down 5% (triggers)
- **Expected**: Stop triggers on 4th bar
- **Complexity**: High (peak tracking)
- **Platforms**: ml4t.backtest, VectorBT (Backtrader/Zipline may differ)

**Scenario 010: Mixed Order Types - Single Trade**
- Entry: LIMIT at specific price
- Exits: Take profit LIMIT + Stop loss STOP (OCO behavior)
- **Expected**: Whichever fills first closes position
- **Complexity**: High (multiple exit conditions)
- **Platforms**: ml4t.backtest (test bracket manager)

---

#### **ADVANCED TIER (Scenarios 011-015)**

**Scenario 011: Bracket Orders - Basic**
- 1 BUY MARKET with TP at +5% and SL at -2%
- Bar sequence: Up 6% (TP triggers), exit at TP price
- **Expected**: Position closed at profit target
- **Complexity**: High (bracket logic)
- **Platforms**: All

**Scenario 012: Bracket Orders - SL Triggers**
- 1 BUY MARKET with TP at +5% and SL at -3%
- Bar sequence: Down 4% (SL triggers), exit at stop loss
- **Expected**: Position closed at loss
- **Complexity**: High (bracket exit logic)

**Scenario 013: Bracket Orders - Percentage-Based**
- Test `tp_pct`, `sl_pct`, `tsl_pct` parameters
- Validate VectorBT compatibility (Pro feature)
- **Complexity**: High (parameter handling)
- **Platforms**: ml4t.backtest, VectorBT Pro

**Scenario 014: Multiple Bracket Orders**
- 3 entries with different bracket parameters
- Each with unique TP/SL levels
- **Expected**: Correct exit prices for each
- **Complexity**: High (multi-bracket tracking)

**Scenario 015: Conditional Orders**
- IF/THEN logic: "Exit AAPL if GOOGL hits level X"
- Tests cross-asset dependencies
- **Complexity**: Very High
- **Platforms**: ml4t.backtest only (custom implementation)

---

#### **COMPLEX TIER (Scenarios 016-020)**

**Scenario 016: Re-Entry While Position Open**
- 1 BUY, then another BUY signal arrives before SELL
- How do platforms handle doubling up?
- **Expected**:
  - ml4t.backtest: Position increases (configurable)
  - VectorBT: May re-entry differently
  - Backtrader: Depends on strategy logic
- **Complexity**: High (position accumulation)

**Scenario 017: Partial Fills - Volume Limited**
- 1 BUY for 10,000 shares but volume only 5,000
- Test multi-bar fill logic (Zipline, Backtrader)
- **Expected**: Some platforms fill over 2+ bars
- **Complexity**: High (volume tracking)
- **Platforms**: Zipline (native), others (manual simulation)

**Scenario 018: Order Cancellation**
- BUY LIMIT order issued but price never reaches level
- Order expires or gets canceled programmatically
- **Expected**: Order status = CANCELED
- **Complexity**: Medium (order lifecycle)

**Scenario 019: Time-In-Force (GTC vs DAY)**
- 1 BUY on day 1, SELL signal on day 5 but different TIF
- DAY order should cancel EOD, GTC should persist
- **Complexity**: High (time-based expiration)

**Scenario 020: Slippage Modeling**
- Same trades with varying slippage models:
  - Fixed: $0.10 per trade
  - Percentage: 0.05% of price
  - Volume-weighted: Depends on liquidity
- **Expected**: Consistent P&L impact across platforms
- **Complexity**: Medium (model comparison)

---

#### **STRESS/EDGE CASE TIER (Scenarios 021+)**

**Scenario 021: Liquidity Crisis**
- Market gaps: low > previous close (gap up)
- Stop orders can't fill at stop price
- **Expected**: Fill at available price, slippage recorded

**Scenario 022: Margin/Short Selling**
- 1 SHORT signal on AAPL
- Validate negative position tracking
- Short stops are reversed (HIGH/LOW checks inverted)

**Scenario 023: Corporate Actions**
- Stock split, dividend, reverse split
- Position quantities and prices adjusted
- **Note**: ml4t.backtest has corporate_actions module

**Scenario 024: Multi-Timeframe**
- Signals on different timeframes (daily + 4H)
- Cross-timeframe execution coordination
- **Complexity**: Very High

**Scenario 025: Large Portfolio (100+ assets)**
- Performance and correctness at scale
- Tests vectorization efficiency
- **Platforms**: VectorBT (strength), others (stress test)

---

### 5.3 Scenario Dependency Graph

```
001 (Simple Market)
├── 002 (Same-Bar) ← Tests timing variant
├── 003 (Position Accumulation) ← Tests tracking
├── 004 (Multi-Asset) ← Tests isolation
├── 005 (High-Frequency) ← Tests queuing
│
├─→ 006 (Limit Orders)
│   ├── 008 (Stop-Limit)
│   └── 010 (Mixed Orders)
│
├─→ 007 (Stop Orders)
│   ├── 009 (Trailing Stop)
│   └── 010 (Mixed Orders)
│
├─→ 011-014 (Bracket Orders) ← Depends on stops + limits
│
└─→ 016-020 (Complex Scenarios) ← Depends on basics
    └─→ 021+ (Stress Tests)
```

---

## Part 6: Testing Architecture Proposal

### 6.1 Fixture Framework

```python
# fixtures/conftest.py

class BaseScenarioFixture:
    """Reusable components for scenario testing."""
    
    @staticmethod
    def create_synthetic_data(
        days: int = 252,
        start_price: float = 100.0,
        volatility: float = 0.02,
        trend: float = 0.0,
        assets: list[str] = None,
    ) -> pl.DataFrame:
        """Generate synthetic OHLCV data with configurable properties."""
        # Generates realistic price paths with given volatility/trend
    
    @staticmethod
    def create_signal_series(
        timestamps: list[datetime],
        assets: list[str],
        actions: list[str],  # 'BUY', 'SELL'
        quantities: list[float],
        order_types: list[str] = None,  # 'MARKET', 'LIMIT', etc.
    ) -> list[Signal]:
        """Create standardized signal specifications."""
    
    @staticmethod
    def run_scenario_on_platform(
        scenario_class,
        platform: str,
        executor_kwargs: dict = None,
    ) -> PlatformResult:
        """Run scenario on specified platform with configurable options."""
    
    @staticmethod
    def extract_and_match_trades(
        results: dict[str, PlatformResult],
        tolerance: dict[str, float] = None,
    ) -> list[TradeMatch]:
        """Extract trades and match across platforms."""
```

### 6.2 Scenario Template

```python
# scenarios/scenario_NNN_description.py

class ScenarioNNN:
    """Clear title and purpose."""
    
    name = "NNN_short_identifier"
    description = "Detailed description for test reports"
    category = "basic|intermediate|advanced|complex|stress"
    
    # Data configuration
    data_config = {
        'days': 252,
        'start_price': 100.0,
        'volatility': 0.02,
        'trend': 0.0,
        'assets': ['AAPL'],
    }
    
    @staticmethod
    def get_data() -> pl.DataFrame:
        """Return OHLCV data."""
    
    # Signal specification
    @staticmethod
    def get_signals() -> list[Signal]:
        """Return trading signals."""
    
    # Backtest configuration
    config = {
        'initial_capital': 100_000.0,
        'commission': 0.001,
        'slippage': 0.0,
        'execution_timing': 'next_bar',  # or 'same_bar'
    }
    
    # Expected results for validation
    expected = {
        'trade_count': 2,
        'final_position': 0,
        'gross_pnl_range': (1000, 2000),  # Allow for slippage variations
    }
    
    # Platform-specific behavior notes
    platform_notes = {
        'vectorbt': 'Same-bar execution requires specific signal design',
        'backtrader': 'Uses next-bar open for market orders',
        'zipline': 'May take multiple bars for large volumes',
    }
    
    # Analysis helpers
    @staticmethod
    def analyze_execution_points():
        """Print OHLC at each signal timestamp for debugging."""
    
    @staticmethod
    def validate_results(trades: list[StandardTrade]) -> dict[str, bool]:
        """Custom validation beyond standard comparisons."""
```

### 6.3 Configuration Management

```python
# config/execution_models.py

class ExecutionModel:
    """Encapsulates execution timing configuration."""
    
    name: str
    timing: str  # 'same_bar', 'next_bar_open', 'next_bar_close'
    fill_logic: str  # 'end_of_bar', 'intrabar_range', 'volume_limited'
    partial_fills_allowed: bool
    multi_bar_fills: bool
    
    def __init__(self, name, timing, fill_logic, **kwargs):
        # Initialize execution model

# Predefined models matching each framework
EXECUTION_MODELS = {
    'vectorbt_same_bar': ExecutionModel(...),
    'backtrader_next_bar': ExecutionModel(...),
    'zipline_volume_limited': ExecutionModel(...),
    'ml4t.backtest_flexible': ExecutionModel(...),
}
```

### 6.4 Validation Assertion Helpers

```python
# validators/assertions.py

def assert_trade_count(trades: list[StandardTrade], expected: int, tolerance: int = 0):
    """Assert exact or approximate trade count."""

def assert_prices_match(
    platform1_trades: list[StandardTrade],
    platform2_trades: list[StandardTrade],
    tolerance_pct: float = 0.1,
):
    """Assert entry/exit prices match within tolerance."""

def assert_execution_timing(
    trades: list[StandardTrade],
    expected: str,  # 'same_bar', 'next_bar'
    data: pl.DataFrame,
):
    """Validate that execution timing matches expectation."""

def assert_position_tracking(
    trades: list[StandardTrade],
    expected_final: float = 0,
):
    """Validate final position and intermediate tracking."""

def assert_no_lookahead(
    trades: list[StandardTrade],
    signals: list[Signal],
    data: pl.DataFrame,
):
    """Validate no look-ahead bias (prices at execution are within OHLC)."""
```

### 6.5 Integration with Runner

```python
# runner.py (enhanced)

class EnhancedScenarioRunner:
    """Support incremental scenario execution."""
    
    def run_scenario_range(self, start: int, end: int, platforms: list[str]):
        """Run scenarios 001-N in sequence."""
        
    def run_scenario_category(self, category: str, platforms: list[str]):
        """Run all 'basic' or 'intermediate' scenarios."""
    
    def generate_coverage_report(self):
        """Show which order types/features covered by which scenarios."""
        
    def run_with_custom_execution_model(
        self, scenario: int, model: ExecutionModel
    ):
        """Test scenario with custom execution timing."""
```

---

## Part 7: Immediate Implementation Plan

### Phase 1: Foundation (Week 1)

**Priority 1a: Infrastructure**
1. Create `fixtures/conftest.py` with BaseScenarioFixture class
2. Create `config/execution_models.py` with predefined models
3. Create `validators/assertions.py` with assertion helpers
4. Enhance runner.py to support scenario ranges and categories

**Priority 1b: Basic Scenarios (001-005)**
1. ✅ Scenario 001 already exists - validate it runs
2. Create Scenario 002 (same-bar execution) - test ml4t.backtest capability
3. Create Scenario 003 (position accumulation) - test no-exit case
4. Create Scenario 004 (multi-asset) - test asset isolation
5. Create Scenario 005 (high-frequency) - test queuing

**Estimated Effort**: 3-4 days
**Deliverables**: 5 passing scenarios, updated runner

### Phase 2: Order Types (Week 2-3)

**Priority 2a: Limit Orders (Scenarios 006-008)**
1. Scenario 006: Limit order entry triggering
2. Scenario 007: Stop-loss order functionality
3. Scenario 008: Stop-limit order (dual trigger)

**Priority 2b: Advanced Stops (Scenarios 009-010)**
1. Scenario 009: Trailing stop functionality
2. Scenario 010: Mixed order types (LIMIT + STOP)

**Estimated Effort**: 5-7 days
**Deliverables**: 5 new scenarios, comprehensive stop/limit testing

### Phase 3: Bracket Orders (Week 4)

**Priority 3a: Bracket Implementation (Scenarios 011-014)**
1. Scenario 011: Basic bracket (TP triggers)
2. Scenario 012: Bracket (SL triggers)
3. Scenario 013: Percentage-based brackets
4. Scenario 014: Multiple brackets

**Estimated Effort**: 3-4 days
**Deliverables**: 4 scenarios validating bracket manager

### Phase 4: Complex Scenarios (Week 5-6)

**Priority 4a: Position Dynamics (Scenarios 016-017)**
1. Scenario 016: Re-entry with open position
2. Scenario 017: Partial/multi-bar fills

**Priority 4b: Order Lifecycle (Scenarios 018-020)**
1. Scenario 018: Order cancellation
2. Scenario 019: Time-in-force (GTC vs DAY)
3. Scenario 020: Slippage modeling variants

**Estimated Effort**: 4-5 days
**Deliverables**: 5 scenarios, edge case coverage

---

## Part 8: Feature Coverage Matrix

| Feature | Unit Tests | Scenario Tests | Status |
|---|---|---|---|
| **MARKET ORDERS** | ✅ (34 tests) | 001-005 | Ready |
| **LIMIT ORDERS** | ✅ | 006a, 006b, 006c, 008 | To Build |
| **STOP ORDERS** | ✅ | 007, 009 | To Build |
| **TRAILING STOPS** | ✅ (test_trailing_stop_bug) | 009 | To Build |
| **BRACKET ORDERS** | ✅ (test_bracket_*) | 011-014 | To Build |
| **OCO (One-Cancels-All)** | ⚠️ (listed but partial) | 010, 015 | Needs Validation |
| **POSITION TRACKING** | ✅ | 003-005, 016 | Ready |
| **MULTI-ASSET** | ✅ | 004 | To Build |
| **SLIPPAGE** | ✅ | 020 | To Build |
| **COMMISSION** | ✅ | 001-005 | Ready |
| **PnL CALCULATION** | ✅ | 001-005 | Ready |
| **CASH CONSTRAINTS** | ✅ (test_cash_constraints) | 002-004 | Ready |
| **MARGIN** | ✅ (test_margin) | 022 | To Build |
| **TIME-IN-FORCE** | ✅ (enum defined) | 019 | To Build |
| **CORPORATE ACTIONS** | ✅ (module exists) | 023 | To Build |

---

## Part 9: Required ml4t.backtest Enhancements

### 9.1 Current Gaps

**Minor**:
1. OCO order validation - currently listed but needs testing
2. Documentation of execution_delay parameter semantics
3. Clear API for querying execution timing configuration

**Medium**:
1. Consider adding `execution_timing` configuration (same_bar vs next_bar)
2. Documentation of intrabar fill logic (currently in can_fill method)
3. Enhanced logging for trade execution for debugging

**Major**:
1. ✅ All major features already implemented
2. No breaking changes needed

### 9.2 Recommended Enhancements

**Convenience Methods**:
```python
# In SimulationBroker or BacktestEngine
def set_execution_model(model: ExecutionModel):
    """Configure execution timing behavior."""
    
def get_execution_config() -> dict:
    """Return current execution configuration."""
```

**Documentation**:
1. Add execution timing guide to docs/
2. Document bracket order behavior with examples
3. Add "Order Types Reference" to docs/

**Testing**:
1. Add comprehensive docstring tests to order.py
2. Create reference implementation for each order type

---

## Part 10: Cross-Platform Validation Strategy

### 10.1 VectorBT Baseline

VectorBT should be primary reference for:
- Same-bar execution behavior
- Intrabar limit/stop order triggering
- Bracket order P&L calculation

**Current Status**: scenario_001 runs but needs detailed result comparison.

### 10.2 Backtrader as Reality Check

Backtrader should validate:
- Next-bar open execution (most common real trading)
- Order queuing and cancellation
- Position sizing and cash constraints

### 10.3 Zipline for Realism

Zipline should validate:
- Multi-bar partial fills
- Volume-based execution
- Realistic slippage modeling

### 10.4 ml4t.backtest Alignment

ml4t.backtest should match:
- All same-bar scenarios where VectorBT matches
- All next-bar scenarios where Backtrader matches
- More flexible than Zipline (instant fills, no volume limits by default)

---

## Part 11: Success Metrics

### Quantitative
- **Coverage**: 20+ scenarios covering 15+ order types/features
- **Pass Rate**: 100% of basic scenarios (001-005) pass on all platforms
- **Match Rate**: 95%+ trade price matching across platforms
- **Documentation**: Full docstrings, examples, execution timing guides

### Qualitative
- **Clarity**: Any developer can understand scenario specification
- **Extensibility**: Easy to add new scenarios following template
- **Debuggability**: analyze_data_at_signals() helps identify discrepancies
- **Reliability**: No flaky tests, deterministic results

### Timeline
- **Weeks 1-2**: Phases 1-2 (10 scenarios, 2-3 order types)
- **Weeks 3-4**: Phase 3 (bracket orders, 4 scenarios)
- **Weeks 5-6**: Phase 4 (complex scenarios, 5+ scenarios)
- **Total**: 20 scenarios in 6 weeks (1 month intensive)

---

## Part 12: Testing Checklist

### Before Running Scenarios

- [ ] All 34 unit tests pass
- [ ] VectorBT environment configured (if testing cross-platform)
- [ ] Backtrader environment configured
- [ ] Zipline environment configured (optional first phase)
- [ ] Data fixtures available (synthetic or real)

### For Each Scenario

- [ ] Scenario class defined with all required methods
- [ ] Data generated or loaded successfully
- [ ] Signals list is complete and correct
- [ ] Expected results documented
- [ ] Runner executes on ml4t.backtest (primary test)
- [ ] Trade extraction works
- [ ] Trades pass basic validation (no look-ahead)

### Before Committing

- [ ] All scenarios 001-N pass independently
- [ ] No regressions in unit tests
- [ ] README updated with scenario list
- [ ] Scenario docstrings complete
- [ ] Edge cases documented in scenario

---

## Conclusion

ml4t.backtest has solid testing foundations. The next phase should focus on:

1. **Building** 20+ incremental validation scenarios
2. **Leveraging** the existing StandardTrade infrastructure
3. **Validating** all order types systematically
4. **Documenting** execution timing clearly
5. **Establishing** cross-platform alignment

The proposed architecture is:
- **Extensible**: Templates and fixtures make new scenarios easy
- **Debuggable**: analyze_data_at_signals() helps diagnose issues
- **Maintainable**: Clear separation of concerns
- **Comprehensive**: Covers basic to stress-test scenarios

With 6 weeks of focused development, ml4t.backtest can have a production-grade test suite that validates institutional-quality execution fidelity.

