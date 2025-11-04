# Contributing to QEngine Validation Framework

Guide for adding new validation scenarios to the framework.

## Table of Contents

1. [Overview](#overview)
2. [Scenario Development Pattern](#scenario-development-pattern)
3. [TDD Workflow](#tdd-workflow)
4. [Scenario File Structure](#scenario-file-structure)
5. [Test File Structure](#test-file-structure)
6. [Signal Design Guidelines](#signal-design-guidelines)
7. [Platform Considerations](#platform-considerations)
8. [Testing & Validation](#testing--validation)
9. [Documentation Requirements](#documentation-requirements)
10. [Examples](#examples)

## Overview

The QEngine validation framework uses a **Test-Driven Development (TDD)** approach with the RED-GREEN-REFACTOR cycle. This proven methodology delivers:

- **71% time savings** vs estimates (scenarios complete in ~1h vs 3.5h)
- **100% test coverage** for all scenarios
- **Consistent quality** across all platforms

## Scenario Development Pattern

### High-Level Steps

1. **Analyze Market Data**: Choose signal dates from real AAPL/MSFT 2017 data
2. **Design Scenario**: Define what behavior you want to test
3. **RED**: Write failing test first
4. **GREEN**: Create scenario file with signals
5. **GREEN**: Verify all 4 platforms execute
6. **REFACTOR**: Defer until 3+ similar scenarios exist (rule of three)

### Time Expectations

- **Scenario Creation**: 30-40 minutes (design signals, write scenario file)
- **Test Creation**: 10-15 minutes (follow established pattern)
- **Platform Verification**: 5-10 minutes (run tests, verify execution)
- **Total**: ~1 hour per scenario

## TDD Workflow

### Phase 1: RED (Write Failing Test)

Create `test_scenario_00X_description.py` with 3 test functions:

```python
import pytest
from runner import run_backtest
from scenarios.scenario_00X_description import get_scenario

def test_all_platforms_scenario_00X():
    """Test all 4 platforms execute scenario 00X"""
    scenario = get_scenario()

    # Test each platform
    for platform in ['qengine', 'vectorbt', 'backtrader', 'zipline']:
        results = run_backtest(
            scenario=scenario,
            platform=platform,
            verbose=False
        )
        assert results is not None, f"{platform} failed to execute"
        assert len(results.trades) > 0, f"{platform} extracted zero trades"

def test_specific_validation():
    """Verify scenario-specific behavior"""
    # Test what makes THIS scenario unique
    # Examples:
    # - Limit prices respected
    # - Stop losses triggered
    # - Positions accumulated correctly
    # - Multi-asset isolation
    pass

def test_execution_timing():
    """Document platform execution timing differences"""
    # Run all platforms
    # Show expected timing variations
    # Validate no critical differences
    pass
```

### Phase 2: GREEN (Create Scenario)

Create `scenarios/scenario_00X_description.py`:

```python
"""
Scenario 00X: Description

Purpose: What this scenario tests
Assets: AAPL, MSFT, etc.
Signals: N signals creating M complete trades
Period: 2017-01-01 to 2017-12-31

Expected Behavior:
- Describe what should happen
- Document platform differences
- Note any special considerations
"""
import pytz
from datetime import datetime
from typing import NamedTuple, List

class Signal(NamedTuple):
    """Trading signal"""
    timestamp: datetime
    ticker: str
    action: str  # 'BUY' or 'SELL'
    quantity: int
    order_type: str = 'MARKET'  # or 'LIMIT'
    limit_price: float | None = None

class Scenario00XDescription:
    """Scenario 00X: Description"""

    # Scenario metadata
    name = "Scenario 00X: Description"
    description = "Brief description of what this tests"
    tickers = ['AAPL']  # or ['AAPL', 'MSFT'] for multi-asset
    start_date = '2017-01-01'
    end_date = '2017-12-31'

    # Trading signals (ALWAYS UTC-aware!)
    signals = [
        Signal(
            timestamp=datetime(2017, 3, 1, tzinfo=pytz.UTC),
            ticker='AAPL',
            action='BUY',
            quantity=100,
            order_type='MARKET'
        ),
        Signal(
            timestamp=datetime(2017, 3, 15, tzinfo=pytz.UTC),
            ticker='AAPL',
            action='SELL',
            quantity=100,
            order_type='MARKET'
        ),
        # Add more signals...
    ]

    # Configuration
    initial_capital = 100_000.0
    commission = 0.001  # 0.1% per trade

def get_scenario():
    """Return scenario instance for runner"""
    return Scenario00XDescription()
```

### Phase 3: GREEN (Verify Platforms)

```bash
# Run test
cd /home/stefan/ml4t/software/backtest/tests/validation
uv run python -m pytest test_scenario_00X_description.py -v

# Run with all platforms
uv run python runner.py --scenario 00X --platforms qengine,vectorbt,backtrader,zipline --report summary

# Expected: All 4 platforms execute successfully
```

### Phase 4: REFACTOR (Defer Until Rule of Three)

- **Don't refactor immediately**
- Wait until 3+ scenarios share common patterns
- Then extract shared logic into helpers
- Current scenarios already follow clean patterns

## Scenario File Structure

### Required Components

1. **Module Docstring**: Purpose, assets, signals, expected behavior
2. **Signal Class**: NamedTuple defining signal structure
3. **Scenario Class**: Contains signals, config, metadata
4. **get_scenario() Function**: Returns scenario instance

### Naming Conventions

- **File**: `scenario_00X_description.py` (3-digit number, lowercase with underscores)
- **Class**: `Scenario00XDescription` (PascalCase)
- **Test File**: `test_scenario_00X_description.py` (matches scenario filename)

### Example Naming

| Scenario | File | Class | Test File |
|----------|------|-------|-----------|
| 001 | `scenario_001_simple_market_orders.py` | `Scenario001SimpleMarketOrders` | `test_scenario_001_simple_market_orders.py` |
| 002 | `scenario_002_limit_orders.py` | `Scenario002LimitOrders` | `test_scenario_002_limit_orders.py` |
| 006 | `scenario_006_margin_trading.py` | `Scenario006MarginTrading` | `test_scenario_006_margin_trading.py` |

## Test File Structure

### Required Test Functions

Every test file must have **3 test functions**:

1. **`test_all_platforms_scenario_00X()`**: Integration test verifying all 4 platforms execute
2. **`test_specific_validation()`**: Scenario-specific behavior validation
3. **`test_execution_timing()`**: Document and validate platform timing differences

### Test Pattern (Copy This!)

```python
"""Tests for Scenario 00X: Description"""
import pytest
from runner import run_backtest
from comparison.matcher import match_trades
from scenarios.scenario_00X_description import get_scenario

def test_all_platforms_scenario_00X():
    """Test all 4 platforms execute scenario 00X"""
    scenario = get_scenario()
    platforms = ['qengine', 'vectorbt', 'backtrader', 'zipline']
    all_trades = {}

    # Run each platform
    for platform in platforms:
        results = run_backtest(
            scenario=scenario,
            platform=platform,
            verbose=False
        )
        assert results is not None, f"{platform} failed to execute"
        assert len(results.trades) > 0, f"{platform} extracted zero trades"
        all_trades[platform] = results.trades
        print(f"\n{platform}: {len(results.trades)} trades extracted")

    # Match trades across platforms
    matched_groups = match_trades(all_trades)
    print(f"\n{len(matched_groups)} trade groups matched")

    # Validate no critical differences
    for group in matched_groups:
        assert group.severity != 'critical', f"Critical difference: {group}"

def test_specific_validation():
    """Verify [SPECIFIC BEHAVIOR]"""
    scenario = get_scenario()

    # Run qengine (or whichever platform is most appropriate)
    results = run_backtest(
        scenario=scenario,
        platform='qengine',
        verbose=False
    )

    # Validate specific behavior
    # Examples:
    # - Check limit prices: assert all(t.entry_price <= t.limit_price for t in trades)
    # - Verify positions: assert max_position == 200
    # - Check asset isolation: assert len(set(t.symbol for t in trades)) == 2
    pass

def test_execution_timing():
    """Document platform execution timing differences"""
    scenario = get_scenario()
    platforms = ['qengine', 'vectorbt', 'backtrader', 'zipline']

    for platform in platforms:
        results = run_backtest(
            scenario=scenario,
            platform=platform,
            verbose=False
        )
        print(f"\n{platform}:")
        print(f"  Trades: {len(results.trades)}")
        print(f"  Execution model: [same-bar close | next-bar close | next-bar open]")
        # Document expected timing for this scenario
```

## Signal Design Guidelines

### Critical Rules

1. **ALWAYS use UTC-aware timestamps**: `datetime(..., tzinfo=pytz.UTC)`
2. **Verify signal dates exist in market data**: Check AAPL/MSFT 2017 dataset
3. **Use class attributes (not methods)** for signals and config
4. **Complete trades**: Ensure every BUY has a corresponding SELL (or vice versa)

### Signal Timing

```python
# ✅ CORRECT: UTC-aware timestamp
Signal(
    timestamp=datetime(2017, 3, 1, tzinfo=pytz.UTC),
    ticker='AAPL',
    action='BUY',
    quantity=100
)

# ❌ WRONG: Timezone-naive (will cause platform failures)
Signal(
    timestamp=datetime(2017, 3, 1),  # Missing tzinfo!
    ticker='AAPL',
    action='BUY',
    quantity=100
)
```

### Signal Quantity

- **Start simple**: 2-4 complete trades (4-8 signals)
- **Complex scenarios**: Up to 10 signals for advanced testing
- **Multi-asset**: Split signals evenly between assets

### Analyzing Market Data for Signal Dates

```python
# Load AAPL 2017 data
from fixtures.market_data import get_ticker_data
df = get_ticker_data('AAPL', '2017-01-01', '2017-12-31')

# Find good dates with price movements
print(df[['open', 'high', 'low', 'close']].head(20))

# Choose dates with:
# - Sufficient price movement (for limit orders)
# - Volume (for realistic execution)
# - Gaps between signals (avoid same-day re-entry issues)
```

## Platform Considerations

### Execution Models

Different platforms execute at different times:

| Platform | Execution | Signal T → Entry |
|----------|-----------|------------------|
| VectorBT | Same-bar close | T close |
| QEngine | Next-bar close | T+1 close |
| Zipline | Next-bar close | T+1 close |
| Backtrader | Next-bar open | T+1 open |

**Implication**: Your scenario will create different trade groups. **This is expected!**

### Platform Quirks

1. **VectorBT**:
   - Same-bar execution (most optimistic)
   - `stop_loss` parameter exists but not uniformly supported across platforms

2. **QEngine & Zipline**:
   - Next-bar close execution (realistic)
   - Zipline requires timezone-naive start/end dates (but signals must be UTC-aware!)

3. **Backtrader**:
   - Next-bar open execution (most realistic for market orders)
   - Returns timezone-naive datetimes (extractors handle conversion)

### Multi-Asset Support

For multi-asset scenarios:

1. **Data**: Use `get_ticker_data()` for each asset
2. **Signals**: Specify `ticker` field for each signal
3. **Interleaving**: Mix signals between assets to test concurrent positions
4. **Extractors**: Already fixed for multi-asset (DataFrame input with duplicate timestamps)

Example:
```python
signals = [
    Signal(datetime(2017, 3, 1, tzinfo=pytz.UTC), 'AAPL', 'BUY', 100),
    Signal(datetime(2017, 3, 5, tzinfo=pytz.UTC), 'MSFT', 'BUY', 100),
    Signal(datetime(2017, 3, 15, tzinfo=pytz.UTC), 'AAPL', 'SELL', 100),
    Signal(datetime(2017, 3, 20, tzinfo=pytz.UTC), 'MSFT', 'SELL', 100),
]
```

## Testing & Validation

### Running Tests

```bash
# Single scenario test
uv run python -m pytest test_scenario_00X_description.py -v

# All scenario tests
uv run python -m pytest test_scenario_*.py -v

# Run scenario with specific platform
uv run python runner.py --scenario 00X --platforms qengine

# Run with all platforms and detailed report
uv run python runner.py --scenario 00X --platforms qengine,vectorbt,backtrader,zipline --report detailed
```

### Validation Checklist

Before marking scenario complete:

- [ ] All 4 platforms execute successfully (no errors)
- [ ] Each platform extracts > 0 trades
- [ ] Test file has 3 test functions (all passing)
- [ ] Scenario-specific validation passes
- [ ] No critical differences in trade matching
- [ ] Documentation added to scenario docstring
- [ ] Signal dates verified to exist in market data
- [ ] All timestamps are UTC-aware

### Expected Results

- **Trade Groups**: Expect 6-11 trade groups (platform execution differences)
- **Perfect Matches**: Aim for 50%+ perfect matches
- **Minor Differences**: 20-40% minor differences (acceptable)
- **Major Differences**: <20% major differences
- **Critical Differences**: 0% critical differences (must investigate if any)

## Documentation Requirements

### Scenario Docstring

```python
"""
Scenario 00X: Description

Purpose: What this scenario tests
Assets: AAPL, MSFT, etc.
Signals: N signals creating M complete trades
Period: 2017-01-01 to 2017-12-31

Expected Behavior:
- Platform A will execute at T close
- Platform B will execute at T+1 open
- Expect N trades per platform (with X-Y variation due to execution models)

Key Validations:
- [What makes this scenario unique]
- [What behavior is being tested]
- [Any special platform considerations]

Design Decisions:
- [Why certain signal dates were chosen]
- [Any compromises or tradeoffs made]
"""
```

### README Update

After completing scenario, update `README.md`:

```markdown
### Scenario 00X: Description ✅
- **Purpose**: Brief description
- **Signals**: N signals, M assets
- **Results**: X trade groups matched, Y perfect matches
- **Validation**: Key behavior verified
- **File**: `scenarios/scenario_00X_description.py`
```

## Examples

### Example 1: Simple Scenario (Market Orders)

See `scenario_001_simple_market_orders.py` for the baseline pattern.

**Key Features**:
- 4 signals (2 BUY, 2 SELL)
- Market orders only
- Single asset (AAPL)
- ~250 lines of code

### Example 2: Limit Orders

See `scenario_002_limit_orders.py` for limit order pattern.

**Key Features**:
- 4 limit order signals
- Price validation (at-or-better than limit)
- Marketable limits (immediate execution)
- ~400 lines of code

### Example 3: Position Accumulation

See `scenario_004_position_reentry.py` for complex position tracking.

**Key Features**:
- 7 signals testing accumulation
- Multiple entries without exit
- Re-entry after partial exit
- Position size validation
- ~450 lines of code

### Example 4: Multi-Asset

See `scenario_005_multi_asset.py` for multi-asset pattern.

**Key Features**:
- 8 signals across 2 assets (AAPL + MSFT)
- Interleaved signals
- Asset isolation verification
- Concurrent positions
- ~330 lines of code

## Common Issues & Solutions

### Issue: Zero Trades Extracted

**Causes**:
1. Signals are timezone-naive (missing `tzinfo=pytz.UTC`)
2. Signal dates don't exist in market data
3. Platform-specific execution issue

**Solution**:
```python
# Verify signals are UTC-aware
assert all(s.timestamp.tzinfo is not None for s in signals)

# Verify dates exist in data
from fixtures.market_data import get_ticker_data
df = get_ticker_data('AAPL', '2017-01-01', '2017-12-31')
assert all(s.timestamp.date() in df.index.date for s in signals)
```

### Issue: Platform Execution Differences

**Not An Issue**: This is expected! Different platforms have different execution models.

**Validation**: Ensure differences are categorized as "minor" or "major", not "critical".

### Issue: Multi-Asset Extraction Fails

**Causes**:
1. Extractors not handling DataFrame input
2. Duplicate timestamps breaking dict lookups

**Solution**: Already fixed in all 3 extractors (vectorbt, backtrader, zipline). See `scenario_005_multi_asset.py` for reference.

## Best Practices

1. **Start Small**: Begin with simple scenarios, add complexity gradually
2. **Follow Patterns**: Copy from scenario_002 or scenario_004 (proven templates)
3. **Test Early**: Run platforms individually before running all 4 together
4. **Document Decisions**: Explain why you chose specific signal dates/behavior
5. **Validate Behavior**: Focus on testing behavior, not exact trade matching
6. **Defer Refactoring**: Wait for rule of three before extracting shared code

## Getting Help

- **Execution Models**: See `docs/PLATFORM_EXECUTION_MODELS.md` (3,500+ lines)
- **Troubleshooting**: See `docs/TROUBLESHOOTING.md` (1,800+ lines)
- **Quick Reference**: See `docs/QUICK_REFERENCE.md` (350 lines)
- **Examples**: Review existing scenarios in `scenarios/` directory

## Questions?

Contact the QEngine Validation Team or review existing scenarios for patterns.

---

**Last Updated**: 2025-11-04
**Framework Version**: 1.0
**Contributing Guide Version**: 1.0
