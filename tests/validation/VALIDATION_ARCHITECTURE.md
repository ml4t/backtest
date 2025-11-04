# Validation Architecture - Clean Platform Comparison

## Goal

Validate that qengine can replicate backtesting results from VectorBT Pro, Backtrader, and Zipline through controlled, transparent scenarios.

**Focus**: Trading mechanics, NOT strategy logic
- Order execution (market, limit, stop)
- Timing correctness (same-bar vs next-bar)
- Position management
- Fill modeling
- Fee/slippage calculation

## Core Principle: Signals as Input

**All platforms receive identical signals**:
```python
# Signal format (platform-independent)
Signal(
    timestamp=datetime(2024, 1, 15, 10, 30),
    asset='BTC',
    action='BUY',  # or 'SELL', 'CLOSE'
    quantity=100,
    order_type='MARKET',  # or 'LIMIT', 'STOP'
    limit_price=50000.0,  # if LIMIT
    stop_price=49000.0,   # if STOP
    take_profit=52000.0,  # optional
    stop_loss=48000.0,    # optional
    trailing_stop_pct=0.02,  # optional
)
```

**No computation in test scenarios** - just execution of pre-defined signals.

## Test Scenario Structure

```
tests/validation/
├── scenarios/              # Test scenario definitions
│   ├── scenario_001_simple_market_orders.py
│   ├── scenario_002_limit_orders.py
│   ├── scenario_003_stop_orders.py
│   ├── scenario_004_bracket_orders.py
│   ├── scenario_005_same_bar_reentry.py
│   ├── scenario_006_multiple_assets.py
│   ├── scenario_007_minute_data_with_gaps.py
│   └── ...
│
├── fixtures/               # Test data (OHLCV + signals)
│   ├── data/
│   │   ├── single_asset_daily.parquet
│   │   ├── single_asset_minute.parquet
│   │   ├── multi_asset_daily.parquet
│   │   └── minute_with_gaps.parquet
│   └── signals/
│       ├── simple_entries.parquet
│       ├── bracket_orders.parquet
│       └── ...
│
├── platforms/              # Platform adapters
│   ├── qengine_adapter.py
│   ├── vectorbt_adapter.py
│   ├── backtrader_adapter.py
│   └── zipline_adapter.py
│
├── comparison/             # Result comparison
│   ├── trade_validator.py
│   └── metrics_validator.py
│
└── runner.py               # Main test runner
```

## Example Scenario: Simple Market Orders

```python
# scenarios/scenario_001_simple_market_orders.py

class Scenario001_SimpleMarketOrders:
    """
    Test basic market order execution with no exits.

    Purpose: Validate that all platforms:
    - Execute market orders immediately
    - Use correct price (open next bar vs close signal bar)
    - Calculate fees correctly
    - Track positions accurately
    """

    name = "001_simple_market_orders"
    description = "Basic market order execution"

    # Data specification
    data_config = {
        'symbol': 'AAPL',
        'timeframe': 'daily',
        'start_date': '2020-01-01',
        'end_date': '2020-12-31',
        'bars': 252,
    }

    # Signals (hardcoded, not computed)
    signals = [
        Signal(
            timestamp=datetime(2020, 2, 3),
            asset='AAPL',
            action='BUY',
            quantity=100,
            order_type='MARKET',
        ),
        Signal(
            timestamp=datetime(2020, 4, 15),
            asset='AAPL',
            action='SELL',
            quantity=100,
            order_type='MARKET',
        ),
        Signal(
            timestamp=datetime(2020, 7, 20),
            asset='AAPL',
            action='BUY',
            quantity=100,
            order_type='MARKET',
        ),
    ]

    # Backtest configuration
    config = {
        'initial_capital': 100_000,
        'commission': 0.001,  # 0.1%
        'slippage': 0.0,      # None for this test
    }

    # Expected outcomes
    expected = {
        'trade_count': 2,  # 2 complete round trips or 1?
        'execution_timing': 'next_bar',  # or 'same_bar'?
        'price_used': 'open',  # or 'close'?
    }

    # Comparison rules
    comparison = {
        'price_tolerance': 0.01,  # 1% price difference acceptable
        'pnl_tolerance': 1.0,     # $1 PnL difference acceptable
        'timestamp_exact': True,  # Timestamps must match exactly
    }
```

## What Gets Compared

### Trade-Level Comparison
```python
@dataclass
class TradeComparison:
    """Result of comparing a single trade across platforms."""

    # Identifiers
    trade_id: int
    signal_timestamp: datetime

    # Entry comparison
    entry_timestamp_match: bool
    entry_price_diff: float
    entry_price_pct_diff: float

    # Exit comparison (if applicable)
    exit_timestamp_match: bool
    exit_price_diff: float

    # PnL comparison
    pnl_diff: float
    pnl_pct_diff: float

    # Commission comparison
    commission_diff: float

    # Status
    matches: bool  # Within tolerance
    notes: str
```

### Aggregate Comparison
```python
@dataclass
class ScenarioComparison:
    """Result of comparing entire scenario across platforms."""

    scenario_name: str
    platforms: list[str]

    # Trade counts
    trade_counts: dict[str, int]
    trade_count_match: bool

    # Timing analysis
    execution_model: dict[str, str]  # 'same_bar' vs 'next_bar'
    timing_consistent: bool

    # Price analysis
    avg_price_diff: dict[str, float]
    max_price_diff: dict[str, float]

    # PnL analysis
    total_pnl: dict[str, float]
    pnl_diff: dict[str, float]

    # Summary
    all_match: bool
    differences_explained: bool
    explanation: str
```

## Test Progression

### Phase 1: Single Asset, Daily Data
1. ✅ Scenario 001: Simple market orders (BUY → SELL)
2. ✅ Scenario 002: Limit orders
3. ✅ Scenario 003: Stop orders
4. ✅ Scenario 004: Market orders with TP/SL brackets
5. ✅ Scenario 005: Trailing stop loss

### Phase 2: Execution Edge Cases
6. ✅ Same-bar re-entry after exit
7. ✅ Multiple orders same bar
8. ✅ Order cancellation
9. ✅ Partial fills
10. ✅ Price gaps (stop triggered beyond stop price)

### Phase 3: Multi-Asset
11. ✅ Two assets, independent signals
12. ✅ Two assets, simultaneous signals
13. ✅ Portfolio-level position limits

### Phase 4: Intraday Data
14. ✅ Minute data, continuous trading
15. ✅ Minute data with daily maintenance gaps (crypto)
16. ✅ Minute data with market hours gaps (equities)

### Phase 5: Advanced Order Types
17. ✅ OCO orders (One-Cancels-Other)
18. ✅ Conditional orders
19. ✅ Scaled entries/exits

## Platform Configuration Matrix

Test each scenario with different platform configurations:

| Config | VectorBT | Backtrader | Zipline | qengine |
|--------|----------|------------|---------|---------|
| Exec timing | same_bar | next_bar | next_bar | configurable |
| Price used | close | open | open | configurable |
| Slippage model | fixed_pct | fixed_pct | volume-based | configurable |
| Partial fills | no | yes | yes | yes |

**Goal**: Document configuration that makes qengine match each platform.

## Output Format

### HTML Report
```
Scenario 001: Simple Market Orders
=====================================

Configuration:
- Asset: AAPL
- Period: 2020-01-01 to 2020-12-31
- Signals: 3 market orders
- Commission: 0.1%

Results:
┌────────────┬────────┬───────────────┬──────────┬────────────┐
│ Platform   │ Trades │ Total PnL     │ Timing   │ Price Used │
├────────────┼────────┼───────────────┼──────────┼────────────┤
│ qengine    │ 2      │ $1,234.56     │ next_bar │ open       │
│ VectorBT   │ 2      │ $1,235.10     │ same_bar │ close      │
│ Backtrader │ 2      │ $1,234.50     │ next_bar │ open       │
│ Zipline    │ 2      │ $1,234.60     │ next_bar │ open       │
└────────────┴────────┴───────────────┴──────────┴────────────┘

Match Analysis:
✅ Trade count: All platforms agree (2 trades)
⚠️  PnL difference: $0.54 (0.04%) - Within tolerance
✅ Execution timing: qengine matches Backtrader and Zipline
❌ VectorBT uses different timing (same-bar execution)

Recommendation:
- qengine behavior validated against Backtrader/Zipline
- VectorBT difference explained: same-bar vs next-bar execution
- Configure qengine with same_bar_execution=True to match VectorBT
```

### Trade-by-Trade Comparison
```
Trade 1: Entry 2020-02-03, Exit 2020-04-15
──────────────────────────────────────────

Entry:
  qengine    : 2020-02-04 open  $73.50
  VectorBT   : 2020-02-03 close $73.85  [diff: $0.35]
  Backtrader : 2020-02-04 open  $73.50  [MATCH]
  Zipline    : 2020-02-04 open  $73.50  [MATCH]

Exit:
  qengine    : 2020-04-16 open  $71.10
  VectorBT   : 2020-04-15 close $71.35  [diff: $0.25]
  Backtrader : 2020-04-16 open  $71.10  [MATCH]
  Zipline    : 2020-04-16 open  $71.10  [MATCH]

PnL:
  qengine    : -$240.00
  VectorBT   : -$250.00  [diff: $10.00]
  Backtrader : -$240.00  [MATCH]
  Zipline    : -$242.00  [diff: $2.00, commission calculation]
```

## Success Criteria

### Per Scenario
- ✅ All platforms execute all signals
- ✅ Differences < tolerance OR explained
- ✅ Configuration documented to replicate any platform

### Overall
- ✅ 20+ scenarios passing
- ✅ All order types validated
- ✅ Multi-asset validated
- ✅ Intraday data validated
- ✅ Clear documentation of design choices

## Key Insights to Capture

For every difference found:
1. **What**: Exact nature of difference (price, timing, PnL)
2. **Why**: Root cause (algorithm, configuration, data)
3. **How**: Configuration to match (if desired)
4. **Decision**: Accept as design choice or implement option

Example:
```
DIFFERENCE: qengine vs VectorBT entry timing

What: qengine enters at next bar open, VectorBT at signal bar close
Why: Event-driven vs vectorized execution model
How: Set qengine.config.execution_timing = 'same_bar'
Decision: Offer both modes, default to 'next_bar' (more realistic)
```

## Implementation Priority

**Week 1**: Foundation
- Clean up existing test/validation/ structure
- Implement Scenario 001-003 (market, limit, stop)
- Get qengine, VectorBT, Backtrader working in parallel

**Week 2**: Order types
- Implement Scenario 004-005 (brackets, trailing stops)
- Document execution differences
- Add configuration options to qengine

**Week 3**: Edge cases & multi-asset
- Implement Scenario 006-013
- Validate position management
- Test simultaneous signals

**Week 4**: Intraday & polish
- Implement Scenario 014-016
- Generate comprehensive HTML reports
- Document all findings

---

**Philosophy**: We're not trying to match VectorBT exactly. We're validating that qengine's trading mechanics are correct and transparent, with clear documentation of design choices.
