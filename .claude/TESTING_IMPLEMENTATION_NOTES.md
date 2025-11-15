# ml4t.backtest Testing Implementation - Technical Notes

**Document**: Supplementary technical guidance for building validation scenarios
**Audience**: Developers implementing the test scenarios roadmap

---

## Quick Reference: Scenario Implementation

### Minimal Scenario Template

```python
"""
Scenario NNN: [Title]

Purpose: [One-sentence explanation of what this tests]
"""

from dataclasses import dataclass
from datetime import datetime
import polars as pl


@dataclass
class Signal:
    """Platform-independent signal specification."""
    timestamp: datetime
    asset: str
    action: str  # 'BUY', 'SELL'
    quantity: float
    order_type: str = 'MARKET'
    limit_price: float | None = None
    stop_price: float | None = None


class ScenarioNNN:
    name = "NNN_title"
    description = "Full description for test reports"
    
    @staticmethod
    def get_data() -> pl.DataFrame:
        """Return synthetic or real OHLCV data."""
        import pandas as pd
        dates = pd.date_range(start='2020-01-01', periods=252, freq='D')
        
        # Generate synthetic price data
        prices = [100.0 + i*0.1 for i in range(252)]
        
        return pl.DataFrame({
            'timestamp': dates,
            'symbol': ['AAPL'] * 252,
            'open': [p + 0.5 for p in prices],
            'high': [p + 2.0 for p in prices],
            'low': [p - 1.5 for p in prices],
            'close': prices,
            'volume': [1_000_000] * 252,
        })
    
    signals = [
        Signal(datetime(2020, 2, 3), 'AAPL', 'BUY', 100),
        Signal(datetime(2020, 4, 15), 'AAPL', 'SELL', 100),
    ]
    
    config = {
        'initial_capital': 100_000.0,
        'commission': 0.001,
        'slippage': 0.0,
    }
    
    expected = {
        'trade_count': 1,
        'final_position': 0,
    }
```

---

## Common Pitfalls and Solutions

### 1. Signal Timing Issues

**Problem**: Signal at timestamp T - which bar's price to use?

**Solution**: Document clearly in scenario:
- `execution_timing: 'same_bar'` → Use close price of signal bar
- `execution_timing: 'next_bar'` → Use open price of next bar

**Example in scenario**:
```python
config = {
    'execution_timing': 'next_bar',  # CRITICAL: Document this
}

platform_notes = {
    'ml4t.backtest': 'Executes at next bar open (default)',
    'vectorbt': 'May execute same bar if price touched',
    'backtrader': 'Next bar open for market orders',
}
```

### 2. Data Boundaries

**Problem**: What if signal is on last bar of data?

**Solution**: In scenario, ensure signals don't hit data boundaries:
```python
# BAD: Signal on 2020-12-31, no next bar
Signal(datetime(2020, 12, 31), 'AAPL', 'SELL', 100),

# GOOD: Signal on 2020-12-30, next bar available
Signal(datetime(2020, 12, 30), 'AAPL', 'SELL', 100),
```

### 3. Limit Orders Not Triggering

**Problem**: Limit order price never in OHLC range

**Solution**: Verify price is actually reachable:
```python
# Data: high=105, low=95
# GOOD: Limit price 100 is within range
Signal(..., order_type='LIMIT', limit_price=100.0),

# BAD: Limit price 80 never reached (below low of 95)
Signal(..., order_type='LIMIT', limit_price=80.0),
```

### 4. Stop Orders in Trending Markets

**Problem**: Stop triggers immediately if price moves against you

**Solution**: Design data sequence carefully:
```python
# Position: LONG at 100
# Day 1: high=102, low=98 (no SL trigger at 97)
# Day 2: high=99, low=92 (SL at 97 triggers!)
# Day 3: high=105, low=100
```

### 5. Bracket Order Timing

**Problem**: Both TP and SL marked as filled - which one actually executed?

**Solution**: Use scenarios where only ONE can trigger:
```python
# Entry: BUY at 100 with TP=105, SL=95
# Day sequence:
#   Day 1: Close 101 (no action)
#   Day 2: High 106, Low 100 (TP triggers, exit at 105)
#   Day 3: (SL never checked - position already closed)
```

---

## Execution Timing Deep Dive

### ml4t.backtest Event Loop

```
FOR each bar in data:
    1. Update market data (new OHLCV)
    2. Check open orders:
       a. LIMIT: If low <= limit_price (buy) OR high >= limit_price (sell) → FILL
       b. STOP: If low <= stop_price (sell) OR high >= stop_price (buy) → FILL
       c. MARKET: Always fill at open price
    3. Generate stop events if triggered
    4. Call strategy.on_event(market_event)
    5. Process any new orders from strategy
```

### Execution Timing Options

**Option A: Same-Bar (Close)**
- Signal arrives at timestamp T
- Fill at close price of bar T
- Used by: VectorBT (sometimes)
- Risk: Look-ahead bias if not careful

**Option B: Next-Bar (Open)**
- Signal arrives at timestamp T
- Fill at open price of bar T+1
- Used by: Backtrader (default), Zipline (default)
- Advantage: Realistic (market closed, can't execute until next open)

**Option C: Intrabar (OHLC Range)**
- Signal at T, check if price reached during bar T+1
- Most realistic for stop/limit orders
- Used by: VectorBT Pro, realistic simulators
- Risk: Need accurate intrabar data

---

## Platform-Specific Behaviors

### VectorBT Quirks

1. **Single execution per bar**: Can't execute both entry and exit on same bar
2. **Fees timing**: Applied at fill time, not separately
3. **Array-based**: All signals processed together, not event-by-event
4. **No partial fills**: Either fills completely or doesn't fill

**Testing strategy**: Use simple signals that don't conflict

### Backtrader Quirks

1. **next() called at bar close**: Signal delays are inherent
2. **Cheat-on-close mode**: Available for same-bar execution
3. **Live trading difference**: Orders in live may behave differently
4. **Commission method**: setcommission() vs direct order cost

**Testing strategy**: Stick with next_bar execution for consistency

### Zipline Quirks

1. **Data bundle requirement**: Need to set up custom bundle
2. **Volume constraints**: Orders limited to 2.5% of volume
3. **Slippage model**: Affects fill prices directly
4. **Context object**: State machine for portfolio

**Testing strategy**: Pre-generate data in correct format

---

## How to Debug Scenario Failures

### Step 1: Run Data Analysis

```python
scenario = Scenario001()
scenario.analyze_data_at_signals()
# Output:
# Signal 1: BUY at 2020-02-03
#   Signal bar: O=70.50 H=72.00 L=68.50 C=70.00
#   Next bar:   O=70.10 H=72.10 L=68.60 C=70.10
```

### Step 2: Check Manual P&L Calculation

```python
trades = scenario.expected_pnl()
# Should print expected values
# Compare with actual extracted trades
```

### Step 3: Extract Platform Trades

```python
from runner import ScenarioRunner
import importlib

module = importlib.import_module(f'scenarios.scenario_001_...')
runner = ScenarioRunner(module)

# Run ml4t.backtest only
results = runner.run(['ml4t.backtest'])
result = results['ml4t.backtest']

# Check for errors
if result.errors:
    print("Errors:", result.errors)
else:
    print("Results:", result.raw_results)
```

### Step 4: Extract and Inspect Trades

```python
from extractors import extract_ml4t.backtest_trades

trades = extract_ml4t.backtest_trades(result.raw_results, result.data)
for trade in trades:
    print(f"Entry: {trade.entry_timestamp} @ {trade.entry_price}")
    print(f"Exit:  {trade.exit_timestamp} @ {trade.exit_price if trade.exit_price else 'OPEN'}")
    print(f"PnL:   {trade.net_pnl}")
    print()
```

### Step 5: Check Price Components

```python
# Infer which OHLC component was used
from core.trade import infer_price_component, get_bar_at_timestamp

bar = get_bar_at_timestamp(data, trade.entry_timestamp)
component = infer_price_component(trade.entry_price, bar)
print(f"Entry price ${trade.entry_price} matches {component}")
```

### Step 6: Compare Across Platforms

```python
# Run all platforms
results = runner.run(['ml4t.backtest', 'vectorbt', 'backtrader'])

# Extract trades from each
from extractors import extract_ml4t.backtest_trades, extract_vectorbt_trades, extract_backtrader_trades

trades_ml4t.backtest = extract_ml4t.backtest_trades(results['ml4t.backtest'].raw_results, results['ml4t.backtest'].data)
trades_vbt = extract_vectorbt_trades(results['vectorbt'].raw_results['portfolio'], results['vectorbt'].raw_results['data'])
trades_bt = extract_backtrader_trades(results['backtrader'].raw_results['trades'], results['backtrader'].raw_results['data'])

# Match trades
from comparison import match_trades

matches = match_trades({
    'ml4t.backtest': trades_ml4t.backtest,
    'vectorbt': trades_vbt,
    'backtrader': trades_bt,
})

# Inspect differences
for match in matches:
    if match.differences:
        print(f"Trade {match.reference_trade.trade_id}:")
        for diff in match.differences:
            print(f"  - {diff}")
```

---

## Creating Synthetic Data Utilities

### Time-Series Data Generation

```python
# In fixtures/data_generator.py

import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class SyntheticDataGenerator:
    """Create realistic OHLCV data for testing."""
    
    @staticmethod
    def geometric_brownian_motion(
        S0: float = 100.0,
        mu: float = 0.0,  # drift
        sigma: float = 0.02,  # volatility
        days: int = 252,
        freq: str = 'D',
    ) -> np.ndarray:
        """Generate GBM price paths."""
        dt = 1.0 if freq == 'D' else (1/252 if freq == 'H' else 1/252/6.5/60)
        returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), days)
        prices = S0 * np.exp(np.cumsum(returns))
        return prices
    
    @staticmethod
    def create_ohlcv(
        dates: list[datetime],
        close_prices: list[float],
        volatility_intrabar: float = 0.01,
    ) -> pl.DataFrame:
        """Convert close prices to OHLCV bars with realistic ranges."""
        opens = close_prices.copy()
        highs = []
        lows = []
        
        for close, open_p in zip(close_prices, opens):
            # Random intrabar movement
            high = max(open_p, close) * (1 + abs(np.random.normal(0, volatility_intrabar)))
            low = min(open_p, close) * (1 - abs(np.random.normal(0, volatility_intrabar)))
            highs.append(high)
            lows.append(low)
        
        return pl.DataFrame({
            'timestamp': dates,
            'symbol': ['AAPL'] * len(dates),
            'open': opens,
            'high': highs,
            'low': lows,
            'close': close_prices,
            'volume': [1_000_000] * len(dates),
        })
```

### Asset Configuration

```python
# In fixtures/assets.py

SYNTHETIC_ASSETS = {
    'AAPL': {
        'start_price': 100.0,
        'volatility': 0.02,  # 2% daily
        'trend': 0.0005,  # 0.05% daily drift
        'liquidity': 'high',
    },
    'TSLA': {
        'start_price': 200.0,
        'volatility': 0.04,  # Higher volatility
        'trend': 0.001,
        'liquidity': 'high',
    },
    'SMALL_CAP': {
        'start_price': 50.0,
        'volatility': 0.05,  # High volatility
        'trend': 0.0,
        'liquidity': 'medium',  # For volume-limited fills
    },
}
```

---

## Scenario Naming Convention

**Format**: `scenario_NNN_descriptive_name.py`

**Examples**:
- `scenario_001_simple_market_orders.py` - Basic market order execution
- `scenario_006_limit_orders_entry.py` - Limit orders for entry
- `scenario_007_stop_loss_orders.py` - Stop-loss functionality
- `scenario_011_bracket_take_profit.py` - Bracket orders (TP triggers)
- `scenario_012_bracket_stop_loss.py` - Bracket orders (SL triggers)
- `scenario_016_reentry_open_position.py` - Re-entry with existing position
- `scenario_022_short_selling.py` - Short selling and margin
- `scenario_025_large_portfolio.py` - Stress test with 100+ assets

**Numbering**:
- 001-005: Basic market orders
- 006-010: Limit/stop orders
- 011-015: Bracket orders and OCO
- 016-020: Complex scenarios
- 021-025: Stress and edge cases
- 026+: Future expansion

---

## Assertion Patterns

### Trade Count Assertions

```python
def assert_trade_count(trades, expected):
    """Assert exact number of completed trades."""
    closed_trades = [t for t in trades if t.is_closed()]
    assert len(closed_trades) == expected, \
        f"Expected {expected} closed trades, got {len(closed_trades)}"
```

### Price Matching Assertions

```python
def assert_prices_within_tolerance(price1, price2, tolerance_pct=0.1):
    """Assert two prices match within percentage tolerance."""
    pct_diff = abs(price1 - price2) / ((price1 + price2) / 2) * 100
    assert pct_diff <= tolerance_pct, \
        f"Price {price1} vs {price2} differ by {pct_diff:.2f}% (limit: {tolerance_pct}%)"
```

### Timing Assertions

```python
def assert_execution_timing(trade, data, expected_timing):
    """Verify execution occurred at expected time."""
    signal_bar = get_bar_at_timestamp(data, trade.signal_timestamp)
    next_bar = get_next_bar(data, trade.signal_timestamp)
    
    if expected_timing == 'same_bar':
        assert trade.entry_price in [signal_bar['open'], signal_bar['close']]
    elif expected_timing == 'next_bar':
        assert trade.entry_price == next_bar['open']
```

### No Look-Ahead Assertions

```python
def assert_no_lookahead(trade, data):
    """Verify execution price is achievable at execution time."""
    bar = get_bar_at_timestamp(data, trade.entry_timestamp)
    assert bar['low'] <= trade.entry_price <= bar['high'], \
        f"Entry price ${trade.entry_price} not in OHLC range ${bar['low']}-${bar['high']}"
```

---

## Testing Different Execution Models

### How to Test Same-Bar vs Next-Bar

Create **two** scenarios with same signals but different timing:

```python
# scenario_002a_market_same_bar.py
class Scenario002A:
    name = "002a_market_same_bar"
    config = {'execution_timing': 'same_bar'}
    # Signals...
    expected = {'entry_prices': [70.0]}  # Use signal bar close

# scenario_002b_market_next_bar.py
class Scenario002B:
    name = "002b_market_next_bar"
    config = {'execution_timing': 'next_bar'}
    # Same signals...
    expected = {'entry_prices': [70.1]}  # Use next bar open
```

Then run both and compare platform behaviors.

---

## Maintenance and Evolution

### When to Add New Scenarios

1. **New order type discovered**: Create scenario validating it
2. **Bug fixed**: Create regression test scenario
3. **Platform difference found**: Create scenario highlighting it
4. **New feature implemented**: Create scenario testing it

### Scenario Lifecycle

```
IDEA → DRAFT → IMPLEMENTATION → VALIDATION → ACCEPTANCE → MAINTENANCE
```

**Draft**: Document what you want to test, why, and expected results
**Implementation**: Code the scenario following template
**Validation**: Run on ml4t.backtest, verify results make sense
**Cross-platform**: Run on VectorBT/Backtrader, document differences
**Acceptance**: All platforms pass within tolerance OR document expected differences
**Maintenance**: Keep up-to-date with code changes

---

## Key Files to Modify

When implementing the roadmap, you'll create/modify:

**New Files**:
- `tests/validation/scenarios/scenario_002_*.py` through `scenario_025_*.py` (20+ files)
- `tests/validation/fixtures/conftest.py` (fixture framework)
- `tests/validation/fixtures/data_generator.py` (synthetic data utilities)
- `tests/validation/config/execution_models.py` (execution timing configs)
- `tests/validation/validators/assertions.py` (assertion helpers)

**Modified Files**:
- `tests/validation/runner.py` (add scenario range/category support)
- `tests/validation/__init__.py` (export new utilities)
- `docs/TESTING_GUIDE.md` (new - comprehensive testing documentation)

**Documentation**:
- Update `README.md` with scenario list
- Add execution timing guide to `docs/`
- Create `docs/SCENARIOS.md` with all scenario descriptions

---

## Next Steps

1. **Review this document** with the team
2. **Create fixtures framework** (3-4 hours)
3. **Implement scenarios 002-005** (2-3 days)
4. **Validate on ml4t.backtest** (1 day)
5. **Run cross-platform comparison** (1 day)
6. **Iterate** on remaining scenarios

