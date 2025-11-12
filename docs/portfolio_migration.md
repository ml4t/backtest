# Portfolio Migration Guide

Upgrading from legacy Portfolio classes to the new unified Portfolio.

## Table of Contents

1. [Overview](#overview)
2. [Quick Migration](#quick-migration)
3. [SimplePortfolio Migration](#simpleportfolio-migration)
4. [PortfolioAccounting Migration](#portfolioaccounting-migration)
5. [Breaking Changes](#breaking-changes)
6. [Benefits of Upgrading](#benefits-of-upgrading)

---

## Overview

### What Changed?

**Removed (Legacy):**
- `SimplePortfolio` - Basic portfolio tracking
- `PortfolioAccounting` - Wrapper with additional metrics

**New (Unified):**
- `Portfolio` - Facade combining all functionality

### Migration Path

```python
# Old code
from qengine.portfolio.simple import SimplePortfolio
portfolio = SimplePortfolio(initial_capital=100000)

# New code
from qengine.portfolio import Portfolio
portfolio = Portfolio(initial_cash=100000)
```

---

## Quick Migration

### 1. Update Imports

```python
# Before
from qengine.portfolio.simple import SimplePortfolio
from qengine.portfolio.accounting import PortfolioAccounting

# After
from qengine.portfolio import Portfolio
```

### 2. Update Initialization

```python
# Before (SimplePortfolio)
portfolio = SimplePortfolio(
    initial_capital=100000,
    currency="USD"
)

# After (Portfolio)
portfolio = Portfolio(
    initial_cash=100000,  # Note: parameter name changed
    currency="USD"
)
```

```python
# Before (PortfolioAccounting)
accounting = PortfolioAccounting(
    initial_cash=100000,
    track_history=True
)

# After (Portfolio)
portfolio = Portfolio(
    initial_cash=100000,
    track_analytics=True  # Note: parameter name changed
)
```

### 3. Update Method Calls

Most methods remain the same, but some have new names:

```python
# Position tracking - UNCHANGED
position = portfolio.get_position("BTC")
positions = portfolio.positions

# Price updates - UNCHANGED
portfolio.update_prices({"BTC": 51000.0})

# Metrics - SLIGHTLY DIFFERENT
# Before: accounting.get_performance_metrics()
# After:  portfolio.get_performance_metrics()  # Same!

# Trade data - SLIGHTLY DIFFERENT
# Before: accounting.get_trades_df()
# After:  portfolio.get_trades()  # Note: df suffix removed
```

---

## SimplePortfolio Migration

### Complete Mapping

| SimplePortfolio | New Portfolio | Notes |
|-----------------|---------------|-------|
| `initial_capital` param | `initial_cash` param | Parameter renamed |
| `cash` property | `cash` property | Unchanged |
| `equity` property | `equity` property | Unchanged |
| `positions` property | `positions` property | Unchanged |
| `get_position(asset_id)` | `get_position(asset_id)` | Unchanged |
| `get_total_value()` | `equity` property | Use property instead |
| `get_positions()` | `positions` property | Use property instead |
| `get_trades()` | `get_trades()` | Unchanged |
| `get_returns()` | `returns` property | Use property instead |
| `calculate_metrics()` | `get_performance_metrics()` | Method renamed |
| `update_market_value(event)` | `update_prices(prices)` | Signature changed |
| `process_fill(event)` | `on_fill_event(event)` | Method renamed |
| `reset()` | `reset()` | Unchanged |

### Code Examples

#### Before: SimplePortfolio

```python
from qengine.portfolio.simple import SimplePortfolio

# Initialization
portfolio = SimplePortfolio(initial_capital=100000, currency="USD")

# Process fills
portfolio.process_fill(fill_event)

# Get metrics
total_value = portfolio.get_total_value()
positions_dict = portfolio.get_positions()
metrics = portfolio.calculate_metrics()

# Update prices
market_event = MarketEvent(...)
portfolio.update_market_value(market_event)

# Reset
portfolio.reset()
```

#### After: Portfolio

```python
from qengine.portfolio import Portfolio

# Initialization
portfolio = Portfolio(initial_cash=100000, currency="USD")

# Process fills
portfolio.on_fill_event(fill_event)

# Get metrics
total_value = portfolio.equity  # Property, not method
positions_dict = portfolio.positions  # Property, not method
metrics = portfolio.get_performance_metrics()

# Update prices
prices = {"BTC": 51000.0, "ETH": 3200.0}
portfolio.update_prices(prices)  # Dict, not event

# Reset
portfolio.reset()
```

### Migration Script

```python
def migrate_simpleportfolio_to_portfolio(old_portfolio):
    """Migrate SimplePortfolio instance to new Portfolio."""
    from qengine.portfolio import Portfolio

    # Create new portfolio with same initial state
    new_portfolio = Portfolio(
        initial_cash=old_portfolio.initial_capital,
        currency=getattr(old_portfolio, 'currency', 'USD')
    )

    # Copy state (if needed during migration)
    new_portfolio.cash = old_portfolio.cash

    # Note: Position objects are compatible
    for asset_id, position in old_portfolio.positions.items():
        new_portfolio._tracker.positions[asset_id] = position

    return new_portfolio
```

---

## PortfolioAccounting Migration

### Complete Mapping

| PortfolioAccounting | New Portfolio | Notes |
|---------------------|---------------|-------|
| `initial_cash` param | `initial_cash` param | Unchanged |
| `track_history` param | `track_analytics` param | Parameter renamed |
| `portfolio` property | `tracker` property | Internal component renamed |
| `process_fill(event)` | `on_fill_event(event)` | Method renamed |
| `update_prices(prices, ts)` | `update_prices(prices)` | Timestamp removed |
| `calculate_win_rate()` | `journal.calculate_win_rate()` | Moved to journal |
| `calculate_profit_factor()` | `journal.calculate_profit_factor()` | Moved to journal |
| `calculate_avg_commission()` | `journal.calculate_avg_commission()` | Moved to journal |
| `calculate_avg_slippage()` | `journal.calculate_avg_slippage()` | Moved to journal |
| `get_performance_metrics()` | `get_performance_metrics()` | Unchanged |
| `get_trades_df()` | `get_trades()` | df suffix removed |
| `get_equity_curve_df()` | `analyzer.equity_curve` | Access directly |
| `get_positions_df()` | `positions` property | Use property |
| `get_summary()` | `get_position_summary()` | Method renamed |
| `reset()` | `reset()` | Unchanged |

### Code Examples

#### Before: PortfolioAccounting

```python
from qengine.portfolio.accounting import PortfolioAccounting

# Initialization
accounting = PortfolioAccounting(
    initial_cash=100000,
    track_history=True
)

# Process fills
accounting.process_fill(fill_event)

# Update prices
accounting.update_prices(
    prices={"BTC": 51000.0},
    timestamp=datetime.now()
)

# Get metrics
metrics = accounting.get_performance_metrics()
win_rate = accounting.calculate_win_rate()
profit_factor = accounting.calculate_profit_factor()
avg_commission = accounting.calculate_avg_commission()

# Get DataFrames
trades_df = accounting.get_trades_df()
equity_df = accounting.get_equity_curve_df()
positions_df = accounting.get_positions_df()

# Reset
accounting.reset()
```

#### After: Portfolio

```python
from qengine.portfolio import Portfolio

# Initialization
portfolio = Portfolio(
    initial_cash=100000,
    track_analytics=True  # Note: parameter renamed
)

# Process fills
portfolio.on_fill_event(fill_event)

# Update prices (no timestamp needed)
portfolio.update_prices({"BTC": 51000.0})

# Get metrics
metrics = portfolio.get_performance_metrics()
win_rate = portfolio.journal.calculate_win_rate()  # Via journal
profit_factor = portfolio.journal.calculate_profit_factor()
avg_commission = portfolio.journal.calculate_avg_commission()

# Get DataFrames
trades_df = portfolio.get_trades()  # df suffix removed
equity_curve = portfolio.analyzer.equity_curve  # Direct access
positions_dict = portfolio.positions  # Dict, not DataFrame

# Reset
portfolio.reset()
```

### Migration Script

```python
def migrate_accounting_to_portfolio(old_accounting):
    """Migrate PortfolioAccounting instance to new Portfolio."""
    from qengine.portfolio import Portfolio

    # Create new portfolio with same settings
    new_portfolio = Portfolio(
        initial_cash=old_accounting.portfolio.initial_cash,
        track_analytics=old_accounting.track_history
    )

    # Copy state (if needed during migration)
    new_portfolio.cash = old_accounting.portfolio.cash

    # Copy positions
    for asset_id, position in old_accounting.portfolio.positions.items():
        new_portfolio._tracker.positions[asset_id] = position

    # Copy metrics state
    if old_accounting.track_history and new_portfolio.analyzer:
        new_portfolio.analyzer.high_water_mark = old_accounting.high_water_mark
        new_portfolio.analyzer.max_drawdown = old_accounting.max_drawdown
        new_portfolio.analyzer.equity_curve = old_accounting.equity_curve.copy()
        new_portfolio.analyzer.daily_returns = old_accounting.daily_returns.copy()
        new_portfolio.analyzer.timestamps = old_accounting.timestamps.copy()

    # Copy fills to journal
    new_portfolio.journal.fills = old_accounting.fills.copy()

    return new_portfolio
```

---

## Breaking Changes

### 1. Parameter Names

```python
# Breaking: initial_capital → initial_cash
# Before
portfolio = SimplePortfolio(initial_capital=100000)

# After
portfolio = Portfolio(initial_cash=100000)

# Breaking: track_history → track_analytics
# Before
accounting = PortfolioAccounting(track_history=True)

# After
portfolio = Portfolio(track_analytics=True)
```

### 2. Method Names

```python
# Breaking: process_fill() → on_fill_event()
# Before
portfolio.process_fill(fill_event)

# After
portfolio.on_fill_event(fill_event)

# Breaking: calculate_metrics() → get_performance_metrics()
# Before
metrics = portfolio.calculate_metrics()

# After
metrics = portfolio.get_performance_metrics()

# Breaking: get_trades_df() → get_trades()
# Before
trades = accounting.get_trades_df()

# After
trades = portfolio.get_trades()
```

### 3. Properties vs Methods

```python
# Breaking: get_total_value() → equity property
# Before
total_value = portfolio.get_total_value()

# After
total_value = portfolio.equity  # Property

# Breaking: get_positions() → positions property
# Before
positions_dict = portfolio.get_positions()

# After
positions_dict = portfolio.positions  # Property

# Breaking: get_returns() → returns property
# Before
returns = portfolio.get_returns()

# After
returns = portfolio.returns  # Property
```

### 4. Component Access

```python
# Breaking: Trade metrics moved to journal
# Before
win_rate = accounting.calculate_win_rate()

# After
win_rate = portfolio.journal.calculate_win_rate()

# Breaking: Equity curve access
# Before
equity_df = accounting.get_equity_curve_df()

# After
equity_list = portfolio.analyzer.equity_curve
timestamps = portfolio.analyzer.timestamps
```

### 5. DataFrame Changes

```python
# Breaking: get_positions_df() returns dict, not DataFrame
# Before
positions_df = accounting.get_positions_df()  # Polars DataFrame

# After
positions_dict = portfolio.positions  # dict[AssetId, Position]

# If you need DataFrame format:
import polars as pl
positions_data = [
    {
        "asset_id": pos.asset_id,
        "quantity": pos.quantity,
        "cost_basis": pos.cost_basis,
        "market_value": pos.market_value,
        "unrealized_pnl": pos.unrealized_pnl
    }
    for pos in portfolio.positions.values()
]
positions_df = pl.DataFrame(positions_data) if positions_data else None
```

---

## Benefits of Upgrading

### 1. Better Performance

```python
# HFT mode: 2.9x faster, 90% less memory
portfolio = Portfolio(
    initial_cash=100000,
    track_analytics=False  # Disable analytics overhead
)
```

### 2. Cleaner API

```python
# Before: Mix of methods and properties
total_value = portfolio.get_total_value()
positions = portfolio.get_positions()

# After: Consistent properties
total_value = portfolio.equity
positions = portfolio.positions
```

### 3. Better Testing

```python
# New architecture: 97-100% test coverage
# - 14 PositionTracker tests
# - 20 PerformanceAnalyzer tests
# - 24 TradeJournal tests
# - 21 Portfolio integration tests
# Total: 79 tests with comprehensive coverage
```

### 4. Extensibility

```python
# Easy to customize
from qengine.portfolio.analytics import PerformanceAnalyzer

class MyAnalyzer(PerformanceAnalyzer):
    def calculate_custom_metric(self):
        # Your logic
        pass

portfolio = Portfolio(
    initial_cash=100000,
    analyzer_class=MyAnalyzer  # Inject custom analyzer
)
```

### 5. Maintained Codebase

```python
# Old: 620 lines across 2 classes (deprecated)
# New: 306 lines across 3 components (actively maintained)
# Result: Cleaner, more maintainable code
```

---

## Common Migration Issues

### Issue 1: Missing track_history Parameter

```python
# Error: __init__() got an unexpected keyword argument 'track_history'

# Fix: Rename to track_analytics
portfolio = Portfolio(
    initial_cash=100000,
    track_analytics=True  # Was: track_history=True
)
```

### Issue 2: Missing process_fill Method

```python
# AttributeError: 'Portfolio' object has no attribute 'process_fill'

# Fix: Rename to on_fill_event
portfolio.on_fill_event(fill_event)  # Was: process_fill(fill_event)
```

### Issue 3: Component Access

```python
# AttributeError: 'Portfolio' object has no attribute 'portfolio'

# Fix: Use 'tracker' instead
# Before
cash = accounting.portfolio.cash

# After
cash = portfolio.tracker.cash
# Or better: use facade property
cash = portfolio.cash
```

### Issue 4: DataFrame Methods

```python
# AttributeError: 'Portfolio' object has no attribute 'get_trades_df'

# Fix: Use get_trades() without df suffix
trades = portfolio.get_trades()  # Was: get_trades_df()
```

### Issue 5: Trade Metrics

```python
# AttributeError: 'Portfolio' object has no attribute 'calculate_win_rate'

# Fix: Access via journal component
win_rate = portfolio.journal.calculate_win_rate()
profit_factor = portfolio.journal.calculate_profit_factor()
```

---

## Automated Migration Tool

### Migration Script

```python
#!/usr/bin/env python3
"""
Automated migration script for Portfolio API changes.

Usage:
    python migrate_portfolio.py <source_file> [--dry-run]
"""

import re
import sys
from pathlib import Path

REPLACEMENTS = [
    # Imports
    (r'from qengine\.portfolio\.simple import SimplePortfolio',
     'from qengine.portfolio import Portfolio'),
    (r'from qengine\.portfolio\.accounting import PortfolioAccounting',
     'from qengine.portfolio import Portfolio'),

    # Class names
    (r'\bSimplePortfolio\b', 'Portfolio'),
    (r'\bPortfolioAccounting\b', 'Portfolio'),

    # Parameters
    (r'initial_capital=', 'initial_cash='),
    (r'track_history=', 'track_analytics='),

    # Methods
    (r'\.process_fill\(', '.on_fill_event('),
    (r'\.calculate_metrics\(', '.get_performance_metrics('),
    (r'\.get_trades_df\(', '.get_trades('),
    (r'\.get_equity_curve_df\(', '.analyzer.equity_curve  # '),
    (r'\.get_positions_df\(', '.positions  # '),

    # Properties (method calls to properties)
    (r'\.get_total_value\(\)', '.equity'),
    (r'\.get_positions\(\)', '.positions'),
    (r'\.get_returns\(\)', '.returns'),

    # Component access
    (r'\.calculate_win_rate\(\)', '.journal.calculate_win_rate()'),
    (r'\.calculate_profit_factor\(\)', '.journal.calculate_profit_factor()'),
    (r'\.calculate_avg_commission\(\)', '.journal.calculate_avg_commission()'),
    (r'\.calculate_avg_slippage\(\)', '.journal.calculate_avg_slippage()'),
]

def migrate_file(filepath: Path, dry_run: bool = False) -> None:
    """Migrate a single Python file."""
    content = filepath.read_text()
    original = content

    # Apply all replacements
    for pattern, replacement in REPLACEMENTS:
        content = re.sub(pattern, replacement, content)

    # Show changes
    if content != original:
        if dry_run:
            print(f"Would modify: {filepath}")
            # Could add diff output here
        else:
            filepath.write_text(content)
            print(f"Modified: {filepath}")
    else:
        print(f"No changes: {filepath}")

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    filepath = Path(sys.argv[1])
    dry_run = '--dry-run' in sys.argv

    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    migrate_file(filepath, dry_run=dry_run)

if __name__ == "__main__":
    main()
```

### Usage

```bash
# Dry run (show changes without modifying)
python migrate_portfolio.py my_strategy.py --dry-run

# Apply changes
python migrate_portfolio.py my_strategy.py

# Migrate entire directory
find . -name "*.py" -exec python migrate_portfolio.py {} \;
```

---

## Testing After Migration

### Verification Checklist

- [ ] All imports updated
- [ ] All parameter names updated
- [ ] All method names updated
- [ ] All property accesses updated
- [ ] Component accesses updated (journal, analyzer)
- [ ] Tests pass
- [ ] Backtest results match (if comparing)

### Comparison Test

```python
def test_migration_equivalence():
    """Verify new Portfolio produces same results as old."""
    from qengine.portfolio import Portfolio

    # Setup
    portfolio = Portfolio(initial_cash=100000)

    # Run same fills as before
    for fill in historical_fills:
        portfolio.on_fill_event(fill)

    # Compare results
    assert abs(portfolio.equity - expected_equity) < 0.01
    assert abs(portfolio.returns - expected_returns) < 0.0001

    metrics = portfolio.get_performance_metrics()
    assert abs(metrics["sharpe_ratio"] - expected_sharpe) < 0.01
```

---

## See Also

- [API Reference](portfolio_api.md) - Complete new API
- [Architecture Guide](portfolio_architecture.md) - Design rationale
- [Extension Guide](portfolio_extensions.md) - Customization
