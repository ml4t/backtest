# ml4t.backtest Documentation Index

**Version**: 0.2.0 | **Python**: 3.11+ | **Status**: Production Ready

This index provides navigation to all documentation for the ml4t.backtest library.

---

## Getting Started

| Document | Description |
|----------|-------------|
| [Quick Start](guides/QUICKSTART.md) | 5-minute introduction to core concepts |
| [Migration Guide](guides/MIGRATION_GUIDE.md) | Migrating from other backtesting frameworks |
| [Differentiators](guides/DIFFERENTIATORS.md) | What makes ml4t.backtest unique |
| [Features](FEATURES.md) | Comprehensive feature inventory |

---

## Architecture

| Document | Description |
|----------|-------------|
| [Architecture Overview](architecture/ARCHITECTURE.md) | Core design principles and components |
| [Point-in-Time Correctness](architecture/PIT_AND_STATE.md) | How look-ahead bias is prevented |

---

## API Reference

| Document | Description |
|----------|-------------|
| [Complete API Reference](api/complete_reference.md) | Full API documentation |
| [Risk Management API](api/risk_management.md) | Position and portfolio risk rules |
| [Data Layer API](api/data_layer.md) | DataFeed and data handling |

---

## Guides

### Data & Feeds

| Document | Description |
|----------|-------------|
| [Data Feeds](guides/data_feeds.md) | Working with price and signal data |
| [Data Architecture](guides/data_architecture.md) | Internal data handling design |
| [Data Optimization](guides/data_optimization.md) | Performance tuning for large datasets |

### Risk Management

| Document | Description |
|----------|-------------|
| [Risk Management Quickstart](guides/risk_management_quickstart.md) | Getting started with risk rules |
| [Troubleshooting](TROUBLESHOOTING.md) | Common issues and solutions |
| [Integrated ML Risk](guides/integrated_ml_risk.md) | Combining ML signals with risk management |
| [Migration to Integrated](guides/migration_to_integrated.md) | Upgrading to integrated risk system |

### Configuration

| Document | Description |
|----------|-------------|
| [Configuration Guide](configuration_guide.md) | BacktestConfig and presets |
| [ML Signals](ml_signals.md) | Using machine learning signals |
| [Margin Calculations](margin_calculations.md) | Understanding margin requirements |

---

## Technical Deep Dives

### Fixes & Improvements

| Document | Description |
|----------|-------------|
| [Cash Constraint Fix](CASH_CONSTRAINT_FIX.md) | How capital validation works |
| [Clock Sync Fix](CLOCK_SYNC_FIX.md) | Time synchronization across components |
| [Lookahead Bias Fix](LOOKAHEAD_BIAS_FIX.md) | Preventing data leakage |
| [PnL Calculation Fix](PNL_CALCULATION_FIX.md) | Accurate profit/loss computation |

### Design & Reviews

| Document | Description |
|----------|-------------|
| [QEngine Design Review](qengine_design_review.md) | Original architecture review |
| [QEngine Review](qengine-review.md) | Follow-up design analysis |
| [Backtester Comparison](BACKTESTER_COMPARISON.md) | Comparison with other frameworks |
| [Competitive Positioning](competitive-positioning.md) | Market positioning analysis |

---

## Portfolio & Execution

| Document | Description |
|----------|-------------|
| [Portfolio API](portfolio_api.md) | Working with portfolios |
| [Portfolio Architecture](portfolio_architecture.md) | Portfolio system design |
| [Portfolio Migration](portfolio_migration.md) | Upgrading portfolio handling |
| [Portfolio Extensions](portfolio_extensions.md) | Advanced portfolio features |
| [Trade Reporting](TRADE_REPORTING.md) | Trade analysis and reporting |

---

## Advanced Topics

| Document | Description |
|----------|-------------|
| [Irregular Timestamps Support](IRREGULAR_TIMESTAMPS_SUPPORT.md) | Handling non-uniform data |
| [Cross Library API](cross_library_api.md) | Integration with other ml4t libraries |
| [Functionality Inventory](FUNCTIONALITY_INVENTORY.md) | Complete capability list |
| [Delivery Summary](DELIVERY_SUMMARY.md) | Release notes and delivery history |
| [Rust Backend Feasibility](rust-backend-feasibility.md) | Performance optimization analysis |

---

## Sphinx Documentation

For auto-generated API documentation from source code:

- [Sphinx HTML Docs](sphinx/build/html/index.html)
- [Sphinx README](sphinx/README.md)

---

## Examples

See the [`examples/`](../examples/) directory for runnable examples:

1. `01_basic_multi_asset.py` - Multi-asset equal-weight rebalancing
2. `02_riskfolio_integration.py` - Riskfolio-Lib portfolio optimization
3. `03_skfolio_integration.py` - skfolio portfolio optimization
4. `04_risk_parity.py` - Risk parity strategy

---

## Quick Reference

### Import Patterns

```python
# Core usage
from ml4t.backtest import Engine, Strategy, DataFeed, Broker

# Types
from ml4t.backtest import Order, Position, Fill, Trade
from ml4t.backtest import OrderType, OrderSide, ExecutionMode

# Risk management
from ml4t.backtest import StopLoss, TakeProfit, TrailingStop
from ml4t.backtest import MaxPositions, MaxExposure

# Configuration
from ml4t.backtest import BacktestConfig, run_backtest
```

### Minimal Example

```python
from ml4t.backtest import Engine, Strategy, DataFeed

class MyStrategy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        for asset, bar in data.items():
            if bar['signal'] > 0.5 and asset not in broker.get_positions():
                broker.submit_order(asset=asset, quantity=100)

feed = DataFeed(prices_df=df)
engine = Engine(feed=feed, strategy=MyStrategy())
result = engine.run()
```

---

*Last updated: January 2026*
