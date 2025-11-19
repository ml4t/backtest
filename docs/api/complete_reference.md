# ml4t.backtest - Complete API Reference

**Package:** `ml4t.backtest`
**Version:** 0.1.0
**Last Updated:** November 2025

## Quick Links

- [Sphinx HTML Documentation](../sphinx/build/html/index.html)
- [Risk Management API](risk_management.md) (Manual Reference)
- [Data Layer API](data_layer.md) (Manual Reference)

---

## Table of Contents

1. [Engine & Configuration](#engine--configuration)
2. [Core Module](#core-module)
3. [Data Module](#data-module)
4. [Execution Module](#execution-module)
5. [Portfolio Module](#portfolio-module)
6. [Risk Management Module](#risk-management-module)
7. [Strategy Module](#strategy-module)
8. [Reporting Module](#reporting-module)

---

## Engine & Configuration

### BacktestEngine

The main orchestrator for event-driven backtesting.

**Location:** `ml4t.backtest.engine`

**Key Methods:**
- `run()` - Execute the backtest
- `step()` - Execute single event step
- `get_results()` - Retrieve backtest results

**Auto-Generated Docs:** [Sphinx HTML - Engine Module](../sphinx/build/html/modules/engine.html)

### BacktestConfig

Configuration schema for backtest setup.

**Location:** `ml4t.backtest.config`

**Auto-Generated Docs:** [Sphinx HTML - Engine Module](../sphinx/build/html/modules/engine.html)

### BacktestResults

Results container with performance metrics.

**Location:** `ml4t.backtest.results`

**Auto-Generated Docs:** [Sphinx HTML - Engine Module](../sphinx/build/html/modules/engine.html)

---

## Core Module

Fundamental types and infrastructure for the backtesting engine.

**Auto-Generated Docs:** [Sphinx HTML - Core Module](../sphinx/build/html/modules/core.html)

### Key Classes

- **Event** (`ml4t.backtest.core.event`) - MarketEvent, OrderEvent, FillEvent
- **Clock** (`ml4t.backtest.core.clock`) - Multi-feed event synchronization
- **Asset** (`ml4t.backtest.core.assets`) - Asset definitions and registry
- **Types** (`ml4t.backtest.core.types`) - Type aliases (AssetId, Price, Quantity)
- **Precision** (`ml4t.backtest.core.precision`) - Price/quantity precision management
- **Context** (`ml4t.backtest.core.context`) - Market context data structures

---

## Data Module

Data feed abstractions, feature providers, and validation.

**Manual Reference:** [Data Layer API](data_layer.md)
**Auto-Generated Docs:** [Sphinx HTML - Data Module](../sphinx/build/html/modules/data.html)

### Key Classes

- **PolarsDataFeed** - High-performance data feed with lazy loading
- **FeatureProvider** - Feature computation interface
- **PrecomputedFeatureProvider** - Pre-computed features from DataFrame
- **CallableFeatureProvider** - On-demand feature computation
- **SignalTimingMode** - Lookahead bias validation modes
- **DataFeed** - Abstract base class for data feeds

---

## Execution Module

Order execution, position sizing, commission, and slippage models.

**Auto-Generated Docs:** [Sphinx HTML - Execution Module](../sphinx/build/html/modules/execution.html)

### Key Classes

- **SimulationBroker** - Broker simulation with realistic fills
- **Order** - Order types (Market, Limit, Stop, StopLimit)
- **FillSimulator** - OHLC-based fill model
- **CommissionModel** - Per-share, percentage, tiered commissions
- **SlippageModel** - Fixed, percentage, volume-share slippage
- **MarketImpactModel** - Kyle, Almgren-Chriss impact models
- **LiquidityModel** - Liquidity constraints
- **TradeTracker** - Trade reconciliation and P&L tracking
- **BracketOrderManager** - OCO, OTO bracket orders
- **OrderRouter** - Order routing and validation
- **CorporateActions** - Stock splits, dividends, mergers

---

## Portfolio Module

Portfolio state management, positions, and analytics.

**Auto-Generated Docs:** [Sphinx HTML - Portfolio Module](../sphinx/build/html/modules/portfolio.html)

### Key Classes

- **Portfolio** - Portfolio state and position tracking
- **Position** - Position representation (core types)
- **PortfolioState** - State snapshots for analysis
- **PortfolioAnalytics** - Performance metrics (Sharpe, drawdown, etc.)
- **MarginAccount** - Margin calculations and constraints

---

## Risk Management Module

Composable risk management framework for position controls.

**Manual Reference:** [Risk Management API](risk_management.md) ⭐ **Complete Manual Reference**
**Auto-Generated Docs:** [Sphinx HTML - Risk Module](../sphinx/build/html/modules/risk.html)

### Key Classes

- **RiskManager** - Orchestrates risk rule evaluation
- **RiskContext** - Immutable state snapshot for risk evaluation
- **RiskDecision** - Risk rule output (exit, update stops, etc.)
- **RiskRule** - Abstract base class for risk rules
- **CompositeRule** - Combine multiple rules

### Built-In Rules

- **TimeBasedExit** - Exit after N bars
- **PriceBasedStopLoss** - Price-based stop loss
- **PriceBasedTakeProfit** - Price-based take profit
- **VolatilityScaledStop** - ATR-based stops
- **DynamicTrailingStop** - Trailing stops
- **RegimeDependentExit** - Exit based on market regime
- **PortfolioConstraints** - Portfolio-level risk limits

**See [Risk Management API](risk_management.md) for complete documentation with examples.**

---

## Strategy Module

Base classes and adapters for implementing trading strategies.

**Auto-Generated Docs:** [Sphinx HTML - Strategy Module](../sphinx/build/html/modules/strategy.html)

### Key Classes

- **Strategy** - Abstract base class for strategies
- **StrategyAdapter** - Adapter for external strategy frameworks
- **CryptoBasisAdapter** - Example: crypto basis trading
- **SPYOrderFlowAdapter** - Example: SPY order flow strategy

---

## Reporting Module

Backtest results reporting and visualization.

**Auto-Generated Docs:** [Sphinx HTML - Reporting Module](../sphinx/build/html/modules/reporting.html)

### Key Classes

- **Reporter** - Unified reporting API
- **TradeAnalysis** - Trade-by-trade analysis
- **TradeSchema** - Trade data schemas
- **Visualizations** - Equity curves, drawdowns, returns
- **HTMLReporter** - HTML tearsheet generation
- **ParquetReporter** - Parquet export for analysis

---

## Cross-References

### Manual API References

These modules have detailed manual API documentation with examples:

1. **[Risk Management API](risk_management.md)** - Complete manual reference with usage examples
2. **[Data Layer API](data_layer.md)** - Complete manual reference for PolarsDataFeed and FeatureProvider

### Auto-Generated API References

All modules have auto-generated Sphinx documentation with docstrings:

- **[HTML Documentation](../sphinx/build/html/index.html)** - Full Sphinx HTML site
- **[Module Index](../sphinx/build/html/py-modindex.html)** - All modules and classes
- **[Search](../sphinx/build/html/search.html)** - Full-text search

---

## Building Documentation

To regenerate this documentation:

```bash
# From repository root
cd docs/
./build_docs.sh
```

This will:
1. Build Sphinx HTML documentation
2. Generate this consolidated API reference
3. Verify all cross-references

---

## Contributing

When adding new classes or methods:

1. Add Google-style docstrings to source code
2. Update relevant module RST file in `docs/sphinx/source/modules/`
3. For complex modules, create manual reference in `docs/api/`
4. Run `./build_docs.sh` to regenerate
5. Verify cross-references work

---

**Last Generated:** 2025-11-18 18:43:39 EST
**Build Script:** `docs/build_docs.sh`
**Sphinx Version:** sphinx-build 8.2.3
