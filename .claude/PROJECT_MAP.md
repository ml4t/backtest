# Project Map: ml4t.backtest Backtesting Library

*Generated: 2025-11-15T07:00:00Z*
*Last Updated: 2025-11-15T07:00:00Z*

## Quick Overview
- **Type**: Python library for event-driven backtesting (part of ml4t namespace)
- **Primary Language**: Python 3.9+
- **Package Name**: `ml4t-backtest` (import as `ml4t.backtest`)
- **Namespace Structure**: Namespaced package within `ml4t.*` ecosystem
- **Frameworks**: Polars (data), Numba (performance), Pydantic (validation)
- **Purpose**: Institutional-grade backtesting engine for quantitative trading strategies
- **Location**: `/home/stefan/ml4t/software/backtest/`
- **Status**: Beta (pre-1.0), active development

## Architecture Overview

ml4t.backtest is an **event-driven backtesting engine** designed to replicate real trading conditions with point-in-time correctness and realistic execution modeling.

**Key Design Principles:**
1. **Event-Driven**: Iterates through market events sequentially (no look-ahead bias)
2. **Point-in-Time Correctness**: All decisions use only data available at decision time
3. **Realistic Execution**: Simulates order fills, slippage, commissions, market impact
4. **Institutional Features**: Corporate actions, margin, position tracking, trade reconciliation
5. **Performance**: Hybrid vectorized/event-driven architecture (100k+ events/second target)

**Differentiators vs Competitors:**
- **vs Backtrader/Zipline**: Faster (Polars/Numba), ML-first design, cleaner API
- **vs VectorBT**: Stateful event-driven logic (not just vectorized), live trading path
- **vs QuantConnect**: Open-source, self-hosted, Python-native (no C# bridge)

## Directory Structure

### Core Source Code (`src/ml4t/backtest/` - 41 files)

**Namespace Structure:** This is a namespaced package within the `ml4t.*` ecosystem, alongside:
- `ml4t.data` - Market data management
- `ml4t.features` - Feature engineering
- `ml4t.eval` - Statistical validation

```
src/ml4t/backtest/
├── __init__.py              # Package exports, version (ml4t.backtest)
├── engine.py                # Main BacktestEngine class (event loop orchestration)
│
├── core/                    # Fundamental types and utilities
│   ├── types.py            # Type aliases (AssetId, Quantity, Price, etc.)
│   ├── event.py            # Event system (MarketEvent, OrderEvent, FillEvent)
│   ├── clock.py            # Multi-feed event synchronization
│   ├── assets.py           # Asset definitions (Stock, Future, Option)
│   ├── precision.py        # Precision management for prices/quantities
│   └── constants.py        # Enums and constants
│
├── data/                    # Data ingestion and management
│   ├── feed.py             # DataFeed interface (historical + live)
│   ├── schemas.py          # Data validation schemas
│   └── asset_registry.py   # Asset metadata registry
│
├── execution/               # Order execution and simulation (13 files)
│   ├── broker.py           # SimulationBroker (position tracking, order routing)
│   ├── order.py            # Order types (Market, Limit, Stop, StopLimit)
│   ├── order_router.py     # Order routing and validation
│   ├── fill_simulator.py   # Fill model (OHLC-based realistic fills)
│   ├── slippage.py         # Slippage models (Fixed, Percentage, VolumeShare)
│   ├── commission.py       # Commission models (PerShare, Percentage, Tiered)
│   ├── market_impact.py    # Market impact models (Kyle, Almgren-Chriss)
│   ├── liquidity.py        # Liquidity constraints and modeling
│   ├── trade_tracker.py    # Trade reconciliation and P&L tracking
│   ├── position_sizer.py   # Position sizing strategies
│   ├── bracket_manager.py  # Bracket order management (OCO, OTO)
│   └── corporate_actions.py # Stock splits, dividends, mergers
│
├── portfolio/               # Portfolio state and analytics
│   ├── portfolio.py        # Portfolio class (positions, cash, equity)
│   ├── core.py             # Position class
│   ├── state.py            # Portfolio state snapshots
│   ├── analytics.py        # Performance metrics (Sharpe, drawdown, etc.)
│   └── margin.py           # Margin calculations and constraints
│
├── strategy/                # Strategy framework
│   ├── base.py             # Strategy base class (on_market_data, on_fill)
│   ├── adapters.py         # Strategy adapters (VectorBT, Backtrader compat)
│   └── crypto_basis_adapter.py # Example: crypto basis trading strategy
│
└── reporting/               # Results reporting and visualization
    ├── base.py             # Reporter interface
    ├── parquet.py          # Parquet export (trades, portfolio states)
    ├── html.py             # HTML report generation
    └── reporter.py         # Unified reporting API
```

### Test Organization (`tests/` - 152 files)

```
tests/
├── unit/                   # Unit tests (isolated component testing)
│   ├── test_order.py
│   ├── test_broker.py
│   ├── test_fill_simulator.py
│   ├── test_slippage.py
│   ├── test_commission.py
│   ├── test_clock_*.py    # Clock synchronization tests
│   └── ... (46 files total)
│
├── integration/            # Integration tests (multi-component)
│   ├── test_strategy_ml4t.backtest_comparison.py
│   ├── test_corporate_action_integration.py
│   └── ... (cross-component workflows)
│
├── validation/             # Cross-framework validation
│   ├── test_cross_framework_alignment.py  # ml4t.backtest vs VectorBT vs Backtrader
│   ├── adapters/
│   │   ├── ml4t.backtest_adapter.py
│   │   ├── vectorbt_adapter.py
│   │   ├── backtrader_adapter.py
│   │   └── zipline_adapter.py
│   └── frameworks/          # Framework wrapper implementations
│
└── comparison/             # Performance benchmarks
    └── ... (framework performance comparisons)
```

**Test Coverage:** 44% overall (goal: 80%+)
- Core modules: 50-90% (clock, types, events)
- Execution: 30-50% (broker, fills, orders)
- Portfolio: 70%+
- Strategy: 30-40%

### Documentation (`docs/` + root)

```
docs/
├── architecture/           # Architectural decision records
├── api/                    # API documentation (auto-generated)
└── guides/                 # User guides and tutorials

Root documentation:
├── README.md              # Project overview and quickstart
├── CLAUDE.md              # Development context for Claude Code
├── FRAMEWORK_VALIDATION_REPORT.md  # Cross-framework validation results
└── .claude/
    ├── memory/            # Persistent design decisions
    │   ├── ml_signal_architecture.md
    │   ├── multi_source_context_architecture.md
    │   ├── project_state.md
    │   ├── library_overview.md
    │   └── conventions.md
    └── transitions/       # Session handoff documents
```

### Configuration & Build

```
├── pyproject.toml         # Package config, dependencies, tools (pytest, ruff, mypy)
├── uv.lock                # Dependency lock file (uv package manager)
├── .github/               # CI/CD workflows
├── .gitignore
├── LICENSE
└── CHANGELOG.md
```

## Key Files

### Application Entry Points
- `src/ml4t/backtest/engine.py` - Main `BacktestEngine` class that orchestrates the event loop
- `src/ml4t/backtest/__init__.py` - Package exports and public API (import as `ml4t.backtest`)
- `src/ml4t/backtest/execution/broker.py` - `SimulationBroker` for backtesting (963 lines)
- `src/ml4t/backtest/strategy/base.py` - `Strategy` base class for user strategies

### Critical Infrastructure
- `src/ml4t/backtest/core/clock.py` - Multi-feed synchronization (176 lines)
- `src/ml4t/backtest/core/event.py` - Event system and types
- `src/ml4t/backtest/execution/fill_simulator.py` - Realistic fill modeling (196 lines)
- `src/ml4t/backtest/execution/order.py` - Order types and lifecycle (241 lines)
- `src/ml4t/backtest/portfolio/portfolio.py` - Portfolio state management

### Validation & Testing
- `tests/validation/test_cross_framework_alignment.py` - Framework comparison tests
- `tests/unit/test_clock_multi_feed.py` - Multi-feed synchronization validation
- `tests/integration/test_corporate_action_integration.py` - Corporate action workflows

## Patterns & Conventions

### Architecture Pattern
**Event-Driven + Hybrid Vectorized**

```python
# Typical event loop
for event in clock:
    if event.type == EventType.MARKET:
        strategy.on_market_data(event)    # User strategy logic
        broker.process_pending_orders()   # Execute orders
    elif event.type == EventType.FILL:
        strategy.on_fill(event)           # Fill notification
```

**Key Characteristics:**
- **Sequential processing**: Events processed in chronological order
- **No look-ahead bias**: Decisions only use past data
- **Vectorized hot paths**: Fill simulation, P&L calc use Numba/Polars
- **Pluggable components**: Swappable slippage, commission, impact models

### Code Organization
**Layered architecture:**
1. **Core** - Fundamental types, no business logic dependencies
2. **Data** - Data ingestion, depends only on Core
3. **Execution** - Order processing, depends on Core + Data
4. **Portfolio** - State tracking, depends on Execution
5. **Strategy** - User interface, depends on all layers
6. **Engine** - Orchestration, coordinates all layers

**Dependency Flow:** Core ← Data ← Execution ← Portfolio ← Strategy ← Engine

### Testing Strategy
**Three-tier approach:**
1. **Unit tests** - Isolated component testing (pytest)
2. **Integration tests** - Multi-component workflows
3. **Validation tests** - Cross-framework comparison (ml4t.backtest vs VectorBT vs Backtrader)

**Critical validation:**
- `test_cross_framework_alignment.py` - Proves ml4t.backtest matches VectorBT/Backtrader within 0.002% variance
- `test_clock_multi_feed.py` - 7/7 tests passing after fixing position sync bug

### Code Style
- **Type hints**: Mandatory, enforced by mypy --strict
- **Docstrings**: Google-style docstrings for public APIs
- **Line length**: 100 characters (ruff enforced)
- **Formatting**: ruff auto-format
- **Imports**: Absolute imports preferred

### Error Handling
- **Validation**: Pydantic for data validation at boundaries
- **Assertions**: For invariant checking in development
- **Exceptions**: Custom exceptions in `ml4t.backtest.core.types`
- **Logging**: Structured logging for debugging (not yet fully implemented)

## Dependencies

### Production Dependencies (Key)
```toml
python = ">=3.9,<3.14"
polars = "^1.15.0"        # High-performance DataFrames
numba = "^0.61.0"         # JIT compilation for hot paths
numpy = "^1.26.4"         # Numerical computing
pydantic = "^2.9.2"       # Data validation
pandas = "^2.2.3"         # Legacy compat (being phased out)
```

### Development Dependencies
```toml
pytest = "^8.3.3"         # Testing framework
pytest-cov = "^6.0.0"     # Coverage reporting
ruff = "^0.8.0"           # Linting and formatting
mypy = "^1.13.0"          # Type checking
ipython = "^8.29.0"       # Interactive development

# Framework validation
vectorbtpro = "^2.0.5"    # Cross-validation (VectorBT)
backtrader = "^1.9.78.123" # Cross-validation (Backtrader)
zipline-reloaded = "^3.0.4" # Cross-validation (Zipline)
```

### Internal Module Dependencies
**Core modules** (no internal deps):
- `ml4t.backtest.core.types`
- `ml4t.backtest.core.constants`
- `ml4t.backtest.core.event`

**Most depended-on modules:**
- `ml4t.backtest.core.types` - Used by all modules
- `ml4t.backtest.execution.broker` - Central execution hub
- `ml4t.backtest.portfolio.portfolio` - Portfolio state

**Circular dependency risk:**
- Broker ↔ Portfolio (resolved via dependency injection)

## Entry Points & Common Tasks

### Development Workflow
```bash
# Setup environment
cd /home/stefan/ml4t/software/backtest
source .venv/bin/activate  # or: uv sync

# Run tests
pytest tests/                               # All tests
pytest tests/unit/                          # Unit tests only
pytest tests/validation/                    # Cross-framework validation
pytest --cov=src/ml4t.backtest --cov-report=html # With coverage

# Code quality
ruff check src/                             # Lint
ruff format src/                            # Format
mypy src/                                   # Type check

# Run specific validation
pytest tests/validation/test_cross_framework_alignment.py::TestCrossFrameworkAlignment::test_frameworks_with_predefined_signals -v -s
```

### Building & Publishing
```bash
# Build package
uv build

# Install locally
pip install -e .

# Run example (if exists)
python examples/simple_ma_strategy.py

# Test imports in Python REPL
python -c "from ml4t.backtest import BacktestEngine; print('✓ Import successful')"
```

### Key Workflows

**Adding a New Feature:**
1. Write tests first (`tests/unit/test_new_feature.py`)
2. Implement in appropriate module (`src/ml4t.backtest/...`)
3. Add type hints and docstrings
4. Run tests: `pytest tests/unit/test_new_feature.py`
5. Check coverage: `pytest --cov=src/ml4t.backtest/module`
6. Update CLAUDE.md or memory files if architectural

**Debugging Execution Issues:**
1. Check clock synchronization: `tests/unit/test_clock_*.py`
2. Verify fill model: `tests/unit/test_fill_simulator.py`
3. Validate against VectorBT: `tests/validation/test_cross_framework_alignment.py`
4. Check broker position tracking: `broker.get_position()` API

**Validating Correctness:**
1. Run cross-framework validation: `pytest tests/validation/`
2. Expected: ml4t.backtest within 0.002% of VectorBT/Backtrader
3. If divergence > 1%, investigate fill model or order execution

## Recent Work & Current State

### Completed (Nov 2025)
- ✅ Corporate action integration (stock splits, dividends)
- ✅ 3-way cross-framework validation (ml4t.backtest, VectorBT, Backtrader)
- ✅ Signal-based validation framework (eliminates calculation variance)
- ✅ Multi-feed clock synchronization (7/7 tests passing)
- ✅ Broker position sync fix (dual portfolio tracking bug resolved)

### In Progress
- 🔨 ML signal integration architecture (design phase)
  - See `.claude/memory/ml_signal_architecture.md`
  - See `.claude/memory/multi_source_context_architecture.md`
- 🔨 Context-dependent trading logic (VIX filtering, regime switching)

### Known Issues
- ⚠️ Test coverage at 44% (target: 80%+)
- ⚠️ Zipline validation excluded (bundle data incompatibility)
- ⚠️ Bracket orders (OCO, OTO) not fully tested
- ⚠️ Market impact models need validation

### Next Milestones
1. **Phase 1a**: ML signal workflow (1 week)
   - Add signals dict to MarketData
   - Strategy helper methods (get_position, buy_percent)
   - Example ML strategy notebook

2. **Phase 1b**: Basic context support (1 week)
   - Context class for market-wide data (VIX, SPY)
   - Context-dependent logic ("if VIX > 30, don't trade")

3. **Phase 2**: Multi-asset optimization (2 weeks, if needed)
   - Memory-efficient shared context
   - 500-stock universe testing

## Focus Areas for Development

### High Priority
1. **ML Signal Integration** - Core feature for modern strategies
2. **Test Coverage** - Increase from 44% to 80%+
3. **Performance Optimization** - Numba compilation of hot paths
4. **Documentation** - User guides, API docs, examples

### Medium Priority
5. **Live Trading Adapters** - LiveDataFeed, LiveBroker interfaces
6. **Advanced Order Types** - Iceberg, TWAP, VWAP
7. **Risk Management** - Portfolio-level risk constraints
8. **Reporting** - HTML/PDF tearsheets, trade analysis

### Low Priority (Post-1.0)
9. **Multi-asset strategies** - Pairs trading, stat arb
10. **Options support** - Greeks, volatility strategies
11. **GPU acceleration** - Cupy/CUDA for massive backtests

## Architecture Decisions (Recent)

### AD-001: Exclude Zipline from Signal-Based Validation
**Date**: 2025-11-14
**Status**: Accepted
**Context**: Zipline's `run_algorithm(bundle='quandl')` fetches its own price data (~4.3x different from test DataFrame)
**Decision**: Exclude Zipline from signal-based validation, use 3-way comparison (ml4t.backtest, VectorBT, Backtrader)
**Consequences**: Can't validate against Zipline with custom data; acceptable trade-off

### AD-002: Hybrid Context Approach (Single vs Multi-Asset)
**Date**: 2025-11-14
**Status**: Proposed
**Context**: Single-asset strategies can embed context in signals; multi-asset needs shared context for memory efficiency
**Decision**: Support both patterns - join for single-asset, separate Context object for multi-asset
**Consequences**: Slightly more complex API, but 500x memory savings for large universes

### AD-003: Signals Computed Outside Engine
**Date**: 2025-11-14
**Status**: Proposed
**Context**: ML inference is expensive; users may use external services
**Decision**: Users pre-compute signals; engine just executes
**Consequences**: Clean separation of concerns; cannot adapt signals based on execution

## Quick Reference: File Locations

**Need to understand event loop?**
→ `src/ml4t/backtest/engine.py` (main orchestration)
→ `src/ml4t/backtest/core/clock.py` (multi-feed sync)

**Need to understand order execution?**
→ `src/ml4t/backtest/execution/broker.py` (order routing, position tracking)
→ `src/ml4t/backtest/execution/fill_simulator.py` (fill model)

**Need to add a new strategy?**
→ `src/ml4t/backtest/strategy/base.py` (base class to inherit)
→ `examples/` (reference implementations)

**Need to validate correctness?**
→ `tests/validation/test_cross_framework_alignment.py` (framework comparison)

**Need to understand architecture decisions?**
→ `.claude/memory/` (design documents, decisions, conventions)

## Import Examples

**Basic usage:**
```python
from ml4t.backtest import BacktestEngine, Strategy, DataFeed
from ml4t.backtest.execution import SimulationBroker
from ml4t.backtest.core import Clock, Event
```

**Advanced features:**
```python
from ml4t.backtest.execution import (
    Order, MarketOrder, LimitOrder,
    FixedSlippage, PercentageCommission
)
from ml4t.backtest.portfolio import Portfolio, Position
from ml4t.backtest.reporting import Reporter
```

---

*This project map is automatically imported into CLAUDE.md and provides persistent context for development sessions.*
