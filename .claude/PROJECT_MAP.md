# Project Map: ml4t.backtest Backtesting Library

*Generated: 2025-11-15T07:00:00Z*
*Last Updated: 2025-12-26T10:50:00Z*

## Quick Overview
- **Type**: Python library for event-driven backtesting (part of ml4t namespace)
- **Primary Language**: Python 3.11+
- **Package Name**: `ml4t-backtest` (import as `ml4t.backtest`)
- **Version**: 0.2.0
- **Frameworks**: Polars (data), Pydantic (validation), Numba (performance)
- **Purpose**: Institutional-grade event-driven backtesting engine for quantitative trading
- **Location**: `/home/stefan/ml4t/software/backtest/`
- **Status**: Beta (474 tests, 73% coverage, all validation scenarios passing)

## Architecture Overview

ml4t.backtest is a **minimal event-driven backtesting engine** with institutional-grade execution fidelity.

**Key Design Principles:**
1. **Event-Driven**: Iterates through market events sequentially (no look-ahead bias)
2. **Point-in-Time Correctness**: All decisions use only data available at decision time
3. **Realistic Execution**: Simulates order fills, slippage, commissions, market impact
4. **Account Policies**: Pluggable cash vs margin account handling with gatekeeper validation
5. **Exit-First Processing**: Exit orders processed before entries to free capital
6. **Risk Management**: Comprehensive position and portfolio risk rules

## Directory Structure

### Core Source Code (`src/ml4t/backtest/` - 34 files, ~7,700 lines)

```
src/ml4t/backtest/
├── __init__.py          # Package exports (80+ public symbols)
├── engine.py            # Engine class (event loop orchestration)
├── broker.py            # Broker (order execution, position tracking) - 37KB
├── datafeed.py          # DataFeed (price + signal iteration)
├── strategy.py          # Strategy base class (on_data callback)
├── types.py             # Core types (Order, Position, Fill, Trade, ContractSpec)
├── models.py            # Commission/slippage models
├── config.py            # BacktestConfig and presets
├── calendar.py          # Trading calendar integration (pandas_market_calendars)
├── analysis.py          # BacktestAnalyzer, TradeStatistics
│
├── accounting/          # Account policies (~1,187 lines)
│   ├── __init__.py
│   ├── account.py       # AccountState class
│   ├── policy.py        # CashAccountPolicy, MarginAccountPolicy
│   └── gatekeeper.py    # Order validation (capital, margin checks)
│
├── analytics/           # Performance metrics (~549 lines)
│   ├── __init__.py
│   ├── equity.py        # EquityCurve analysis
│   ├── metrics.py       # Sharpe, Sortino, Calmar, drawdown
│   └── trades.py        # TradeAnalyzer
│
├── execution/           # Advanced execution (~820 lines)
│   ├── __init__.py
│   ├── impact.py        # MarketImpactModel (Linear, SquareRoot)
│   ├── limits.py        # ExecutionLimits, VolumeParticipationLimit
│   ├── rebalancer.py    # TargetWeightExecutor for portfolio rebalancing
│   └── result.py        # ExecutionResult
│
├── risk/                # Risk management (~1,500+ lines)
│   ├── __init__.py
│   ├── types.py         # RiskContext, RiskDecision, RuleResult
│   ├── position/        # Position-level rules
│   │   ├── static.py    # StopLoss, TakeProfit, TrailingStop
│   │   ├── dynamic.py   # VolatilityStop, BreakEvenStop, TimeStop
│   │   ├── signal.py    # SignalBasedExit
│   │   ├── composite.py # CompositeRule, ConditionalRule, SequentialRule
│   │   └── protocol.py  # PositionRule protocol
│   └── portfolio/       # Portfolio-level limits
│       ├── limits.py    # MaxPositions, MaxExposure, MaxDrawdown
│       └── manager.py   # PortfolioRiskManager
│
└── presets/             # Configuration presets
    └── *.yaml           # Pre-built config files
```

### Test Organization (`tests/` - 23 files, ~7,626 lines)

```
tests/
├── __init__.py
├── test_core.py         # Main engine tests
├── test_broker.py       # Broker-specific tests
├── test_calendar.py     # Calendar function tests
│
├── accounting/          # Account system tests
│   ├── test_account_state.py
│   ├── test_cash_account_policy.py
│   ├── test_margin_account_policy.py
│   ├── test_gatekeeper.py
│   └── test_position.py
│
├── execution/           # Execution model tests
│   ├── test_impact.py
│   └── test_rebalancer.py
│
└── risk/                # Risk management tests
    ├── test_static_rules.py
    ├── test_dynamic_rules.py
    ├── test_signal_rules.py
    ├── test_composite_rules.py
    ├── test_portfolio_limits.py
    └── test_portfolio_manager.py
```

### Validation (`validation/` - Cross-framework validation)

```
validation/
├── README.md                 # Per-framework validation strategy
├── benchmark_suite.py        # Unified benchmark runner
├── risk_validation.py        # Risk system validation
├── vectorbt_pro/             # VectorBT Pro validation (internal)
├── vectorbt_oss/             # VectorBT OSS validation
├── backtrader/               # Backtrader validation
└── zipline/                  # Zipline validation
```

**Validation Strategy**: Per-framework in isolated venvs (NOT unified pytest).

## Key Files

### Application Entry Points
- `src/ml4t/backtest/engine.py` - Main `Engine` class (event loop)
- `src/ml4t/backtest/__init__.py` - Package exports (80+ public symbols)
- `src/ml4t/backtest/broker.py` - `Broker` class (~37KB, core order execution)

### Core Components
- `src/ml4t/backtest/types.py` - Order, Position, Fill, Trade, ContractSpec dataclasses
- `src/ml4t/backtest/models.py` - Commission/slippage models
- `src/ml4t/backtest/datafeed.py` - DataFeed iterator
- `src/ml4t/backtest/config.py` - BacktestConfig with presets

### Account System
- `src/ml4t/backtest/accounting/policy.py` - Cash vs Margin account policies
- `src/ml4t/backtest/accounting/gatekeeper.py` - Order validation

### Risk Management
- `src/ml4t/backtest/risk/position/static.py` - StopLoss, TakeProfit, TrailingStop
- `src/ml4t/backtest/risk/position/dynamic.py` - VolatilityStop, BreakEvenStop
- `src/ml4t/backtest/risk/portfolio/manager.py` - Portfolio-level risk

## Patterns & Conventions

### Event Loop Pattern

```python
# Engine.run() - main event loop
for timestamp, assets_data, context in feed:
    broker._update_time(timestamp, prices, opens, highs, lows, volumes, signals)

    # Process pending exits (NEXT_BAR_OPEN mode)
    broker._process_pending_exits()

    # Evaluate position rules (stops, trails, etc.)
    broker.evaluate_position_rules()

    if execution_mode == NEXT_BAR:
        broker._process_orders(use_open=True)
        strategy.on_data(timestamp, assets_data, context, broker)
    else:  # SAME_BAR
        broker._process_orders()
        strategy.on_data(timestamp, assets_data, context, broker)
        broker._process_orders()
```

### Strategy Interface

```python
class Strategy(ABC):
    @abstractmethod
    def on_data(self, timestamp, data, context, broker) -> None:
        pass

    def on_start(self, broker) -> None: pass
    def on_end(self, broker) -> None: pass
```

### Order Execution (Exit-First)

```python
# Broker._process_orders() - critical sequencing
# 1. Process exit orders first (frees capital)
# 2. Update account equity
# 3. Process entry orders (validated via Gatekeeper)
```

### Code Style
- **Python**: 3.11+ (modern type hints, no `from __future__ import annotations` needed)
- **Type hints**: Mandatory
- **Line length**: 100 characters (ruff enforced)
- **Testing**: pytest with 474 tests (73% coverage)

## Dependencies

### Production Dependencies
```toml
python = ">=3.11"
polars = ">=0.20.0"        # DataFrames
pandas = ">=2.0.0"         # Legacy compat
numpy = ">=1.24.0"         # Numerical computing
pyarrow = ">=14.0.0"       # Data interchange
pydantic = ">=1.10.0,<3.0.0"  # Configuration validation
numba = ">=0.57.0"         # JIT compilation
sortedcontainers = ">=2.4.0"  # Efficient priority queue
tables = ">=3.8.0"         # HDF5 support
```

### Development Dependencies
```toml
pytest = "^8.3.3"
ruff = "^0.8.0"
mypy = "^1.13.0"
```

## Virtual Environments

**Only 3 venvs needed:**

```
.venv               # Main development (Python 3.12)
.venv-validation    # VBT OSS + Backtrader + Zipline (all open-source together)
.venv-vectorbt-pro  # VectorBT Pro only (commercial, cannot coexist with OSS)
```

**VectorBT OSS/Pro Conflict**: Both register pandas `.vbt` accessor. Cannot coexist.

## Entry Points & Common Tasks

### Development Workflow
```bash
cd /home/stefan/ml4t/software/backtest
source .venv/bin/activate

# Run tests
pytest tests/ -q                    # All tests (474 pass)

# Code quality
ruff check src/                     # Lint
ruff format src/                    # Format

# Test imports
python -c "from ml4t.backtest import Engine; print('✓')"
```

### VectorBT Pro Validation
```bash
source .venv-vectorbt-pro/bin/activate
cd validation/vectorbt_pro
python scenario_01_long_only.py
```

### Backtrader Validation
```bash
source .venv-validation/bin/activate
cd validation/backtrader
python scenario_01_long_only.py
```

## Recent Work & Current State (Dec 2025)

### Completed
- ✅ Major repository cleanup (739K → 7.7K lines)
- ✅ Clean modular structure
- ✅ Account system (cash/margin policies)
- ✅ Exit-first order processing
- ✅ Risk management system (position + portfolio rules)
- ✅ Execution model (volume limits, market impact)
- ✅ Portfolio rebalancing (TargetWeightExecutor)
- ✅ Ruff passing, 474 tests passing

### Completed Validation
- ✅ VectorBT Pro: All scenarios pass (EXACT MATCH)
- ✅ VectorBT OSS: All scenarios pass (EXACT MATCH)
- ✅ Backtrader: All scenarios pass (EXACT MATCH)
- ✅ Zipline: All scenarios pass (within tolerance)

## Architecture Decisions

### AD-004: Validation Environment Strategy (Nov 2025)
**Status**: Accepted
**Decision**: Use 2 validation venvs: .venv-validation (OSS+BT+ZL), .venv-vectorbt-pro
**Reason**: VectorBT OSS/Pro cannot coexist (pandas .vbt accessor conflict)

### AD-005: Python 3.11+ Target (Nov 2025)
**Status**: Accepted
**Decision**: Target Python 3.11+ only
**Benefit**: Modern syntax, cleaner code

### AD-006: Exit-First Order Processing
**Status**: Accepted
**Decision**: Always process exit orders before entries
**Reason**: Frees capital for new positions, matches real broker behavior

## Import Examples

```python
# Core usage
from ml4t.backtest import Engine, Strategy, DataFeed, Broker

# Types
from ml4t.backtest import Order, Position, Fill, Trade
from ml4t.backtest import OrderType, OrderSide, OrderStatus, ExecutionMode

# Models
from ml4t.backtest import (
    PercentageCommission, PerShareCommission, NoCommission,
    FixedSlippage, PercentageSlippage, NoSlippage,
)

# Configuration
from ml4t.backtest import BacktestConfig, run_backtest

# Analytics
from ml4t.backtest import EquityCurve, TradeAnalyzer
from ml4t.backtest import sharpe_ratio, max_drawdown, cagr

# Calendar
from ml4t.backtest import get_calendar, get_trading_days, is_trading_day

# Execution model
from ml4t.backtest import (
    ExecutionLimits, VolumeParticipationLimit,
    MarketImpactModel, LinearImpact, SquareRootImpact,
    RebalanceConfig, TargetWeightExecutor,
)
```

---

*This project map reflects the current state as of Dec 2025.*
