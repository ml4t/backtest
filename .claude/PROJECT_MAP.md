# Project Map: ml4t.backtest Backtesting Library

*Generated: 2025-11-15T07:00:00Z*
*Last Updated: 2025-11-22T14:00:00Z*

## Quick Overview
- **Type**: Python library for event-driven backtesting (part of ml4t namespace)
- **Primary Language**: Python 3.11+
- **Package Name**: `ml4t-backtest` (import as `ml4t.backtest`)
- **Version**: 0.2.0
- **Frameworks**: Polars (data)
- **Purpose**: Minimal event-driven backtesting engine for quantitative trading strategies
- **Location**: `/home/stefan/ml4t/software/backtest/`
- **Status**: Beta, post-cleanup (99.2% code reduction completed Nov 2025)

## Architecture Overview

ml4t.backtest is a **minimal event-driven backtesting engine** with clean, focused implementation.

**Key Design Principles:**
1. **Minimal Core**: ~2,800 lines of source code (down from 739K after cleanup)
2. **Event-Driven**: Iterates through market events sequentially (no look-ahead bias)
3. **Point-in-Time Correctness**: All decisions use only data available at decision time
4. **Realistic Execution**: Simulates order fills, slippage, commissions
5. **Account Policies**: Pluggable cash vs margin account handling

## Directory Structure

### Core Source Code (`src/ml4t/backtest/` - 13 files, ~2,788 lines)

```
src/ml4t/backtest/
├── __init__.py          # Package exports, version
├── engine.py            # Engine class (event loop orchestration)
├── broker.py            # Broker (order execution, position tracking)
├── datafeed.py          # DataFeed (price + signal iteration)
├── strategy.py          # Strategy base class (on_data callback)
├── types.py             # Core types (Order, Position, Fill, Trade)
├── models.py            # Commission/slippage models
├── config.py            # BacktestConfig and presets
└── accounting/          # Account policies
    ├── __init__.py
    ├── account.py       # AccountState class
    ├── policy.py        # CashAccountPolicy, MarginAccountPolicy
    ├── gatekeeper.py    # Order validation
    └── models.py        # Position dataclass
```

### Test Organization (`tests/` - 8 files, ~2,704 lines)

```
tests/
├── __init__.py
├── test_core.py         # Main engine tests (154 tests)
└── accounting/
    ├── __init__.py
    ├── test_account_state.py
    ├── test_cash_account_policy.py
    ├── test_margin_account_policy.py
    ├── test_gatekeeper.py
    └── test_position.py
```

### Validation (`validation/` - NEW)

```
validation/
├── README.md            # Per-framework validation strategy
├── vectorbt_pro/        # VectorBT Pro validation (internal only)
└── backtrader/          # Backtrader validation (open source)
```

**Validation Strategy**: Per-framework in isolated venvs, NOT unified pytest.

## Key Files

### Application Entry Points
- `src/ml4t/backtest/engine.py` - Main `Engine` class (event loop)
- `src/ml4t/backtest/__init__.py` - Package exports (40+ public symbols)
- `src/ml4t/backtest/broker.py` - `Broker` class (~480 lines)

### Core Components
- `src/ml4t/backtest/types.py` - Order, Position, Fill, Trade dataclasses
- `src/ml4t/backtest/models.py` - Commission/slippage models
- `src/ml4t/backtest/datafeed.py` - DataFeed iterator
- `src/ml4t/backtest/config.py` - BacktestConfig with presets

### Account System
- `src/ml4t/backtest/accounting/policy.py` - Cash vs Margin account policies
- `src/ml4t/backtest/accounting/gatekeeper.py` - Order validation

## Patterns & Conventions

### Event Loop Pattern

```python
# Engine.run() - main event loop
for timestamp, assets_data, context in feed:
    broker._update_time(timestamp, prices, opens, volumes, signals)

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
- **Python**: 3.11+ (no `from __future__ import annotations` needed)
- **Type hints**: Mandatory
- **Line length**: 100 characters (ruff enforced)
- **Testing**: pytest with ~154 tests passing

## Dependencies

### Production Dependencies
```toml
python = ">=3.11,<3.14"
polars = "^1.15.0"        # DataFrames
numpy = "^1.26.4"         # Numerical computing
pydantic = "^2.9.2"       # Data validation
pandas = "^2.2.3"         # Legacy compat
```

### Development Dependencies
```toml
pytest = "^8.3.3"
ruff = "^0.8.0"
mypy = "^1.13.0"          # Note: relaxed config, 16 errors remain
```

## Virtual Environments

**CRITICAL - Use the right venv for each task:**

```
.venv               # Main development (Python 3.12)
.venv-backtrader    # Backtrader validation only
.venv-vectorbt      # VectorBT OSS only
.venv-vectorbt-pro  # VectorBT Pro (internal, commercial)
.venv-zipline       # Zipline (low priority)
.venv-validation    # DEPRECATED - has dependency conflicts
```

## Entry Points & Common Tasks

### Development Workflow
```bash
cd /home/stefan/ml4t/software/backtest
source .venv/bin/activate

# Run tests
pytest tests/ -q                    # All tests (154 pass in ~0.5s)

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
python run_scenarios.py
```

### Backtrader Validation
```bash
source .venv-backtrader/bin/activate
cd validation/backtrader
python run_scenarios.py
```

## Recent Work & Current State (Nov 2025)

### Completed
- ✅ Major repository cleanup (739K → 5.5K lines, 99.2% reduction)
- ✅ Deleted `archive/`, `resources/`, chaotic `tests/validation/`
- ✅ Clean flat module structure (no deep nesting)
- ✅ Account system (cash/margin policies)
- ✅ Exit-first order processing
- ✅ Ruff passing, 154 tests passing

### Known Issues
- ⚠️ Mypy: 16 errors remain (relaxed config, not blocking)
- ⚠️ VectorBT OSS/Pro conflict in same env
- ⚠️ Zipline bundle issues (excluded from validation)

### Next Steps
1. Create VectorBT Pro validation scripts
2. Create Backtrader validation scripts
3. Fix remaining mypy errors
4. Document configuration presets

## Architecture Decisions

### AD-004: Per-Framework Validation (Nov 2025)
**Status**: Accepted
**Context**: Two days wasted on dependency conflicts in unified pytest suite
**Decision**: Validate each framework in isolated venv with standalone scripts
**Consequence**: No unified test runner, but no dependency hell

### AD-005: Python 3.11+ Target (Nov 2025)
**Status**: Accepted
**Context**: No need for `from __future__ import annotations`
**Decision**: Target Python 3.11+ only
**Consequence**: Cleaner code, modern syntax

### AD-006: Relaxed Mypy (Nov 2025)
**Status**: Accepted
**Context**: 16 real type issues remain, not critical
**Decision**: Set `strict = false` in pyproject.toml
**Consequence**: Can proceed with development, fix types later

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
```

---

*This project map reflects the post-cleanup state as of Nov 2025.*
