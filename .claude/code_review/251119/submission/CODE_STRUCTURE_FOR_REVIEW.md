# Code Structure for External Review

## Summary Statistics

- **Total Lines**: 22,709
- **Total Files**: 64 Python files
- **Core Files** (essential): ~8,000 lines, 15-20 files
- **Bloat** (potentially unnecessary): ~14,000 lines, 40+ files

## Core Architecture Files (KEEP)

### Engine & Orchestration
- `engine.py` (639 lines) - **CRITICAL** - Main event loop, needs refactoring for batch mode
- `core/clock.py` (460 lines) - Event synchronization across feeds
- `core/event.py` (~200 lines) - MarketEvent, FillEvent, OrderEvent
- `core/types.py` (~150 lines) - Type definitions
- `core/constants.py` (~50 lines) - Enums

**Subtotal: ~1,500 lines**

### Data Layer
- `data/feed.py` (~150 lines) - DataFeed base class
- `data/multi_symbol_feed.py` (201 lines) - **CRITICAL** - Multi-asset data feed (just added)
- `data/schemas.py` (~100 lines) - Schema validation

**Subtotal: ~450 lines**

### Execution
- `execution/broker.py` (1379 lines) - **CRITICAL** - Needs refactoring for batch fills
- `execution/fill_simulator.py` (663 lines) - Realistic fill modeling
- `execution/order.py` (478 lines) - Order types and lifecycle
- `execution/slippage.py` (647 lines) - Slippage models
- `execution/commission.py` (~200 lines) - Commission models

**Subtotal: ~3,367 lines**

### Portfolio Tracking
- `portfolio/portfolio.py` (373 lines) - Position and P&L tracking
- `portfolio/core.py` (~200 lines) - Position class
- `portfolio/state.py` (~150 lines) - Portfolio state

**Subtotal: ~723 lines**

### Strategy Interface
- `strategy/base.py` (851 lines) - **NEEDS FIXING** - Broken API (buy_percent, etc.)

**Subtotal: ~851 lines**

### Risk Management (Optional but Valuable)
- `risk/manager.py` (621 lines) - Rule orchestration
- `risk/context.py` (433 lines) - Risk context construction
- `risk/decision.py` (453 lines) - Decision merging
- `risk/rule.py` (~100 lines) - Base rule interface

**Subtotal: ~1,607 lines**

**CORE TOTAL: ~8,500 lines (37% of codebase)**

---

## Potentially Dead Code (REVIEW/REMOVE)

### Strategy Adapters (Unnecessary Complexity)
- `strategy/adapters.py` (371 lines) - VectorBT/Backtrader compatibility layer
- `strategy/crypto_basis_adapter.py` (485 lines) - Specific crypto strategy
- `strategy/spy_order_flow_adapter.py` (465 lines) - Specific SPY strategy

**Subtotal: ~1,321 lines (not needed for core engine)**

### Data Layer Bloat
- `data/polars_feed.py` (618 lines) - **DEPRECATED** - Single-asset only, replaced by MultiSymbolDataFeed
- `data/validation.py` (848 lines) - Schema validation layer (may be overkill)
- `data/feature_provider.py` (~300 lines) - Feature provider abstraction (may be unnecessary)
- `data/asset_registry.py` (~200 lines) - Asset metadata (may be unnecessary)

**Subtotal: ~1,966 lines**

### Reporting (Nice-to-Have, Not Core)
- `reporting/html.py` (599 lines) - HTML report generation
- `reporting/visualizations.py` (707 lines) - Plotly charts
- `reporting/trade_analysis.py` (385 lines) - Trade analysis
- `reporting/parquet.py` (402 lines) - Parquet export
- `reporting/trade_schema.py` (515 lines) - Trade schema
- `reporting/reporter.py` (~200 lines) - Reporter interface
- `reporting/base.py` (~100 lines) - Base classes

**Subtotal: ~2,908 lines (nice for demos, not essential)**

### Risk Rules (Specific Implementations)
- `risk/rules/volatility_scaled.py` (438 lines)
- `risk/rules/portfolio_constraints.py` (449 lines)
- `risk/rules/regime_dependent.py` (371 lines)
- `risk/rules/dynamic_trailing.py` (~300 lines)
- `risk/rules/time_based.py` (~200 lines)
- `risk/rules/price_based.py` (~150 lines)

**Subtotal: ~1,908 lines (examples, not core architecture)**

### Execution Extras
- `execution/corporate_actions.py` (840 lines) - Stock splits, dividends (complex, probably premature)
- `execution/bracket_manager.py` (~300 lines) - Bracket orders (OCO, OTO - advanced)
- `execution/market_impact.py` (507 lines) - Market impact models (advanced)
- `execution/trade_tracker.py` (637 lines) - Trade reconciliation

**Subtotal: ~2,284 lines**

### Configuration & Utilities
- `config.py` (510 lines) - Configuration system (unnecessary?)
- `core/assets.py` (490 lines) - Asset class definitions (probably unnecessary)
- `core/precision.py` (~200 lines) - Decimal precision management
- `core/context.py` (~150 lines) - Context cache

**Subtotal: ~1,350 lines**

### Portfolio Extras
- `portfolio/analytics.py` (~300 lines) - Performance analytics
- `portfolio/margin.py` (~200 lines) - Margin calculations

**Subtotal: ~500 lines**

### Package Files
- `__init__.py` files (~20 files × 30 lines avg = 600 lines)

**Subtotal: ~600 lines**

**BLOAT TOTAL: ~14,209 lines (63% of codebase)**

---

## Recommended Cleanup Actions

### Phase 1: Immediate Removal (No Impact)
**Remove these files entirely:**
- `strategy/adapters.py`, `strategy/crypto_basis_adapter.py`, `strategy/spy_order_flow_adapter.py` (1,321 lines)
- `data/polars_feed.py` (618 lines) - Deprecated
- `reporting/*` (2,908 lines) - Move to separate package later
- `config.py` (510 lines) - Unnecessary complexity

**Total removed: ~5,357 lines (24% reduction)**

### Phase 2: Defer to Later (Useful but Not Now)
**Keep but mark as "future work":**
- `execution/corporate_actions.py` (840 lines) - Stock splits needed eventually
- `execution/bracket_manager.py` (300 lines) - Advanced order types
- `execution/market_impact.py` (507 lines) - Realism enhancement
- `risk/rules/*` (1,908 lines) - Specific rule implementations

**Deferred: ~3,555 lines**

### Phase 3: Consider Simplifying
**Review and possibly simplify:**
- `data/validation.py` (848 lines) - Do we need all this validation?
- `data/feature_provider.py` (300 lines) - Could be simpler
- `core/assets.py` (490 lines) - Do we need all asset classes?
- `execution/trade_tracker.py` (637 lines) - Reconciliation overkill?

**Potential simplification: ~2,275 lines**

---

## Post-Cleanup Target

**After Phase 1 cleanup:**
- Total lines: ~17,352 (down from 22,709)
- Core files: ~8,500 lines (49%)
- Supporting files: ~8,852 lines (51%)

**After Phase 2 (defer to later):**
- Active development: ~13,797 lines
- Archived/future: ~3,555 lines

**After Phase 3 (simplification):**
- Target: <12,000 lines for core engine
- Stretch goal: <8,000 lines (core only)

---

## Files to Include in RepoMix Submission

### Critical for Review (Must Read)
1. `EXTERNAL_REVIEW_PROMPT.md` - This document
2. `ARCHITECTURE_DIAGNOSIS.md` - Full technical analysis
3. `HONEST_ENGINE_COMPARISON.md` - Performance comparison
4. `engine.py` - Main orchestration
5. `execution/broker.py` - Order execution
6. `strategy/base.py` - Strategy interface
7. `data/multi_symbol_feed.py` - Data feed
8. `examples/integrated/top25_ml_strategy_complete.py` - Working manual loop
9. `examples/integrated/top25_using_engine.py` - Broken engine version

### Supporting Context
- Core files listed above (~8,500 lines)
- `README.md` if it exists
- `pyproject.toml` for dependencies

### Exclude from RepoMix
- `tests/` directory (too large, not needed for architecture review)
- `.venv/`, `__pycache__/`, build artifacts
- Data files (`.parquet`, `.csv`)
- Documentation builds
- Bloat files from Phase 1 above

---

## Key Questions for Reviewer

1. **Is the core architecture (8,500 lines) salvageable?**
2. **What should be deleted vs refactored vs kept?**
3. **How to fix the broker for batch fills?**
4. **Correct API for portfolio rebalancing?**
5. **Path to 10x performance improvement?**

---

## Submission Command

```bash
cd /home/stefan/ml4t/software/backtest

# Run RepoMix to generate XML
repomix --output repomix-output.xml

# This will create a compressed XML file containing:
# - All source code (excluding tests, data, builds per .repomixignore)
# - Documentation files (EXTERNAL_REVIEW_PROMPT.md, etc.)
# - File structure
# - Ready for LLM review
```

The XML file can be uploaded to an external code review service or shared with an expert via Claude/ChatGPT with a large context window.
