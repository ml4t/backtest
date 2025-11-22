# CLAUDE.md - ml4t.backtest Development Guidelines

## Project Understanding
@.claude/PROJECT_MAP.md

## Context & Location

**You are in**: `/home/stefan/ml4t/software/backtest/` (ml4t.backtest library)
**Parent directory**: `/home/stefan/ml4t/software/` (multi-library coordination)
**Sibling libraries**: `../data/`, `../features/`, `../evaluation/`
**Integration projects**: `../projects/` (use ml4t.backtest in workflows)

**Division of Labor**: See `../.claude/memory/division_of_labor.md` for when to work here vs parent directory.

**Work in THIS directory for**:
- Event-driven execution engine improvements
- Order types and execution logic
- Position tracking and portfolio management
- Broker simulation and fill models
- Unit tests for ml4t.backtest functionality
- API improvements and documentation

**Work in PARENT directory for**:
- Multi-library workflows using ml4t.backtest + others
- Integration testing in `../projects/`
- Validation studies (e.g., VectorBT exact matching)

## Vision & Goals

ml4t.backtest provides an event-driven backtesting engine with institutional-grade execution fidelity.

**Core Mission**: Replicate real trading conditions with point-in-time correctness and realistic execution.

## Key Architecture

- **Event-Driven**: Market, signal, order, fill events
- **Point-in-Time Safety**: No look-ahead bias
- **Vectorized Hybrid**: Event-driven control + vectorized execution
- **Pluggable Components**: Broker, commission, slippage models
- **Performance**: 100k+ events/second

## Critical Known Issues

### Position Sync Issue (Fixed)
**Problem**: Dual position tracking
- `broker.position_tracker`: Updated on fills (source of truth)
- `broker.portfolio.positions`: Separate object (stale after fills)

**Solution**: Strategies query `broker.get_position()` instead of `portfolio.positions.get()`.

See `../projects/crypto_futures/` for validation work.

### VectorBT OSS/Pro Conflict (Critical)
**Problem**: VectorBT OSS and Pro CANNOT coexist in the same environment.
Both register a pandas `.vbt` accessor which collide. Running OSS imports before Pro causes:
```
AttributeError: 'OHLCVDFAccessor' object has no attribute 'has_ohlc'
```

**Solution**: Use separate virtual environments:
- `.venv` - Main development with VectorBT OSS only (validation tests work here)
- `.venv-pro` - VectorBT Pro only for Pro-specific tests (private, commercial license)

**Test conftest.py behavior**: Don't import VectorBT OSS if Pro is available to avoid accessor conflict.

### Python 3.12 Traceback Formatting Bug
**Problem**: `traceback.format_exc()` can crash with `RuntimeError: generator raised StopIteration` when formatting certain exception chains (especially from Zipline).

**Solution**: Wrap traceback calls in try/except:
```python
try:
    result.errors.append(traceback.format_exc())
except RuntimeError:
    result.errors.append(f"Exception type: {type(e).__name__}")
```

### Zipline Bundle/Symbol Resolution (Excluded)
**Problem**: Zipline `run_algorithm()` has environment-specific bundle/symbol issues.
The bundle may have AAPL registered but `symbol('AAPL')` fails at runtime.

**Decision**: Zipline excluded from cross-framework validation (see AD-001 in docs).
Test file `test_zipline_adapter.py` is fully skipped.

## Development Standards

- **Python**: 3.9+ with type hints
- **Testing**: pytest with comprehensive coverage
- **Linting**: ruff (100 char line length)
- **Performance**: Numba JIT for hot paths
- **Quality**: Pre-commit hooks
- **Correctness**: No look-ahead bias, point-in-time accuracy

## Projects Awareness

`../projects/` uses ml4t.backtest for:
- Strategy backtesting
- VectorBT replication and validation
- Execution fidelity testing
- Performance benchmarking

**Integration Work**: The VectorBT exact matching study (`../projects/crypto_futures/`) validates ml4t.backtest produces identical results to VectorBT Pro.

Coordinate breaking changes through parent `.claude/`.

## Key Validation Requirements

When modifying execution logic:
1. Ensure point-in-time correctness (no look-ahead)
2. Test same-bar re-entry scenarios
3. Validate position sync after fills
4. Check against VectorBT reference results
5. Verify fill prices are within OHLC bounds

## 🚨 CRITICAL: Framework Source Code Availability

**ALL BENCHMARK FRAMEWORKS HAVE COMPLETE SOURCE CODE LOCALLY AVAILABLE**

**Locations**:
- ✅ **Zipline-reloaded**: `resources/zipline-reloaded-main/src/zipline/`
- ✅ **Backtrader**: `resources/backtrader-master/backtrader/`
- ✅ **VectorBT OSS**: `resources/vectorbt/vectorbt/`
- ✅ **VectorBT Pro**: `resources/vectorbt.pro-main/vectorbtpro/`

### Zero Tolerance Policy for "I Don't Know"

**NEVER ACCEPTABLE**:
- ❌ "Unclear how VectorBT executes fills"
- ❌ "Need to research Backtrader's order logic"
- ❌ "Not sure why Zipline produces different results"

**ALWAYS REQUIRED**:
1. Read the actual source code (`Read resources/framework/relevant_file.py`)
2. Cite specific files and line numbers
3. Explain the exact implementation difference with code evidence
4. Document findings in validation report

### Investigation Protocol (Mandatory)

When frameworks produce different results:

```bash
# 1. Search for relevant code
grep -rn "fill.*price\|execution" resources/vectorbt/vectorbt/portfolio/
grep -rn "fill.*price\|execution" resources/backtrader-master/backtrader/brokers/

# 2. Read the implementation
Read resources/vectorbt/vectorbt/portfolio/base.py
Read resources/backtrader-master/backtrader/brokers/bbroker.py

# 3. Compare and cite specific lines
# Example: "VectorBT fills at close (base.py:3245),
#           Backtrader fills at next open (bbroker.py:467)"

# 4. Use Serena for semantic search (if available)
mcp__serena__find_symbol("from_signals", "resources/vectorbt/")
```

**This is not optional. This is mandatory for all validation work.**

### Key Framework Files

**VectorBT OSS/Pro**:
- Portfolio API: `portfolio/base.py` (from_signals, from_orders, from_holding)
- Numba execution: `portfolio/nb/from_signals.py` (vectorized fill logic)
- Orders: `portfolio/orders.py` (order types, execution)

**Backtrader**:
- Broker: `brokers/bbroker.py` (order execution, COO/COC, fills)
- Orders: `order.py` (order types, status)
- Cerebro: `cerebro.py` (engine orchestration)

**Zipline**:
- Execution: `finance/execution.py` (order placement, fills)
- Commission: `finance/commission.py` (PerShare, PerTrade, PerDollar)
- Slippage: `finance/slippage.py` (FixedSlippage, VolumeShareSlippage)

## References

- Event-driven architecture patterns
- VectorBT Pro fill model documentation
- Position tracking best practices
- Framework source code in `resources/` (see above)
