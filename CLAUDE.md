# CLAUDE.md - ml4t.backtest Development Guidelines

## IRONCLAD RULES

**NEVER REMOVE ANY DATA OR OTHER FILES THAT HAVE NOT BEEN PREVIOUSLY GIT COMMITTED WITHOUT EXPRESS PERMISSION.**

This includes downloaded data, generated files, caches, or any user content. Data costs money and cannot be recovered.

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

**Solution**: Use separate virtual environments (ALL ALREADY EXIST):
```
.venv                 - Main development (VectorBT OSS)
.venv-vectorbt-pro    - VectorBT Pro only (for Pro-specific tests)
.venv-vectorbt        - VectorBT OSS only
.venv-backtrader      - Backtrader only
.venv-zipline         - Zipline only
.venv-validation      - Combined validation (BROKEN - OSS/Pro conflict)
```

**To run tests with VectorBT Pro:**
```bash
source .venv-vectorbt-pro/bin/activate
python -m pytest tests/validation/ -q
```

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

## Validation Strategy

**Per-framework validation in isolated environments** (NOT unified pytest).

See `validation/README.md` and `.claude/memory/validation_methodology.md` for details.

### Virtual Environments

| Environment | Purpose |
|-------------|---------|
| `.venv` | Main development |
| `.venv-vectorbt-pro` | VectorBT Pro validation (internal) |
| `.venv-backtrader` | Backtrader validation |
| `.venv-zipline` | Zipline (excluded - bundle issues) |

### Key Framework Behaviors

**VectorBT Pro**: Vectorized, uses close price, `accumulate=False` for no re-entry
**Backtrader**: Event-driven, COO/COC flags, integer shares
**Zipline**: EXCLUDED - uses bundle data instead of test DataFrame

## References

- Event-driven architecture patterns
- Position tracking best practices
- See `.claude/memory/` for architectural decisions
