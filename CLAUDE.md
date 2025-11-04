# CLAUDE.md - QEngine Development Guidelines

## Context & Location

**You are in**: `/home/stefan/ml4t/software/backtest/` (qengine library)
**Parent directory**: `/home/stefan/ml4t/software/` (multi-library coordination)
**Sibling libraries**: `../data/`, `../features/`, `../evaluation/`
**Integration projects**: `../projects/` (use qengine in workflows)

**Division of Labor**: See `../.claude/memory/division_of_labor.md` for when to work here vs parent directory.

**Work in THIS directory for**:
- Event-driven execution engine improvements
- Order types and execution logic
- Position tracking and portfolio management
- Broker simulation and fill models
- Unit tests for qengine functionality
- API improvements and documentation

**Work in PARENT directory for**:
- Multi-library workflows using qengine + others
- Integration testing in `../projects/`
- Validation studies (e.g., VectorBT exact matching)

## Vision & Goals

QEngine provides an event-driven backtesting engine with institutional-grade execution fidelity.

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

## Development Standards

- **Python**: 3.9+ with type hints
- **Testing**: pytest with comprehensive coverage
- **Linting**: ruff (100 char line length)
- **Performance**: Numba JIT for hot paths
- **Quality**: Pre-commit hooks
- **Correctness**: No look-ahead bias, point-in-time accuracy

## Projects Awareness

`../projects/` uses qengine for:
- Strategy backtesting
- VectorBT replication and validation
- Execution fidelity testing
- Performance benchmarking

**Integration Work**: The VectorBT exact matching study (`../projects/crypto_futures/`) validates qengine produces identical results to VectorBT Pro.

Coordinate breaking changes through parent `.claude/`.

## Key Validation Requirements

When modifying execution logic:
1. Ensure point-in-time correctness (no look-ahead)
2. Test same-bar re-entry scenarios
3. Validate position sync after fills
4. Check against VectorBT reference results
5. Verify fill prices are within OHLC bounds

## References

- Event-driven architecture patterns
- VectorBT Pro fill model documentation
- Position tracking best practices
