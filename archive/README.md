# Archive

This directory contains code from the original 19,876-line implementation that was replaced by the minimal 744-line `engine_v2.py`.

## Structure

### `/obsolete/`
Old code that is no longer used. Kept for reference only.

- **src/** - Old modular source code (core/, data/, execution/, portfolio/, risk/, strategy/)
- **examples/** - Examples written for the old engine
- **tests/** - Tests for the old engine (unit/, integration/, etc.)
- **Root files** - Old config.py, engine.py, results.py, profiling outputs

### `/possibly-relevant/`
Files that might be useful for reference.

- **comparison/** - Framework comparison data and results
- **data/** - Test data files
- **extractors/** - Data extraction utilities
- **results/** - Previous backtest results

## Why Archived?

The original implementation grew to 19,876 lines across 41 files with:
- Complex interdependencies
- Low test coverage (~30%)
- Over-engineered abstractions

The new `engine_v2.py` (744 lines) provides:
- Same core functionality
- 82% test coverage
- 21x faster than Backtrader
- Cleaner, extensible API
- Same-bar and next-bar execution modes

## Migration

To use the new engine:

```python
from ml4t.backtest import (
    Engine, Strategy, DataFeed,
    PercentageCommission, PercentageSlippage,
    ExecutionMode
)
```

See `tests/test_core.py` for examples.
