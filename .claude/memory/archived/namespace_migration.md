# Namespace Package Migration

**Date:** 2025-11-15
**Status:** Complete

## Change Summary

Migrated from standalone `qengine` package to namespaced `ml4t.backtest` package structure.

### Before (Old Structure)
```
src/qengine/
├── __init__.py
├── engine.py
├── core/
├── data/
├── execution/
├── portfolio/
├── strategy/
└── reporting/
```

**Imports:**
```python
from qengine import Engine, Strategy
from qengine.core import Clock, Event
from qengine.execution import SimulationBroker
```

### After (New Structure)
```
src/ml4t/backtest/
├── __init__.py
├── engine.py
├── core/
├── data/
├── execution/
├── portfolio/
├── strategy/
└── reporting/
```

**Imports:**
```python
from ml4t.backtest import BacktestEngine, Strategy
from ml4t.backtest.core import Clock, Event
from ml4t.backtest.execution import SimulationBroker
```

## Rationale

**Namespace Package Architecture:**
The ml4t project is being organized as a namespaced package with four independent libraries:
- `ml4t.data` - Market data management
- `ml4t.features` - Feature engineering
- `ml4t.backtest` - Backtesting engine
- `ml4t.eval` - Statistical validation

**Benefits:**
1. **Clear organization** - All ML4T libraries under single namespace
2. **No naming conflicts** - Each library has its own subpackage
3. **Independent versioning** - Libraries can be released independently
4. **Consistent imports** - All use `ml4t.*` pattern

## Package Details

**Package Name:** `ml4t-backtest`
**Import Name:** `ml4t.backtest`
**Location:** `/home/stefan/ml4t/software/backtest/src/ml4t/backtest/`

**PyPI Name:** `ml4t-backtest` (when published)
**Local Development:** `pip install -e .` installs as `ml4t.backtest`

## Migration Checklist

- [x] Moved source from `src/qengine/` to `src/ml4t/backtest/`
- [x] Updated `__init__.py` imports to use `ml4t.backtest.*`
- [x] Updated PROJECT_MAP.md with new structure
- [x] Verified import paths in core modules
- [ ] Update all example code (when examples exist)
- [ ] Update documentation (when written)
- [ ] Update README.md import examples
- [ ] Migration guide for external users (if needed)

## Breaking Changes

**Import changes required for any code using this library:**

```python
# Old
from qengine import Engine
from qengine.core import Clock

# New
from ml4t.backtest import BacktestEngine
from ml4t.backtest.core import Clock
```

**Note:** This is pre-1.0 software, so breaking changes are expected.

## Related Files

- `PROJECT_MAP.md` - Updated with namespace structure
- `pyproject.toml` - Package name remains `ml4t-backtest`
- `src/ml4t/backtest/__init__.py` - Package entry point

## Future Considerations

**When other ml4t libraries are published:**
- Users can install full suite: `pip install ml4t-data ml4t-features ml4t-backtest ml4t-eval`
- Or just what they need: `pip install ml4t-backtest`
- All import as: `from ml4t.{library} import ...`

**Namespace package features:**
- Each library is independently installable
- All share the `ml4t` namespace
- No `__init__.py` in `src/ml4t/` (namespace package pattern)

---

**Last Updated:** 2025-11-15
