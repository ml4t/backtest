# Monorepo Integration Update - August 6, 2025

## Critical Changes

### 1. Simplified Development Setup
- QuantLab now uses a **single virtual environment** for all three projects
- Unified `pyproject.toml` at monorepo root manages all dependencies
- Package manager changed to `uv` (10-100x faster than pip)

### 2. Your Role in the Ecosystem

You are the **backtesting layer** in a three-library pipeline:

```
qfeatures → qeval → qengine (YOU)
     ↓         ↓         ↓
 Features  Validation  Backtest
```

### 3. Expected Input Sources

You consume data from two upstream libraries:

**From qfeatures (features and labels):**
| Column | Type | Purpose |
|--------|------|---------|
| `event_time` | datetime64[ns] | Point-in-time anchor |
| `asset_id` | str | Instrument identifier |
| `features...` | float64 | Feature columns |
| `label` | int8 | ML target |
| `t_exit` | datetime64[ns] | Position exit time |

**From qeval (validated models):**
- Trained model instances
- Performance metrics
- Statistical significance tests

### 4. Critical Integration Requirements

Your engine MUST:
- Use `event_time` for point-in-time correctness
- Never look ahead past `event_time`
- Support multi-asset backtesting via `asset_id`
- Handle model signals from qeval's validated models

### 5. Testing Your Code

From monorepo root:
```bash
make test-qng    # Test qengine only
make test        # Test everything
make cycle       # Format, lint, type-check, and test
```

### 6. Import Simplification

You can now directly import from sibling projects:
```python
# These work from anywhere in the monorepo
from qfeatures import Pipeline, bars, labeling
from qeval import Evaluator, CombinatorialPurgedKFold
```

### 7. Your Previous Makefile Commands

Your project-specific Makefile commands are now integrated into the monorepo Makefile:
- `make format` → Works for all projects
- `make lint` → Works for all projects
- `make type-check` → Now `make type`
- `make test` → Now `make test-qng` for qengine only

### 8. Integration Testing

See `/integration_tests/` for examples of the complete pipeline from features through backtesting.

## What This Means For You

1. **Continue** your backtesting engine development
2. **Respect** the input schemas from upstream
3. **Maintain** point-in-time correctness
4. **Test** using monorepo commands
5. **Remember** you're the final validation step before trading

## Project Status

You are the newest addition to QuantLab (added August 2025). Your core event system is implemented, with execution and portfolio modules currently in progress.

## Key Files to Reference

- `/CLAUDE.md` - Monorepo guidelines
- `/pyproject.toml` - Unified dependencies
- `/Makefile` - Common commands
- `/integration_tests/` - End-to-end pipeline examples
