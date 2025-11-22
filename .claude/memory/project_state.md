# Project State: ml4t.backtest

*Last updated: 2025-11-22*

## Current Status

**Version**: 0.2.0 (Beta, post-cleanup)
**Tests**: 154 passing in ~1s
**Code**: ~2,800 lines source, ~2,700 lines tests (5,500 total)
**Python**: 3.11+

## Architecture Summary

Minimal event-driven backtesting engine with:
- **Engine**: Event loop orchestration
- **Broker**: Order execution, position tracking, exit-first processing
- **DataFeed**: Price + signal iteration via Polars
- **Strategy**: Abstract base class with `on_data()` callback
- **Accounting**: Pluggable cash/margin account policies

## Key Components

```
src/ml4t/backtest/
├── engine.py          # Main event loop
├── broker.py          # Order execution (~480 lines)
├── datafeed.py        # Data iteration
├── strategy.py        # Strategy interface
├── types.py           # Core dataclasses
├── models.py          # Commission/slippage models
├── config.py          # Configuration and presets
└── accounting/        # Account policies (4 files)
```

## Validation Strategy

**Per-framework in isolated venvs** (NOT unified pytest):
- `.venv-vectorbt-pro` - VectorBT Pro (internal, commercial)
- `.venv-backtrader` - Backtrader (open source)
- `.venv-zipline` - Zipline (excluded - bundle issues)

## Recent Changes (Nov 2025)

1. **99.2% code reduction** - From 739K to 5.5K lines
2. **Deleted**: `archive/`, `resources/`, chaotic `tests/validation/`
3. **Added**: `validation/` directory for per-framework scripts
4. **Simplified**: Flat module structure, no deep nesting

## Known Issues

- Mypy: 16 errors remain (relaxed config)
- VectorBT OSS/Pro conflict in same environment
- Zipline bundle data issues

## Next Steps

1. Create VectorBT Pro validation scripts
2. Create Backtrader validation scripts
3. Fix mypy errors
4. Document configuration presets
