# Project State: ml4t.backtest

*Last updated: 2025-11-22*

## Current Status

**Version**: 0.2.0 (Beta, post-cleanup)
**Tests**: 154 passing in ~1s
**Code**: ~2,800 lines source, ~2,700 lines tests (5,500 total)
**Python**: 3.11+

## Architecture Summary

Event-driven backtesting engine with:
- **Engine**: Event loop orchestration with same-bar or next-bar execution
- **Broker**: Order execution, position tracking, exit-first processing
- **DataFeed**: Polars-based multi-asset iteration with lazy loading
- **Strategy**: Abstract base class with `on_data()` callback
- **Accounting**: Pluggable cash/margin account policies
- **Config**: Framework compatibility presets (VectorBT, Backtrader, Zipline)

## Key Capabilities

- **Assets**: Single-asset and multi-asset strategies
- **Account Types**: Cash (no leverage) and margin (with shorts)
- **Order Types**: Market, limit, stop, stop-limit, trailing stop, bracket
- **Order Management**: Update pending orders, cancel, partial liquidation
- **Execution Modes**: Same-bar (close) and next-bar (open)
- **P&L Tracking**: Per-trade and portfolio-level with commission/slippage

## Key Components

```
src/ml4t/backtest/
├── engine.py          # Main event loop
├── broker.py          # Order execution (~480 lines)
├── datafeed.py        # Polars data iteration with lazy loading
├── strategy.py        # Strategy interface
├── types.py           # Order, Position, Fill, Trade
├── models.py          # Commission/slippage models
├── config.py          # Framework presets
└── accounting/        # Cash and margin account policies
```

## Virtual Environments (Established)

| Environment | Contents | Status |
|-------------|----------|--------|
| `.venv` | Main development | Active |
| `.venv-vectorbt-pro` | VectorBT Pro 2025.x | Installed |
| `.venv-backtrader` | Backtrader only | Available |
| `.venv-zipline` | Zipline-reloaded | Low priority |
| `.venv-validation` | DEPRECATED | OSS/Pro conflict |

**Critical**: VectorBT OSS and Pro CANNOT coexist - both register `.vbt` accessor.

## Validation Approach

**Scenario-based, per-framework validation** in isolated environments:

1. Define specific test scenarios (long-only, long/short, with costs)
2. Run identical scenarios in ml4t.backtest AND external framework
3. Compare trade-by-trade: timestamps, prices, P&L
4. Document any configuration needed to match external behavior

**Goal**: Configurable behavior matching, not default matching. Users should be able to replicate any framework's exact behavior through configuration options.

## Known Issues

- Mypy: 16 errors remain (relaxed config)
- Zipline excluded (bundle data incompatibility)

## Next Steps

See work unit for Phase 1 ML Data Foundation validation effort.
