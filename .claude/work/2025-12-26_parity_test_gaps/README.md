# Work Unit: Parity Test Gaps

**Created**: 2025-12-26
**Source**: ML4T book repo review
**Priority**: High

## Summary

Cross-framework validation tests are missing for:
- Commission models (% and per-share)
- Slippage models (fixed and %)
- Trailing stops
- Bracket orders (OCO)

## Files

- `requirements.md` - Detailed requirements with framework config mappings
- `plan.md` - Implementation plan with task checklist

## Quick Start

1. Read `requirements.md` for what needs to be done
2. Follow `plan.md` task order (Phase 1 → Phase 2 → Phase 3)
3. Use existing scenario scripts as templates
4. Update `validation/README.md` with results

## Context

The book repo (Chapter 9) references ml4t-backtest as achieving exact parity with VectorBT/Backtrader/Zipline. The existing tests validate core functionality (long/short, stops), but commission and slippage models - which significantly affect realistic backtesting - have not been validated cross-framework.
