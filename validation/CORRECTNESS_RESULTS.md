# Correctness Validation Results

**Generated**: 2026-01-01 11:03:36

## Summary

| Framework | Scenario | Status |
|-----------|----------|--------|
| VectorBT Pro | 01: Long Only | ✅ PASS |
| VectorBT Pro | 02: Long/Short | ✅ PASS |
| VectorBT Pro | 03: Stop Loss | ✅ PASS |
| VectorBT Pro | 04: Take Profit | ✅ PASS |
| VectorBT Pro | 05: Commission (Pct) | ✅ PASS |
| VectorBT Pro | 06: Commission (Per-Share) | ✅ PASS |
| VectorBT Pro | 07: Slippage (Fixed) | ✅ PASS |
| VectorBT Pro | 08: Slippage (Pct) | ✅ PASS |
| VectorBT Pro | 09: Trailing Stop | ❌ FAIL |
| VectorBT Pro | 10: Bracket Order | ❌ FAIL |

## Statistics

- **Passed**: 8
- **Failed**: 2
- **Skipped**: 0
- **Total**: 10

## Failures

### VectorBT Pro - 09: Trailing Stop

**Error**: None

```
======================================================================
Scenario 09: Trailing Stop (5%)
======================================================================

  Bars: 100, Entry signals: 2

  Running VectorBT Pro...
Traceback (most recent call last):
  File "/home/stefan/ml4t/software/backtest/.venv-vectorbt-pro/lib/python3.13/site-packages/pandas/core/indexes/base.py", line 3812, in get_loc
    return self._engine.get_loc(casted_key)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^
 
```

### VectorBT Pro - 10: Bracket Order

**Error**: None

```
======================================================================
Scenario 10: Bracket Order (SL=5%, TP=10%)
======================================================================

  Bars: 100, Entry signals: 3

  Running VectorBT Pro...
Traceback (most recent call last):
  File "/home/stefan/ml4t/software/backtest/validation/vectorbt_pro/scenario_10_bracket_order.py", line 246, in <module>
    sys.exit(main())
             ~~~~^^
  File "/home/stefan/ml4t/software/backtest/validation/vecto
```
