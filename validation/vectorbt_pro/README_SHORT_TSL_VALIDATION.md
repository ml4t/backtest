# SHORT Trailing Stop Validation Results

**Date**: 2026-01-20
**Status**: ✅ VALIDATED - SHORT TSL logic matches VBT Pro

---

## Summary

The investigation confirmed that the SHORT trailing stop logic in ml4t-backtest is **correct and matches VBT Pro exactly**.

The initial suspicion was based on the observation that:
- `scenario_09_trailing_stop.py` tests **LONG ONLY** (`allow_short_selling=False`)
- `scenario_11_short_only.py` tests SHORT but with **NO trailing stops**

This led to concerns that SHORT + TSL was never properly tested.

## Validation Results

### scenario_12_short_trailing_stop.py
- Tests 5% trailing stop for SHORT positions
- Entry at downtrend starts, exit via TSL on reversals
- **Result: 100% match** - Trade count, exit bars, exit prices, PnL all match

### scenario_12b_short_tsl_stress.py
- **Gap-Through (3% TSL)**: ✅ PASS - Price gaps above TSL level handled correctly
- **Tight Stop (2% TSL)**: ✅ PASS - More frequent exits, all match
- **Multiple Entries (5% TSL)**: ✅ PASS - 3 entries, 2 closed trades match exactly

## Key Finding: Open Position Reporting Difference

VBT Pro and ml4t handle open positions differently:

| Framework | Open Position at End of Backtest |
|-----------|----------------------------------|
| VBT Pro | Reports as trade with `Status="Open"`, shows mark-to-market PnL |
| ml4t | Does NOT report open positions in `trades` list |

This is a **reporting difference**, not a logic bug. The TSL exit logic is identical.

## Technical Details

### SHORT TSL Algorithm (INTRABAR mode)
1. Compute live LWM = `min(previous_lwm, current_bar_low)`
2. Compute TSL level = `LWM * (1 + trail_pct)`
3. If `bar_high >= TSL`: trigger exit
4. Fill at stop price (or bar open if gap-through)
5. Update LWM at end of bar

### Configuration for VBT Pro Compatibility
```python
engine = Engine(
    ...
    trail_hwm_source=TrailHwmSource.HIGH,  # Uses LOW for LWM
    initial_hwm_source=InitialHwmSource.BAR_CLOSE,
    stop_fill_mode=StopFillMode.STOP_PRICE,
    trail_stop_timing=TrailStopTiming.INTRABAR,
)
```

## Files Created

| File | Purpose |
|------|---------|
| `scenario_12_short_trailing_stop.py` | Main SHORT + TSL validation |
| `scenario_12b_short_tsl_stress.py` | Stress tests (gaps, tight stops, multiple entries) |

## Conclusion

**No fix required.** The SHORT trailing stop logic in ml4t-backtest is correct and produces identical results to VBT Pro for all tested scenarios.
