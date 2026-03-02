# Archived Validation Scripts

Development scripts from the parity validation work (2025-2026). These were used during
iterative debugging to achieve 97.5% cross-framework parity. Archived because they are
one-off investigation tools, not part of the repeatable validation suite.

## Debug Scripts (edge-case investigation)

| Script | Purpose |
|--------|---------|
| `basic_entry_debug.py` | Debug entry signal processing |
| `continuous_signal_debug.py` | Debug continuous vs discrete signal handling |
| `debug_asset000.py` | Single-asset trace for asset #000 |
| `debug_asset001_442.py` | Single-asset trace for assets #001 and #442 |
| `debug_exit_price_diff.py` | Investigate exit price discrepancies |
| `debug_exit_price_mismatch.py` | Trace exit price mismatch root cause |
| `debug_hwm_precise.py` | High-water-mark precision comparison |
| `debug_hwm_tracking.py` | Trailing stop HWM tracking investigation |
| `debug_pnl_mismatch.py` | PnL difference root cause analysis |
| `debug_reentry.py` | Same-bar re-entry after stop exit |
| `debug_trailing_fill.py` | Trailing stop fill price investigation |
| `reentry_debug.py` | Re-entry timing investigation |
| `single_asset_debug.py` | Generic single-asset debug harness |
| `trailing_reentry_debug.py` | Trailing stop + re-entry combination |
| `trailing_stop_debug.py` | Trailing stop mechanics investigation |
| `trailing_trigger_debug.py` | Trailing stop trigger timing |

## Comparison Scripts (parity matching)

| Script | Purpose |
|--------|---------|
| `analyze_remaining_mismatches.py` | Categorize remaining trade-level gaps |
| `backtrader_strict_trace_compare.py` | Bar-by-bar trace comparison vs Backtrader |
| `signal_only_compare.py` | Compare signal processing across frameworks |
| `trailing_stop_compare.py` | Compare trailing stop behavior |
| `vbt_pro_perf_comparison.py` | VBT Pro performance parity check |
| `single_asset_exact_match.py` | Exact match verification (single asset) |
| `multi_asset_exact_match.py` | Exact match verification (multi-asset) |
| `ml4t_vbt_scale_match.py` | ml4t vs VBT scale match verification |

## Scale Tests (stress/performance)

| Script | Purpose |
|--------|---------|
| `scale_test.py` | Basic scale test harness |
| `large_scale_perf.py` | Large-scale performance profiling |
| `large_scale_validation.py` | Large-scale correctness validation |
| `calendar_scale_test.py` | Calendar performance at scale |
| `rebalancing_scale_test.py` | Rebalancing at scale (many assets) |

## Nautilus Evaluation (experimental, incomplete)

| Script | Purpose |
|--------|---------|
| `quickstart_test.py` | Nautilus Trader quickstart evaluation |
| `EVALUATION_RESULTS.md` | Nautilus evaluation findings |
