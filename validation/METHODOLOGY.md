# Validation Methodology

*Last updated: 2026-03-02*

## Core Principle

**ml4t-backtest achieves validation confidence by being configurable enough to perfectly replicate
the behavior of every major external backtesting framework through profiles.**

There are NO "expected differences." Every trade count gap, every value gap, is a signal that a
configurable knob is missing or misconfigured. The gap must be driven to zero.

## How It Works

### 1. Configurable Code Paths

Every behavioral choice that differs between backtesting frameworks is expressed as a configurable
parameter in `BacktestConfig`. These are not workarounds or compatibility shims -- they are
first-class, well-documented behavioral dimensions that represent real design choices.

Examples of behavioral dimensions:
- **Execution timing**: same-bar vs next-bar
- **Fill price**: open, close, VWAP, midpoint
- **Stop level basis**: fill price vs signal price
- **Cash policy**: constrained vs unconstrained, credit vs lock_notional
- **Fill ordering**: exit-first vs FIFO
- **Share type**: integer vs fractional
- **Commission model**: none, percentage, per-share
- **Rebalance mode**: incremental vs snapshot

### 2. Profiles for Each External Framework

Each external backtester gets a profile that sets ALL configurable knobs to the values that
replicate that framework's behavior:

| Profile | Emulates | Goal |
|---------|----------|------|
| `vectorbt` / `vectorbt_strict` | VectorBT Pro/OSS | 0% trade gap, 0% value gap |
| `backtrader` / `backtrader_strict` | Backtrader | 0% trade gap, 0% value gap |
| `zipline` / `zipline_strict` | Zipline Reloaded | 0% trade gap, 0% value gap |
| `lean` / `lean_strict` | QuantConnect LEAN | 0% trade gap, 0% value gap |
| `default` | ml4t's own opinion | Best-practice defaults |
| `realistic` | Conservative simulation | Adds costs, integer shares |

### 3. ml4t Default = Our Best Opinion

The `default` profile represents ml4t's own opinion on the most reasonable settings:

| Dimension | ml4t Default | Rationale |
|-----------|-------------|-----------|
| Execution | next_bar_open | No look-ahead bias |
| Costs | 0.1% commission + 0.1% slippage | Conservative but not punishing |
| Short selling | disabled | Require explicit opt-in |
| Leverage | disabled | Require explicit opt-in |
| Share type | fractional | Simpler for research |
| Fill ordering | exit_first | Capital-efficient |
| Rebalance | incremental | Most accurate cash tracking |
| Stop basis | fill_price | Based on actual execution |

Every choice is documented and justified. Users can see exactly what the default profile does
and why, and they can override any setting.

### 4. Transparency Through Configuration

The configuration system makes ALL behavioral choices explicit and visible:
- A user can inspect any profile and see exactly which settings differ from default
- The behavioral difference matrix (below) documents what each framework does
- No hidden behaviors -- if a framework does something differently, it's a named config parameter

## Why This Approach

### Confidence Through Convergence

If ml4t can perfectly replicate VectorBT, Backtrader, Zipline, and LEAN -- all independently
developed frameworks with different architectures -- then the core execution logic must be correct.
All four frameworks would have to be wrong in the same way for ml4t to be wrong.

### Fair Performance Comparison

When ml4t runs with a framework's profile, the comparison is apples-to-apples. Performance
benchmarks compare identical behavioral semantics, not different execution models.

### User Trust

Users migrating from another framework can verify that ml4t produces identical results with the
matching profile before switching to ml4t's default or realistic profiles.

## Validation Process

### Step 1: Identify Behavioral Differences

Run the benchmark suite against an external framework. Any gap -- no matter how small -- indicates
a behavioral difference that needs to be captured.

### Step 2: Root-Cause the Gap

For each gap, identify the specific behavioral dimension that differs. This is NOT "debug the bug"
-- it's "which design choice does this framework make differently?"

Common root causes:
- Fill price timing (close vs open vs next-bar open)
- Cash constraint model (constrained vs unconstrained)
- Order processing sequence (exit-first vs FIFO)
- Position sizing (integer vs fractional shares)
- Commission/slippage application

### Step 3: Add the Config Knob

If the behavioral dimension isn't already configurable in `BacktestConfig`, add it:
1. Add the parameter to `BacktestConfig` with a sensible default
2. Wire it through the execution path (broker, engine, accounting)
3. Add tests for both values of the parameter
4. Document it in the behavioral difference matrix

### Step 4: Update the Profile

Set the new parameter in the framework's profile to match its behavior.

### Step 5: Re-run and Verify Zero Gap

Run the benchmark suite again. The gap for this dimension should now be zero.
If not, there are additional behavioral differences to capture -- go back to step 1.

### Step 6: Update Documentation

Update the behavioral difference matrix, profile diffs, and parity results.

## Anti-Patterns

**DO NOT** accept gaps as "structural" or "expected":
- "LEAN fills at bar close, ml4t at next-bar open -- this is a structural difference" is WRONG
- CORRECT: "We need to add `execution_price=close` to the lean profile to close this gap"

**DO NOT** say "this is good enough for production":
- 95% parity is not the goal -- 100% parity is the goal
- Every trade difference represents a behavioral dimension we haven't captured yet

**DO NOT** compare frameworks with mismatched profiles:
- Comparing ml4t[default] vs Backtrader is meaningless
- Always compare ml4t[backtrader_strict] vs Backtrader

---

## Behavioral Difference Matrix

Complete matrix of how each framework handles every behavioral dimension.

### Execution Timing

| Knob | ml4t Default | VectorBT | Backtrader | Zipline | LEAN |
|------|-------------|----------|------------|---------|------|
| `execution_mode` | next_bar | **same_bar** | next_bar | next_bar | **same_bar** |
| `fill_timing` | next_bar_open | **same_bar** | next_bar_open | next_bar_open | **same_bar** |
| `execution_price` | open | **close** | open | open | **close** |

VectorBT and LEAN both use same-bar execution with close fills. VectorBT is vectorized
(inherently same-bar), while LEAN processes market orders immediately at bar close in its
event-driven loop. All event-driven frameworks (BT, Zipline) except LEAN use next-bar open.

### Stop/Risk Configuration

| Knob | ml4t Default | VectorBT | Backtrader | Zipline | LEAN |
|------|-------------|----------|------------|---------|------|
| `stop_fill_mode` | stop_price | stop_price | stop_price | stop_price | stop_price |
| `stop_level_basis` | fill_price | fill_price | **signal_price** | fill_price | fill_price |
| `trail_hwm_source` | close | **bar_extreme** | close | close | close |
| `initial_hwm_source` | fill_price | **bar_high** | fill_price | fill_price | fill_price |
| `trail_stop_timing` | lagged | **intrabar** | lagged | lagged | lagged |

Backtrader calculates stop levels from **signal bar close** (the price when the strategy decided
to trade), not the actual fill price. This matters when next-bar open differs significantly from
previous close.

### Account & Cash

| Knob | ml4t Default | VectorBT | Backtrader | Zipline | LEAN |
|------|-------------|----------|------------|---------|------|
| `allow_short_selling` | false | **true** | **true** | false | true |
| `allow_leverage` | false | false | **true** | false | true |
| `short_cash_policy` | credit | credit | credit | credit | credit |
| `initial_margin` | 0.5 | -- | **0.5** | -- | varies |
| `long_maint_margin` | 0.25 | -- | **0.25** | -- | varies |
| `short_maint_margin` | 0.30 | -- | **0.30** | -- | varies |

### Order Handling

| Knob | ml4t Default | VectorBT | Backtrader | Zipline | LEAN |
|------|-------------|----------|------------|---------|------|
| `fill_ordering` | exit_first | exit_first | **fifo** | exit_first | **exit_first** |
| `entry_order_priority` | submission | submission | submission | submission | submission |
| `rebalance_mode` | incremental | **hybrid** | **snapshot** | **snapshot** | snapshot |
| `rebalance_headroom_pct` | 1.0 | 1.0 | **0.998** | **0.998** | 0.998 |
| `reject_on_insufficient_cash` | true | **false** | true | true | true |
| `partial_fills_allowed` | false | **true** | false | **true** | false |
| `missing_price_policy` | skip | **use_last** | **use_last** | **use_last** | use_last |

### Position Sizing & Costs

| Knob | ml4t Default | VectorBT | Backtrader | Zipline | LEAN |
|------|-------------|----------|------------|---------|------|
| `share_type` | fractional | fractional | **integer** | **integer** | **integer** |
| `signal_processing` | check_position | **process_all** | check_position | check_position | check_position |
| `commission_model` | percentage | **none** | percentage | **per_share** | per_share |
| `commission_rate` | 0.1% | 0% | 0.1% | -- | -- |
| `commission_per_share` | -- | -- | -- | **$0.005** | varies |
| `commission_minimum` | $0 | -- | -- | **$1.00** | varies |
| `slippage_model` | percentage | **none** | percentage | **volume_based** | volume_based |
| `slippage_rate` | 0.1% | 0% | 0.1% | **10%** | varies |

### "Strict" Profile Additions (for validation parity)

| Knob | vbt_strict | bt_strict | zip_strict | lean_strict |
|------|-----------|-----------|------------|-------------|
| `short_cash_policy` | **lock_notional** | credit | **credit** | credit |
| `skip_cash_validation` | false | false | **true** | false |
| `reject_on_insufficient_cash` | **true** | true | true | true |
| `fill_ordering` | **fifo** | fifo | exit_first | **exit_first** |
| `next_bar_submission_precheck` | -- | **true** | -- | **true** |
| `next_bar_simple_cash_check` | -- | **true** | -- | **false** |
| `allow_short_selling` | -- | -- | **true** | -- |

---

## Current Parity Status

Validated on both synthetic and **real market data** (250 US equities, 1998-2018).

### Real Data: 250 assets x 20 years (us_equities.parquet)

| Profile | ml4t Trades | Ext Trades | Trade Gap | ml4t Value | Ext Value | Value Gap | Speedup |
|---------|-------------|------------|-----------|------------|-----------|-----------|---------|
| **zipline_strict** | 226,723 | 226,723 | **0 (0.00%)** | -$18,846,482 | -$18,846,501 | **$19 (0.0001%)** | **8.4x** |
| **backtrader_strict** | 216,980 | 216,981 | -1 (0.0005%) | $13,880,796 | $13,881,298 | $503 (0.004%) | **19.3x** |
| **vectorbt_strict** | 210,352 | 210,261 | +91 (0.04%) | $135,539 | $135,539 | $0 (0.00%) | 0.04x |
| **lean_strict** | 226,246 | 225,583 | +663 (0.29%) | $1,106,708 | $1,093,558 | $13,150 (1.2%) | **5.3x** |

### Parity Confidence

| Framework | Trade Diff (real) | Value Diff (real) | Confidence | Status |
|-----------|-------------------|-------------------|------------|--------|
| Zipline | **0 (0.00%)** | **$19 (0.0001%)** | **99.9%+** | DONE |
| Backtrader | **-1 (0.0005%)** | $503 (0.004%) | **99%+** | Production |
| VBT OSS | +91 (0.04%) | $0 (0.00%) | **99%+** | Production |
| LEAN | +663 (0.29%) | $13K (1.2%) | **97%+** | Buying-power reservation gap |

---

## Validation Harness

### Scenario Tests (16 scenarios x 4 frameworks)

The validation suite tests 16 scenarios against 4 external frameworks:

| ID | Scenario | What It Tests |
|----|----------|---------------|
| 01 | Long Only | Basic execution, fill price |
| 02 | Long/Short | Direction switching, short selling |
| 03 | Stop Loss | Risk rule triggering, stop fill mode |
| 04 | Take Profit | Limit order execution |
| 05 | Commission (Pct) | Percentage commission model |
| 06 | Commission (Per-Share) | Per-share commission model |
| 07 | Slippage (Fixed) | Fixed slippage model |
| 08 | Slippage (Pct) | Percentage slippage model |
| 09 | Trailing Stop | High-water-mark tracking, exit timing |
| 10 | Bracket Order | OCO orders, rule chain priority |
| 11 | Short Only | Short position mechanics |
| 12 | Short Trailing Stop | Short-side high-water-mark |
| 13 | TSL + TP Combo | Rule chain evaluation order |
| 14 | TSL + SL Combo | Rule chain evaluation order |
| 15 | Triple Rule | Three-rule chain with priority |
| 16 | Stress Test | 1500 bars, 9 market regimes |

### Large-Scale Validation (250 assets x 20 years)

Beyond scenario tests, the benchmark suite runs a full ranking strategy on 250 US equities
from 1998-2018. This produces 200K+ trades that are compared trade-by-trade between frameworks.
See `benchmark_suite.py` for the implementation.

### Running the Suite

```bash
# Single scenario
python validation/run_scenario.py --scenario 01 --framework backtrader

# All scenarios for one framework
python validation/run_scenario.py --framework vectorbt_oss

# Full matrix
python validation/run_scenario.py --all

# Large-scale benchmark
python validation/benchmark_suite.py --profile backtrader_strict --framework backtrader
```

---

## File References

| Resource | Path |
|----------|------|
| Config (40+ knobs) | `src/ml4t/backtest/config.py` |
| Profiles (6 core + 4 strict) | `src/ml4t/backtest/profiles.py` |
| Scenario definitions | `validation/scenarios/definitions.py` |
| Framework drivers | `validation/frameworks/` |
| Benchmark suite | `validation/benchmark_suite.py` |
| Scenario runner | `validation/run_scenario.py` |
