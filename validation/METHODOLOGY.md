# Validation Methodology

*Last updated: 2026-08-13*

## Core Principle

**ml4t-backtest is designed to express external backtesting behavior through configurable
profiles and to measure every remaining difference.**

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
| `lean` | Frozen LEAN daily US-equity protocol | 0 fill gap, 0 canonical value gap |
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

Convergence with independently developed frameworks provides evidence for the execution behavior
covered by the retained workloads. It does not establish correctness for behavior that the
comparison matrix does not exercise.

### Fair Performance Comparison

Performance comparisons are valid only after the profile reproduces the behavior exercised by the
benchmark and every framework runs the same retained workload.

### User Trust

Users migrating from another framework can inspect retained comparison results for the matching
profile before switching to ml4t's default or realistic profiles.

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

Release comparisons use equality after all adapters serialize numeric values to an eight-decimal
fixed-point representation. This representation removes framework-specific binary floating-point
noise without allowing a nonzero value difference. Comparison artifacts retain raw values, raw
differences, canonical values, record counts, hashes, and the first divergent record. Scenario
diagnostic thresholds provide context in failure messages and never change pass or fail status.

### Step 6: Update Documentation

Update the behavioral difference matrix, profile diffs, and parity results.

## Anti-Patterns

**DO NOT** accept gaps as "structural" or "expected":
- "The engines use different order timing, so the gap is expected" is wrong.
- Identify the native event sequence, configure the corresponding profile, or narrow the claim.

**DO NOT** say "this is good enough for production":
- 95% parity is not the goal -- 100% parity is the goal
- Every trade difference represents a behavioral dimension we haven't captured yet

**DO NOT** compare frameworks with mismatched profiles:
- Comparing ml4t[default] vs Backtrader is meaningless
- Always compare ml4t[backtrader_strict] vs Backtrader

---

## Behavioral Difference Matrix

Complete matrix of how each framework handles every behavioral dimension.

### VectorBT native evidence

The VectorBT column cites checks from `native/vectorbt_behavior.py`. The checks import only the
frozen framework and its dependencies. Both current targets passed all checks:

- [VectorBT OSS 1.1.0 evidence](native/evidence/vectorbt_oss-1.1.0.json)
- [VectorBT Pro 2026.6.27 evidence](native/evidence/vectorbt_pro-2026.6.27.json)

`VBT-S` refers to the `signal_timing_and_default_fill` and `explicit_fill_price` checks. `VBT-R`
refers to `stop_fill` and `trailing_stop_extreme_and_intrabar_fill`. `VBT-C` refers to
`short_cash`, `cash_sharing_and_call_sequence`, and `insufficient_cash_partial_fill`. `VBT-O`
refers to `target_percent_sizing`, `accumulation`, `long_signal_conflict`, and
`missing_order_price`. `VBT-F` refers to `defaults`, `fees_and_slippage`, and
`record_construction`.

The evidence records the official source address for each frozen implementation. OSS uses
[`Portfolio.from_orders`](https://github.com/polakowo/vectorbt/blob/259d2d89fe2e7638baf3ca76c394937cd32b656d/vectorbt/portfolio/base.py#L1616)
and [`Portfolio.from_signals`](https://github.com/polakowo/vectorbt/blob/259d2d89fe2e7638baf3ca76c394937cd32b656d/vectorbt/portfolio/base.py#L2047).
Pro uses the corresponding methods at
[`Portfolio.from_orders`](https://github.com/polakowo/vectorbt.pro/blob/6e18cf0aa37849cfc20848f40f1d26ecfdc771b4/vectorbtpro/portfolio/base.py#L4145)
and [`Portfolio.from_signals`](https://github.com/polakowo/vectorbt.pro/blob/6e18cf0aa37849cfc20848f40f1d26ecfdc771b4/vectorbtpro/portfolio/base.py#L4646).

### Backtrader native evidence

The Backtrader column cites checks from `native/backtrader_behavior.py`. The probe imports only
Backtrader 1.9.78.123 and pandas. It passed all 14 checks in the retained
[Backtrader evidence](native/evidence/backtrader-1.9.78.123.json). The evidence identifies the
installed wheel and its SHA-256 digest, and records source locations inside that artifact.

`BT-T` refers to `next_bar_open_and_gap`, `cheat_on_close`, and `final_bar_market_order`. `BT-S`
refers to `integer_target_percent` and `commission_headroom`. `BT-C` refers to `defaults`,
`cash_rejection_and_configured_leverage`, `short_cash`, and `trade_record`. `BT-O` refers to
`submission_sequence`. `BT-R` refers to `signal_price_stop_basis` and
`trailing_stop_signal_close_and_lagged`. `BT-D` refers to `missing_bar_uses_last_value` and
`late_feed_start`.

### Zipline native evidence and comparison protocol

The retained [Zipline Reloaded 3.1.1 evidence](native/evidence/zipline-3.1.1.json) contains 13
checks executed through `zipline.run_algorithm` and direct construction of the frozen models.
The source locations resolve to commit `09885a2ebc7567d40942c891b3879dc03c745070`, and the
evidence identifies the source artifact and its SHA-256 digest.

The native defaults and the comparison protocol are different:

| Behavior | Zipline Reloaded 3.1.1 default | Suite comparison protocol |
|---|---|---|
| Daily fill | Next session close plus 5 basis points | Next session open |
| Equity commission | `$0.001` per share, `$0` minimum | No commission unless the scenario specifies one |
| Equity slippage | `FixedBasisPointsSlippage(5 bps, volume_limit=0.1)` | Custom open-price model with no volume cap |
| Optional volume-share model | `volume_limit=0.025`, `price_impact=0.1` | Used only by an explicit scenario or native check |
| Stop loss, take profit, trailing stop | No portfolio risk-rule protocol | Adapter evaluates daily OHLC and submits an exit |
| Closed trades | Native transactions and positions, no closed-trade column | Adapter reconstructs round trips from transactions |

The `0.1` in `VolumeShareSlippage` is a coefficient in a quadratic price-impact formula. It is
not a 10% slippage rate. The profile and scenario matrix reproduce the suite protocol, not the
framework defaults. Each scenario record identifies adapter-emulated risk rules and reconstructed
trade records in `provenance.comparison_protocol`.

`ZL-D` refers to `defaults` and `default_next_bar_close_fill`. `ZL-P` refers to
`configured_next_bar_open_fill` and `explicit_minimum_commission`. `ZL-S` refers to
`integer_target_percent` and `target_percent_snapshot`. `ZL-C` refers to
`cash_and_short_proceeds`. `ZL-O` refers to `submission_sequence` and
`final_bar_market_order`. `ZL-V` refers to `volume_share_partial_fills`. `ZL-X` refers to
`transaction_records_not_native_trades`. `ZL-M` refers to `session_calendar` and `missing_bar`.

`LN-T` refers to LEAN native `timing` and `final_bar_order`. `LN-M` refers to `default_models`,
`buying_power_allowed`, and `buying_power_rejected`. `LN-O` refers to `submission_sequence` and
`buying_power_sequence`. `LN-S` refers to `target_sizing`. `LN-C` refers to `explicit_costs`.
`LN-D` refers to `fill_forward`. `LN-F` refers to `default_full_fill`. `LN-X` refers to
`terminal_holding` and `liquidation`. These checks use LEAN engine 18001 at commit
`278fcb3d1b815b63ccadba68d7ae54422e34b792`, CLI 1.0.228, `DefaultBrokerageModel`, a margin
account, daily adjusted US equities, and 2x security leverage.

The tables below describe configured profile values. They are not all cross-engine equivalence
claims. `behavior_coverage.toml` maps every profile field to its native checks and cross-engine
scenarios. A dimension is publishable only when both references are present. The current scenario
matrix does not publish equivalence for target-percent sizing, insufficient-cash boundaries,
competing same-session orders, partial fills, missing bars, or late assets. Those settings remain
available profile choices backed by native evidence where the map identifies it.

### Execution Timing

| Knob | ml4t Default | VectorBT | VBT evidence | Backtrader | Zipline protocol | LEAN |
|------|-------------|----------|--------------|------------|---------|------|
| `execution_mode` | next_bar | **same_bar** | VBT-S | next_bar (BT-T) | next_bar (ZL-P) | next_bar (LN-T) |
| `fill_timing` | next_bar_open | **same_bar** | VBT-S | next_bar_open (BT-T) | next_bar_open (ZL-P) | next session (LN-T) |
| `execution_price` | open | **close** | VBT-S | open (BT-T) | open (ZL-P) | open (LN-T) |

LEAN emits a daily US-equity bar to `OnData` at the market close. A market order submitted from
that callback is therefore submitted while the market is closed, converted to market-on-open, and
filled at the next session's open. A final-bar order remains submitted and unfilled. VectorBT uses
same-bar close execution. Backtrader and the configured Zipline protocol use the next session's
open.

### Stop/Risk Configuration

| Knob | ml4t Default | VectorBT | VBT evidence | Backtrader | Zipline protocol | LEAN |
|------|-------------|----------|--------------|------------|---------|------|
| `stop_fill_mode` | stop_price | stop_price | VBT-R | stop_price (BT-R) | next_bar_open (adapter) | profile fallback; not claimed |
| `stop_level_basis` | fill_price | fill_price | VBT-R | **signal_price (BT-R)** | fill_price (adapter) | profile fallback; not claimed |
| `trail_hwm_source` | close | **bar_extreme** | VBT-R | close (BT-R) | bar_extreme (adapter) | profile fallback; not claimed |
| `initial_hwm_source` | fill_price | **bar_high** | VBT-R | signal_price (BT-R) | fill_price (adapter) | profile fallback; not claimed |
| `trail_stop_timing` | lagged | **intrabar** | VBT-R | lagged (BT-R) | intrabar (adapter) | profile fallback; not claimed |

Backtrader calculates stop levels from **signal bar close** (the price when the strategy decided
to trade), not the actual fill price. This matters when next-bar open differs significantly from
previous close.

### Account & Cash

| Knob | ml4t Default | VectorBT | VBT evidence | Backtrader | Zipline protocol | LEAN |
|------|-------------|----------|--------------|------------|---------|------|
| `allow_short_selling` | false | **true** | VBT-C | **true (BT-C)** | true (ZL-C) | true |
| `allow_leverage` | false | false | VBT-C | false by default (BT-C) | cash validation disabled (ZL-C) | true |
| `short_cash_policy` | credit | credit | VBT-C | credit (BT-C) | credit (ZL-C) | credit |
| `initial_margin` | 0.5 | -- | VBT-C | -- (BT-C) | -- | 0.5 (LN-M) |
| `long_maint_margin` | 0.25 | -- | VBT-C | -- (BT-C) | -- | 0.5 (LN-M) |
| `short_maint_margin` | 0.30 | -- | VBT-C | -- (BT-C) | -- | 0.5 (LN-M) |

These LEAN margin values belong to `SecurityMarginModel` at 2x leverage. Other security and
brokerage models can select different requirements.

### Order Handling

| Knob | ml4t Default | VectorBT | VBT evidence | Backtrader | Zipline protocol | LEAN |
|------|-------------|----------|--------------|------------|---------|------|
| `fill_ordering` | exit_first | exit_first | VBT-C, VBT-O | **fifo (BT-O)** | fifo (ZL-O) | sequential (LN-O) |
| `entry_order_priority` | submission | submission | VBT-C | submission (BT-O) | submission (ZL-O) | submission |
| `immediate_fill` | false | **true** | VBT-S, VBT-O | false (BT-T) | false (ZL-P, ZL-O) | false (LN-T) |
| `rebalance_mode` | incremental | **hybrid** | VBT-O | **snapshot (BT-O)** | snapshot (ZL-S) | snapshot |
| `rebalance_headroom_pct` | 1.0 | 1.0 | VBT-O | 1.0 (BT-S) | 1.0 (ZL-S) | 0.9975 (LN-S) |
| `reject_on_insufficient_cash` | true | **false** | VBT-C | true (BT-C, BT-O) | false (ZL-C) | true |
| `partial_fills_allowed` | false | **true** | VBT-C | false (BT-C) | false in protocol (ZL-P) | false in default fill model (LN-F) |
| `missing_price_policy` | skip | skip order; forward-fill valuation | VBT-O | **use_last (BT-D)** | last price for sizing; stale bar defers fill (ZL-M) | fill-forward data; defer to next real open (LN-D) |
| `late_asset_policy` | allow after 1 bar | allow after 1 bar | VBT-O | allow after 1 bar (BT-D) | allow (ZL-M) | allow |

### Position Sizing & Costs

| Knob | ml4t Default | VectorBT | VBT evidence | Backtrader | Zipline protocol | LEAN |
|------|-------------|----------|--------------|------------|---------|------|
| `share_type` | integer | fractional | VBT-O | integer (BT-S) | integer (ZL-S) | integer (LN-S) |
| repeated entry | check position | ignore unless accumulation is enabled | VBT-O | adapter checks position | adapter checks position | adapter submits target delta |
| `commission_model` | none | none | VBT-F | none (BT-C) | none (ZL-P) | `InteractiveBrokersFeeModel` (LN-M) |
| `commission_rate` | 0% | 0% | VBT-F | 0% (BT-C) | -- | -- |
| `commission_per_share` | -- | -- | VBT-F | -- | -- (ZL-P) | $0.005 for US equities (LN-M) |
| `commission_minimum` | $0 | -- | VBT-F | -- | -- (ZL-P) | $1 for US equities (LN-M) |
| `slippage_model` | none | none | VBT-F | none (BT-C) | custom open, zero cost (ZL-P) | `NullSlippageModel` (LN-M) |
| `slippage_rate` | 0% | 0% | VBT-F | 0% (BT-C) | 0% (ZL-P) | 0% (LN-M) |

The fee and slippage entries name the models selected by `DefaultBrokerageModel` for a US equity.
They are not universal LEAN defaults. `LN-C` separately verifies explicit constant fee and
percentage slippage overrides.

### Comparison profiles

`vectorbt_strict` uses locked short collateral, partial fills, and FIFO processing to match the
controlled `Portfolio.from_orders` protocol. The retained native check verifies that short-sale
cash is reported in the cash series but cannot fund new exposure. `backtrader` uses the measured
framework defaults for costs, leverage, target headroom, missing bars, and late feeds.
`backtrader_strict` adds submission-time cash checks to reproduce Backtrader's enabled
`checksubmit` path. `zipline_strict` resolves to the same settings as `zipline`; both represent the
explicit comparison protocol described above. The native Zipline defaults remain separate. The
`lean` profile represents only the named daily US-equity protocol above. Its stop settings are not
part of the retained LEAN equivalence claim.

Large-scale evidence compares native ordered fills first, then reconstructs closed trades from the
canonical 1e-8 fill records for both engines. This prevents sub-quantum differences in a
framework's post-hoc trade-table arithmetic from changing a trade derived from equivalent fills.
The scenario suite separately compares each framework's native trade records.

---

## Current Parity Status

<!-- parity-claims:start -->
<!-- Generated by validation/generate_parity_claims.py. Do not edit by hand. -->

Scenario claims use the retained release-candidate matrix. "Exact" means terminal values, ordered closed trades, and ordered fills match after 1e-8 quantization for every retained surface the framework exposes.

| Profile | Pinned framework | Required scenarios | Evidence |
|---|---|---:|---|
| `vectorbt_strict` | [VectorBT Pro 2026.6.27](https://github.com/polakowo/vectorbt.pro) | 17/17 exact | [scenario evidence](https://github.com/ml4t/backtest/blob/main/validation/CORRECTNESS_RESULTS.json) |
| `vectorbt` | [VectorBT OSS 1.1.0](https://pypi.org/project/vectorbt/1.1.0/) | 16/16 exact | [scenario evidence](https://github.com/ml4t/backtest/blob/main/validation/CORRECTNESS_RESULTS.json) |
| `backtrader_strict` | [Backtrader 1.9.78.123](https://pypi.org/project/backtrader/1.9.78.123/) | 17/17 exact | [scenario evidence](https://github.com/ml4t/backtest/blob/main/validation/CORRECTNESS_RESULTS.json) |
| `zipline_strict` | [Zipline Reloaded 3.1.1](https://pypi.org/project/zipline-reloaded/3.1.1/) | 16/16 exact | [scenario evidence](https://github.com/ml4t/backtest/blob/main/validation/CORRECTNESS_RESULTS.json) |

Large-scale claims are published only when a retained workload has zero canonical gap.

| Profile | Pinned framework | Compared | Trade gap | Terminal value | Evidence |
|---|---|---:|---:|---:|---|
| `vectorbt_strict` | VectorBT Pro 2025.12.31 (`1305a1e19743`) | 225,844 trades | 0 | 685179.007330 | [large-scale evidence](https://github.com/ml4t/backtest/blob/main/validation/vectorbt_pro/large_scale_parity.json) |

No large-scale claim is published for Backtrader, Zipline, VectorBT OSS, or LEAN without a passing retained artifact.
<!-- parity-claims:end -->

---

## Validation Harness

### Scenario Tests (17 scenarios x 4 frameworks)

The validation suite tests 17 scenarios against 4 external frameworks:

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
| 16 | Regime Path Coverage | 1500 bars, 9 market regimes and sparse entries |
| 17 | High Event Count | 1800 bars, up to 600 completed round trips |

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
