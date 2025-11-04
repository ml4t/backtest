# Root Cause Analysis Plan: qengine vs VectorBT Discrepancies

## Executive Summary

**Observed Discrepancies:**
- **Trade Count**: qengine 337 vs VectorBT 482 (30% fewer trades)
- **Exit Types**: qengine 100% TSL vs VectorBT 87.7% TSL + 11.8% TP + 0.6% SL
- **Performance**: qengine -24.89% vs VectorBT baseline (unknown)

**Investigation Strategy:** 4-layer diagnostic approach from quick data inspection to isolated component testing, following evidence chain to avoid trial-and-error.

---

## Root Cause Categories (6 Categories Identified)

### Category 1: Signal Generation Differences
**Hypothesis**: Entry signals are being generated or interpreted differently between engines.

**Possible Causes:**
- Different signal interpretation (close > upper vs different comparison)
- Signal timing differences (bar close vs bar open)
- Data alignment issues (timestamp mismatches)
- Crossover vs level detection (VectorBT might use transitions, qengine uses absolute values)

### Category 2: Position Management Differences
**Hypothesis**: Re-entry logic prevents consecutive positions that VectorBT allows.

**Possible Causes:**
- Re-entry restrictions after position close
- Position state tracking errors (is_flat check failing)
- Signal crossover detection (checking signal==True vs False->True transition)
- Same-bar entry/exit handling differences

### Category 3: Exit Mechanism Differences
**Hypothesis**: TSL triggers too aggressively, preventing TP exits.

**Possible Causes:**
- TP threshold never reached before TSL triggers
- TSL tracking more aggressive (updating more frequently)
- TP/TSL priority logic differences
- Percentage calculations applied to different base prices

### Category 4: Execution Timing Differences
**Hypothesis**: Event-driven vs vectorized timing causes systematic differences.

**Possible Causes:**
- Order execution timing (event-driven vs close-based)
- Fill price calculation differences
- Bracket order activation timing
- Same-bar execution handling

### Category 5: Bracket Order Implementation
**Hypothesis**: TSL update logic or exit priority differs from VectorBT.

**Possible Causes:**
- TSL update frequency (every bar vs only when profitable)
- Base price for TSL calculation (highest high vs entry price + profit)
- TP calculation base price
- Exit priority when both conditions met on same bar

### Category 6: Data Quality and Alignment
**Hypothesis**: Data misalignment causes signal lookups to fail.

**Possible Causes:**
- Timestamp alignment between indicators and futures data
- Missing data handling (failed signal lookups)
- Signal generation timing (forward-looking indicators)
- Date range filtering differences

---

## 4-Layer Diagnostic Plan

### Layer 1: Quick Data Inspection (15 minutes)
**Goal**: Identify high-level patterns without code changes.

**Steps:**
1. **Compare First 10 Trades**
   - Load qengine trades: `data/trade_logs/qengine_trades_q1_2024.parquet`
   - Load VectorBT trades: `data/trade_logs/vectorbt_trades_q1_2024.parquet` (if exists)
   - Compare entry timestamps
   - **Decision Point**: If timestamps match → signal OK, investigate exits (go to step 3)
   - **Decision Point**: If timestamps differ → signal generation issue (go to Layer 2-A)

2. **Count Consecutive Signal Utilization**
   - For first 50 signals, count how many result in trades
   - Calculate utilization rate for both engines
   - **Decision Point**: If qengine skips signals VectorBT doesn't → re-entry blocking (go to Layer 2-B)

3. **Compare Exit Types and Timing**
   - For trades with matching entry timestamps, compare exit types
   - Calculate time-in-trade statistics
   - **Decision Point**: If qengine exits earlier systematically → TSL too aggressive (go to Layer 2-C)

4. **Analyze Trade Duration Distribution**
   - Plot histogram of trade durations for both engines
   - Calculate mean, median, std dev
   - **Expected**: If TSL is too aggressive, qengine trades will be systematically shorter

**Expected Outcomes:**
- Clear identification of which category (1-6) is primary root cause
- Evidence-based decision on which code to inspect next

---

### Layer 2: Code Inspection (30-45 minutes)
**Goal**: Understand implementation details causing observed discrepancies.

#### Layer 2-A: Signal Generation (if Layer 1 Step 1 failed)
**Files to Read:**
- `projects/crypto_futures/scripts/run_qengine_backtest.py` (signal generation logic)
- Check: Line ~100-110 where signal is created from donchian_120min_upper

**Questions to Answer:**
- Is signal using `close > donchian_upper` or different comparison?
- Is signal generated at bar close or bar open?
- Are timestamps aligned between indicators df and futures df?

#### Layer 2-B: Re-entry Logic (if Layer 1 Step 2 failed)
**Files to Read:**
- `projects/crypto_futures/scripts/run_qengine_backtest.py` (on_market_event method)
- `backtest/src/qengine/portfolio/portfolio.py` (position state tracking)

**Questions to Answer:**
- How is `is_flat` determined?
- Is there any delay or restriction on re-entry after exit?
- Does signal detection use crossover (False->True) or level (True)?

#### Layer 2-C: TSL Triggering Logic (if Layer 1 Step 3 failed)
**Files to Read:**
- `backtest/src/qengine/execution/bracket_manager.py` (on_market_event method)
- Focus on TSL update logic and trigger conditions

**Questions to Answer:**
- How often is TSL updated? (every bar or only when profitable?)
- What base price is used for TSL calculation?
- What base price is used for TP calculation?
- What is the exit priority when both TSL and TP trigger on same bar?

**Key Code Sections:**
```python
# bracket_manager.py - TSL update logic
def on_market_event(self, event):
    # Find where TSL price is updated
    # Check: tsl_price = entry_price * (1 - tsl_pct)  OR
    #        tsl_price = highest_high * (1 - tsl_pct)

# bracket_manager.py - Exit trigger logic
def check_exit_conditions(self, bracket, current_price):
    # Check order of TSL vs TP checks
    # Verify percentage calculations
```

---

### Layer 3: Instrumentation (1 hour)
**Goal**: Add logging to track exact behavior during backtest.

**Instrumentation Points:**

1. **TSL/TP Distance Tracking**
   ```python
   # In bracket_manager on_market_event
   print(f"[BRACKET] Bar {event.timestamp}: "
         f"TSL={tsl_price:.2f} (dist={current_price - tsl_price:.2f}), "
         f"TP={tp_price:.2f} (dist={tp_price - current_price:.2f})")
   ```

2. **Re-entry Attempt Logging**
   ```python
   # In strategy on_market_event
   if has_signal:
       print(f"[SIGNAL] {event.timestamp}: signal=True, is_flat={is_flat}, "
             f"position_qty={broker_position.quantity if broker_position else 0}")
   ```

3. **Position State Transitions**
   ```python
   # In portfolio on fill
   print(f"[POSITION] {timestamp}: qty {old_qty} -> {new_qty}")
   ```

**Expected Outcomes:**
- Detailed trace of first 20 trades showing exact TSL/TP values
- Evidence of when TSL updates and why TP never triggers
- Clear view of re-entry blocking if it occurs

---

### Layer 4: Isolated Component Tests (2 hours)
**Goal**: Create controlled tests to validate specific behaviors.

**Test 1: Bracket Manager TSL Update**
```python
# Test TSL updates on every bar vs only when profitable
def test_tsl_update_frequency():
    # Create bracket with entry at 100, TSL 1%
    # Feed market events: 101, 100.5, 100, 99.5, 99
    # Verify TSL price after each event
    # Expected: TSL should update to 101 * 0.99 = 99.99 on first bar
```

**Test 2: TP vs TSL Priority**
```python
# Test what happens when both trigger on same bar
def test_exit_priority_same_bar():
    # Create bracket with entry at 100, TP 2.5%, TSL 1%
    # Feed market event with high=102.6 (triggers TP), low=98.99 (triggers TSL)
    # Verify which exit executes
```

**Test 3: Re-entry After Exit**
```python
# Test if immediate re-entry is allowed
def test_immediate_reentry():
    # Enter position, exit via TSL on bar N
    # On bar N+1, signal is still True
    # Verify new position is opened
```

---

## Execution Sequence

### Session 1: Quick Diagnosis (30 min)
1. Run Layer 1 diagnostics (15 min)
2. Based on results, identify primary category (5 min)
3. Run appropriate Layer 2 code inspection (10 min)

**Deliverable**: Identified 1-3 specific root causes with evidence

### Session 2: Deep Dive (1-2 hours)
1. Add Layer 3 instrumentation for identified issues
2. Run instrumented backtest
3. Analyze detailed logs
4. Create Layer 4 isolated tests if needed

**Deliverable**: Confirmed root causes with detailed evidence

### Session 3: Fix Implementation (1-2 hours)
1. Implement fixes for confirmed root causes
2. Re-run backtest
3. Compare with VectorBT baseline
4. Iterate if needed

**Deliverable**: qengine matching VectorBT behavior

---

## Success Criteria

### Minimum Success (Phase 2 Completion)
- ✅ Identified all root causes of discrepancies
- ✅ qengine trade count within 5% of VectorBT (459-505 trades)
- ✅ Exit type distribution matches VectorBT within 10%
- ✅ First 100 trades match VectorBT exactly on entry timing

### Ideal Success (100% Matching)
- ✅ Exact trade count match (482 trades)
- ✅ Exact exit type distribution match
- ✅ All trades match VectorBT on entry/exit timing and prices
- ✅ Performance metrics within 0.1% of VectorBT

---

## Risk Mitigation

**Risk 1**: Multiple interacting root causes
- **Mitigation**: Fix in priority order (TSL first, then re-entry, then signal)
- **Validation**: Re-run after each fix to measure impact

**Risk 2**: VectorBT behavior is not well documented
- **Mitigation**: Run VectorBT comparison script to extract exact behavior
- **Validation**: Document VectorBT behavior as ground truth

**Risk 3**: Time investment vs reward trade-off
- **Mitigation**: Set time box of 4-6 hours total
- **Decision Point**: If not achieving minimum success after 6 hours, document findings and pivot to other Phase 2 tasks

---

## Tools and Scripts

### Diagnostic Scripts to Create
1. `scripts/diagnose_trade_comparison.py` - Layer 1 quick comparison
2. `scripts/trace_bracket_behavior.py` - Layer 3 instrumented backtest
3. `tests/validation/test_bracket_tsl_update.py` - Layer 4 isolated tests

### Existing Resources
- `data/trade_logs/qengine_trades_q1_2024.parquet` - qengine results
- `data/trade_logs/vectorbt_trades_q1_2024.parquet` - VectorBT baseline (if exists)
- `backtest/src/qengine/execution/bracket_manager.py` - TSL/TP implementation
- Previous investigation docs in `projects/crypto_futures/docs/TASK-*_COMPLETION.md`

---

**Plan Created**: 2025-10-28
**Next Session**: Execute Layer 1 diagnostics and identify primary root causes
