# Framework Source Code Investigation Protocol

**Status**: ACTIVE - Applies to all validation work
**Created**: 2025-11-16
**Purpose**: Ensure zero tolerance for uncertainty about framework behavior

---

## The Core Principle

**With complete source code available locally, there is ZERO acceptable reason to claim uncertainty about how any benchmark framework works.**

This is not a suggestion. This is a **mandatory protocol** for all cross-framework validation work.

---

## Available Source Code

### Complete Local Access

| Framework | Location | Size | Status |
|-----------|----------|------|--------|
| **Zipline-reloaded** | `resources/zipline-reloaded-main/` | Full source | ✅ Available |
| **Backtrader** | `resources/backtrader-master/` | Full source | ✅ Available |
| **VectorBT OSS** | `resources/vectorbt/` | Full source | ✅ Available (Nov 2025) |
| **VectorBT Pro** | `resources/vectorbt.pro-main/` | Full source | ✅ Available |

### Key Files Reference

**VectorBT OSS/Pro**:
```
vectorbt/portfolio/
├── base.py              # Main Portfolio API (from_signals, from_orders, from_holding)
├── nb/                  # Numba-compiled execution logic
│   ├── from_signals.py  # Vectorized signal-based execution
│   ├── from_orders.py   # Order-based execution
│   └── core.py          # Core portfolio calculations
├── orders.py            # Order types and execution
├── trades.py            # Trade tracking and P&L
└── logs.py              # Order/trade logging
```

**Backtrader**:
```
backtrader/
├── brokers/
│   └── bbroker.py       # BackBroker (order execution, COO/COC, fills)
├── order.py             # Order types and lifecycle
├── cerebro.py           # Main engine orchestration
├── strategy.py          # Strategy base class
└── commission.py        # Commission schemes
```

**Zipline**:
```
zipline/finance/
├── execution.py         # Order placement and fills
├── commission.py        # PerShare, PerTrade, PerDollar models
├── slippage.py          # FixedSlippage, VolumeShareSlippage
└── blotter.py           # Order management and tracking
```

---

## Investigation Protocol

### When Frameworks Produce Different Results

**MANDATORY STEPS** (no shortcuts allowed):

#### 1. Identify the Discrepancy

Be specific:
- ✅ "VectorBT shows 62 trades, ml4t.backtest shows 60 trades"
- ✅ "Final portfolio value differs by $128.50 (0.13%)"
- ❌ "The results don't match" (too vague)

#### 2. Search for Relevant Code

Use `grep` to locate the implementation:

```bash
# Example: Find fill price logic
grep -rn "fill.*price\|execution.*price" resources/vectorbt/vectorbt/portfolio/
grep -rn "fill.*price\|execution.*price" resources/backtrader-master/backtrader/brokers/

# Example: Find same-bar re-entry logic
grep -rn "same.*bar\|re.*entry\|accumulate" resources/vectorbt/vectorbt/portfolio/

# Example: Find commission calculation
grep -rn "commission\|fee.*calc" resources/backtrader-master/backtrader/
```

#### 3. Read the Implementation

**Don't guess - READ THE CODE**:

```bash
# Use the Read tool to examine actual implementation
Read resources/vectorbt/vectorbt/portfolio/base.py

# Look for specific functions
Read resources/backtrader-master/backtrader/brokers/bbroker.py
# Then search within file for relevant methods
```

#### 4. Cite Specific Lines

**Every finding MUST include**:
- File path
- Line number(s)
- Code snippet (if relevant)
- Explanation of behavior

**Example** ✅:
> "VectorBT fills at close price by default (`portfolio/base.py:3245`) because the `price` parameter defaults to the `close` Series. Backtrader fills at next bar's open (`brokers/bbroker.py:467`) unless `coc=True` is set. This explains the $128.50 difference in final value."

**Not acceptable** ❌:
> "The frameworks seem to handle fills differently."

#### 5. Compare Implementations

Create a comparison table:

| Aspect | VectorBT | Backtrader | ml4t.backtest |
|--------|----------|------------|---------------|
| Default fill price | Close (base.py:3245) | Next open (bbroker.py:467) | Next open (fill_simulator.py:123) |
| Same-bar re-entry | Allowed with `accumulate=True` | Prevented by default | Prevented (state.py:73) |
| Commission timing | After fill | Before fill | After fill |

#### 6. Document in Validation Report

Add to the appropriate validation document with:
- Discrepancy description
- Source code evidence
- Conclusion (legitimate difference vs bug)
- Configuration to align behaviors

---

## Example Investigations

### Example 1: Trade Count Difference

**Scenario**: VectorBT shows 62 trades, ml4t.backtest shows 60

**Investigation**:

1. **Search**:
```bash
grep -rn "accumulate\|same.*bar" resources/vectorbt/vectorbt/portfolio/base.py
```

2. **Read**: `resources/vectorbt/vectorbt/portfolio/base.py` lines 3400-3450

3. **Finding**: VectorBT's `accumulate` parameter (line 3245) controls whether multiple entries on the same bar are allowed:
```python
accumulate: tp.ArrayLike = None,  # Whether to accumulate signals
```

4. **Comparison**:
- VectorBT with `accumulate=True`: Allows same-bar re-entry (62 trades)
- VectorBT with `accumulate=False`: One position at a time (60 trades expected)
- ml4t.backtest: Prevents same-bar re-entry in `PositionTracker.update()` (`portfolio/state.py:73`)

5. **Conclusion**: This is a **legitimate configuration difference**. To match ml4t.backtest, set `accumulate=False` in VectorBT.

### Example 2: Fill Price Difference

**Scenario**: Final values differ by $128.50

**Investigation**:

1. **Hypothesis**: Different fill prices

2. **Search**:
```bash
grep -rn "def _execute\|fill.*price" resources/backtrader-master/backtrader/brokers/bbroker.py
```

3. **Read**: `resources/backtrader-master/backtrader/brokers/bbroker.py` lines 450-500

4. **Finding** (line 467):
```python
# Backtrader fills at next bar's open by default
if order.type == Order.Market:
    order.execute(next_bar.open, ...)
```

5. **Compare with VectorBT**: `resources/vectorbt/vectorbt/portfolio/base.py` line 3245:
```python
# VectorBT fills at close by default
price=None,  # Defaults to close prices
```

6. **Conclusion**: **Legitimate design difference**. Backtrader uses next-bar open, VectorBT uses same-bar close. To align:
- Set Backtrader `coc=True` for same-bar close
- Or set VectorBT `price=open.shift(-1)` for next-bar open

### Example 3: Commission Calculation

**Scenario**: Total commission differs by $12.35

**Investigation**:

1. **Read commission models**:
- VectorBT: `resources/vectorbt/vectorbt/portfolio/base.py`
- Backtrader: `resources/backtrader-master/backtrader/commission.py`

2. **Finding**: Order of operations differs
- VectorBT: Slippage first, then commission on slipped price
- Backtrader: Commission on market price, then slippage

3. **Math**:
```python
# VectorBT: commission on (price + slippage)
fill_price = 100.0 * (1 + 0.001) = 100.10  # 0.1% slippage
commission = 100.10 * 100 * 0.001 = $10.01

# Backtrader: commission on base price
commission = 100.0 * 100 * 0.001 = $10.00
# Then slippage applied separately
```

4. **Conclusion**: **Implementation difference**, both are valid. Document expected variance.

---

## Tools for Investigation

### 1. Grep (Fast Search)
```bash
grep -rn "pattern" resources/framework/
grep -A 10 "function_name" file.py  # Show 10 lines after match
grep -B 5 "class.*Order" file.py    # Show 5 lines before match
```

### 2. Read (Examine Files)
```bash
Read resources/vectorbt/vectorbt/portfolio/base.py
Read resources/backtrader-master/backtrader/brokers/bbroker.py (lines 400-500)
```

### 3. Serena (Semantic Search - if available)
```bash
mcp__serena__find_symbol("from_signals", "resources/vectorbt/")
mcp__serena__find_symbol("execute_order", "resources/backtrader-master/")
```

### 4. WebFetch (Official Docs - supplementary)
```bash
WebFetch "https://vectorbt.dev/api/portfolio/" "explain from_signals API"
```

**Priority**: Always check source code FIRST, docs second.

---

## What This Means for Validation Work

### Acceptable Statements ✅

- "VectorBT fills at close (base.py:3245), ml4t.backtest fills at next open (fill_simulator.py:123)"
- "Backtrader prevents same-bar re-entry by default (bbroker.py:580), causing 2 fewer trades"
- "Commission is calculated post-slippage in VectorBT (base.py:3567), pre-slippage in Backtrader (commission.py:45)"

### Unacceptable Statements ❌

- "Unclear how VectorBT handles fills"
- "Not sure why trade counts differ"
- "Need to research Backtrader's execution model"
- "The frameworks probably differ in some way"

**If you catch yourself about to say any of the ❌ statements, STOP and read the source code instead.**

---

## Enforcement

### Code Review Checklist

When reviewing validation work, verify:
- [ ] All discrepancies investigated via source code
- [ ] Specific file paths and line numbers cited
- [ ] Code snippets included where relevant
- [ ] Legitimate differences documented
- [ ] Configuration alignments specified

### Red Flags 🚩

These indicate the protocol was not followed:
- 🚩 Vague statements ("seems to differ")
- 🚩 No file/line citations
- 🚩 Guessing about behavior ("probably", "maybe")
- 🚩 Claiming uncertainty without source code investigation

---

## Memory Persistence

This protocol is documented in:
1. **CLAUDE.md** - Project-level instructions (loaded every session)
2. **This file** - Detailed procedures (referenced during validation)
3. **Work unit state.json** - Active protocol tracking
4. **Handoff documents** - Investigation trail across sessions

**Reinforcement**: Every time you investigate by reading source code, cite this protocol in your notes.

---

## Commitment Statement

**I will:**
- ✅ ALWAYS read source code when frameworks differ
- ✅ ALWAYS cite specific files and line numbers
- ✅ ALWAYS explain implementation differences with evidence
- ✅ ALWAYS document findings in validation reports

**I will NEVER:**
- ❌ Claim uncertainty about framework behavior
- ❌ Guess about implementation details
- ❌ Skip source code investigation
- ❌ Provide explanations without code references

---

**This protocol is non-negotiable for all validation work.**
