# Implementation Plan: Robust Accounting Logic

**Work Unit**: 2025-11-20_01_accounting
**Created**: 2025-11-20
**Estimated Total Effort**: 10-12 hours (2-3 days)

## Project Overview

### Objective
Build proper accounting constraints for the backtest engine supporting **both cash accounts** (simple, no leverage, no shorts) and **margin accounts** (leverage enabled, shorts allowed) to replace the current "unlimited debt" model that allows cash to go to -$652k.

### Critical Bug Being Fixed
**Line 587 in engine.py**:
```python
self.cash += cash_change  # No constraint check! Can go negative!
```

This causes 99.4% difference vs VectorBT in validation tests.

### Scope
**In Scope:**
- Cash account with constraint enforcement (cash >= 0)
- Margin account with NLV/BP/MM calculations
- Unified Position class with cost basis tracking
- Pre-execution order validation (Gatekeeper pattern)
- Exit-first sequencing for capital efficiency
- Integration with existing Broker class
- Comprehensive test suite (unit + integration + validation)

**Out of Scope (Future):**
- Portfolio margin (cross-position margin benefits)
- Options margin (complex multi-leg rules)
- Futures margin (per-contract requirements)
- Partial fills when insufficient capital
- Volume-based liquidity constraints
- T+1 settlement rules

### Success Criteria
- [ ] VectorBT validation within 0.1% (fix 99.4% diff)
- [ ] All 17 existing tests pass (with updates)
- [ ] Bankruptcy Test passes (equity floor at 0)
- [ ] Flipping Test passes (commission tracking)
- [ ] 30x+ speed maintained (no performance regression)
- [ ] Clean code (no technical debt, clear separation)

## Technical Architecture

### Core Design: Policy Pattern

```
Broker
  ├── AccountState
  │     ├── cash: float
  │     ├── positions: dict[str, Position]
  │     ├── policy: AccountPolicy
  │     ├── total_equity property
  │     └── buying_power property
  ├── Gatekeeper
  │     ├── account: AccountState
  │     ├── commission_model: CommissionModel
  │     └── validate_order() → (bool, str)
  └── existing: orders, fills, trades, commission, slippage
```

### Account Policies

**CashAccountPolicy:**
- Buying power = max(0, cash)
- No short selling (positions must be >= 0)
- Constraint: order_cost <= available_cash

**MarginAccountPolicy:**
- Net Liquidating Value: NLV = Cash + Σ(qty × price)
- Maintenance Margin: MM = Σ(|qty × price| × mm_rate)
- Buying Power: BP = (NLV - MM) / im_rate
- Constraint: new_margin_requirement <= buying_power

### Module Structure

```
src/ml4t/backtest/
├── engine.py (modified)
└── accounting/ (NEW package)
    ├── __init__.py
    ├── models.py          # Position class
    ├── policy.py          # AccountPolicy interface + implementations
    ├── account.py         # AccountState
    └── gatekeeper.py      # Order validation
```

## Implementation Phases

### Phase 1: Accounting Infrastructure (3 hours)
Create the accounting package with core classes and cash account support.

**Tasks:**
- TASK-001: Create accounting package structure
- TASK-002: Implement unified Position class
- TASK-003: Create AccountPolicy interface
- TASK-004: Implement CashAccountPolicy
- TASK-005: Write unit tests for accounting package

### Phase 2: Cash Account Integration (3 hours)
Integrate accounting into Broker and add validation.

**Tasks:**
- TASK-006: Update Broker initialization with account_type
- TASK-007: Implement Gatekeeper validation
- TASK-008: Add exit-first order sequencing
- TASK-009: Update existing 17 unit tests
- TASK-010: Run VectorBT validation (fix 99.4% diff)

### Phase 3: Margin Account Support (4 hours)
Add margin account policy and short selling support.

**Tasks:**
- TASK-011: Implement MarginAccountPolicy
- TASK-012: Add short position tracking
- TASK-013: Handle position reversals (long→short)
- TASK-014: Write margin account unit tests
- TASK-015: Create Bankruptcy Test
- TASK-016: Create Flipping Test

### Phase 4: Documentation & Cleanup (2 hours)
Document the new system and finalize.

**Tasks:**
- TASK-017: Update README with account type examples
- TASK-018: Document margin calculations
- TASK-019: Create architecture decision record
- TASK-020: Final cleanup and polish

## Task Details

### TASK-001: Create accounting package structure
**Type**: foundation
**Estimated**: 30 minutes
**Dependencies**: None

**Description:**
Create the `src/ml4t/backtest/accounting/` package with proper structure and exports.

**Acceptance Criteria:**
- [ ] Directory `src/ml4t/backtest/accounting/` created
- [ ] `__init__.py` with proper exports
- [ ] Empty module files: `models.py`, `policy.py`, `account.py`, `gatekeeper.py`
- [ ] Package importable: `from ml4t.backtest.accounting import ...`

**Files to Create:**
- `src/ml4t/backtest/accounting/__init__.py`
- `src/ml4t/backtest/accounting/models.py`
- `src/ml4t/backtest/accounting/policy.py`
- `src/ml4t/backtest/accounting/account.py`
- `src/ml4t/backtest/accounting/gatekeeper.py`

---

### TASK-002: Implement unified Position class
**Type**: feature
**Estimated**: 1 hour
**Dependencies**: TASK-001

**Description:**
Create unified Position class with cost basis tracking, supporting both long and short positions.

**Acceptance Criteria:**
- [ ] Position dataclass with: asset, quantity, avg_entry_price, current_price, entry_time, bars_held
- [ ] `market_value` property (quantity × current_price)
- [ ] `unrealized_pnl` property ((current - avg) × quantity)
- [ ] Supports negative quantities (shorts)
- [ ] Type hints and docstrings complete

**Implementation Notes:**
- Positive quantity = Long position
- Negative quantity = Short position
- Market value is positive for longs, negative for shorts
- Replaces existing Position class in engine.py (will be unified later)

**Files to Modify:**
- `src/ml4t/backtest/accounting/models.py`

---

### TASK-003: Create AccountPolicy interface
**Type**: foundation
**Estimated**: 45 minutes
**Dependencies**: TASK-002

**Description:**
Define AccountPolicy abstract base class for different account types.

**Acceptance Criteria:**
- [ ] Abstract `AccountPolicy` class with ABC
- [ ] Abstract method: `calculate_buying_power(cash, positions) -> float`
- [ ] Abstract method: `allows_short_selling() -> bool`
- [ ] Abstract method: `validate_new_position(asset, qty, price, positions, cash) -> (bool, str)`
- [ ] Type hints and comprehensive docstrings

**Implementation Notes:**
- Use Python ABC (Abstract Base Class)
- Clear docstrings explaining each method's purpose
- Return tuple: (is_valid: bool, reason: str) for validation

**Files to Modify:**
- `src/ml4t/backtest/accounting/policy.py`

---

### TASK-004: Implement CashAccountPolicy
**Type**: feature
**Estimated**: 1 hour
**Dependencies**: TASK-003

**Description:**
Implement cash account constraints: cash >= 0, no shorts, buying_power = cash.

**Acceptance Criteria:**
- [ ] `CashAccountPolicy` class inherits from `AccountPolicy`
- [ ] `calculate_buying_power()` returns max(0, cash)
- [ ] `allows_short_selling()` returns False
- [ ] `validate_new_position()` rejects shorts and checks cash >= order_cost
- [ ] Clear rejection messages for each failure case
- [ ] Type hints complete

**Implementation Notes:**
- Simple implementation (no margin calculations)
- Rejection messages: "Cash accounts do not allow short selling" and "Insufficient cash: need $X, have $Y"
- Order cost = quantity × price (commission added separately)

**Files to Modify:**
- `src/ml4t/backtest/accounting/policy.py`

---

### TASK-005: Write unit tests for accounting package
**Type**: testing
**Estimated**: 1 hour
**Dependencies**: TASK-004

**Description:**
Create comprehensive unit tests for Position, AccountPolicy, and CashAccountPolicy.

**Acceptance Criteria:**
- [ ] Test file: `tests/accounting/test_position.py`
- [ ] Test file: `tests/accounting/test_cash_account_policy.py`
- [ ] Position: test cost basis, market value, unrealized PnL
- [ ] CashAccountPolicy: test buying power, short rejection, cash constraint
- [ ] All tests pass with pytest
- [ ] Coverage >= 90% for accounting package

**Test Cases:**
Position:
- Long position market value and PnL
- Short position market value and PnL
- Cost basis updates when adding to position

CashAccountPolicy:
- Buying power equals cash when positive
- Buying power is zero when cash negative
- Short selling rejected
- Order approved when cash sufficient
- Order rejected when cash insufficient

**Files to Create:**
- `tests/accounting/__init__.py`
- `tests/accounting/test_position.py`
- `tests/accounting/test_cash_account_policy.py`

---

### TASK-006: Update Broker initialization with account_type
**Type**: integration
**Estimated**: 45 minutes
**Dependencies**: TASK-005

**Description:**
Add account_type parameter to Broker.__init__() and create AccountState with appropriate policy.

**Acceptance Criteria:**
- [ ] Broker.__init__() has `account_type: str = "cash"` parameter
- [ ] Broker.__init__() has `initial_margin: float = 0.5` parameter (for margin accounts)
- [ ] Broker.__init__() has `maintenance_margin: float = 0.25` parameter
- [ ] Creates CashAccountPolicy when account_type="cash"
- [ ] Raises ValueError for unknown account_type
- [ ] Creates AccountState and stores as `self.account`
- [ ] Backward compatible (existing tests work with default account_type="cash")

**Implementation Notes:**
```python
if account_type == "cash":
    policy = CashAccountPolicy()
elif account_type == "margin":
    raise NotImplementedError("Margin accounts in Phase 3")
else:
    raise ValueError(f"Unknown account type: {account_type}")

self.account = AccountState(initial_cash, policy)
```

**Files to Modify:**
- `src/ml4t/backtest/engine.py` (Broker.__init__)

---

### TASK-007: Implement Gatekeeper validation
**Type**: feature
**Estimated**: 1.5 hours
**Dependencies**: TASK-006

**Description:**
Create Gatekeeper class to validate orders before execution, checking account constraints.

**Acceptance Criteria:**
- [ ] Gatekeeper class in `gatekeeper.py`
- [ ] `__init__(account: AccountState, commission_model: CommissionModel)`
- [ ] `validate_order(order, price) -> (bool, str)` method
- [ ] Identifies reducing vs opening trades
- [ ] Reducing trades always approved
- [ ] Opening trades validated via policy
- [ ] Commission included in cost calculation
- [ ] Type hints and docstrings complete

**Implementation Notes:**
- Reducing trade: position exists and order has opposite sign
- Opening trade: no position or same sign as existing
- Estimate commission for cost calculation
- Return clear rejection reasons

**Files to Modify:**
- `src/ml4t/backtest/accounting/gatekeeper.py`

**Integration:**
- Create Gatekeeper in Broker.__init__()
- Call gatekeeper.validate_order() in Broker._execute_fill() BEFORE fill

---

### TASK-008: Add exit-first order sequencing
**Type**: feature
**Estimated**: 1 hour
**Dependencies**: TASK-007

**Description:**
Modify Broker.process_pending_orders() to process exits before entries for capital efficiency.

**Acceptance Criteria:**
- [ ] Broker.process_pending_orders() splits orders into exits and entries
- [ ] Exits processed first
- [ ] AccountState.mark_to_market() called after exits
- [ ] Entries processed second with updated buying power
- [ ] Broker._is_exit_order(order) helper method
- [ ] Logic: exit if position exists and order has opposite sign

**Implementation Notes:**
```python
def process_pending_orders(self):
    exits = [o for o in self.pending_orders if self._is_exit_order(o)]
    entries = [o for o in self.pending_orders if not self._is_exit_order(o)]

    for order in exits:
        self._check_and_fill(order)

    self.account.mark_to_market(self._current_prices)

    for order in entries:
        self._check_and_fill(order)
```

**Files to Modify:**
- `src/ml4t/backtest/engine.py` (Broker.process_pending_orders, add _is_exit_order)

---

### TASK-009: Update existing 17 unit tests
**Type**: testing
**Estimated**: 1.5 hours
**Dependencies**: TASK-008

**Description:**
Update existing tests in test_core.py to work with new accounting system.

**Acceptance Criteria:**
- [ ] All 17 tests in tests/test_core.py pass
- [ ] Tests explicitly use account_type="cash"
- [ ] Tests expect order rejections where appropriate
- [ ] Updated assertions for new Position class properties
- [ ] No test relies on unlimited debt behavior

**Expected Changes:**
- Broker initialization: add account_type="cash"
- Some tests may need more initial cash to avoid rejections
- Position assertions use avg_entry_price instead of entry_price
- Tests expecting unlimited orders may need updates

**Files to Modify:**
- `tests/test_core.py`

---

### TASK-010: Run VectorBT validation (fix 99.4% diff)
**Type**: validation
**Estimated**: 1 hour
**Dependencies**: TASK-009

**Description:**
Run VectorBT comparison test and verify cash account mode matches VectorBT within 0.1%.

**Acceptance Criteria:**
- [ ] tests/validation/test_validation.py runs successfully
- [ ] Cash account mode P&L matches VectorBT within 0.1%
- [ ] 50-asset scenario with cash constraints passes
- [ ] Trade count matches VectorBT ±1%
- [ ] No orders execute with negative cash

**Expected Results:**
- Current: -$652,580 (our engine, broken)
- VectorBT: -$3,891 (correct)
- Target: Within $40 of VectorBT (0.1% tolerance)

**Files to Run:**
- `pytest tests/validation/test_validation.py -v`

**If test fails:** Debug and fix accounting logic before proceeding to Phase 3.

---

### TASK-011: Implement MarginAccountPolicy
**Type**: feature
**Estimated**: 2 hours
**Dependencies**: TASK-010

**Description:**
Implement margin account with NLV, MM, BP calculations and leverage support.

**Acceptance Criteria:**
- [ ] `MarginAccountPolicy` class in `policy.py`
- [ ] `__init__(initial_margin=0.5, maintenance_margin=0.25)`
- [ ] `calculate_buying_power()` implements (NLV - MM) / IM formula
- [ ] `allows_short_selling()` returns True
- [ ] `validate_new_position()` checks margin requirement vs buying power
- [ ] Handles both long and short positions correctly
- [ ] Type hints and docstrings complete

**Formulas:**
```
NLV = Cash + Σ(position.market_value)  # Longs positive, shorts negative
MM = Σ(|position.market_value| × maintenance_rate)
BP = (NLV - MM) / initial_rate
```

**Files to Modify:**
- `src/ml4t/backtest/accounting/policy.py`

---

### TASK-012: Add short position tracking
**Type**: feature
**Estimated**: 1 hour
**Dependencies**: TASK-011

**Description:**
Update Broker and AccountState to properly track short positions (negative quantities).

**Acceptance Criteria:**
- [ ] AccountState.apply_fill() handles negative quantities correctly
- [ ] Short positions tracked with negative quantity
- [ ] Cost basis calculation correct for adding to shorts
- [ ] Cash increases when opening shorts (proceeds received)
- [ ] Cash decreases when covering shorts (cost to close)
- [ ] Market value calculation correct for shorts

**Implementation Notes:**
- Opening short: sell first without position (creates negative qty)
- Adding to short: sell more when position already negative
- Covering short: buy when position is negative
- Cost basis only updates when increasing short (more negative)

**Files to Modify:**
- `src/ml4t/backtest/accounting/account.py` (AccountState.apply_fill)
- `src/ml4t/backtest/engine.py` (Broker.submit_order, handle shorts)

---

### TASK-013: Handle position reversals (long→short)
**Type**: feature
**Estimated**: 1.5 hours
**Dependencies**: TASK-012

**Description:**
Implement atomic handling of position reversals (long to short or vice versa).

**Acceptance Criteria:**
- [ ] Gatekeeper detects position reversals
- [ ] Reversal split into: close existing + open new
- [ ] Close portion always executes
- [ ] Open portion validated against buying power
- [ ] Cash account rejects reversals (no shorts allowed)
- [ ] Margin account handles reversals correctly

**Example:**
Long 100 → Sell 200 → Split into:
1. Sell 100 (close long) - always allowed
2. Sell 100 (open short) - needs validation

**Files to Modify:**
- `src/ml4t/backtest/accounting/gatekeeper.py` (validate_order, detect and split reversals)

---

### TASK-014: Write margin account unit tests
**Type**: testing
**Estimated**: 1.5 hours
**Dependencies**: TASK-013

**Description:**
Create comprehensive unit tests for MarginAccountPolicy and short positions.

**Acceptance Criteria:**
- [ ] Test file: `tests/accounting/test_margin_account_policy.py`
- [ ] Test buying power calculation (NLV, MM, BP)
- [ ] Test short selling allowed
- [ ] Test margin requirement validation
- [ ] Test long and short position tracking
- [ ] Test position reversals
- [ ] All tests pass
- [ ] Coverage >= 90%

**Test Cases:**
- Buying power with no positions
- Buying power with long positions
- Buying power with short positions
- Buying power with mixed long/short
- Order approved when sufficient BP
- Order rejected when insufficient BP
- Position reversal validation

**Files to Create:**
- `tests/accounting/test_margin_account_policy.py`

---

### TASK-015: Create Bankruptcy Test
**Type**: testing
**Estimated**: 45 minutes
**Dependencies**: TASK-014

**Description:**
Create Martingale strategy test that doubles down on losses until equity reaches zero.

**Acceptance Criteria:**
- [ ] Test file: `tests/validation/test_bankruptcy.py`
- [ ] Martingale strategy: double position size on each loss
- [ ] Margin account starts with $100k
- [ ] Strategy continues until equity ≈ 0
- [ ] Final equity >= 0 (never goes negative)
- [ ] Orders rejected when BP exhausted
- [ ] Test passes

**Implementation Notes:**
- Start with $100k, initial trade $1k
- Each losing trade: next trade is 2x size
- Eventually runs out of buying power
- Equity should be close to 0 but never negative

**Files to Create:**
- `tests/validation/test_bankruptcy.py`

---

### TASK-016: Create Flipping Test
**Type**: testing
**Estimated**: 45 minutes
**Dependencies**: TASK-015

**Description:**
Create test that alternates long/short every bar to validate commission tracking.

**Acceptance Criteria:**
- [ ] Test file: `tests/validation/test_flipping.py`
- [ ] Strategy: Long 1 share → Short 1 share → Long 1 share... (100 flips)
- [ ] Starting cash: $10,000
- [ ] Commission: $1 per trade
- [ ] Expected cash decrease: ~$200 (100 flips × $2 commission per flip)
- [ ] Test calculates exact commission + spread cost
- [ ] Final cash matches calculation within $1
- [ ] Test passes

**Implementation Notes:**
- Each flip = close position + open opposite = 2 trades = 2× commission
- Track position reversals correctly
- Verify no P&L leakage

**Files to Create:**
- `tests/validation/test_flipping.py`

---

### TASK-017: Update README with account type examples
**Type**: documentation
**Estimated**: 45 minutes
**Dependencies**: TASK-016

**Description:**
Update README.md with clear examples of cash and margin account usage.

**Acceptance Criteria:**
- [ ] Section: "Account Types"
- [ ] Cash account example with code
- [ ] Margin account example with code
- [ ] Explanation of constraints for each type
- [ ] Example of order rejection scenarios
- [ ] API reference for account_type parameter

**Content to Add:**
```python
# Cash Account (default)
broker = Broker(
    initial_cash=100000,
    account_type="cash",  # No leverage, no shorts
    commission_model=PerShareCommission(0.005)
)

# Margin Account
broker = Broker(
    initial_cash=100000,
    account_type="margin",
    initial_margin=0.5,      # 2x leverage
    maintenance_margin=0.25,  # 25% maintenance
    commission_model=PerShareCommission(0.005)
)
```

**Files to Modify:**
- `README.md`

---

### TASK-018: Document margin calculations
**Type**: documentation
**Estimated**: 45 minutes
**Dependencies**: TASK-017

**Description:**
Create detailed documentation of margin calculations and formulas.

**Acceptance Criteria:**
- [ ] Document: `docs/margin_calculations.md`
- [ ] Formulas: NLV, MM, BP with examples
- [ ] Example scenarios with step-by-step calculations
- [ ] Explanation of initial vs maintenance margin
- [ ] Examples of order approval/rejection
- [ ] References to RegT standards

**Content Structure:**
1. Overview of margin accounts
2. Formula definitions (NLV, MM, BP)
3. Example 1: Long-only with leverage
4. Example 2: Long/short portfolio
5. Example 3: Order rejection scenario
6. Reg T reference (50% initial, 25% maintenance)

**Files to Create:**
- `docs/margin_calculations.md`

---

### TASK-019: Create architecture decision record
**Type**: documentation
**Estimated**: 30 minutes
**Dependencies**: TASK-018

**Description:**
Document key architectural decisions made during implementation.

**Acceptance Criteria:**
- [ ] Document: `.claude/memory/accounting_architecture_adr.md`
- [ ] ADR-001: Policy pattern for account types
- [ ] ADR-002: Unified Position class
- [ ] ADR-003: Pre-execution validation (Gatekeeper)
- [ ] ADR-004: Exit-first sequencing
- [ ] Each ADR has: Context, Decision, Consequences

**ADR Format:**
```markdown
## ADR-001: Policy Pattern for Account Types

**Context**: Need to support both cash and margin accounts with different constraints.

**Decision**: Use Strategy/Policy pattern with AccountPolicy interface.

**Consequences**:
- Pros: Clean separation, extensible, clear user intent
- Cons: Slightly more code than single unified approach
- Trade-off: Accepted for better clarity and maintainability
```

**Files to Create:**
- `.claude/memory/accounting_architecture_adr.md`

---

### TASK-020: Final cleanup and polish
**Type**: refactoring
**Estimated**: 1 hour
**Dependencies**: TASK-019

**Description:**
Final pass for code quality, remove dead code, ensure consistency.

**Acceptance Criteria:**
- [ ] All tests passing (17 core + accounting + validation)
- [ ] No commented-out code
- [ ] Consistent naming conventions
- [ ] All docstrings complete
- [ ] Type hints on all functions
- [ ] Remove old Position class from engine.py (unified now)
- [ ] Final performance benchmark (maintain 30x vs Backtrader)
- [ ] git status clean (no uncommitted changes in work)

**Checks:**
- Run pytest with coverage report
- Run mypy type checking
- Check for unused imports
- Verify all TODOs addressed
- Final VectorBT validation pass

**Files to Review:**
- All files in `src/ml4t/backtest/accounting/`
- `src/ml4t/backtest/engine.py`
- All test files

## Dependency Graph

```
TASK-001 (foundation)
  └─→ TASK-002 (Position)
       └─→ TASK-003 (AccountPolicy interface)
            └─→ TASK-004 (CashAccountPolicy)
                 └─→ TASK-005 (unit tests)
                      └─→ TASK-006 (Broker init)
                           └─→ TASK-007 (Gatekeeper)
                                └─→ TASK-008 (exit-first)
                                     └─→ TASK-009 (update tests)
                                          └─→ TASK-010 (VectorBT validation)
                                               └─→ TASK-011 (MarginAccountPolicy)
                                                    ├─→ TASK-012 (short tracking)
                                                    └─→ TASK-013 (reversals)
                                                         └─→ TASK-014 (margin tests)
                                                              ├─→ TASK-015 (bankruptcy)
                                                              └─→ TASK-016 (flipping)
                                                                   └─→ TASK-017 (README)
                                                                        └─→ TASK-018 (margin docs)
                                                                             └─→ TASK-019 (ADR)
                                                                                  └─→ TASK-020 (cleanup)
```

**Critical Path** (must complete in order):
TASK-001 → TASK-002 → TASK-003 → TASK-004 → TASK-005 → TASK-006 → TASK-007 → TASK-008 → TASK-009 → TASK-010

**Parallel Opportunities**:
- After TASK-014: TASK-015 and TASK-016 can run in parallel
- After TASK-016: TASK-017, TASK-018, TASK-019 can run in parallel

## Risk Assessment

### High Risk Items

**TASK-010: VectorBT Validation**
- **Risk**: May not match VectorBT exactly due to subtle differences
- **Impact**: High (blocks Phase 3)
- **Mitigation**: Manual spreadsheet validation for 5-trade scenario, detailed debugging

**TASK-013: Position Reversals**
- **Risk**: Complex logic, easy to get wrong
- **Impact**: Medium (margin accounts broken)
- **Mitigation**: Comprehensive test cases, step-by-step debugging

### Medium Risk Items

**TASK-009: Update 17 Existing Tests**
- **Risk**: Tests may rely on unlimited debt behavior
- **Impact**: Medium (delays Phase 2)
- **Mitigation**: Incremental updates, increase initial cash where needed

**TASK-011: MarginAccountPolicy**
- **Risk**: Margin formulas could be incorrect
- **Impact**: High (incorrect accounting)
- **Mitigation**: Compare to industry standards (RegT), manual calculations

### Low Risk Items

**TASK-001-005**: Foundation tasks are straightforward
**TASK-017-020**: Documentation tasks are low risk

## Quality Gates

### Phase 1 Complete
- [ ] Accounting package structure exists
- [ ] Position class with cost basis works
- [ ] CashAccountPolicy implements constraints
- [ ] Unit tests pass with >= 90% coverage

### Phase 2 Complete
- [ ] Broker uses AccountState
- [ ] Gatekeeper validates orders
- [ ] Exit-first sequencing implemented
- [ ] All 17 existing tests pass
- [ ] VectorBT validation within 0.1%

### Phase 3 Complete
- [ ] MarginAccountPolicy implemented
- [ ] Short positions tracked correctly
- [ ] Position reversals handled
- [ ] Bankruptcy Test passes
- [ ] Flipping Test passes

### Phase 4 Complete
- [ ] README updated with examples
- [ ] Margin calculations documented
- [ ] ADR created
- [ ] Final cleanup done
- [ ] All tests passing

## Performance Targets

### Speed
- **Maintain**: 30x faster than Backtrader
- **Target**: Cash account adds < 5% overhead
- **Target**: Margin account adds < 10% overhead

### Accuracy
- **Critical**: VectorBT validation within 0.1%
- **Critical**: Bankruptcy Test equity >= 0
- **Critical**: Flipping Test commission tracking exact

### Code Quality
- **Target**: >= 85% test coverage
- **Target**: Zero mypy type errors
- **Target**: All docstrings present

## Next Steps After Planning

1. **Review this plan**: Verify it meets requirements
2. **Start Phase 1**: Run `/next` to begin TASK-001
3. **Track progress**: Use `/status` to monitor completion
4. **Adapt as needed**: Adjust plan based on discoveries

## Specialist Agent Recommendations

### Phase 1-2 Tasks
- **TASK-003**: Consider `/agent architect` for AccountPolicy interface design
- **TASK-007**: Consider `/agent code-reviewer` for Gatekeeper security review

### Phase 3 Tasks
- **TASK-011**: Consider `/agent architect` for margin formula validation
- **TASK-013**: Consider `/agent code-reviewer` for position reversal logic review

### Testing Tasks
- **TASK-014**: Consider `/agent test-engineer` for comprehensive margin test strategy
- **TASK-015-016**: Consider `/agent test-engineer` for validation test design

### Documentation
- **TASK-019**: Consider `/agent auditor` for framework compliance validation

*Note: Agent use is optional but recommended for complex or critical tasks.*

---

**Plan Status**: ✅ Complete and ready for execution
**Total Tasks**: 20
**Total Estimated Hours**: 20.5 hours (2.5 days)
**Next Command**: `/next` to start TASK-001
