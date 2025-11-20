# Work Unit: Robust Accounting Logic

**Created**: 2025-11-20
**Status**: Exploration Complete, Ready for Planning
**Estimated Effort**: 10-12 hours (2-3 days)

## Quick Summary

Building proper accounting constraints for the backtest engine to support **both cash accounts** (simple, no leverage) and **margin accounts** (shorts allowed, leverage enabled).

**Critical Bug**: Engine currently executes with unlimited debt (-$652k), failing VectorBT validation by 99.4%.

**Solution**: Policy pattern for account types with pre-execution validation.

## Architecture Overview

```
Broker
  └── AccountState
        ├── CashAccountPolicy: cash >= 0, no shorts
        └── MarginAccountPolicy: NLV/BP/MM, shorts allowed
  └── Gatekeeper (validates orders before execution)
```

## Key Design Decisions

1. **Policy Pattern**: Explicit account_type choice, not forced margin model
2. **Unified Position**: Single class with cost basis tracking for both types
3. **Pre-execution Validation**: Gatekeeper checks constraints before fill
4. **Exit-First Sequencing**: Process exits before entries for capital efficiency
5. **Default to Cash**: Safer, simpler default account type

## Files

- **requirements.md**: Complete functional and non-functional requirements
- **exploration.md**: Detailed codebase analysis and implementation approach
- **metadata.json**: Work unit metadata and phase breakdown

## Implementation Phases

### Phase 1: Accounting Infrastructure (2-3 hours)
- Create `accounting/` package
- Unified Position class with cost basis
- AccountPolicy interface
- CashAccountPolicy implementation
- Unit tests

### Phase 2: Cash Account Integration (2-3 hours)
- Update Broker.__init__() with account_type parameter
- Integrate Gatekeeper validation
- Add exit-first sequencing
- Update 17 existing tests
- VectorBT validation (fix 99.4% diff)

### Phase 3: Margin Account Support (3-4 hours)
- MarginAccountPolicy with NLV/BP/MM
- Short selling support
- Position reversal handling
- Margin-specific tests (Bankruptcy, Flipping)

### Phase 4: Documentation (2 hours)
- README with account type examples
- Margin calculation documentation
- Architecture decision records

## Next Steps

### Run Planning
```bash
/plan
```

This will generate detailed task breakdown with dependencies, creating `plan.md` and `state.json`.

### Start Implementation
```bash
/next
```

After planning, this will start executing tasks in Phase 1.

## Success Criteria

- [ ] VectorBT validation passes (±0.1% for cash accounts)
- [ ] All 17 existing tests pass (with updates)
- [ ] Bankruptcy Test passes (equity floor at 0)
- [ ] Flipping Test passes (commission tracking)
- [ ] 30x+ speed maintained (no performance regression)
- [ ] Clean separation of concerns (no technical debt)

## References

- **External Review**: `.claude/code_review/251120-01/gemini_01.md`
- **Handoff**: `.claude/transitions/2025-11-20/135656.md`
- **Design Questions**: `DESIGN_QUESTIONS.md` (root)
- **Current Tests**: `tests/test_core.py` (17 tests), `tests/validation/test_validation.py`

## Key Insights from Exploration

### What's Wrong Now
1. No cash constraint check (line 587: `self.cash += cash_change`)
2. Simple Position class (no cost basis, no mark-to-market)
3. No order validation (everything executes)
4. Random order processing (no exit-first)

### What's Right (Keep These)
- DataFeed design (Polars-based, clean)
- Order types (MARKET, LIMIT, STOP, etc.)
- Commission/Slippage models (pluggable)
- Fill tracking (complete details)
- Trade recording (P&L, bars held)

### External Review Recommendations (Adapted)
- ✅ Keep custom engine (32x speed advantage worth it)
- ✅ Add proper accounting (NLV, BP, MM for margin accounts)
- ✅ Exit-first sequencing (institutional best practice)
- ✅ Gatekeeper pattern (pre-execution validation)
- ⚠️ Adapted: Policy pattern instead of forced margin model
- ⚠️ Extension: Support both cash and margin accounts

## Testing Strategy

### Unit Tests (Accounting Package)
- Position cost basis updates
- CashAccountPolicy constraints
- MarginAccountPolicy calculations
- Gatekeeper validation logic

### Integration Tests (Broker + Accounting)
- Cash account order rejection
- Margin account short selling
- Exit-first sequencing
- Multi-asset tracking

### Validation Tests (Framework Comparison)
- VectorBT cash constraints (fix 99.4% diff)
- Bankruptcy Test (Martingale strategy)
- Flipping Test (long↔short commission)

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Test breakage (17 tests) | Update incrementally, default to cash accounts |
| Integration complexity | Phased approach (cash first, then margin) |
| Performance regression | Benchmark before/after, profile hot paths |
| Edge cases (reversals) | Comprehensive test suite, spreadsheet validation |

---

**Status**: 📊 Exploration Complete
**Next**: 📋 Run `/plan` to generate implementation plan
**Then**: ⚙️ Run `/next` to start coding
