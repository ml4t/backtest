# Validation Framework Documentation

**Complete reference materials for multi-platform backtesting validation.**

---

## Documentation Index

### 🚀 Start Here

**New to the validation framework?** Read these in order:

1. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - One-page cheat sheet (⭐ print this!)
2. **[PLATFORM_EXECUTION_MODELS.md](PLATFORM_EXECUTION_MODELS.md)** - How each platform works
3. **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Solutions to common problems

---

## Document Purposes

### QUICK_REFERENCE.md
**When**: You need to look something up fast
**Contains**:
- Platform execution model one-liners
- Critical timezone rule
- Zero trades diagnostic (3 checks)
- Common error messages and fixes
- Expected trade differences table
- Quick diagnostic script

**Time to read**: 2 minutes
**Printable**: Yes (1 page)

### PLATFORM_EXECUTION_MODELS.md
**When**: You need deep understanding of platform behavior
**Contains**:
- Complete execution model documentation for all 4 platforms
- Data format requirements
- Signal handling patterns
- Trade extraction details
- Known quirks and gotchas
- Trade matching guidelines
- Scenario creation checklist

**Time to read**: 20-30 minutes
**Reference**: Keep open while developing scenarios

### TROUBLESHOOTING.md
**When**: Something isn't working
**Contains**:
- Zero trades extracted → 4 common causes + solutions
- Timezone errors → 2 types + fixes
- Bundle issues → 3 Zipline problems + solutions
- Import errors → Path setup
- Trade matching errors → Tolerance configuration
- Performance issues → Optimization tips
- Quick diagnostic script

**Time to read**: 5-10 minutes (scan for your issue)
**Interactive**: Includes copy-paste diagnostic scripts

---

## Learning Path

### Phase 1: Getting Started (30 minutes)

1. Read [QUICK_REFERENCE.md](QUICK_REFERENCE.md) completely
2. Skim [PLATFORM_EXECUTION_MODELS.md](PLATFORM_EXECUTION_MODELS.md) introduction
3. Run a test:
   ```bash
   cd tests/validation
   uv run python runner.py --scenario 001 --platforms ml4t.backtest,vectorbt
   ```
4. If it fails, use [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

### Phase 2: Understanding Platforms (1 hour)

1. Read full [PLATFORM_EXECUTION_MODELS.md](PLATFORM_EXECUTION_MODELS.md)
2. Focus on section for your primary platform
3. Review "Common Gotchas" section
4. Read `../scenarios/scenario_001_simple_market_orders.py` as example

### Phase 3: Creating Scenarios (2 hours)

1. Use scenario creation template from [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
2. Follow TDD methodology (see TASK-001_COMPLETION_REPORT.md)
3. Test with diagnostic script from [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
4. Run across all platforms
5. Document expected differences

### Phase 4: Advanced Topics (ongoing)

1. Custom execution models
2. Intrabar precision
3. Complex order types (bracket, OCO)
4. Portfolio rebalancing
5. Multi-asset scenarios

---

## Quick Start Commands

### Run Tests
```bash
# Navigate to validation directory
cd /home/stefan/ml4t/software/backtest/tests/validation

# Single platform test
uv run python runner.py --scenario 001 --platforms ml4t.backtest

# Multi-platform comparison
uv run python runner.py --scenario 001 --platforms ml4t.backtest,vectorbt,backtrader --report both

# Diagnostic test (if zero trades)
uv run python test_ml4t.backtest_signal_processing.py

# Run troubleshooting diagnostic
uv run python docs/diagnose.py  # (see TROUBLESHOOTING.md for script)
```

### Check Platform Status
```bash
# Verify all platforms available
uv run python -c "
import sys
for pkg in ['ml4t.backtest', 'vectorbtpro', 'backtrader', 'zipline']:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except Exception as e:
        print(f'❌ {pkg}: {e}')
"
```

### Validate Signal Dates
```bash
# Check if signals are in dataset
uv run python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'fixtures'))
sys.path.insert(0, str(Path.cwd() / 'scenarios'))

from market_data import get_ticker_data
from scenario_001_simple_market_orders import Scenario001

data = Scenario001.get_data()
signals = Scenario001.signals
data_dates = data['timestamp'].to_list()

print('Signal Validation:')
for i, sig in enumerate(signals):
    in_data = sig.timestamp in data_dates
    status = '✅' if in_data else '❌'
    print(f'  {i+1}. {sig.timestamp} {status}')
"
```

---

## FAQ

### Q: Why do I get different results across platforms?

**A**: Different execution models! See [Platform Execution Models Comparison](PLATFORM_EXECUTION_MODELS.md#execution-model-comparison).

- VectorBT executes same-bar at close (lookahead bias)
- Others execute next-bar (realistic)
- Backtrader uses open price, others use close

**These differences are expected and correct.**

### Q: Why does Backtrader need special handling?

**A**: Backtrader returns timezone-naive datetimes. See [Backtrader Quirks](PLATFORM_EXECUTION_MODELS.md#backtrader).

Must create dual signal dictionary:
```python
self.signals_naive = {sig.timestamp.replace(tzinfo=None): sig for sig in signals}
self.signals_tz = {sig.timestamp: sig for sig in signals}
```

### Q: How do I know if my timestamps are correct?

**A**: Use the 3-check diagnostic from [QUICK_REFERENCE.md](QUICK_REFERENCE.md#zero-trades-check-these-3-things):

```python
print(f"1. Signal tz: {signal.timestamp.tzinfo}")  # Should be UTC
print(f"2. Data dtype: {data['timestamp'].dtype}")  # Should show UTC
print(f"3. In data: {signal.timestamp in data['timestamp'].to_list()}")  # Should be True
```

### Q: What tolerance should I use for trade matching?

**A**: Depends on comparison type. See [Trade Matching Guidelines](PLATFORM_EXECUTION_MODELS.md#trade-matching-guidelines).

**Comparing different execution models** (VectorBT vs ml4t.backtest):
```python
timestamp_tolerance_seconds = 86400  # ±1 day
price_tolerance_pct = 2.0            # ±2%
```

**Comparing same execution model** (ml4t.backtest vs ml4t.backtest):
```python
timestamp_tolerance_seconds = 60     # ±1 minute
price_tolerance_pct = 0.1            # ±0.1%
```

### Q: Where do I find working examples?

**A**: Check these files:

1. `../scenarios/scenario_001_simple_market_orders.py` - Complete scenario
2. `../test_ml4t.backtest_signal_processing.py` - Diagnostic test
3. `../runner.py` - Integration patterns
4. `../TASK-001_COMPLETION_REPORT.md` - Case study

### Q: How do I add a new scenario?

**A**: Follow this process:

1. Use template from [QUICK_REFERENCE.md](QUICK_REFERENCE.md#scenario-creation-template)
2. **Write failing test first** (TDD Red phase)
3. Implement scenario
4. Verify test passes (TDD Green phase)
5. Document expected platform differences
6. Run across all platforms

See [PLATFORM_EXECUTION_MODELS.md - Scenario Creation Checklist](PLATFORM_EXECUTION_MODELS.md#quick-reference-scenario-creation-checklist)

---

## Maintenance

### When to Update These Docs

**PLATFORM_EXECUTION_MODELS.md**:
- New platform added
- Execution model changed
- New quirk discovered
- Commission/pricing logic modified

**TROUBLESHOOTING.md**:
- New error pattern encountered
- New solution discovered
- Common issue identified

**QUICK_REFERENCE.md**:
- Quick command changed
- New critical rule added
- Common pattern identified

### How to Update

1. Identify which document needs update
2. Add content to appropriate section
3. Update "Last Updated" date
4. Update version number if major changes
5. Update this README if adding new document

### Document Owners

- **PLATFORM_EXECUTION_MODELS.md**: Primary platform reference
- **TROUBLESHOOTING.md**: Support and debugging
- **QUICK_REFERENCE.md**: Quick lookups and patterns
- **README.md** (this file): Documentation index

---

## Related Resources

### Internal Documentation
- `../TASK-001_COMPLETION_REPORT.md` - Timezone issue case study
- `../scenarios/scenario_001_simple_market_orders.py` - Working example
- `../.claude/work/current/005_validation_infrastructure_real_data/` - Work unit docs

### External References
- [VectorBT Pro Docs](https://vectorbt.pro/)
- [Backtrader Documentation](https://www.backtrader.com/docu/)
- [Zipline-Reloaded Docs](https://zipline.ml4trading.io/)
- ml4t.backtest: See `../../../src/ml4t.backtest/` source code

### Framework Standards
- TDD Methodology: Red-Green-Refactor cycle
- Timezone Policy: Always UTC-aware timestamps
- Commission Policy: Document per-platform differences
- Tolerance Policy: Explicit tolerance configuration

---

## Help and Support

### Getting Help

1. **Quick question?** → [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
2. **Something broken?** → [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
3. **Need understanding?** → [PLATFORM_EXECUTION_MODELS.md](PLATFORM_EXECUTION_MODELS.md)
4. **Still stuck?** → Check `../TASK-001_COMPLETION_REPORT.md` for similar issue

### Contributing

Found an issue? Fixed a bug? Learned something new?

**Update the docs!** These are living documents that improve with use.

---

## Document Status

| Document | Version | Last Updated | Status |
|----------|---------|--------------|--------|
| QUICK_REFERENCE.md | 1.0 | 2025-11-04 | ✅ Current |
| PLATFORM_EXECUTION_MODELS.md | 1.0 | 2025-11-04 | ✅ Current |
| TROUBLESHOOTING.md | 1.0 | 2025-11-04 | ✅ Current |
| README.md (this file) | 1.0 | 2025-11-04 | ✅ Current |

---

## Summary

**Three documents, three purposes:**

📄 **QUICK_REFERENCE.md** → Fast lookups (print this!)
📚 **PLATFORM_EXECUTION_MODELS.md** → Deep understanding (read this!)
🔧 **TROUBLESHOOTING.md** → Fix problems (scan this!)

**Start with QUICK_REFERENCE.md and you'll know where to look next.**

---

**Questions?** These docs answer 95% of validation framework questions. If not, update them!

**Last Updated**: 2025-11-04
**Maintainer**: Claude (AI Assistant) + Stefan (Human)
**Framework**: ML4T Validation Infrastructure (Work Unit 005)
