# Framework Comparison Matrix

**Purpose**: Quick reference for framework execution differences
**Created**: 2025-11-16
**Status**: Initial version - will be updated as we verify via source code

---

## Execution Timing

| Framework | Default Fill | Configuration | Look-Ahead Risk | Source Code Reference |
|-----------|--------------|---------------|-----------------|----------------------|
| **VectorBT** | Same-bar close | `price=open.shift(-1)` | 🚨 HIGH | `portfolio/base.py:3245` |
| **Backtrader** | Next-bar open | `coo/coc` flags | ✅ LOW | `brokers/bbroker.py:175-189` |
| **Zipline** | Next-bar open | Not configurable | ✅ LOW | `finance/execution.py` (TBD) |
| **ml4t.backtest** | Next-bar open | TBD | ✅ LOW | `execution/fill_simulator.py` (TBD) |

**Recommended for validation**: All frameworks configured for **next-bar open** execution to avoid look-ahead bias.

---

## Order Types Supported

| Order Type | VectorBT | Backtrader | Zipline | ml4t.backtest |
|------------|----------|------------|---------|---------------|
| Market | ✅ | ✅ | ✅ | ✅ |
| Limit | ✅ | ✅ | ✅ | ✅ |
| Stop | ✅ | ✅ | ✅ | ✅ |
| Stop-Limit | ✅ | ✅ | ✅ | ⏳ TBD |
| Trailing Stop | ✅ (built-in) | ✅ (custom) | ❌ | ⏳ TBD |
| Bracket (OCO) | ❌ | ✅ | ❌ | ⏳ TBD |

---

## Commission Models

| Model Type | VectorBT | Backtrader | Zipline | ml4t.backtest |
|------------|----------|------------|---------|---------------|
| **Percentage** | `fees=0.001` | `setcommission(0.001)` | `PerDollar(0.001)` | `PercentageCommission(0.001)` |
| **Fixed per trade** | `fixed_fees=2.0` | Custom `CommInfoBase` | `PerTrade(2.0)` | Custom |
| **Per share** | ❌ | Custom `CommInfoBase` | `PerShare(0.01)` | Custom |
| **Combined** | `fees=0.001, fixed_fees=2.0` | Custom | Manual | ⏳ TBD |

**Source code to verify**:
- VectorBT: `portfolio/base.py` (fees parameter)
- Backtrader: `commission.py`
- Zipline: `finance/commission.py`
- ml4t.backtest: `execution/commission.py`

---

## Slippage Models

| Model Type | VectorBT | Backtrader | Zipline | ml4t.backtest |
|------------|----------|------------|---------|---------------|
| **Percentage** | `slippage=0.0005` | `slip_perc=0.0005` | N/A | `PercentageSlippage(0.0005)` |
| **Fixed price** | N/A | `slip_fixed=0.01` | `FixedSlippage(spread=0.02)` | `FixedSlippage(0.01)` |
| **Volume-based** | ❌ | Custom | `VolumeShareSlippage()` | ⏳ TBD |

**Note**: Zipline's `FixedSlippage` uses **spread** (buy higher, sell lower), not simple percentage.

---

## Commission & Slippage Order of Operations

| Framework | Sequence | Source Code Reference |
|-----------|----------|----------------------|
| **VectorBT** | 1. Slippage → 2. Commission | `portfolio/nb/from_signals.py` (TBD) |
| **Backtrader** | 1. Commission → 2. Slippage (separate) | `brokers/bbroker.py` (TBD) |
| **Zipline** | TBD | `finance/slippage.py`, `finance/commission.py` (TBD) |
| **ml4t.backtest** | 1. Slippage → 2. Commission | `execution/fill_simulator.py` (TBD) |

**Impact**: Different order can cause $0.01-$1.00 per trade variance.

**TODO**: Read source code to confirm each framework's exact sequence.

---

## Same-Bar Re-Entry

| Framework | Behavior | Configuration | Source Reference |
|-----------|----------|---------------|------------------|
| **VectorBT** | Allowed with `accumulate=True` | Default: `False` | `portfolio/base.py:3245` |
| **Backtrader** | Prevented by default | Can override in strategy | `brokers/bbroker.py` (TBD) |
| **Zipline** | Prevented (next-bar fill) | N/A | Design limitation |
| **ml4t.backtest** | Prevented | TBD | `portfolio/state.py:73` |

**For validation**: Set `accumulate=False` in VectorBT to match other frameworks.

---

## Position Sizing

| Framework | API | Max Position | Fractional Shares |
|-----------|-----|--------------|-------------------|
| **VectorBT** | `size=np.inf` (all cash) | Via `max_size` param | ✅ Yes |
| **Backtrader** | `self.order_target_percent(1.0)` | Via strategy logic | ⏳ Configurable |
| **Zipline** | `order_target_percent(asset, 1.0)` | Via strategy logic | ⏳ Configurable |
| **ml4t.backtest** | `buy_percent(1.0)` | Via broker config | ⏳ Configurable |

---

## Data Requirements

| Framework | Format | Frequency | MultiIndex |
|-----------|--------|-----------|------------|
| **VectorBT** | Series/DataFrame | Any (inferred from `freq`) | ✅ Multi-asset |
| **Backtrader** | Feed objects | Daily/minute/tick | ✅ Via cerebro.adddata() |
| **Zipline** | Bundle system | Daily/minute | ✅ Via context.assets |
| **ml4t.backtest** | Polars/Pandas | Daily/minute | ✅ Multi-asset |

---

## Performance Characteristics (Preliminary)

Based on 30-stock, 5-year backtest:

| Framework | Time | Memory | Throughput | Notes |
|-----------|------|--------|------------|-------|
| **ml4t.backtest** | 0.433s | 9 MB | 11,455 trades/sec | Event-driven + vectorized |
| **VectorBT** | 7.244s | 12 MB | 685 trades/sec | Fully vectorized |
| **Backtrader** | ~3.3s | 18 MB | ~1,500 trades/sec | Pure event-driven |
| **Zipline** | ❌ Failed | N/A | N/A | Bundle data issues |

**Next**: Test at 100, 500, 1000 assets for scalability analysis.

---

## Configuration Equivalence Guide

### Align All Frameworks for Validation

**Goal**: Same fills, same costs, same results

```python
# Common settings
commission_pct = 0.001  # 0.1%
slippage_pct = 0.0005   # 0.05%
initial_capital = 100000

# VectorBT
pf = vbt.Portfolio.from_signals(
    close=data['close'],
    entries=entries,
    exits=exits,
    price=data['open'].shift(-1),  # Next-bar open fill
    fees=commission_pct,
    slippage=slippage_pct,
    accumulate=False,  # No same-bar re-entry
    init_cash=initial_capital,
)

# Backtrader
cerebro.broker = bt.brokers.BackBroker(
    coo=False,  # No cheat-on-open
    coc=False,  # No cheat-on-close
    slip_perc=slippage_pct,
    cash=initial_capital,
)
cerebro.broker.setcommission(commission=commission_pct)

# Zipline
def initialize(context):
    set_commission(us_equities=PerDollar(cost=commission_pct))
    set_slippage(us_equities=FixedSlippage(spread=slippage_pct * 2))
    context.set_portfolio_value(initial_capital)

# ml4t.backtest
engine = BacktestEngine(
    data_feed=feed,
    strategy=strategy,
    broker=SimulationBroker(
        initial_cash=initial_capital,
        commission=PercentageCommission(commission_pct),
        slippage=PercentageSlippage(slippage_pct),
    )
)
```

---

## Known Legitimate Differences

These are **expected** and **documented**:

1. **Fill timing default**: VectorBT (same-bar) vs others (next-bar)
   - **Solution**: Configure VectorBT with `price=open.shift(-1)`

2. **Commission/slippage order**: May differ slightly ($0.01-$1.00 per trade)
   - **Solution**: Document expected variance, verify math is correct

3. **Zipline bundle data**: Uses own price data (can't use custom DataFrame easily)
   - **Solution**: Exclude Zipline from signal-based validation

4. **Fractional shares**: VectorBT allows, others may not
   - **Solution**: Use integer share quantities

5. **Same-bar re-entry**: VectorBT allows with flag, others prevent
   - **Solution**: Set `accumulate=False` in VectorBT

---

## Investigation Priorities

### Phase 1 (Current)
- [x] Document framework locations
- [x] Initial comparison matrix
- [ ] Read VectorBT fill logic (`from_signals.py`)
- [ ] Read Backtrader `_execute()` method
- [ ] Read Zipline execution timing
- [ ] Read ml4t.backtest fill simulator

### Phase 2
- [ ] Verify commission calculation order (all frameworks)
- [ ] Verify slippage application order (all frameworks)
- [ ] Test configuration equivalence
- [ ] Document any $0.01+ variance causes

### Phase 3
- [ ] Limit order logic comparison
- [ ] Stop order logic comparison
- [ ] Bracket order support analysis

---

## Updates

**2025-11-16**: Initial version created
- Framework locations documented
- Preliminary comparison from earlier research
- Marked areas needing source code verification (TBD)

**Next update**: After reading source code for exact fill timing/commission logic.

---

**This matrix will be updated as we verify implementation details via source code reading.**
