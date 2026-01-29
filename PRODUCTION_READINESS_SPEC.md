# ml4t-backtest Production Readiness Specification

**Created**: 2026-01-22
**Author**: Claude Code (via Wyden Long-Short project validation)
**Status**: Draft - Ready for Implementation

---

## Executive Summary

This document specifies the changes required to make `ml4t-backtest` production-ready. The issues were identified during validation against VectorBT Pro in the Wyden Long-Short cryptocurrency trading project.

### Critical Issues Identified

| Issue | Severity | Location | Impact |
|-------|----------|----------|--------|
| Sharpe computed from bar-level returns | **CRITICAL** | `analytics/equity.py` | Wrong Sharpe for minute data |
| Hardcoded 252 trading days | **HIGH** | `analytics/metrics.py` | Wrong annualization for crypto (24/7) |
| No integration with ml4t-diagnostic | **MEDIUM** | N/A | Duplicate metric implementations |

### Validation Evidence

From `/home/stefan/clients/wyden/long-short/development/results/ml4t_validation/comparison_summary.csv`:
```
vbt_trades,ml4t_trades,vbt_pnl,ml4t_pnl,vbt_sharpe,ml4t_sharpe
455,455,-10904.87,-10904.87,-0.104,-0.003
```

- **Trade count**: EXACT MATCH (455 vs 455)
- **P&L**: EXACT MATCH (-10904.87)
- **Sharpe**: MISMATCH (-0.104 vs -0.003) ← **This is the problem**

---

## Issue 1: Sharpe Computed from Bar-Level Returns

### Current Behavior (WRONG)

**File**: `/home/stefan/ml4t/libraries/ml4t-backtest/src/ml4t/backtest/analytics/equity.py`

```python
@property
def returns(self) -> np.ndarray:
    """Daily returns."""  # MISLEADING COMMENT!
    if len(self.values) < 2:
        return np.array([])
    return returns_from_values(self.values)  # Returns PER-BAR returns

def sharpe(self, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio."""
    return sharpe_ratio(self.returns, risk_free_rate)  # Computes on bar-level returns!
```

**Problem**: For minute-level data, this computes Sharpe from 525,600 minute-level returns per year, not 365 daily returns. The sqrt(252) annualization is then completely wrong.

### Required Behavior (CORRECT)

1. Aggregate bar-level returns to **daily returns** before computing Sharpe
2. Support calendar-aware daily aggregation:
   - **Stocks**: 252 trading days/year (M-F only, excluding holidays)
   - **Crypto**: 365 days/year (24/7 trading)
3. Use session-aware aggregation for futures (e.g., CME 5pm-4pm CT sessions)

### Implementation Specification

#### Option A: Add `to_daily_returns()` method (Recommended)

```python
def to_daily_returns(
    self,
    calendar: str = "crypto",  # "crypto" | "NYSE" | "CME_Equity" | etc.
    session_aligned: bool = False,
    timezone: str = "UTC",
) -> pd.Series:
    """Aggregate bar-level returns to daily returns.

    Parameters
    ----------
    calendar : str
        Trading calendar to use:
        - "crypto": 365 days/year, 24/7
        - "NYSE", "NASDAQ": US stock market hours
        - "CME_Equity": CME futures sessions
    session_aligned : bool
        If True, align to trading sessions (e.g., CME 5pm-4pm CT)
    timezone : str
        Timezone for aggregation

    Returns
    -------
    pd.Series
        Daily returns with DatetimeIndex (dates only)
    """
    pass
```

#### Option B: Use ml4t-diagnostic (Preferred)

Delegate all metrics computation to `ml4t-diagnostic`:

```python
from ml4t.diagnostic.evaluation.metrics.risk_adjusted import sharpe_ratio, sortino_ratio

def sharpe(self, risk_free_rate: float = 0.0, calendar: str = "crypto") -> float:
    """Annualized Sharpe ratio computed on daily returns."""
    daily_returns = self.to_daily_returns(calendar=calendar)
    annualization = 365 if calendar == "crypto" else 252
    return sharpe_ratio(daily_returns, risk_free_rate, annualization_factor=annualization)
```

---

## Issue 2: Hardcoded 252 Trading Days

### Current Behavior (WRONG)

**File**: `/home/stefan/ml4t/libraries/ml4t-backtest/src/ml4t/backtest/analytics/metrics.py`

```python
TRADING_DAYS_PER_YEAR = 252  # HARDCODED!

def sharpe_ratio(returns: ReturnsLike, risk_free_rate: float = 0.0, annualize: bool = True) -> float:
    # ...
    if annualize:
        sharpe *= math.sqrt(TRADING_DAYS_PER_YEAR)  # Always uses 252!
```

**Problem**: Crypto markets trade 24/7 = 365 days/year. Using sqrt(252) instead of sqrt(365) causes ~17% error in annualized Sharpe.

### Required Behavior (CORRECT)

Make annualization factor configurable:

```python
def sharpe_ratio(
    returns: ReturnsLike,
    risk_free_rate: float = 0.0,
    annualization_factor: float | None = None,  # None = infer from data
    calendar: str | None = None,  # "crypto" | "NYSE" | etc.
) -> float:
    """Calculate Sharpe ratio.

    Parameters
    ----------
    annualization_factor : float | None
        If None, infer from calendar (365 for crypto, 252 for stocks)
    calendar : str | None
        Trading calendar. Overrides annualization_factor.
    """
    pass
```

### Calendar-Aware Annualization Table

| Calendar | Trading Days/Year | sqrt(N) |
|----------|-------------------|---------|
| crypto | 365 | 19.10 |
| NYSE, NASDAQ | 252 | 15.87 |
| CME_Equity | ~250 | 15.81 |
| LSE | ~253 | 15.91 |

---

## Issue 3: No Integration with ml4t-diagnostic

### Current State

`ml4t-backtest` has its own metrics implementation in `analytics/metrics.py`:
- `sharpe_ratio()`
- `sortino_ratio()`
- `maximum_drawdown()`
- etc.

`ml4t-diagnostic` has a more complete implementation in `evaluation/metrics/risk_adjusted.py`:
- `sharpe_ratio()` with bootstrap confidence intervals
- `sortino_ratio()`
- `maximum_drawdown()` with duration and peak/trough dates
- More statistical rigor

### Required Behavior

**Option A: Thin wrapper (Recommended)**

`ml4t-backtest` should be a thin wrapper that:
1. Runs the backtest loop
2. Produces trades and equity curves
3. **Delegates all metrics to ml4t-diagnostic**

```python
# ml4t-backtest/src/ml4t/backtest/result.py

class BacktestResult:
    """Backtest result with trade-level data."""

    def __init__(self, trades: pl.DataFrame, equity: np.ndarray, timestamps: pd.DatetimeIndex):
        self._trades = trades
        self._equity = equity
        self._timestamps = timestamps
        self._daily_returns: pd.Series | None = None

    def to_daily_pnl(self, calendar: str = "crypto", session_aligned: bool = False) -> pl.DataFrame:
        """Get daily P&L. Uses existing implementation."""
        pass

    def to_daily_returns(self, calendar: str = "crypto") -> pd.Series:
        """Get daily returns for analytics."""
        if self._daily_returns is None:
            daily_pnl = self.to_daily_pnl(calendar=calendar)
            self._daily_returns = daily_pnl["return"].to_pandas()
        return self._daily_returns

    def metrics(self, calendar: str = "crypto", confidence_intervals: bool = False) -> dict:
        """Compute all metrics via ml4t-diagnostic.

        Returns
        -------
        dict
            {
                "sharpe_ratio": float,
                "sortino_ratio": float,
                "max_drawdown": float,
                "max_drawdown_duration": int,
                "calmar_ratio": float,
                "annual_return": float,
                "annual_volatility": float,
                ...
            }
        """
        from ml4t.diagnostic.evaluation.metrics.risk_adjusted import (
            sharpe_ratio, sortino_ratio, maximum_drawdown
        )

        daily_returns = self.to_daily_returns(calendar=calendar)
        ann_factor = 365 if calendar == "crypto" else 252

        return {
            "sharpe_ratio": sharpe_ratio(
                daily_returns,
                annualization_factor=ann_factor,
                confidence_intervals=confidence_intervals,
            ),
            "sortino_ratio": sortino_ratio(daily_returns, annualization_factor=ann_factor),
            "max_drawdown": maximum_drawdown(daily_returns),
            # ... more metrics
        }
```

**Option B: Remove duplicate metrics code**

Delete `ml4t-backtest/analytics/metrics.py` entirely and require `ml4t-diagnostic` as a dependency.

---

## Implementation Checklist

### Phase 1: Fix Daily Returns (Critical)

- [ ] Add `to_daily_returns()` method to `EquityCurve` class
- [ ] Support calendar-aware aggregation (crypto=365, stocks=252)
- [ ] Update `sharpe()` method to use daily returns
- [ ] Update `sortino()` method to use daily returns
- [ ] Add unit tests with minute-level data

### Phase 2: Integrate ml4t-diagnostic (High)

- [ ] Add `ml4t-diagnostic` as optional dependency
- [ ] Create `metrics()` method that delegates to ml4t-diagnostic
- [ ] Deprecate local metrics implementations
- [ ] Update documentation

### Phase 3: Validate Against VBT Pro (High)

- [ ] Run Wyden validation suite
- [ ] Confirm trade count parity
- [ ] Confirm P&L parity
- [ ] Confirm Sharpe parity (within 0.01)

### Phase 4: Validate Rust Implementation (Medium)

- [ ] Ensure backtest-rs returns daily returns (not bar returns)
- [ ] Add calendar-aware annualization
- [ ] Validate against Python ml4t-backtest
- [ ] Performance benchmark

---

## Test Cases

### Test 1: Sharpe from Minute Data

```python
def test_sharpe_minute_data():
    """Sharpe should be same whether computed from minute or daily data."""
    # Generate 1 year of minute data
    n_bars = 525600  # 365 * 24 * 60
    returns_minute = np.random.normal(0.0001, 0.001, n_bars)  # Per-minute returns

    # Aggregate to daily
    returns_daily = returns_minute.reshape(365, 1440).sum(axis=1)  # 365 days

    # Compute Sharpe both ways
    sharpe_from_minute = sharpe_ratio(returns_minute, annualization_factor=525600)  # WRONG
    sharpe_from_daily = sharpe_ratio(returns_daily, annualization_factor=365)  # CORRECT

    # They should be approximately equal
    assert abs(sharpe_from_minute - sharpe_from_daily) < 0.01
```

### Test 2: Calendar-Aware Annualization

```python
def test_calendar_annualization():
    """Crypto should use 365, stocks should use 252."""
    daily_returns = np.random.normal(0.001, 0.02, 252)

    sharpe_crypto = sharpe_ratio(daily_returns, calendar="crypto")  # sqrt(365)
    sharpe_stocks = sharpe_ratio(daily_returns, calendar="NYSE")    # sqrt(252)

    # Crypto annualization is higher
    expected_ratio = np.sqrt(365) / np.sqrt(252)  # ~1.20
    assert abs(sharpe_crypto / sharpe_stocks - expected_ratio) < 0.01
```

### Test 3: VBT Pro Parity

```python
def test_vbt_pro_parity():
    """Sharpe should match VBT Pro within 0.01."""
    # Load test data from Wyden project
    ohlcv = pl.read_parquet("test_data/btc_1min.parquet")
    signals = pl.read_parquet("test_data/btc_signals.parquet")

    # Run ml4t-backtest
    result = backtest(ohlcv, signals, direction="short", tp=0.03, tsl=0.015)
    ml4t_sharpe = result.metrics(calendar="crypto")["sharpe_ratio"]

    # Run VBT Pro (from saved results)
    vbt_sharpe = -0.104  # From validation file

    # Should match
    assert abs(ml4t_sharpe - vbt_sharpe) < 0.01
```

---

## Reference Implementations

### ml_tools (Working Implementation)

**File**: `/home/stefan/quant/ml_tools/src/ml_tools/backtest/report.py`

```python
from empyrical import sharpe_ratio, sortino_ratio, max_drawdown

class BacktestReport:
    def __init__(self, daily_returns: pd.DataFrame, ...):
        self.daily_returns = daily_returns  # Already daily!

    def compute_metrics(self):
        return {
            "sharpe": self.daily_returns.apply(ep.sharpe_ratio),  # Uses empyrical
            ...
        }
```

### ml4t-diagnostic (Target API)

**File**: `/home/stefan/ml4t/libraries/ml4t-diagnostic/src/ml4t/diagnostic/evaluation/metrics/risk_adjusted.py`

```python
def sharpe_ratio(
    returns: Union[pl.Series, pd.Series, NDArray],
    risk_free_rate: float = 0.0,
    annualization_factor: float | None = None,  # Configurable!
    confidence_intervals: bool = False,
    ...
) -> float | dict:
    """Sharpe with optional confidence intervals."""
    pass
```

---

## Questions for User

1. **Backward Compatibility**: Should we deprecate the old API or break it immediately?
2. **Default Calendar**: Should the default be "crypto" or require explicit specification?
3. **ml4t-diagnostic Dependency**: Required or optional?
4. **Rust Implementation**: Should backtest-rs also delegate to ml4t-diagnostic (via Python bridge)?

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/ml4t/backtest/analytics/equity.py` | Add `to_daily_returns()`, fix `sharpe()` |
| `src/ml4t/backtest/analytics/metrics.py` | Add calendar-aware annualization |
| `src/ml4t/backtest/result.py` | Add `metrics()` method |
| `pyproject.toml` | Add ml4t-diagnostic dependency |
| `tests/test_metrics.py` | Add VBT Pro parity tests |

---

## Appendix: VBT Pro Sharpe Calculation

VBT Pro computes Sharpe from **daily returns**:

```python
# VBT Pro internal (simplified)
daily_returns = portfolio.returns().resample('D').sum()
sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)  # or 365 for crypto
```

The key is the `.resample('D').sum()` which aggregates bar-level returns to daily returns BEFORE computing Sharpe.
