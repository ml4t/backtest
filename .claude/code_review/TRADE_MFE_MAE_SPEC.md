# Spec: Add MFE/MAE to Trade Class

**Date**: 2025-11-23
**Priority**: High (data loss bug)
**Effort**: ~30 minutes
**Source**: Cross-library analysis (evaluation + backtest)

## Problem

The `Position` class correctly tracks MFE/MAE during its lifetime:

```python
# types.py:178-181
class Position:
    high_water_mark: float | None = None
    low_water_mark: float | None = None
    max_favorable_excursion: float = 0.0  # Best unrealized return
    max_adverse_excursion: float = 0.0    # Worst unrealized return
```

But when a Position closes and becomes a `Trade`, **MFE/MAE data is lost**:

```python
# types.py:301-317
@dataclass
class Trade:
    asset: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_percent: float
    bars_held: int
    commission: float = 0.0
    slippage: float = 0.0
    entry_signals: dict[str, float] = field(default_factory=dict)
    exit_signals: dict[str, float] = field(default_factory=dict)
    # NO MFE/MAE FIELDS!
```

## Impact

- **Exit efficiency analysis impossible**: Can't answer "Did we capture most of MFE?" or "Did MAE exceed our stop?"
- **Trade optimization blocked**: Can't analyze if TP/SL levels are optimal based on actual excursions
- **TradeAnalyzer limited**: `analytics/trades.py` can't provide MFE/MAE statistics

## Solution

### 1. Add MFE/MAE Fields to Trade

```python
@dataclass
class Trade:
    asset: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_percent: float
    bars_held: int
    commission: float = 0.0
    slippage: float = 0.0
    entry_signals: dict[str, float] = field(default_factory=dict)
    exit_signals: dict[str, float] = field(default_factory=dict)
    # NEW: Preserve MFE/MAE from Position
    max_favorable_excursion: float = 0.0  # Best unrealized return during trade
    max_adverse_excursion: float = 0.0    # Worst unrealized return during trade
```

### 2. Update Position→Trade Conversion

Find where Position is converted to Trade (likely in broker.py or execution code) and ensure MFE/MAE is preserved:

```python
# Wherever this conversion happens:
trade = Trade(
    asset=position.asset,
    entry_time=position.entry_time,
    exit_time=exit_time,
    entry_price=position.entry_price,
    exit_price=exit_price,
    quantity=position.quantity,
    pnl=pnl,
    pnl_percent=pnl_percent,
    bars_held=position.bars_held,
    commission=commission,
    slippage=slippage,
    entry_signals=position.context.get("entry_signals", {}),
    exit_signals=exit_signals,
    # ADD THESE:
    max_favorable_excursion=position.max_favorable_excursion,
    max_adverse_excursion=position.max_adverse_excursion,
)
```

### 3. Enhance TradeAnalyzer

Add MFE/MAE analysis methods to `analytics/trades.py`:

```python
@property
def avg_mfe(self) -> float:
    """Average maximum favorable excursion across trades."""
    if not self.trades:
        return 0.0
    mfes = [t.max_favorable_excursion for t in self.trades]
    return float(np.mean(mfes))

@property
def avg_mae(self) -> float:
    """Average maximum adverse excursion across trades."""
    if not self.trades:
        return 0.0
    maes = [t.max_adverse_excursion for t in self.trades]
    return float(np.mean(maes))

@property
def mfe_capture_ratio(self) -> float:
    """Average ratio of realized return to MFE.

    Values close to 1.0 indicate exits near peak profit.
    Values close to 0.0 indicate exits gave back most gains.
    """
    if not self.trades:
        return 0.0
    ratios = []
    for t in self.trades:
        if t.max_favorable_excursion > 0:
            ratios.append(t.pnl_percent / t.max_favorable_excursion)
    return float(np.mean(ratios)) if ratios else 0.0

@property
def mae_breach_rate(self) -> float:
    """Percentage of trades where final loss exceeded MAE.

    High values may indicate stop losses being hit.
    """
    if not self.trades:
        return 0.0
    losers = [t for t in self.trades if t.pnl < 0]
    if not losers:
        return 0.0
    breaches = sum(1 for t in losers if t.pnl_percent <= t.max_adverse_excursion)
    return breaches / len(losers)
```

Update `to_dict()` to include new metrics:

```python
def to_dict(self) -> dict:
    return {
        # ... existing fields ...
        "avg_mfe": self.avg_mfe,
        "avg_mae": self.avg_mae,
        "mfe_capture_ratio": self.mfe_capture_ratio,
        "mae_breach_rate": self.mae_breach_rate,
    }
```

## Testing

1. **Unit test**: Create trade with known MFE/MAE, verify preserved
2. **Integration test**: Run backtest, verify Trade objects have correct MFE/MAE
3. **TradeAnalyzer test**: Verify new metrics compute correctly

## Files to Modify

1. `src/ml4t/backtest/types.py` - Add fields to Trade class
2. `src/ml4t/backtest/broker.py` (or wherever Position→Trade conversion happens)
3. `src/ml4t/backtest/analytics/trades.py` - Add MFE/MAE analysis methods
4. `tests/` - Add tests for new functionality

## Backward Compatibility

New fields have defaults (`= 0.0`), so existing code creating Trade objects without MFE/MAE will continue to work. This is fully backward compatible.

## Related Work

- **evaluation library**: Price excursion analysis (different feature - for parameter selection)

---

## Future Enhancements (from deleted prototype code)

The following ideas were captured from prototype code in `evaluation/reporting/backtest/` that was deleted (broken imports, wrong location). These could be future enhancements:

### ExitReason Enum

Add structured exit reason tracking:

```python
class ExitReason(str, Enum):
    """Enumeration of trade exit reasons for analysis."""
    SIGNAL = "signal"           # Normal signal-based exit
    STOP_LOSS = "stop_loss"     # Stop-loss triggered
    TAKE_PROFIT = "take_profit" # Take-profit triggered
    TIME_STOP = "time_stop"     # Maximum hold time exceeded
    RISK_RULE = "risk_rule"     # Risk rule triggered
    END_OF_DATA = "end_of_data" # Backtest ended with open position
    UNKNOWN = "unknown"         # Unknown or unspecified
```

Add to Trade class: `exit_reason: ExitReason = ExitReason.UNKNOWN`

### TradeAnalyzer Exit Analysis

```python
def win_rate_by_exit_reason(self) -> dict[str, float]:
    """Win rate grouped by exit reason."""

def avg_hold_time_by_exit_reason(self) -> dict[str, float]:
    """Average bars held grouped by exit reason."""

def pnl_by_exit_reason(self) -> dict[str, float]:
    """Total PnL grouped by exit reason."""
```

### Visualization Ideas (for evaluation library)

When Trade has MFE/MAE, evaluation library can implement:

1. `plot_mfe_mae_scatter()` - Scatter plot of MFE vs MAE per trade
2. `plot_exit_efficiency()` - Histogram of MFE capture ratios
3. `plot_exit_reason_breakdown()` - Pie/bar chart of exit reasons
4. `plot_hold_time_by_outcome()` - Duration analysis winners vs losers

These visualizations should live in **evaluation library** (not backtest) since they're diagnostic/analytical.
