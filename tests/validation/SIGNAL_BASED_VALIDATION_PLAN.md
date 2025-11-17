# Signal-Based Cross-Framework Validation Plan

## Current State

**Existing infrastructure in `tests/validation/`:**
- ✅ `frameworks/` - Adapters for ml4t.backtest, Backtrader, VectorBT, Zipline
- ✅ `bundles/` - Zipline bundle ingest code
- ✅ `base.py` - ValidationResult, TradeRecord, BaseFrameworkAdapter
- ⚠️  Adapters use `strategy_params` (indicators calculated per framework)

**Problem:**
- Current approach has each framework calculate indicators
- Can't prove equivalence if signal generation differs
- Need signal-based approach: pre-calculate signals, feed to all frameworks

## Proposed Modifications

### 1. Add Signal-Based Interface to BaseFrameworkAdapter

**File**: `tests/validation/frameworks/base.py`

Add new method:
```python
class BaseFrameworkAdapter(ABC):

    # Existing method (keep for backward compatibility)
    @abstractmethod
    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_params: dict[str, Any],
        initial_capital: float = 10000,
    ) -> ValidationResult:
        """Run backtest with strategy parameters (old approach)."""
        pass

    # NEW: Signal-based method
    @abstractmethod
    def run_with_signals(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,  # Boolean entry/exit
        initial_capital: float = 10000,
        commission_rate: float = 0.001
    ) -> ValidationResult:
        """
        Run backtest with pre-calculated signals.

        Args:
            data: OHLCV DataFrame (index=datetime)
            signals: DataFrame with boolean columns ['entry', 'exit']
            initial_capital: Starting cash
            commission_rate: Commission as fraction

        Returns:
            ValidationResult with trades and metrics

        Note:
            Signals are PRE-CALCULATED. No indicator computation
            should occur in this method.
        """
        pass
```

### 2. Create Signal Generation Module

**New file**: `tests/validation/signals/generate.py`

```python
"""
Generate pre-calculated signals for framework validation.

Signals are computed ONCE, independently, and saved to disk.
All frameworks load these SAME signals for validation.
"""

import pandas as pd
from pathlib import Path

SIGNAL_DIR = Path(__file__).parent
DATA_SOURCE = Path(__file__).parent.parent.parent.parent.parent / 'projects'

def load_crypto_data() -> dict[str, pd.DataFrame]:
    """Load crypto data from ../../projects/crypto_futures/"""
    crypto_dir = DATA_SOURCE / 'crypto_futures' / 'data'
    # Load BTC, ETH, SOL

def generate_sma_crossover(prices: pd.Series, fast=10, slow=20) -> pd.DataFrame:
    """Generate SMA crossover signals (entry/exit booleans)"""
    sma_fast = prices.rolling(fast).mean()
    sma_slow = prices.rolling(slow).mean()

    entry = (sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))
    exit = (sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1))

    return pd.DataFrame({'entry': entry, 'exit': exit})

def save_signal_set(name: str, data: pd.DataFrame, signals: pd.DataFrame):
    """Save data + signals as pickle"""
    output = {'data': data, 'signals': signals}
    pd.to_pickle(output, SIGNAL_DIR / f'{name}.pkl')
```

### 3. Update Framework Adapters

**Files to modify:**
- `tests/validation/frameworks/qengine_adapter.py` (ml4t.backtest)
- `tests/validation/frameworks/backtrader_adapter.py`
- `tests/validation/frameworks/vectorbt_adapter.py`
- `tests/validation/frameworks/zipline_adapter.py`

**Add to each:**
```python
class ML4TAdapter(BaseFrameworkAdapter):

    # Existing method (keep)
    def run_backtest(self, data, strategy_params, initial_capital):
        # Old approach with strategy parameters
        pass

    # NEW: Signal-based execution
    def run_with_signals(self, data, signals, initial_capital, commission_rate):
        """Execute pre-calculated signals (NO indicator computation)"""

        # Create strategy that ONLY executes signals
        class SignalStrategy(Strategy):
            def __init__(self, signals_df):
                super().__init__()
                self.signals = signals_df

            def on_market_event(self, event):
                ts = event.timestamp
                if ts not in self.signals.index:
                    return

                signal = self.signals.loc[ts]

                # Execute pre-calculated signal (no calculation)
                if signal['entry']:
                    self.order_percent(event.asset_id, 1.0, event.close)
                elif signal['exit']:
                    self.close_position(event.asset_id, event.close)

        # Run backtest with signal strategy
        # Extract trades, return ValidationResult
```

### 4. Create Signal-Based Test Runner

**New file**: `tests/validation/run_signal_validation.py`

```python
"""
Run cross-framework validation with pre-calculated signals.

Usage:
    python run_signal_validation.py --signal btc_sma_crossover
"""

import argparse
from pathlib import Path
import pandas as pd

from frameworks.qengine_adapter import ML4TAdapter
from frameworks.backtrader_adapter import BacktraderAdapter
from frameworks.vectorbt_adapter import VectorBTAdapter
from frameworks.zipline_adapter import ZiplineAdapter

def load_signal_set(name: str) -> dict:
    """Load pre-generated signal set"""
    path = Path(__file__).parent / 'signals' / f'{name}.pkl'
    return pd.read_pickle(path)

def run_validation(signal_name: str):
    # Load signals
    signal_set = load_signal_set(signal_name)
    data = signal_set['data']
    signals = signal_set['signals']

    # Run all frameworks with SAME signals
    adapters = [
        ML4TAdapter('ml4t.backtest'),
        BacktraderAdapter('Backtrader'),
        VectorBTAdapter('VectorBT'),
        ZiplineAdapter('Zipline')
    ]

    results = []
    for adapter in adapters:
        result = adapter.run_with_signals(
            data=data,
            signals=signals,
            initial_capital=100_000,
            commission_rate=0.001
        )
        results.append(result)

    # Compare results
    compare_trades(results)
    compare_pnl(results)
    generate_report(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--signal', required=True)
    args = parser.parse_args()

    run_validation(args.signal)
```

## Implementation Steps

### Phase 1: Signal Generation (2 hours)
1. Create `tests/validation/signals/` directory
2. Implement `generate.py` with SMA crossover signals
3. Generate signal sets from crypto data:
   - `btc_sma_crossover_daily.pkl`
   - `multi_sma_crossover_daily.pkl`

### Phase 2: Update Base Interface (1 hour)
1. Add `run_with_signals()` to `BaseFrameworkAdapter`
2. Keep `run_backtest()` for backward compatibility
3. Document signal-based approach in docstrings

### Phase 3: Implement Signal Execution (4 hours)
1. Update `qengine_adapter.py` - Add signal-based strategy
2. Update `backtrader_adapter.py` - Adapt to signal execution
3. Update `vectorbt_adapter.py` - Load signals directly
4. Update `zipline_adapter.py` - Signal-based algorithm

### Phase 4: Create Test Runner (2 hours)
1. Implement `run_signal_validation.py`
2. Add trade comparison logic
3. Add P&L comparison logic
4. Generate validation report

### Phase 5: Documentation (1 hour)
1. Update `FRAMEWORK_GUIDE.md` with signal-based approach
2. Add examples to README
3. Document expected outputs

**Total: ~10 hours**

## Success Criteria

Validation passes when all frameworks produce:
1. ✅ Same number of trades (exact)
2. ✅ Same trade timestamps (exact)
3. ✅ Same trade prices (within $0.01)
4. ✅ Same final portfolio value (within $1.00)

## Example Workflow

```bash
# 1. Generate signals (once)
cd tests/validation/signals
python generate.py

# 2. Run validation
cd tests/validation
python run_signal_validation.py --signal btc_sma_crossover_daily

# Output:
# ✅ ml4t.backtest: 42 trades, $125,432.15 final value
# ✅ Backtrader: 42 trades, $125,432.15 final value
# ✅ VectorBT: 42 trades, $125,432.15 final value
# ✅ Zipline: 42 trades, $125,432.15 final value
#
# VALIDATION PASSED ✅
```

## Reference Implementation

Use existing examples from ML3T book:
- Zipline: `/home/stefan/ml3t/ch11_strategy_backtesting/code/04_ml4t_workflow_with_zipline`
- Backtrader: `/home/stefan/ml3t/ch11_strategy_backtesting/code/03_backtesting_with_backtrader.py`

Adapt these to:
1. Accept pre-calculated signals instead of calculating indicators
2. Execute signals without modification
3. Return standardized ValidationResult

## Next Steps

1. Review and approve this plan
2. Start Phase 1: Signal generation
3. Test with single signal set (BTC SMA crossover)
4. Expand to multi-asset after single-asset validation works

---

**Key Principle**: Generate signals ONCE, independently. All frameworks execute SAME signals. Compare execution fidelity only.
