This is a high-quality codebase with a clean, separation-of-concerns architecture. However, the performance profile you
are seeing is typical for "pure Python" event-driven systems. The slowness comes from **object churn** and **dictionary
overhead** inside the tightest loops.

To bridge the gap between 31s (Python) and 0.9s (Vectorized), you must minimize what happens inside the `for timestamp`
loop.

Here is my analysis and prioritized recommendations.

-----

### 1\. Profile-Informed Analysis

Your intuition is correct, but the magnitude of the bottlenecks might be surprising.

1. **The DataFeed "Death by 1000 Cuts":**

    * In `datafeed.py`, `ts_df.iter_rows(named=True)` converts performant C/Rust memory (Polars) into heavy Python
      dictionaries.
    * **Impact:** You are creating 500 dictionaries + 2,500 floats *every single bar*. Over 2,520 bars, that is \~1.25
      million dictionary allocations and \~6 million float objects created and immediately discarded.

2. **Engine Dictionary Re-allocation:**

    * In `engine.py`, lines 58-63 create **new** dictionaries (`prices`, `opens`, `highs`, etc.) for every bar.
    * **Impact:** This forces Python to hash 500 strings and allocate memory 5 times per bar.

3. **Dataclass Instantiation Overhead:**

    * In `broker.evaluate_position_rules`, you call `_build_position_state`.
    * This creates a `PositionState` dataclass for every open position, every bar. If you hold 100 assets, that is
      250,000 dataclass instantiations just to check if a stop loss hit.

4. **Dictionary Lookups vs. Array Indexing:**

    * `self._current_prices.get(asset)` is $O(1)$, but the constant factor of hashing a string and looking it up in a
      generic PyDict is significantly slower than `prices[i]` in a NumPy array.

-----

### 2\. Prioritized Optimization Recommendations

#### Priority 1: Hybrid Data Architecture (High Impact, Low Risk)

**Estimated Speedup: 3-5x**

Stop iterating Polars rows and creating dictionaries per bar. Convert your data to **NumPy arrays** (or a 2D aligned
structure) at the start.

* **Concept:** Map every asset string to an integer ID (`0` to `499`).
* **Storage:** Store prices as `(n_bars, n_assets)` NumPy arrays.
* **Access:** In the loop, slice the array by the current time index. `current_prices = all_prices[t_idx]`.

#### Priority 2: Vectorized "Hot Path" Risk Checks (High Impact, Medium Effort)

**Estimated Speedup: 2-3x (Cumulative)**

Your stop-loss logic checks every position individually. Since your rules are mostly static (e.g., "5% stop"), you can
vectorise this check *inside* the event loop.

* **Concept:** Instead of `for pos in positions: rule.evaluate(pos)`, use NumPy boolean masks.
* **Logic:** `hit_stops = (lows[t] < entry_prices * 0.95) & (quantities > 0)`.

#### Priority 3: Internal Structure of Arrays (SOA) (High Impact, High Effort)

**Estimated Speedup: 2-4x (Cumulative)**

Refactor `Broker` to store position state in NumPy arrays instead of a `dict[str, Position]`. Only reconstruct the
`Position` object when the API `get_position()` is explicitly called by the user's strategy.

-----

### 3\. Concrete Implementation Steps

Here is exactly how I would refactor your code.

#### Step A: Refactor DataFeed for Array Access

Change `DataFeed` to pre-align data into NumPy arrays.

```python
# datafeed.py
import numpy as np


class FastDataFeed:
    def __init__(self, prices_df, ...):
        # 1. Get unique sorted timestamps and assets
        self.timestamps = np.array(sorted(prices_df["timestamp"].unique()))
        self.assets = sorted(prices_df["asset"].unique())
        self.asset_map = {asset: i for i, asset in enumerate(self.assets)}

        n_bars = len(self.timestamps)
        n_assets = len(self.assets)

        # 2. Pre-allocate arrays (NaN initialized)
        self.opens = np.full((n_bars, n_assets), np.nan, dtype=np.float64)
        self.closes = np.full((n_bars, n_assets), np.nan, dtype=np.float64)
        # ... highs, lows, volumes ...

        # 3. Fill arrays (Pivot polars to (time x asset))
        # This is a one-time cost at startup
        pivoted = prices_df.pivot(index="timestamp", columns="asset", values="close")
        # Ensure alignment with sorted assets
        self.closes = pivoted.select(self.assets).to_numpy()

        # Repeat for Open, High, Low...

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx >= len(self.timestamps):
            raise StopIteration

        # Return lightweight views, NOT dictionaries
        # We return the index (t) so the broker can slice arrays directly
        t = self._idx
        self._idx += 1
        return self.timestamps[t], t
```

#### Step B: Refactor Engine Loop

Stop creating dictionaries. Pass the time index `t` to the broker.

```python
# engine.py
def run(self):
    # ... setup ...

    # 1. Access raw arrays directly from feed
    opens_arr = self.feed.opens
    closes_arr = self.feed.closes

    for timestamp, t_idx in self.feed:
        # Pass the ROW of prices to broker (View, no copy)
        # current_closes is now a (500,) float64 array
        self.broker._update_time_arrays(
            timestamp,
            t_idx,
            opens_arr[t_idx],
            closes_arr[t_idx],
            # ...
        )

        # Vectorized checks inside broker
        self.broker.evaluate_position_rules_fast()

        # ... logic ...
```

#### Step C: The Broker SOA (Struct of Arrays)

This is the most critical change.

```python
# broker.py
import numpy as np


class Broker:
    def __init__(self, assets: list[str], ...):
        self.assets = assets
        self.asset_map = {a: i for i, a in enumerate(assets)}
        n_assets = len(assets)

        # CORE STATE: NumPy Arrays instead of Dict[Position]
        self.pos_quantities = np.zeros(n_assets, dtype=np.float64)
        self.pos_entry_prices = np.zeros(n_assets, dtype=np.float64)

        # Pre-computed current bar data (updated every bar)
        self._cur_closes = None
        self._cur_lows = None
        self._cur_highs = None

    def _update_time_arrays(self, ts, t_idx, opens, closes, highs, lows, ...):
        self._current_time = ts
        self._cur_closes = closes  # Zero-copy view
        self._cur_highs = highs
        self._cur_lows = lows

    def evaluate_position_rules_fast(self):
        """Vectorized Stop Loss Check"""
        # 1. Identify active positions (Boolean mask)
        active_longs = self.pos_quantities > 0

        if not np.any(active_longs):
            return

        # 2. Check stops using array math (SIMD speed)
        # Assumes a global 5% stop rule for this example
        stop_levels = self.pos_entry_prices * 0.95

        # "Did the low of this bar hit the stop?"
        triggered_mask = (self._cur_lows <= stop_levels) & active_longs

        # 3. Process triggers (only iterate the hits)
        if np.any(triggered_mask):
            triggered_indices = np.where(triggered_mask)[0]
            for idx in triggered_indices:
                asset = self.assets[idx]
                qty = self.pos_quantities[idx]
                # Fallback to standard order logic for the actual exit
                self.submit_order(asset, -qty, ...)

    # Keep API compatibility
    def get_position(self, asset: str) -> Position | None:
        idx = self.asset_map.get(asset)
        qty = self.pos_quantities[idx]
        if qty == 0:
            return None
        # Reconstruct object on demand only
        return Position(asset, qty, self.pos_entry_prices[idx], ...)
```

-----

### 4\. Specific Questions Answered

**1. DataFeed: Pre-build or Iterate?**
Pre-build into **NumPy Arrays** (Struct of Arrays). Do not build a `list[dict]`. A `list[dict]` still incurs hashing
overhead during lookup. Use `(n_bars, n_assets)` arrays.

**2. Position Tracking: `dict` vs Arrays?**
Arrays. `dict[str, Position]` is your biggest bottleneck for scale. Using `pos_qty[i]` and `pos_price[i]` with integer
IDs allows you to use Numba or vectorized NumPy operations for risk checks.

**3. Numba Applicability?**
Numba is **excellent** here, but *only* if you switch to arrays. You cannot Numba JIT `evaluate_position_rules()` as it
currently stands because it uses Python objects (`Position`, `Order`).
Once you switch to `self.pos_quantities` (arrays), you can write a `@jit(nopython=True)` function that takes
`current_prices`, `entry_prices`, and `quantities` and returns a list of indices that need to be closed.

**4. Memory Layout: SoA or AoS?**
**Struct-of-Arrays (SoA)** is strictly better here.

* Good: `closes = np.array([...])` (contiguous floats).
* Why: CPU cache locality. When checking stops, the CPU loads a cache line of prices. In SoA, the next price is right
  there. In AoS (list of dicts or structs), the next price is memory-pointer-jumps away.

**5. Batching Orders?**
Yes. In `_process_orders`, separates the "validation" logic from "execution".

1. Filter: `fills = check_fills_vectorized(pending_orders_arrays, ohlc_arrays)`.
2. Update: Update `pos_quantities` array in one shot for all fills.

-----

### 5\. What NOT to do

1. **Don't use Threads:** Python's GIL will make threading slower for this CPU-bound workload.
2. **Don't optimize `Decimal`:** If you are tempted to use `Decimal` for precision, don't. It is too slow for
   backtesting. Stick to `float64`.
3. **Don't keep `PositionState`:** Completely remove this class from the hot path. Pass raw variables to your rule
   logic.

### 6\. Architecture Consideration: The Hybrid Approach

You asked about a hybrid approach. This is the winning strategy.

* **Pre-Loop (Vectorized):** Calculate indicators (SMA, RSI) using Polars/TA-Lib on the full columns.
* **The Loop (Event-Driven):** Iterate bar-by-bar using the integer index `t`.
* **Inside the Loop (Vectorized):** Use NumPy masking to check stops/limits for *all* assets at step `t` simultaneously.
* **Inside the Loop (Event-Driven):** Only drop into standard Python logic (`if`, `order = ...`) when a trade actually
  occurs.

This allows you to scan 500 assets in microseconds (Vectorized speed) but handle the complex order logic for the 5
assets that actually traded with flexible Python code (Event-driven flexibility).