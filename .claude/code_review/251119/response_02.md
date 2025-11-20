The most meaningful performance enhancement is to move from an **Atomic Event Loop** to a **Batch-Event Hybrid
Architecture**.

Your goal is to keep the *usability* of `on_bar(context, event)` while achieving the *speed* of vectorization. The
current bottleneck is not just "Python is slow"; it is that your engine is paying the Python interpreter overhead tax
126,000 times (500 assets × 252 days).

Here are the three specific architectural changes that will yield the highest ROI:

### 1\. Pivot to "Time-Slice" Batching

Instead of the engine iterating over every single market event (tick/bar) for every single asset, the main loop should
iterate over **timestamps**.

* **Current (Slow):**

  ```python
  # 126,000 iterations
  while events_queue:
      event = queue.pop()
      strategy.on_bar(event) # Context switch 126k times
  ```

* **Recommended (Fast):**

  ```python
  # 252 iterations (Daily)
  for timestamp, batch_df in data.group_by("timestamp"):
      # 1. Bulk update portfolio state for ALL assets
      portfolio.mark_to_market(timestamp, batch_df)

      # 2. Pass the entire slice to the strategy
      strategy.on_bar_batch(timestamp, batch_df)
  ```

**Why this wins:** You reduce the interpreter overhead by a factor of N (number of assets). Polars excels here because
slicing a DataFrame by timestamp (`df.filter(pl.col("time") == t)`) is essentially free if the data is sorted.

### 2\. The "Extract -\> Process -\> Wrap" Pattern with Numba

Polars and Numba are powerful individually but don't talk to each other directly. Polars is fast for *data
manipulation* (joins, grouping), while Numba is fast for *calculation* (loops, logic).

Use this pattern for your heavy lifting (e.g., signal generation or complex risk checks):

1. **Extract:** Pull the underlying Numpy array from the Polars Series (zero-copy).
2. **Process:** Pass that array to a Numba-compiled function (`@njit`).
3. **Wrap:** Put the result back into a Polars Series.

<!-- end list -->

```python
import polars as pl
from numba import njit


# The "Hot Path" Logic - Compiled to Machine Code
@njit(nogil=True)
def calc_signals_numba(prices, window):
    signals = np.zeros_like(prices)
    # ... complex looping logic ...
    return signals


class Strategy:
    def on_bar_batch(self, timestamp, batch_df):
        # 1. Extract (Zero-Copy)
        prices_array = batch_df["close"].to_numpy()

        # 2. Process (C-Speed)
        signals = calc_signals_numba(prices_array, 20)

        # 3. Wrap (User-Friendly)
        batch_df = batch_df.with_columns(pl.Series("signal", signals))
```

### 3\. "Lazy" Context Objects

To maintain the user-friendly `context` interface without killing performance, make the context object **lazy**.

Don't pre-calculate every possible value a user *might* ask for. Instead, give the user an object that holds the *index*
of the current time slice.

* **Bad:** Creating a new `Position` object for every asset at every timestep.
* **Good:** A single `PortfolioContext` object that holds big Numpy arrays of positions. When the user asks for
  `context.portfolio["AAPL"]`, it looks up the index for "AAPL" and returns that specific slice of the array.

### Summary of Technologies

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Data Storage** | **Polars** | Fast loading, aligning, and slicing of history. |
| **Outer Loop** | **Python** | Iterates over *Timestamps* (not events). Orchestrates the batch. |
| **Inner Logic** | **Numba** | Handles the "per-asset" logic inside the loop. |
| **State** | **Numpy** | Continuous arrays for Portfolio/Positions (fed into Numba). |

**Recommendation:**
Focus on refactoring your `Engine.run()` method to be a **Time-Slice Loop**. This single change will likely get you 80%
of the performance gain you are looking for, effectively hybridizing the "Backtrader feel" with "VectorBT speed."