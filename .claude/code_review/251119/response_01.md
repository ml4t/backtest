This is a high-quality, comprehensive review package. You have correctly identified that you are trying to fit a square
peg (Cross-Sectional Portfolio Rebalancing) into a round hole (Single-Asset Event-Driven Execution).

Here is my external review, broken down by your requested deliverables.

### 1\. Architectural Assessment

**Verdict:** **Salvageable, but requires a pivot to "Batch-First" Architecture.**

Do not start over. You have built valuable components (Polars data feed, Risk Manager). However, the `BacktestEngine`
and `SimulationBroker` are currently modeled after **Backtrader**, which was designed in a pre-pandas/polars era where
iterating bar-by-bar per asset was standard.

For a modern, 500+ asset ML strategy, **pure event-driven processing (one event per asset per timestamp) is the wrong
abstraction.**

**The Structural Mismatch:**

* **Current:** The engine treats `(AAPL, T1)` and `(MSFT, T1)` as totally separate events.
* **Required:** The engine must treat `T1` as the event, containing a payload of `{AAPL, MSFT, ...}`.

You do not need a separate `PortfolioEngine`. You need to refactor the existing `BacktestEngine` to operate on **Time
Slices**, not atomic Market Events.

### 2\. Performance Analysis

**Why is it 3.7x slower than the manual loop?**

1. **The Python Loop Overhead:**

    * **Manual Loop:** Iterates 252 times (once per day). Inside, it does fast vectorized lookups or bulk list
      comprehensions.
    * **Engine:** Iterates 126,000 times (500 assets \* 252 days).
    * **The Math:** Python function calls have overhead. Dispatching 126,000 events through
      `Engine -> Strategy -> Risk -> Broker` involves millions of stack frame creations.

2. **Context Switching:**
   The Risk Manager and Portfolio likely re-calculate state (or check cache) 126,000 times. In a batch system, you
   update the Portfolio state **once** per timestamp after processing all fills.

**Path to 10x Speedup:**
To beat Backtrader and approach VectorBT speeds, you must minimize the "Inner Loop":

1. **Batch Dispatch:** The Engine loop should iterate over `timestamp`, not `event`.
2. **Vectorized Signals:** If using Polars, pre-calculate as much as possible before the loop.
3. **Bulk Execution:** The Broker should accept a list of orders and match them against a slice of price data in one
   operation, not `for order in orders: try_fill()`.

### 3\. API Design Recommendations

The current API (`buy_percent`) is dangerous for rebalancing because it is stateless regarding the *current* position.

**Recommended API: Target-Based Sizing**
Adopt the Zipline/Quantopian standard, which is the industry standard for a reason.

```python
# The "Correct" Abstraction
def rebalance(self, context, data):
    # 1. Define ideal state (Stateless logic)
    target_weights = {"AAPL": 0.04, "MSFT": 0.04, ...}

    # 2. Execution Engine handles the "Diff" (Stateful logic)
    for asset, weight in target_weights.items():
        self.order_target_percent(asset, weight)


# Under the hood implementation of order_target_percent:
def order_target_percent(self, asset, target_pct):
    target_value = self.portfolio.total_equity * target_pct
    current_value = self.portfolio.positions[asset].value
    delta_value = target_value - current_value
    qty = delta_value / current_price
    self.submit_order(asset, qty)
```

**Handling Asynchronicity:**

* Strategies should implement `on_data_batch(timestamp, data_slice)` rather than `on_market_event`.
* If an asset is missing from `data_slice` (asynchronous), the strategy logic decides whether to hold, close, or ignore.

### 4\. Implementation Roadmap

Here is the prioritized fix list to get you from **Broken** to **Superior**.

#### Phase 1: Correctness (The "Must Haves")

* **Refactor Strategy Helper:** Delete `buy_percent`. Implement `order_target_percent` and `order_target_value`. This
  immediately fixes the position explosion.
* **Fix Execution Delay:** In `SimulationBroker`, create a `process_fills(timestamp)` method. This method looks at *all*
  open orders and fills them against the current timestamp's prices *before* the strategy runs for that timestamp.
* **Batch Orchestration:** Modify `BacktestEngine.run()`:
  ```python
  # Pseudo-code for fixed Engine
  for timestamp, event_batch in data_feed.stream_by_timestamp():
      # 1. Update Portfolio (Mark to Market)
      broker.process_fills(timestamp, event_batch) 
      
      # 2. Strategy Logic (Batch)
      strategy.on_data_batch(timestamp, event_batch)
      
      # 3. Risk/Housekeeping
      portfolio.record_history()
  ```

#### Phase 2: Performance (The "Should Haves")

* **Polars Integration:** Ensure `event_batch` passed to strategy is a Polars slice/DataFrame, not a list of Python
  objects.
* **Bulk Order Processing:** Update `Broker` to match orders using array operations/masking rather than iterating.

### 5\. Code Cleanup Plan

Your "Bloat" assessment is accurate. You are carrying technical debt for features you don't use.

**Delete Immediately:**

* `strategy/adapters.py` & `strategy/*_adapter.py` (1,300 lines): If you want to use Backtrader, use Backtrader. Don't
  maintain a compatibility layer.
* `data/polars_feed.py`: Dead code.
* `reporting/*`: Generate a CSV/Parquet of the trade log. Use a separate Jupyter notebook for analysis. Do not embed
  HTML generation in your core engine.

**Keep & Refactor:**

* `execution/broker.py`: Simplify. Remove complex slippage/market impact models until the basic fill logic works.
* `strategy/base.py`: Strip it down to the essential lifecycle methods (`initialize`, `on_data`, `order_target...`).

### Summary Statistics & Expectation Management

| Metric            | Current          | Target (With Fixes)  | Why?                                 |
|:------------------|:-----------------|:---------------------|:-------------------------------------|
| **Architecture**  | Atomic Event     | Batch Event          | Atomic is O(N\*T), Batch is O(T).    |
| **API**           | Accumulation     | Target/Diff          | Accumulation ignores current state.  |
| **Lines of Code** | 22,000+          | \< 6,000             | 70% is unused boilerplate/reporting. |
| **Speed**         | 3,000 events/sec | \> 50,000 events/sec | Removing Python loop overhead.       |

### Final Recommendation

**Fix the engine, don't write a new one.**
The `Manual Loop` you wrote is essentially what the `BacktestEngine.run()` method *should* look like.

1. Take the logic from `top25_ml_strategy_complete.py`.
2. Move the "Get all events" and "Process all fills" loops *into* `BacktestEngine.run()`.
3. Move the "Calculate DIFFERENCES" logic into `Strategy.order_target_percent`.

This bridges the gap between your robust DataFeed/RiskManager and the performant execution of the manual loop.

### Next Step

Would you like me to generate the refactored `BacktestEngine.run` method and the `order_target_percent` implementation
to replace your current broken components?

Please generate the refactored BacktestEngine.run method and the order_target_percent implementation

Here is the concrete code refactoring to fix your critical issues.

This implementation shifts your architecture from **Atomic Event Processing** (Per-Asset) to **Batch Time-Slice
Processing** (Portfolio-Aware).

### 1\. The Fix for Issue \#1: `Strategy.order_target_percent`

Replace your broken `buy_percent` method with this state-aware logic. This calculates the **difference** between what
you *want* and what you *have*.

```python
# In src/ml4t/backtest/strategy/base.py

def order_target_percent(self, asset_id: str, target_pct: float):
    """
    Place an order to adjust a position to a target percent of portfolio equity.
    
    Fixes Critical Issue #1: 
    Instead of blindly adding to a position, this calculates the delta 
    required to reach the target.
    """
    # 1. Get current portfolio state
    # Assuming self.portfolio updates automatically via the engine loop
    current_equity = self.portfolio.equity
    current_position = self.portfolio.get_position(asset_id)
    current_qty = current_position.quantity if current_position else 0

    # 2. Get the most recent price for this asset
    # (Implementation depends on how your strategy stores current bar data)
    price = self.get_current_price(asset_id)

    if price is None or price <= 0:
        return  # Cannot trade without price

    # 3. Calculate target value and quantity
    target_value = current_equity * target_pct
    target_qty = target_value / price

    # 4. Calculate the DIFFERENCE (The Delta)
    delta_qty = target_qty - current_qty

    # 5. Round to integer (assuming no fractional shares for now)
    # Use simple rounding or floor depending on conservatism
    trade_qty = int(delta_qty)

    # 6. Submit Order
    # Only submit if the quantity is meaningful (avoid noise)
    if abs(trade_qty) > 0:
        side = OrderSide.BUY if trade_qty > 0 else OrderSide.SELL

        order = Order(
            asset_id=asset_id,
            side=side,
            quantity=abs(trade_qty),
            order_type=OrderType.MARKET  # Standard for rebalancing
        )
        self.broker.submit_order(order)


def get_current_price(self, asset_id: str) -> float:
    """Helper to look up price from the current batch context."""
    # This assumes the strategy maintains a cache of the current 
    # timestamp's data, updated by the Engine before calling on_bar
    if asset_id in self._current_market_data:
        return self._current_market_data[asset_id].close
    return None
```

### 2\. The Fix for Issues \#2 & \#3: `BacktestEngine.run` (Batch Mode)

Refactor the engine to group events by timestamp. This ensures that **fills happen before strategy logic**, preventing
the "duplicate order" bug.

```python
# In src/ml4t/backtest/engine.py

from collections import defaultdict


class BacktestEngine:
    # ... existing init ...

    def run(self):
        """
        Execute the backtest using Batch Time-Slice Processing.
        
        Fixes Critical Issues #2 & #3:
        1. Groups events by timestamp (Batching).
        2. Processes Fills FIRST (simulating Market Open/Close mechanics).
        3. Runs Strategy SECOND (seeing correct portfolio state).
        """
        self.is_running = True
        self.broker.start()
        self.strategy.start()

        # Buffer for batching events by timestamp
        current_timestamp = None
        batch_events = []

        # 1. Stream events from DataFeed
        for event in self.data_feed.stream():
            if not self.is_running:
                break

            # If we hit a new timestamp, process the PREVIOUS batch
            if current_timestamp is not None and event.timestamp != current_timestamp:
                self._process_time_slice(current_timestamp, batch_events)
                batch_events = []  # Reset buffer

            current_timestamp = event.timestamp
            batch_events.append(event)

        # Process the final batch
        if batch_events:
            self._process_time_slice(current_timestamp, batch_events)

        self._finalize_backtest()

    def _process_time_slice(self, timestamp, events):
        """
        Core Logic: The sequence here determines correctness.
        """
        # A. Map events for easy lookup
        # This allows the broker/strategy to find 'AAPL' data instantly
        market_map = {e.asset_id: e for e in events if e.type == EventType.MARKET}

        # --- PHASE 1: EXECUTION (Broker) ---
        # Broker looks at OLD orders and tries to fill them using NEW prices.
        # This effectively handles the execution delay correctly.
        # If an order was submitted yesterday, it fills NOW, updating positions.
        self.broker.process_batch_fills(timestamp, market_map)

        # --- PHASE 2: UPDATE STATE ---
        # Mark-to-Market: Update portfolio equity based on new prices
        self.portfolio.update_market_values(market_map)

        # --- PHASE 3: STRATEGY (Decision) ---
        # Strategy now sees the UPDATED portfolio (with Phase 1 fills included).
        # It can essentially do the "Manual Loop" logic here.
        self.strategy._current_market_data = market_map  # Update strategy cache
        self.strategy.on_data_batch(timestamp, market_map)

        # --- PHASE 4: RISK (Validation) ---
        # Check constraints after strategy generates new orders
        # (Optional: depending on where your risk hooks sit)

    def _finalize_backtest(self):
        self.strategy.stop()
        self.broker.stop()
        # Generate summary...
```

### 3\. The Glue: Batch Broker Logic

You need to add `process_batch_fills` to your Broker. This replaces the atomic `on_market_event`.

```python
# In src/ml4t/backtest/execution/broker.py

def process_batch_fills(self, timestamp, market_data_map):
    """
    Attempt to fill ALL open orders against the current market data batch.
    """
    # Iterate over a copy of orders because we might remove them upon fill
    for order_id, order in list(self.open_orders.items()):

        # If we don't have price data for this asset this timestamp, skip
        if order.asset_id not in market_data_map:
            continue

        market_event = market_data_map[order.asset_id]

        # Check fill conditions (Limit, Stop, or Market)
        # This reuses your existing logic, just called in a batch loop
        fill_event = self._check_fill(order, market_event)

        if fill_event:
            # Apply fill immediately so Portfolio is updated before Strategy runs
            self._handle_fill_event(fill_event)

            # If fully filled, remove from open orders
            if order.status == OrderStatus.FILLED:
                del self.open_orders[order_id]
```

### Summary of Changes

1. **Strategy:** Stopped adding positions blindly. Now calculates `Target - Current`.
2. **Engine:** Stopped passing one event at a time. Now gathers a full day's data, fills *yesterday's* orders first,
   then asks the strategy for *today's* decisions.
3. **Performance:** This loop runs 252 times (days) instead of 126,000 times (events). While the inner loops still
   iterate over assets, the overhead of function calls and context switching is reduced by orders of magnitude.

