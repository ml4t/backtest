Here is a review of the `ml4t.backtest` architecture based on your request.

This is a well-structured and comprehensive proposal. The existing abstractions (Broker, DataFeed, event loop) are solid
and provide a strong, testable foundation. Your design philosophy correctly identifies the key gap in the market: a
library that prioritizes **ML-first, stateful logic, and transparency** over the raw vectorized speed of frameworks like
VectorBT.

My recommendations are focused on reinforcing this niche, prioritizing flexibility and usability for the ML
practitioner, and deferring performance optimizations until they are proven necessary.

-----

## Question 1: Strategy Formulation API

**Recommendation:** Option 1A (Event Callbacks with Helper Methods)

**Reasoning:**

This option is the *only one* that fully satisfies your top-ranked goals:

1. **ML-First Design (Goal \#1) & Stateful Support (Goal \#3):** ML strategies are often complex. They involve state
   machines (e.g., "am I in a high-volatility regime?"), dynamic calculations (e.g., "size position based on confidence
   *and* current portfolio risk"), and hybrid logic. A pure Python callback is the *only* way to provide the arbitrary
   flexibility needed for this.
2. **Transparency (Goal \#2):** This API is the easiest to debug. Users can set a breakpoint inside `on_market_data` and
   inspect all variables. This is critical for building trust and ensuring reproducibility.
3. **Live Trading (Goal \#4):** The logic is 1-to-1 portable.

Option 1B (Declarative) is too restrictive and fails Goals \#1 and \#3. Option 1C (Hybrid) fails Goal \#2 by default and
forces complexity (Numba, hot/cold paths) onto the user, contradicting your "pragmatic performance" (Goal \#5) and "
transparency" goals.

**Risks:**

* **Performance:** This is the slowest option. However, given your non-goal of competing with VectorBT and your focus on
  daily/hourly strategies, this is an acceptable trade-off.
* **User Bugs:** Users can introduce look-ahead bias *within their own code* (e.g., by accessing an external,
  non-point-in-time data source). This is a documentation and education challenge, not an architectural one.

**Alternatives:**

* Focus on making Option 1A *excellent* by providing a rich set of helper methods (`self.get_position`,
  `self.buy_percent`, `self.get_unrealized_pnl`, `self.get_last_price`, etc.). This gives users the power of 1A with the
  convenience of 1B's "rules."

-----

## Question 2: ML Signal Data Structure

**Recommendation:** Option 2A (Signals as MarketEvent Attributes)

**Reasoning:**

This is the most pragmatic and user-friendly approach.

* **Simplicity:** It matches the user's mental model. ML practitioners pre-compute predictions and join them to their
  OHLCV data. A `ParquetDataFeed(df, signal_columns=['...'])` API is perfectly intuitive.
* **Zero Overhead:** The strategy logic is simple: `event.signals['my_pred']`. No user-managed caching (as in 2B) is
  required.
* **Live Trading Compatibility:** This design *does not* prevent live trading. It simply moves the complexity of
  asynchronicity to the correct place: **inside the `LiveFeed` implementation**. The `LiveFeed` (e.g.,
  `AlpacaStreamFeed`) will be responsible for buffering market data from one WebSocket and ML signals from another (or
  an API). Only when it has *both* for a given timestamp does it emit a single, complete `MarketEvent` (with the
  `signals` dict populated). This keeps the `Strategy` code 100% portable (Goal \#4).

**Risks:**

* **Inflexibility for Latency Modeling:** This design makes it hard to explicitly model *signal latency* (e.g., "what if
  my signal arrives 50ms after the bar closes?"). This is a niche use case that can be added later (as an advanced
  Option 2B) if required.

**Alternatives:**

* Implement 2A now. It serves 95% of use cases. Add 2B later as an *advanced* feature if users demand it.

-----

## Question 3: Context and Cross-Asset Data

**Recommendation:** Option 3B (Separate Context Object)

**Reasoning:**

The memory inefficiency of Option 3A is a critical flaw for any serious multi-asset backtest. Your "100 assets" use case
makes 3B a requirement, not an option.

* **Memory Efficiency:** Storing VIX once instead of 500 times is a fundamental optimization.
* **Performance:** The proposed timestamp-based caching mechanism in the `Context` object is excellent. It ensures
  context lookups are virtually free (O(1) after the first asset on a given bar).
* **Clarity:** It correctly separates *per-asset* data (signals) from *market-wide* data (context).

I recommend a slight refinement to your proposed API to make it even simpler for the user:

```python
# Engine implementation (simplified)
def _run_loop(self):
    while event := self.clock.get_next_event():
        # Engine resolves context *for* the user
        bar_context = self.context.at(event.timestamp) if self.context else {}

        # Strategy API remains simple
        self.strategy.on_market_data(event, bar_context)


# Strategy API
class MyStrategy(Strategy):
    # User gets a simple dict, not the Context object
    def on_market_data(self, event: MarketEvent, context: dict):
        pred = event.signals['ml_pred']
        vix = context.get('VIX', 50)  # Default if VIX is missing
```

**Risks:**

* **Slight API Complexity:** The user has to learn about the `context` argument. This is a minor, acceptable trade-off
  for the massive performance gain.

-----

## Question 4: Performance Architecture

**Recommendation:** Option 4C (Delay Optimization)

**Reasoning:**

The proposal states: "**We do not have performance data.**" Therefore, any optimization (Option 4B) is premature and a
violation of engineering best practices.

* **Focus on Correctness:** Your priorities are flexibility and transparency. Optimizing with Numba (4B) is a direct
  contradiction to this, as it obscures logic and limits flexibility.
* **Achievable Targets:** Your hypothetical targets (Daily, 100 assets, 5 years in \< 1 minute) involve only \~125,000
  events. A pure Python event loop (Option 4A) can *almost certainly* handle this. The 10M-event scenario (Minute data)
  is the only one at risk, but you should **profile first** to see if it's even a problem.
* **Profile-Driven:** Implement the features (Q1, Q2, Q3). Then, write benchmark scripts for your 4 scenarios. The
  profiler (e.g., `cProfile`, `snakeviz`) will tell you *exactly* where the bottlenecks are. It's often not the event
  loop itself but I/O or data-structure instantiation (e.g., inside `Portfolio`).

**Risks:**

* You may find that the minute-level use case is too slow. This is an acceptable risk. The library's niche can be "
  stateful ML for daily and hourly strategies," which is still incredibly valuable.

-----

## Question 5: Live Trading Portability

**Recommendation:** Yes, the current abstraction (Option 5A) is sufficient.

**Reasoning:**

The `Broker` and `DataFeed` abstract base classes are the correct, industry-standard pattern. They perfectly isolate the
`Strategy` from the environment.

* **No Code Change:** The goal of `strategy=MyStrategy()` being identical for backtest and live is achieved.
* **Async Handling:** As discussed in Q2, the complexity of async signal/market data synchronization belongs *inside*
  the `LiveFeed` implementation. The `Strategy` should remain synchronous and simple.
* **Error Handling:** The `LiveBroker` and `LiveFeed` implementations will need to handle reconnections, API errors,
  etc., and can emit `ErrorEvent` types (which a strategy can *optionally* listen for) without changing the core
  `on_market_data` loop.

**Risks:**

* The implementation of the live components (e.g., `AlpacaStreamFeed`) will be challenging. It will require robust
  buffering, timestamp alignment, and error handling. This is an *implementation* challenge, not an *architectural*
  flaw.

**Alternatives:**

* Do not introduce `async/await` into the `Strategy` class. It would destroy the simplicity of the backtesting loop and
  violate Goal \#4 by making the backtest and live logic paradigms different.

-----

## Question 6: Integration and Trade-offs

**Recommendation:** Prioritize **Flexibility, Simplicity, and Portability** over **Raw Performance**.

**Reasoning:**

Your library's "reason for being" is to serve the ML user who finds VectorBT too restrictive and Backtrader too clunky.
Your architecture should lean into this 100%.

* **What to Prioritize:**

    1. **Flexibility (1A):** The Python callback is non-negotiable.
    2. **Simplicity (2A):** The embedded signal dict is the simplest API for the user.
    3. **Portability (5A):** The `Broker`/`Feed` abstraction must be strictly maintained.
    4. **Scalability (3B):** The `Context` object is a necessary optimization.
    5. **Performance (4C):** Profile last.

* **Specific Trade-offs:**

    * **Live Signals (2A):** Go with 2A (Embedded). It *does not* prevent live streaming; it just correctly assigns the
      async complexity to the `LiveFeed`, not the `Strategy`.
    * **Numba (4B):** Avoid it. It fights your core goals.
    * **Context (3B):** The small complexity of 3B is *worth it* for the 50x memory reduction.
    * **Helper Methods (1A):** These are *essential* for usability. They are part of the "simple API" goal, not a
      performance risk.

-----

## Overall Assessment

### Implementation Priorities

Your proposed phasing is logical. I would refine it as follows:

1. **Phase 1 (Week 1):** **Core ML API (Q1 + Q2)**

    * Implement **Option 1A** (Callbacks).
    * Implement the *essential helper methods* (`get_position`, `buy_percent`, `close_position`, `get_cash`). This is
      critical for usability.
    * Implement **Option 2A** (Embedded Signals) by updating `ParquetDataFeed` to accept `signal_columns`.

2. **Phase 2 (Week 2):** **Multi-Asset Context (Q3)**

    * Implement **Option 3B** (Separate `Context` object) with the timestamp-caching.
    * Update the `BacktestEngine` to pass the `context` dict to `on_market_data` as recommended.

3. **Phase 3 (Week 3):** **Profiling (Q4)**

    * *Now* you have a feature-complete backtester.
    * Write benchmark scripts for your 4 scenarios (plus a 500-asset, 10-year daily stress test).
    * Profile and identify *actual* bottlenecks.

4. **Phase 4 (Week 4+):** **Live Trading & Optimization**

    * Implement `AlpacaBroker` and `AlpacaStreamFeed` (**Q5**). This will be a complex task.
    * *Only if* Phase 3 showed critical bottlenecks, optimize those specific functions.

### Performance Expectations

* **Targets:** Your targets are reasonable. The Daily/100 asset goal of `< 1 minute` should be easily achievable in pure
  Python.
* **When to Profile:** After Phase 2.
* **Pure Python:** Yes, it is acceptable. It is the core of your **Transparency** goal. Do not compromise this unless
  profiling shows the system is unusable for your primary (daily) use case.

### API Design Critique

* The proposed APIs are generally clean and Pythonic.
* **Critique:** My only change is for Q3: Pass `context: dict` to `on_market_data`, not `context: Context`. The engine
  should resolve the timestamp-specific data *before* calling the strategy. This simplifies the strategy-writer's job.
* **Suggestion:** Avoid implementing Option 1B (Declarative) entirely. It will confuse users and create two competing
  APIs. Instead, pour that effort into making the **helper methods** for Option 1A powerful and convenient (e.g.,
  `self.buy_at_signal_threshold(...)`).

### Live Trading Concerns

* **Async Delivery:** As stated, this is the job of the `LiveFeed`.
* **Race Conditions:** Your single-threaded, synchronous event loop *prevents* race conditions in the `Strategy`/
  `Portfolio`. This is a major strength.
* **Buffering:** `LiveFeed` is responsible for this. It must have logic to handle missing signals (e.g., timeout and
  emit `MarketEvent` with empty `signals` dict) or partial data.

### Niche to Fill

Your library should be the **"ML-First Backtrader"** or the **"Stateful, Transparent VectorBT."** It's for users who
have outgrown the inflexibility of vectorized engines and need to test complex, stateful ML models with a focus on
correctness and debuggability before going live. This is a fantastic and underserved niche.