Here is a detailed analysis of your backtesting engine draft, comparing it to the vision outlined in the product design
documents, along with a concrete action plan for improvement.

### 🌎 Executive Summary: Analysis at a Glance

Your initial draft has **excellent bones but a flawed skeleton**.

You have implemented the *pluggable components* of the PRD vision (SOTA, OPUS, GEM) beautifully. The separation of
concerns in the `execution` module—with distinct, pluggable models for `FillSimulator`, `CommissionModel`,
`SlippageModel`, and `TradeTracker`—is outstanding and perfectly aligns with the "Extensible Realism" principle. The use
of Polars in the data feeds is also a major win.

However, the core architecture contains **three critical design flaws** that prevent it from being a true event-driven
engine.

1. **It is Data-Feed-Driven, not Event-Driven:** The main `engine.py` loop is a simple `while` loop over a *single* data
   feed. This makes it impossible to handle asynchronous events (like ML signals, corporate actions, or data from a
   second asset) in a point-in-time correct manner.
2. **It Has Two "Brains":** There are two sources of truth for cash and positions: `execution.PositionTracker` (used by
   the Broker) and `portfolio.Portfolio` (used by the Engine/Strategy). These will inevitably diverge and lead to
   critical accounting errors.
3. **It Has Two "Hearts":** The code contains two separate priority queues: one in `core.Clock` and another in
   `core.EventBus`. The engine uses the `EventBus` but drives its loop from the `DataFeed`, rendering the `Clock`'s true
   purpose (a global event sequencer) useless.

The code *has* the right pieces, but they are assembled incorrectly. The action plan below focuses on re-wiring these
core components to build the robust, event-driven "heart" and "brain" envisioned in the PRDs.

-----

### 👍 Strengths of the Current Draft

* **Excellent Separation of Concerns (in `execution`):** The `SimulationBroker` is not a "god object." It correctly
  composes specialized components like `FillSimulator`, `OrderRouter`, `PositionTracker`, and `BracketOrderManager`.
  This is a clean, modern design.
* **Pluggable Realism:** The `CommissionModel` and `SlippageModel` systems are fantastic. Providing `VectorBTCommission`
  and `VectorBTSlippage` is a brilliant move for validation and for attracting your target "pro-am" audience who may be
  migrating.
* **Modern Data Layer:** The `data.feed` module correctly uses Polars for `ParquetDataFeed` and `CSVDataFeed`. This
  directly implements the "Polars-First" vision from SOTA.
* **Efficient Trade Logging:** `execution.trade_tracker.py` is a high-performance, well-designed component. It correctly
  understands that "Fills" are used for accounting, while "Trades" (entry + exit) are needed for performance analysis,
  and it builds the latter from the former.
* **Strong Asset Modeling:** `core.assets.py` shows a deep understanding of the problem domain. It correctly identifies
  the unique properties of futures, options, and crypto (e.g., `contract_size`, `expiry`, `maker_fee`), which sets the
  stage for v1/v2 features.
* **Standardized Reporting:** The `reporting` module, especially `ParquetReportGenerator`, is a perfect implementation
  of the PRD vision. It creates a standard, machine-readable artifact that separates simulation from analysis.

-----

### 📉 Weaknesses & Critical Design Flaws

1. **[CRITICAL FLAW] Data-Feed-Driven Loop:**

    * **Problem:** The main `BacktestEngine.run()` loop is driven by `while not self.data_feed.is_exhausted:`. This is a
      single-feed-driven loop.
    * **Why it's Wrong:** This architecture *cannot* fulfill the PRD vision for a leakage-safe ML engine. If you have a
      `ParquetSignalSource` that needs to deliver a signal at `10:30:05`, but your main `MarketEvent` feed only has bars
      at `10:30:00` and `10:31:00`, your signal will be processed at the *wrong time* (batched with the 10:31:00 bar).
    * **PRD Vision:** The `Clock` (as described in SOTA/OPUS) should own a **single global priority queue (`heapq`)**.
      The main loop should be `while (event := self.clock.get_next_event()):`. The `Clock` would be primed with the
      *first* event from *all* sources (market data, signal data, corporate actions) and would *always* return the next
      event in true chronological order, regardless of its source.

2. **[CRITICAL FLAW] Dual State Management (Two "Brains"):**

    * **Problem:** `execution.broker.SimulationBroker` creates its own `PositionTracker` (`self.position_tracker`). The
      `BacktestEngine` creates a `SimplePortfolio` (`self.portfolio`). Both objects track cash and positions
      independently.
    * **Why it's Wrong:** This is a race condition waiting to happen. The Broker will think you have X cash, while the
      Portfolio thinks you have Y. The strategy will make decisions based on the Portfolio's state, but the Broker will
      execute based on its *own* state. This is the \#1 cause of impossible-to-debug backtest errors.
    * **PRD Vision:** There must be a **single source of truth for state**. The `BacktestEngine` should instantiate
      *one* `Portfolio` object. This *single instance* must be passed to both the `Strategy` (
      `strategy.portfolio = ...`) and the `Broker` (`broker.portfolio = ...`). The Broker *reads* from this portfolio to
      check for cash/margin and *publishes* `FillEvent`s, which the `Portfolio` *subscribes* to in order to update
      itself.

3. **[CRITICAL FLAW] Redundant Queues (Two "Hearts"):**

    * **Problem:** The code has two, independent event queues: `core.clock.Clock._event_queue` and
      `core.event.EventBus._queue`.
    * **Why it's Wrong:** It's confusing and shows a misunderstanding of the architecture. The `Clock` *is* the event
      bus and the main loop. The `EventBus` class is completely redundant if the `Clock` is implemented correctly.
    * **PRD Vision:** The `Clock` *is* the event bus. It handles time advancement, queueing, and dispatching.

4. **`AssetSpec` is a "God Object":**

    * **Problem:** `core.assets.AssetSpec` contains complex logic like `get_margin_requirement()` and `calculate_pnl()`.
    * **Why it's Wrong:** This violates the Single Responsibility Principle. An `AssetSpec` should be a "dumb" data
      container that holds *properties* (e.g., `contract_size`, `tick_size`, `margin_rate`). The *logic* for calculating
      P\&L or margin belongs in the `Portfolio` module, which would *read* those properties from the spec.
    * **PRD Vision:** A clean, modular design. `AssetSpec` holds data; `Portfolio` holds logic.

5. **Confusing Portfolio Layer:**

    * **Problem:** The distinction between `Portfolio`, `SimplePortfolio`, and `PortfolioAccounting` is unclear.
      `SimplePortfolio` inherits from `Portfolio`, and `PortfolioAccounting` *composes* a `Portfolio`.
    * **Why it's Wrong:** It's redundant and hard to maintain.
    * **PRD Vision:** A single, authoritative `Portfolio` class that handles state, accounting, and P\&L.

6. **Corporate Actions are an Afterthought:**

    * **Problem:** The `engine.py` loop *manually* checks the date and calls
      `self.corporate_action_processor.process_actions(...)`.
    * **Why it's Wrong:** This is not event-driven. A corporate action is just another event that happens at a specific
      time.
    * **PRD Vision:** A `CorporateActionFeed` should be added to the `Clock`. A `CorporateActionEvent` (e.g., a "
      SplitEvent" on its `ex_date`) should be pulled from the `Clock`'s priority queue and processed in its correct
      chronological order, just like any other event.

-----

## 🛠️ Detailed Action Plan for Refactoring to PRD Vision

This plan prioritizes fixing the critical architectural flaws first, then aligning with the Pro-Am target audience.

### Phase 1: Fix the Engine's "Heart" (The Core Loop)

**Objective:** Implement the true, multi-source, event-driven loop from the PRDs.

1. **Deprecate `EventBus`:** Delete `core/event.py:EventBus`. Its functionality will be merged into the `Clock`.
2. **Elevate `Clock`:** The `core/clock.py:Clock` will become the central component.
    * Add subscriber methods to `Clock`: `subscribe(self, event_type: EventType, handler: Callable)`.
    * Store subscribers in a `dict`: `self._subscribers: dict[EventType, list[Callable]] = defaultdict(list)`.
3. **Refactor `BacktestEngine`:**
    * Remove `self.event_bus`.
    * The `Engine` *owns* the `Clock`: `self.clock = Clock(...)`.
    * `_setup_event_handlers()` will now call `self.clock.subscribe(...)` to register the `Strategy`, `Broker`, and
      `Portfolio`.
    * The `BacktestEngine` must pass *all* data/signal feeds to the `Clock` (e.g., `self.clock.add_data_feed(...)`,
      `self.clock.add_signal_source(...)`).
4. **Rewrite the `BacktestEngine.run()` Loop:**
    * **DELETE** the old loop: `while not self.data_feed.is_exhausted: ...`
    * **REPLACE** with the PRD-vision loop:
      ```python
      def run(self, ...):
          # ... (initialization) ...
          # Priming is handled by clock.add_data_feed()
          
          while True:
              # 1. Get the next event from the *single* priority queue
              event_tuple = self.clock.get_next_event()
              if event_tuple is None:
                  break # All feeds are exhausted
              
              # 2. Extract event (Clock's get_next_event should return the event)
              event = event_tuple 

              # 3. Log progress (optional)
              self.events_processed += 1
              if self.events_processed % PROGRESS_LOG_INTERVAL == 0:
                   logger.info(f"Processed {self.events_processed:,} events. Time: {event.timestamp}")

              # 4. Dispatch event to all subscribers
              handlers = self.clock._subscribers.get(event.event_type, [])
              for handler in handlers:
                  handler(event) # e.g., strategy.on_event(event), broker.on_event(event)
          
          # ... (finalization) ...
      ```
    * **Note:** The `Clock`'s `get_next_event` method (which you've already written) correctly uses `heapq.heappop` and
      then *replenishes* the queue from the *same source* that event came from. This is the correct implementation.

### Phase 2: Fix the Engine's "Brain" (State Management)

**Objective:** Create a single, authoritative source of truth for cash and positions.

1. **Deprecate `execution.PositionTracker`:** Delete this class. Its function will be absorbed by the `Portfolio`.
2. **Consolidate `portfolio` Layer:**
    * Refactor `portfolio.Portfolio` to be the single, authoritative class.
    * Merge the best parts of `PortfolioAccounting` (history tracking, metrics) and `SimplePortfolio` into it.
    * Delete `SimplePortfolio` and `PortfolioAccounting`.
3. **Refactor `BacktestEngine.__init__`:**
    * Instantiate *one* portfolio: `self.portfolio = Portfolio(initial_capital=...)`.
    * **Inject** this single instance into the `Broker` and `Strategy`:
        * `self.broker.portfolio = self.portfolio`
        * `self.strategy.portfolio = self.portfolio`
4. **Refactor `SimulationBroker`:**
    * Remove `self.position_tracker`.
    * The `Broker`'s job is *not* to update state. Its job is to *check* state and *publish* fills.
    * Modify `on_market_event` to *read* from `self.portfolio` (e.g., `current_cash = self.portfolio.cash`,
      `current_position = self.portfolio.get_position(asset_id)`).
    * When `FillSimulator` generates a `FillEvent`, the `Broker`'s *only* action is to **publish it** (which it doesn't
      do yet—it needs to call `self.clock.publish(fill_event)`).
5. **Refactor `Portfolio`:**
    * The `Portfolio` must subscribe to `FillEvent`s.
    * Create an `on_fill_event(self, event: FillEvent)` method inside `Portfolio` that updates its *own* cash and
      positions.

### Phase 3: Clean Up Architectural Smells

1. **Refactor `AssetSpec`:**
    * Move `get_margin_requirement()` logic into `ml4t.backtest.portfolio.Margin`.
    * Move `calculate_pnl()` logic into `ml4t.backtest.portfolio.Portfolio`.
    * `AssetSpec` should only hold data: `contract_size`, `tick_size`, `initial_margin`, `taker_fee`, etc.
2. **Event-Drive Corporate Actions:**
    * Create a `CorporateActionFeed(DataFeed)` that reads from a file and generates `CorporateActionEvent`s.
    * The `BacktestEngine` will add this feed to the `Clock`.
    * The `Portfolio` will subscribe to `CorporateActionEvent` and adjust its state (e.g., update quantity for a
      `SplitEvent`, add cash for a `DividendEvent`) when it receives the event at the correct `ex_date` timestamp.
    * This removes the manual `if current_date != self._last_processed_date:` hack from `engine.py`.

### Phase 4: Feature Alignment (Targeting the Pro-Am Trader)

1. **Implement "Config-First" (from OPUS/GEM):** Create a top-level `run.py` script that takes a single YAML file. This
   file will define which `DataFeed`, `Strategy`, `SlippageModel`, and `CommissionModel` to load, making experiments
   reproducible and easy to launch.
2. **Build out Crypto & Futures:** With the core architecture fixed, you can now safely build out v1 features.
    * **Futures:** Implement daily Mark-to-Market (MTM) logic in the `Portfolio`. This should be triggered by a "
      MarketClose" event.
    * **Crypto Perps:** Create a simple `FundingRateFeed` that generates periodic `FundingEvent`s. The `Portfolio`
      subscribes to these and debits/credits cash based on the funding rate and position size, which it reads from its
      *own* state.