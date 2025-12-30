Here is the code review and enhancement architecture proposal.

-----

# Part 1: Codebase Review

## 1\. Repository Map and Mental Model

**Mental Model:**
This is a classic event-driven loop architecture similar to Zipline, but refactored to use modern Python typing and data
structures (Polars).

* **Data Ingestion (`DataFeed`):** Ingests OHLCV/Signal data. Currently acts as a synchronous iterator yielding
  `(timestamp, asset_data, context)` tuples.
* **Orchestration (`Engine`):** The main loop. It fetches the next bar from `DataFeed`, updates the `Broker` time/state,
  triggers the `Strategy`, and processes orders.
* **State Management (`Broker` & `Accounting`):**
    * The `Broker` holds the "street-side" state (orders, fills).
    * The `AccountState` holds the "books" (cash, positions, equity).
    * The `Gatekeeper` acts as a pre-trade risk check (buying power, short logic).
* **Execution (`Broker._process_orders`):** Orders are processed in an "Exit-First" sequence to free up capital before
  attempting entries.

**Extension Points:**

* **Present:** Commission models, Slippage models, Account policies (Cash/Margin), Strategy interface.
* **Missing:** Risk Management (stops/limits are hardcoded in Broker/Order objects rather than a managed logic layer),
  Market Impact (slippage is price-only, no volume limiting), Corporate Actions handling.

## 2\. Correctness and Robustness

### Critical Correctness Issues

1. **Limit/Stop Order Logic relies on Open/Close only:**
   In `src/ml4t/backtest/broker.py`, the `_check_fill` method checks limits/stops against `price` (which is either Open
   or Close).

    * *Bug:* It ignores `High` and `Low`. A Limit Buy at $100 might be filled if the Low was $99, even if Open
      was $105 and Close was $102. The current engine will miss this fill.
    * *Fix:* The `_check_fill` method must accept the full OHLC bar to validate if the price traded through the
      limit/stop level during the period.

2. **Look-ahead Bias in `DataFeed` timestamps:**
   The `DataFeed` sorts unique timestamps from prices, signals, and context. If `signals` contain forward-looking
   timestamps (e.g., prediction for T+5), the engine might advance time incorrectly or try to trade on future dates.

3. **Partial Fills & Volume Limits:**
   The current `VolumeShareSlippage` model calculates a *price impact*, but it does not restrict the *quantity*. If a
   strategy orders 1M shares on a bar with 100k volume, it fills 1M shares with a price penalty. In reality, you simply
   cannot trade more than the volume.

### Robustness & Invariants

* **Invariant:** `AccountState.cash` + `MarketValue(Positions)` must equal `AccountState.total_equity`.
* **Edge Case:** Delisted assets. If an asset disappears from the feed, the `Broker` retains the position indefinitely
  without updating the price. The `Broker` needs a mechanism to handle `KeyError` or stale prices during
  `mark_to_market`.

## 3\. Code Quality and Maintainability

* **Separation of Concerns:** The separation between `Broker` (execution) and `AccountState`/`Gatekeeper` (accounting)
  is excellent. This makes adding "Portfolio Margin" or "Crypto Futures" support much easier.
* **Type Hinting:** Strong use of Python type hints makes the code readable and safer.
* **Method Length:** `Broker._process_orders` is becoming a "God Method." It handles retrieving prices, splitting order
  sides, validating via gatekeeper, executing fills, and cleaning lists. This should be refactored.

**Refactoring Suggestion:**
Move the order processing logic into an `OrderProcessor` class or split `_process_orders` into `_process_exits()`,
`_process_entries()`, and `_validate_and_fill(order)`.

## 4\. Performance and Scalability

**Major Performance Bottleneck Identified:**
The `DataFeed` implementation is the primary reason the engine takes \~35s (vs VectorBT's 1s).

1. **Eager Loading:** `pl.scan_parquet(...).collect()` in `__init__` loads the entire history into RAM. This defeats the
   purpose of Polars "lazy" loading.
2. **The O(N\*M) Filter:**
   In `src/ml4t/backtest/datafeed.py`:
   ```python
   prices_at_ts = self.prices.filter(pl.col("timestamp") == ts)
   ```
   This line runs inside the event loop. For every single day (N), it scans the entire 10-year dataframe (M) to filter
   for that specific timestamp. This is catastrophic for performance.

**Optimizations (Ranked):**

1. **Pre-partition / GroupBy Iterator (High Impact):** Instead of filtering inside the loop, convert the Polars
   DataFrame into a dictionary of DataFrames keyed by timestamp, or use `df.partition_by("timestamp")` *once* at
   startup. Even better, iterate over the sorted DataFrame once using pointers.
2. **Vectorized State Checks:** The engine loops through pending orders in Python. For 500 stocks, this is fine, but for
   5,000 it will slow down. Keep orders in a specialized structure (e.g., `SortedDict` by asset) to avoid iterating
   non-relevant orders.

## 5\. Test Coverage and Validation

**Missing Critical Tests:**

1. **Intraday High/Low Penetration:** Test a Limit Buy below Open/Close but above Low. Current code will fail this.
2. **Margin Call Logic:** Verify behavior when equity drops below Maintenance Margin. Does it liquidate? (Currently, it
   seems it just prevents *new* orders, but doesn't force close existing ones).
3. **Split/Dividend Adjustments:** Ensure a 2:1 split doesn't halve the portfolio value instantly.

-----

# Part 2: Enhancement Architecture Proposals

## A. Flexible, Dynamic Risk Management Framework

The current system relies on strategies managing their own exits or hardcoded order types (Stop/Limit). We need a
dedicated `RiskManager` layer that intercepts signals/orders.

### 1\. Component Model

```python
class RiskManager(ABC):
    """Orchestrator that validates state and modifies orders."""

    def on_market_event(self, timestamp, data, context, portfolio_state) -> list[Order]:
        # Check portfolio level stops (e.g., max drawdown)
        pass

    def on_signal(self, signals, portfolio_state) -> dict[str, float]:
        # Adjust target weights based on volatility/risk
        pass

    def review_orders(self, orders: list[Order], portfolio_state) -> list[Order]:
        # Final veto power before Gatekeeper
        pass
```

### 2\. Risk Rule Interface

We will implement a "Chain of Responsibility" pattern where rules can veto or modify orders.

```python
@dataclass
class RiskState:
    # Persistent state for dynamic rules
    peak_equity: float = 0.0
    entry_prices: dict[str, float] = field(default_factory=dict)
    highest_prices: dict[str, float] = field(default_factory=dict)  # For trailing stops


class RiskRule(Protocol):
    def process(self, state: RiskState, orders: list[Order], broker: Broker) -> list[Order]: ...
```

### 3\. Integration Integration Plan

**Target:** `src/ml4t/backtest/risk.py` (New module) and `Broker` class.

1. **Hook into Broker:** Add `self.risk_manager` to Broker.
2. **Event Hooks:**
    * `on_bar_open`: Risk manager checks for global liquidations (e.g., Portfolio Drawdown \> 10%).
    * `pre_submission`: When strategy calls `submit_order`, pass it through `risk_manager.review_orders`.

### 4\. Example: Dynamic Trailing Stop Logic

```python
class DynamicTrailingStop(RiskRule):
    def __init__(self, activation_profit_pct: float, trail_pct: float):
        self.activation_pct = activation_profit_pct
        self.trail_pct = trail_pct

    def process(self, state: RiskState, orders: list[Order], broker: Broker) -> list[Order]:
        generated_orders = []

        for asset, position in broker.positions.items():
            # Update highest price tracking
            current_price = broker._current_prices[asset]
            state.highest_prices[asset] = max(state.highest_prices.get(asset, 0), current_price)

            # Calculate unrealized PnL %
            pnl_pct = (current_price - position.avg_entry_price) / position.avg_entry_price

            # Check activation
            if pnl_pct > self.activation_pct:
                # Logic: Stop price is X% below highest price seen since entry
                stop_price = state.highest_prices[asset] * (1.0 - self.trail_pct)

                if current_price < stop_price:
                    # Generate forced liquidation
                    generated_orders.append(
                        Order(asset, OrderSide.SELL, abs(position.quantity), OrderType.MARKET)
                    )

        return orders + generated_orders
```

-----

## B. Slippage and Market Impact Models

We need to move from simple price adjustment to a model that affects **Price** and **Quantity**.

### 1\. ExecutionCostModel Interface

```python
@dataclass
class ExecutionResult:
    filled_quantity: float
    fill_price: float
    commission: float
    status: OrderStatus  # FILLED or PARTIAL


class ExecutionModel(ABC):
    @abstractmethod
    def simulate_fill(
            self,
            asset: str,
            order: Order,
            bar_data: dict,  # Open, High, Low, Close, Volume
            current_spread: float = 0.0
    ) -> ExecutionResult:
        pass
```

### 2\. Implementation: Volume Participation Model

This model limits trading to a percentage of the bar's volume and penalizes price based on square-root law.

```python
class VolumeParticipationModel(ExecutionModel):
    def __init__(self, max_participation=0.10, impact_constant=0.1):
        self.max_vol_pct = max_participation
        self.k = impact_constant

    def simulate_fill(self, asset, order, bar_data, spread=0.0):
        market_volume = bar_data.get('volume', float('inf'))

        # 1. Limit Quantity
        max_tradable = market_volume * self.max_vol_pct
        fill_qty = min(order.quantity, max_tradable)

        # 2. Calculate Price Impact
        # Basic sqrt impact: Cost ~ k * sigma * sqrt(size / volume)
        # Simplified: Price deviates by k * (participation_rate)^0.5
        participation_rate = fill_qty / market_volume if market_volume > 0 else 0
        impact_pct = self.k * (participation_rate ** 0.5)

        base_price = bar_data['open']  # Or arrival price

        if order.side == OrderSide.BUY:
            fill_price = base_price * (1 + impact_pct + spread / 2)
        else:
            fill_price = base_price * (1 - impact_pct - spread / 2)

        status = OrderStatus.FILLED if fill_qty == order.quantity else OrderStatus.FILLED_PARTIAL

        return ExecutionResult(fill_qty, fill_price, 0.0, status)
```

### 3\. Integration

In `broker.py`, modify `_execute_fill`. Instead of calculating slippage internally, delegate to
`self.execution_model.simulate_fill(...)`.

-----

## C. Additional Feature Gaps

1. **Corporate Actions Pipeline (High Priority):**

    * *Why:* Stock splits destroy backtests (price drops 50%, engine thinks you lost 50%).
    * *Impl:* `DataFeed` needs an `adjustments` dataframe. `Broker` checks this map daily. If split, multiply pos
      quantity, divide pos cost basis.

2. **Short Availability (Hard-to-Borrow):**

    * *Why:* Institutional strategies often can't short volatile names.
    * *Impl:* `Gatekeeper` checks a `ShortAvailability` source before validating sell orders.

3. **Transaction Cost Analysis (TCA):**

    * *Why:* To debug why a strategy loses money (alpha vs. execution).
    * *Impl:* Record `arrival_price` (price when order generated) vs `fill_price` in the `Trade` object. Calculate
      implementation shortfall.

-----

# Priority and Roadmap

**Phase 1: Core Stability & Performance (Weeks 1-2)**

1. **Fix DataFeed Performance:** Replace `iter_rows` loop with a pre-partitioned structure. This is the single highest
   ROI change.
2. **Fix Limit/Stop Logic:** Update `_check_fill` to use High/Low data.
3. **Add Corporate Actions:** Basic split handling is mandatory for 10-year backtests.

**Phase 2: Architecture Extensions (Weeks 3-5)**

1. **Refactor Execution Model:** Implement the `ExecutionModel` interface allowing partial fills.
2. **Implement Risk Manager:** Create the hook points in `Broker` and the base `RiskRule` classes.

**Phase 3: Advanced Features (Month 2+)**

1. **Dynamic Risk Rules:** Implement the trailing stops and portfolio heat checks.
2. **TCA Reporting:** Enhanced trade logs.

### Next Action for User:

I recommend starting with the **DataFeed optimization**.
Would you like me to write the optimized `DataFeed` class using a generator or a partitioned dictionary approach to fix
the 35s runtime immediately?
