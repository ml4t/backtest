This is a high-quality, well-structured proposal. You have correctly identified a significant gap in the `ml4t.backtest`
library: the bridge between **signal generation** (alpha) and **execution mechanics** (rebalancing).

Here is my review of your proposal, validating your recommendation and offering specific technical refinements.

### Executive Summary of Review

I **strongly validate the recommendation of Option B (TargetWeightExecutor)**.

* **Separation of Concerns:** It decouples the *execution* of a portfolio (boring, error-prone math) from the
  *construction* of a portfolio (creative, complex math).
* **Library Bloat:** Option C introduces a heavy opinion on how returns are calculated (timeframes, data alignment),
  which often conflicts with advanced user needs (e.g., users trading on tick bars or volume bars).
* **Flexibility:** Option B allows users to use `riskfolio`, `cvxpy`, or simple heuristics like `1/N` without fighting
  the framework.

However, there is one **critical architectural blind spot** in the current draft regarding **Pending Orders**, which I
detail below.

-----

### Critical Feedback: The "Pending Order" Trap

Your current implementation of `_get_current_weights` looks like this:

```python
# Current Draft
weights = {}
for asset, pos in broker.positions.items():
# ... calculates value based on HELD positions ...
```

**The Risk:**
In an event-driven system, a Strategy might run multiple times before an order is filled (e.g., if using
`ExecutionMode.NEXT_BAR` or if the order is a LIMIT order sitting in the book).

If the user calls `executor.execute()` twice in succession before the first batch of orders fills:

1. **T=1:** Executor sees 0% AAPL. Target 10%. Submits BUY order.
2. **T=2:** Order is still pending. Executor sees 0% AAPL. Target 10%. Submits **another** BUY order.
3. **Result:** The user ends up with 20% AAPL (double allocation).

**The Fix:**
The `TargetWeightExecutor` must account for the `delta` of pending orders when calculating current weights.

```python
def _get_effective_weights(self, broker: Broker, data: dict) -> dict[str, float]:
    equity = broker.get_account_value()
    # 1. Start with actual positions
    effective_value = {
        a: p.quantity * data.get(a, {}).get('close', p.entry_price)
        for a, p in broker.positions.items()
    }

    # 2. Add net value of pending orders
    for order in broker.pending_orders:
        # Estimate value of the pending order
        price = order.limit_price or data.get(order.asset, {}).get('close')
        if price:
            val = order.quantity * price * (1 if order.side == OrderSide.BUY else -1)
            effective_value[order.asset] = effective_value.get(order.asset, 0) + val

    # 3. Convert to weights
    return {k: v / equity for k, v in effective_value.items()}
```

-----

### Answers to Your Specific Questions

#### 1\. Is TargetWeightExecutor the right abstraction? Class vs. Function?

**Keep it as a Class.**
You need state configuration (`RebalanceConfig`). A function signature would become unwieldy with 8 different boolean
flags. Furthermore, if you later decide to add "Drift Tracking" (statefulness to track how far the portfolio has drifted
since the last rebalance), a class is required.

#### 2\. Should we handle partial fills?

**No.** Keep the executor stateless regarding fill history.
If the strategy is event-driven, it should simply re-evaluate on the next bar.

* **Scenario:** Target 100 shares. Fill 60.
* **Next Bar:** Executor calculates Current = 60, Target = 100. Delta = 40. Submits order for 40.
  The loop naturally corrects partial fills without complex state tracking.

#### 3\. Transaction cost awareness?

**Yes, partially.**
Instead of complex cost modeling, add a `ignore_if_cost_exceeds_bps` threshold to `RebalanceConfig`.
If the estimated trade size is $1,000 and the commission is $10 (1%), and the alpha edge is only 0.5%, the trade
destroys value.

* *Suggestion:* Add `min_trade_value` (which you have) is usually sufficient for 90% of users. Don't over-engineer this
  yet.

#### 4\. Constraints integration (Sector limits, etc.)?

**Strictly No.**
This belongs in the **Optimizer** (Strategy layer), not the **Executor** (Execution layer). If the Executor starts
deciding "I can't buy this because it's Tech sector," it is silently overriding the Optimizer's math. The Executor's job
is to blindly obey the target weights provided.

#### 5\. What about live trading?

The design translates well, **provided you solve the Pending Order issue** mentioned above. In live trading, latency
means pending orders exist for longer durations.

-----

### Refined Code Snippet (Option B+)

Here is the robust version of the `execute` method I recommend, incorporating the "Pending Order" fix and a "
Cancel/Replace" logic.

```python
    def execute(self, target_weights: dict[str, float], data: dict, broker: Broker) -> list[Order]:
    equity = broker.get_account_value()

    # 1. Cancel existing open orders to prevent race conditions
    # (Optional: dependent on aggressive vs. passive style. 
    # Safe default is to cancel open orders before rebalancing)
    for order in broker.pending_orders:
        broker.cancel_order(order.order_id)

    # 2. Calculate effective positions (current + nothing pending now)
    current_weights = self._get_current_weights(broker, data)
    orders = []

    for asset, target_wt in target_weights.items():
        # ... (existing logic) ...

        # 3. Logic check for Cash
        # Ensure we don't spend more than 100% of equity unless leverage is allowed
        # This prevents floating point errors from triggering Margin Calls

        # 4. Logic for fractional vs whole shares
        shares = delta_value / price
        if not self.config.allow_fractional:
            shares = int(shares)

        if shares == 0:
            continue

        # ... submit order ...

    return orders
```

### Summary of Recommendations

1. **Proceed with Option B.**
2. **Update `_get_current_weights`** (or the execution logic) to account for or cancel existing pending orders.
3. **Add `allow_fractional`** to `RebalanceConfig` (defaults to `False`). Many backtests fail because they try to buy
   10.43 shares of SPY.
4. **Add `targeted_cash` handling.** If weights sum to 0.95, the executor implicitly leaves 5% cash. Ensure the math
   supports `sum(weights) < 1.0`.

### Next Step

Would you like me to refactor the **`TargetWeightExecutor`** code to include the "Pending Order Awareness" logic and the
updated `RebalanceConfig` with fractional support?