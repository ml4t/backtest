This is a comprehensive and high-quality code bundle. The architecture is clean, the separation of concerns (Broker vs.
Accounting vs. Execution) is excellent, and the use of Polars for data handling is modern.

However, there are **critical correctness issues in the Margin Account logic** and some optimistic assumptions in order
execution that need to be addressed before v1.0.

Here is the detailed review.

-----

# Code Review: ml4t.backtest v0.2.0

## 1\. Critical Correctness Issues (High Priority)

### 🔴 Account Policy: Margin Buying Power Formula

**File:** `accounting/policy.py`
**Severity:** **High**

Your current formula for buying power allows users to leverage up to the **Maintenance Margin** limit immediately,
rather than the **Initial Margin (Reg T)** limit. This allows users to open positions that would immediately place them
in a margin call or be rejected by a real broker.

* **Current Code:**

  ```python
  # Buying Power based on Maintenance Margin (MM)
  buying_power = (nlv - maintenance_margin_requirement) / self.initial_margin
  ```

* **The Issue:**
  Reg T requires you to have 50% equity (usually) at the time of the trade. Your formula calculates the surplus equity
  above the *25% maintenance line* and then leverages that.

    * *Example:* Long $100k stock, $50k cash (borrowed $50k).
    * NLV = $50k. IM (50%) = $50k. Excess Initial Equity = $0. **Buying Power should be $0.**
    * *Your Formula:* MM (25%) = $25k. Surplus = $50k - $25k = $25k. BP = $25k / 0.5 = **$50,000\*\*.
    * *Result:* The engine allows buying $50k more stock. New Portfolio: $150k Stock, $0 Equity (100k debt). You are now
      at 33% equity, below the 50% Reg T requirement.

* **Correction:**
  Buying power must be calculated based on **Excess Liquidity relative to Initial Margin**, not Maintenance Margin.

  ```python
  # Correct Reg T Buying Power
  total_market_value = sum(pos.market_value for pos in positions.values())
  required_initial_margin = total_market_value * self.initial_margin
  excess_equity = nlv - required_initial_margin
  buying_power = max(0.0, excess_equity / self.initial_margin)
  ```

### 🟡 Broker: Stop Order Gap Handling

**File:** `broker.py:_check_fill`
**Severity:** **Medium**

Your logic for filling stop orders is slightly optimistic on gap days.

* **Current Code:** `max(order.stop_price, low)` (for buys).
* **Scenario:** Stock closes at 100. Stop Buy at 105. Next Open is 110. Low is 108.
* **Result:** You fill at 108 (`max(105, 108)`).
* **Reality:** If the stock opens at 110, a Stop Market order triggers immediately. You would likely fill at 110 (Open)
  or higher. Filling at the Low (108) implies the trader perfectly timed the pullback during the day.
* **Recommendation:** If `Open > StopPrice` (for buys), the fill price should default to `Open` (or `Open + Slippage`).

### 🟡 Gatekeeper: Position Flip Validation

**File:** `accounting/gatekeeper.py`
**Severity:** **Medium**

You correctly identify reversals, but you might be double-counting the margin requirement for the closing portion of a
flip.

* **Logic:** When flipping Long 100 to Short 100 (Order -200), the `validate_position_change` logic checks if you have
  buying power for the *risk increase*.
* **The Issue:** The `risk_increase` variable in `policy.py` seems to calculate the risk on the *resulting* short
  position correctly. However, does `AccountState` allow the release of the Long margin *before* checking the Short
  margin?
* **Observation:** Since you don't update `AccountState` until *after* validation, `buying_power` is calculated based on
  the *current* Long position.
* **Result:** A user might have enough equity to hold the Short position, but not enough to hold the Long position
  *plus* the new Short margin requirement simultaneously.
* **Fix:** For reversals, you should simulate the "Close" first (add proceeds to theoretical cash, remove Long market
  value) to get the *Post-Close Buying Power*, then validate the New Short against that.

-----

## 2\. Feature Completeness vs. Competitors

### What's Missing (That Matters)

1. **Futures Margin Logic (Critical for "Institutional Grade")**

    * **Status:** You support `multiplier` in `Position`, which is great.
    * **Gap:** You apply `MarginAccountPolicy` (percentage-based) to everything. Futures use **SPAN** or **Fixed Dollar
      Margin per Contract** (e.g., ES=$12,000 per contract), not a percentage of notional value.
    * **Recommendation:** Add a `FuturesMarginPolicy` that looks up margin requirements from `ContractSpec`.

2. **Order Status: PARTIALLY\_FILLED**

    * **Status:** You keep partial orders as `PENDING`.
    * **Gap:** This is confusing for API users. Standard FIX/Exchange protocol distinguishes between `NEW` (Pending) and
      `PARTIALLY_FILLED`.
    * **Recommendation:** Add `OrderStatus.PARTIALLY_FILLED`.

3. **Dividend/Split Handling (Corporate Actions)**

    * **Status:** Missing.
    * **Impact:** This renders long-term equity backtests ( \> 1 year) inaccurate due to price adjustments.
    * **Recommendation:** Since you use Polars, you can implement a `CorporateActionAdjuster` that applies a ratio to
      positions/prices on ex-dates.

### Feature "Nice-to-Haves" (Not Critical)

* **Options:** Skipping these is the right choice for v0.2.0. They add massive complexity.
* **Optimizers:** Keep them external. Good design choice.

-----

## 3\. Edge Case Analysis

| Edge Case | Risk | Recommendation |
| :--- | :--- | :--- |
| **Zero Volume Bar** | **High** | If `volume=0`, `VolumeShareSlippage` returns 0 cost. It should probably block the trade entirely or assume infinite slippage (untradeable). |
| **Market Halt** | **Medium** | Similar to zero volume. If data is missing for a timestamp, the engine typically skips it, but open orders should not trigger on stale prices. Ensure `current_price` updates strictly match the current bar timestamp. |
| **Gap Day Stop Loss** | **Medium** | As noted above, ensure fills happen at `Open`, not `Stop Price`, on gaps. |
| **Penny Stocks** | **Low** | `PercentageSlippage` works poorly on $0.05 stocks. `FixedSlippage` or `VolumeShare` is better. Ensure your default isn't breaking low-priced assets. |

-----

## 4\. API Design Review

### 1\. Order Return Type

**Current:** Returns `Order` object.
**Verdict:** **Keep it.**
Returning the object is Pythonic and allows `order.status` checks immediately. Returning just an ID forces the user to
write `broker.get_order(id)` constantly.

### 2\. Position Access

**Current:** `get_position("AAPL")` vs `broker.positions`.
**Verdict:** **Standardize on Property.**
`broker.positions["AAPL"]` is more Pythonic. `get_position` is Java-esque. Keep `get_position` as a convenience helper
that handles `None` safely (returns a flat position object instead of None), but encourage dictionary access.

### 3\. ExecutionResult.success

**Verdict:** **"Success" = "Processed without Error".**
If an order is skipped due to risk limits, `success` should probably be `False` (or a specific `SKIPPED` state). If it
is partially filled, `success` is `True`.

### 4\. DataFeed Iteration

**Current:** Tuple `(ts, data, context)`.
**Verdict:** **Change to Dataclass.**
Tuples are brittle. If you add a fourth element later (e.g., `fundamental_data`), you break all unpacking loops. Use a
`BarData` dataclass.

-----

## 5\. Next Steps for You

### Step 1: Fix the Margin Formula

Apply this patch to `accounting/policy.py` immediately to fix the leverage calculation bug.

```python
# accounting/policy.py

def calculate_buying_power(self, cash: float, positions: dict[str, Position]) -> float:
    nlv = cash + sum(pos.market_value for pos in positions.values())

    # Calculate Reg T Initial Margin Requirement
    # Note: Ideally this comes from a ContractSpec, but using flat rate for now
    total_market_value = sum(abs(pos.market_value) for pos in positions.values())
    required_im = total_market_value * self.initial_margin

    # Buying Power is based on Excess Equity relative to Initial Margin
    excess_equity = nlv - required_im

    # If we are already in a deficit, 0 buying power
    if excess_equity < 0:
        return 0.0

    # Standard Reg T: Buying Power = Excess Equity / Initial Margin
    return excess_equity / self.initial_margin
```

### Step 2: Implement "Snapshot" Logic for Reversals

In `Gatekeeper`, when detecting a reversal:

1. Clone the `AccountState`.
2. Apply the "Close" portion of the trade to the clone.
3. Validate the "Open" portion against the clone's Buying Power.

### Step 3: Add Unit Tests for Leverage

Create a test case:

1. Cash: $10,000.
2. Margin: 50% IM.
3. Buy $20,000 worth of stock. (Should Succeed).
4. Try to buy $1 more. (Should Fail).
   *Your current code likely allows buying up to $40,000 (4x leverage) because of the math error.*

Would you like me to write the specific **Unit Test** for the Margin Logic verification to confirm the bug?