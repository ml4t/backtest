This is a comprehensive architectural review based on the provided documentation (README\_REVIEW.md,
DESIGN\_QUESTIONS.md) and the code context.

### **Executive Summary**

Verdict: Build (Refactor).
Do not abandon your engine. A 32x speed advantage over Backtrader is a massive competitive advantage for optimization
and walk-forward analysis. Adopting Backtrader returns you to a performance bottleneck, and VectorBT, while fast, lacks
the granular event-driven flexibility needed for institutional complexity (e.g., complex order types, conditional
rebalancing).
However, your current "unlimited debt" model is fatal. You must decouple the **Execution Engine** from the **Accounting
Logic**. The accounting must be a strict gatekeeper.

---

## **Part 1: Accounting Model Review**

**Status:** The proposed model is insufficient for leverage handling.

### **1.1 The Correct Identity**

Your proposed identity Account\_Value \= Cash \+ Σ(position\_qty × current\_price) is mathematically correct for **Net
Liquidating Value (NLV)**, but it fails to model **Buying Power**, which is what actually constrains trades.

![Image of margin account structure][image1]

Shutterstock

Recommended Model:
You need to track three distinct states:

1. Total Equity (NLV):

   $$NLV \= Cash \+ \\sum\_{i} (Quantity\_i \\times Price\_i)$$

   (Note: Short quantities are negative, so this naturally subtracts short obligations.)
2. Maintenance Margin (MM):

   $$MM \= \\sum\_{i} (|Quantity\_i \\times Price\_i| \\times MarginRate\_i)$$

   (Note: Margin is based on absolute value—gross exposure.)
3. Buying Power (BP):

   $$BP \= \\frac{NLV \- MM}{InitialMarginRate}$$

### **1.2 Handling Shorts & Proceeds**

Question: Is short sale cash available for longs?
Answer: Yes, but it doesn't increase Equity.
When you short $10k of stock:

1. **Cash** increases by $10k.
2. **Liability** (Short Market Value) increases by $10k.
3. **Equity** remains unchanged (ignoring fees).
4. **Buying Power** decreases because the short position consumes margin.

Fix for your Engine:
Do not simply check if total\_cost \> cash. You must check:
if new\_trade\_margin\_impact \> current\_available\_margin.

---

## **Part 2: Execution Sequencing Review**

**Status:** Your "Next-Bar" logic is correct for backtesting, but the specific sequence needs refinement to maximize
capital efficiency.

### **2.1 The Golden Sequence**

To resolve the "Exit before Entry" dilemma, the standard institutional backtesting sequence for Daily bars is:

1. **Market Open (Day N+1):**
2. **Process Exits:** Execute all CLOSE, SELL (of long), and COVER (of short) orders.
    * *Why:* This releases margin/cash immediately.
3. **Mark-to-Market:** Update Account Equity and Buying Power based on open prices and realized P\&L from step 2\.
4. **Process Entries:** Execute BUY and SHORT orders using the *newly freed* Buying Power.

### **2.2 Position Reversals (The "Flip")**

Question: How to handle Long 100 $\\to$ Short 100?
Recommendation: Treat this as an atomic transaction sequence within the same timestamp:

1. Generate logical order: Sell 200\.
2. **Engine internals:**
    * Split into: Sell 100 (Close) \+ Sell 100 (Open Short).
    * Execute Close $\\to$ Update Margin $\\to$ Execute Open Short.
      If you try to execute "Sell 200" as a single block against a "Long 100" position without splitting, simple
      accounting engines often crash or miscalculate margin.

---

## **Part 3: Liquidity & Market Impact**

**Status:** Critical missing piece. Speed is irrelevant if you execute $10M volume on a $50k volume bar.

### **3.1 Zero-Volume Bars**

**Rule:** If Volume \== 0 (common in overnight futures), **execution is forbidden.**

* **Action:** The order remains PENDING. It does not fill. It carries over to the next bar.
* **Why:** You cannot trade if no one else traded. Filling at the "Close" price of a zero-volume bar is a classic "
  look-ahead" variation that yields fake alpha.

### **3.2 Partial Fills vs. Rejection**

Question: Scale down or reject?
Recommendation: Partial Fill (Scale Down).
If a strategy signals a buy of 10,000 shares but available volume is 5,000 (or cash supports only 5,000):

1. Fill 5,000.
2. Cancel the remainder (or leave pending, depending on order type TIF).
   Rejecting the whole order creates discontinuous cliff-edges in optimization landscapes.

---

## **Part 4: Architecture & Code Fixes**

### **4.1 The "Broker" is the Guardrail**

Your Broker class currently acts as a passive recorder. It must become an active gatekeeper.

**Refactored \_execute\_fill Logic:**

Python

def \_execute\_fill(self, order, fill\_price):
\# 1\. Calculate Costs
commission \= self.commission\_model.calculate(order, fill\_price)
gross\_value \= order.quantity \* fill\_price

    \# 2\. Determine Margin Impact (Simplified)
    \# Assume 1.0 for cash, 0.5 for RegT, etc.
    margin\_required \= abs(gross\_value) \* self.leverage\_ratio

    \# 3\. CHECK CONSTRAINTS (The missing piece)
    \# If closing a position, margin releases, so always allow.
    \# If opening/increasing, check capability.
    is\_reducing\_exposure \= self.\_is\_reducing\_position(order)

    if not is\_reducing\_exposure:
        if margin\_required \> self.get\_available\_buying\_power():
            \# Option A: Reject
            return RejectedFill("Insufficient Buying Power")
            \# Option B: Scale (Better)
            \# order.quantity \= ...

    \# 4\. Update State
    self.cash \-= (gross\_value \+ commission) \# Note: gross\_value is negative for Sells
    self.\_update\_position(order)

### **4.2 Implementation Roadmap (Build Plan)**

Since you have a working event loop that is 32x faster, **stick with it**. Here is the minimum viable scope to fix the
accounting:

1. **Phase 1: The Ledger (1 Week)**
    * Implement AccountState class (tracks Cash, Equity, Margin).
    * Implement Position class (tracks Quantity, Avg Price, Cost Basis).
2. **Phase 2: The Gatekeeper (1 Week)**
    * Update Broker.submit\_order to check Buying Power.
    * Implement "Exit-First" sorting in the order processing loop.
3. **Phase 3: Validation (Continuous)**
    * Re-run your VectorBT comparison. The goal is to match VectorBT's PnL exactly (or within rounding error) when cash
      constraints are hit.

---

## **Part 5: Testing Strategy**

**Question:** *How to validate without ground truth?*

1. **The "Bankruptcy Test":** Create a strategy that doubles down on losing positions exponentially (Martingale).
    * *Pass:* The engine stops trading when equity hits 0\.
    * *Current Fail:* Your engine likely trades until equity is negative millions.
2. **The "Flipping Test":** Long 1 share $\\leftrightarrow$ Short 1 share every bar.
    * Verify cash balance decreases exactly by (Commission \+ Spread) per bar.
3. **Spreadsheet Validation:** Do not trust other libraries blindly. Build a manual Excel sheet for **5 trades**.
   Calculate the exact cash/margin flow. Assert your engine matches the Excel sheet to the penny.

---

## **Summary of Recommendations**

| Area             | Recommendation                                                              | Priority |
|:-----------------|:----------------------------------------------------------------------------|:---------|
| **Architecture** | **Keep Custom Build.** 32x speed is worth the accounting refactor.          | High     |
| **Accounting**   | Switch from Cash-Check to **Margin/Buying Power** Check.                    | Critical |
| **Sequencing**   | **Exits First.** Process all closing trades before opening trades.          | High     |
| **Logic**        | **Atomic Reversals.** Split "Flip" orders into Close \+ Open.               | Medium   |
| **Liquidity**    | **Volume Limits.** Cap order size at 10% of bar volume. Reject on 0 volume. | Medium   |

### **Next Step**

Would you like me to write the skeleton code for the **AccountState** and **Gatekeeper** classes to replace your current
Broker implementation? This will directly solve the "unlimited debt" issue.

This is a foundational refactor. We are moving from a simple "Checkbook" model (Cash only) to a "Margin Account" model (
Equity \+ Buying Power).

I have generated three files/blocks for you:

1. **models.py**: The data structures.
2. **accounting.py**: The math engine (AccountState & MarginModel).
3. **gatekeeper.py**: The enforcer logic.

### **1\. Data Structures (models.py)**

We need a robust Position class that tracks cost basis separately from current market value.

src/ml4t/backtest/models.py
Nov 20, 9:36 AM
Open

### **2\. The Ledger (accounting.py)**

This is the heart of the fix. AccountState calculates **Equity**, not just Cash.

**Key Concept:**

* **Cash:** Can go negative (margin debt) or be artificially high (short proceeds). It is not the limit.
* **Equity (NLV):** The true liquidation value. Cash \+ Market Value of Positions.
* **Buying Power:** The derived value that limits new trades.

src/ml4t/backtest/accounting.pyNov 20, 9:36 AM
Open

### **3\. The Enforcer (gatekeeper.py)**

This replaces the logic inside your Broker.submit\_order. It decides if a trade is legal.

src/ml4t/backtest/gatekeeper.py
Nov 20, 9:36 AM
Open

### **How to integrate this into your Engine**

1. **Initialize:**
   Python
   margin\_model \= MarginModel(initial\_margin=0.5) \# 2x leverage
   account \= AccountState(initial\_cash=100\_000, margin\_model=margin\_model)
   gatekeeper \= Gatekeeper(account)

2. **In your Loop (Processing Orders):**
   Python
   \# 1\. Update prices first\!
   account.mark\_to\_market(current\_bar\_prices)

   \# 2\. Process Exits (Release margin)
   for order in exit\_orders:
   account.apply\_fill(...)

   \# 3\. Process Entries (Check Gatekeeper)
   for order in entry\_orders:
   is\_valid, reason \= gatekeeper.validate\_order(order)
   if is\_valid:
   \# Execute
   account.apply\_fill(asset, qty, price, comm)
   else:
   \# Log rejection
   print(f"Order Rejected: {reason}")

This structure specifically solves your "Unlimited Debt" problem by calculating buying\_power based on total\_equity
rather than just cash.
