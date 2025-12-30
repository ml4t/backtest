# Margin Account Calculations

This document provides detailed explanations of margin calculations used in ml4t.backtest, including formulas, examples, and regulatory context.

---

## Overview

Margin accounts allow traders to:
1. **Use leverage** - Trade with borrowed funds (up to 2x buying power)
2. **Short sell** - Borrow and sell assets they don't own

This flexibility comes with constraints enforced through margin requirements.

---

## Key Formulas

### 1. Net Liquidation Value (NLV)

**Formula**:
```
NLV = Cash + Σ(Position Market Values)
```

**Definition**: Total account value if all positions were liquidated at current market prices.

**Components**:
- `Cash`: Available cash in account
- `Position Market Value`: `quantity × current_price` for each position

**Example**:
```
Cash:         $50,000
Long AAPL:    +100 shares @ $150 = $15,000
Short TSLA:   -50 shares @ $200  = -$10,000

NLV = $50,000 + $15,000 + (-$10,000) = $55,000
```

**Notes**:
- Long positions add to NLV
- Short positions subtract from NLV (liability)
- NLV represents actual account equity

---

### 2. Maintenance Margin (MM)

**Formula**:
```
MM = Σ(|Position Market Value| × maintenance_margin_rate)
```

**Definition**: Minimum equity required to maintain current positions without triggering margin call.

**Standard Rates** (Regulation T):
- Long positions: 25% (0.25)
- Short positions: 30% (0.30)

**ml4t.backtest Default**: 25% (0.25) for longs, 30% (0.30) for shorts

**Example**:
```
Long AAPL:    |+100 × $150| = $15,000 → MM = $15,000 × 0.25 = $3,750
Short TSLA:   |-50 × $200|  = $10,000 → MM = $10,000 × 0.30 = $3,000

Total MM = $3,750 + $3,000 = $6,750
```

**Notes**:
- Uses absolute value (both long and short contribute)
- Higher MM = less available buying power
- Account must maintain NLV ≥ MM to avoid margin call

---

### 3. Initial Margin (IM)

**Formula**:
```
IM = initial_margin_rate × |Order Value|
```

**Definition**: Equity required to open a new position.

**Standard Rate** (Regulation T): 50% (0.50) = 2x leverage

**Example**:
```
Want to buy $20,000 worth of stock:
Required equity = $20,000 × 0.50 = $10,000

With $10,000 equity, can control $20,000 position (2x leverage)
```

**Notes**:
- IM only applies to opening new positions
- IM ≥ MM (initial requirement is stricter)
- Closing positions doesn't require additional IM

---

### 4. Buying Power (BP)

**Formula**:
```
BP = (NLV - MM) / initial_margin
```

**Definition**: Maximum value of new positions that can be opened.

**Intuition**:
- `(NLV - MM)` = Excess equity above maintenance requirement
- Divide by IM rate to get total position value allowed

**Example**:
```
NLV = $55,000
MM = $6,250
IM = 0.50 (50%)

BP = ($55,000 - $6,250) / 0.50 = $97,500
```

**Notes**:
- Buying power includes leverage (can exceed cash)
- Accounts with no positions: BP = Cash / IM
- Cash account: BP = Cash (no leverage, IM = 1.0)

---

## Step-by-Step Calculation Examples

### Example 1: Initial Account State

**Starting Conditions**:
```
Initial Cash:     $100,000
Positions:        None
IM:               50% (0.50)
MM:               25% (0.25)
```

**Calculations**:
```
NLV = $100,000 + $0 = $100,000
MM = $0 (no positions)
BP = ($100,000 - $0) / 0.50 = $200,000
```

**Interpretation**:
- Can open positions worth up to $200,000 (2x leverage)
- If buying $200/share stock: max 1,000 shares

---

### Example 2: After Opening Long Position

**Action**: Buy 400 shares of AAPL @ $150

**New Account State**:
```
Cash:             $100,000 - (400 × $150) = $40,000
Position:         +400 AAPL @ $150 = $60,000
```

**Calculations**:
```
NLV = $40,000 + $60,000 = $100,000
MM = $60,000 × 0.25 = $15,000
BP = ($100,000 - $15,000) / 0.50 = $170,000
```

**Interpretation**:
- NLV unchanged (used cash to buy assets)
- MM now $15,000 (must maintain this equity)
- BP now $170,000 (can open $170k more positions)

---

### Example 3: After Price Movement (Profit)

**Event**: AAPL rises to $175

**New Account State**:
```
Cash:             $40,000 (unchanged)
Position:         +400 AAPL @ $175 = $70,000 (was $60,000)
```

**Calculations**:
```
NLV = $40,000 + $70,000 = $110,000 (+$10,000)
MM = $70,000 × 0.25 = $17,500
BP = ($110,000 - $17,500) / 0.50 = $185,000
```

**Interpretation**:
- NLV increased by $10,000 (unrealized P&L)
- BP increased to $185,000 (more room to trade)
- No cash change until position sold

---

### Example 4: After Price Movement (Loss)

**Event**: AAPL drops to $130 (from original $150 entry)

**New Account State**:
```
Cash:             $40,000 (unchanged)
Position:         +400 AAPL @ $130 = $52,000 (was $60,000)
```

**Calculations**:
```
NLV = $40,000 + $52,000 = $92,000 (-$8,000)
MM = $52,000 × 0.25 = $13,000
BP = ($92,000 - $13,000) / 0.50 = $158,000
```

**Interpretation**:
- NLV decreased by $8,000 (unrealized loss)
- Still above MM ($92,000 > $13,000) - no margin call
- BP reduced to $158,000 (less room to trade)

---

### Example 5: Margin Call Scenario

**Event**: AAPL crashes to $50 (from $150 entry)

**New Account State**:
```
Cash:             $40,000
Position:         +400 AAPL @ $50 = $20,000 (was $60,000)
```

**Calculations**:
```
NLV = $40,000 + $20,000 = $60,000 (-$40,000 loss)
MM = $20,000 × 0.25 = $5,000
BP = ($60,000 - $5,000) / 0.50 = $110,000
```

**Interpretation**:
- Lost $40,000 (40% of initial capital)
- NLV ($60,000) > MM ($5,000) - NO margin call
- BP still $110,000 (can open more positions)

**Why no margin call?**
- Margin call triggers when NLV < MM
- Here: $60,000 > $5,000 (safe)
- Would need AAPL to drop to near $0 for margin call

---

### Example 6: Short Position

**Action**: Short 100 shares of TSLA @ $200

**New Account State**:
```
Cash:             $100,000 + (100 × $200) = $120,000 (proceeds)
Position:         -100 TSLA @ $200 = -$20,000 (liability)
```

**Calculations**:
```
NLV = $120,000 + (-$20,000) = $100,000
MM = |-$20,000| × 0.25 = $5,000
BP = ($100,000 - $5,000) / 0.50 = $190,000
```

**Interpretation**:
- Cash increased by short sale proceeds
- NLV unchanged (proceeds offset by liability)
- MM = $5,000 (minimum equity to hold short)
- BP = $190,000 (still have room to trade)

---

### Example 7: Short Position Loss

**Event**: TSLA rises to $250 (from $200 entry)

**New Account State**:
```
Cash:             $120,000 (unchanged)
Position:         -100 TSLA @ $250 = -$25,000 (liability increased)
```

**Calculations**:
```
NLV = $120,000 + (-$25,000) = $95,000 (-$5,000 loss)
MM = |-$25,000| × 0.25 = $6,250
BP = ($95,000 - $6,250) / 0.50 = $177,500
```

**Interpretation**:
- Lost $5,000 on short (stock went up)
- NLV decreased to $95,000
- MM increased to $6,250 (larger liability)
- BP reduced to $177,500

---

### Example 8: Position Reversal (Long → Short)

**Starting State**:
```
Cash:             $50,000
Position:         +100 AAPL @ $150 = $15,000
NLV:              $65,000
```

**Action**: Sell 200 shares AAPL @ $150 (reversal: close +100, open -100)

**Step 1: Close Long Position**:
```
Cash:             $50,000 + (100 × $150) = $65,000
Position:         0
NLV:              $65,000
```

**Step 2: Open Short Position**:
```
Cash:             $65,000 + (100 × $150) = $80,000
Position:         -100 AAPL @ $150 = -$15,000
NLV:              $80,000 + (-$15,000) = $65,000
```

**Final Calculations**:
```
NLV = $65,000
MM = |-$15,000| × 0.25 = $3,750
BP = ($65,000 - $3,750) / 0.50 = $122,500
```

**Notes**:
- Reversal split into close + open (two operations)
- Close always executes (reduces margin requirement)
- Open validated against buying power
- Cash account would reject the open short (no shorts allowed)

---

## Order Validation Logic

### Order Approval Decision Tree

```
Is order reducing existing position?
├─ YES → Approve (always allowed)
└─ NO (opening or increasing)
   └─ Is it a short sale?
      ├─ YES → Is short selling allowed?
      │  ├─ YES (margin account) → Check buying power
      │  └─ NO (cash account) → REJECT
      └─ NO (long position)
         └─ Check buying power
            ├─ Order cost ≤ BP → APPROVE
            └─ Order cost > BP → REJECT
```

### Example: Order Validation

**Account State**:
```
Cash:             $50,000
Position:         +100 AAPL @ $150
NLV:              $65,000
MM:               $3,750
BP:               $122,500
Account Type:     Margin
```

**Order 1**: Buy 800 shares AAPL @ $150 (long addition)
```
Order value = 800 × $150 = $120,000
BP = $122,500
Result: APPROVED ($120,000 < $122,500)
```

**Order 2**: Buy 900 shares AAPL @ $150 (exceeds BP)
```
Order value = 900 × $150 = $135,000
BP = $122,500
Result: REJECTED ($135,000 > $122,500)
```

**Order 3**: Sell 50 shares AAPL @ $150 (reducing)
```
Current position: +100
New position: +50
Result: APPROVED (reducing always allowed)
```

**Order 4**: Short 100 shares TSLA @ $200 (new short)
```
Order value = 100 × $200 = $20,000
BP = $122,500
Account allows shorts: YES (margin account)
Result: APPROVED ($20,000 < $122,500)
```

---

## Regulation T (RegT) Standards

### Overview

**Regulation T** is a Federal Reserve rule governing margin accounts in the United States.

**Key Requirements**:
1. **Initial Margin**: 50% (customers must pay half the purchase price)
2. **Maintenance Margin**: 25% minimum (can be higher at broker discretion)
3. **Pattern Day Trader**: Requires $25,000 minimum equity

### ml4t.backtest Implementation

**Defaults**:
```python
initial_margin = 0.50                # 50% (RegT standard)
long_maintenance_margin = 0.25       # 25% (RegT minimum for longs)
short_maintenance_margin = 0.30      # 30% (RegT standard for shorts)
```

**Equities (Percentage-based Margin)**:
```python
engine = Engine(
    feed,
    strategy,
    initial_cash=100_000.0,
    account_type="margin",
    initial_margin=0.60,                  # 60% (more conservative)
    long_maintenance_margin=0.30,         # 30% for longs
    short_maintenance_margin=0.35,        # 35% for shorts
)
```

**Futures (Fixed-Dollar Margin)**:
```python
engine = Engine(
    feed,
    strategy,
    initial_cash=100_000.0,
    account_type="margin",
    fixed_margin_schedule={
        "ES": (15_400.0, 14_000.0),       # S&P 500 E-mini: $15.4k initial, $14k maintenance
        "NQ": (20_900.0, 19_000.0),       # Nasdaq 100 E-mini: $20.9k initial, $19k maintenance
        "CL": (8_200.0, 7_500.0),         # Crude Oil: $8.2k initial, $7.5k maintenance
    },
)
```

**Mixed Portfolio (Equities + Futures)**:
```python
engine = Engine(
    feed,
    strategy,
    initial_cash=100_000.0,
    account_type="margin",
    initial_margin=0.50,                  # 50% for equities
    long_maintenance_margin=0.25,
    short_maintenance_margin=0.30,
    fixed_margin_schedule={               # Fixed margin for futures
        "ES": (15_400.0, 14_000.0),
    },
)
```

**Notes**:
- ml4t.backtest now supports asymmetric maintenance margins (different for longs vs shorts)
- Futures use fixed-dollar margin per contract (SPAN-style), not percentage of notional
- Assets in `fixed_margin_schedule` use fixed margin; others use percentage
- Pattern Day Trader rules NOT enforced in backtesting
- No distinction between "Reg T excess" and "SMA" (Simplified)

---

## Margin Call Handling

### What is a Margin Call?

A margin call occurs when:
```
NLV < MM
```

**Example**:
```
NLV = $10,000
MM = $12,000

Margin call triggered: $10,000 < $12,000
```

### ml4t.backtest Behavior

**Current Implementation**: Orders rejected when BP exhausted

```
BP = (NLV - MM) / IM

If BP ≤ 0:
  - Cannot open new positions
  - Can only close existing positions
```

**No Forced Liquidation**: ml4t.backtest does NOT automatically liquidate positions on margin call (future enhancement).

**Prevention**:
- Orders rejected before account becomes underwater
- Gatekeeper validates BP before execution
- Exit orders always execute (reduce margin requirement)

---

## Cash Account vs Margin Account

### Cash Account

```python
account_type = "cash"
```

**Constraints**:
```
Buying Power = Cash
Initial Margin = 1.0 (100%, no leverage)
Short Selling = Not Allowed
```

**Example**:
```
Cash: $50,000
BP = $50,000 / 1.0 = $50,000 (no leverage)
Can buy: $50,000 worth of stock
```

### Margin Account

```python
account_type = "margin"
initial_margin = 0.50
maintenance_margin = 0.25
```

**Constraints**:
```
Buying Power = (NLV - MM) / IM
Initial Margin = 0.50 (2x leverage)
Short Selling = Allowed
```

**Example**:
```
Cash: $50,000
BP = ($50,000 - $0) / 0.50 = $100,000 (2x leverage)
Can buy: $100,000 worth of stock
```

---

## Common Pitfalls

### Pitfall 1: Confusing Cash with Buying Power

❌ **Wrong**:
```python
cash = broker.account.cash
if cash >= order_value:
    broker.submit_order(asset, quantity)
```

✅ **Correct**:
```python
buying_power = broker.get_buying_power()
if buying_power >= order_value:
    broker.submit_order(asset, quantity)
```

**Why**: Cash doesn't account for margin or existing positions.

---

### Pitfall 2: Ignoring Unrealized P&L

❌ **Wrong**:
```
Equity = Cash
```

✅ **Correct**:
```
Equity = NLV = Cash + Σ(Position Values)
```

**Why**: Position value changes affect buying power even if not realized.

---

### Pitfall 3: Assuming Infinite Buying Power

❌ **Wrong**:
```python
# Margin account = unlimited trading
broker.submit_order("AAPL", 1_000_000)  # May be rejected
```

✅ **Correct**:
```python
buying_power = broker.get_buying_power()
max_shares = int(buying_power / price)
broker.submit_order("AAPL", max_shares)
```

**Why**: Margin accounts have leverage limits (typically 2x).

---

## API Usage Examples

### Check Buying Power Before Order

```python
class MyStrategy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        price = data["AAPL"]["close"]
        buying_power = broker.get_buying_power()

        # Calculate max affordable shares
        max_shares = int(buying_power / price)

        if max_shares >= 100:
            broker.submit_order("AAPL", 100)
```

### Handle Order Rejections

```python
class RobustStrategy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        price = data["AAPL"]["close"]

        # Try to buy 500 shares
        order = broker.submit_order("AAPL", 500)

        if order is None:
            # Order rejected - try smaller size
            buying_power = broker.get_buying_power()
            affordable_qty = int(buying_power / price)

            if affordable_qty > 0:
                broker.submit_order("AAPL", affordable_qty)
```

### Query Account State

```python
class MonitoringStrategy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        # Get account metrics
        cash = broker.account.cash
        equity = broker.account.equity
        positions = broker.account.positions
        buying_power = broker.get_buying_power()

        # Log state
        print(f"Cash: ${cash:,.0f}")
        print(f"Equity: ${equity:,.0f}")
        print(f"Buying Power: ${buying_power:,.0f}")
        print(f"Positions: {len(positions)}")
```

---

## References

1. **Regulation T**: [Federal Reserve Board](https://www.federalreserve.gov/supervisionreg/regt.htm)
2. **FINRA Margin Requirements**: [FINRA Rule 4210](https://www.finra.org/rules-guidance/rulebooks/finra-rules/4210)
3. **Pattern Day Trader Rules**: [SEC Day Trading](https://www.sec.gov/reportspubs/investor-publications/investorpubsdaytradinghtml.html)
4. **ml4t.backtest Source**: `src/ml4t/backtest/accounting/policy.py`

---

## Summary

**Key Takeaways**:
1. **NLV** = Total account value (cash + position values)
2. **MM** = Minimum equity to hold positions
3. **BP** = Maximum value of new positions allowed
4. **IM** = Equity required to open positions (50% = 2x leverage)
5. **Margin Call** = When NLV < MM (orders rejected)
6. **Cash Account** = No leverage, no shorts (BP = Cash)
7. **Margin Account** = 2x leverage, shorts allowed (BP > Cash)

**Formula Cheat Sheet**:
```
NLV = Cash + Σ(Position Values)
MM = Σ(|Position Values| × maintenance_margin)
BP = (NLV - MM) / initial_margin

Margin Call: NLV < MM
Order Approved: Order Value ≤ BP
```

---

**Last Updated**: 2025-11-24
**Version**: 1.1
**Author**: ml4t.backtest documentation team

**Changelog**:
- v1.1 (2025-11-24): Added asymmetric maintenance margins and futures fixed-dollar margin support
- v1.0 (2025-11-20): Initial version
