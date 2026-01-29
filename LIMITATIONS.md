# ml4t-backtest: Known Limitations and Assumptions

This document describes the known limitations, assumptions, and edge cases in ml4t-backtest.
Understanding these helps set realistic expectations for backtest results.

## Settlement and Clearing

### T+0 Settlement (Default)
- **Assumption**: All trades settle immediately (T+0)
- **Reality**: US equities settle T+2, futures T+1
- **Impact**: Buying power is available immediately after a sale
- **Workaround**: Use `cash_buffer_pct` in BacktestConfig to reserve liquidity

### No Settlement Queue
- There is no modeling of pending settlements
- Cash from sales is immediately available for new purchases
- This can overestimate trading capacity in high-turnover strategies

## Corporate Actions

### Pre-Adjusted Data Assumed
- **Assumption**: Input data is already split/dividend adjusted
- Backtesting with unadjusted data will produce incorrect results
- Recommendation: Use adjusted close prices from data providers

### No Automatic Liquidation on Delisting
- Delisted securities are not automatically closed
- Positions persist until explicit exit signal or end of data
- Manual handling required for bankruptcy/delisting scenarios

### Dividends and Interest
- Dividend payments are not modeled (assumed reinvested in price)
- Short selling borrowing costs are not modeled
- Bond coupon payments are not supported

## Execution Model

### SAME_BAR Mode (Look-Ahead Bias Risk)
- Orders filled at the same bar's close introduce look-ahead bias
- Use `NEXT_BAR` mode for realistic execution
- SAME_BAR mode useful for vectorized strategy comparison only

### NEXT_BAR Mode Edge Cases
1. **Deferred Exit Re-Entry**: When a stop-loss triggers and defers to next bar's open,
   entry signals on the same bar are blocked to prevent same-bar re-entry.
   This matches VectorBT Pro behavior.

2. **Order Queue Priority**: Exits are processed before entries each bar to ensure
   capital freed by exits is available for new entries.

### No Intrabar Order Priority
- Within a single bar, order execution sequence is:
  1. Deferred exits from previous bar (at open)
  2. Risk rule exits (stop-loss, take-profit)
  3. Strategy entry orders
- Cannot model specific intrabar order sequences

### Gap Handling
- Gap opens are handled: if bar opens beyond stop price, fill at open
- This models slippage during fast market conditions
- Large gaps may fill at worse prices than the stop level

## Market Impact

### Fixed Slippage Model (Default)
- Default slippage is percentage-based, not volume-dependent
- Does not account for order book depth or market impact
- Large orders in illiquid securities may face higher slippage

### Volume Limits (Optional)
- `ExecutionLimits` can constrain fills to percentage of bar volume
- Requires volume data in the feed
- Partial fills create multi-bar execution

### No Order Book Modeling
- All orders assume sufficient liquidity at quoted prices
- Limit orders fill if price is touched (no queue position)
- Market orders fill at close/open (no spread modeling)

## Short Selling

### Unlimited Borrowing Assumed
- No locate requirements or borrow availability checks
- Any security can be shorted without constraint
- No hard-to-borrow fees or rebate rates

### Margin Calls
- Margin accounts enforce maintenance margin requirements
- Position liquidation on margin call is not automatic
- Strategies should monitor margin via `get_buying_power()`

### Uptick Rule
- No uptick rule (Rule 201) enforcement
- Short sales allowed at any price level

## Risk Management

### Position-Level vs Portfolio-Level
- Position rules (StopLoss, TakeProfit, etc.) operate independently per position
- Portfolio rules (MaxDrawdown, MaxPositions) check aggregate exposure
- Cross-position hedging is not automatically recognized

### Stop Order Fill Modes
- `STOP_PRICE`: Fill at exact stop price (default)
- `BAR_EXTREME`: Fill at bar's low/high (conservative)
- `NEXT_BAR_OPEN`: Defer to next bar's open (Zipline-style)
- `CLOSE_PRICE`: Fill at bar's close (VectorBT Pro-style)

### Trailing Stop Initialization
- New positions use close price as initial high-water mark
- Entry bar's high is NOT used (matches VectorBT Pro)
- High-water mark updates begin on the bar after entry

## Data Requirements

### OHLCV Data Expected
- Full OHLCV bars provide most accurate execution simulation
- Close-only data limits stop/limit order accuracy
- Volume data required for `ExecutionLimits`

### No Tick Data Support
- Bar-based execution only
- Tick-by-tick order book simulation not available
- Use smaller bar intervals for higher precision

### Timezone Handling
- Naive datetimes default to UTC
- Calendar sessions use exchange timezone
- Mixed timezones may cause unexpected behavior

## Calendar and Sessions

### Trading Calendar (Optional)
- Exchange calendars enforce trading hours
- `enforce_sessions=True` skips bars outside sessions
- Overnight gaps handled when calendar is enabled

### CME/Futures Overnight Sessions
- Overnight sessions (e.g., CME equity futures) are supported
- Set `overnight=True` for calendar types that trade overnight
- Session boundaries respect next-day roll

## Numerical Precision

### Fractional Shares
- Fractional quantities supported by default
- Set `share_type=INTEGER` for whole shares only
- Very small positions may cause rounding issues

### Price Precision
- All prices stored as Python floats (IEEE 754 double)
- May introduce minor floating-point errors
- Not suitable for sub-penny precision requirements

## What's NOT Modeled

The following real-world factors are **not** simulated:

1. **Broker API latency** - Orders execute instantly
2. **Quote stuffing / HFT interference** - Clean price discovery assumed
3. **Exchange halts / circuit breakers** - All bars are tradeable
4. **Order cancellation delays** - Cancels are instant
5. **Partial fills due to queue position** - Limit orders fill completely or not at all
6. **Regulatory restrictions** - PDT, accredited investor, etc.
7. **Tax implications** - No wash sale tracking
8. **Currency conversion** - Single currency assumed

## Validation Status

ml4t-backtest has been validated against:

| Framework | Trade Count | Match Rate | Notes |
|-----------|-------------|------------|-------|
| VectorBT Pro | 119,000+ | 100% | Full feature coverage |
| Backtrader | 119,577 | 100% | LONG positions only for risk rules |
| Zipline | 119,577 | 100% | LONG positions only for risk rules |
| VectorBT OSS | 114,607 | 100% | LONG positions only for risk rules |

### What's Validated

**Core Execution** (119k+ trades across 500 assets × 10 years):
- Entry/exit timing
- Fill prices
- Position tracking
- P&L calculations
- Multi-asset portfolio management
- Cash constraints and margin

**Risk Rules - LONG Positions** (Scenario tests):
- Stop-loss triggers
- Take-profit triggers
- Trailing stop (close-based HWM)
- Bracket orders

### What's NOT Validated (Gaps)

**SHORT Position Risk Rules** (VBT Pro only):
- SHORT stop-loss triggers - VBT Pro validated, others untested
- SHORT take-profit triggers - VBT Pro validated, others untested
- SHORT trailing stop - VBT Pro validated, others untested
- SHORT bracket orders - VBT Pro validated, others untested

**Rule Combinations** (Not tested anywhere):
- TSL + TP when both trigger same bar
- TSL + SL when both trigger same bar
- TSL + TP + SL (triple rule interaction)
- Rule priority in simultaneous breach

**Stress Conditions** (Limited testing):
- Gap openings through stop levels
- High volatility periods
- Extended (1000+ bar) trailing stop tracking

## Recommendations

### For Realistic Backtests
1. Use `NEXT_BAR` execution mode
2. Set `slippage_rate` >= 0.001 (0.1%)
3. Set `commission_rate` >= 0.001 (0.1%)
4. Enable `reject_on_insufficient_cash`
5. Use exchange calendar for accurate session handling

### For Strategy Comparison
1. Use `SAME_BAR` mode for vectorized comparison with VectorBT
2. Use identical settings across all frameworks
3. Focus on relative performance, not absolute returns

### For Production Readiness
1. Validate with `Mode.REALISTIC` preset
2. Run with historical crisis periods (2008, 2020, 2022)
3. Test with varied slippage and commission assumptions
4. Paper trade before live deployment
