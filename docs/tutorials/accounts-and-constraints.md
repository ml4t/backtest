# Compare accounts, share sizing, and exposure caps

This tutorial holds prices and orders fixed while changing one account or sizing
setting at a time. The declared USD prices are $100 on each of three synthetic
daily bars. AAPL and MSFT are treated as equities with a unit multiplier, zero
fees, and next-bar market fills. The fractional-share case assumes a venue that
supports decimal equity quantities. The margin case uses the default 50% initial
margin requirement; it is an example of buying power, not a claim that every
broker offers the same terms. The [account policy](../user-guide/accounts.md)
and [rebalancing](../user-guide/rebalancing.md) guides explain the settings in
full.

## Run the controlled cases

The first six runs submit one AAPL order at the January 2 close and inspect the
January 3 fill or rejection. `rejected_orders` supplies a structured code when
a gatekeeper declines an order. The last three runs request 80% each in AAPL
and MSFT. Each cap is applied before sizing the next-bar orders. The final
portfolio-state row records cash and signed market exposure after the fills.

<!-- ml4t-doc-test: tutorial-accounts-constraints -->
```python
from datetime import datetime
from importlib.metadata import version

import polars as pl
from ml4t.backtest import BacktestConfig, DataFeed, Engine, OrderSide, Strategy
from ml4t.backtest import RebalanceConfig, TargetWeightExecutor
from ml4t.backtest.config import ShareType


prices = pl.DataFrame({
    "timestamp": [datetime(2024, 1, day) for day in (2, 3, 4) for _ in range(2)],
    "asset": [asset for _ in range(3) for asset in ("AAPL", "MSFT")],
    "open": [100.0] * 6,
    "high": [100.0] * 6,
    "low": [100.0] * 6,
    "close": [100.0] * 6,
    "volume": [10_000.0] * 6,
})


class OneOrder(Strategy):
    def __init__(self, quantity, side):
        self.quantity = quantity
        self.side = side
        self.submitted = False

    def on_data(self, timestamp, data, context, broker):
        if not self.submitted:
            broker.submit_order("AAPL", self.quantity, self.side)
            self.submitted = True


def report(label, result):
    state = result.to_portfolio_state_dataframe().to_dicts()[-1]
    fills = [(fill.asset, fill.side.value, fill.quantity) for fill in result.fills]
    codes = [row["rejection_code"] for row in result.to_rejected_orders_dataframe().to_dicts()]
    print(f"{label}: fills={fills} cash={state['cash']:.0f} "
          f"net={state['net_exposure']:.0f} rejected={codes}")


print("ml4t-backtest " + version("ml4t-backtest"))
aapl = prices.filter(pl.col("asset") == "AAPL")
account_cases = [
    ("cash buy", 150, OrderSide.BUY, {}),
    ("margin buy", 150, OrderSide.BUY, {"allow_leverage": True}),
    ("cash short", 10, OrderSide.SELL, {}),
    ("short enabled", 10, OrderSide.SELL, {"allow_short_selling": True}),
    ("integer", 10.5, OrderSide.BUY, {"share_type": ShareType.INTEGER}),
    ("fractional", 10.5, OrderSide.BUY, {"share_type": ShareType.FRACTIONAL}),
]
for label, quantity, side, options in account_cases:
    result = Engine(DataFeed(prices_df=aapl), OneOrder(quantity, side),
                    BacktestConfig(initial_cash=10_000, **options)).run()
    report(label, result)


class TwoTargets(Strategy):
    def __init__(self, single_cap, gross_cap):
        self.executor = TargetWeightExecutor(RebalanceConfig(
            max_single_weight=single_cap, max_gross_leverage=gross_cap,
            allow_fractional=True,
        ))
        self.submitted = False

    def on_data(self, timestamp, data, context, broker):
        if not self.submitted:
            self.executor.execute({"AAPL": 0.8, "MSFT": 0.8}, data, broker,
                                  timestamp=timestamp)
            self.submitted = True


for label, single_cap, gross_cap in [
    ("uncapped targets", 1.0, None),
    ("50% position cap", 0.5, None),
    ("100% gross cap", 1.0, 1.0),
]:
    result = Engine(DataFeed(prices_df=prices), TwoTargets(single_cap, gross_cap),
                    BacktestConfig(initial_cash=10_000, allow_leverage=True,
                                   share_type=ShareType.FRACTIONAL)).run()
    report(label, result)
```

<!-- ml4t-doc-output: tutorial-accounts-constraints -->
```text
ml4t-backtest {package_version}
cash buy: fills=[] cash=10000 net=0 rejected=['insufficient_cash']
margin buy: fills=[('AAPL', 'buy', 150.0)] cash=-5000 net=15000 rejected=[]
cash short: fills=[] cash=10000 net=0 rejected=['account_restriction']
short enabled: fills=[('AAPL', 'sell', 10.0)] cash=11000 net=-1000 rejected=[]
integer: fills=[('AAPL', 'buy', 10.0)] cash=9000 net=1000 rejected=[]
fractional: fills=[('AAPL', 'buy', 10.5)] cash=8950 net=1050 rejected=[]
uncapped targets: fills=[('AAPL', 'buy', 80.0), ('MSFT', 'buy', 80.0)] cash=-6000 net=16000 rejected=[]
50% position cap: fills=[('AAPL', 'buy', 50.0), ('MSFT', 'buy', 50.0)] cash=0 net=10000 rejected=[]
100% gross cap: fills=[('AAPL', 'buy', 50.0), ('MSFT', 'buy', 50.0)] cash=0 net=10000 rejected=[]
```

## Read the account results

A cash account rejects the $15,000 purchase because it has $10,000. With
leverage enabled, the identical 150-share order fills and leaves $5,000 of
negative cash against $15,000 of long exposure. A cash account rejects a new
short with `account_restriction`; enabling short selling admits the same
10-share sell, leaves $11,000 cash including sale proceeds, and records
-$1,000 signed exposure. Cash alone is therefore not the account's equity or
spendable buying power. The two share-type runs submit the same 10.5-share
request: integer execution fills 10 shares, while fractional execution fills
10.5. Check the venue's actual share and lot rules before using the latter.

## Read the exposure results

Without a rebalance cap, two 80% targets buy 80 shares each and borrow $6,000.
`max_single_weight=0.5` cuts each target to 50%; `max_gross_leverage=1.0`
scales their combined 160% target down to 100%. Both changes produce two
50-share fills and no borrowing in this symmetric example. The caps cause
constrained sizing, not a rejected order; the account gatekeeper still checks
cash and margin when the resulting orders execute. Compare `fills`,
`to_rejected_orders_dataframe()`, and `to_portfolio_state_dataframe()` together
instead of inferring acceptance from requested weights.

## In the book

Chapter 17, Section 17.1, [Defining the allocation problem](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/17_portfolio_construction/README.md),
sets out the role of constraints and leverage. [Notebook 07, Conformal position
sizing](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/17_portfolio_construction/07_conformal_position_sizing.ipynb)
uses registered ETF and futures predictions to study a larger sizing problem.
The controlled runs here make the account and execution layer inspectable
before applying a book allocator.
