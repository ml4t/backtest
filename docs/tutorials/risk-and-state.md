# Follow stops and portfolio risk through a backtest

This run uses two synthetic unit-multiplier assets, A and B, with $2,500
starting cash and no fees. Both orders are placed on January 2 and filled at
$100 on January 3. B has a 5% stop loss. A has no position stop, so it remains
available for a portfolio drawdown rule that halves exposure after a 10%
drawdown. All bars are daily; the declared open, high, low, and close are equal,
so the $94 B exit is an observable gap below its $95 stop level. The run uses
default next-bar order timing.

## Run the strategy

`ManagedRisk` retains its entry flag, risk manager, and callback log between
bars. Its `on_start` installs B's stop and initializes the portfolio's high
water mark. On each callback it passes the broker's current marked equity and
positions to `RiskManager.update`. That call may queue a reduction for the next
bar. The log prints the manager's reported action and the broker's pending
orders separately: a repeated breach can report `reduce` without submitting
another order.

<!-- ml4t-doc-test: tutorial-risk-state -->
```python
from datetime import datetime
from importlib.metadata import version

import polars as pl
from ml4t.backtest import BacktestConfig, DataFeed, Engine, StopLoss, Strategy
from ml4t.backtest.risk.portfolio import MaxDrawdownLimit, RiskManager


days = (2, 3, 4, 5, 6, 7, 8)
marks = {"A": (100, 100, 80, 80, 120, 80, 80), "B": (100, 100, 94, 94, 94, 94, 94)}
prices = pl.DataFrame({
    "timestamp": [datetime(2024, 1, day) for day in days for _ in ("A", "B")],
    "asset": [asset for _ in days for asset in ("A", "B")],
    "open": [marks[asset][i] for i in range(len(days)) for asset in ("A", "B")],
    "high": [marks[asset][i] for i in range(len(days)) for asset in ("A", "B")],
    "low": [marks[asset][i] for i in range(len(days)) for asset in ("A", "B")],
    "close": [marks[asset][i] for i in range(len(days)) for asset in ("A", "B")],
    "volume": [1000.0] * (len(days) * 2),
})

class ManagedRisk(Strategy):
    def on_start(self, broker):
        broker.set_position_rules(StopLoss(pct=0.05), asset="B")
        self.manager = RiskManager(limits=[MaxDrawdownLimit(
            max_drawdown=0.10, action="reduce", reduction_pct=0.5,
        )])
        self.manager.initialize(initial_equity=broker.get_account_value())
        self.events = []
        self.entered = False

    def on_data(self, timestamp, data, context, broker):
        positions = {asset: pos.market_value for asset, pos in broker.positions.items()}
        results = self.manager.update(
            equity=broker.get_account_value(), positions=positions,
            timestamp=timestamp, broker=broker,
        )
        self.events.append((timestamp, round(broker.get_account_value(), 2),
                            {a: p.quantity for a, p in broker.positions.items()},
                            [r.action for r in results],
                            [(o.asset, o.quantity) for o in broker.get_pending_orders()]))
        if not self.entered:
            broker.submit_order("A", 16)
            broker.submit_order("B", 4)
            self.entered = True

strategy = ManagedRisk()
result = Engine(DataFeed(prices_df=prices), strategy,
                BacktestConfig(initial_cash=2500)).run()
print("ml4t-backtest " + version("ml4t-backtest"))
for timestamp, equity, positions, actions, pending in strategy.events:
    print(f"{timestamp:%Y-%m-%d}: equity={equity:.0f} positions={positions} "
          f"risk={actions} pending={pending}")
for fill in result.fills:
    if fill.side.value == "sell":
        print(f"exit {fill.timestamp:%Y-%m-%d}: {fill.asset} {fill.quantity:g} "
              f"at {fill.price:.0f} reason={fill.exit_reason_detail}")
for state in result.to_portfolio_state_dataframe().to_dicts():
    if state["timestamp"].day in (4, 5, 6, 7, 8):
        print(f"exposure {state['timestamp']:%Y-%m-%d}: "
              f"gross={state['gross_exposure']:.0f} cash={state['cash']:.0f}")
```

<!-- ml4t-doc-output: tutorial-risk-state -->
```text
ml4t-backtest {package_version}
2024-01-02: equity=2500 positions={} risk=[] pending=[]
2024-01-03: equity=2500 positions={'A': 16.0, 'B': 4.0} risk=[] pending=[]
2024-01-04: equity=2156 positions={'A': 16.0} risk=['reduce'] pending=[('A', 8.0)]
2024-01-05: equity=2156 positions={'A': 8.0} risk=['reduce'] pending=[]
2024-01-06: equity=2476 positions={'A': 8.0} risk=[] pending=[]
2024-01-07: equity=2156 positions={'A': 8.0} risk=['reduce'] pending=[('A', 4.0)]
2024-01-08: equity=2156 positions={'A': 4.0} risk=['reduce'] pending=[]
exit 2024-01-04: B 4 at 94 reason=stop_loss_5.0%
exit 2024-01-05: A 8 at 80 reason=risk reduction: drawdown 13.8% >= 10.0%
exit 2024-01-08: A 4 at 80 reason=risk reduction: drawdown 13.8% >= 10.0%
exposure 2024-01-04: gross=1280 cash=876
exposure 2024-01-05: gross=640 cash=1516
exposure 2024-01-06: gross=960 cash=1516
exposure 2024-01-07: gross=640 cash=1516
exposure 2024-01-08: gross=320 cash=1836
```

## Read the event order

1. On January 3, the queued entries fill and the strategy observes 16 A and 4 B.
2. On January 4, the new $94 B bar triggers its stop. The engine fills that
   stop before calling the strategy. The callback therefore sees only A and
   $2,156 equity, a 13.8% drawdown from $2,500. It queues a sell for half of
   A's 16 shares.
3. On January 5, the queued reduction fills at $80 before the callback. Gross
   exposure drops from $1,280 to $640. The drawdown remains above 10%, so the
   manager reports a breach but does not queue a duplicate reduction.
4. On January 6, A reaches $120 and equity rises to $2,476. This clears the
   drawdown breach and allows a later risk event.
5. On January 7, A returns to $80. The new breach queues a four-share sell,
   half of the eight shares then held. The January 8 fill leaves four shares
   and $320 gross exposure.

The risk manager stores whether a continuous breach has already caused a
reduction. The strategy retains whether it has submitted the initial orders.
Neither decision can be reconstructed safely from a precomputed vector of
entry signals alone. Inspect fill `exit_reason_detail` and portfolio-state
rows to separate the stop exit, queued reductions, and changes in exposure.
The [Risk Management](../user-guide/risk-management.md) guide covers the rules
and the [stateful strategy examples](https://github.com/ml4t/backtest/blob/main/examples/stateful_strategies.py)
show other uses of callback state.

## In the book

Chapter 19, Section 19.4, [Drawdowns, path risk, and time to recovery](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/19_risk_management/README.md),
puts this event trace in a broader risk framework. [Notebook 10, ml4t-backtest
risk demo](https://github.com/stefan-jansen/machine-learning-for-trading/blob/366e1d51ace2d851776499a68da3d6e3c2641b02/19_risk_management/10_ml4t_backtest_risk_demo.ipynb)
uses the library's position rules and portfolio controls on larger examples.
