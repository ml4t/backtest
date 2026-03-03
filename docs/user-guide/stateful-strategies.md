# Stateful Strategies

Stateful strategies are the reason event-driven backtesting exists. In a vectorized framework, every signal is computed in advance from historical data alone. In an event-driven engine, each trading decision can depend on the **entire history of prior decisions** — fills, P&L, position state, equity path. This feedback loop is impossible to vectorize.

## When You Need Event-Driven

Use vectorized backtesting when your signal is a pure function of price history:

```
signal[t] = f(prices[0:t])    # No feedback — vectorizable
```

Use event-driven backtesting when your trading decision depends on execution state:

```
action[t] = g(prices[0:t], fills[0:t], equity[0:t])    # Feedback — requires event loop
```

Five categories of stateful patterns:

| Pattern | State Dependency | Example |
|---------|-----------------|---------|
| Feedback loops | Position size depends on realized P&L | Kelly sizing |
| Conditional chains | Entry N depends on P&L of entries 1..N-1 | Pyramiding |
| Cross-asset coordination | Asset A's order depends on asset B's fill | Pairs trading |
| Path-dependent sizing | Equity curve drives future position sizes | Drawdown circuit breaker |
| Reactive order management | Each fill triggers new orders | Grid trading |

## Pattern 1: Feedback Loops (Adaptive Kelly Sizing)

Position size adapts based on realized win rate and payoff ratio. The feedback: `position_size → P&L → Kelly_fraction → next_position_size`.

```python
from ml4t.backtest import Strategy

class AdaptiveKellySizingStrategy(Strategy):
    def __init__(self, base_size=0.10, min_size=0.02, max_size=0.25,
                 kelly_fraction=0.5, min_trades=5):
        self.base_size = base_size
        self.min_size = min_size
        self.max_size = max_size
        self.kelly_fraction = kelly_fraction
        self.min_trades = min_trades

    def _kelly_size(self, broker, asset):
        """Half-Kelly position sizing from realized trade stats."""
        stats = broker.get_asset_stats(asset)
        if stats.total_trades < self.min_trades:
            return self.base_size

        w = stats.recent_win_rate
        wins = [p for p in stats.recent_pnls if p > 0]
        losses = [p for p in stats.recent_pnls if p <= 0]
        if not wins or not losses:
            return self.base_size

        r = sum(wins) / len(wins) / abs(sum(losses) / len(losses))
        f_star = max(0.0, w - (1 - w) / r) * self.kelly_fraction
        return max(self.min_size, min(self.max_size, f_star))

    def on_data(self, timestamp, data, context, broker):
        for asset, bar in data.items():
            signal = bar.get("signals", {}).get("signal", 0) or 0
            price = bar.get("close", 0)
            if price <= 0:
                continue

            position = broker.get_position(asset)
            if position is None and signal > 0.5:
                size_frac = self._kelly_size(broker, asset)
                shares = (broker.get_account_value() * size_frac) / price
                if shares > 0:
                    broker.submit_order(asset, shares)
            elif position is not None and signal < -0.5:
                broker.close_position(asset)
```

**Why vectorized fails**: The Kelly fraction at bar N depends on the win rate from trades 0..N-1, but each trade's P&L depends on its size, which was set by the Kelly fraction at entry time. This circular dependency requires sequential execution.

## Pattern 2: Conditional Chains (Pyramiding)

Add to winners: each new entry triggers only when prior entries have accumulated enough unrealized profit. The chain: `entry_1 → profit_check → entry_2 → profit_check → entry_3`.

```python
from collections import defaultdict
from ml4t.backtest import Strategy

class PyramidingStrategy(Strategy):
    def __init__(self, max_levels=3, profit_threshold=0.02,
                 base_size=0.10, size_decay=0.5):
        self.max_levels = max_levels
        self.profit_threshold = profit_threshold
        self.base_size = base_size
        self.size_decay = size_decay
        self.pyramid_levels = defaultdict(int)

    def on_data(self, timestamp, data, context, broker):
        for asset, bar in data.items():
            signal = bar.get("signals", {}).get("signal", 0) or 0
            price = bar.get("close", 0)
            if price <= 0:
                continue

            position = broker.get_position(asset)

            if position is None:
                if signal > 0.5:
                    equity = broker.get_account_value()
                    shares = (equity * self.base_size) / price
                    if shares > 0:
                        broker.submit_order(asset, shares)
                        self.pyramid_levels[asset] = 1
                continue

            if signal < -0.5:
                broker.close_position(asset)
                self.pyramid_levels[asset] = 0
                continue

            # Pyramid up on profit
            level = self.pyramid_levels[asset]
            pnl_pct = position.pnl_percent()
            if level < self.max_levels and pnl_pct > self.profit_threshold * level:
                decay = self.size_decay ** level
                equity = broker.get_account_value()
                shares = (equity * self.base_size * decay) / price
                if shares > 0:
                    broker.submit_order(asset, shares)
                    self.pyramid_levels[asset] = level + 1
```

**Why vectorized fails**: Whether entry 2 happens depends on the unrealized P&L of entry 1, which depends on entry 1's fill price and size. The fill price includes slippage, which may depend on volume and order size. Each link in the chain is only knowable at execution time.

## Pattern 3: Cross-Asset Coordination (Pairs Trading)

Trade the spread between two correlated assets. Entry and exit of asset A is conditioned on the price relationship with asset B.

```python
from ml4t.backtest import Strategy

class PairsTradingStrategy(Strategy):
    def __init__(self, asset_a="A", asset_b="B", lookback=20,
                 entry_zscore=2.0, exit_zscore=0.5, position_size=0.10):
        self.asset_a = asset_a
        self.asset_b = asset_b
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.position_size = position_size
        self.price_history_a = []
        self.price_history_b = []
        self.pair_status = "flat"

    def _compute_zscore(self):
        if len(self.price_history_a) < self.lookback:
            return None
        ratios = [b / a for a, b in zip(
            self.price_history_a[-self.lookback:],
            self.price_history_b[-self.lookback:]) if a > 0]
        if len(ratios) < 2:
            return None
        mean_r = sum(ratios) / len(ratios)
        std_r = (sum((r - mean_r) ** 2 for r in ratios) / (len(ratios) - 1)) ** 0.5
        if std_r == 0:
            return None
        return (self.price_history_b[-1] / self.price_history_a[-1] - mean_r) / std_r

    def on_data(self, timestamp, data, context, broker):
        bar_a, bar_b = data.get(self.asset_a), data.get(self.asset_b)
        if bar_a is None or bar_b is None:
            return

        price_a, price_b = bar_a.get("close", 0), bar_b.get("close", 0)
        if price_a <= 0 or price_b <= 0:
            return

        self.price_history_a.append(price_a)
        self.price_history_b.append(price_b)

        z = self._compute_zscore()
        if z is None:
            return

        equity = broker.get_account_value()

        if self.pair_status == "flat":
            if z > self.entry_zscore:
                shares_a = (equity * self.position_size) / price_a
                shares_b = (equity * self.position_size) / price_b
                if shares_a > 0 and shares_b > 0:
                    broker.submit_order(self.asset_a, shares_a)
                    broker.submit_order(self.asset_b, -shares_b)
                    self.pair_status = "short_spread"
            elif z < -self.entry_zscore:
                shares_a = (equity * self.position_size) / price_a
                shares_b = (equity * self.position_size) / price_b
                if shares_a > 0 and shares_b > 0:
                    broker.submit_order(self.asset_a, -shares_a)
                    broker.submit_order(self.asset_b, shares_b)
                    self.pair_status = "long_spread"
        elif abs(z) < self.exit_zscore:
            broker.close_position(self.asset_a)
            broker.close_position(self.asset_b)
            self.pair_status = "flat"
```

**Why vectorized fails**: Position in A affects available capital for B. If A's order gets rejected (insufficient cash, margin limits), B shouldn't be entered either — the pair is meaningless as a single leg. Capital allocation across the two legs depends on execution outcomes.

## Pattern 4: Path-Dependent Sizing (Drawdown Circuit Breaker)

Reduce or halt trading when portfolio drawdown exceeds thresholds. The feedback: `equity_curve → drawdown → sizing_multiplier → future_equity_curve`.

```python
from ml4t.backtest import Strategy

class DrawdownCircuitBreakerStrategy(Strategy):
    def __init__(self, base_size=0.10, caution_threshold=0.05,
                 halt_threshold=0.10, reduction_factor=0.5, recovery_rate=0.01):
        self.base_size = base_size
        self.caution_threshold = caution_threshold
        self.halt_threshold = halt_threshold
        self.reduction_factor = reduction_factor
        self.recovery_rate = recovery_rate
        self.peak_equity = 0.0
        self.sizing_multiplier = 1.0

    def on_data(self, timestamp, data, context, broker):
        equity = broker.get_account_value()

        # Update peak and compute drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
        dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0

        # Adjust sizing multiplier
        if dd < self.caution_threshold:
            self.sizing_multiplier = min(1.0, self.sizing_multiplier + self.recovery_rate)
        elif dd < self.halt_threshold:
            range_pct = (dd - self.caution_threshold) / (self.halt_threshold - self.caution_threshold)
            self.sizing_multiplier = self.reduction_factor * (1 - range_pct)
        else:
            self.sizing_multiplier = 0.0

        for asset, bar in data.items():
            signal = bar.get("signals", {}).get("signal", 0) or 0
            price = bar.get("close", 0)
            if price <= 0:
                continue

            position = broker.get_position(asset)
            if position is None and signal > 0.5:
                if self.sizing_multiplier <= 0:
                    continue  # Trading halted
                effective_size = self.base_size * self.sizing_multiplier
                shares = (equity * effective_size) / price
                if shares > 0:
                    broker.submit_order(asset, shares)
            elif position is not None and signal < -0.5:
                broker.close_position(asset)
```

**Why vectorized fails**: The sizing multiplier at bar N depends on the drawdown from bars 0..N-1, but the equity at each prior bar depends on the sizing decisions made at those bars. The equity path and the sizing path are co-determined — you can't compute one without the other.

## Pattern 5: Reactive Order Management (Grid Trading)

Place limit orders on a grid; each fill triggers a new order at the adjacent level. The grid state is fully dynamic and depends on fill history.

```python
from ml4t.backtest import Strategy
from ml4t.backtest.types import OrderType

class GridTradingStrategy(Strategy):
    def __init__(self, asset="ASSET", grid_spacing=0.01, num_levels=5,
                 order_size=100, max_position=500, recenter_threshold=0.05):
        self.asset = asset
        self.grid_spacing = grid_spacing
        self.num_levels = num_levels
        self.order_size = order_size
        self.max_position = max_position
        self.recenter_threshold = recenter_threshold
        self.reference_price = 0.0
        self.grid_orders = {}  # level → order_id
        self.initialized = False

    def _place_grid(self, broker, price):
        self.reference_price = price
        self.grid_orders.clear()
        for i in range(1, self.num_levels + 1):
            buy_order = broker.submit_order(
                self.asset, self.order_size,
                order_type=OrderType.LIMIT,
                limit_price=price * (1 - self.grid_spacing * i))
            if buy_order:
                self.grid_orders[-i] = buy_order.order_id
            sell_order = broker.submit_order(
                self.asset, -self.order_size,
                order_type=OrderType.LIMIT,
                limit_price=price * (1 + self.grid_spacing * i))
            if sell_order:
                self.grid_orders[i] = sell_order.order_id

    def on_data(self, timestamp, data, context, broker):
        bar = data.get(self.asset)
        if bar is None:
            return
        price = bar.get("close", 0)
        if price <= 0:
            return

        if not self.initialized:
            self._place_grid(broker, price)
            self.initialized = True
            return

        # React to fills
        for level, order_id in list(self.grid_orders.items()):
            order = broker.get_order(order_id)
            if order is not None and order.status.value == "filled":
                del self.grid_orders[level]
                # Buy filled → place sell above; Sell filled → place buy below
                new_level = level + 1 if level < 0 else level - 1
                if new_level != 0 and new_level not in self.grid_orders:
                    new_qty = -self.order_size if level < 0 else self.order_size
                    new_price = self.reference_price * (
                        1 + self.grid_spacing * abs(new_level) * (-1 if new_qty > 0 else 1))
                    order = broker.submit_order(
                        self.asset, new_qty,
                        order_type=OrderType.LIMIT, limit_price=new_price)
                    if order:
                        self.grid_orders[new_level] = order.order_id

        # Recenter if price drifted too far
        if abs(price - self.reference_price) / self.reference_price > self.recenter_threshold:
            for oid in self.grid_orders.values():
                broker.cancel_order(oid)
            self.grid_orders.clear()
            self._place_grid(broker, price)
```

**Why vectorized fails**: The entire order book is reactive — each fill changes the grid, which changes which orders exist, which changes future fills. The full state evolution requires sequential event processing.

## Combining Patterns

Real strategies often combine multiple stateful patterns. For example, a pairs trading strategy with drawdown protection:

```python
class ProtectedPairsStrategy(Strategy):
    def __init__(self, asset_a, asset_b):
        self.pairs = PairsTradingStrategy(asset_a, asset_b)
        self.breaker = DrawdownCircuitBreakerStrategy()

    def on_data(self, timestamp, data, context, broker):
        # Update drawdown state
        self.breaker.on_data(timestamp, {}, context, broker)

        # Only trade pairs if circuit breaker allows
        if self.breaker.sizing_multiplier > 0:
            self.pairs.on_data(timestamp, data, context, broker)
```

## Broker State API

Stateful strategies depend on querying execution state. Key broker methods:

| Method | Returns | Used By |
|--------|---------|---------|
| `get_position(asset)` | Position or None | All patterns |
| `get_positions()` | Dict of all positions | Multi-asset |
| `get_account_value()` | Total portfolio value | Sizing |
| `get_cash()` | Available cash | Capital allocation |
| `get_asset_stats(asset)` | Trade statistics | Kelly sizing |
| `get_order(order_id)` | Order status | Grid trading |
| `get_rejected_orders()` | List of rejections | Error handling |

`Position` objects expose:

| Attribute | Description |
|-----------|-------------|
| `quantity` | Current shares held |
| `entry_price` | Average entry price |
| `bars_held` | Bars since entry |
| `pnl_percent()` | Unrealized P&L as percentage |
| `high_water_mark` | Highest price since entry |

## Testing Stateful Strategies

Stateful strategies need tests that verify state transitions, not just final P&L:

```python
def test_kelly_adapts_to_losses():
    """After consecutive losses, Kelly should reduce position size."""
    strategy = AdaptiveKellySizingStrategy(base_size=0.10, min_trades=3)
    # ... run with losing signals ...
    assert strategy.size_history[-1] < strategy.size_history[0]

def test_pyramiding_respects_max_levels():
    """Should not exceed max_levels even with strong profits."""
    strategy = PyramidingStrategy(max_levels=3)
    # ... run with continuously profitable signal ...
    assert max(strategy.pyramid_levels.values()) <= 3
```

See `examples/test_stateful_strategies.py` for 45+ test methods covering all five patterns.

## Next Steps

- [Strategies](strategies.md) — strategy interface and broker methods
- [Risk Management](risk-management.md) — automatic position rules
- [Execution Semantics](execution-semantics.md) — fill timing and ordering
