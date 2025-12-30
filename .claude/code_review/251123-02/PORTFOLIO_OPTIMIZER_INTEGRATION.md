# Proposal: Portfolio Optimizer Integration

**Date**: 2025-11-23
**Author**: Claude Code (prepared for external review)
**Status**: Draft - Seeking Validation & Alternative Ideas
**Library**: ml4t.backtest

---

## Executive Summary

This proposal addresses how users can integrate external portfolio optimizers (like riskfolio-lib, PyPortfolioOpt, cvxpy) with ml4t.backtest for position sizing. We present three design options, analyze tradeoffs, and recommend a minimal-footprint approach.

**Key Question**: Where should portfolio optimization logic live in the backtest architecture?

---

## Background

### Current Architecture

ml4t.backtest uses an event-driven architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                         Engine.run()                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  for timestamp, data, context in feed:                   │   │
│  │      broker._update_time(...)                            │   │
│  │      broker.evaluate_position_rules()  # Risk mgmt       │   │
│  │      broker._process_orders()          # Fill orders     │   │
│  │      strategy.on_data(...)             # USER LOGIC      │   │
│  │      broker._process_orders()          # Fill new orders │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Current Flow**:
1. `Strategy.on_data()` receives price data and signals
2. Strategy decides what to trade and calls `broker.submit_order()`
3. Broker executes orders with commission/slippage

**The Gap**: No built-in support for:
- Converting alpha signals → target weights
- Target weights → orders (rebalancing)
- Integration with external optimizers

### Use Cases

1. **Mean-Variance Optimization**: Use riskfolio-lib to compute optimal weights from expected returns and covariance
2. **Risk Parity**: Equal risk contribution across assets
3. **Hierarchical Risk Parity**: Tree-based diversification
4. **Black-Litterman**: Combine market equilibrium with views
5. **Factor-based allocation**: Target factor exposures
6. **Custom convex optimization**: User-defined objectives via cvxpy

---

## Design Options

### Option A: Strategy-Level (Minimal Change)

**Philosophy**: User handles everything in their strategy; library provides no special support.

```python
import riskfolio as rp
import pandas as pd

class OptimizedStrategy(Strategy):
    def __init__(self, lookback: int = 60, rebalance_freq: int = 20):
        self.lookback = lookback
        self.rebalance_freq = rebalance_freq
        self.returns_buffer: dict[str, list[float]] = {}
        self.bar_count = 0

    def on_data(self, timestamp, data, context, broker):
        # 1. Collect returns
        for asset, bars in data.items():
            if asset not in self.returns_buffer:
                self.returns_buffer[asset] = []
            if 'close' in bars:
                self.returns_buffer[asset].append(bars['close'])

        self.bar_count += 1

        # Only rebalance every N bars
        if self.bar_count % self.rebalance_freq != 0:
            return

        if len(next(iter(self.returns_buffer.values()))) < self.lookback:
            return

        # 2. Build returns DataFrame
        returns_df = self._compute_returns()

        # 3. Run optimizer (riskfolio-lib)
        port = rp.Portfolio(returns=returns_df)
        port.assets_stats(method_mu='hist', method_cov='hist')
        weights = port.optimization(model='Classic', rm='MV', obj='Sharpe')

        # 4. Convert to orders
        self._rebalance(weights, data, broker)

    def _compute_returns(self) -> pd.DataFrame:
        prices = pd.DataFrame(self.returns_buffer)
        return prices.pct_change().dropna().tail(self.lookback)

    def _rebalance(self, weights, data, broker):
        equity = broker.get_account_value()
        target_weights = weights['weights'].to_dict()

        for asset, weight in target_weights.items():
            price = data.get(asset, {}).get('close')
            if not price:
                continue

            target_value = equity * weight
            current_pos = broker.get_position(asset)
            current_value = (current_pos.quantity * price) if current_pos else 0

            delta_value = target_value - current_value
            delta_shares = delta_value / price

            if abs(delta_shares) > 0.5:  # Min trade threshold
                side = OrderSide.BUY if delta_shares > 0 else OrderSide.SELL
                broker.submit_order(asset, abs(delta_shares), side)
```

**Pros**:
- Zero library changes
- Maximum flexibility
- Users can integrate ANY optimizer
- No new abstractions to learn

**Cons**:
- Boilerplate for common patterns
- Easy to get rebalancing wrong
- No standardization

---

### Option B: TargetWeightExecutor Utility

**Philosophy**: Provide a utility class for the common weight→order conversion.

```python
# NEW: src/ml4t/backtest/execution/rebalancer.py

from dataclasses import dataclass
from typing import Protocol

from ..broker import Broker
from ..types import Order, OrderSide


class WeightProvider(Protocol):
    """Protocol for anything that produces target weights."""
    def get_weights(self, data: dict, broker: Broker) -> dict[str, float]:
        """Return target weights (asset -> weight, should sum to <= 1.0)."""
        ...


@dataclass
class RebalanceConfig:
    """Configuration for rebalancing behavior."""
    min_trade_value: float = 100.0      # Skip trades smaller than this
    min_weight_change: float = 0.01     # Skip if weight change < 1%
    round_lots: bool = False            # Round to 100-share lots
    lot_size: int = 100                 # Lot size for rounding
    allow_short: bool = False           # Allow short positions
    max_single_weight: float = 1.0      # Max weight per asset


class TargetWeightExecutor:
    """Convert target portfolio weights to orders.

    Handles the common pattern of rebalancing to target weights:
    - Computes required trades from current vs target positions
    - Applies minimum trade thresholds
    - Handles lot rounding (optional)
    - Respects position limits

    Example:
        executor = TargetWeightExecutor(config=RebalanceConfig(min_trade_value=500))

        # In strategy:
        target_weights = {'AAPL': 0.3, 'GOOG': 0.3, 'MSFT': 0.4}
        orders = executor.execute(target_weights, data, broker)
    """

    def __init__(self, config: RebalanceConfig | None = None):
        self.config = config or RebalanceConfig()

    def execute(
        self,
        target_weights: dict[str, float],
        data: dict[str, dict],
        broker: Broker,
    ) -> list[Order]:
        """Execute rebalancing to target weights.

        Args:
            target_weights: Dict of asset -> target weight (0.0 to 1.0)
            data: Current bar data (for prices)
            broker: Broker instance for order submission

        Returns:
            List of submitted orders
        """
        equity = broker.get_account_value()
        orders = []

        # Compute current weights
        current_weights = self._get_current_weights(broker, data)

        # Determine trades needed
        for asset, target_wt in target_weights.items():
            # Validate weight
            target_wt = min(target_wt, self.config.max_single_weight)
            if target_wt < 0 and not self.config.allow_short:
                target_wt = 0

            current_wt = current_weights.get(asset, 0.0)
            weight_delta = target_wt - current_wt

            # Skip small weight changes
            if abs(weight_delta) < self.config.min_weight_change:
                continue

            # Get price
            price = data.get(asset, {}).get('close')
            if not price or price <= 0:
                continue

            # Compute trade
            target_value = equity * target_wt
            current_value = equity * current_wt
            delta_value = target_value - current_value

            # Skip small trades
            if abs(delta_value) < self.config.min_trade_value:
                continue

            # Compute shares
            shares = delta_value / price

            # Round to lots if configured
            if self.config.round_lots:
                shares = round(shares / self.config.lot_size) * self.config.lot_size

            if shares == 0:
                continue

            # Submit order
            side = OrderSide.BUY if shares > 0 else OrderSide.SELL
            order = broker.submit_order(asset, abs(shares), side)
            orders.append(order)

        # Close positions not in target (if any)
        for asset in current_weights:
            if asset not in target_weights:
                pos = broker.get_position(asset)
                if pos and pos.quantity != 0:
                    order = broker.close_position(asset)
                    if order:
                        orders.append(order)

        return orders

    def _get_current_weights(
        self,
        broker: Broker,
        data: dict[str, dict]
    ) -> dict[str, float]:
        """Get current portfolio weights."""
        equity = broker.get_account_value()
        if equity <= 0:
            return {}

        weights = {}
        for asset, pos in broker.positions.items():
            price = data.get(asset, {}).get('close', pos.entry_price)
            value = pos.quantity * price
            weights[asset] = value / equity

        return weights

    def preview(
        self,
        target_weights: dict[str, float],
        data: dict[str, dict],
        broker: Broker,
    ) -> list[dict]:
        """Preview trades without executing.

        Returns:
            List of trade previews with asset, current_weight, target_weight,
            shares, and value.
        """
        equity = broker.get_account_value()
        current_weights = self._get_current_weights(broker, data)
        previews = []

        for asset, target_wt in target_weights.items():
            current_wt = current_weights.get(asset, 0.0)
            price = data.get(asset, {}).get('close', 0)

            if price > 0:
                delta_value = equity * (target_wt - current_wt)
                shares = delta_value / price

                previews.append({
                    'asset': asset,
                    'current_weight': current_wt,
                    'target_weight': target_wt,
                    'shares': shares,
                    'value': delta_value,
                })

        return previews
```

**Usage with riskfolio-lib**:

```python
from ml4t.backtest.execution.rebalancer import TargetWeightExecutor, RebalanceConfig

class RiskfolioStrategy(Strategy):
    def __init__(self):
        self.executor = TargetWeightExecutor(
            config=RebalanceConfig(min_trade_value=500, round_lots=True)
        )
        self.returns_buffer = {}

    def on_data(self, timestamp, data, context, broker):
        # ... collect returns ...

        # Get weights from riskfolio-lib
        weights = self._optimize(returns_df)

        # Execute rebalancing (one line!)
        orders = self.executor.execute(weights, data, broker)
```

**Pros**:
- Solves the common weight→order problem correctly
- Preview mode for debugging
- Configurable thresholds
- Still allows any optimizer
- Small, focused addition (~100 lines)

**Cons**:
- Another class to learn
- Doesn't help with optimization itself

---

### Option C: PortfolioStrategy Base Class

**Philosophy**: Provide a specialized base class with optimization hooks.

```python
# NEW: src/ml4t/backtest/strategy/portfolio.py

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from ..strategy import Strategy
from ..execution.rebalancer import TargetWeightExecutor, RebalanceConfig


@dataclass
class PortfolioStrategyConfig:
    """Configuration for portfolio strategies."""
    lookback: int = 60                     # Bars of history for optimization
    rebalance_frequency: int = 20          # Bars between rebalances
    warmup_bars: int = 60                  # Bars before first optimization
    rebalance_config: RebalanceConfig = field(default_factory=RebalanceConfig)


class PortfolioStrategy(Strategy):
    """Base class for portfolio optimization strategies.

    Provides infrastructure for:
    - Collecting historical returns
    - Periodic rebalancing
    - Weight → order execution

    Subclasses implement:
    - compute_alpha(): Generate alpha signals (optional)
    - optimize(): Return target weights given returns history

    Example:
        class MeanVarianceStrategy(PortfolioStrategy):
            def optimize(self, returns: pd.DataFrame, data: dict) -> dict[str, float]:
                import riskfolio as rp
                port = rp.Portfolio(returns=returns)
                port.assets_stats(method_mu='hist', method_cov='hist')
                weights = port.optimization(model='Classic', rm='MV', obj='Sharpe')
                return weights['weights'].to_dict()
    """

    def __init__(self, config: PortfolioStrategyConfig | None = None):
        self.config = config or PortfolioStrategyConfig()
        self.executor = TargetWeightExecutor(self.config.rebalance_config)
        self._prices: dict[str, list[float]] = {}
        self._timestamps: list[datetime] = []
        self._bar_count = 0

    def on_data(self, timestamp, data, context, broker):
        # Collect prices
        self._collect_prices(timestamp, data)
        self._bar_count += 1

        # Check if we should rebalance
        if not self._should_rebalance():
            return

        # Build returns DataFrame
        returns = self._compute_returns()
        if returns is None or len(returns) < self.config.lookback:
            return

        # Call user-defined optimization
        target_weights = self.optimize(returns, data, context, broker)

        if target_weights:
            self.executor.execute(target_weights, data, broker)

    @abstractmethod
    def optimize(
        self,
        returns: pd.DataFrame,
        data: dict,
        context: dict,
        broker,
    ) -> dict[str, float] | None:
        """Return target portfolio weights.

        Args:
            returns: Historical returns DataFrame (assets as columns)
            data: Current bar data
            context: Context data from feed
            broker: Broker instance (for current positions, etc.)

        Returns:
            Dict of asset -> weight (0.0 to 1.0), or None to skip rebalance
        """
        pass

    def _should_rebalance(self) -> bool:
        if self._bar_count < self.config.warmup_bars:
            return False
        return self._bar_count % self.config.rebalance_frequency == 0

    def _collect_prices(self, timestamp: datetime, data: dict):
        self._timestamps.append(timestamp)
        for asset, bars in data.items():
            if 'close' in bars:
                if asset not in self._prices:
                    self._prices[asset] = []
                self._prices[asset].append(bars['close'])

    def _compute_returns(self) -> pd.DataFrame | None:
        if not self._prices:
            return None

        # Build price DataFrame
        prices = pd.DataFrame(self._prices)

        # Compute returns
        returns = prices.pct_change().dropna()

        return returns.tail(self.config.lookback)
```

**Usage**:

```python
import riskfolio as rp

class RiskParityStrategy(PortfolioStrategy):
    """Risk parity using riskfolio-lib."""

    def optimize(self, returns, data, context, broker):
        port = rp.Portfolio(returns=returns)
        port.assets_stats(method_mu='hist', method_cov='hist')
        weights = port.rp_optimization(
            model='Classic',
            rm='MV',
            rf=0,
            b=None,
            hist=True
        )
        return weights['weights'].to_dict()


class HRPStrategy(PortfolioStrategy):
    """Hierarchical Risk Parity."""

    def optimize(self, returns, data, context, broker):
        port = rp.HCPortfolio(returns=returns)
        weights = port.optimization(
            model='HRP',
            rm='MV',
            rf=0,
            linkage='ward',
        )
        return weights['weights'].to_dict()
```

**Pros**:
- Clean abstraction for portfolio strategies
- Handles returns collection automatically
- Built-in rebalancing logic
- Easy to implement different optimizers

**Cons**:
- More complex abstraction
- May not fit all use cases
- Opinionated about returns computation
- Requires pandas dependency in core

---

## Recommendation

**Implement Option B (TargetWeightExecutor) only.**

Rationale:
1. **Minimal footprint**: ~100 lines, one new file
2. **Solves the right problem**: Weight→order is the tricky part
3. **Stays flexible**: Works with ANY optimizer
4. **No new dependencies**: Pure Python
5. **Doesn't constrain users**: Can still do Option A or build Option C themselves

Option C is appealing but:
- Introduces opinions about returns computation
- May not fit factor-based or signal-based strategies
- Can be built on top of Option B by users

---

## Questions for Reviewers

1. **Is TargetWeightExecutor the right abstraction?** Should it be a function instead of a class?

2. **Should we handle partial fills?** If rebalancing generates orders that partially fill, should the executor track and retry?

3. **Transaction cost awareness**: Should the executor estimate transaction costs before deciding to trade?

4. **Constraints integration**: Should the executor support passing constraints to optimizers (e.g., sector limits)?

5. **What about live trading?** Does this design translate to live trading with real brokers?

6. **Alternative approaches we're missing?**
   - Signal-based sizing (Kelly criterion)
   - Factor-targeted portfolios
   - Multi-period optimization

---

## Appendix: Relevant Source Code

### A. Strategy Base Class

```python
# src/ml4t/backtest/strategy.py

"""Base strategy class for backtesting."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any


class Strategy(ABC):
    """Base strategy class."""

    @abstractmethod
    def on_data(
        self,
        timestamp: datetime,
        data: dict[str, dict],
        context: dict[str, Any],
        broker: Any,  # Avoid circular import, use Any for broker type
    ) -> None:
        """Called for each timestamp with all available data."""
        pass

    def on_start(self, broker: Any) -> None:  # noqa: B027
        """Called before backtest starts."""
        pass

    def on_end(self, broker: Any) -> None:  # noqa: B027
        """Called after backtest ends."""
        pass
```

### B. Engine Event Loop

```python
# src/ml4t/backtest/engine.py (core run method)

def run(self) -> dict:
    """Run backtest and return results."""
    self.strategy.on_start(self.broker)

    for timestamp, assets_data, context in self.feed:
        prices = {a: d["close"] for a, d in assets_data.items() if d.get("close")}
        opens = {a: d.get("open", d.get("close")) for a, d in assets_data.items()}
        highs = {a: d.get("high", d.get("close")) for a, d in assets_data.items()}
        lows = {a: d.get("low", d.get("close")) for a, d in assets_data.items()}
        volumes = {a: d.get("volume", 0) for a, d in assets_data.items()}
        signals = {a: d.get("signals", {}) for a, d in assets_data.items()}

        self.broker._update_time(timestamp, prices, opens, highs, lows, volumes, signals)

        # Process pending exits from NEXT_BAR_OPEN mode (fills at open)
        self.broker._process_pending_exits()

        # Evaluate position rules (stops, trails, etc.) - generates exit orders
        self.broker.evaluate_position_rules()

        if self.execution_mode == ExecutionMode.NEXT_BAR:
            # Next-bar mode: process pending orders at open price
            self.broker._process_orders(use_open=True)
            # Strategy generates new orders
            self.strategy.on_data(timestamp, assets_data, context, self.broker)
            # New orders will be processed next bar
        else:
            # Same-bar mode: process before and after strategy
            self.broker._process_orders()
            self.strategy.on_data(timestamp, assets_data, context, self.broker)
            self.broker._process_orders()

        self.equity_curve.append((timestamp, self.broker.get_account_value()))

    self.strategy.on_end(self.broker)
    return self._generate_results()
```

### C. Broker Order Submission

```python
# src/ml4t/backtest/broker.py (relevant methods)

class Broker:
    """Broker interface - same for backtest and live trading."""

    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission_model: CommissionModel | None = None,
        slippage_model: SlippageModel | None = None,
        execution_mode: ExecutionMode = ExecutionMode.SAME_BAR,
        # ... other params ...
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, Position] = {}
        self.orders: list[Order] = []
        self.pending_orders: list[Order] = []
        self.fills: list[Fill] = []
        self.trades: list[Trade] = []
        # ...

    def get_position(self, asset: str) -> Position | None:
        return self.positions.get(asset)

    def get_cash(self) -> float:
        return self.cash

    def get_account_value(self) -> float:
        """Calculate total account value (cash + position values)."""
        value = self.cash
        for asset, pos in self.positions.items():
            price = self._current_prices.get(asset, pos.entry_price)
            multiplier = self.get_multiplier(asset)
            value += pos.quantity * price * multiplier
        return value

    def submit_order(
        self,
        asset: str,
        quantity: float,
        side: OrderSide | None = None,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        trail_amount: float | None = None,
    ) -> Order:
        if side is None:
            side = OrderSide.BUY if quantity > 0 else OrderSide.SELL
            quantity = abs(quantity)

        self._order_counter += 1
        order = Order(
            asset=asset,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            trail_amount=trail_amount,
            order_id=f"ORD-{self._order_counter}",
            created_at=self._current_time,
        )

        self.orders.append(order)
        self.pending_orders.append(order)
        return order

    def close_position(self, asset: str) -> Order | None:
        pos = self.positions.get(asset)
        if pos and pos.quantity != 0:
            side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY
            return self.submit_order(asset, abs(pos.quantity), side)
        return None
```

### D. Core Types

```python
# src/ml4t/backtest/types.py (relevant excerpts)

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


@dataclass
class Order:
    asset: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    trail_amount: float | None = None
    parent_id: str | None = None
    order_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime | None = None
    filled_at: datetime | None = None
    filled_price: float | None = None
    filled_quantity: float = 0.0


@dataclass
class Position:
    """Unified position tracking for strategy and accounting."""
    asset: str
    quantity: float  # Positive for long, negative for short
    entry_price: float  # Weighted average cost basis
    entry_time: datetime
    current_price: float | None = None  # Mark-to-market price
    bars_held: int = 0
    # Risk tracking fields
    high_water_mark: float | None = None
    low_water_mark: float | None = None
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    initial_quantity: float | None = None
    context: dict = field(default_factory=dict)
    multiplier: float = 1.0  # Contract multiplier (for futures)

    @property
    def market_value(self) -> float:
        """Current market value of position."""
        price = self.current_price or self.entry_price
        return self.quantity * price * self.multiplier

    def unrealized_pnl(self, price: float | None = None) -> float:
        """Calculate unrealized P&L at given or current price."""
        p = price or self.current_price or self.entry_price
        return (p - self.entry_price) * self.quantity * self.multiplier
```

---

## Changelog

- 2025-11-23: Initial draft
