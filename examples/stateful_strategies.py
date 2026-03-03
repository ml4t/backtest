"""Stateful strategy examples demonstrating why event-driven backtesting matters.

Each strategy here maintains state across bars — trading decisions feed back
into future decisions. This is fundamentally impossible in vectorized frameworks
like VectorBT, where all signals must be computed in advance.

Five reasons you need event-driven backtesting:

1. **Feedback loops** — position size depends on realized P&L (AdaptiveKellySizing)
2. **Conditional chains** — entry N depends on P&L of entries 1..N-1 (Pyramiding)
3. **Cross-asset coordination** — two legs managed as one position (PairsTrading)
4. **Path-dependent state** — equity curve drives future sizing (DrawdownCircuitBreaker)
5. **Reactive order management** — each fill triggers new orders (GridTrading)

These are NOT part of the public API. They are importable demonstrations with
full test coverage in test_stateful_strategies.py.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

from ml4t.backtest.strategy import Strategy
from ml4t.backtest.types import OrderType

if TYPE_CHECKING:
    from ml4t.backtest.broker import Broker


# ---------------------------------------------------------------------------
# 1. Adaptive Kelly Sizing — feedback loop
# ---------------------------------------------------------------------------


class AdaptiveKellySizingStrategy(Strategy):
    """Position size adapts based on realized win rate and payoff ratio.

    The feedback loop: position_size → P&L → Kelly_fraction → next_position_size.
    In a vectorized framework, you cannot compute the Kelly fraction because it
    depends on future fills that depend on the fraction itself.

    Kelly formula: f* = W - (1 - W) / R
        where W = win rate, R = avg_win / avg_loss
    We use half-Kelly (kelly_fraction=0.5) for safety.
    """

    def __init__(
        self,
        signal_column: str = "signal",
        entry_threshold: float = 0.5,
        exit_threshold: float = -0.5,
        base_size: float = 0.10,
        min_size: float = 0.02,
        max_size: float = 0.25,
        kelly_fraction: float = 0.5,
        min_trades: int = 5,
    ):
        self.signal_column = signal_column
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.base_size = base_size
        self.min_size = min_size
        self.max_size = max_size
        self.kelly_fraction = kelly_fraction
        self.min_trades = min_trades
        # Track the size used for each entry (for test verification)
        self.size_history: list[float] = []

    def _kelly_size(self, broker: Broker, asset: str) -> float:
        """Compute position size as fraction of equity using Kelly criterion."""
        stats = broker.get_asset_stats(asset)
        if stats.total_trades < self.min_trades:
            return self.base_size

        w = stats.recent_win_rate
        # Compute average win / average loss ratio from recent PnLs
        wins = [p for p in stats.recent_pnls if p > 0]
        losses = [p for p in stats.recent_pnls if p <= 0]
        if not wins or not losses:
            return self.base_size

        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))
        if avg_loss == 0:
            return self.max_size

        r = avg_win / avg_loss
        f_star = w - (1 - w) / r  # Kelly fraction
        f_star = max(0.0, f_star) * self.kelly_fraction  # half-Kelly
        size = max(self.min_size, min(self.max_size, f_star))
        return size

    def on_data(
        self,
        timestamp: datetime,
        data: dict[str, dict],
        context: dict[str, Any],
        broker: Broker,
    ) -> None:
        for asset, bar in data.items():
            signals = bar.get("signals", {})
            signal = signals.get(self.signal_column, 0) if signals else 0
            if signal is None:
                signal = 0

            price = bar.get("close", 0)
            if price <= 0:
                continue

            position = broker.get_position(asset)

            if position is None and signal > self.entry_threshold:
                size_frac = self._kelly_size(broker, asset)
                self.size_history.append(size_frac)
                equity = broker.get_account_value()
                shares = (equity * size_frac) / price
                if shares > 0:
                    broker.submit_order(asset, shares)
            elif position is not None and signal < self.exit_threshold:
                broker.close_position(asset)


# ---------------------------------------------------------------------------
# 2. Pyramiding — conditional chains
# ---------------------------------------------------------------------------


class PyramidingStrategy(Strategy):
    """Add to winners: each pyramid level triggers when unrealized P&L hits a threshold.

    The conditional chain: entry_1 → profit check → entry_2 → profit check → entry_3.
    Each entry depends on the P&L of all prior entries, which depends on their fill
    prices, which are only known at execution time.
    """

    def __init__(
        self,
        signal_column: str = "signal",
        entry_threshold: float = 0.5,
        exit_threshold: float = -0.5,
        max_levels: int = 3,
        profit_threshold: float = 0.02,
        base_size: float = 0.10,
        size_decay: float = 0.5,
        adverse_threshold: float = 0.03,
    ):
        self.signal_column = signal_column
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.max_levels = max_levels
        self.profit_threshold = profit_threshold
        self.base_size = base_size
        self.size_decay = size_decay
        self.adverse_threshold = adverse_threshold
        # State: current pyramid level per asset
        self.pyramid_levels: dict[str, int] = defaultdict(int)
        # Track entry prices per level for the adverse check
        self.level_entries: dict[str, list[float]] = defaultdict(list)

    def on_data(
        self,
        timestamp: datetime,
        data: dict[str, dict],
        context: dict[str, Any],
        broker: Broker,
    ) -> None:
        for asset, bar in data.items():
            signals = bar.get("signals", {})
            signal = signals.get(self.signal_column, 0) if signals else 0
            if signal is None:
                signal = 0

            price = bar.get("close", 0)
            if price <= 0:
                continue

            position = broker.get_position(asset)

            if position is None:
                # No position — enter on signal
                if signal > self.entry_threshold:
                    equity = broker.get_account_value()
                    shares = (equity * self.base_size) / price
                    if shares > 0:
                        broker.submit_order(asset, shares)
                        self.pyramid_levels[asset] = 1
                        self.level_entries[asset] = [price]
                continue

            # Have a position — check exit first
            if signal < self.exit_threshold:
                broker.close_position(asset)
                self.pyramid_levels[asset] = 0
                self.level_entries[asset] = []
                continue

            level = self.pyramid_levels[asset]
            pnl_pct = position.pnl_percent()

            # Scale out on adverse move
            if (
                position.high_water_mark is not None
                and price < position.high_water_mark * (1 - self.adverse_threshold)
                and level > 1
            ):
                broker.reduce_position(asset, 0.5)
                self.pyramid_levels[asset] = max(1, level - 1)
                continue

            # Pyramid up on profit
            if level < self.max_levels and pnl_pct > self.profit_threshold * level:
                decay = self.size_decay**level
                equity = broker.get_account_value()
                shares = (equity * self.base_size * decay) / price
                if shares > 0:
                    broker.submit_order(asset, shares)
                    self.pyramid_levels[asset] = level + 1
                    self.level_entries[asset].append(price)


# ---------------------------------------------------------------------------
# 3. Pairs Trading — cross-asset coordination
# ---------------------------------------------------------------------------


class PairsTradingStrategy(Strategy):
    """Trade the spread between two correlated assets.

    Cross-asset coordination: the entry/exit of asset A is conditioned on the
    price of asset B. You cannot vectorize this because position in A affects
    available capital for B, and fills in A affect the timing of orders for B.
    """

    def __init__(
        self,
        asset_a: str = "A",
        asset_b: str = "B",
        lookback: int = 20,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        position_size: float = 0.10,
    ):
        self.asset_a = asset_a
        self.asset_b = asset_b
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.position_size = position_size
        # State
        self.price_history_a: list[float] = []
        self.price_history_b: list[float] = []
        self.pair_status: str = "flat"  # "flat", "long_spread", "short_spread"
        self.entry_zscore_value: float = 0.0

    def _compute_zscore(self) -> float | None:
        """Compute z-score of the price ratio B/A."""
        if len(self.price_history_a) < self.lookback:
            return None
        ratios = [
            b / a
            for a, b in zip(
                self.price_history_a[-self.lookback :],
                self.price_history_b[-self.lookback :],
            )
            if a > 0
        ]
        if len(ratios) < 2:
            return None
        mean_r = sum(ratios) / len(ratios)
        var_r = sum((r - mean_r) ** 2 for r in ratios) / (len(ratios) - 1)
        std_r = var_r**0.5
        if std_r == 0:
            return None
        current_ratio = self.price_history_b[-1] / self.price_history_a[-1]
        return (current_ratio - mean_r) / std_r

    def on_data(
        self,
        timestamp: datetime,
        data: dict[str, dict],
        context: dict[str, Any],
        broker: Broker,
    ) -> None:
        bar_a = data.get(self.asset_a)
        bar_b = data.get(self.asset_b)
        if bar_a is None or bar_b is None:
            return

        price_a = bar_a.get("close", 0)
        price_b = bar_b.get("close", 0)
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
                # Spread too wide (B expensive relative to A) → short B, long A
                shares_a = (equity * self.position_size) / price_a
                shares_b = (equity * self.position_size) / price_b
                if shares_a > 0 and shares_b > 0:
                    broker.submit_order(self.asset_a, shares_a)
                    broker.submit_order(self.asset_b, -shares_b)
                    self.pair_status = "short_spread"
                    self.entry_zscore_value = z
            elif z < -self.entry_zscore:
                # Spread too narrow (A expensive relative to B) → long B, short A
                shares_a = (equity * self.position_size) / price_a
                shares_b = (equity * self.position_size) / price_b
                if shares_a > 0 and shares_b > 0:
                    broker.submit_order(self.asset_a, -shares_a)
                    broker.submit_order(self.asset_b, shares_b)
                    self.pair_status = "long_spread"
                    self.entry_zscore_value = z
        else:
            # Check for exit: z-score reverted toward zero
            if abs(z) < self.exit_zscore:
                broker.close_position(self.asset_a)
                broker.close_position(self.asset_b)
                self.pair_status = "flat"
                self.entry_zscore_value = 0.0


# ---------------------------------------------------------------------------
# 4. Drawdown Circuit Breaker — path-dependent state
# ---------------------------------------------------------------------------


class DrawdownCircuitBreakerStrategy(Strategy):
    """Reduce or halt trading when portfolio drawdown exceeds thresholds.

    Path-dependent feedback: equity_curve → drawdown → sizing_multiplier →
    future_equity_curve. The sizing multiplier at bar N depends on the entire
    equity path from bar 0 to bar N-1, which depends on all prior sizing
    decisions. Impossible to vectorize.
    """

    def __init__(
        self,
        signal_column: str = "signal",
        entry_threshold: float = 0.5,
        exit_threshold: float = -0.5,
        base_size: float = 0.10,
        caution_threshold: float = 0.05,
        halt_threshold: float = 0.10,
        reduction_factor: float = 0.5,
        recovery_rate: float = 0.01,
    ):
        self.signal_column = signal_column
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.base_size = base_size
        self.caution_threshold = caution_threshold
        self.halt_threshold = halt_threshold
        self.reduction_factor = reduction_factor
        self.recovery_rate = recovery_rate
        # State
        self.peak_equity: float = 0.0
        self.sizing_multiplier: float = 1.0
        # Track for test verification
        self.multiplier_history: list[float] = []

    def on_data(
        self,
        timestamp: datetime,
        data: dict[str, dict],
        context: dict[str, Any],
        broker: Broker,
    ) -> None:
        equity = broker.get_account_value()

        # Update peak and compute drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
        drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0

        # Update sizing multiplier based on drawdown
        if drawdown < self.caution_threshold:
            # Below caution: recover toward 1.0
            self.sizing_multiplier = min(1.0, self.sizing_multiplier + self.recovery_rate)
        elif drawdown < self.halt_threshold:
            # Between caution and halt: linearly interpolate from reduction_factor to 0
            range_pct = (drawdown - self.caution_threshold) / (
                self.halt_threshold - self.caution_threshold
            )
            self.sizing_multiplier = self.reduction_factor * (1 - range_pct)
        else:
            # At or beyond halt: no new entries
            self.sizing_multiplier = 0.0

        self.multiplier_history.append(self.sizing_multiplier)

        for asset, bar in data.items():
            signals = bar.get("signals", {})
            signal = signals.get(self.signal_column, 0) if signals else 0
            if signal is None:
                signal = 0

            price = bar.get("close", 0)
            if price <= 0:
                continue

            position = broker.get_position(asset)

            if position is None and signal > self.entry_threshold:
                if self.sizing_multiplier <= 0:
                    continue  # Halted
                effective_size = self.base_size * self.sizing_multiplier
                shares = (equity * effective_size) / price
                if shares > 0:
                    broker.submit_order(asset, shares)
            elif position is not None and signal < self.exit_threshold:
                broker.close_position(asset)


# ---------------------------------------------------------------------------
# 5. Grid Trading — reactive order management
# ---------------------------------------------------------------------------


class GridTradingStrategy(Strategy):
    """Place limit orders on a grid; each fill triggers a new order at the adjacent level.

    Reactive order management: when a buy limit fills, place a sell limit one
    grid level above. When a sell limit fills, place a buy limit one grid level
    below. The grid state is fully dynamic and depends on the fill history.
    """

    def __init__(
        self,
        asset: str = "ASSET",
        grid_spacing: float = 0.01,
        num_levels: int = 5,
        order_size: float = 100,
        max_position: float = 500,
        recenter_threshold: float = 0.05,
    ):
        self.asset = asset
        self.grid_spacing = grid_spacing
        self.num_levels = num_levels
        self.order_size = order_size
        self.max_position = max_position
        self.recenter_threshold = recenter_threshold
        # State
        self.reference_price: float = 0.0
        self.grid_orders: dict[int, str] = {}  # level → order_id
        self.initialized: bool = False
        # Track for test verification
        self.fills_seen: int = 0

    def _place_grid(self, broker: Broker, price: float) -> None:
        """Place buy limits below and sell limits above reference price."""
        self.reference_price = price
        self.grid_orders.clear()

        for i in range(1, self.num_levels + 1):
            # Buy limits below
            buy_price = price * (1 - self.grid_spacing * i)
            order = broker.submit_order(
                self.asset,
                self.order_size,
                order_type=OrderType.LIMIT,
                limit_price=buy_price,
            )
            if order:
                self.grid_orders[-i] = order.order_id

            # Sell limits above
            sell_price = price * (1 + self.grid_spacing * i)
            order = broker.submit_order(
                self.asset,
                -self.order_size,
                order_type=OrderType.LIMIT,
                limit_price=sell_price,
            )
            if order:
                self.grid_orders[i] = order.order_id

    def _cancel_all(self, broker: Broker) -> None:
        """Cancel all outstanding grid orders."""
        for order_id in self.grid_orders.values():
            broker.cancel_order(order_id)
        self.grid_orders.clear()

    def on_data(
        self,
        timestamp: datetime,
        data: dict[str, dict],
        context: dict[str, Any],
        broker: Broker,
    ) -> None:
        bar = data.get(self.asset)
        if bar is None:
            return

        price = bar.get("close", 0)
        if price <= 0:
            return

        # Initialize grid on first bar
        if not self.initialized:
            self._place_grid(broker, price)
            self.initialized = True
            return

        # Check for filled orders and react
        filled_levels: list[int] = []
        for level, order_id in list(self.grid_orders.items()):
            order = broker.get_order(order_id)
            if order is not None and order.status.value == "filled":
                filled_levels.append(level)
                self.fills_seen += 1

        # React to fills
        position = broker.get_position(self.asset)
        current_qty = position.quantity if position else 0.0

        for level in filled_levels:
            del self.grid_orders[level]

            if level < 0:
                # Buy filled → place sell one level above (closer to reference)
                new_level = level + 1
                if new_level != 0 and new_level not in self.grid_orders:
                    sell_price = self.reference_price * (1 + self.grid_spacing * abs(new_level))
                    if abs(current_qty - self.order_size) <= self.max_position:
                        order = broker.submit_order(
                            self.asset,
                            -self.order_size,
                            order_type=OrderType.LIMIT,
                            limit_price=sell_price,
                        )
                        if order:
                            self.grid_orders[new_level] = order.order_id
            else:
                # Sell filled → place buy one level below (closer to reference)
                new_level = level - 1
                if new_level != 0 and new_level not in self.grid_orders:
                    buy_price = self.reference_price * (1 - self.grid_spacing * abs(new_level))
                    if abs(current_qty + self.order_size) <= self.max_position:
                        order = broker.submit_order(
                            self.asset,
                            self.order_size,
                            order_type=OrderType.LIMIT,
                            limit_price=buy_price,
                        )
                        if order:
                            self.grid_orders[new_level] = order.order_id

        # Recenter if price drifted too far
        if abs(price - self.reference_price) / self.reference_price > self.recenter_threshold:
            self._cancel_all(broker)
            self._place_grid(broker, price)
