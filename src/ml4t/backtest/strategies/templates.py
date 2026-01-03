"""Strategy templates for common trading patterns.

These templates provide a starting point for implementing trading strategies.
Override the required methods to customize behavior.
"""

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from datetime import datetime
from statistics import mean, stdev
from typing import TYPE_CHECKING, Any

from ..strategy import Strategy

if TYPE_CHECKING:
    from ..broker import Broker


class SignalFollowingStrategy(Strategy):
    """Template for strategies that follow pre-computed signals.

    Use this when you have ML predictions, technical indicators, or any
    pre-computed signal column in your DataFrame.

    Class Attributes:
        signal_column: Name of the signal column in data (default: "signal")
        position_size: Fraction of equity per position (default: 0.10)
        allow_shorts: Whether to allow short positions (default: False)

    Example:
        >>> class MyMLStrategy(SignalFollowingStrategy):
        ...     signal_column = "rf_prediction"
        ...     position_size = 0.05
        ...
        ...     def should_enter_long(self, signal):
        ...         return signal > 0.7
        ...
        ...     def should_exit(self, signal):
        ...         return signal < 0.3
    """

    signal_column: str = "signal"
    position_size: float = 0.10
    allow_shorts: bool = False

    @abstractmethod
    def should_enter_long(self, signal: float) -> bool:
        """Return True to open a long position.

        Args:
            signal: Current signal value for the asset

        Returns:
            True if should enter long position
        """

    @abstractmethod
    def should_exit(self, signal: float) -> bool:
        """Return True to close current position.

        Args:
            signal: Current signal value for the asset

        Returns:
            True if should exit position
        """

    def should_enter_short(self, signal: float) -> bool:
        """Return True to open a short position.

        Override this method for short strategies. Default returns False.

        Args:
            signal: Current signal value for the asset

        Returns:
            True if should enter short position
        """
        return False

    def on_data(
        self,
        timestamp: datetime,
        data: dict[str, dict],
        context: dict[str, Any],
        broker: Broker,
    ) -> None:
        """Process each bar and generate orders based on signals."""
        for asset, bar in data.items():
            # Signals are nested under 'signals' dict in DataFeed output
            signals = bar.get("signals", {})
            signal = signals.get(self.signal_column, 0) if signals else 0
            if signal is None:
                signal = 0

            position = broker.get_position(asset)
            price = bar.get("close", 0)

            if position is None:
                # No position - check for entry
                if self.should_enter_long(signal):
                    equity = broker.get_account_value()
                    shares = int((equity * self.position_size) / price) if price > 0 else 0
                    if shares > 0:
                        broker.submit_order(asset, shares)
                elif self.allow_shorts and self.should_enter_short(signal):
                    equity = broker.get_account_value()
                    shares = int((equity * self.position_size) / price) if price > 0 else 0
                    if shares > 0:
                        broker.submit_order(asset, -shares)
            else:
                # Have position - check for exit
                if self.should_exit(signal):
                    broker.close_position(asset)


class MomentumStrategy(Strategy):
    """Template for momentum/trend-following strategies.

    Enters long when asset has positive momentum over lookback period,
    exits when momentum turns negative.

    Class Attributes:
        lookback: Number of bars for momentum calculation (default: 20)
        entry_threshold: Minimum return to enter (default: 0.05 = 5%)
        exit_threshold: Return level to exit (default: -0.02 = -2%)
        position_size: Fraction of equity per position (default: 0.10)

    Example:
        >>> class MyMomentum(MomentumStrategy):
        ...     lookback = 60  # 60-day momentum
        ...     entry_threshold = 0.10  # Enter on 10% gain
        ...     exit_threshold = 0.0  # Exit when momentum turns negative
    """

    lookback: int = 20
    entry_threshold: float = 0.05
    exit_threshold: float = -0.02
    position_size: float = 0.10

    def __init__(self) -> None:
        self.price_history: dict[str, list[float]] = defaultdict(list)

    def calculate_momentum(self, prices: list[float]) -> float:
        """Calculate momentum as return over lookback period.

        Args:
            prices: List of prices (most recent last)

        Returns:
            Return from first to last price
        """
        if len(prices) < 2 or prices[0] == 0:
            return 0.0
        return (prices[-1] / prices[0]) - 1

    def on_data(
        self,
        timestamp: datetime,
        data: dict[str, dict],
        context: dict[str, Any],
        broker: Broker,
    ) -> None:
        """Process each bar and trade based on momentum."""
        for asset, bar in data.items():
            close = bar.get("close")
            if close is None or close <= 0:
                continue

            # Track price history
            self.price_history[asset].append(close)

            # Wait for enough history
            if len(self.price_history[asset]) < self.lookback:
                continue

            # Keep only lookback period
            self.price_history[asset] = self.price_history[asset][-self.lookback :]

            # Calculate momentum
            momentum = self.calculate_momentum(self.price_history[asset])
            position = broker.get_position(asset)

            if position is None and momentum > self.entry_threshold:
                # Enter long on strong momentum
                equity = broker.get_account_value()
                shares = int((equity * self.position_size) / close)
                if shares > 0:
                    broker.submit_order(asset, shares)
            elif position is not None and momentum < self.exit_threshold:
                # Exit on weak momentum
                broker.close_position(asset)


class MeanReversionStrategy(Strategy):
    """Template for mean-reversion strategies.

    Buys when price is below moving average by a threshold,
    sells when price reverts to the mean.

    Class Attributes:
        lookback: Number of bars for mean calculation (default: 20)
        entry_zscore: Z-score threshold to enter (default: -2.0)
        exit_zscore: Z-score threshold to exit (default: 0.0)
        position_size: Fraction of equity per position (default: 0.10)

    Example:
        >>> class MyMeanReversion(MeanReversionStrategy):
        ...     lookback = 30
        ...     entry_zscore = -2.5  # More extreme entry
        ...     exit_zscore = 0.5   # Take profit above mean
    """

    lookback: int = 20
    entry_zscore: float = -2.0
    exit_zscore: float = 0.0
    position_size: float = 0.10

    def __init__(self) -> None:
        self.price_history: dict[str, list[float]] = defaultdict(list)

    def calculate_zscore(self, prices: list[float], current: float) -> float | None:
        """Calculate z-score of current price vs historical distribution.

        Args:
            prices: Historical prices
            current: Current price

        Returns:
            Z-score or None if insufficient data
        """
        if len(prices) < 2:
            return None

        try:
            avg = mean(prices)
            std = stdev(prices)
            if std == 0:
                return None
            return (current - avg) / std
        except Exception:
            return None

    def on_data(
        self,
        timestamp: datetime,
        data: dict[str, dict],
        context: dict[str, Any],
        broker: Broker,
    ) -> None:
        """Process each bar and trade based on mean reversion."""
        for asset, bar in data.items():
            close = bar.get("close")
            if close is None or close <= 0:
                continue

            # Track price history
            self.price_history[asset].append(close)

            # Wait for enough history
            if len(self.price_history[asset]) < self.lookback:
                continue

            # Keep only lookback period
            prices = self.price_history[asset][-self.lookback :]
            self.price_history[asset] = prices

            # Calculate z-score
            zscore = self.calculate_zscore(prices[:-1], close)
            if zscore is None:
                continue

            position = broker.get_position(asset)

            if position is None and zscore < self.entry_zscore:
                # Enter long on oversold condition
                equity = broker.get_account_value()
                shares = int((equity * self.position_size) / close)
                if shares > 0:
                    broker.submit_order(asset, shares)
            elif position is not None and zscore > self.exit_zscore:
                # Exit on mean reversion
                broker.close_position(asset)


class LongShortStrategy(Strategy):
    """Template for long/short equity strategies.

    Ranks assets by a signal and goes long top N, short bottom N.

    Class Attributes:
        signal_column: Column to rank assets by (default: "signal")
        long_count: Number of assets to go long (default: 5)
        short_count: Number of assets to go short (default: 5)
        position_size: Fraction of equity per position (default: 0.05)
        rebalance_frequency: Bars between rebalancing (default: 20)

    Example:
        >>> class MyLongShort(LongShortStrategy):
        ...     signal_column = "momentum_score"
        ...     long_count = 10
        ...     short_count = 10
        ...     rebalance_frequency = 21  # Monthly
    """

    signal_column: str = "signal"
    long_count: int = 5
    short_count: int = 5
    position_size: float = 0.05
    rebalance_frequency: int = 20

    def __init__(self) -> None:
        self.bar_count = 0

    def rank_assets(self, data: dict[str, dict]) -> tuple[list[str], list[str]]:
        """Rank assets by signal and return long/short lists.

        Args:
            data: Current bar data for all assets

        Returns:
            Tuple of (long_assets, short_assets)
        """
        # Collect signals (signals are nested under 'signals' dict)
        signals: list[tuple[str, float]] = []
        for asset, bar in data.items():
            bar_signals = bar.get("signals", {})
            signal = bar_signals.get(self.signal_column) if bar_signals else None
            if signal is not None:
                signals.append((asset, signal))

        if not signals:
            return [], []

        # Sort by signal (high to low)
        signals.sort(key=lambda x: x[1], reverse=True)

        # Top N for long, bottom N for short
        long_assets = [s[0] for s in signals[: self.long_count]]
        short_assets = [s[0] for s in signals[-self.short_count :]]

        # Don't short the same assets we're going long
        short_assets = [a for a in short_assets if a not in long_assets]

        return long_assets, short_assets

    def on_data(
        self,
        timestamp: datetime,
        data: dict[str, dict],
        context: dict[str, Any],
        broker: Broker,
    ) -> None:
        """Rebalance portfolio periodically based on rankings."""
        self.bar_count += 1

        # Only rebalance on schedule
        if self.bar_count % self.rebalance_frequency != 1:
            return

        # Get current rankings
        long_assets, short_assets = self.rank_assets(data)
        target_assets = set(long_assets + short_assets)

        # Close positions not in target
        for asset in list(broker.get_positions().keys()):
            if asset not in target_assets:
                broker.close_position(asset)

        # Open/adjust positions
        equity = broker.get_account_value()

        for asset in long_assets:
            price = data.get(asset, {}).get("close", 0)
            if price <= 0:
                continue

            position = broker.get_position(asset)
            target_shares = int((equity * self.position_size) / price)

            if position is None and target_shares > 0:
                broker.submit_order(asset, target_shares)

        for asset in short_assets:
            price = data.get(asset, {}).get("close", 0)
            if price <= 0:
                continue

            position = broker.get_position(asset)
            target_shares = int((equity * self.position_size) / price)

            if position is None and target_shares > 0:
                broker.submit_order(asset, -target_shares)
