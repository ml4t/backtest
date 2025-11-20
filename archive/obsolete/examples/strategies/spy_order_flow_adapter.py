"""
SPY Order Flow Strategy Adapter for ml4t.backtest

This module provides an external strategy and adapter for SPY order flow trading
using microstructure features to predict short-term movements.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from ml4t.backtest.core.types import AssetId
from ml4t.backtest.strategy.adapters import (
    DataFrameAdapter,
    ExternalStrategyInterface,
    PITData,
    StrategySignal,
)


@dataclass
class OrderFlowState:
    """State tracking for order flow strategy."""

    # Price and volume history
    prices: list[float] = field(default_factory=list)
    volumes: list[float] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)
    buy_volumes: list[float] = field(default_factory=list)
    sell_volumes: list[float] = field(default_factory=list)
    volume_imbalances: list[float] = field(default_factory=list)

    # Derived features
    price_momentum_5: float = 0.0
    price_momentum_20: float = 0.0
    volume_momentum_5: float = 0.0
    price_mean_20: float = 0.0
    price_std_10: float = 0.0
    imbalance_ratio: float = 0.5

    # Position tracking
    current_position: float = 0.0
    signal_count: int = 0
    last_signal_time: datetime | None = None

    # Statistical features
    imbalance_mean: float = 0.0
    imbalance_std: float = 0.0


class SPYOrderFlowExternalStrategy(ExternalStrategyInterface):
    """
    External SPY order flow strategy implementation.

    Uses microstructure features from order flow to predict short-term movements.
    Generates directional signals based on order flow imbalances and momentum.
    """

    def __init__(
        self,
        asset_id: AssetId = "SPY",
        lookback_window: int = 100,
        momentum_window_short: int = 5,
        momentum_window_long: int = 20,
        imbalance_threshold: float = 0.65,
        momentum_threshold: float = 0.002,
        min_data_points: int = 20,
        signal_cooldown: int = 5,  # Minimum bars between signals
    ):
        """
        Initialize SPY order flow strategy.

        Args:
            asset_id: Asset identifier for SPY
            lookback_window: Rolling window for statistics
            momentum_window_short: Short momentum calculation window
            momentum_window_long: Long momentum calculation window
            imbalance_threshold: Threshold for order flow imbalance signals
            momentum_threshold: Threshold for momentum signals
            min_data_points: Minimum data points before generating signals
            signal_cooldown: Minimum bars between signals
        """
        self.asset_id = asset_id
        self.lookback_window = lookback_window
        self.momentum_window_short = momentum_window_short
        self.momentum_window_long = momentum_window_long
        self.imbalance_threshold = imbalance_threshold
        self.momentum_threshold = momentum_threshold
        self.min_data_points = min_data_points
        self.signal_cooldown = signal_cooldown

        # Initialize state
        self.state = OrderFlowState()

    def initialize(self) -> None:
        """Initialize strategy state (required by interface)."""
        self.state = OrderFlowState()
        print(f"[SPYOrderFlowStrategy] Initialized with asset {self.asset_id}")

    def finalize(self) -> None:
        """Clean up strategy state (required by interface)."""
        print(f"[SPYOrderFlowStrategy] Generated {self.state.signal_count} signals")
        print(f"[SPYOrderFlowStrategy] Final position: {self.state.current_position}")

    def on_start(self) -> None:
        """Initialize strategy state (alias for initialize)."""
        self.initialize()

    def on_end(self) -> None:
        """Clean up strategy state (alias for finalize)."""
        self.finalize()

    def generate_signal(
        self, timestamp: datetime, pit_data: PITData
    ) -> StrategySignal | None:
        """
        Generate trading signal based on order flow analysis.

        Args:
            timestamp: Current timestamp
            pit_data: Point-in-time data snapshot

        Returns:
            Trading signal or None
        """
        # Get current market data
        price = pit_data.get_price(self.asset_id)

        # Get volume from asset data if available
        asset_data = pit_data.asset_data.get(self.asset_id, {})
        volume = asset_data.get("volume", 0)

        if price is None or volume == 0:
            return None

        # Update state with new data
        self._update_state(timestamp, price, volume, pit_data)

        # Need minimum data for analysis
        if len(self.state.prices) < self.min_data_points:
            return None

        # Check signal cooldown
        if self.state.last_signal_time is not None:
            # Find the most recent timestamp index
            try:
                last_signal_idx = self.state.timestamps.index(
                    self.state.last_signal_time
                )
                bars_since_signal = len(self.state.timestamps) - last_signal_idx - 1
            except ValueError:
                # If timestamp not found, calculate based on current time
                bars_since_signal = self.signal_cooldown + 1  # Allow signal

            if bars_since_signal < self.signal_cooldown:
                return None

        # Calculate current features
        self._calculate_features()

        # Generate signal based on order flow and momentum
        signal = self._generate_order_flow_signal(timestamp)

        if signal:
            self.state.signal_count += 1
            self.state.last_signal_time = timestamp

        return signal

    def _update_state(
        self,
        timestamp: datetime,
        price: float,
        volume: float,
        pit_data: PITData,
    ) -> None:
        """Update internal state with new market data."""
        # Add basic data
        self.state.prices.append(price)
        self.state.volumes.append(volume)
        self.state.timestamps.append(timestamp)

        # Extract order flow features from PITData if available
        asset_data = pit_data.asset_data.get(self.asset_id, {})

        # Get buy/sell volumes (use heuristics if not available)
        buy_volume = asset_data.get("buy_volume", volume * 0.5)
        sell_volume = asset_data.get("sell_volume", volume * 0.5)

        self.state.buy_volumes.append(buy_volume)
        self.state.sell_volumes.append(sell_volume)

        # Calculate volume imbalance
        total_volume = buy_volume + sell_volume + 1e-10
        imbalance = (buy_volume - sell_volume) / total_volume
        self.state.volume_imbalances.append(imbalance)

        # Keep only lookback window
        if len(self.state.prices) > self.lookback_window:
            self.state.prices = self.state.prices[-self.lookback_window :]
            self.state.volumes = self.state.volumes[-self.lookback_window :]
            self.state.timestamps = self.state.timestamps[-self.lookback_window :]
            self.state.buy_volumes = self.state.buy_volumes[-self.lookback_window :]
            self.state.sell_volumes = self.state.sell_volumes[-self.lookback_window :]
            self.state.volume_imbalances = self.state.volume_imbalances[
                -self.lookback_window :
            ]

    def _calculate_features(self) -> None:
        """Calculate order flow and momentum features."""
        prices = np.array(self.state.prices)
        volumes = np.array(self.state.volumes)
        imbalances = np.array(self.state.volume_imbalances)

        # Price momentum
        if len(prices) >= self.momentum_window_short:
            self.state.price_momentum_5 = (
                prices[-1] / prices[-self.momentum_window_short] - 1
            )

        if len(prices) >= self.momentum_window_long:
            self.state.price_momentum_20 = (
                prices[-1] / prices[-self.momentum_window_long] - 1
            )
            self.state.price_mean_20 = np.mean(prices[-self.momentum_window_long :])

        # Volume momentum
        if len(volumes) >= self.momentum_window_short:
            self.state.volume_momentum_5 = (
                volumes[-1] / np.mean(volumes[-self.momentum_window_short :]) - 1
            )

        # Price volatility
        if len(prices) >= 10:
            self.state.price_std_10 = np.std(prices[-10:])

        # Imbalance statistics
        if len(imbalances) >= 20:
            self.state.imbalance_mean = np.mean(imbalances[-20:])
            self.state.imbalance_std = np.std(imbalances[-20:])

        # Current imbalance ratio
        if len(self.state.buy_volumes) > 0:
            recent_buy = np.mean(self.state.buy_volumes[-5:])
            recent_sell = np.mean(self.state.sell_volumes[-5:])
            self.state.imbalance_ratio = recent_buy / (recent_buy + recent_sell + 1e-10)

    def _generate_order_flow_signal(self, timestamp: datetime) -> StrategySignal | None:
        """Generate signal based on order flow imbalance and momentum."""
        # Signal strength based on multiple factors
        signal_strength = 0.0
        factors = []

        # 1. Order flow imbalance signal
        if self.state.imbalance_ratio > self.imbalance_threshold:
            signal_strength += 0.4
            factors.append("buy_pressure")
        elif self.state.imbalance_ratio < (1 - self.imbalance_threshold):
            signal_strength -= 0.4
            factors.append("sell_pressure")

        # 2. Price momentum confirmation
        if abs(self.state.price_momentum_5) > self.momentum_threshold:
            if self.state.price_momentum_5 > 0:
                signal_strength += 0.3
                factors.append("positive_momentum")
            else:
                signal_strength -= 0.3
                factors.append("negative_momentum")

        # 3. Volume surge detection
        if self.state.volume_momentum_5 > 0.5:  # 50% above average
            signal_strength += 0.2 * np.sign(signal_strength)
            factors.append("volume_surge")

        # 4. Mean reversion opportunity
        if len(self.state.prices) >= 20:
            price_deviation = (self.state.prices[-1] - self.state.price_mean_20) / (
                self.state.price_std_10 + 1e-10
            )
            if abs(price_deviation) > 2:  # 2 standard deviations
                signal_strength -= 0.2 * np.sign(price_deviation)  # Mean reversion
                factors.append("mean_reversion")

        # Generate signal if strong enough
        threshold = 0.5
        if abs(signal_strength) >= threshold:
            # Determine position
            if signal_strength > 0:
                position = min(1.0, signal_strength)  # Long
                signal_type = "BUY"
            else:
                position = max(-1.0, signal_strength)  # Short
                signal_type = "SELL"

            # Confidence based on signal strength
            confidence = min(1.0, abs(signal_strength) / 1.5)

            # Update position tracking
            self.state.current_position = position

            return StrategySignal(
                timestamp=timestamp,
                asset_id=self.asset_id,
                position=position,
                confidence=confidence,
                metadata={
                    "signal_type": signal_type,
                    "factors": factors,
                    "imbalance_ratio": round(self.state.imbalance_ratio, 3),
                    "price_momentum_5": round(self.state.price_momentum_5, 4),
                    "volume_momentum": round(self.state.volume_momentum_5, 3),
                    "signal_strength": round(signal_strength, 3),
                },
            )

        return None

    def get_current_statistics(self) -> dict[str, Any]:
        """Get current strategy statistics."""
        return {
            "data_points": len(self.state.prices),
            "current_position": self.state.current_position,
            "imbalance_ratio": self.state.imbalance_ratio,
            "price_momentum_5": self.state.price_momentum_5,
            "price_momentum_20": self.state.price_momentum_20,
            "volume_momentum_5": self.state.volume_momentum_5,
            "signal_count": self.state.signal_count,
        }


class SPYOrderFlowAdapter(DataFrameAdapter):
    """
    Complete adapter for SPY order flow trading strategy.

    This combines the external strategy with DataFrame support
    and provides complete ml4t.backtest integration for order flow analysis.
    """

    def __init__(
        self,
        asset_id: AssetId = "SPY",
        lookback_window: int = 100,
        momentum_window_short: int = 5,
        momentum_window_long: int = 20,
        imbalance_threshold: float = 0.65,
        momentum_threshold: float = 0.002,
        position_scaling: float = 0.2,
        window_size: int = 1000,
        **kwargs,
    ):
        """
        Initialize SPY order flow adapter.

        Args:
            asset_id: Asset identifier for SPY
            lookback_window: Rolling window for statistics
            momentum_window_short: Short momentum window
            momentum_window_long: Long momentum window
            imbalance_threshold: Order flow imbalance threshold
            momentum_threshold: Price momentum threshold
            position_scaling: Scaling factor for position size
            window_size: DataFrame history window
            **kwargs: Additional arguments for DataFrameAdapter
        """
        # Create external strategy
        external_strategy = SPYOrderFlowExternalStrategy(
            asset_id=asset_id,
            lookback_window=lookback_window,
            momentum_window_short=momentum_window_short,
            momentum_window_long=momentum_window_long,
            imbalance_threshold=imbalance_threshold,
            momentum_threshold=momentum_threshold,
        )

        # Create custom position sizer for order flow strategy
        def order_flow_position_sizer(signal: StrategySignal, cash: float) -> float:
            # Scale position based on signal strength and confidence
            base_value = cash * position_scaling

            # Adjust for confidence
            position_value = base_value * abs(signal.position) * signal.confidence

            # Apply maximum position limits
            max_position = cash * 0.5  # Maximum 50% of capital
            position_value = min(position_value, max_position)

            # Return signed position value
            return position_value if signal.position > 0 else -position_value

        # Filter kwargs for parent constructor
        parent_kwargs = {k: v for k, v in kwargs.items() if k in ["risk_manager", "name"]}

        # Initialize adapter
        super().__init__(
            external_strategy=external_strategy,
            window_size=window_size,
            position_sizer=order_flow_position_sizer,
            name=f"SPYOrderFlowAdapter_{asset_id}",
            **parent_kwargs,
        )

        # Store configuration
        self.asset_id = asset_id
        self._last_statistics: dict[str, Any] = {}

    def on_start(self) -> None:
        """Start strategy and subscribe to data feeds."""
        super().on_start()

        # Subscribe to SPY market data
        self.subscribe(asset=self.asset_id, event_type="market")

        self.log(f"Subscribed to {self.asset_id} order flow data")

    def on_market_event(self, event) -> None:
        """Process market events with order flow analysis."""
        # Update data history
        self._update_data_history(event)

        # Process through parent's market event handler
        super().on_market_event(event)

        # Update statistics for monitoring
        if hasattr(self.external_strategy, "get_current_statistics"):
            self._last_statistics = self.external_strategy.get_current_statistics()

        # Log significant order flow events
        if self._last_statistics.get("imbalance_ratio", 0.5):
            imbalance = self._last_statistics["imbalance_ratio"]
            if imbalance > 0.7 or imbalance < 0.3:
                self.log(
                    f"Significant order flow: imbalance={imbalance:.3f}, "
                    f"momentum={self._last_statistics.get('price_momentum_5', 0):.4f}",
                    level="INFO",
                )

    def get_strategy_diagnostics(self) -> dict[str, Any]:
        """Get detailed diagnostics for strategy monitoring."""
        base_state = self.get_strategy_state()

        # Add order flow specific diagnostics
        order_flow_stats = self._last_statistics.copy() if self._last_statistics else {}

        return {
            **base_state,
            "order_flow_statistics": order_flow_stats,
            "strategy_type": "SPY Order Flow Momentum",
            "asset": self.asset_id,
        }


def create_spy_order_flow_strategy(**kwargs) -> SPYOrderFlowAdapter:
    """
    Factory function to create SPY order flow strategy adapter.

    Args:
        **kwargs: Configuration parameters for the adapter

    Returns:
        Configured SPYOrderFlowAdapter instance
    """
    return SPYOrderFlowAdapter(**kwargs)
