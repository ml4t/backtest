"""
Crypto Basis Strategy QEngine Adapter
=====================================

Adapter that integrates the CryptoBasisStrategy with QEngine's event-driven architecture.
This allows the basis trading strategy to run within QEngine and benefit from
advanced execution models, slippage simulation, and commission structures.
"""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from qengine.core.types import AssetId
from qengine.strategy.adapters import (
    DataFrameAdapter,
    ExternalStrategyInterface,
    PITData,
    StrategySignal,
)


@dataclass
class BasisState:
    """State tracking for basis calculations."""

    spot_prices: list[float] = field(default_factory=list)
    futures_prices: list[float] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)
    basis_history: list[float] = field(default_factory=list)

    # Rolling statistics
    basis_mean: float | None = None
    basis_std: float | None = None
    current_z_score: float | None = None

    # Position tracking
    current_position: float = 0.0
    entry_basis: float | None = None
    entry_timestamp: datetime | None = None


class CryptoBasisExternalStrategy(ExternalStrategyInterface):
    """
    External strategy implementation for crypto basis trading.

    This wraps the original CryptoBasisStrategy logic in the interface
    required for QEngine integration.
    """

    def __init__(
        self,
        spot_asset_id: AssetId = "BTC",
        futures_asset_id: AssetId = "BTC_FUTURE",
        lookback_window: int = 120,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        max_position: float = 0.3,
        min_data_points: int = 50,
    ):
        """
        Initialize crypto basis strategy.

        Args:
            spot_asset_id: Asset ID for spot prices
            futures_asset_id: Asset ID for futures prices
            lookback_window: Number of periods for rolling statistics
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
            max_position: Maximum position size
            min_data_points: Minimum data points before generating signals
        """
        self.spot_asset_id = spot_asset_id
        self.futures_asset_id = futures_asset_id
        self.lookback_window = lookback_window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.max_position = max_position
        self.min_data_points = min_data_points

        # State
        self.state = BasisState()
        self.volatility_lookback = 20

    def initialize(self) -> None:
        """Initialize strategy state."""
        self.state = BasisState()

    def finalize(self) -> None:
        """Cleanup strategy state."""
        # Log final statistics
        if self.state.basis_history:
            total_signals = len([b for b in self.state.basis_history if b != 0])
            print(f"[CryptoBasisStrategy] Generated {total_signals} signals")
            print(f"[CryptoBasisStrategy] Final position: {self.state.current_position}")

    def generate_signal(
        self,
        timestamp: datetime,
        pit_data: PITData,
    ) -> StrategySignal | None:
        """
        Generate trading signal based on basis analysis.

        Args:
            timestamp: Current timestamp
            pit_data: Point-in-time data snapshot

        Returns:
            Trading signal or None
        """
        # Get current prices
        spot_price = pit_data.get_price(self.spot_asset_id)
        futures_price = pit_data.get_price(self.futures_asset_id)

        if spot_price is None or futures_price is None:
            return None

        # Update state with new data
        self._update_state(timestamp, spot_price, futures_price)

        # Calculate current statistics if we have enough data
        if len(self.state.basis_history) >= 2:
            self._calculate_statistics()

        # Need minimum data points for signal generation
        if len(self.state.basis_history) < self.min_data_points:
            return None

        # Generate signal
        signal = self._generate_basis_signal(timestamp)

        return signal

    def _update_state(
        self,
        timestamp: datetime,
        spot_price: float,
        futures_price: float,
    ) -> None:
        """Update internal state with new price data."""
        # Calculate basis
        basis = futures_price - spot_price

        # Update history
        self.state.timestamps.append(timestamp)
        self.state.spot_prices.append(spot_price)
        self.state.futures_prices.append(futures_price)
        self.state.basis_history.append(basis)

        # Maintain window size
        if len(self.state.basis_history) > self.lookback_window:
            self.state.timestamps = self.state.timestamps[-self.lookback_window :]
            self.state.spot_prices = self.state.spot_prices[-self.lookback_window :]
            self.state.futures_prices = self.state.futures_prices[-self.lookback_window :]
            self.state.basis_history = self.state.basis_history[-self.lookback_window :]

    def _calculate_statistics(self) -> None:
        """Calculate rolling statistics for basis."""
        if len(self.state.basis_history) < 2:
            return

        basis_array = np.array(self.state.basis_history)

        # Rolling mean and std
        self.state.basis_mean = np.mean(basis_array)
        self.state.basis_std = np.std(basis_array)

        # Current z-score
        if self.state.basis_std > 1e-8:  # Avoid division by zero
            current_basis = self.state.basis_history[-1]
            self.state.current_z_score = (
                current_basis - self.state.basis_mean
            ) / self.state.basis_std
        else:
            self.state.current_z_score = 0.0

    def _generate_basis_signal(self, timestamp: datetime) -> StrategySignal | None:
        """Generate trading signal based on basis z-score."""
        if self.state.current_z_score is None:
            return None

        z_score = self.state.current_z_score
        current_basis = self.state.basis_history[-1]

        # Calculate volatility for position sizing
        spot_returns = np.diff(np.log(self.state.spot_prices[-self.volatility_lookback :]))
        volatility = np.std(spot_returns) if len(spot_returns) > 1 else 0.01

        position = 0.0
        confidence = 0.0

        # Entry logic
        if abs(self.state.current_position) < 1e-6:  # Flat position
            if z_score > self.entry_threshold:
                # Basis too high: short futures, long spot (negative position)
                position = -1.0
                confidence = min((z_score - self.entry_threshold) / 2, 1.0)
                self.state.entry_basis = current_basis
                self.state.entry_timestamp = timestamp

            elif z_score < -self.entry_threshold:
                # Basis too low: long futures, short spot (positive position)
                position = 1.0
                confidence = min((abs(z_score) - self.entry_threshold) / 2, 1.0)
                self.state.entry_basis = current_basis
                self.state.entry_timestamp = timestamp

        # Exit logic
        else:
            if self.state.current_position > 0:  # Long futures/short spot
                if z_score > -self.exit_threshold:  # Basis normalized
                    position = 0.0
                    confidence = 1.0
                elif z_score > self.entry_threshold:  # Reversal
                    position = -1.0
                    confidence = min((z_score - self.entry_threshold) / 2, 1.0)
                else:
                    position = self.state.current_position  # Hold

            elif self.state.current_position < 0:  # Short futures/long spot
                if z_score < self.exit_threshold:  # Basis normalized
                    position = 0.0
                    confidence = 1.0
                elif z_score < -self.entry_threshold:  # Reversal
                    position = 1.0
                    confidence = min((abs(z_score) - self.entry_threshold) / 2, 1.0)
                else:
                    position = self.state.current_position  # Hold

        # Apply volatility adjustment and position limits
        if position != 0:
            volatility_scalar = 1 / (1 + volatility * 10)  # Reduce size in high vol
            position = position * min(confidence * volatility_scalar, self.max_position)

        # Update position state
        old_position = self.state.current_position
        self.state.current_position = position

        # Only generate signal if position changed significantly
        if abs(position - old_position) > 0.001:
            return StrategySignal(
                timestamp=timestamp,
                asset_id=self.spot_asset_id,  # Use spot as primary asset
                position=position,
                confidence=confidence,
                metadata={
                    "basis": current_basis,
                    "z_score": z_score,
                    "volatility": volatility,
                    "entry_basis": self.state.entry_basis,
                    "strategy_type": "crypto_basis",
                    "spot_price": self.state.spot_prices[-1],
                    "futures_price": self.state.futures_prices[-1],
                },
            )

        return None

    def get_current_statistics(self) -> dict[str, float]:
        """Get current basis statistics for monitoring."""
        if not self.state.basis_history:
            return {}

        return {
            "current_basis": self.state.basis_history[-1],
            "basis_mean": self.state.basis_mean or 0,
            "basis_std": self.state.basis_std or 0,
            "z_score": self.state.current_z_score or 0,
            "current_position": self.state.current_position,
            "data_points": len(self.state.basis_history),
        }


class CryptoBasisAdapter(DataFrameAdapter):
    """
    Complete adapter for crypto basis trading strategy.

    This combines the external strategy with DataFrame support
    and provides a complete QEngine integration.
    """

    def __init__(
        self,
        spot_asset_id: AssetId = "BTC",
        futures_asset_id: AssetId = "BTC_FUTURE",
        lookback_window: int = 120,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        max_position: float = 0.3,
        position_scaling: float = 0.1,
        window_size: int = 1000,
        **kwargs,
    ):
        """
        Initialize crypto basis adapter.

        Args:
            spot_asset_id: Asset ID for spot prices
            futures_asset_id: Asset ID for futures prices
            lookback_window: Rolling window for statistics
            entry_threshold: Z-score threshold for entries
            exit_threshold: Z-score threshold for exits
            max_position: Maximum position size
            position_scaling: Scaling factor for position size
            **kwargs: Additional arguments for DataFrameAdapter
        """
        # Create external strategy
        external_strategy = CryptoBasisExternalStrategy(
            spot_asset_id=spot_asset_id,
            futures_asset_id=futures_asset_id,
            lookback_window=lookback_window,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            max_position=max_position,
        )

        # Create custom position sizer
        def basis_position_sizer(signal: StrategySignal, cash: float) -> float:
            # Scale position based on available cash and signal strength
            base_value = cash * position_scaling  # Use X% of cash
            position_value = base_value * abs(signal.position) * signal.confidence

            # Return signed position value
            return position_value if signal.position > 0 else -position_value

        # Filter kwargs for parent constructor
        parent_kwargs = {
            k: v for k, v in kwargs.items() if k in ["position_sizer", "risk_manager", "name"]
        }

        # Initialize adapter
        super().__init__(
            external_strategy=external_strategy,
            window_size=window_size,
            position_sizer=basis_position_sizer,
            name=f"CryptoBasisAdapter_{spot_asset_id}_{futures_asset_id}",
            **parent_kwargs,
        )

        # Store configuration
        self.spot_asset_id = spot_asset_id
        self.futures_asset_id = futures_asset_id
        self._last_statistics = {}

        # Event synchronization for multi-asset basis calculation
        self._event_buffer: dict[datetime, dict[AssetId, MarketEvent]] = {}
        self._required_assets = {spot_asset_id, futures_asset_id}

    def on_start(self) -> None:
        """Start strategy and subscribe to data feeds."""
        super().on_start()

        # Subscribe to both spot and futures data
        self.subscribe(asset=self.spot_asset_id, event_type="market")
        self.subscribe(asset=self.futures_asset_id, event_type="market")

        self.log(f"Subscribed to {self.spot_asset_id} (spot) and {self.futures_asset_id} (futures)")

    def on_market_event(self, event) -> None:
        """Process market events with synchronization for basis calculation."""
        # Buffer the event by timestamp and asset
        if event.timestamp not in self._event_buffer:
            self._event_buffer[event.timestamp] = {}

        self._event_buffer[event.timestamp][event.asset_id] = event

        # Check if we have both required assets for this timestamp
        buffered_assets = set(self._event_buffer[event.timestamp].keys())

        if self._required_assets.issubset(buffered_assets):
            # We have both spot and futures data - process synchronously
            self._process_synchronized_events(event.timestamp)

            # Clean up old buffer entries (keep only last 10 timestamps)
            timestamps = sorted(self._event_buffer.keys())
            if len(timestamps) > 10:
                for old_ts in timestamps[:-10]:
                    del self._event_buffer[old_ts]
        else:
            # Still waiting for the other asset - update both DataFrame and parent's history
            super()._update_data_history(event)

    def _process_synchronized_events(self, timestamp: datetime) -> None:
        """Process events when both spot and futures data are available."""
        buffered_events = self._event_buffer[timestamp]

        # Process both events to update DataFrames AND internal history
        for event in buffered_events.values():
            # Call parent's update which maintains _data_history dict
            super()._update_data_history(event)

        # Now create PITData - parent's _create_pit_data should have both prices now
        try:
            # Create point-in-time data snapshot with both prices
            pit_data = self._create_pit_data(timestamp)

            # Debug: check if PITData has both prices
            spot_price = pit_data.get_price(self.spot_asset_id)
            futures_price = pit_data.get_price(self.futures_asset_id)

            if spot_price is None or futures_price is None:
                self.log(
                    f"Missing prices in PITData: spot={spot_price}, futures={futures_price}",
                    level="WARNING",
                )
                return

            # Generate signal from external strategy (now has both prices)
            signal = self.external_strategy.generate_signal(timestamp, pit_data)

            if signal:
                self.log(
                    f"Signal generated: pos={signal.position:.3f}, conf={signal.confidence:.3f}",
                )
                self._process_signal(signal)
            else:
                # Check if we expected a signal but didn't get one
                stats = self.external_strategy.get_current_statistics()
                if abs(stats.get("z_score", 0)) > 1.0:
                    self.log(f"Expected signal but got None: z_score={stats.get('z_score', 0):.2f}")

            # Update statistics for monitoring
            if hasattr(self.external_strategy, "get_current_statistics"):
                self._last_statistics = self.external_strategy.get_current_statistics()

                # Log all statistics for debugging
                self.log(
                    f"Basis stats: z={self._last_statistics['z_score']:.3f}, "
                    f"mean={self._last_statistics.get('basis_mean', 0):.1f}, "
                    f"std={self._last_statistics.get('basis_std', 0):.3f}, "
                    f"current={self._last_statistics.get('current_basis', 0):.0f}, "
                    f"data_pts={self._last_statistics.get('data_points', 0)}",
                    level="INFO",
                )

        except Exception as e:
            self.log(f"Error processing synchronized events: {e}", level="ERROR")

    def get_strategy_diagnostics(self) -> dict[str, any]:
        """Get detailed diagnostics for strategy monitoring."""
        base_state = self.get_strategy_state()

        # Add basis-specific statistics
        base_state.update(
            {
                "basis_statistics": self._last_statistics,
                "spot_asset": self.spot_asset_id,
                "futures_asset": self.futures_asset_id,
                "dataframe_sizes": {
                    asset: len(df) for asset, df in self.get_all_dataframes().items()
                },
            },
        )

        return base_state


def create_crypto_basis_strategy(
    spot_asset_id: AssetId = "BTC",
    futures_asset_id: AssetId = "BTC_FUTURE",
    **strategy_params,
) -> CryptoBasisAdapter:
    """
    Factory function to create a crypto basis strategy.

    Args:
        spot_asset_id: Asset ID for spot prices
        futures_asset_id: Asset ID for futures prices
        **strategy_params: Strategy parameters

    Returns:
        Configured CryptoBasisAdapter
    """
    return CryptoBasisAdapter(
        spot_asset_id=spot_asset_id,
        futures_asset_id=futures_asset_id,
        **strategy_params,
    )
