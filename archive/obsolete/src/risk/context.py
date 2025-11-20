"""Risk context for position and portfolio state evaluation.

RiskContext is an immutable snapshot of all state needed for risk rule evaluation.
It captures position state, market prices, portfolio metrics, and features/context
at a specific point in time.

Design Principles:
    - Immutable (@dataclass(frozen=True)) - contexts are snapshots in time
    - Lazy evaluation (@cached_property) - expensive calculations only when accessed
    - Explicit separation - per-asset features vs market-wide context
    - Complete state - all data needed for risk decisions in one object

Examples:
    >>> # Single-asset strategy with position management
    >>> context = RiskContext.from_state(
    ...     market_event=event,  # Has signals dict and context dict
    ...     position=broker.get_position("AAPL"),
    ...     portfolio=portfolio,
    ...     feature_provider=None
    ... )
    >>>
    >>> # Risk rule: volatility-scaled stop loss
    >>> if context.position_quantity > 0:
    ...     atr = context.features.get('atr_20', 0.0)
    ...     stop_price = context.entry_price - 2.0 * atr
    ...     if context.close < stop_price:
    ...         strategy.sell(context.asset_id, context.position_quantity)
    >>>
    >>> # Risk rule: VIX filter
    >>> vix = context.market_features.get('vix', 15.0)
    >>> if vix > 30:
    ...     # High volatility - don't enter new positions
    ...     return
    >>>
    >>> # Risk rule: max adverse excursion (MAE) exit
    >>> if context.position_quantity > 0:
    ...     mae_pct = context.max_adverse_excursion_pct
    ...     if mae_pct < -0.05:  # -5% MAE
    ...         strategy.sell(context.asset_id, context.position_quantity)
    >>>
    >>> # Multi-asset strategy with context-dependent logic
    >>> contexts = {
    ...     asset_id: RiskContext.from_state(event, pos, portfolio)
    ...     for asset_id, (event, pos) in asset_data.items()
    ... }
    >>> vix = contexts['SPY'].market_features.get('vix', 15.0)
    >>> for asset_id, ctx in contexts.items():
    ...     # Adjust position sizing based on market volatility
    ...     if vix > 25:
    ...         size = ctx.features.get('target_size', 0) * 0.5  # Half size in high vol
    ...     else:
    ...         size = ctx.features.get('target_size', 0)
"""

from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from typing import Optional

from ml4t.backtest.core.event import MarketEvent
from ml4t.backtest.core.types import AssetId, Price, Quantity
from ml4t.backtest.data.feature_provider import FeatureProvider
from ml4t.backtest.portfolio.portfolio import Portfolio
from ml4t.backtest.portfolio.state import Position


@dataclass(frozen=True)
class RiskContext:
    """Immutable snapshot of position, market, and portfolio state for risk evaluation.

    This dataclass captures all state needed for risk rule evaluation at a specific
    point in time. It's designed to be:

    1. **Immutable** - frozen dataclass ensures contexts are snapshots
    2. **Lazy** - expensive calculations (MAE, MFE) only computed when accessed
    3. **Complete** - all risk-relevant data in one place
    4. **Typed** - full type hints for IDE support

    Attributes:
        timestamp: Event timestamp
        asset_id: Asset identifier

        # Market prices (OHLCV)
        open: Open price (None if not bar data)
        high: High price (None if not bar data)
        low: Low price (None if not bar data)
        close: Close/last price
        volume: Volume (None if not bar data)

        # Quote prices (bid/ask)
        bid_price: Best bid price (None if not quote data)
        ask_price: Best ask price (None if not quote data)

        # Position state (None if no position)
        position_quantity: Current position quantity (0 if no position)
        entry_price: Average entry price (0.0 if no position)
        entry_time: First entry timestamp (None if no position)
        bars_held: Number of bars since entry (0 if no position)

        # Portfolio state
        equity: Total portfolio equity (cash + positions)
        cash: Available cash
        leverage: Current leverage ratio

        # Features (from MarketEvent.signals - per-asset)
        features: Per-asset numerical features (ML scores, ATR, RSI, etc.)

        # Context (from MarketEvent.context - market-wide)
        market_features: Market-wide features (VIX, SPY returns, regime, etc.)

        # Lazy properties (computed on first access)
        # - unrealized_pnl: Position unrealized P&L
        # - unrealized_pnl_pct: Position unrealized P&L as percentage
        # - max_favorable_excursion: Highest unrealized profit since entry
        # - max_adverse_excursion: Lowest unrealized profit since entry
        # - max_favorable_excursion_pct: MFE as percentage
        # - max_adverse_excursion_pct: MAE as percentage

    Examples:
        >>> # Access market prices
        >>> if context.close > context.high * 0.95:
        ...     # Near session high
        ...     pass
        >>>
        >>> # Access position state
        >>> if context.position_quantity > 0 and context.bars_held > 20:
        ...     # Long position held for 20+ bars
        ...     pass
        >>>
        >>> # Access per-asset features (from MarketEvent.signals)
        >>> atr = context.features.get('atr_20', 0.0)
        >>> ml_score = context.features.get('ml_score', 0.0)
        >>> rsi = context.features.get('rsi_14', 50.0)
        >>>
        >>> # Access market-wide features (from MarketEvent.context)
        >>> vix = context.market_features.get('vix', 15.0)
        >>> spy_return = context.market_features.get('spy_return', 0.0)
        >>> regime = context.market_features.get('market_regime', 0.0)
        >>>
        >>> # Lazy property - only computed when accessed
        >>> pnl_pct = context.unrealized_pnl_pct  # Computed on first access
        >>> mae_pct = context.max_adverse_excursion_pct  # Computed on first access
    """

    # Event metadata
    timestamp: datetime
    asset_id: AssetId

    # Market prices (OHLCV)
    open: Optional[Price]
    high: Optional[Price]
    low: Optional[Price]
    close: Price
    volume: Optional[float]

    # Quote prices
    bid_price: Optional[Price]
    ask_price: Optional[Price]

    # Position state
    position_quantity: Quantity
    entry_price: Price
    entry_time: Optional[datetime]
    bars_held: int

    # Portfolio state
    equity: float
    cash: float
    leverage: float

    # Features (per-asset from signals, market-wide from context)
    features: dict[str, float]
    market_features: dict[str, float]

    @cached_property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L for current position.

        Returns:
            Unrealized P&L in currency units (0.0 if no position)
        """
        if self.position_quantity == 0:
            return 0.0
        return float(self.position_quantity * (self.close - self.entry_price))

    @cached_property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage of entry value.

        Returns:
            Percentage return (0.0 if no position or zero entry price)
        """
        if self.position_quantity == 0 or self.entry_price == 0:
            return 0.0
        return float((self.close - self.entry_price) / self.entry_price)

    @cached_property
    def max_favorable_excursion(self) -> float:
        """Maximum favorable excursion (MFE) - highest unrealized profit since entry.

        If RiskManager is tracking MFE across bars (via PositionTradeState),
        it will inject the tracked value into features['_tracked_mfe'].
        Otherwise, this computes intra-bar MFE from current bar's high/low.

        For long positions: (high - entry_price) * quantity
        For short positions: (entry_price - low) * abs(quantity)

        Returns:
            MFE in currency units (0.0 if no position)
        """
        # Use tracked MFE if available (from PositionTradeState via RiskManager)
        if '_tracked_mfe' in self.features:
            return float(self.features['_tracked_mfe'])

        # Otherwise compute intra-bar MFE from OHLC
        if self.position_quantity == 0:
            return 0.0

        if self.position_quantity > 0:  # Long position
            if self.high is not None:
                return float(self.position_quantity * (self.high - self.entry_price))
        else:  # Short position
            if self.low is not None:
                return float(abs(self.position_quantity) * (self.entry_price - self.low))

        # Fallback to unrealized P&L if no high/low
        return self.unrealized_pnl

    @cached_property
    def max_adverse_excursion(self) -> float:
        """Maximum adverse excursion (MAE) - lowest unrealized profit since entry.

        If RiskManager is tracking MAE across bars (via PositionTradeState),
        it will inject the tracked value into features['_tracked_mae'].
        Otherwise, this computes intra-bar MAE from current bar's high/low.

        For long positions: (low - entry_price) * quantity
        For short positions: (entry_price - high) * abs(quantity)

        Returns:
            MAE in currency units (0.0 if no position)
        """
        # Use tracked MAE if available (from PositionTradeState via RiskManager)
        if '_tracked_mae' in self.features:
            return float(self.features['_tracked_mae'])

        # Otherwise compute intra-bar MAE from OHLC
        if self.position_quantity == 0:
            return 0.0

        if self.position_quantity > 0:  # Long position
            if self.low is not None:
                return float(self.position_quantity * (self.low - self.entry_price))
        else:  # Short position
            if self.high is not None:
                return float(abs(self.position_quantity) * (self.entry_price - self.high))

        # Fallback to unrealized P&L if no high/low
        return self.unrealized_pnl

    @cached_property
    def max_favorable_excursion_pct(self) -> float:
        """MFE as percentage of entry value.

        Returns:
            MFE percentage (0.0 if no position or zero entry price)
        """
        if self.position_quantity == 0 or self.entry_price == 0:
            return 0.0

        position_value = abs(self.position_quantity) * self.entry_price
        if position_value == 0:
            return 0.0

        return float(self.max_favorable_excursion / position_value)

    @cached_property
    def max_adverse_excursion_pct(self) -> float:
        """MAE as percentage of entry value.

        Returns:
            MAE percentage (0.0 if no position or zero entry price)
        """
        if self.position_quantity == 0 or self.entry_price == 0:
            return 0.0

        position_value = abs(self.position_quantity) * self.entry_price
        if position_value == 0:
            return 0.0

        return float(self.max_adverse_excursion / position_value)

    @property
    def current_price(self) -> Price:
        """Current market price (alias for close).

        This property provides a semantic alias for the close price,
        making rule code more readable when checking current price levels.

        Returns:
            Current market price (same as close)
        """
        return self.close

    @classmethod
    def from_state(
        cls,
        market_event: MarketEvent,
        position: Optional[Position] = None,
        portfolio: Optional[Portfolio] = None,
        feature_provider: Optional[FeatureProvider] = None,
        entry_time: Optional[datetime] = None,
        bars_held: int = 0,
    ) -> "RiskContext":
        """Build RiskContext from market event, position, and portfolio state.

        This is the primary way to create RiskContext objects. It extracts
        all relevant data from the provided objects and creates an immutable
        snapshot.

        Args:
            market_event: MarketEvent with prices, signals, and context
            position: Optional Position object (None if no position)
            portfolio: Optional Portfolio object (None to use defaults)
            feature_provider: Optional FeatureProvider to fetch additional features
                             (usually not needed - MarketEvent.signals already has features)
            entry_time: Optional entry timestamp (None if no position or unknown)
            bars_held: Number of bars held (0 if no position)

        Returns:
            RiskContext snapshot

        Examples:
            >>> # Basic usage - position and portfolio from broker
            >>> context = RiskContext.from_state(
            ...     market_event=event,
            ...     position=broker.get_position(asset_id),
            ...     portfolio=broker.portfolio
            ... )
            >>>
            >>> # No position case
            >>> context = RiskContext.from_state(
            ...     market_event=event,
            ...     position=None,
            ...     portfolio=portfolio
            ... )
            >>> assert context.position_quantity == 0
            >>>
            >>> # With entry tracking
            >>> context = RiskContext.from_state(
            ...     market_event=event,
            ...     position=position,
            ...     portfolio=portfolio,
            ...     entry_time=position_entry_time,  # From strategy tracking
            ...     bars_held=15  # Strategy tracks bars since entry
            ... )
            >>> if context.bars_held > 20:
            ...     # Exit after 20 bars
            ...     pass
        """
        # Extract position state
        if position is not None:
            position_quantity = position.quantity
            entry_price = (
                float(position.cost_basis) / position.quantity
                if position.quantity != 0
                else 0.0
            )
        else:
            position_quantity = 0.0
            entry_price = 0.0

        # Extract portfolio state
        if portfolio is not None:
            equity = portfolio.equity
            cash = portfolio.cash
            # Calculate leverage as total position value / equity
            total_position_value = sum(
                abs(p.market_value) for p in portfolio.positions.values()
            )
            leverage = total_position_value / equity if equity > 0 else 0.0
        else:
            equity = 0.0
            cash = 0.0
            leverage = 0.0

        # Extract features from MarketEvent.signals (per-asset)
        features = market_event.signals.copy() if market_event.signals else {}

        # Extract context from MarketEvent.context (market-wide)
        market_features = market_event.context.copy() if market_event.context else {}

        # Optionally fetch additional features from provider
        if feature_provider is not None:
            # Merge provider features with event signals (event takes precedence)
            provider_features = feature_provider.get_features(
                market_event.asset_id, market_event.timestamp
            )
            features = {**provider_features, **features}

            # Merge provider market features with event context (event takes precedence)
            provider_market = feature_provider.get_market_features(market_event.timestamp)
            market_features = {**provider_market, **market_features}

        return cls(
            # Event metadata
            timestamp=market_event.timestamp,
            asset_id=market_event.asset_id,
            # Market prices
            open=market_event.open,
            high=market_event.high,
            low=market_event.low,
            close=market_event.close or market_event.price or 0.0,
            volume=market_event.volume,
            # Quote prices
            bid_price=market_event.bid_price,
            ask_price=market_event.ask_price,
            # Position state
            position_quantity=position_quantity,
            entry_price=entry_price,
            entry_time=entry_time,
            bars_held=bars_held,
            # Portfolio state
            equity=equity,
            cash=cash,
            leverage=leverage,
            # Features
            features=features,
            market_features=market_features,
        )


__all__ = ["RiskContext"]
