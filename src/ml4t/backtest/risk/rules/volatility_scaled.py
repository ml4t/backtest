"""Volatility-scaled risk management rules (ATR-based stops and targets).

These rules adapt stop-loss and take-profit levels to market volatility,
providing tighter protection in calm markets and wider tolerance in volatile markets.
This approach reduces whipsaw exits while maintaining consistent risk control.
"""

import logging
from typing import Optional

from ml4t.backtest.core.types import Price
from ml4t.backtest.risk.context import RiskContext
from ml4t.backtest.risk.decision import RiskDecision
from ml4t.backtest.risk.rule import RiskRule

logger = logging.getLogger(__name__)


class VolatilityScaledStopLoss(RiskRule):
    """Stop loss scaled to market volatility (ATR or realized volatility).

    This rule dynamically adjusts stop-loss distance based on market volatility,
    providing:
    - **Adaptive protection**: Wider stops in volatile markets (fewer whipsaws)
    - **Tighter risk control**: Narrower stops in calm markets (better protection)
    - **Consistent risk**: Stop distance scales proportionally to price movement

    The stop loss level is calculated as:
    - **Long positions**: entry_price - (atr_multiplier × ATR)
    - **Short positions**: entry_price + (atr_multiplier × ATR)

    Args:
        atr_multiplier: How many ATRs away from entry (typical: 1.5-2.5)
        volatility_key: Which feature to use ('atr' or 'realized_volatility')
        priority: Rule priority for conflict resolution (default: 100)

    Examples:
        >>> # Typical usage: 2.0x ATR stop loss
        >>> stop_loss = VolatilityScaledStopLoss(atr_multiplier=2.0)
        >>> risk_manager.add_rule(stop_loss)
        >>>
        >>> # Conservative stop (tighter): 1.5x ATR
        >>> stop_loss = VolatilityScaledStopLoss(atr_multiplier=1.5)
        >>>
        >>> # Use realized volatility instead of ATR
        >>> stop_loss = VolatilityScaledStopLoss(
        ...     atr_multiplier=2.0,
        ...     volatility_key='realized_volatility'
        ... )

    Recommended Multipliers:
        - **1.5x ATR**: Tight stop, more exits, good for mean-reversion
        - **2.0x ATR**: Balanced, suitable for most trend-following strategies
        - **2.5x ATR**: Wide stop, fewer exits, let trends run longer

    ATR Calculation:
        ATR (Average True Range) must be pre-computed and passed via
        MarketEvent.indicators (which becomes context.features['atr']).

        Example with Polars:
        ```python
        from ml4t.features.indicators import calculate_atr

        df = df.with_columns([
            calculate_atr('high', 'low', 'close', period=14).alias('atr')
        ])
        ```

    Note:
        - Returns `update_stops()` not `exit_now()` - updates levels, doesn't exit
        - Handles missing ATR gracefully (logs warning, returns no_action)
        - Handles zero/negative ATR (invalid, returns no_action)
        - Only active when position exists
    """

    def __init__(
        self,
        atr_multiplier: float,
        *,
        volatility_key: str = "atr",
        priority: int = 100,
    ):
        """Initialize VolatilityScaledStopLoss rule.

        Args:
            atr_multiplier: How many ATRs away from entry (e.g., 2.0)
            volatility_key: Which feature to use ('atr' or 'realized_volatility')
            priority: Rule priority for conflict resolution (default: 100)
        """
        if atr_multiplier <= 0:
            raise ValueError(f"atr_multiplier must be positive, got {atr_multiplier}")

        self.atr_multiplier = atr_multiplier
        self.volatility_key = volatility_key
        self._priority = priority

    @property
    def priority(self) -> int:
        """Priority for conflict resolution (higher = more important)."""
        return self._priority

    def evaluate(self, context: RiskContext) -> RiskDecision:
        """Evaluate volatility-scaled stop loss level.

        Args:
            context: Current risk context with position and market data

        Returns:
            RiskDecision with update_stop_loss if position exists and ATR available,
            otherwise no_action
        """
        # No position - no stop loss needed
        if context.position_quantity == 0:
            return RiskDecision.no_action(
                reason="No position to apply stop loss",
                asset_id=context.asset_id,
            )

        # Get volatility from features
        volatility = context.features.get(self.volatility_key)

        if volatility is None:
            logger.warning(
                f"Volatility key '{self.volatility_key}' not found in features for {context.asset_id}. "
                f"Available features: {list(context.features.keys())}. Skipping stop loss update."
            )
            return RiskDecision.no_action(
                reason=f"Missing {self.volatility_key} in features",
                metadata={"available_features": list(context.features.keys())},
                asset_id=context.asset_id,
            )

        # Validate volatility is positive
        if volatility <= 0:
            logger.warning(
                f"Invalid {self.volatility_key}={volatility} for {context.asset_id}. "
                f"Volatility must be positive. Skipping stop loss update."
            )
            return RiskDecision.no_action(
                reason=f"Invalid {self.volatility_key}={volatility:.4f} (must be > 0)",
                metadata={self.volatility_key: volatility},
                asset_id=context.asset_id,
            )

        # Calculate stop loss level based on position direction
        is_long = context.position_quantity > 0

        if is_long:
            # Long: stop below entry
            stop_loss_price = context.entry_price - (self.atr_multiplier * volatility)
        else:
            # Short: stop above entry
            stop_loss_price = context.entry_price + (self.atr_multiplier * volatility)

        # Calculate stop distance for metadata
        stop_distance = abs(context.entry_price - stop_loss_price)
        stop_distance_pct = (
            (stop_distance / context.entry_price) if context.entry_price > 0 else 0.0
        )

        return RiskDecision.update_stops(
            update_stop_loss=stop_loss_price,
            reason=(
                f"Volatility-scaled stop: {self.atr_multiplier:.1f}x "
                f"{self.volatility_key}={volatility:.4f}, "
                f"SL={stop_loss_price:.2f} ({stop_distance_pct:.2%} from entry)"
            ),
            priority=self.priority,
            metadata={
                "volatility_key": self.volatility_key,
                "volatility_value": volatility,
                "atr_multiplier": self.atr_multiplier,
                "stop_loss_price": stop_loss_price,
                "entry_price": context.entry_price,
                "stop_distance": stop_distance,
                "stop_distance_pct": stop_distance_pct,
                "position_direction": "long" if is_long else "short",
            },
            asset_id=context.asset_id,
        )


class VolatilityScaledTakeProfit(RiskRule):
    """Take profit scaled to market volatility (ATR or realized volatility).

    This rule dynamically adjusts take-profit distance based on market volatility,
    providing:
    - **Adaptive targets**: Wider targets in volatile markets (let winners run)
    - **Tighter targets**: Narrower targets in calm markets (lock in gains faster)
    - **Consistent risk/reward**: Target distance scales proportionally to price movement

    The take profit level is calculated as:
    - **Long positions**: entry_price + (atr_multiplier × ATR)
    - **Short positions**: entry_price - (atr_multiplier × ATR)

    Args:
        atr_multiplier: How many ATRs away from entry (typical: 2.5-4.0)
        volatility_key: Which feature to use ('atr' or 'realized_volatility')
        priority: Rule priority for conflict resolution (default: 100)

    Examples:
        >>> # Typical usage: 3.0x ATR take profit (wider than 2.0x stop)
        >>> take_profit = VolatilityScaledTakeProfit(atr_multiplier=3.0)
        >>> risk_manager.add_rule(take_profit)
        >>>
        >>> # Let winners run: 4.0x ATR target
        >>> take_profit = VolatilityScaledTakeProfit(atr_multiplier=4.0)
        >>>
        >>> # Combined with stop loss for balanced risk/reward
        >>> stop_loss = VolatilityScaledStopLoss(atr_multiplier=2.0)
        >>> take_profit = VolatilityScaledTakeProfit(atr_multiplier=3.0)
        >>> # Risk/reward ratio = 3.0 / 2.0 = 1.5:1

    Recommended Multipliers:
        - **2.5x ATR**: Conservative target, faster profit-taking
        - **3.0x ATR**: Balanced target, typical for trend-following
        - **4.0x ATR**: Aggressive target, maximize trend capture

    Risk/Reward Ratios:
        Common combinations with VolatilityScaledStopLoss:
        - **1.5x stop, 3.0x target**: 2:1 reward/risk ratio
        - **2.0x stop, 3.0x target**: 1.5:1 reward/risk ratio
        - **2.0x stop, 4.0x target**: 2:1 reward/risk ratio
        - **2.5x stop, 4.0x target**: 1.6:1 reward/risk ratio

    ATR Calculation:
        ATR (Average True Range) must be pre-computed and passed via
        MarketEvent.indicators (which becomes context.features['atr']).

        Example with Polars:
        ```python
        from ml4t.features.indicators import calculate_atr

        df = df.with_columns([
            calculate_atr('high', 'low', 'close', period=14).alias('atr')
        ])
        ```

    Note:
        - Returns `update_stops()` not `exit_now()` - updates levels, doesn't exit
        - Handles missing ATR gracefully (logs warning, returns no_action)
        - Handles zero/negative ATR (invalid, returns no_action)
        - Only active when position exists
    """

    def __init__(
        self,
        atr_multiplier: float,
        *,
        volatility_key: str = "atr",
        priority: int = 100,
    ):
        """Initialize VolatilityScaledTakeProfit rule.

        Args:
            atr_multiplier: How many ATRs away from entry (e.g., 3.0)
            volatility_key: Which feature to use ('atr' or 'realized_volatility')
            priority: Rule priority for conflict resolution (default: 100)
        """
        if atr_multiplier <= 0:
            raise ValueError(f"atr_multiplier must be positive, got {atr_multiplier}")

        self.atr_multiplier = atr_multiplier
        self.volatility_key = volatility_key
        self._priority = priority

    @property
    def priority(self) -> int:
        """Priority for conflict resolution (higher = more important)."""
        return self._priority

    def evaluate(self, context: RiskContext) -> RiskDecision:
        """Evaluate volatility-scaled take profit level.

        Args:
            context: Current risk context with position and market data

        Returns:
            RiskDecision with update_take_profit if position exists and ATR available,
            otherwise no_action
        """
        # No position - no take profit needed
        if context.position_quantity == 0:
            return RiskDecision.no_action(
                reason="No position to apply take profit",
                asset_id=context.asset_id,
            )

        # Get volatility from features
        volatility = context.features.get(self.volatility_key)

        if volatility is None:
            logger.warning(
                f"Volatility key '{self.volatility_key}' not found in features for {context.asset_id}. "
                f"Available features: {list(context.features.keys())}. Skipping take profit update."
            )
            return RiskDecision.no_action(
                reason=f"Missing {self.volatility_key} in features",
                metadata={"available_features": list(context.features.keys())},
                asset_id=context.asset_id,
            )

        # Validate volatility is positive
        if volatility <= 0:
            logger.warning(
                f"Invalid {self.volatility_key}={volatility} for {context.asset_id}. "
                f"Volatility must be positive. Skipping take profit update."
            )
            return RiskDecision.no_action(
                reason=f"Invalid {self.volatility_key}={volatility:.4f} (must be > 0)",
                metadata={self.volatility_key: volatility},
                asset_id=context.asset_id,
            )

        # Calculate take profit level based on position direction
        is_long = context.position_quantity > 0

        if is_long:
            # Long: target above entry
            take_profit_price = context.entry_price + (self.atr_multiplier * volatility)
        else:
            # Short: target below entry
            take_profit_price = context.entry_price - (self.atr_multiplier * volatility)

        # Calculate target distance for metadata
        target_distance = abs(take_profit_price - context.entry_price)
        target_distance_pct = (
            (target_distance / context.entry_price) if context.entry_price > 0 else 0.0
        )

        return RiskDecision.update_stops(
            update_take_profit=take_profit_price,
            reason=(
                f"Volatility-scaled target: {self.atr_multiplier:.1f}x "
                f"{self.volatility_key}={volatility:.4f}, "
                f"TP={take_profit_price:.2f} ({target_distance_pct:.2%} from entry)"
            ),
            priority=self.priority,
            metadata={
                "volatility_key": self.volatility_key,
                "volatility_value": volatility,
                "atr_multiplier": self.atr_multiplier,
                "take_profit_price": take_profit_price,
                "entry_price": context.entry_price,
                "target_distance": target_distance,
                "target_distance_pct": target_distance_pct,
                "position_direction": "long" if is_long else "short",
            },
            asset_id=context.asset_id,
        )


__all__ = [
    "VolatilityScaledStopLoss",
    "VolatilityScaledTakeProfit",
]
