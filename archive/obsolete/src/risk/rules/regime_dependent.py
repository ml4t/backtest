"""Regime-dependent risk management with adaptive rule selection.

This module implements composite risk rules that adapt to market regime,
enabling context-aware risk management that adjusts parameters based on
market conditions (high/low volatility, trending/mean-reverting, etc.).
"""

import logging
from typing import Optional

from ml4t.backtest.risk.context import RiskContext
from ml4t.backtest.risk.decision import RiskDecision
from ml4t.backtest.risk.rule import RiskRule

logger = logging.getLogger(__name__)


class RegimeDependentRule(RiskRule):
    """Composite risk rule that delegates to different rules based on market regime.

    This allows strategies to adapt risk management parameters to changing market
    conditions, such as tightening stops during high volatility or widening them
    during low volatility trending markets.

    The rule selects the appropriate sub-rule based on the current market regime,
    which can be determined from:
    1. Pre-classified regime labels in context.market_features['regime']
    2. VIX-based classification (high_vol vs low_vol)
    3. Custom regime indicators (user-defined)

    Supported Regime Sources:
        - **Direct regime labels**: Read from market_features[regime_key]
        - **VIX-based**: Classify high_vol/low_vol based on VIX threshold
        - **Custom indicators**: Any market_feature can be used for regime detection

    Common Use Cases:
        - **VIX adaptation**: Tighter stops in high VIX (panic), wider in low VIX (greed)
        - **Trend/mean-reversion**: Different rules for trending vs ranging markets
        - **Sector-specific**: Adapt risk by market sector or industry group
        - **Time-based**: Different rules for intraday vs overnight, seasonal patterns

    Args:
        regime_rules: Mapping of regime label to RiskRule to use in that regime
            Example: {'high_vol': VolatilityScaledStopLoss(1.5),
                      'low_vol': VolatilityScaledStopLoss(2.5)}
        regime_key: Which market_feature to read for regime (default: 'regime')
        default_regime: Regime to use if key missing or regime not in mapping (default: None)
        priority: Rule priority for conflict resolution (default: 100)

    Examples:
        >>> # Example 1: VIX-based risk adaptation
        >>> # Tighten stops when VIX spikes above 20
        >>> rule = RegimeDependentRule.from_vix_threshold(
        ...     vix_threshold=20.0,
        ...     high_vol_rule=VolatilityScaledStopLoss(1.5),  # 1.5x ATR (tight)
        ...     low_vol_rule=VolatilityScaledStopLoss(2.5)    # 2.5x ATR (wide)
        ... )
        >>> risk_manager.add_rule(rule)
        >>>
        >>> # Example 2: Custom regime classification
        >>> # Use pre-computed regime labels from your strategy
        >>> rule = RegimeDependentRule(
        ...     regime_rules={
        ...         'trending': DynamicTrailingStop(0.03, 0.001),      # trail tightly
        ...         'mean_reverting': PriceBasedStopLoss(entry * 0.98) # fixed stop
        ...     },
        ...     regime_key='market_regime',  # Read from market_features['market_regime']
        ...     default_regime='trending'     # Use if regime unknown
        ... )
        >>>
        >>> # Example 3: Multi-regime strategy
        >>> rule = RegimeDependentRule(
        ...     regime_rules={
        ...         'high_vol': VolatilityScaledStopLoss(1.5),   # Tight in panic
        ...         'low_vol': VolatilityScaledStopLoss(2.5),    # Wide in calm
        ...         'trending': DynamicTrailingStop(0.05, 0.001), # Trail in trends
        ...         'mean_reverting': PriceBasedStopLoss(0.02)   # Fixed in ranges
        ...     },
        ...     regime_key='regime'
        ... )

    VIX-Based Example (Recommended Pattern):
        ```python
        # Setup: Pre-compute VIX in your data pipeline
        # df = df.with_columns([
        #     pl.col('vix').alias('vix')  # VIX from external data
        # ])

        # Strategy: Use VIX threshold to adapt stops
        rule = RegimeDependentRule.from_vix_threshold(
            vix_threshold=20.0,
            high_vol_rule=VolatilityScaledStopLoss(1.5),  # Tight in panic
            low_vol_rule=VolatilityScaledStopLoss(2.5)    # Wide in calm
        )

        # Effect:
        # - VIX = 15: 2.5x ATR stop (wide, let trends run)
        # - VIX = 25: 1.5x ATR stop (tight, protect capital)
        # - Transition: Automatic when VIX crosses 20
        ```

    Custom Regime Example:
        ```python
        # Setup: Pre-compute regime in your data pipeline
        # regime_classifier = RegimeClassifier()
        # df = df.with_columns([
        #     regime_classifier(pl.col('returns')).alias('market_regime')
        # ])

        # Strategy: Use custom regime labels
        rule = RegimeDependentRule(
            regime_rules={
                'trending': DynamicTrailingStop(0.03, 0.001),
                'mean_reverting': PriceBasedStopLoss(0.02),
                'volatile': VolatilityScaledStopLoss(1.5)
            },
            regime_key='market_regime',
            default_regime='trending'
        )
        ```

    Note:
        - Regime classification must be done BEFORE the backtest (in data pipeline)
        - The rule only reads regime, it doesn't compute it
        - Missing regime → uses default_regime or returns no_action
        - Metadata includes regime label and delegated rule name
        - Works with any RiskRule subclass as sub-rules
    """

    def __init__(
        self,
        regime_rules: dict[str, RiskRule],
        *,
        regime_key: str = "regime",
        default_regime: Optional[str] = None,
        priority: int = 100,
    ):
        """Initialize RegimeDependentRule.

        Args:
            regime_rules: Mapping of regime label to RiskRule to use
            regime_key: Which market_feature to read for regime (default: 'regime')
            default_regime: Regime to use if key missing or regime not in mapping
            priority: Rule priority for conflict resolution (default: 100)

        Raises:
            ValueError: If regime_rules is empty or invalid
        """
        if not regime_rules:
            raise ValueError("regime_rules cannot be empty")

        if default_regime is not None and default_regime not in regime_rules:
            raise ValueError(
                f"default_regime '{default_regime}' not found in regime_rules. "
                f"Available regimes: {list(regime_rules.keys())}"
            )

        self.regime_rules = regime_rules
        self.regime_key = regime_key
        self.default_regime = default_regime
        self._priority = priority

        # Internal flags for VIX-based mode
        self._is_vix_based = False
        self._vix_threshold: Optional[float] = None

    @classmethod
    def from_vix_threshold(
        cls,
        vix_threshold: float,
        high_vol_rule: RiskRule,
        low_vol_rule: RiskRule,
        *,
        priority: int = 100,
    ) -> "RegimeDependentRule":
        """Create regime-dependent rule based on VIX threshold.

        This factory method creates a rule that switches between two sub-rules
        based on whether VIX is above or below a threshold.

        Args:
            vix_threshold: VIX value that separates regimes (e.g., 20.0)
            high_vol_rule: Rule to use when VIX > threshold
            low_vol_rule: Rule to use when VIX <= threshold
            priority: Rule priority for conflict resolution (default: 100)

        Returns:
            RegimeDependentRule configured for VIX-based switching

        Raises:
            ValueError: If vix_threshold is not positive

        Example:
            >>> # Tighten stops when VIX spikes above 20
            >>> rule = RegimeDependentRule.from_vix_threshold(
            ...     vix_threshold=20.0,
            ...     high_vol_rule=VolatilityScaledStopLoss(1.5),  # tighter in panic
            ...     low_vol_rule=VolatilityScaledStopLoss(2.5)    # wider in calm
            ... )
            >>>
            >>> # Effect:
            >>> # VIX = 15: Use low_vol_rule (2.5x ATR stop)
            >>> # VIX = 25: Use high_vol_rule (1.5x ATR stop)
        """
        if vix_threshold <= 0:
            raise ValueError(f"vix_threshold must be positive, got {vix_threshold}")

        instance = cls(
            regime_rules={
                "high_vol": high_vol_rule,
                "low_vol": low_vol_rule,
            },
            regime_key="vix",  # Will be used internally for metadata
            default_regime="low_vol",  # Default to low volatility
            priority=priority,
        )

        # Set internal flags for VIX-based mode
        instance._is_vix_based = True
        instance._vix_threshold = vix_threshold

        return instance

    @property
    def priority(self) -> int:
        """Priority for conflict resolution (higher = more important)."""
        return self._priority

    def evaluate(self, context: RiskContext) -> RiskDecision:
        """Evaluate risk by delegating to regime-appropriate rule.

        Algorithm:
            1. Determine current regime from context.market_features
            2. If regime unknown and no default, return no_action
            3. Select the appropriate rule for this regime
            4. Delegate to selected rule's evaluate() method
            5. Enhance decision metadata with regime info

        Args:
            context: Current risk context with position and market data

        Returns:
            RiskDecision from delegated rule, enhanced with regime metadata
        """
        # 1. Determine current regime
        regime = self._get_current_regime(context)

        # 2. If regime unknown and no default, return no action
        if regime is None:
            available_features = list(context.market_features.keys())
            return RiskDecision.no_action(
                reason=f"Regime key '{self.regime_key}' not found in market_features",
                metadata={
                    "regime_key": self.regime_key,
                    "available_features": available_features,
                    "is_vix_based": self._is_vix_based,
                },
                asset_id=context.asset_id,
            )

        # 3. Get the appropriate rule for this regime
        rule = self.regime_rules.get(regime)

        if rule is None:
            # Regime not in mapping, use default or no action
            if self.default_regime and self.default_regime in self.regime_rules:
                regime = self.default_regime
                rule = self.regime_rules[self.default_regime]
                logger.debug(
                    f"Regime '{regime}' not in rules mapping, using default regime '{self.default_regime}'"
                )
            else:
                return RiskDecision.no_action(
                    reason=f"Regime '{regime}' not in rules mapping and no default",
                    metadata={
                        "regime": regime,
                        "available_regimes": list(self.regime_rules.keys()),
                        "default_regime": self.default_regime,
                    },
                    asset_id=context.asset_id,
                )

        # 4. Delegate to the selected rule
        decision = rule.evaluate(context)

        # 5. Enhance metadata with regime info
        # Create new metadata dict preserving existing metadata
        enhanced_metadata = decision.metadata.copy() if decision.metadata else {}
        enhanced_metadata.update(
            {
                "regime": regime,
                "delegated_to": rule.__class__.__name__,
                "regime_key": self.regime_key,
            }
        )

        # If VIX-based, include VIX value and threshold
        if self._is_vix_based and self._vix_threshold is not None:
            vix_value = context.market_features.get("vix")
            if vix_value is not None:
                enhanced_metadata.update(
                    {
                        "vix": vix_value,
                        "vix_threshold": self._vix_threshold,
                    }
                )

        # Return new decision with enhanced metadata
        # We need to create a new decision since it's immutable
        return RiskDecision(
            should_exit=decision.should_exit,
            exit_type=decision.exit_type,
            exit_price=decision.exit_price,
            update_stop_loss=decision.update_stop_loss,
            update_take_profit=decision.update_take_profit,
            reason=decision.reason,
            priority=decision.priority,
            metadata=enhanced_metadata,
            asset_id=decision.asset_id,
        )

    def _get_current_regime(self, context: RiskContext) -> Optional[str]:
        """Determine current market regime from context.

        Args:
            context: Risk context with market_features

        Returns:
            Regime label string, or None if regime cannot be determined
        """
        if self._is_vix_based:
            # VIX-based regime classification
            vix = context.market_features.get("vix")
            if vix is None:
                logger.warning(
                    f"VIX not found in market_features for {context.asset_id}. "
                    f"Available features: {list(context.market_features.keys())}"
                )
                return None

            # Classify based on threshold
            return "high_vol" if vix > self._vix_threshold else "low_vol"
        else:
            # Direct regime label from market_features
            regime = context.market_features.get(self.regime_key)

            if regime is None:
                logger.debug(
                    f"Regime key '{self.regime_key}' not found in market_features for {context.asset_id}"
                )
                return None

            # Convert to string if needed (could be int/float from data)
            return str(regime) if regime is not None else None

    def __repr__(self) -> str:
        """String representation for debugging."""
        if self._is_vix_based:
            return (
                f"RegimeDependentRule(VIX-based, threshold={self._vix_threshold}, "
                f"regimes={list(self.regime_rules.keys())}, priority={self.priority})"
            )
        else:
            return (
                f"RegimeDependentRule(regime_key='{self.regime_key}', "
                f"regimes={list(self.regime_rules.keys())}, "
                f"default='{self.default_regime}', priority={self.priority})"
            )


__all__ = ["RegimeDependentRule"]
