"""Pluggable commission and slippage models."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

from .config import ExecutionPrice

if TYPE_CHECKING:
    from .types import Order

# === Protocols ===


@runtime_checkable
class CommissionModel(Protocol):
    """Return a non-negative fee for a proposed or executed fill.

    Pre-trade estimates call ``calculate`` on a deep copy. Only the fill-time
    result on the original model is charged. Stateful models can advance tiers
    on filled executions, but must support deepcopy and avoid external effects
    from ``calculate`` because those cannot be isolated by copying.
    """

    def calculate(self, asset: str, quantity: float, price: float) -> float: ...


@runtime_checkable
class SlippageModel(Protocol):
    """Protocol for slippage/market impact calculation."""

    def calculate(
        self, asset: str, quantity: float, price: float, volume: float | None
    ) -> float: ...


@dataclass(frozen=True)
class CommissionContext:
    """A quote's actual fill increment and the order's previously committed fees."""

    order_id: str | None
    asset: str
    signed_quantity: float
    price: float
    multiplier: float
    filled_quantity: float
    commission_paid: float
    filled_notional: float


@dataclass(frozen=True)
class SlippageContext:
    """Actual fill size, observed traded volume and already committed asset/bar volume."""

    asset: str
    quantity: float
    price: float
    traded_volume: float | None
    volume_consumed: float
    quoted_size: float | None
    timestamp: datetime
    bar_high: float | None = None
    bar_low: float | None = None


@dataclass(frozen=True)
class VolatilityVolumeSlippage:
    """Square-root impact from explicit completed-history ADV and annualized volatility.

    The read-only statistics callback receives the asset and actual fill clock.
    None or zero ADV selects the explicit 7.5bp insufficient-history convention.
    Choose the high/low midpoint execution basis for the legacy futures method.
    """

    eta: float
    statistics: Callable[[str, datetime], tuple[float, float] | None] | None = None
    requires_traded_volume: ClassVar[bool] = True
    requires_observed_ohlc: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.eta) or self.eta < 0:
            raise ValueError("eta must be finite and nonnegative")
        if self.statistics is not None and not callable(self.statistics):
            raise TypeError("statistics must be a read-only callable or None")

    def calculate(self, asset: str, quantity: float, price: float, volume: float | None) -> float:
        raise TypeError("VolatilityVolumeSlippage requires an observed SlippageContext")

    def __deepcopy__(self, memo):
        return type(self)(self.eta, self.statistics)

    def calculate_with_context(self, context: SlippageContext) -> float:
        if any(
            value is None or not math.isfinite(value) or value <= 0
            for value in (context.bar_high, context.bar_low)
        ):
            raise ValueError("Volatility-volume impact requires observed positive high and low")
        assert context.bar_high is not None and context.bar_low is not None
        if context.bar_high < context.bar_low:
            raise ValueError("Observed high cannot be below low")
        values = (
            self.statistics(context.asset, context.timestamp)
            if self.statistics is not None
            else None
        )
        if values is None:
            return context.price * 7.5 / 10000.0
        mean_volume, annualized_volatility = values
        if (
            not math.isfinite(mean_volume)
            or mean_volume < 0
            or not math.isfinite(annualized_volatility)
            or annualized_volatility < 0
        ):
            raise ValueError("Impact statistics must be finite and nonnegative")
        if mean_volume == 0:
            return context.price * 7.5 / 10000.0
        from .execution.impact import SquareRootImpact

        return SquareRootImpact(
            coefficient=self.eta / 10000.0, volatility=annualized_volatility
        ).calculate(context.quantity, context.price, mean_volume, True)


class AssetSlippage:
    """Route stable keys to native models and explicit execution-price bases."""

    def __init__(
        self,
        models: Mapping[str, SlippageModel],
        *,
        default: SlippageModel,
        price_sources: Mapping[str, ExecutionPrice | str] | None = None,
    ) -> None:
        if not isinstance(default, SlippageModel) or any(
            not isinstance(key, str) or not key or not isinstance(model, SlippageModel)
            for key, model in models.items()
        ):
            raise TypeError("Asset slippage requires stable string keys and native SlippageModels")
        if isinstance(default, AssetSlippage) or any(
            isinstance(model, AssetSlippage) for model in models.values()
        ):
            raise TypeError("Declare one asset slippage routing table, without nested routes")
        if any(not isinstance(key, str) or not key for key in (price_sources or {})):
            raise ValueError("Execution-price sources require stable string asset keys")
        self.models = MappingProxyType(dict(models))
        self.default = default
        self.price_sources = MappingProxyType(
            {key: ExecutionPrice(value) for key, value in (price_sources or {}).items()}
        )

    def for_asset(self, asset: str) -> SlippageModel:
        return self.models.get(asset, self.default)

    @property
    def requires_observed_ohlc(self) -> bool:
        return any(
            getattr(model, "requires_observed_ohlc", False)
            for model in [self.default, *self.models.values()]
        )

    def calculate(self, asset: str, quantity: float, price: float, volume: float | None) -> float:
        return self.for_asset(asset).calculate(asset, quantity, price, volume)

    def calculate_with_context(self, context: SlippageContext) -> float:
        return calculate_slippage(
            self.for_asset(context.asset),
            context.asset,
            context.quantity,
            context.price,
            context.traded_volume,
            context=context,
        )


def calculate_commission(
    model: CommissionModel,
    asset: str,
    quantity: float,
    price: float,
    *,
    order: Order | None = None,
    multiplier: float = 1.0,
) -> float:
    quote = getattr(model, "quote", None)
    value = (
        quote(
            CommissionContext(
                order.order_id if order is not None else None,
                asset,
                -abs(quantity)
                if order is not None and order.side.value == "sell"
                else abs(quantity),
                price,
                multiplier,
                order.filled_quantity if order is not None else 0.0,
                order.commission_paid if order is not None else 0.0,
                abs((order.filled_price or 0.0) * order.filled_quantity * multiplier)
                if order is not None
                else 0.0,
            )
        )
        if quote is not None
        else model.calculate(asset, quantity, price)
    )
    return _validate_nonnegative_model_output("commission", model, value)


def estimate_commission(
    model: CommissionModel,
    asset: str,
    quantity: float,
    price: float,
    *,
    order: Order | None = None,
    multiplier: float = 1.0,
) -> float:
    """Estimate a fee without advancing a stateful model's execution state."""
    if type(model) is AssetCommission:
        return estimate_commission(
            model.models.get(asset, model.default),
            asset,
            quantity,
            price,
            order=order,
            multiplier=multiplier,
        )
    try:
        estimate_model = deepcopy(model)
    except Exception as exc:
        raise TypeError("commission model must support deepcopy for pre-trade estimates") from exc
    if estimate_model is model:
        raise TypeError("commission model deepcopy must return an independent instance")
    return calculate_commission(
        estimate_model, asset, quantity, price, order=order, multiplier=multiplier
    )


def quote_commission(
    model: CommissionModel,
    asset: str,
    quantity: float,
    price: float,
    *,
    order: Order | None = None,
    multiplier: float = 1.0,
) -> float:
    """Quote without consuming mutable models; retain native stateless validation cost."""
    if type(model) is AssetCommission:
        return quote_commission(
            model.models.get(asset, model.default),
            asset,
            quantity,
            price,
            order=order,
            multiplier=multiplier,
        )
    if type(model) in (
        NoCommission,
        PercentageCommission,
        PerShareCommission,
        PerOrderCommission,
        PerOrderPerShareCommission,
        TieredCommission,
        CombinedCommission,
        FuturesCommission,
    ):
        return calculate_commission(
            model, asset, quantity, price, order=order, multiplier=multiplier
        )
    return estimate_commission(model, asset, quantity, price, order=order, multiplier=multiplier)


def calculate_slippage(
    model: SlippageModel,
    asset: str,
    quantity: float,
    price: float,
    volume: float | None,
    *,
    context: SlippageContext | None = None,
) -> float:
    contextual = getattr(model, "calculate_with_context", None) if context is not None else None
    value = (
        contextual(context)
        if callable(contextual)
        else model.calculate(asset, quantity, price, volume)
    )
    return _validate_nonnegative_model_output("slippage", model, value)


def _validate_nonnegative_model_output(kind: str, model: object, value: float) -> float:
    model_name = type(model).__name__
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid {kind} from {model_name}: expected a finite non-negative number, "
            f"got {value!r}"
        ) from exc
    if not math.isfinite(numeric_value) or numeric_value < 0.0:
        raise ValueError(
            f"Invalid {kind} from {model_name}: expected a finite non-negative number, "
            f"got {value!r}"
        )
    return numeric_value


# === Commission Models ===


class NoCommission:
    """Zero commission."""

    def calculate(self, asset: str, quantity: float, price: float) -> float:
        return 0.0


class PercentageCommission:
    """Commission as percentage of trade value."""

    def __init__(self, rate: float = 0.0):
        self.rate = rate

    def calculate(self, asset: str, quantity: float, price: float) -> float:
        return abs(quantity * price * self.rate)


class PerShareCommission:
    """Fixed commission per share with optional minimum."""

    def __init__(self, per_share: float = 0.0, minimum: float = 0.0):
        self.per_share = per_share
        self.minimum = minimum

    def calculate(self, asset: str, quantity: float, price: float) -> float:
        return max(abs(quantity) * self.per_share, self.minimum)


class PerOrderCommission:
    """Compose unit, notional and one-time fees with a cumulative order minimum."""

    def __init__(
        self,
        per_share: float = 0.0,
        rate: float = 0.0,
        per_trade: float = 0.0,
        minimum: float = 0.0,
    ):
        if any(
            not math.isfinite(value) or value < 0 for value in (per_share, rate, per_trade, minimum)
        ):
            raise ValueError("Commission rates and minimum must be finite and nonnegative")
        self.per_share, self.rate = per_share, rate
        self.per_trade, self.minimum = per_trade, minimum

    def calculate(self, asset: str, quantity: float, price: float) -> float:
        raise TypeError("Per-order commission requires a CommissionContext quote")

    def quote(self, context: CommissionContext) -> float:
        quantity = context.filled_quantity + abs(context.signed_quantity)
        notional = context.filled_notional + abs(
            context.signed_quantity * context.price * context.multiplier
        )
        total = quantity * self.per_share + notional * self.rate + self.per_trade
        return max(0.0, max(total, self.minimum) - context.commission_paid)


class PerOrderPerShareCommission(PerOrderCommission):
    """Cumulative per-order minimum; each fill pays only the remaining amount due."""

    def __init__(self, per_share: float = 0.0, minimum: float = 0.0):
        super().__init__(per_share=per_share, minimum=minimum)


class AssetCommission:
    """Route explicit stable asset keys to existing native fee models.

    Quote context, including committed fills and fees, is passed unchanged.
    Missing mappings use the caller's required default model; symbols and asset
    classes are never guessed from an identifier or contract multiplier.
    """

    def __init__(self, models: Mapping[str, CommissionModel], *, default: CommissionModel) -> None:
        if not isinstance(default, CommissionModel) or any(
            not isinstance(key, str) or not key or not isinstance(model, CommissionModel)
            for key, model in models.items()
        ):
            raise TypeError("Asset fees require stable string keys and native CommissionModels")
        self.models = dict(models)
        self.default = default

    def calculate(self, asset: str, quantity: float, price: float) -> float:
        return self.models.get(asset, self.default).calculate(asset, quantity, price)

    def quote(self, context: CommissionContext) -> float:
        model = self.models.get(context.asset, self.default)
        quote = getattr(model, "quote", None)
        return (
            quote(context)
            if quote is not None
            else model.calculate(context.asset, context.signed_quantity, context.price)
        )


class TieredCommission:
    """Tiered commission based on trade value."""

    def __init__(self, tiers: list[tuple[float, float]]):
        # [(threshold, rate), ...] e.g. [(10000, 0.001), (50000, 0.0008), (inf, 0.0005)]
        self.tiers = sorted(tiers, key=lambda x: x[0])

    def calculate(self, asset: str, quantity: float, price: float) -> float:
        value = abs(quantity * price)
        for threshold, rate in self.tiers:
            if value <= threshold:
                return value * rate
        return value * self.tiers[-1][1]


class CombinedCommission:
    """Combined percentage + fixed commission per trade."""

    def __init__(self, percentage: float = 0.0, fixed: float = 0.0):
        self.percentage = percentage
        self.fixed = fixed

    def calculate(self, asset: str, quantity: float, price: float) -> float:
        value = abs(quantity * price)
        return value * self.percentage + self.fixed


class FuturesCommission:
    """PySystemTrade-compatible futures commission cost calculator.

    Calculates commission as the MAX of three cost components:
    1. per_trade: Fixed cost per trade (regardless of size)
    2. per_block: Cost per contract (block)
    3. percentage: Percentage of notional value (qty * price * multiplier)

    This matches PySystemTrade's formula:
        total = max(per_trade, per_block * abs(qty), percentage * notional)

    Note: When used without explicit multiplier (default=1.0), the per_block
    and per_trade components work correctly. The percentage component requires
    the actual contract multiplier for accurate notional-based commission.

    Reference: sysobjects/instruments.py in PySystemTrade

    Args:
        per_trade: Fixed cost per trade in currency units (default 0.0)
        per_block: Cost per contract/block (default 0.0)
        percentage: Percentage of notional as decimal (e.g., 0.0001 = 1bp)

    Example:
        # Interactive Brokers ES futures: $2.25 per contract
        comm = FuturesCommission(per_block=2.25)

        # With 10 ES at $4000: max(0, 2.25*10, 0) = $22.50
        cost = comm.calculate("ES", 10, 4000.0, multiplier=50.0)

        # Percentage-based: 1bp on notional
        comm = FuturesCommission(percentage=0.0001)
        # 10 ES at $4000 = $2M notional, 1bp = $200
        cost = comm.calculate("ES", 10, 4000.0, multiplier=50.0)
    """

    def __init__(
        self,
        per_trade: float = 0.0,
        per_block: float = 0.0,
        percentage: float = 0.0,
    ):
        self.per_trade = per_trade
        self.per_block = per_block
        self.percentage = percentage

    def calculate(
        self, asset: str, quantity: float, price: float, multiplier: float = 1.0
    ) -> float:
        """Calculate commission for a futures trade.

        Args:
            asset: Asset symbol (unused, for protocol compatibility)
            quantity: Number of contracts (signed, will use absolute value)
            price: Contract price
            multiplier: Contract multiplier (point value)

        Returns:
            Commission in currency units (MAX of the three components)
        """
        notional = abs(quantity) * price * multiplier
        return max(
            self.per_trade,
            self.per_block * abs(quantity),
            self.percentage * notional,
        )


# === Slippage Models ===


class NoSlippage:
    """Zero slippage."""

    def calculate(self, asset: str, quantity: float, price: float, volume: float | None) -> float:
        return 0.0


class FixedSlippage:
    """Fixed slippage per share (per-unit price adjustment).

    The amount is added to the fill price for buys and subtracted for sells.
    For example, with amount=0.01:
    - Buy at $100.00 fills at $100.01
    - Sell at $100.00 fills at $99.99
    """

    def __init__(self, amount: float = 0.0):
        self.amount = amount

    def calculate(self, asset: str, quantity: float, price: float, volume: float | None) -> float:
        # Return per-unit price adjustment (same as PercentageSlippage contract)
        # Broker adds this to fill price: fill = base_price ± slippage
        return self.amount


class SpreadSlippage:
    """Approximate bid-ask spread cost in currency units.

    The model returns a per-unit price adjustment to be applied on top of the
    configured execution price. By default, input values are treated as full
    quoted spreads and converted to half-spread crossing cost per side.
    """

    def __init__(
        self,
        spread: float = 0.0,
        asset_spreads: dict[str, float] | None = None,
        convention: str = "full_spread",
    ):
        self.spread = spread
        self.asset_spreads = asset_spreads or {}
        self.convention = convention

    def calculate(self, asset: str, quantity: float, price: float, volume: float | None) -> float:
        spread = self.asset_spreads.get(asset, self.spread)
        if self.convention == "full_spread":
            return spread / 2.0
        return spread


class PercentageSlippage:
    """Slippage as percentage of price (per-unit price adjustment)."""

    def __init__(self, rate: float = 0.0):
        self.rate = rate

    def calculate(self, asset: str, quantity: float, price: float, volume: float | None) -> float:
        # Return per-unit price adjustment (not total dollars)
        # Broker adds this to fill price: fill = base_price ± slippage
        return price * self.rate


class VolumeShareSlippage:
    """Slippage based on order size vs volume (market impact)."""

    def __init__(self, impact_factor: float = 0.0):
        self.impact_factor = impact_factor

    def calculate(self, asset: str, quantity: float, price: float, volume: float | None) -> float:
        if volume is None or volume == 0:
            return 0.0
        volume_fraction = abs(quantity) / volume
        impact = volume_fraction * self.impact_factor
        # Return per-unit price adjustment (not total dollars)
        return price * impact


class CumulativeVolumeShareSlippage:
    """Quadratic adverse impact using committed total absolute asset/bar participation.

    This is distinct from the existing per-fill linear VolumeShareSlippage. A
    participation cap belongs to ExecutionLimits, not this price-impact model.
    """

    requires_traded_volume = True

    def __init__(self, price_impact: float = 0.1):
        if not math.isfinite(price_impact) or price_impact < 0:
            raise ValueError("price_impact must be finite and nonnegative")
        self.price_impact = price_impact

    def calculate(self, asset: str, quantity: float, price: float, volume: float | None) -> float:
        raise TypeError("CumulativeVolumeShareSlippage requires SlippageContext")

    def calculate_with_context(self, context: SlippageContext) -> float:
        volume = context.traded_volume
        if volume is None or not math.isfinite(volume) or volume <= 0:
            raise ValueError("Cumulative volume impact requires positive observed traded volume")
        participation = (context.volume_consumed + abs(context.quantity)) / volume
        return context.price * self.price_impact * participation**2


class FuturesSlippage:
    """Futures slippage cost calculator.

    Calculates total slippage cost in currency units for portfolio cost
    analysis and position sizing. Based on PySystemTrade's formula:
        slippage = abs(qty) * price_slippage_points * multiplier

    Note: This returns TOTAL cost, not per-unit price adjustment. It is
    designed for direct cost analysis, not for use as a FillExecutor
    slippage model (which expects per-unit returns). For fill-time
    slippage on futures, use PercentageSlippage or FixedSlippage.

    Args:
        slippage_points: Slippage in price points per contract (default 0.0)

    Example:
        slip = FuturesSlippage(slippage_points=0.25)
        # 2 ES contracts: 2 * 0.25 * 50 = $25 total cost
        cost = slip.calculate("ES", 2, 4000.0, multiplier=50.0)
    """

    def __init__(self, slippage_points: float = 0.0):
        self.slippage_points = slippage_points

    def calculate(
        self,
        asset: str,
        quantity: float,
        price: float,
        volume: float | None = None,
        multiplier: float = 1.0,
    ) -> float:
        """Calculate total slippage cost for a futures trade.

        Args:
            asset: Asset symbol (unused, for signature compatibility)
            quantity: Number of contracts (signed, uses absolute value)
            price: Contract price (unused in this model)
            volume: Bar volume (unused, for signature compatibility)
            multiplier: Contract multiplier (point value)

        Returns:
            Total slippage cost in currency units
        """
        return abs(quantity) * self.slippage_points * multiplier
