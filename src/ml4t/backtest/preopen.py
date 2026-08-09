"""Causal pre-open target lowering and reconciliation."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ml4t.specs import (
    BarPathPolicy,
    CanonicalChildOrderIntent,
    CanonicalTargetIntent,
    ExecutionBehavior,
    ExecutionCapability,
    ExecutionPolicy,
    FillEligibility,
    LifecyclePhase,
    LifecycleVersion,
    OrderParameters,
    ResidualPolicy,
    RoundingPolicy,
    SessionPolicy,
    TargetMeasure,
    TimeInForce,
)
from ml4t.specs import OrderSide as SpecOrderSide
from ml4t.specs import OrderType as SpecOrderType

from .calendar import get_schedule
from .config import (
    BacktestConfig,
    CommissionType,
    ExecutionPrice,
    ShareType,
    SlippageType,
)
from .core.shared import SubmitOrderOptions
from .models import calculate_commission, calculate_slippage
from .types import OrderSide, OrderStatus, OrderType

if TYPE_CHECKING:
    from .broker import Broker
    from .risk.position import PositionRule


class PreOpenIntentError(ValueError):
    """Base class for rejected pre-open target behavior."""


class LateAuctionIntentError(PreOpenIntentError):
    """Raised when intent registration or decision misses the opening cutoff."""


class UnsupportedPreOpenPolicyError(PreOpenIntentError):
    """Raised before orders when a declared opening policy cannot be simulated."""


class AmbiguousBarPathError(PreOpenIntentError):
    """Raised when post-open rule outcomes depend on unknown high-low order."""


class IntentOutcome(StrEnum):
    """Observed state of one canonical child intent."""

    FULL = "full"
    PARTIAL = "partial"
    PENDING = "pending"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class IntentReconciliation:
    """Fill, remainder, rejection, and rule-activation evidence for one child."""

    target_intent_id: str
    child_intent_id: str
    order_id: str
    event_time: datetime
    requested_quantity: float
    filled_quantity: float
    remaining_quantity: float
    outcome: IntentOutcome
    rejection_reason: str | None = None
    rule_policy_id: str | None = None
    rule_activated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible reconciliation record."""
        return {
            "target_intent_id": self.target_intent_id,
            "child_intent_id": self.child_intent_id,
            "order_id": self.order_id,
            "event_time": self.event_time.isoformat(),
            "requested_quantity": self.requested_quantity,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "outcome": self.outcome.value,
            "rejection_reason": self.rejection_reason,
            "rule_policy_id": self.rule_policy_id,
            "rule_activated_at": (
                self.rule_activated_at.isoformat() if self.rule_activated_at is not None else None
            ),
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> IntentReconciliation:
        """Restore one reconciliation record."""
        activation = value.get("rule_activated_at")
        return cls(
            target_intent_id=value["target_intent_id"],
            child_intent_id=value["child_intent_id"],
            order_id=value["order_id"],
            event_time=datetime.fromisoformat(value["event_time"]),
            requested_quantity=value["requested_quantity"],
            filled_quantity=value["filled_quantity"],
            remaining_quantity=value["remaining_quantity"],
            outcome=IntentOutcome(value["outcome"]),
            rejection_reason=value.get("rejection_reason"),
            rule_policy_id=value.get("rule_policy_id"),
            rule_activated_at=datetime.fromisoformat(activation) if activation else None,
        )


def default_execution_policy(config: BacktestConfig) -> ExecutionPolicy:
    """Build the explicit assumptions used when no policy is supplied."""
    fill_phase = (
        LifecyclePhase.OPENING_AUCTION
        if config.execution_price is ExecutionPrice.OPEN
        else LifecyclePhase.MARKET_EVENT
    )
    return ExecutionPolicy(
        policy_id="ml4t-backtest-default-v1",
        market_fill_phase=fill_phase,
        opening_auction=ExecutionBehavior.CLIENT,
        moc=ExecutionBehavior.CLIENT,
        limit=ExecutionBehavior.CLIENT,
        stop=ExecutionBehavior.CLIENT,
        stop_limit=ExecutionBehavior.DISABLED,
        trailing=ExecutionBehavior.CLIENT,
        contingent=ExecutionBehavior.CLIENT,
        fee_bps=(
            config.commission_rate * 10_000
            if config.commission_type is CommissionType.PERCENTAGE
            else 0.0
        ),
        slippage_bps=(
            config.slippage_rate * 10_000
            if config.slippage_type is SlippageType.PERCENTAGE
            else 0.0
        ),
        spread_bps=0.0,
        impact_bps=0.0,
        latency_ms=0.0,
        liquidity_fraction=1.0,
        allow_partial_fills=config.partial_fills_allowed,
        bar_path=BarPathPolicy.CONSERVATIVE,
    )


class PreOpenTargetManager:
    """Accept, lower, execute, and reconcile canonical opening targets."""

    def __init__(
        self,
        broker: Broker,
        policy: ExecutionPolicy,
        lifecycle_version: LifecycleVersion,
        *,
        calendar: str | None,
    ) -> None:
        self.broker = broker
        self.policy = policy
        self.lifecycle_version = lifecycle_version
        self.calendar = calendar
        self._targets: dict[str, CanonicalTargetIntent] = {}
        self._idempotency: dict[str, str] = {}
        self._children: dict[str, CanonicalChildOrderIntent] = {}
        self._order_by_child: dict[str, str] = {}
        self._processed_targets: set[str] = set()
        self._reconciliations: list[IntentReconciliation] = []
        self._position_rules: dict[str, PositionRule] = {}

    @property
    def targets(self) -> tuple[CanonicalTargetIntent, ...]:
        """Return accepted targets in registration order."""
        return tuple(self._targets.values())

    @property
    def children(self) -> tuple[CanonicalChildOrderIntent, ...]:
        """Return all derived canonical child intents."""
        return tuple(self._children.values())

    @property
    def reconciliations(self) -> tuple[IntentReconciliation, ...]:
        """Return retained reconciliation history."""
        return tuple(self._reconciliations)

    def register(
        self,
        intent: CanonicalTargetIntent,
        *,
        position_rules: PositionRule | None = None,
    ) -> CanonicalTargetIntent:
        """Accept one idempotent target without reading feed prices."""
        if intent.lifecycle_version is not self.lifecycle_version:
            raise UnsupportedPreOpenPolicyError(
                f"target lifecycle {intent.lifecycle_version.value!r} does not match engine "
                f"lifecycle {self.lifecycle_version.value!r}"
            )
        if intent.effective_phase is not LifecyclePhase.PRE_OPEN:
            raise UnsupportedPreOpenPolicyError(
                "opening target effective_phase must be LifecyclePhase.PRE_OPEN"
            )
        if self.policy.opening_auction is ExecutionBehavior.DISABLED:
            raise UnsupportedPreOpenPolicyError(
                f"execution policy {self.policy.policy_id!r} disables opening-auction execution"
            )
        active_phase = self.broker._active_lifecycle_phase
        if active_phase is not None and not self._registration_is_causal(intent, active_phase):
            raise LateAuctionIntentError(
                f"target {intent.intent_id!r} for {intent.effective_session} was registered during "
                f"{active_phase.value} after its pre-open decision phase"
            )
        policy_id = intent.position_rule_policy_id
        if position_rules is not None and policy_id is None:
            raise PreOpenIntentError(
                "position_rules require CanonicalTargetIntent.position_rule_policy_id"
            )
        if position_rules is not None:
            assert policy_id is not None
            self.register_position_rule_policy(policy_id, position_rules)
        elif policy_id is not None and policy_id not in self._position_rules:
            raise UnsupportedPreOpenPolicyError(
                f"position rule policy {policy_id!r} has no registered implementation"
            )
        existing_id = self._idempotency.get(intent.idempotency_key)
        if existing_id is not None:
            existing = self._targets[existing_id]
            if existing != intent:
                raise PreOpenIntentError(
                    f"idempotency key {intent.idempotency_key!r} identifies a different target"
                )
            return existing
        if intent.intent_id in self._targets:
            raise PreOpenIntentError(f"duplicate target intent_id {intent.intent_id!r}")
        assets = {target.asset for target in intent.targets}
        for registered in self._targets.values():
            overlap = assets.intersection(target.asset for target in registered.targets)
            if registered.effective_session == intent.effective_session and overlap:
                raise PreOpenIntentError(
                    f"target {intent.intent_id!r} overlaps target {registered.intent_id!r} for "
                    f"{', '.join(sorted(overlap))} in session {intent.effective_session}"
                )
        self._targets[intent.intent_id] = intent
        self._idempotency[intent.idempotency_key] = intent.intent_id
        return intent

    def register_position_rule_policy(self, policy_id: str, rules: PositionRule) -> None:
        """Bind a portable policy identity to the local rule implementation."""
        if not policy_id:
            raise ValueError("policy_id must be non-empty")
        existing = self._position_rules.get(policy_id)
        if existing is not None and existing != rules:
            raise PreOpenIntentError(f"position rule policy {policy_id!r} is already registered")
        self._position_rules[policy_id] = rules

    def process_opening(self, timestamp: datetime) -> None:
        """Lower and execute targets eligible for the current opening."""
        session = timestamp.date()
        eligible = [
            intent
            for intent in self._targets.values()
            if intent.intent_id not in self._processed_targets
            and intent.effective_session <= session
        ]
        for intent in eligible:
            if intent.effective_session < session:
                raise LateAuctionIntentError(
                    f"target {intent.intent_id!r} missed effective session "
                    f"{intent.effective_session}"
                )
            opening_time = self._opening_time(session, timestamp)
            if intent.decision_time >= opening_time or intent.information_cutoff >= opening_time:
                raise LateAuctionIntentError(
                    f"target {intent.intent_id!r} decision and information cutoff must precede "
                    f"opening {opening_time.isoformat()}"
                )
            children = self._lower(intent)
            validated_children: list[tuple[CanonicalChildOrderIntent, OrderSide]] = []
            for child in children:
                side = OrderSide.BUY if child.side is SpecOrderSide.BUY else OrderSide.SELL
                available_size = self.broker.get_available_size(child.asset, side)
                policy_fill = min(
                    child.quantity,
                    available_size * self.policy.liquidity_fraction
                    if available_size is not None
                    else child.quantity,
                )
                actual_fill = child.quantity
                if self.broker.execution_limits is not None:
                    actual_fill = self.broker.execution_limits.calculate(
                        child.quantity,
                        available_size,
                        self.broker._current_opens[child.asset],
                    ).fillable_quantity
                    if self.broker.share_type is ShareType.INTEGER:
                        actual_fill = float(int(actual_fill))
                if not math.isclose(actual_fill, policy_fill, abs_tol=1e-12):
                    raise UnsupportedPreOpenPolicyError(
                        f"execution limits allow {actual_fill:g} units for {child.asset}, but "
                        f"execution policy {self.policy.policy_id!r} declares {policy_fill:g}"
                    )
                if not self.policy.allow_partial_fills and child.quantity > actual_fill:
                    raise UnsupportedPreOpenPolicyError(
                        f"target {intent.intent_id!r} requires a partial opening fill for "
                        f"{child.asset}, but policy {self.policy.policy_id!r} disallows it"
                    )
                validated_children.append((child, side))

            order_ids: set[str] = set()
            for child, side in validated_children:
                order = self.broker.submit_order(
                    child.asset,
                    child.quantity,
                    side,
                    OrderType.MARKET,
                    _options=SubmitOrderOptions(
                        eligible_in_next_bar_mode=True,
                        rebalance_id=intent.intent_id,
                        target_intent_id=intent.intent_id,
                        child_intent_id=child.child_intent_id,
                        intent_idempotency_key=child.idempotency_key,
                    ),
                )
                if order is not None:
                    self._order_by_child[child.child_intent_id] = order.order_id
                    order_ids.add(order.order_id)
            if order_ids:
                self.broker._process_orders(use_open=True, order_ids=order_ids)
            self._processed_targets.add(intent.intent_id)
            self.reconcile(timestamp, target_intent_id=intent.intent_id)

    def reconcile(self, timestamp: datetime, *, target_intent_id: str | None = None) -> None:
        """Record current child outcomes and activate rules after observed fills."""
        for child in self._children.values():
            if target_intent_id is not None and child.target_intent_id != target_intent_id:
                continue
            order_id = self._order_by_child.get(child.child_intent_id)
            if order_id is None:
                continue
            order = self.broker.get_order(order_id)
            if order is None:
                continue
            filled = order.filled_quantity
            remaining = child.remaining_after_fill(min(filled, child.quantity))
            if order.status is OrderStatus.REJECTED:
                outcome = IntentOutcome.REJECTED
            elif order.status is OrderStatus.CANCELLED:
                outcome = IntentOutcome.CANCELLED
            elif remaining == 0:
                outcome = IntentOutcome.FULL
            elif filled > 0:
                outcome = IntentOutcome.PARTIAL
            else:
                outcome = IntentOutcome.PENDING
            intent = self._targets[child.target_intent_id]
            activation = self._activate_rules(intent, child.asset, timestamp, filled)
            record = IntentReconciliation(
                target_intent_id=child.target_intent_id,
                child_intent_id=child.child_intent_id,
                order_id=order_id,
                event_time=self._as_utc(timestamp),
                requested_quantity=child.quantity,
                filled_quantity=filled,
                remaining_quantity=remaining,
                outcome=outcome,
                rejection_reason=order.rejection_reason,
                rule_policy_id=intent.position_rule_policy_id,
                rule_activated_at=activation,
            )
            if not self._reconciliations or self._reconciliations[-1] != record:
                self._reconciliations.append(record)

    def to_state(self) -> dict[str, Any]:
        """Serialize target state for restart without duplicating accepted intent."""
        return {
            "targets": [intent.to_dict() for intent in self.targets],
            "children": [child.to_dict() for child in self.children],
            "order_by_child": dict(self._order_by_child),
            "processed_targets": sorted(self._processed_targets),
            "reconciliations": [record.to_dict() for record in self._reconciliations],
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore target evidence and idempotency state into a configured broker."""
        if self._targets or self._children or self._reconciliations:
            raise PreOpenIntentError(
                "target intent state can only be restored into an empty manager"
            )
        for raw in state.get("targets", ()):
            intent = CanonicalTargetIntent.from_mapping(raw)
            self._targets[intent.intent_id] = intent
            self._idempotency[intent.idempotency_key] = intent.intent_id
        for raw in state.get("children", ()):
            child = CanonicalChildOrderIntent.from_mapping(raw)
            self._children[child.child_intent_id] = child
        self._order_by_child = dict(state.get("order_by_child", {}))
        self._processed_targets = set(state.get("processed_targets", ()))
        self._reconciliations = [
            IntentReconciliation.from_mapping(raw) for raw in state.get("reconciliations", ())
        ]

    def capture_transaction_state(self) -> tuple[dict[str, Any], dict[str, PositionRule]]:
        """Capture manager state for callback rollback."""
        return self.to_state(), copy.deepcopy(self._position_rules)

    def restore_transaction_state(
        self, state: tuple[dict[str, Any], dict[str, PositionRule]]
    ) -> None:
        """Restore manager state after a callback failure."""
        self._targets.clear()
        self._idempotency.clear()
        self._children.clear()
        self._order_by_child.clear()
        self._processed_targets.clear()
        self._reconciliations.clear()
        self.restore_state(state[0])
        self._position_rules = state[1]

    def _registration_is_causal(
        self, intent: CanonicalTargetIntent, active_phase: LifecyclePhase
    ) -> bool:
        if active_phase is LifecyclePhase.CAUSAL_INITIALIZATION:
            return True
        if active_phase in {LifecyclePhase.PRE_OPEN, LifecyclePhase.RUN_START}:
            return active_phase is LifecyclePhase.PRE_OPEN
        current_time = self.broker._current_time
        return current_time is not None and intent.effective_session > current_time.date()

    def _lower(self, intent: CanonicalTargetIntent) -> list[CanonicalChildOrderIntent]:
        open_prices: dict[str, float] = {}
        for target in intent.targets:
            price = self.broker._current_opens.get(target.asset)
            if price is None or not math.isfinite(price) or price <= 0:
                raise PreOpenIntentError(
                    f"target {intent.intent_id!r} has no valid opening price for {target.asset}"
                )
            open_prices[target.asset] = price

        equity = self.broker.cash + sum(
            position.quantity
            * open_prices.get(asset, self.broker._last_prices.get(asset, position.entry_price))
            * position.multiplier
            for asset, position in self.broker.positions.items()
        )
        desired: dict[str, float] = {}
        raw_desired: dict[str, float] = {}
        effective_cash_buffer = max(intent.cash_buffer, self.broker.cash_buffer_pct)
        for target in intent.targets:
            if target.measure is TargetMeasure.WEIGHT:
                multiplier = self.broker.get_multiplier(target.asset)
                target_notional = equity * (1.0 - effective_cash_buffer) * target.value
                raw = target_notional / (open_prices[target.asset] * multiplier)
                current = self.broker.positions.get(target.asset)
                current_quantity = current.quantity if current is not None else 0.0
                if target.value > 0:
                    raw = self._cost_adjusted_buy_quantity(
                        target.asset,
                        target_notional,
                        open_prices[target.asset],
                        multiplier,
                        current_quantity,
                    )
            else:
                raw = target.value
            raw_desired[target.asset] = raw
            desired[target.asset] = self._round_quantity(raw, intent.rounding)

        if intent.residual is ResidualPolicy.REJECT and any(
            not math.isclose(raw_desired[asset], quantity, abs_tol=1e-12)
            for asset, quantity in desired.items()
        ):
            raise PreOpenIntentError(
                f"target {intent.intent_id!r} leaves a rounding residual under reject policy"
            )
        if intent.residual is ResidualPolicy.LARGEST_REMAINDER:
            self._allocate_largest_remainders(
                raw_desired,
                desired,
                open_prices,
                effective_cash_buffer,
            )

        children: list[CanonicalChildOrderIntent] = []
        for target in sorted(intent.targets, key=lambda item: item.asset):
            current = self.broker.positions.get(target.asset)
            current_quantity = current.quantity if current is not None else 0.0
            delta = desired[target.asset] - current_quantity
            if math.isclose(delta, 0.0, abs_tol=1e-12):
                continue
            child = CanonicalChildOrderIntent(
                child_intent_id=f"{intent.intent_id}:{target.asset}",
                target_intent_id=intent.intent_id,
                idempotency_key=f"{intent.idempotency_key}:{target.asset}",
                asset=target.asset,
                side=SpecOrderSide.BUY if delta > 0 else SpecOrderSide.SELL,
                quantity=abs(delta),
                order_type=SpecOrderType.MARKET,
                parameters=OrderParameters(),
                eligibility_phase=LifecyclePhase.OPENING_AUCTION,
                fill_eligibility=FillEligibility.OPENING_AUCTION,
                time_in_force=TimeInForce.OPG,
                session_policy=SessionPolicy.REGULAR,
                capabilities=(ExecutionCapability.OPENING_AUCTION,),
                reason=intent.reason,
                lifecycle_version=intent.lifecycle_version,
            )
            self._children[child.child_intent_id] = child
            children.append(child)
        return children

    def _cost_adjusted_buy_quantity(
        self,
        asset: str,
        target_notional: float,
        price: float,
        multiplier: float,
        current_quantity: float,
    ) -> float:
        """Find the largest target quantity whose notional and entry costs fit its allocation."""
        upper = target_notional / (price * multiplier)
        if upper <= current_quantity:
            return upper
        available_size = self.broker.get_available_size(asset, OrderSide.BUY)
        declared_cost_rate = (
            self.policy.fee_bps
            + self.policy.slippage_bps
            + self.policy.spread_bps
            + self.policy.impact_bps
        ) / 10_000

        def required_notional(quantity: float) -> float:
            entry_quantity = max(0.0, quantity - current_quantity)
            if entry_quantity == 0:
                return quantity * price * multiplier
            impacted_price = price
            if self.broker.market_impact_model is not None:
                impact = self.broker.market_impact_model.calculate(
                    entry_quantity,
                    price,
                    available_size,
                    True,
                )
                if not math.isfinite(impact) or impact < 0:
                    raise UnsupportedPreOpenPolicyError(
                        f"market impact for {asset} must be finite and non-negative"
                    )
                impacted_price += impact
            slippage = calculate_slippage(
                self.broker.slippage_model,
                asset,
                entry_quantity,
                impacted_price,
                available_size,
            )
            fill_price = impacted_price + slippage
            if not math.isfinite(fill_price) or fill_price <= 0:
                raise UnsupportedPreOpenPolicyError(
                    f"estimated opening execution price for {asset} must be finite and positive"
                )
            commission = calculate_commission(
                self.broker.commission_model,
                asset,
                entry_quantity,
                fill_price,
            )
            actual_entry_cost = entry_quantity * (fill_price - price) * multiplier + commission
            declared_entry_cost = entry_quantity * price * multiplier * declared_cost_rate
            return quantity * price * multiplier + max(actual_entry_cost, declared_entry_cost)

        low = current_quantity
        high = upper
        for _ in range(64):
            midpoint = (low + high) / 2
            if required_notional(midpoint) <= target_notional:
                low = midpoint
            else:
                high = midpoint
        return low

    def _allocate_largest_remainders(
        self,
        raw: dict[str, float],
        rounded: dict[str, float],
        prices: dict[str, float],
        cash_buffer: float,
    ) -> None:
        if self.broker.share_type is ShareType.FRACTIONAL:
            rounded.update(raw)
            return
        available_cash = max(0.0, self.broker.cash * (1.0 - cash_buffer))
        target_notional = sum(max(0.0, quantity) * prices[asset] for asset, quantity in raw.items())
        committed = sum(max(0.0, quantity) * prices[asset] for asset, quantity in rounded.items())
        residual_cash = max(0.0, min(available_cash, target_notional) - committed)
        candidates = sorted(
            (
                (abs(raw[asset] - rounded[asset]), asset)
                for asset in raw
                if raw[asset] > rounded[asset] >= 0
            ),
            key=lambda item: (-item[0], item[1]),
        )
        for _, asset in candidates:
            price = prices[asset] * self.broker.get_multiplier(asset)
            if price <= residual_cash:
                rounded[asset] += 1.0
                residual_cash -= price

    def _round_quantity(self, value: float, policy: RoundingPolicy) -> float:
        if policy is RoundingPolicy.NONE:
            if self.broker.share_type is ShareType.INTEGER and not value.is_integer():
                raise UnsupportedPreOpenPolicyError(
                    "rounding=none requires fractional shares or integral target quantities"
                )
            return value
        if policy is RoundingPolicy.TOWARD_ZERO:
            return float(math.trunc(value))
        magnitude = math.floor(abs(value) + 0.5)
        return math.copysign(float(magnitude), value)

    def _activate_rules(
        self,
        intent: CanonicalTargetIntent,
        asset: str,
        timestamp: datetime,
        filled_quantity: float,
    ) -> datetime | None:
        policy_id = intent.position_rule_policy_id
        if policy_id is None or filled_quantity <= 0:
            return None
        child_intent_id = f"{intent.intent_id}:{asset}"
        prior_activation = next(
            (
                record.rule_activated_at
                for record in reversed(self._reconciliations)
                if record.child_intent_id == child_intent_id
                and record.rule_activated_at is not None
            ),
            None,
        )
        if prior_activation is not None:
            return prior_activation
        rules = self._position_rules.get(policy_id)
        if rules is None:
            raise UnsupportedPreOpenPolicyError(
                f"position rule policy {policy_id!r} has no registered implementation"
            )
        position = self.broker.get_position(asset)
        if position is None:
            return None
        existing = position.context.get("rule_activation_time")
        if existing is not None:
            return datetime.fromisoformat(existing)
        activated_at = self._as_utc(timestamp)
        position.context.update(
            {
                "rule_activation_time": activated_at.isoformat(),
                "rule_activation_bar_index": self.broker._bar_index,
                "bar_path_policy": self.policy.bar_path,
            }
        )
        self.broker.set_position_rules(copy.deepcopy(rules), asset=asset)
        return activated_at

    def _opening_time(self, session: date, fallback: datetime) -> datetime:
        if self.calendar is not None:
            schedule = get_schedule(self.calendar, session, session)
            if schedule.is_empty():
                raise PreOpenIntentError(f"{session} is not a session on {self.calendar}")
            return self._as_utc(schedule["market_open"][0])
        return self._as_utc(fallback)

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


__all__ = [
    "AmbiguousBarPathError",
    "IntentOutcome",
    "IntentReconciliation",
    "LateAuctionIntentError",
    "PreOpenIntentError",
    "PreOpenTargetManager",
    "UnsupportedPreOpenPolicyError",
    "default_execution_policy",
]
