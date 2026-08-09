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
from .execution.limits import VolumeParticipationLimit
from .models import calculate_commission, calculate_slippage
from .sessions import session_date_for_timestamp
from .types import OrderSide, OrderStatus, OrderType

if TYPE_CHECKING:
    from .accounting import AccountState
    from .broker import Broker
    from .core import MarketState
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


@dataclass(slots=True)
class _PreOpenTransactionState:
    """Constant-size checkpoint for append-only callback registrations."""

    targets_length: int
    target_by_session_asset_length: int
    idempotency_length: int
    position_rules_length: int


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
        account: AccountState,
        market: MarketState,
        calendar: str | None,
        timezone: str | None = None,
        session_start_time: str | None = None,
        data_frequency: Any | None = None,
        timestamp_semantics: Any | None = None,
    ) -> None:
        self.broker = broker
        self.account = account
        self.market = market
        self.policy = policy
        self.lifecycle_version = lifecycle_version
        self.calendar = calendar
        self.timezone = timezone
        self.session_start_time = session_start_time
        self.data_frequency = data_frequency
        self.timestamp_semantics = timestamp_semantics
        self._targets: dict[str, CanonicalTargetIntent] = {}
        self._target_by_session_asset: dict[tuple[date, str], str] = {}
        self._idempotency: dict[str, str] = {}
        self._children: dict[str, CanonicalChildOrderIntent] = {}
        self._order_by_child: dict[str, str] = {}
        self._active_children: set[str] = set()
        self._processed_targets: set[str] = set()
        self._reconciliations: list[IntentReconciliation] = []
        self._latest_reconciliation: dict[str, IntentReconciliation] = {}
        self._terminal_children: set[str] = set()
        self._rule_activations: dict[str, datetime] = {}
        self._position_rules: dict[str, PositionRule] = {}
        self._active_rule_policy_by_asset: dict[str, str] = {}
        self._installed_rule_by_asset: dict[str, PositionRule] = {}

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

    @property
    def target_count(self) -> int:
        """Return the number of accepted targets without copying them."""
        return len(self._targets)

    @property
    def child_count(self) -> int:
        """Return the number of derived child intents without copying them."""
        return len(self._children)

    @property
    def reconciliation_count(self) -> int:
        """Return the number of retained reconciliation records without copying them."""
        return len(self._reconciliations)

    def register(
        self,
        intent: CanonicalTargetIntent,
        *,
        position_rules: PositionRule | None = None,
        active_phase: LifecyclePhase | None = None,
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
        if active_phase is LifecyclePhase.RUN_START:
            raise LateAuctionIntentError(
                "register opening targets in on_prepare; on_start cannot bind a pre-open decision"
            )
        if active_phase is not None and not self._registration_is_causal(intent, active_phase):
            raise LateAuctionIntentError(
                f"target {intent.intent_id!r} for {intent.effective_session} was registered during "
                f"{active_phase.value} after its pre-open decision phase"
            )
        if (
            self.broker.share_type is ShareType.FRACTIONAL
            and intent.residual is ResidualPolicy.LARGEST_REMAINDER
        ):
            raise UnsupportedPreOpenPolicyError(
                "largest_remainder is unsupported with fractional shares; use keep_cash"
            )
        policy_id = intent.position_rule_policy_id
        policy_registration: tuple[str, PositionRule] | None = None
        if position_rules is not None and policy_id is None:
            raise PreOpenIntentError(
                "position_rules require CanonicalTargetIntent.position_rule_policy_id"
            )
        if position_rules is not None and policy_id is not None:
            policy_registration = (policy_id, position_rules)
            existing_rules = self._position_rules.get(policy_registration[0])
            if existing_rules is not None and existing_rules != position_rules:
                raise PreOpenIntentError(
                    f"position rule policy {policy_id!r} is already registered"
                )
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
            if policy_registration is not None:
                self.register_position_rule_policy(*policy_registration)
            return existing
        if intent.intent_id in self._targets:
            raise PreOpenIntentError(f"duplicate target intent_id {intent.intent_id!r}")
        assets = {target.asset for target in intent.targets}
        overlapping_ids = {
            registered_id
            for asset in assets
            if (
                registered_id := self._target_by_session_asset.get(
                    (intent.effective_session, asset)
                )
            )
            is not None
        }
        if overlapping_ids:
            registered_id = min(overlapping_ids)
            overlap = {
                asset
                for asset in assets
                if self._target_by_session_asset.get((intent.effective_session, asset))
                == registered_id
            }
            raise PreOpenIntentError(
                f"target {intent.intent_id!r} overlaps target {registered_id!r} for "
                f"{', '.join(sorted(overlap))} in session {intent.effective_session}"
            )
        if policy_registration is not None:
            self.register_position_rule_policy(*policy_registration)
        self._targets[intent.intent_id] = intent
        self._target_by_session_asset.update(
            ((intent.effective_session, asset), intent.intent_id) for asset in assets
        )
        self._idempotency[intent.idempotency_key] = intent.intent_id
        return intent

    def register_position_rule_policy(self, policy_id: str, rules: PositionRule) -> None:
        """Bind a portable policy identity to the local rule implementation."""
        if not policy_id:
            raise ValueError("policy_id must be non-empty")
        existing = self._position_rules.get(policy_id)
        if existing is not None:
            if existing != rules:
                raise PreOpenIntentError(
                    f"position rule policy {policy_id!r} is already registered"
                )
            return
        self._position_rules[policy_id] = rules

    def process_opening(self, timestamp: datetime) -> None:
        """Lower and execute targets eligible for the current opening."""
        if self._processed_targets.issuperset(self._targets):
            return
        session = self._session_date(timestamp)
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
            opening_limits = self.broker.execution_limits
            if opening_limits is None and self.policy.liquidity_fraction < 1.0:
                opening_limits = VolumeParticipationLimit(
                    max_participation=self.policy.liquidity_fraction
                )
            validated_children: list[tuple[CanonicalChildOrderIntent, OrderSide]] = []
            for child in children:
                side = OrderSide.BUY if child.side is SpecOrderSide.BUY else OrderSide.SELL
                available_size = self.broker.get_available_size(child.asset, side)
                policy_fill = child.quantity
                if self.policy.liquidity_fraction < 1.0 and available_size is not None:
                    policy_fill = min(
                        child.quantity,
                        available_size * self.policy.liquidity_fraction,
                    )
                if self.broker.share_type is ShareType.INTEGER:
                    policy_fill = float(int(policy_fill))
                actual_fill = child.quantity
                if opening_limits is not None:
                    actual_fill = opening_limits.calculate(
                        child.quantity,
                        available_size,
                        self.market.opens[child.asset],
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
                    self._active_children.add(child.child_intent_id)
                    order_ids.add(order.order_id)
            if order_ids:
                original_limits = self.broker.execution_limits
                self.broker.execution_limits = opening_limits
                try:
                    self.broker._process_orders(use_open=True, order_ids=order_ids)
                finally:
                    self.broker.execution_limits = original_limits
                for order_id in order_ids:
                    order = self.broker.get_order(order_id)
                    if order is not None and order.status is OrderStatus.PENDING:
                        self.broker.cancel_order(order_id)
                nonterminal = [
                    order_id
                    for order_id in order_ids
                    if (order := self.broker.get_order(order_id)) is None
                    or order.status is OrderStatus.PENDING
                ]
                if nonterminal:
                    raise RuntimeError(
                        "opening-auction child orders must be terminal after the OPG cancel sweep: "
                        + ", ".join(sorted(nonterminal))
                    )
            self._processed_targets.add(intent.intent_id)
            self.reconcile(timestamp, target_intent_id=intent.intent_id)

    def reconcile(self, timestamp: datetime, *, target_intent_id: str | None = None) -> None:
        """Record current child outcomes and activate rules after observed fills."""
        for child_id in tuple(self._active_children):
            child = self._children[child_id]
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
            elif remaining == 0:
                outcome = IntentOutcome.FULL
            elif filled > 0:
                outcome = IntentOutcome.PARTIAL
            elif order.status is OrderStatus.CANCELLED:
                outcome = IntentOutcome.CANCELLED
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
            previous = self._latest_reconciliation.get(child.child_intent_id)
            if previous is None or not self._same_reconciliation_state(previous, record):
                self._reconciliations.append(record)
                self._latest_reconciliation[child.child_intent_id] = record
            if outcome in {
                IntentOutcome.FULL,
                IntentOutcome.PARTIAL,
                IntentOutcome.REJECTED,
                IntentOutcome.CANCELLED,
            }:
                self._terminal_children.add(child.child_intent_id)
                self._active_children.discard(child.child_intent_id)
        self._clear_rules_for_flat_positions()

    def to_state(self) -> dict[str, Any]:
        """Serialize target state for restart without duplicating accepted intent."""
        return {
            "targets": [intent.to_dict() for intent in self.targets],
            "children": [child.to_dict() for child in self.children],
            "order_by_child": dict(self._order_by_child),
            "processed_targets": sorted(self._processed_targets),
            "reconciliations": [record.to_dict() for record in self._reconciliations],
            "active_rule_policy_by_asset": dict(self._active_rule_policy_by_asset),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore target evidence and idempotency state into a configured broker."""
        if (
            self._targets
            or self._target_by_session_asset
            or self._idempotency
            or self._children
            or self._order_by_child
            or self._active_children
            or self._processed_targets
            or self._reconciliations
            or self._latest_reconciliation
            or self._terminal_children
            or self._rule_activations
            or self._active_rule_policy_by_asset
            or self._installed_rule_by_asset
        ):
            raise PreOpenIntentError(
                "target intent state can only be restored into an empty manager"
            )
        order_by_child = dict(state.get("order_by_child", {}))
        missing_orders = sorted(
            child_id
            for child_id, order_id in order_by_child.items()
            if self.broker.get_order(order_id) is None
        )
        if missing_orders:
            names = ", ".join(missing_orders)
            raise PreOpenIntentError(
                "post-opening target intent state cannot be restored without a complete broker "
                f"checkpoint, which Engine does not support: {names}"
            )
        restored_targets = (
            CanonicalTargetIntent.from_mapping(raw) for raw in state.get("targets", ())
        )
        targets = {intent.intent_id: intent for intent in restored_targets}
        restored_children = (
            CanonicalChildOrderIntent.from_mapping(raw) for raw in state.get("children", ())
        )
        children = {child.child_intent_id: child for child in restored_children}
        processed_targets = set(state.get("processed_targets", ()))
        reconciliations = [
            IntentReconciliation.from_mapping(raw) for raw in state.get("reconciliations", ())
        ]
        latest_reconciliation: dict[str, IntentReconciliation] = {}
        rule_activations: dict[str, datetime] = {}
        for record in reconciliations:
            latest_reconciliation[record.child_intent_id] = record
            if record.rule_activated_at is not None:
                rule_activations.setdefault(record.child_intent_id, record.rule_activated_at)
        nonterminal_children = {
            child_id
            for child_id, record in latest_reconciliation.items()
            if record.outcome is IntentOutcome.PENDING
        }
        if nonterminal_children:
            names = ", ".join(sorted(nonterminal_children))
            raise PreOpenIntentError(
                "target intent state contains non-terminal opening orders that cannot be "
                f"resumed in a new broker: {names}"
            )
        unknown_targets = processed_targets.difference(targets)
        unknown_children = set(order_by_child).difference(children)
        if unknown_targets or unknown_children:
            raise PreOpenIntentError("target intent state contains unknown target or child ids")

        self._targets.update(targets)
        self._target_by_session_asset.update(
            ((intent.effective_session, target.asset), intent.intent_id)
            for intent in targets.values()
            for target in intent.targets
        )
        self._idempotency.update(
            (intent.idempotency_key, intent.intent_id) for intent in targets.values()
        )
        self._children.update(children)
        self._order_by_child.update(order_by_child)
        self._processed_targets.update(processed_targets)
        self._reconciliations.extend(reconciliations)
        self._latest_reconciliation.update(latest_reconciliation)
        self._rule_activations.update(rule_activations)
        self._terminal_children.update(latest_reconciliation)
        self._active_rule_policy_by_asset.update(state.get("active_rule_policy_by_asset", {}))
        self._installed_rule_by_asset.update(
            (asset, rules)
            for asset in self._active_rule_policy_by_asset
            if (rules := self.broker._get_position_rule_override(asset)) is not None
        )

    def capture_transaction_state(self) -> _PreOpenTransactionState:
        """Capture manager state for callback rollback."""
        return _PreOpenTransactionState(
            targets_length=len(self._targets),
            target_by_session_asset_length=len(self._target_by_session_asset),
            idempotency_length=len(self._idempotency),
            position_rules_length=len(self._position_rules),
        )

    def restore_transaction_state(self, state: _PreOpenTransactionState) -> None:
        """Restore manager state after a callback failure."""
        self._truncate_mapping(self._targets, state.targets_length)
        self._truncate_mapping(self._target_by_session_asset, state.target_by_session_asset_length)
        self._truncate_mapping(self._idempotency, state.idempotency_length)
        self._truncate_mapping(self._position_rules, state.position_rules_length)

    @staticmethod
    def _truncate_mapping(mapping: dict[Any, Any], length: int) -> None:
        while len(mapping) > length:
            mapping.popitem()

    def _registration_is_causal(
        self, intent: CanonicalTargetIntent, active_phase: LifecyclePhase
    ) -> bool:
        if active_phase is LifecyclePhase.CAUSAL_INITIALIZATION:
            return True
        if active_phase is LifecyclePhase.PRE_OPEN:
            return True
        current_time = self.market.time
        return current_time is not None and intent.effective_session > self._session_date(
            current_time
        )

    def _session_date(self, timestamp: datetime) -> date:
        return session_date_for_timestamp(
            timestamp,
            calendar=self.calendar,
            timezone=self.timezone,
            session_start_time=self.session_start_time,
            data_frequency=self.data_frequency,
            timestamp_semantics=self.timestamp_semantics,
        )

    def _lower(self, intent: CanonicalTargetIntent) -> list[CanonicalChildOrderIntent]:
        open_prices: dict[str, float] = {}
        for target in intent.targets:
            price = self.market.opens.get(target.asset)
            if price is None or not math.isfinite(price) or price <= 0:
                raise PreOpenIntentError(
                    f"target {intent.intent_id!r} has no valid opening price for {target.asset}"
                )
            open_prices[target.asset] = price

        equity = self.account.cash
        for asset, position in self.account.positions.items():
            mark_price = self.market.opens.get(asset)
            if mark_price is None:
                mark_price = self.market.last_prices.get(asset, position.entry_price)
            equity += position.quantity * mark_price * position.multiplier
        desired: dict[str, float] = {}
        raw_desired: dict[str, float] = {}
        effective_cash_buffer = max(intent.cash_buffer, self.broker.cash_buffer_pct)
        for target in intent.targets:
            if target.measure is TargetMeasure.WEIGHT:
                multiplier = self.broker.get_multiplier(target.asset)
                target_notional = equity * (1.0 - effective_cash_buffer) * target.value
                raw = target_notional / (open_prices[target.asset] * multiplier)
                current = self.account.positions.get(target.asset)
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
            current = self.account.positions.get(target.asset)
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
            return
        available_cash = max(0.0, self.account.cash * (1.0 - cash_buffer))
        multipliers = {asset: self.broker.get_multiplier(asset) for asset in raw}
        target_notional = sum(
            max(0.0, quantity) * prices[asset] * multipliers[asset]
            for asset, quantity in raw.items()
        )
        committed_notional = sum(
            max(0.0, quantity) * prices[asset] * multipliers[asset]
            for asset, quantity in rounded.items()
        )
        committed_delta = sum(
            (
                quantity
                - (
                    self.account.positions[asset].quantity
                    if asset in self.account.positions
                    else 0.0
                )
            )
            * prices[asset]
            * multipliers[asset]
            for asset, quantity in rounded.items()
        )
        cash_after_committed = max(0.0, available_cash - committed_delta)
        residual_cash = max(
            0.0,
            min(cash_after_committed, target_notional - committed_notional),
        )
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
        if filled_quantity <= 0:
            return None
        child_intent_id = f"{intent.intent_id}:{asset}"
        existing_activation = self._rule_activations.get(child_intent_id)
        if existing_activation is not None:
            return existing_activation
        position = self.broker.get_position(asset)
        if position is None:
            self._clear_target_managed_rule(asset)
            return None
        if policy_id is None:
            self._clear_target_managed_rule(asset)
            return None
        rules = self._position_rules.get(policy_id)
        if rules is None:
            raise UnsupportedPreOpenPolicyError(
                f"position rule policy {policy_id!r} has no registered implementation"
            )
        activated_at = self._as_utc(timestamp)
        position.context.update(
            {
                "rule_activation_time": activated_at.isoformat(),
                "rule_activation_bar_index": self.market.bar_index,
                "bar_path_policy": self.policy.bar_path,
                "position_rule_policy_id": policy_id,
            }
        )
        installed_rules = copy.deepcopy(rules)
        self.broker.set_position_rules(installed_rules, asset=asset)
        self._active_rule_policy_by_asset[asset] = policy_id
        self._installed_rule_by_asset[asset] = installed_rules
        self._rule_activations[child_intent_id] = activated_at
        return activated_at

    def _clear_rules_for_flat_positions(self) -> None:
        for asset in tuple(self._active_rule_policy_by_asset):
            if self.broker.get_position(asset) is None:
                self._clear_target_managed_rule(asset)

    def _clear_target_managed_rule(self, asset: str) -> None:
        if self._active_rule_policy_by_asset.pop(asset, None) is not None:
            had_installed_rule = asset in self._installed_rule_by_asset
            installed_rules = self._installed_rule_by_asset.pop(asset, None)
            position = self.broker.get_position(asset)
            if position is not None:
                self.broker._capture_lifecycle_mutation(asset=asset)
                for key in (
                    "rule_activation_time",
                    "rule_activation_bar_index",
                    "bar_path_policy",
                    "position_rule_policy_id",
                ):
                    position.context.pop(key, None)
            if (
                had_installed_rule
                and self.broker._get_position_rule_override(asset) is installed_rules
            ):
                self.broker._remove_position_rule_override(asset)

    @staticmethod
    def _same_reconciliation_state(
        previous: IntentReconciliation,
        current: IntentReconciliation,
    ) -> bool:
        return (
            previous.target_intent_id == current.target_intent_id
            and previous.child_intent_id == current.child_intent_id
            and previous.order_id == current.order_id
            and previous.requested_quantity == current.requested_quantity
            and previous.filled_quantity == current.filled_quantity
            and previous.remaining_quantity == current.remaining_quantity
            and previous.outcome is current.outcome
            and previous.rejection_reason == current.rejection_reason
            and previous.rule_policy_id == current.rule_policy_id
            and previous.rule_activated_at == current.rule_activated_at
        )

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
