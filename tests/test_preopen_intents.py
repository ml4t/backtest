from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime

import polars as pl
import pytest
from ml4t.specs import (
    AssetTarget,
    BarPathPolicy,
    CanonicalTargetIntent,
    ExecutionBehavior,
    IntentReason,
    LifecyclePhase,
    ResidualPolicy,
    RoundingPolicy,
    TargetMeasure,
)

from ml4t.backtest import (
    AmbiguousBarPathError,
    BacktestConfig,
    BacktestResult,
    DataFeed,
    Engine,
    IntentOutcome,
    LateAuctionIntentError,
    PreOpenIntentError,
    Strategy,
    UnsupportedPreOpenPolicyError,
    default_execution_policy,
)
from ml4t.backtest.config import CommissionType, ExecutionPrice, SlippageType, TrailStopTiming
from ml4t.backtest.execution import VolumeParticipationLimit
from ml4t.backtest.preopen import PreOpenTargetManager
from ml4t.backtest.risk.position import StopLoss, TrailingStop


def prices(*, bars: int = 1, volume: float = 1_000_000.0, low: float | None = None) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [datetime(2026, 8, 3 + index) for index in range(bars)],
            "asset": ["SPY"] * bars,
            "open": [100.0] * bars,
            "high": [110.0] * bars,
            "low": [low if low is not None else (90.0 if bars == 1 else 99.0)] * bars,
            "close": [105.0] * bars,
            "volume": [volume] * bars,
        }
    )


def target_intent(
    *,
    intent_id: str = "initial-portfolio",
    session: date = date(2026, 8, 3),
    weight: float = 0.5,
    rounding: RoundingPolicy = RoundingPolicy.TOWARD_ZERO,
    residual: ResidualPolicy = ResidualPolicy.KEEP_CASH,
    position_rule_policy_id: str | None = None,
) -> CanonicalTargetIntent:
    decision = datetime(2026, 8, 2, 23, tzinfo=UTC)
    return CanonicalTargetIntent(
        intent_id=intent_id,
        decision_time=decision,
        information_cutoff=decision,
        effective_session=session,
        effective_phase=LifecyclePhase.PRE_OPEN,
        targets=(AssetTarget("SPY", TargetMeasure.WEIGHT, weight),),
        idempotency_key=f"{intent_id}-key",
        measure=TargetMeasure.WEIGHT,
        cash_buffer=0.0,
        rounding=rounding,
        residual=residual,
        reason=IntentReason.REBALANCE,
        position_rule_policy_id=position_rule_policy_id,
    )


class InitialTargetStrategy(Strategy):
    def __init__(self, intent: CanonicalTargetIntent, rules=None) -> None:
        self.intent = intent
        self.rules = rules

    def on_prepare(self, broker, config=None) -> None:
        broker.register_target_intent(self.intent, position_rules=self.rules)

    def on_data(self, timestamp, data, context, broker) -> None:
        return None


def test_initial_weight_target_lowers_at_open_and_rules_see_only_later_movement() -> None:
    intent = target_intent(position_rule_policy_id="stop-5")
    engine = Engine(
        DataFeed(prices_df=prices()),
        InitialTargetStrategy(intent, StopLoss(0.05)),
    )

    result = engine.run()

    child = engine.broker.get_child_order_intents()[0]
    assert child.quantity == 500
    assert child.eligibility_phase is LifecyclePhase.OPENING_AUCTION
    assert [fill.price for fill in result.fills] == [100.0, 95.0]
    assert result.fills[0].target_intent_id == intent.intent_id
    assert result.fills[0].child_intent_id == child.child_intent_id
    reconciliation = engine.broker.get_intent_reconciliations()[0]
    assert reconciliation.outcome is IntentOutcome.FULL
    assert reconciliation.rule_policy_id == "stop-5"
    assert reconciliation.rule_activated_at == datetime(2026, 8, 3, tzinfo=UTC)
    assert result.metrics["lifecycle_version"] == "1"
    assert result.metrics["execution_policy"]["bar_path"] == "conservative"
    assert result.metrics["target_intents"] == [intent.to_dict()]
    assert result.metrics["child_order_intents"] == [child.to_dict()]


def test_scheduled_target_registered_from_prior_event_fills_next_open() -> None:
    scheduled = target_intent(
        intent_id="scheduled",
        session=date(2026, 8, 4),
    )

    class ScheduledStrategy(Strategy):
        def on_data(self, timestamp, data, context, broker) -> None:
            if timestamp.date() == date(2026, 8, 3):
                broker.register_target_intent(scheduled)

    result = Engine(DataFeed(prices_df=prices(bars=2)), ScheduledStrategy()).run()

    assert len(result.fills) == 1
    assert result.fills[0].timestamp == datetime(2026, 8, 4)
    assert result.fills[0].price == 100.0


def test_same_session_target_registered_after_market_event_is_rejected_without_orders() -> None:
    intent = target_intent()

    class LateStrategy(Strategy):
        def on_data(self, timestamp, data, context, broker) -> None:
            broker.register_target_intent(intent)

    engine = Engine(DataFeed(prices_df=prices()), LateStrategy())

    with pytest.raises(LateAuctionIntentError, match="registered during market_event"):
        engine.run()

    assert engine.broker.orders == []
    assert engine.broker.get_target_intents() == ()
    assert engine.broker.cash == engine.config.initial_cash


def test_decision_at_open_is_rejected_before_order_or_account_mutation() -> None:
    intent = replace(
        target_intent(),
        decision_time=datetime(2026, 8, 3, tzinfo=UTC),
        information_cutoff=datetime(2026, 8, 3, tzinfo=UTC),
    )
    engine = Engine(DataFeed(prices_df=prices()), InitialTargetStrategy(intent))

    with pytest.raises(LateAuctionIntentError, match="must precede opening"):
        engine.run()

    assert engine.broker.orders == []
    assert engine.broker.positions == {}
    assert engine.broker.cash == engine.config.initial_cash


def test_disabled_opening_capability_rejects_registration() -> None:
    config = BacktestConfig()
    policy = replace(
        default_execution_policy(config),
        opening_auction=ExecutionBehavior.DISABLED,
    )
    engine = Engine(
        DataFeed(prices_df=prices(low=100.0)),
        InitialTargetStrategy(target_intent()),
        config,
        execution_policy=policy,
    )

    with pytest.raises(UnsupportedPreOpenPolicyError, match="disables opening-auction"):
        engine.run()

    assert engine.broker.orders == []


def test_missing_position_rule_implementation_rejects_before_orders() -> None:
    intent = target_intent(position_rule_policy_id="stop-5")
    engine = Engine(DataFeed(prices_df=prices()), InitialTargetStrategy(intent))

    with pytest.raises(UnsupportedPreOpenPolicyError, match="no registered implementation"):
        engine.run()

    assert engine.broker.orders == []
    assert engine.broker.get_target_intents() == ()


def test_overlapping_targets_for_one_session_are_rejected_atomically() -> None:
    first = target_intent(intent_id="first")
    second = target_intent(intent_id="second")

    class OverlappingStrategy(Strategy):
        def on_prepare(self, broker, config=None) -> None:
            broker.register_target_intent(first)
            broker.register_target_intent(second)

        def on_data(self, timestamp, data, context, broker) -> None:
            return None

    engine = Engine(DataFeed(prices_df=prices()), OverlappingStrategy())

    with pytest.raises(PreOpenIntentError, match="overlaps target"):
        engine.run()

    assert engine.broker.orders == []
    assert engine.broker.get_target_intents() == ()


def test_partial_opening_fill_retains_child_remainder_and_lineage() -> None:
    config = BacktestConfig()
    policy = replace(
        default_execution_policy(config),
        liquidity_fraction=0.1,
        allow_partial_fills=True,
    )
    engine = Engine(
        DataFeed(prices_df=prices(volume=100)),
        InitialTargetStrategy(target_intent()),
        config,
        execution_policy=policy,
    )

    result = engine.run()

    assert result.fills[0].quantity == 10
    reconciliation = engine.broker.get_intent_reconciliations()[-1]
    assert reconciliation.outcome is IntentOutcome.PARTIAL
    assert reconciliation.requested_quantity == 500
    assert reconciliation.filled_quantity == 10
    assert reconciliation.remaining_quantity == 490
    assert engine.broker.pending_orders[0].child_intent_id == reconciliation.child_intent_id


def test_partial_fill_policy_rejects_before_submitting_child_order() -> None:
    config = BacktestConfig()
    policy = replace(
        default_execution_policy(config),
        liquidity_fraction=0.1,
        allow_partial_fills=False,
    )
    engine = Engine(
        DataFeed(prices_df=prices(volume=100)),
        InitialTargetStrategy(target_intent()),
        config,
        execution_policy=policy,
    )

    with pytest.raises(UnsupportedPreOpenPolicyError, match="disallows"):
        engine.run()

    assert engine.broker.orders == []


def test_execution_limit_must_match_declared_liquidity_policy() -> None:
    config = BacktestConfig()
    policy = replace(
        default_execution_policy(config),
        liquidity_fraction=0.1,
        allow_partial_fills=True,
    )
    engine = Engine(
        DataFeed(prices_df=prices(volume=100)),
        InitialTargetStrategy(target_intent()),
        config,
        execution_policy=policy,
        execution_limits=VolumeParticipationLimit(max_participation=0.2),
    )

    with pytest.raises(UnsupportedPreOpenPolicyError, match="execution limits allow 20"):
        engine.run()

    assert engine.broker.orders == []


def test_account_policy_rejection_is_reconciled_with_original_child() -> None:
    intent = target_intent(weight=-0.5)
    engine = Engine(DataFeed(prices_df=prices()), InitialTargetStrategy(intent))

    engine.run()

    reconciliation = engine.broker.get_intent_reconciliations()[-1]
    assert reconciliation.outcome is IntentOutcome.REJECTED
    assert reconciliation.filled_quantity == 0
    assert reconciliation.remaining_quantity == 500
    assert reconciliation.rejection_reason is not None
    assert engine.broker.orders[0].child_intent_id == reconciliation.child_intent_id


def test_cancelled_partial_remainder_is_reconciled_before_run_end() -> None:
    config = BacktestConfig()
    policy = replace(
        default_execution_policy(config),
        liquidity_fraction=0.1,
        allow_partial_fills=True,
    )
    intent = target_intent()

    class CancelRemainderStrategy(InitialTargetStrategy):
        def on_data(self, timestamp, data, context, broker) -> None:
            pending = broker.get_pending_orders("SPY")
            assert len(pending) == 1
            broker.cancel_order(pending[0].order_id)

    engine = Engine(
        DataFeed(prices_df=prices(volume=100)),
        CancelRemainderStrategy(intent),
        config,
        execution_policy=policy,
    )

    engine.run()

    outcomes = [record.outcome for record in engine.broker.get_intent_reconciliations()]
    assert outcomes == [IntentOutcome.PARTIAL, IntentOutcome.CANCELLED]
    assert engine.broker.get_intent_reconciliations()[-1].remaining_quantity == 490


def test_percentage_fees_are_reserved_during_opening_weight_sizing() -> None:
    config = BacktestConfig(
        commission_type=CommissionType.PERCENTAGE,
        commission_rate=0.01,
    )
    engine = Engine(
        DataFeed(prices_df=prices(low=100.0)),
        InitialTargetStrategy(target_intent(weight=1.0)),
        config,
    )

    result = engine.run()

    assert result.fills[0].quantity == 990
    assert result.fills[0].commission == 990
    assert engine.broker.cash == 10


def test_fixed_trade_fee_is_reserved_during_opening_weight_sizing() -> None:
    config = BacktestConfig(
        commission_type=CommissionType.PER_TRADE,
        commission_per_trade=5.0,
    )

    engine = Engine(
        DataFeed(prices_df=prices(low=100.0)),
        InitialTargetStrategy(target_intent(weight=1.0)),
        config,
    )

    result = engine.run()

    assert result.fills[0].quantity == 999
    assert result.fills[0].commission == 5
    assert engine.broker.cash == 95


def test_fixed_slippage_is_reserved_during_opening_weight_sizing() -> None:
    config = BacktestConfig(
        slippage_type=SlippageType.FIXED,
        slippage_fixed=0.25,
    )
    engine = Engine(
        DataFeed(prices_df=prices(low=100.0)),
        InitialTargetStrategy(target_intent(weight=1.0)),
        config,
    )

    result = engine.run()

    assert result.fills[0].quantity == 997
    assert result.fills[0].price == 100.25
    assert engine.broker.cash == pytest.approx(50.75)


def test_opening_target_uses_open_independently_of_ordinary_execution_price() -> None:
    config = BacktestConfig(execution_price=ExecutionPrice.CLOSE)

    result = Engine(
        DataFeed(prices_df=prices(low=100.0)),
        InitialTargetStrategy(target_intent()),
        config,
    ).run()

    assert result.fills[0].price == 100
    assert result.fills[0].price_source == "open"


def test_reject_residual_policy_fails_before_child_order() -> None:
    intent = target_intent(
        weight=0.33333,
        residual=ResidualPolicy.REJECT,
    )
    engine = Engine(DataFeed(prices_df=prices()), InitialTargetStrategy(intent))

    with pytest.raises(PreOpenIntentError, match="rounding residual"):
        engine.run()

    assert engine.broker.orders == []


def test_restart_state_preserves_partial_remainder_and_idempotency() -> None:
    config = BacktestConfig()
    policy = replace(
        default_execution_policy(config),
        liquidity_fraction=0.1,
        allow_partial_fills=True,
    )
    intent = target_intent()
    engine = Engine(
        DataFeed(prices_df=prices(volume=100)),
        InitialTargetStrategy(intent),
        config,
        execution_policy=policy,
    )
    engine.run()
    state = engine.broker.export_target_intent_state()
    original_order_count = len(engine.broker.orders)

    restored = PreOpenTargetManager(
        engine.broker,
        policy,
        engine.lifecycle_version,
        calendar=None,
    )
    engine.broker._preopen_target_manager = restored
    restored.restore_state(state)
    assert restored.register(intent) is restored.targets[0]
    restored.reconcile(datetime(2026, 8, 3))

    assert len(engine.broker.orders) == original_order_count
    assert restored.reconciliations[-1].remaining_quantity == 490


def test_restart_registration_reattaches_position_rule_implementation() -> None:
    intent = target_intent(position_rule_policy_id="stop-5")
    engine = Engine(
        DataFeed(prices_df=prices()),
        InitialTargetStrategy(intent, StopLoss(0.05)),
    )
    engine.run()
    state = engine.preopen_target_manager.to_state()
    restored = PreOpenTargetManager(
        engine.broker,
        engine.execution_policy,
        engine.lifecycle_version,
        calendar=None,
    )

    restored.restore_state(state)
    assert restored.register(intent, position_rules=StopLoss(0.05)) is restored.targets[0]


def test_result_artifact_round_trip_retains_contract_and_intent_evidence(tmp_path) -> None:
    intent = target_intent()
    result = Engine(
        DataFeed(prices_df=prices(low=100.0)),
        InitialTargetStrategy(intent),
    ).run()

    result.to_parquet(tmp_path)
    loaded = BacktestResult.from_parquet(tmp_path)

    assert loaded.metrics["lifecycle_version"] == "1"
    assert loaded.metrics["execution_policy"] == result.metrics["execution_policy"]
    assert loaded.metrics["target_intents"] == [intent.to_dict()]
    assert loaded.metrics["child_order_intents"] == result.metrics["child_order_intents"]
    assert loaded.metrics["intent_reconciliations"] == result.metrics["intent_reconciliations"]


def test_ambiguous_opening_bar_path_requires_declared_resolution() -> None:
    config = BacktestConfig(trail_stop_timing=TrailStopTiming.INTRABAR)
    policy = replace(
        default_execution_policy(config),
        bar_path=BarPathPolicy.REJECT_AMBIGUOUS,
    )
    intent = target_intent(position_rule_policy_id="trail-5")
    engine = Engine(
        DataFeed(prices_df=prices(low=100.0)),
        InitialTargetStrategy(intent, TrailingStop(0.05)),
        config,
        execution_policy=policy,
    )

    with pytest.raises(AmbiguousBarPathError, match="depends on daily high-low order"):
        engine.run()


@pytest.mark.parametrize(
    ("bar_path", "expected_fill_count"),
    [
        (BarPathPolicy.OPEN_HIGH_LOW_CLOSE, 2),
        (BarPathPolicy.OPEN_LOW_HIGH_CLOSE, 1),
    ],
)
def test_explicit_opening_bar_path_controls_trailing_result(
    bar_path: BarPathPolicy,
    expected_fill_count: int,
) -> None:
    config = BacktestConfig(trail_stop_timing=TrailStopTiming.INTRABAR)
    policy = replace(default_execution_policy(config), bar_path=bar_path)
    intent = target_intent(position_rule_policy_id="trail-5")

    result = Engine(
        DataFeed(prices_df=prices(low=100.0)),
        InitialTargetStrategy(intent, TrailingStop(0.05)),
        config,
        execution_policy=policy,
    ).run()

    assert len(result.fills) == expected_fill_count
