from __future__ import annotations

from dataclasses import fields, replace
from datetime import UTC, date, datetime, timedelta

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
    AssetClass,
    BacktestConfig,
    BacktestResult,
    Broker,
    ContractSpec,
    DataFeed,
    Engine,
    ExecutionMode,
    IntentOutcome,
    LateAuctionIntentError,
    Position,
    PreOpenIntentError,
    Strategy,
    UnsupportedPreOpenPolicyError,
    default_execution_policy,
)
from ml4t.backtest.config import (
    CommissionType,
    ExecutionPrice,
    FillOrdering,
    ShareType,
    SlippageType,
    TrailStopTiming,
)
from ml4t.backtest.execution import VolumeParticipationLimit
from ml4t.backtest.risk.position import StopLoss, TrailingStop
from ml4t.backtest.types import OrderType, StopFillMode


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
    decision_time: datetime = datetime(2026, 8, 2, 23, tzinfo=UTC),
    weight: float = 0.5,
    rounding: RoundingPolicy = RoundingPolicy.TOWARD_ZERO,
    residual: ResidualPolicy = ResidualPolicy.KEEP_CASH,
    position_rule_policy_id: str | None = None,
) -> CanonicalTargetIntent:
    return CanonicalTargetIntent(
        intent_id=intent_id,
        decision_time=decision_time,
        information_cutoff=decision_time,
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


def portfolio_intent(
    intent_id: str,
    session: date,
    targets: tuple[tuple[str, float, str | None], ...],
) -> CanonicalTargetIntent:
    decision_time = datetime.combine(session, datetime.min.time(), tzinfo=UTC) - timedelta(hours=1)
    return CanonicalTargetIntent(
        intent_id=intent_id,
        decision_time=decision_time,
        information_cutoff=decision_time,
        effective_session=session,
        effective_phase=LifecyclePhase.PRE_OPEN,
        targets=tuple(
            AssetTarget(
                asset,
                TargetMeasure.WEIGHT,
                weight,
                position_rule_policy_id=policy_id,
            )
            for asset, weight, policy_id in targets
        ),
        idempotency_key=f"{intent_id}-key",
        measure=TargetMeasure.WEIGHT,
        cash_buffer=0.0,
        rounding=RoundingPolicy.TOWARD_ZERO,
        residual=ResidualPolicy.KEEP_CASH,
        reason=IntentReason.REBALANCE,
    )


def portfolio_prices(
    sessions: tuple[date, ...],
    assets: tuple[str, ...],
    *,
    overrides: dict[tuple[date, str], dict[str, float]] | None = None,
) -> pl.DataFrame:
    overrides = overrides or {}
    rows = []
    for session in sessions:
        for asset in assets:
            row = {
                "timestamp": datetime.combine(session, datetime.min.time()),
                "asset": asset,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1_000_000.0,
            }
            row.update(overrides.get((session, asset), {}))
            rows.append(row)
    return pl.DataFrame(rows)


class InitialTargetStrategy(Strategy):
    def __init__(self, intent: CanonicalTargetIntent, rules=None) -> None:
        self.intent = intent
        self.rules = rules

    def on_prepare(self, broker, config=None) -> None:
        broker.register_target_intent(self.intent, position_rules=self.rules)

    def on_data(self, timestamp, data, context, broker) -> None:
        return None


class NoOpStrategy(Strategy):
    def on_data(self, timestamp, data, context, broker) -> None:
        return None


def test_target_intent_api_requires_an_engine_configured_broker() -> None:
    broker = Broker()

    assert broker.get_target_intents() == ()
    assert broker.get_child_order_intents() == ()
    assert broker.get_intent_reconciliations() == ()
    assert broker.export_target_intent_state() == {}
    with pytest.raises(RuntimeError, match="Engine-configured broker"):
        broker.register_target_intent(target_intent())
    with pytest.raises(RuntimeError, match="Engine-configured broker"):
        broker.register_position_rule_policy("stop-5", StopLoss(0.05))
    with pytest.raises(RuntimeError, match="Engine-configured broker"):
        broker.restore_target_intent_state({})


def test_opening_processing_skips_session_resolution_without_outstanding_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = Engine(DataFeed(prices_df=prices(bars=2)), InitialTargetStrategy(target_intent()))
    resolved_sessions = []
    resolve_session = engine.preopen_target_manager._session_date

    def track_session_resolution(timestamp):
        resolved_sessions.append(timestamp)
        return resolve_session(timestamp)

    monkeypatch.setattr(
        engine.preopen_target_manager,
        "_session_date",
        track_session_resolution,
    )

    result = engine.run()

    assert result.metrics["target_intent_count"] == 1
    assert result.metrics["intent_reconciliation_count"] == 1
    assert resolved_sessions == [datetime(2026, 8, 3)]


@pytest.mark.parametrize(
    "execution_mode,target_session,lows",
    [
        (ExecutionMode.SAME_BAR, date(2026, 8, 5), [100.0, 90.0, 100.0]),
        (ExecutionMode.NEXT_BAR, date(2026, 8, 6), [100.0, 100.0, 90.0, 100.0]),
    ],
)
def test_deferred_risk_exit_fills_before_same_asset_opening_target(
    execution_mode: ExecutionMode,
    target_session: date,
    lows: list[float],
) -> None:
    class DeferredExitTargetStrategy(Strategy):
        def on_prepare(self, broker, config=None) -> None:
            broker.set_position_rules(StopLoss(0.05))
            broker.register_target_intent(target_intent(session=target_session, weight=0.15))

        def on_data(self, timestamp, data, context, broker) -> None:
            if timestamp == datetime(2026, 8, 3):
                broker.submit_order("SPY", 100)

    timestamps = [datetime(2026, 8, 3 + offset) for offset in range(len(lows))]
    feed = DataFeed(
        prices_df=pl.DataFrame(
            {
                "timestamp": timestamps,
                "asset": ["SPY"] * len(lows),
                "open": [100.0] * len(lows),
                "high": [100.0] * len(lows),
                "low": lows,
                "close": [100.0] * len(lows),
                "volume": [1_000_000.0] * len(lows),
            }
        )
    )

    engine = Engine(
        feed,
        DeferredExitTargetStrategy(),
        BacktestConfig(
            execution_mode=execution_mode,
            stop_fill_mode=StopFillMode.NEXT_BAR_OPEN,
        ),
    )

    result = engine.run()

    assert result.metrics["intent_reconciliation_count"] == 1
    assert engine.broker.get_position("SPY").quantity == 150
    assert result.metrics["final_value"] == pytest.approx(100_000.0)
    assert result.metrics["num_fills"] == 3


def test_deferred_risk_exit_precedes_fifo_entry_and_releases_cash() -> None:
    class DeferredExitBeforeEntryStrategy(Strategy):
        def on_prepare(self, broker, config=None) -> None:
            broker.set_position_rules(StopLoss(0.05), asset="SPY")

        def on_data(self, timestamp, data, context, broker) -> None:
            if timestamp == datetime(2026, 8, 3):
                broker.submit_order("SPY", 100)
            elif timestamp == datetime(2026, 8, 5):
                broker.submit_order("QQQ", 100)

    timestamps = [datetime(2026, 8, day) for day in range(3, 7)]
    rows = [
        {
            "timestamp": timestamp,
            "asset": asset,
            "open": 100.0,
            "high": 100.0,
            "low": 90.0 if asset == "SPY" and timestamp.day == 5 else 100.0,
            "close": 100.0,
            "volume": 1_000_000.0,
        }
        for timestamp in timestamps
        for asset in ("SPY", "QQQ")
    ]
    engine = Engine(
        DataFeed(prices_df=pl.DataFrame(rows)),
        DeferredExitBeforeEntryStrategy(),
        BacktestConfig(
            initial_cash=10_000.0,
            execution_mode=ExecutionMode.NEXT_BAR,
            stop_fill_mode=StopFillMode.NEXT_BAR_OPEN,
            fill_ordering=FillOrdering.FIFO,
        ),
    )

    result = engine.run()

    assert [fill.asset for fill in result.fills] == ["SPY", "SPY", "QQQ"]
    assert engine.broker.get_position("SPY") is None
    assert engine.broker.get_position("QQQ").quantity == 100


def test_failed_prepare_rolls_back_target_and_position_rule_registration() -> None:
    intent = target_intent(position_rule_policy_id="stop-5")

    class FailingPrepare(Strategy):
        def on_prepare(self, broker, config=None) -> None:
            broker.register_target_intent(intent, position_rules=StopLoss(0.05))
            raise RuntimeError("prepare failed")

        def on_data(self, timestamp, data, context, broker) -> None:
            return None

    engine = Engine(DataFeed(prices_df=prices()), FailingPrepare())

    with pytest.raises(RuntimeError, match="prepare failed"):
        engine.run()

    assert engine.broker.get_target_intents() == ()
    assert engine.broker.get_child_order_intents() == ()
    assert engine.broker.get_intent_reconciliations() == ()
    assert engine.preopen_target_manager._position_rules == {}


def test_callback_rollback_does_not_run_public_restart_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intent = target_intent()

    class FailingPrepare(Strategy):
        def on_prepare(self, broker, config=None) -> None:
            broker.register_target_intent(intent)
            raise RuntimeError("prepare failed")

        def on_data(self, timestamp, data, context, broker) -> None:
            return None

    engine = Engine(DataFeed(prices_df=prices()), FailingPrepare())

    def fail_restart_validation(state):
        raise AssertionError("transaction rollback must not call restore_state")

    monkeypatch.setattr(engine.preopen_target_manager, "restore_state", fail_restart_validation)

    with pytest.raises(RuntimeError, match="prepare failed"):
        engine.run()

    assert engine.broker.get_target_intents() == ()


def test_preopen_transaction_checkpoint_retains_only_collection_lengths() -> None:
    engine = Engine(DataFeed(prices_df=prices()), InitialTargetStrategy(target_intent()))
    engine.preopen_target_manager.register(target_intent())

    state = engine.preopen_target_manager.capture_transaction_state()

    assert all(field.name.endswith("_length") for field in fields(state))
    assert all(isinstance(getattr(state, field.name), int) for field in fields(state))


def test_target_intent_state_restore_is_rejected_inside_lifecycle_callback() -> None:
    class RestoreDuringPrepare(Strategy):
        def on_prepare(self, broker, config=None) -> None:
            broker.restore_target_intent_state({})

        def on_data(self, timestamp, data, context, broker) -> None:
            return None

    engine = Engine(DataFeed(prices_df=prices()), RestoreDuringPrepare())

    with pytest.raises(RuntimeError, match="cannot be restored during a lifecycle callback"):
        engine.run()


def test_opening_target_registration_from_on_start_names_on_prepare_migration() -> None:
    intent = target_intent()

    class StartTarget(Strategy):
        def on_start(self, broker) -> None:
            broker.register_target_intent(intent)

        def on_data(self, timestamp, data, context, broker) -> None:
            return None

    engine = Engine(DataFeed(prices_df=prices()), StartTarget())

    with pytest.raises(LateAuctionIntentError, match="register opening targets in on_prepare"):
        engine.run()

    assert engine.broker.get_target_intents() == ()


def test_initial_weight_target_lowers_at_open_and_rules_see_only_later_movement() -> None:
    intent = target_intent(position_rule_policy_id="stop-5")
    engine = Engine(
        DataFeed(prices_df=prices()),
        InitialTargetStrategy(intent, StopLoss(0.05)),
        BacktestConfig(retain_intent_history=True),
    )

    result = engine.run()

    child = engine.broker.get_child_order_intents()[0]
    assert child.quantity == 500
    assert child.decision_session == intent.effective_session
    assert child.effective_session == intent.effective_session
    assert child.eligibility_phase is LifecyclePhase.PRE_OPEN
    assert [fill.price for fill in result.fills] == [100.0, 95.0]
    assert result.fills[0].target_intent_id == intent.intent_id
    assert result.fills[0].child_intent_id == child.child_intent_id
    reconciliation = engine.broker.get_intent_reconciliations()[0]
    assert reconciliation.outcome is IntentOutcome.FULL
    assert reconciliation.rule_policy_id == "stop-5"
    assert reconciliation.rule_activated_at == datetime(2026, 8, 3, tzinfo=UTC)
    assert {record.rule_activated_at for record in engine.broker.get_intent_reconciliations()} == {
        datetime(2026, 8, 3, tzinfo=UTC)
    }
    assert result.metrics["lifecycle_version"] == "1"
    assert result.metrics["execution_policy"]["bar_path"] == "conservative"
    assert result.metrics["target_intents"] == [intent.to_dict()]
    assert result.metrics["child_order_intents"] == [child.to_dict()]


def test_opening_target_fills_only_its_child_orders_before_the_regular_queue() -> None:
    intent = target_intent()

    class TargetWithRestingOrder(InitialTargetStrategy):
        def on_start(self, broker) -> None:
            broker.submit_order("SPY", 1, order_type=OrderType.LIMIT, limit_price=110.0)

    config = BacktestConfig(next_bar_queue_shadow_validation=True)
    engine = Engine(
        DataFeed(prices_df=prices()),
        TargetWithRestingOrder(intent),
        config,
    )

    result = engine.run()

    assert [fill.child_intent_id for fill in result.fills] == [
        f"{intent.intent_id}:SPY",
        None,
    ]
    assert engine.broker.get_pending_orders("SPY") == []


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


def test_opening_target_marks_existing_non_target_position_at_open() -> None:
    scheduled = target_intent(intent_id="scheduled", session=date(2026, 8, 5))

    class ExistingPositionStrategy(Strategy):
        def on_data(self, timestamp, data, context, broker) -> None:
            if timestamp.date() == date(2026, 8, 3):
                broker.submit_order("HELD", 100)
            elif timestamp.date() == date(2026, 8, 4):
                broker.register_target_intent(scheduled)

    rows = []
    for timestamp in (datetime(2026, 8, 3), datetime(2026, 8, 4), datetime(2026, 8, 5)):
        for asset in ("SPY", "HELD"):
            close = 200.0 if asset == "HELD" and timestamp.date() == date(2026, 8, 5) else 100.0
            rows.append(
                {
                    "timestamp": timestamp,
                    "asset": asset,
                    "open": 100.0,
                    "high": max(100.0, close),
                    "low": 100.0,
                    "close": close,
                    "volume": 1_000_000.0,
                }
            )

    engine = Engine(DataFeed(prices_df=pl.DataFrame(rows)), ExistingPositionStrategy())
    engine.run()

    child = engine.broker.get_child_order_intents()[0]
    assert child.asset == "SPY"
    assert child.quantity == 500


def test_opening_target_uses_exchange_session_date_across_utc_calendar_boundary() -> None:
    intent = target_intent(
        session=date(2024, 1, 8),
        decision_time=datetime(2024, 1, 7, 22, tzinfo=UTC),
    )
    frame = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 7, 23)],
            "asset": ["SPY"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1_000_000.0],
        }
    )
    feed = DataFeed(
        prices_df=frame,
        feed_spec={
            "calendar": "CME_Equity",
            "timezone": "UTC",
            "data_frequency": "1m",
            "timestamp_semantics": "bar_close",
            "session_start_time": "17:00",
        },
    )

    result = Engine(
        feed,
        InitialTargetStrategy(intent),
        BacktestConfig(data_frequency="1m"),
    ).run()

    assert len(result.fills) == 1
    assert result.fills[0].timestamp == datetime(2024, 1, 7, 23)


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
        market_fill_phase=LifecyclePhase.MARKET_EVENT,
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


def test_rejected_overlapping_target_does_not_leak_its_rule_policy() -> None:
    first = target_intent(intent_id="first")
    overlapping = target_intent(
        intent_id="overlapping",
        position_rule_policy_id="leaked-stop",
    )
    later = target_intent(
        intent_id="later",
        session=date(2026, 8, 4),
        position_rule_policy_id="leaked-stop",
    )

    class CaughtOverlapStrategy(Strategy):
        def on_prepare(self, broker, config=None) -> None:
            broker.register_target_intent(first)
            with pytest.raises(PreOpenIntentError, match="overlaps target"):
                broker.register_target_intent(overlapping, position_rules=StopLoss(0.05))
            broker.register_target_intent(later)

        def on_data(self, timestamp, data, context, broker) -> None:
            return None

    engine = Engine(DataFeed(prices_df=prices(bars=2)), CaughtOverlapStrategy())

    with pytest.raises(UnsupportedPreOpenPolicyError, match="no registered implementation"):
        engine.run()

    assert engine.preopen_target_manager._position_rules == {}


def test_later_target_replaces_the_active_position_rule_policy() -> None:
    first = target_intent(
        intent_id="first",
        weight=0.5,
        position_rule_policy_id="stop-50",
    )
    second = target_intent(
        intent_id="second",
        session=date(2026, 8, 4),
        weight=0.6,
        position_rule_policy_id="stop-20",
    )

    class ReplacingRuleStrategy(Strategy):
        def on_prepare(self, broker, config=None) -> None:
            broker.register_target_intent(first, position_rules=StopLoss(0.5))
            broker.register_target_intent(second, position_rules=StopLoss(0.2))

        def on_data(self, timestamp, data, context, broker) -> None:
            return None

    engine = Engine(
        DataFeed(prices_df=prices(bars=2, low=100.0)),
        ReplacingRuleStrategy(),
    )

    engine.run()

    assert engine.broker._position_rules_by_asset["SPY"] == StopLoss(0.2)
    position = engine.broker.get_position("SPY")
    assert position is not None
    assert position.context["position_rule_policy_id"] == "stop-20"
    reconciliations = engine.broker.get_intent_reconciliations()
    assert [record.rule_policy_id for record in reconciliations] == ["stop-50", "stop-20"]
    assert [record.rule_activated_at for record in reconciliations] == [
        datetime(2026, 8, 3, tzinfo=UTC),
        datetime(2026, 8, 4, tzinfo=UTC),
    ]


def test_later_target_without_a_rule_policy_clears_the_target_managed_rule() -> None:
    first = target_intent(
        intent_id="first",
        weight=0.5,
        position_rule_policy_id="stop-50",
    )
    second = target_intent(
        intent_id="second",
        session=date(2026, 8, 4),
        weight=0.6,
    )

    class ClearingRuleStrategy(Strategy):
        def on_prepare(self, broker, config=None) -> None:
            broker.register_target_intent(first, position_rules=StopLoss(0.5))
            broker.register_target_intent(second)

        def on_data(self, timestamp, data, context, broker) -> None:
            return None

    engine = Engine(
        DataFeed(prices_df=prices(bars=2, low=100.0)),
        ClearingRuleStrategy(),
    )

    engine.run()

    assert "SPY" not in engine.broker._position_rules_by_asset
    position = engine.broker.get_position("SPY")
    assert position is not None
    assert "position_rule_policy_id" not in position.context
    reconciliations = engine.broker.get_intent_reconciliations()
    assert [record.rule_policy_id for record in reconciliations] == ["stop-50", None]


def test_one_target_intent_activates_distinct_asset_rule_policies_on_entry_bar() -> None:
    session = date(2026, 8, 3)
    intent = portfolio_intent(
        "mixed-book",
        session,
        (
            ("SPY", 0.25, "stop-5"),
            ("QQQ", 0.25, "trail-5"),
            ("IWM", 0.25, None),
        ),
    )
    feed = portfolio_prices(
        (session,),
        ("SPY", "QQQ", "IWM"),
        overrides={
            (session, "SPY"): {"high": 102.0, "low": 94.0},
            (session, "QQQ"): {"high": 110.0, "low": 100.0, "close": 105.0},
        },
    )

    class MixedRuleStrategy(Strategy):
        def on_prepare(self, broker, config=None) -> None:
            broker.register_target_intent(
                intent,
                position_rules={
                    "stop-5": StopLoss(0.05),
                    "trail-5": TrailingStop(0.05),
                },
            )

        def on_data(self, timestamp, data, context, broker) -> None:
            return None

    engine = Engine(
        DataFeed(prices_df=feed),
        MixedRuleStrategy(),
        BacktestConfig(retain_intent_history=True),
    )

    result = engine.run()

    assert engine.broker.get_target_intents() == (intent,)
    assert {trade.exit_reason for trade in result.trades} == {"stop_loss", "trailing_stop"}
    assert engine.broker.get_position("IWM") is not None
    assert "IWM" not in engine.broker._position_rules_by_asset
    assert [record.asset for record in engine.broker.get_target_rule_reconciliations()] == [
        "IWM",
        "QQQ",
        "SPY",
    ]


def test_unchanged_carried_position_preserves_replaces_and_clears_rule_policy() -> None:
    sessions = tuple(date(2026, 8, day) for day in range(3, 7))
    intents = (
        portfolio_intent("initial", sessions[0], (("SPY", 0.5, "stop-50"),)),
        portfolio_intent("preserve", sessions[1], (("SPY", 0.5, "stop-50"),)),
        portfolio_intent("replace", sessions[2], (("SPY", 0.5, "stop-20"),)),
        portfolio_intent("clear", sessions[3], (("SPY", 0.5, None),)),
    )

    class CarriedRuleStrategy(Strategy):
        def __init__(self) -> None:
            self.snapshots = []

        def on_prepare(self, broker, config=None) -> None:
            broker.register_target_intent(intents[0], position_rules={"stop-50": StopLoss(0.5)})
            broker.register_target_intent(intents[1], position_rules={"stop-50": StopLoss(0.5)})
            broker.register_target_intent(intents[2], position_rules={"stop-20": StopLoss(0.2)})
            broker.register_target_intent(intents[3])

        def on_data(self, timestamp, data, context, broker) -> None:
            position = broker.get_position("SPY")
            assert position is not None
            self.snapshots.append(
                (
                    broker._get_position_rule_override("SPY"),
                    dict(position.context),
                )
            )

    strategy = CarriedRuleStrategy()
    engine = Engine(
        DataFeed(prices_df=portfolio_prices(sessions, ("SPY",))),
        strategy,
        BacktestConfig(retain_intent_history=True),
    )

    engine.run()

    assert len(engine.broker.get_child_order_intents()) == 1
    assert strategy.snapshots[0][0] is strategy.snapshots[1][0]
    assert strategy.snapshots[2][0] is not strategy.snapshots[1][0]
    assert strategy.snapshots[3][0] is None
    assert (
        strategy.snapshots[0][1]["rule_activation_time"]
        == strategy.snapshots[1][1]["rule_activation_time"]
    )
    assert (
        strategy.snapshots[2][1]["rule_activation_time"]
        != strategy.snapshots[1][1]["rule_activation_time"]
    )
    assert "rule_activation_time" not in strategy.snapshots[3][1]
    assert [record.outcome.value for record in engine.broker.get_target_rule_reconciliations()] == [
        "activated",
        "preserved",
        "activated",
        "cleared",
    ]


@pytest.mark.parametrize(
    ("position_rules", "message"),
    [
        (None, "no registered implementation"),
        (
            {"stop-5": StopLoss(0.05), "unused": StopLoss(0.1)},
            "unreferenced.*unused",
        ),
    ],
)
def test_target_level_policy_registration_rejects_atomically(
    position_rules,
    message: str,
) -> None:
    intent = portfolio_intent(
        "atomic",
        date(2026, 8, 3),
        (("SPY", 0.5, "stop-5"),),
    )
    engine = Engine(DataFeed(prices_df=prices(low=100.0)), NoOpStrategy())
    manager = engine.preopen_target_manager
    before = (
        dict(manager._targets),
        dict(manager._target_by_session_asset),
        dict(manager._idempotency),
        dict(manager._position_rules),
    )

    with pytest.raises(PreOpenIntentError, match=message):
        manager.register(intent, position_rules=position_rules)

    assert (
        manager._targets,
        manager._target_by_session_asset,
        manager._idempotency,
        manager._position_rules,
    ) == before


def test_conflicting_target_level_policy_registration_is_atomic() -> None:
    intent = portfolio_intent(
        "conflict",
        date(2026, 8, 3),
        (("SPY", 0.5, "stop-5"),),
    )
    engine = Engine(DataFeed(prices_df=prices(low=100.0)), NoOpStrategy())
    manager = engine.preopen_target_manager
    manager.register_position_rule_policy("stop-5", StopLoss(0.05))
    before = (
        dict(manager._targets),
        dict(manager._target_by_session_asset),
        dict(manager._idempotency),
        dict(manager._position_rules),
    )

    with pytest.raises(PreOpenIntentError, match="already registered"):
        manager.register(intent, position_rules={"stop-5": StopLoss(0.1)})

    assert (
        manager._targets,
        manager._target_by_session_asset,
        manager._idempotency,
        manager._position_rules,
    ) == before


def test_target_managed_rule_does_not_apply_after_flat_and_plain_reentry() -> None:
    intent = target_intent(position_rule_policy_id="stop-50")

    class CloseThenReopenStrategy(Strategy):
        def on_prepare(self, broker, config=None) -> None:
            broker.register_target_intent(intent, position_rules=StopLoss(0.5))

        def on_data(self, timestamp, data, context, broker) -> None:
            if timestamp == datetime(2026, 8, 3):
                position = broker.get_position("SPY")
                assert position is not None
                broker.submit_order("SPY", -position.quantity)
            elif timestamp == datetime(2026, 8, 4):
                broker.submit_order("SPY", 1)

    engine = Engine(
        DataFeed(prices_df=prices(bars=2, low=100.0)),
        CloseThenReopenStrategy(),
        BacktestConfig(execution_mode=ExecutionMode.SAME_BAR),
    )

    engine.run()

    position = engine.broker.get_position("SPY")
    assert position is not None
    assert position.quantity == 1
    assert "SPY" not in engine.broker._position_rules_by_asset


@pytest.mark.parametrize(
    ("disable_rules", "replacement"),
    [(False, StopLoss(0.25)), (True, None)],
)
def test_user_rule_override_survives_target_managed_cleanup(
    disable_rules: bool,
    replacement: StopLoss | None,
) -> None:
    intent = target_intent(position_rule_policy_id="stop-50")

    class OverrideThenReopenStrategy(Strategy):
        def on_prepare(self, broker, config=None) -> None:
            broker.register_target_intent(intent, position_rules=StopLoss(0.5))

        def on_data(self, timestamp, data, context, broker) -> None:
            if timestamp == datetime(2026, 8, 3):
                if disable_rules:
                    broker.clear_position_rules(asset="SPY")
                else:
                    broker.set_position_rules(replacement, asset="SPY")
                position = broker.get_position("SPY")
                assert position is not None
                broker.submit_order("SPY", -position.quantity)
            elif timestamp == datetime(2026, 8, 4):
                broker.submit_order("SPY", 1)

    engine = Engine(
        DataFeed(prices_df=prices(bars=2, low=100.0)),
        OverrideThenReopenStrategy(),
        BacktestConfig(execution_mode=ExecutionMode.SAME_BAR),
    )

    engine.run()

    assert "SPY" in engine.broker._position_rules_by_asset
    assert engine.broker._position_rules_by_asset["SPY"] == replacement


def test_partial_opening_fill_cancels_opg_remainder_and_retains_lineage() -> None:
    config = BacktestConfig()
    policy = replace(
        default_execution_policy(config),
        liquidity_fraction=0.1,
        allow_partial_fills=True,
    )
    engine = Engine(
        DataFeed(prices_df=prices(bars=2, volume=100)),
        InitialTargetStrategy(target_intent()),
        config,
        execution_policy=policy,
    )

    result = engine.run()

    assert result.fills[0].quantity == 10
    assert len(result.fills) == 1
    reconciliation = engine.broker.get_intent_reconciliations()[0]
    assert reconciliation.outcome is IntentOutcome.PARTIAL
    assert reconciliation.requested_quantity == 500
    assert reconciliation.filled_quantity == 10
    assert reconciliation.remaining_quantity == 490
    assert engine.broker.pending_orders == []
    assert engine.broker._order_state.partial_quantities == {}


def test_opening_reconciliation_rejects_a_nonterminal_opg_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr(engine.broker, "cancel_order", lambda _: False)

    with pytest.raises(RuntimeError, match="must be terminal after the OPG cancel sweep"):
        engine.run()


def test_non_integral_liquidity_limit_is_rounded_consistently() -> None:
    config = BacktestConfig()
    policy = replace(
        default_execution_policy(config),
        liquidity_fraction=0.1,
        allow_partial_fills=True,
    )
    engine = Engine(
        DataFeed(prices_df=prices(volume=1005)),
        InitialTargetStrategy(target_intent()),
        config,
        execution_policy=policy,
    )

    result = engine.run()

    assert result.fills[0].quantity == 100
    assert engine.broker.get_intent_reconciliations()[0].outcome is IntentOutcome.PARTIAL


def test_default_policy_does_not_cap_opening_fill_at_bar_volume() -> None:
    result = Engine(
        DataFeed(prices_df=prices(volume=100)),
        InitialTargetStrategy(target_intent()),
    ).run()

    assert result.fills[0].quantity == 500
    assert result.metrics["intent_reconciliation_count"] == 1
    assert result.metrics["target_intent_count"] == 1


def test_opening_liquidity_policy_does_not_cap_regular_strategy_orders() -> None:
    class RegularOrderStrategy(Strategy):
        def __init__(self) -> None:
            self.submitted = False

        def on_data(self, timestamp, data, context, broker) -> None:
            if not self.submitted:
                broker.submit_order("SPY", 500)
                self.submitted = True

    config = BacktestConfig()
    policy = replace(
        default_execution_policy(config),
        liquidity_fraction=0.1,
        allow_partial_fills=True,
    )

    result = Engine(
        DataFeed(prices_df=prices(bars=2, volume=100)),
        RegularOrderStrategy(),
        config,
        execution_policy=policy,
    ).run()

    assert [fill.quantity for fill in result.fills] == [500]


def test_immediate_same_bar_mode_still_executes_opening_child_at_open() -> None:
    config = BacktestConfig(
        execution_price=ExecutionPrice.CLOSE,
        execution_mode=ExecutionMode.SAME_BAR,
        immediate_fill=True,
    )

    result = Engine(
        DataFeed(prices_df=prices(low=100.0)),
        InitialTargetStrategy(target_intent()),
        config,
    ).run()

    assert result.fills[0].price == 100
    assert result.fills[0].price_source == "open"


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


def test_terminal_reconciliation_is_not_duplicated_on_later_bars() -> None:
    config = BacktestConfig()
    policy = replace(
        default_execution_policy(config),
        liquidity_fraction=0.1,
        allow_partial_fills=True,
    )
    intent = target_intent()

    engine = Engine(
        DataFeed(prices_df=prices(bars=10, volume=100)),
        InitialTargetStrategy(intent),
        config,
        execution_policy=policy,
    )

    engine.run()

    reconciliations = engine.broker.get_intent_reconciliations()
    assert len(reconciliations) == 1
    assert reconciliations[0].outcome is IntentOutcome.PARTIAL
    assert reconciliations[0].remaining_quantity == 490


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


def test_largest_remainder_allocation_accounts_for_contract_multipliers() -> None:
    specs = {
        asset: ContractSpec(asset, AssetClass.FUTURE, multiplier=50.0) for asset in ("ES", "NQ")
    }
    engine = Engine(
        DataFeed(prices_df=prices()),
        InitialTargetStrategy(target_intent()),
        BacktestConfig(initial_cash=300_000.0),
        contract_specs=specs,
    )
    rounded = {"ES": 1.0, "NQ": 1.0}

    engine.preopen_target_manager._allocate_largest_remainders(
        {"ES": 1.6, "NQ": 1.6},
        rounded,
        {"ES": 1_000.0, "NQ": 1_000.0},
        0.0,
        "multiplier-target",
    )

    assert rounded == {"ES": 2.0, "NQ": 1.0}


def test_largest_remainder_allocation_uses_cash_released_by_position_trims() -> None:
    engine = Engine(DataFeed(prices_df=prices()), InitialTargetStrategy(target_intent()))
    engine.broker.positions.update(
        {asset: Position(asset, 10.0, 100.0, datetime(2026, 8, 1)) for asset in ("A", "B")}
    )
    engine.broker.cash = 0.0
    rounded = {"A": 4.0, "B": 4.0}

    engine.preopen_target_manager._allocate_largest_remainders(
        {"A": 4.6, "B": 4.6},
        rounded,
        {"A": 100.0, "B": 100.0},
        0.0,
        "trim-target",
    )

    assert rounded == {"A": 5.0, "B": 4.0}


@pytest.mark.parametrize("rounding", [RoundingPolicy.TOWARD_ZERO, RoundingPolicy.NONE])
def test_fractional_largest_remainder_is_rejected(rounding: RoundingPolicy) -> None:
    engine = Engine(
        DataFeed(prices_df=prices()),
        InitialTargetStrategy(
            target_intent(
                weight=0.00175,
                rounding=rounding,
                residual=ResidualPolicy.LARGEST_REMAINDER,
            )
        ),
        BacktestConfig(share_type=ShareType.FRACTIONAL),
    )

    with pytest.raises(
        UnsupportedPreOpenPolicyError,
        match="largest_remainder is unsupported with fractional shares",
    ):
        engine.run()

    assert engine.broker.orders == []


def test_fractional_largest_remainder_is_rejected_when_state_is_restored() -> None:
    intent = target_intent(residual=ResidualPolicy.LARGEST_REMAINDER)
    source = Engine(DataFeed(prices_df=prices()), InitialTargetStrategy(intent))
    source.preopen_target_manager.register(intent)
    state = source.preopen_target_manager.to_state()
    destination = Engine(
        DataFeed(prices_df=prices()),
        InitialTargetStrategy(intent),
        BacktestConfig(share_type=ShareType.FRACTIONAL),
    )

    with pytest.raises(
        UnsupportedPreOpenPolicyError,
        match=r"unsupported with fractional shares.*target 'initial-portfolio'",
    ):
        destination.broker.restore_target_intent_state(state)

    assert destination.broker.get_target_intents() == ()
    assert destination.broker.get_child_order_intents() == ()
    assert destination.broker.get_intent_reconciliations() == ()
    assert destination.preopen_target_manager._idempotency == {}


def test_fractional_accounts_cannot_enter_largest_remainder_allocator() -> None:
    intent = target_intent()
    engine = Engine(
        DataFeed(prices_df=prices()),
        InitialTargetStrategy(intent),
        BacktestConfig(share_type=ShareType.FRACTIONAL),
    )

    with pytest.raises(
        UnsupportedPreOpenPolicyError,
        match=r"target 'initial-portfolio'.*fractional shares; use keep_cash",
    ):
        engine.preopen_target_manager._allocate_largest_remainders(
            {"SPY": 1.5},
            {"SPY": 1.0},
            {"SPY": 100.0},
            0.0,
            intent.intent_id,
        )


def test_restart_state_requires_matching_broker_order_state() -> None:
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

    with pytest.raises(PreOpenIntentError, match="complete broker checkpoint"):
        Engine(
            DataFeed(prices_df=prices(volume=100)),
            InitialTargetStrategy(intent),
            config,
            execution_policy=policy,
            target_intent_state=state,
        )


def test_restart_registration_reattaches_position_rule_implementation() -> None:
    intent = target_intent(position_rule_policy_id="stop-5")
    engine = Engine(
        DataFeed(prices_df=prices()),
        InitialTargetStrategy(intent, StopLoss(0.05)),
    )
    engine.run()
    state = engine.preopen_target_manager.to_state()
    restored = engine.broker._create_preopen_target_manager(
        engine.execution_policy,
        engine.lifecycle_version,
        calendar=None,
    )

    restored.restore_state(state)
    assert restored.register(intent, position_rules=StopLoss(0.05)) is restored.targets[0]


def test_restored_manager_preserves_an_explicit_rule_disable_during_cleanup() -> None:
    intent = target_intent(position_rule_policy_id="stop-50")
    engine = Engine(
        DataFeed(prices_df=prices(low=100.0)),
        InitialTargetStrategy(intent, StopLoss(0.5)),
    )
    engine.run()
    state = engine.preopen_target_manager.to_state()
    engine.broker.clear_position_rules(asset="SPY")
    restored = engine.broker._create_preopen_target_manager(
        engine.execution_policy,
        engine.lifecycle_version,
        calendar=None,
    )
    restored.restore_state(state)

    engine.broker.positions.clear()
    restored.reconcile(datetime(2026, 8, 4))

    assert "SPY" in engine.broker._position_rules_by_asset
    assert engine.broker._position_rules_by_asset["SPY"] is None


def test_result_artifact_round_trip_retains_contract_and_intent_evidence(tmp_path) -> None:
    intent = target_intent()
    result = Engine(
        DataFeed(prices_df=prices(low=100.0)),
        InitialTargetStrategy(intent),
        BacktestConfig(retain_intent_history=True),
    ).run()

    result.to_parquet(tmp_path)
    loaded = BacktestResult.from_parquet(tmp_path)

    assert loaded.metrics["lifecycle_version"] == "1"
    assert loaded.metrics["execution_policy"] == result.metrics["execution_policy"]
    assert loaded.metrics["target_intents"] == [intent.to_dict()]
    assert loaded.metrics["child_order_intents"] == result.metrics["child_order_intents"]
    assert loaded.metrics["intent_reconciliations"] == result.metrics["intent_reconciliations"]


def test_default_result_retains_intent_counts_without_full_history(tmp_path) -> None:
    result = Engine(
        DataFeed(prices_df=prices(low=100.0)),
        InitialTargetStrategy(target_intent()),
    ).run()

    assert result.metrics["target_intent_count"] == 1
    assert result.metrics["child_order_intent_count"] == 1
    assert result.metrics["intent_reconciliation_count"] == 1
    assert result.metrics["target_intents"] == []
    assert result.metrics["child_order_intents"] == []
    assert result.metrics["intent_reconciliations"] == []
    assert result.to_spec_dict()["target_intent_count"] == 1

    result.to_parquet(tmp_path)
    loaded = BacktestResult.from_parquet(tmp_path)
    assert loaded.metrics["target_intent_count"] == 1
    assert loaded.metrics["child_order_intent_count"] == 1
    assert loaded.metrics["intent_reconciliation_count"] == 1


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


def test_reject_ambiguous_policy_accepts_identical_path_outcomes() -> None:
    policy = replace(
        default_execution_policy(BacktestConfig()),
        bar_path=BarPathPolicy.REJECT_AMBIGUOUS,
    )
    intent = target_intent(position_rule_policy_id="stop-5")

    result = Engine(
        DataFeed(prices_df=prices(low=100.0)),
        InitialTargetStrategy(intent, StopLoss(0.05)),
        execution_policy=policy,
    ).run()

    assert len(result.fills) == 1


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
