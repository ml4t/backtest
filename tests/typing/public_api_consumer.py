"""Representative typed use of the installed ml4t-backtest wheel."""

from datetime import UTC, date, datetime
from typing import Any

import polars as pl
from ml4t.specs import (
    AssetTarget,
    CanonicalTargetIntent,
    IntentReason,
    LifecyclePhase,
    ResidualPolicy,
    RoundingPolicy,
    TargetMeasure,
)

from ml4t.backtest import (
    BacktestConfig,
    BacktestResult,
    Broker,
    DataFeed,
    Engine,
    OrderSide,
    Strategy,
)
from ml4t.backtest.risk.position import StopLoss, TrailingStop


class BuyFirstAsset(Strategy):
    def on_data(
        self,
        timestamp: datetime,
        data: dict[str, dict],
        context: dict[str, Any],
        broker: Broker,
    ) -> None:
        del timestamp, context
        if data and not broker.positions:
            broker.submit_order(next(iter(data)), 1.0, side=OrderSide.BUY)


def run_typed_consumer(prices: pl.DataFrame) -> BacktestResult:
    feed = DataFeed(prices_df=prices)
    engine = Engine(feed=feed, strategy=BuyFirstAsset(), config=BacktestConfig())
    return engine.run()


def register_typed_target_rules(broker: Broker) -> None:
    decision = datetime(2026, 8, 2, 20, tzinfo=UTC)
    intent = CanonicalTargetIntent(
        intent_id="typed-mixed-rules",
        decision_time=decision,
        information_cutoff=decision,
        effective_session=date(2026, 8, 3),
        effective_phase=LifecyclePhase.PRE_OPEN,
        targets=(
            AssetTarget(
                "SPY",
                TargetMeasure.WEIGHT,
                0.45,
                position_rule_policy_id="stop-5",
            ),
            AssetTarget(
                "QQQ",
                TargetMeasure.WEIGHT,
                0.45,
                position_rule_policy_id="trail-8",
            ),
        ),
        idempotency_key="typed-mixed-rules-2026-08-03",
        measure=TargetMeasure.WEIGHT,
        cash_buffer=0.1,
        rounding=RoundingPolicy.TOWARD_ZERO,
        residual=ResidualPolicy.KEEP_CASH,
        reason=IntentReason.REBALANCE,
    )
    broker.register_target_intent(
        intent,
        position_rules={
            "stop-5": StopLoss(0.05),
            "trail-8": TrailingStop(0.08),
        },
    )
    broker.get_target_rule_reconciliations()
