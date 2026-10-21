"""Parameter conversion for native pre-submission validators."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from ...calendar import get_calendar, get_schedule
from ...controls import (
    AccountConstraints,
    AccountControlContext,
    OrderControlContext,
    TradingControl,
    TradingControlViolation,
)
from ...sessions import SessionConfig


@dataclass(frozen=True)
class _TradingRule:
    constraint: TradingControl
    on_error: str

    @property
    def counts_order_requests(self) -> bool:
        return getattr(self.constraint, "counts_order_requests", False)

    @property
    def timezone(self) -> str:
        return getattr(self.constraint, "timezone", "UTC")

    def validate(self, request: OrderControlContext) -> None:
        try:
            self.constraint.validate(request)
        except TradingControlViolation as error:
            if self.on_error == "fail":
                raise
            logging.getLogger(__name__).error("%s", error)


@dataclass
class _ExecutionCloseRule:
    rule: AccountConstraints
    closes: frozenset[datetime]
    seen: set[datetime] = field(default_factory=set)

    def validate(self, account: AccountControlContext) -> None:
        if account.timestamp in self.closes:
            self.rule.validate(account)
            self.seen.add(account.timestamp)

    def completed(self, _event_count: int) -> None:
        missing = self.closes - self.seen
        if missing:
            raise ValueError(
                f"Session account controls require execution close observations: {sorted(missing)}"
            )


def _register_account_controls(context, runtime):
    if not context._account_controls:
        return
    engine = runtime._engine
    assert engine is not None
    config = engine.config
    if config.resolved_calendar is None:
        raise ValueError("Session account controls require a declared calendar")
    points = [time for time in engine.feed.timestamps if runtime._accepts(time)]
    if not points:
        raise ValueError("Account controls require at least one accepted market observation")
    clock = SessionConfig(
        config.resolved_calendar,
        config.resolved_timezone,
        config.resolved_session_start_time,
        config.resolved_timestamp_semantics,
        config.resolved_feed_spec.bar_label,
        config.resolved_data_frequency,
    )
    execution_closes = None
    if get_calendar(config.resolved_calendar).name == "us_futures":
        sessions = {clock.date_for_timestamp(point).date() for point in points}
        closes = []
        for row in get_schedule(config.resolved_calendar, min(sessions), max(sessions)).iter_rows(
            named=True
        ):
            if row["session_date"] not in sessions:
                continue
            closed = pd.Timestamp(row["market_close"]) - timedelta(hours=1)
            if points[0].tzinfo is None:
                closed = closed.tz_convert(config.resolved_timezone).tz_localize(None)
            closes.append(closed)
        execution_closes = frozenset(closes)
        if execution_closes - set(points):
            raise ValueError(
                "Session account controls require each legacy execution close observation"
            )
        config.metadata["migration_account_control_execution_closes"] = [
            point.isoformat() for point in sorted(execution_closes)
        ]
    session = clock.date_for_timestamp(points[0]).date()
    for kind, value, grace in context._account_controls:
        if kind == "maximum":
            rule = AccountConstraints(max_leverage=value)
        else:
            deadline = pd.Timestamp(session, tz="UTC") + grace
            if points[0].tzinfo is None:
                deadline = deadline.tz_convert(config.resolved_timezone).tz_localize(None)
            rule = AccountConstraints(min_leverage=value, deadline=deadline)
        if execution_closes is None:
            engine.broker.register_account_control(rule, boundary="session_end")
        else:
            scheduled = _ExecutionCloseRule(rule, execution_closes)
            engine.broker.register_account_control(scheduled, boundary="after_fills")
            engine.broker._register_completion_validator(scheduled, scheduled.completed)
