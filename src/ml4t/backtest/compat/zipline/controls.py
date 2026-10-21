"""Parameter conversion for native pre-submission validators."""

import logging
from dataclasses import dataclass

import pandas as pd

from ...controls import (
    AccountConstraints,
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
    session = clock.date_for_timestamp(points[0]).date()
    for kind, value, grace in context._account_controls:
        if kind == "maximum":
            rule = AccountConstraints(max_leverage=value)
        else:
            deadline = pd.Timestamp(session, tz="UTC") + grace
            if points[0].tzinfo is None:
                deadline = deadline.tz_convert(config.resolved_timezone).tz_localize(None)
            rule = AccountConstraints(min_leverage=value, deadline=deadline)
        engine.broker.register_account_control(rule, boundary="session_end")
