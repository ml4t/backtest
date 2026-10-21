"""Account constraints follow the published close boundary, including early closes."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from ml4t.backtest import DataFeed
from ml4t.backtest.compat.zipline import run_algorithm
from ml4t.backtest.compat.zipline.assets import Asset
from ml4t.backtest.compat.zipline.finance import commission, slippage
from ml4t.backtest.controls import AccountControlViolation


def _reference(name):
    return json.loads((Path(__file__).parent / "fixtures/zipline_original" / name).read_text())


def _feed(reference):
    points = [datetime.fromisoformat(t) for t in reference["minutes"]]
    return DataFeed(
        prices_df=pl.DataFrame(
            {
                "timestamp": points,
                "asset": ["A"] * len(points),
                "open": [90.0] * len(points),
                "high": [101.0] * len(points),
                "low": [89.0] * len(points),
                "close": [100.0] * len(points),
                "volume": [1000.0] * len(points),
            }
        ),
        feed_spec={
            "calendar": "NYSE",
            "time_contract_version": 2,
            "timestamp_semantics": "bar_close",
            "bar_label": "right",
            "data_frequency": "1m",
            "bar_grid_anchor": "session_open",
            "partial_bar_policy": "include",
        },
    )


def _costs(context):
    context.a = context.symbol("TEST")
    context.n = 0
    context.set_commission(commission.PerShare(0, 0))
    context.set_slippage(slippage.FixedSlippage(0))


@pytest.mark.parametrize("recovered", [True, False])
def test_max_leverage_allows_recovered_intraday_breach_but_rejects_persistent_at_close(recovered):
    reference = _reference("account-controls-reference.json")
    expected = reference["cases"][str(recovered)]
    calls, seen = [], []

    def initialize(context):
        _costs(context)
        context.set_max_leverage(1.0)

    def handle(context, data):
        now = context.get_datetime()
        calls.append(now.isoformat())
        if context.n == 0:
            context.order(context.a, 20)
        if context.n == 1 and recovered:
            context.order_target(context.a, 0)
        if context.n in [0, 1, 2, 389]:
            seen.append(
                {
                    "timestamp": now.isoformat(),
                    "quantity": context.portfolio.positions[context.a].amount,
                    "cash": context.portfolio.cash,
                    "equity": context.portfolio.portfolio_value,
                    "leverage": context.account.leverage,
                }
            )
        context.n += 1

    def execute():
        return run_algorithm(
            feed=_feed(reference),
            assets=[Asset("A", "TEST")],
            initialize=initialize,
            handle_data=handle,
            capital_base=1000,
        )

    if recovered:
        execute()
    else:
        with pytest.raises(AccountControlViolation):
            execute()
    assert len(calls) == expected["callback_count"]
    assert calls[-1] == expected["last_callback"]
    assert seen == expected["trace"]


@pytest.mark.parametrize("grace_hours", [0, 8, 20, 24, 48])
def test_min_leverage_deadline_uses_start_session_midnight_and_actual_early_close(grace_hours):
    reference = _reference("min-leverage-reference.json")
    expected = reference["cases"][str(grace_hours)]
    calls, seen = [], []

    def initialize(context):
        _costs(context)
        context.set_min_leverage(0.5, timedelta(hours=grace_hours))

    def handle(context, data):
        now = context.get_datetime()
        calls.append(now.isoformat())
        if now.isoformat() in ["2025-07-02T20:00:00+00:00", "2025-07-03T17:00:00+00:00"]:
            seen.append({"timestamp": now.isoformat(), "leverage": context.account.leverage})

    def execute():
        return run_algorithm(
            feed=_feed(reference),
            assets=[Asset("A", "TEST")],
            initialize=initialize,
            handle_data=handle,
            capital_base=1000,
        )

    if grace_hours < 48:
        with pytest.raises(AccountControlViolation):
            execute()
    else:
        execute()
    assert len(calls) == expected["callback_count"]
    assert calls[-1] == expected["last_callback"]
    assert seen == expected["trace"]


@pytest.mark.parametrize(
    "kind,value",
    [("maximum", -1), ("maximum", float("nan")), ("minimum", -1), ("minimum", float("inf"))],
)
def test_account_controls_reject_invalid_parameters_before_any_market_callback(kind, value):
    calls = []

    def initialize(context):
        if kind == "maximum":
            context.set_max_leverage(value)
        else:
            context.set_min_leverage(value, timedelta(0))

    with pytest.raises(ValueError, match="finite and nonnegative"):
        run_algorithm(
            feed=_feed(_reference("account-controls-reference.json")),
            assets=[Asset("A", "TEST")],
            initialize=initialize,
            handle_data=lambda *args: calls.append(args),
        )
    assert calls == []


def test_account_controls_are_initialize_only_and_record_the_real_native_boundary():
    reference = _reference("account-controls-reference.json")
    snapshots = []

    def initialize(context):
        context.set_max_leverage(3)
        context.set_min_leverage(0, timedelta(days=1))

    def handle(context, data):
        with pytest.raises(RuntimeError, match="during initialize"):
            context.set_max_leverage(1)
        with pytest.raises(RuntimeError, match="during initialize"):
            context.set_min_leverage(1, timedelta(0))
        snapshots.append(context.account)

    result = run_algorithm(
        feed=_feed(reference),
        assets=[Asset("A", "TEST")],
        initialize=initialize,
        handle_data=handle,
    )
    assert len(snapshots) == 390
    assert snapshots[0].equity == 100000 and snapshots[0].leverage == 0
    assert result.attrs["backtest_result"].config.metadata["migration_account_controls"] == [
        {"max_leverage": 3, "boundary": "session_end"},
        {
            "min_leverage": 0,
            "grace_seconds": 86400.0,
            "deadline_origin": "start_session_midnight_UTC",
            "boundary": "session_end",
        },
    ]
