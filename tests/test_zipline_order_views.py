"""Order projections follow real cumulative fills and keep each run's identifiers scoped."""

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from ml4t.backtest import DataFeed

_REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/zipline_original/order-views-reference.json").read_text()
)


def _feed():
    return DataFeed(
        prices_df=pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [6, 7, 8]],
                "asset": ["A"] * 3,
                "close": [100.0, 110.0, 120.0],
                "volume": [5] * 3,
            }
        ),
        feed_spec={
            "time_contract_version": 2,
            "timestamp_semantics": "event_time",
            "calendar": None,
        },
    )


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("cancel", [False, True])
def test_order_views_match_published_signed_partial_full_and_cancelled_state(direction, cancel):
    from ml4t.backtest.compat.zipline import run_algorithm
    from ml4t.backtest.compat.zipline.api import (
        cancel_order,
        get_datetime,
        get_open_orders,
        get_order,
    )
    from ml4t.backtest.compat.zipline.assets import Asset
    from ml4t.backtest.compat.zipline.finance import commission, slippage

    seen = []

    def initialize(context):
        context.a = context.symbol("ALPHA")
        context.oid = None
        context.n = 0
        context.set_commission(commission.PerShare(0.01, 1.0))
        context.set_slippage(slippage.VolumeShareSlippage(1.0, 0.0))

    def handle_data(context, data):
        if context.oid is None:
            context.oid = context.order(context.a, direction * 10)
        actual = get_order(context.oid)
        if cancel and context.n == 1:
            assert cancel_order(actual) is None
            actual = get_order(context.oid)
        seen.append(
            {
                "timestamp": get_datetime().isoformat(),
                "amount": actual.amount,
                "filled": actual.filled,
                "commission": actual.commission,
                "status": int(actual.status),
                "created": actual.created.isoformat(),
                "dt": actual.dt.isoformat(),
                "open_for_asset": len(get_open_orders(context.a)),
                "open_groups": len(get_open_orders()),
            }
        )
        assert get_order("unknown-order") is None
        groups = get_open_orders()
        groups.clear()
        assert len(get_open_orders()) == seen[-1]["open_groups"]
        context.n += 1

    run_algorithm(
        feed=_feed(),
        assets=[Asset("A", "ALPHA")],
        initialize=initialize,
        handle_data=handle_data,
        capital_base=10000,
    )
    assert seen == _REFERENCE["traces"][f"{direction}:{cancel}"]


def test_order_ids_from_a_prior_run_cannot_cancel_a_new_native_order_with_same_local_id():
    from ml4t.backtest.compat.zipline import run_algorithm
    from ml4t.backtest.compat.zipline.assets import Asset

    old = []

    def first(context, data):
        if not old:
            old.append(context.order(context.symbol("ALPHA"), 10))

    run_algorithm(feed=_feed(), assets=[Asset("A", "ALPHA")], handle_data=first)
    checked = []

    def second(context, data):
        if not checked:
            current = context.order(context.symbol("ALPHA"), 10)
            assert current != old[0]
            assert context.get_order(old[0]) is None
            context.cancel_order(old[0])
            assert len(context.get_open_orders(context.symbol("ALPHA"))) == 1
            checked.append(True)

    run_algorithm(feed=_feed(), assets=[Asset("A", "ALPHA")], handle_data=second)
    assert checked == [True]
