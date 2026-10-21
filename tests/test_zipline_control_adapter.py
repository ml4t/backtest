"""Controls use native admission state with explicit legacy request-count semantics."""

import json
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from ml4t.backtest import DataFeed

_REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/zipline_original/controls-reference.json").read_text()
)


def _feed():
    trace = _REFERENCE["traces"]["rejected_attempt"]
    return DataFeed(
        prices_df=pl.DataFrame(
            {
                "timestamp": [datetime.fromisoformat(row["timestamp"]) for row in trace],
                "asset": ["A"] * 3,
                **_REFERENCE["inputs"],
            }
        ),
        feed_spec={
            "time_contract_version": 2,
            "timestamp_semantics": "bar_close",
            "calendar": "NYSE",
            "bar_label": "right",
            "bar_grid_anchor": "session_open",
            "partial_bar_policy": "include",
            "data_frequency": "1d",
        },
    )


@pytest.mark.parametrize(
    "label,quantities",
    [
        ("rejected_attempt", [6, 1, 1]),
        ("zero_request", [0, 6, 1]),
        ("log_and_allow", [6]),
        ("long_only", [-1, 1]),
        ("position_size", [4, 4, -6]),
        ("order_notional", [5, 6]),
        ("position_notional", [4, 4, -6]),
    ],
)
def test_control_traces_match_published_counts_zero_requests_later_rejections_and_log_policy(
    label, quantities
):
    from ml4t.backtest.compat.zipline import run_algorithm
    from ml4t.backtest.compat.zipline.assets import Asset
    from ml4t.backtest.compat.zipline.finance import commission, slippage
    from ml4t.backtest.controls import TradingControlViolation

    seen = []

    def initialize(context):
        context.a = context.symbol("ALPHA")
        context.set_commission(commission.PerShare(0, 0))
        context.set_slippage(slippage.FixedSlippage(0))
        if label in {"rejected_attempt", "zero_request"}:
            context.set_max_order_count(2)
            context.set_max_order_size(max_shares=5)
        elif label == "log_and_allow":
            context.set_max_order_size(max_shares=5, on_error="log")
        elif label == "position_size":
            context.set_max_position_size(max_shares=5)
        elif label == "order_notional":
            context.set_max_order_size(max_notional=500)
        elif label == "position_notional":
            context.set_max_position_size(max_notional=500)
        else:
            context.set_long_only()

    def handle(context, data):
        answers = []
        for amount in quantities:
            try:
                order_id = context.order(context.a, amount)
                answers.append("none" if order_id is None else "submitted")
            except TradingControlViolation:
                answers.append("violation")
        seen.append(
            {
                "timestamp": context.get_datetime().isoformat(),
                "answers": answers,
                "open": len(context.get_open_orders(context.a)),
                "quantity": context.portfolio.positions[context.a].amount,
                "cash": context.portfolio.cash,
                "equity": context.portfolio.portfolio_value,
            }
        )

    run_algorithm(
        feed=_feed(),
        assets=[Asset("A", "ALPHA")],
        initialize=initialize,
        handle_data=handle,
        capital_base=10000,
    )
    assert seen == _REFERENCE["traces"][label]
