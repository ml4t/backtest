"""End-of-day cancellation must occur after close callbacks at the actual execution close."""

import json
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from ml4t.backtest import DataFeed
from ml4t.backtest.compat.zipline import run_algorithm
from ml4t.backtest.compat.zipline.assets import Asset, Future

ROOT = Path(__file__).parent / "fixtures/zipline_cancel"
REFERENCE = json.loads((ROOT / "reference.json").read_text())["cases"]


@pytest.mark.parametrize("case", sorted(REFERENCE))
@pytest.mark.parametrize("naive", [False, True], ids=["utc", "naive_new_york"])
def test_actual_reference_eod_and_never_preserve_close_callback_partial_fill_and_order_clocks(
    case, naive
):
    from ml4t.backtest.compat.zipline.api import set_cancel_policy
    from ml4t.backtest.compat.zipline.finance import commission, slippage
    from ml4t.backtest.compat.zipline.finance.cancel_policy import EODCancel, NeverCancel

    reference = REFERENCE[case]
    calendar, mode = case.split("/")
    frame = pl.read_csv(ROOT / (calendar + "-inputs.csv"), try_parse_dates=True)
    if naive:
        frame = frame.with_columns(
            pl.col("timestamp").dt.convert_time_zone("America/New_York").dt.replace_time_zone(None)
        )
    asset = (
        Future("A", "ESH25", 8, root_symbol="ES", multiplier=50.0, tick_size=0.25)
        if calendar == "us_futures"
        else Asset("A", "A", 8)
    )
    closes = [pd.Timestamp(value) for value in reference["closes"]]
    opens = [pd.Timestamp(value) for value in reference["opens"]]
    trace, final = [], []

    def state(context):
        orders = [context.get_order(key) for key in context.ids]
        return [
            {
                "amount": float(order.amount),
                "filled": float(order.filled),
                "status": int(order.status),
                "dt": order.dt.isoformat(),
            }
            for order in orders
        ]

    def initialize(context):
        context.ids = []
        set_cancel_policy(EODCancel(warn_on_cancel=False) if mode == "eod" else NeverCancel())
        context.set_commission(
            us_equities=commission.PerShare(0), us_futures=commission.PerFutureTrade(0)
        )
        context.set_slippage(
            us_equities=slippage.FixedBasisPointsSlippage(0, 0.5),
            us_futures=slippage.FixedBasisPointsSlippage(0, 0.5),
        )

    def handle(context, data):
        now = context.get_datetime()
        if now == opens[0]:
            context.ids.append(context.order(asset, 2, limit_price=1))
        if now == closes[0] - pd.Timedelta(minutes=1):
            context.ids.append(context.order(asset, 3))
        if now == closes[0]:
            context.ids.append(context.order(asset, 4))
        if now in [closes[0], opens[1]]:
            trace.append({"timestamp": now.isoformat(), "orders": state(context)})

    result = run_algorithm(
        feed=DataFeed(
            prices_df=frame,
            feed_spec={
                "calendar": "NYSE" if calendar == "XNYS" else calendar,
                "time_contract_version": 2,
                "timestamp_semantics": "bar_close",
                "bar_label": "right",
                "bar_grid_anchor": "session_open",
                "partial_bar_policy": "include",
                "data_frequency": "1m",
                "timezone": "America/New_York" if naive else "UTC",
            },
        ),
        assets=[asset],
        initialize=initialize,
        handle_data=handle,
        analyze=lambda context, _performance: final.extend(state(context)),
    )
    assert trace == reference["trace"]
    assert final == reference["final"]
    fills = result.attrs["backtest_result"].to_fills_dataframe()
    actual = [
        {
            "timestamp": pd.Timestamp(row["timestamp"])
            .tz_localize("America/New_York")
            .tz_convert("UTC")
            .isoformat()
            if naive
            else row["timestamp"].isoformat(),
            "amount": row["quantity"],
            "price": row["price"],
        }
        for row in fills.iter_rows(named=True)
    ]
    assert actual == reference["transactions"]


def test_eod_requires_real_close_coverage_before_any_strategy_market_callback():
    from ml4t.backtest.compat.zipline.finance.cancel_policy import EODCancel

    raw = pl.read_csv(ROOT / "XNYS-inputs.csv", try_parse_dates=True)
    first_close = pd.Timestamp(REFERENCE["XNYS/eod"]["closes"][0])
    raw = raw.filter(pl.col("timestamp") != first_close)
    seen = []
    with pytest.raises(ValueError, match="execution-close observation"):
        run_algorithm(
            feed=DataFeed(
                prices_df=raw,
                feed_spec={
                    "calendar": "NYSE",
                    "time_contract_version": 2,
                    "timestamp_semantics": "bar_close",
                    "bar_label": "right",
                    "data_frequency": "1m",
                },
            ),
            assets=[Asset("A", "A", 8)],
            initialize=lambda c: c.set_cancel_policy(EODCancel()),
            handle_data=lambda c, _d: seen.append(c),
        )
    assert seen == []


def test_daily_eod_cancels_all_remainders_instead_of_reproducing_reference_first_only_defect():
    from ml4t.backtest.compat.zipline.finance.cancel_policy import EODCancel

    reference = json.loads((ROOT / "daily-reference.json").read_text())
    times = [pd.Timestamp(value).to_pydatetime() for value in reference["closes"]]
    traces, final = [], []

    def initialize(context):
        context.ids = []
        context.set_cancel_policy(EODCancel(warn_on_cancel=False))

    def handle(context, data):
        if not context.ids:
            context.ids = [context.order(context.sid(8), 1, limit_price=1) for _ in range(3)]
        traces.append([int(context.get_order(key).status) for key in context.ids])

    result = run_algorithm(
        feed=DataFeed(
            prices_df=pl.DataFrame(
                {
                    "timestamp": times,
                    "asset": ["A"] * 2,
                    "close": [100.0] * 2,
                    "volume": [2.0] * 2,
                }
            ),
            feed_spec={
                "calendar": "NYSE",
                "data_frequency": "daily",
                "time_contract_version": 2,
                "timestamp_semantics": "bar_close",
            },
        ),
        assets=[Asset("A", "A", 8)],
        initialize=initialize,
        handle_data=handle,
        analyze=lambda c, _p: final.extend(int(c.get_order(key).status) for key in c.ids),
    )
    assert traces == [[0, 0, 0], [2, 2, 2]] and final == [2, 2, 2]
    assert reference["callback_statuses"] == [[0, 0, 0], [2, 0, 0]]
    assert reference["final_statuses"] == [2, 2, 0]
    assert result.attrs["backtest_result"].to_fills_dataframe().is_empty()
