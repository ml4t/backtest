"""Restriction queries and native order decisions must match actual published workflows."""

import json
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from ml4t.backtest import DataFeed
from ml4t.backtest.compat.zipline import run_algorithm
from ml4t.backtest.compat.zipline.assets import Asset

_REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/zipline_original/restrictions-reference.json").read_text()
)
_ASSETS = [Asset("A", "A", 7), Asset("B", "B", 8)]


def _feed():
    times = [
        datetime.fromisoformat(row["timestamp"]) for row in _REFERENCE["cases"]["fail"]["trace"]
    ]
    return DataFeed(
        prices_df=pl.DataFrame(
            [
                {"timestamp": point, "asset": asset.key, "close": 100.0 + i, "volume": 1000}
                for i, point in enumerate(times)
                for asset in _ASSETS
            ]
        ),
        feed_spec={
            "time_contract_version": 2,
            "calendar": "NYSE",
            "data_frequency": "1d",
            "timestamp_semantics": "bar_close",
            "bar_label": "right",
        },
    )


def test_unconfigured_equity_slippage_matches_actual_published_default_fill_and_cash():
    seen = []

    def handle(context, data):
        if not seen:
            context.order(_ASSETS[0], 1)
        seen.append(context.get_datetime())

    performance = run_algorithm(feed=_feed(), assets=_ASSETS, handle_data=handle, capital_base=1000)
    fills = performance.attrs["backtest_result"].to_fills_dataframe()
    reference = _REFERENCE["cases"]["fail"]
    assert fills["price"].to_list() == [reference["fills"][0]["price"]]
    assert performance.cash.tolist() == pytest.approx(reference["cash"])


@pytest.mark.parametrize("mode", ["fail", "log"])
def test_dated_union_restrictions_match_actual_query_shapes_native_rejections_fills_and_cash(mode):
    import pandas as pd

    from ml4t.backtest.compat.zipline.finance.asset_restrictions import (
        RESTRICTION_STATES,
        HistoricalRestrictions,
        NoRestrictions,
        Restriction,
        StaticRestrictions,
    )

    trace = []

    def initialize(context):
        context.rules = (
            NoRestrictions()
            | StaticRestrictions([_ASSETS[1]])
            | HistoricalRestrictions(
                [
                    Restriction(
                        _ASSETS[0], pd.Timestamp("2025-07-02T20:00:00Z"), RESTRICTION_STATES.FROZEN
                    ),
                    Restriction(
                        _ASSETS[0], pd.Timestamp("2025-07-03T17:00:00Z"), RESTRICTION_STATES.ALLOWED
                    ),
                ]
            )
        )
        context.set_asset_restrictions(context.rules, on_error=mode)

    def handle(context, data):
        point = context.get_datetime()
        item = {
            "timestamp": point.isoformat(),
            "restricted": context.rules.is_restricted(_ASSETS, point).tolist(),
            "can_trade": data.can_trade(_ASSETS).tolist(),
            "submitted": [],
            "rejected": [],
        }
        assert context.rules.is_restricted(_ASSETS[0], point) is item["restricted"][0]
        assert data.can_trade(_ASSETS[0]) is item["can_trade"][0]
        assert data.can_trade([]).dtype == bool and data.can_trade([]).empty
        for asset in _ASSETS:
            try:
                context.order(asset, 1)
            except ValueError as error:
                assert "restricted" in str(error)
                item["rejected"].append(int(asset))
            else:
                item["submitted"].append(int(asset))
        trace.append(item)

    frame = run_algorithm(
        feed=_feed(), assets=_ASSETS, initialize=initialize, handle_data=handle, capital_base=1000
    )
    expected = _REFERENCE["cases"][mode]
    assert trace == expected["trace"]
    fills = [
        {
            "timestamp": fill["dt"].isoformat(),
            "sid": int(fill["sid"]),
            "amount": fill["amount"],
            "price": fill["price"],
        }
        for transaction in frame.transactions
        for fill in transaction
    ]
    assert len(fills) == len(expected["fills"])
    for produced, reference in zip(fills, expected["fills"], strict=True):
        assert {name: value for name, value in produced.items() if name != "price"} == {
            name: value for name, value in reference.items() if name != "price"
        }
        assert produced["price"] == pytest.approx(reference["price"], rel=0, abs=1e-12)
    assert frame.cash.tolist() == pytest.approx(expected["cash"], rel=0, abs=1e-10)
    assert (
        frame.attrs["backtest_result"].config.metadata["migration_trading_controls"][0][
            "definition"
        ]["model"]
        == "union"
    )


@pytest.mark.parametrize("provider", [False, True])
def test_legacy_do_not_order_alias_uses_native_restrictions_for_assets_and_explicit_provider_sids(
    provider,
):
    from types import SimpleNamespace

    seen = []

    def initialize(context):
        source = SimpleNamespace(current_securities=lambda _dt: {8}) if provider else [_ASSETS[1]]
        context.set_do_not_order_list(source)

    def handle(context, data):
        seen.append(data.can_trade(_ASSETS).tolist())
        with pytest.raises(ValueError, match="restricted"):
            context.order(_ASSETS[1], 1)

    result = run_algorithm(feed=_feed(), assets=_ASSETS, initialize=initialize, handle_data=handle)
    assert seen == [[True, False]] * 3
    assert result.attrs["backtest_result"].fills == []


def test_naive_effective_dates_use_query_timezone_and_later_events_do_not_change_earlier_states():
    import pandas as pd

    from ml4t.backtest.compat.zipline.finance.asset_restrictions import (
        RESTRICTION_STATES,
        HistoricalRestrictions,
        Restriction,
    )

    changes = [Restriction(_ASSETS[0], pd.Timestamp("2025-07-02 09:30"), RESTRICTION_STATES.FROZEN)]
    rules = HistoricalRestrictions(changes)
    later = HistoricalRestrictions(
        [
            *changes,
            Restriction(_ASSETS[0], pd.Timestamp("2025-07-03 09:30"), RESTRICTION_STATES.ALLOWED),
        ]
    )
    for timezone in ["UTC", "America/New_York"]:
        before = pd.Timestamp("2025-07-02 09:30", tz=timezone) - pd.Timedelta(1, "ns")
        at = before + pd.Timedelta(1, "ns")
        assert rules.is_restricted(_ASSETS[0], before) is False
        assert rules.is_restricted(_ASSETS[0], at) is True
        assert later.is_restricted(_ASSETS[0], before) is False
        assert later.is_restricted(_ASSETS[0], at) is True


def test_restricted_failed_requests_still_count_and_preexisting_orders_are_not_cancelled():
    import pandas as pd

    from ml4t.backtest.compat.zipline.finance.asset_restrictions import (
        RESTRICTION_STATES,
        HistoricalRestrictions,
        Restriction,
    )

    points = pd.to_datetime(
        [
            "2025-07-01T14:00:00.000000001Z",
            "2025-07-01T14:00:00.000000002Z",
            "2025-07-01T14:00:00.000000003Z",
        ]
    )
    feed = DataFeed(
        prices_df=pl.DataFrame(
            {
                "timestamp": pl.Series(points.asi8).cast(pl.Datetime("ns", "UTC")),
                "asset": ["A"] * 3,
                "close": [100.0, 101.0, 102.0],
                "volume": [1000] * 3,
            }
        ),
        feed_spec={
            "calendar": None,
            "time_contract_version": 2,
            "timestamp_semantics": "event_time",
        },
    )
    errors = []

    def initialize(context):
        context.set_max_order_count(2)
        context.set_asset_restrictions(
            HistoricalRestrictions(
                [
                    Restriction(_ASSETS[0], points[1], RESTRICTION_STATES.FROZEN),
                    Restriction(_ASSETS[0], points[2], RESTRICTION_STATES.ALLOWED),
                ]
            )
        )

    def handle(context, data):
        try:
            context.order(_ASSETS[0], 1)
        except ValueError as error:
            errors.append(str(error))

    frame = run_algorithm(feed=feed, assets=_ASSETS, initialize=initialize, handle_data=handle)
    assert len(errors) == 2 and "restricted" in errors[0] and "order requests" in errors[1].lower()
    fills = frame.attrs["backtest_result"].to_fills_dataframe()
    assert fills["quantity"].to_list() == [1.0]
    assert fills["timestamp"].cast(pl.Int64).to_list() == [points[1].value]


def test_restrictions_are_initialize_scoped_and_runtime_reuse_does_not_keep_prior_rules():
    from ml4t.backtest import RuntimeServices
    from ml4t.backtest.compat.zipline.finance.asset_restrictions import StaticRestrictions

    runtime = RuntimeServices()

    def handle(context, data):
        with pytest.raises(RuntimeError, match="during initialize"):
            context.set_asset_restrictions(StaticRestrictions(_ASSETS))

    run_algorithm(
        feed=_feed(),
        assets=_ASSETS,
        runtime=runtime,
        handle_data=handle,
        initialize=lambda context: context.set_asset_restrictions(StaticRestrictions(_ASSETS)),
    )
    seen = []
    run_algorithm(
        feed=_feed(),
        assets=_ASSETS,
        runtime=runtime,
        handle_data=lambda _context, data: seen.append(data.can_trade(_ASSETS).tolist()),
    )
    assert seen == [[True, True]] * 3


@pytest.mark.parametrize("mode", ["default", "noop", "ten_bps_keyword"])
def test_published_basis_point_defaults_noops_and_keywords_share_volume_across_orders(mode):
    from ml4t.backtest.compat.zipline.api import set_commission, set_slippage
    from ml4t.backtest.compat.zipline.finance.commission import PerShare
    from ml4t.backtest.compat.zipline.finance.slippage import FixedBasisPointsSlippage

    reference = json.loads(
        (
            Path(__file__).parent / "fixtures/zipline_original/basis-points-reference.json"
        ).read_text()
    )["cases"][mode]

    def initialize(context):
        context.submitted = False
        if mode == "noop":
            set_commission()
            set_slippage()
        elif mode == "ten_bps_keyword":
            set_commission(us_equities=PerShare(cost=0.1, min_trade_cost=1.0))
            set_slippage(us_equities=FixedBasisPointsSlippage(basis_points=10.0, volume_limit=0.1))

    def handle(context, data):
        if not context.submitted:
            context.order(_ASSETS[0], 120)
            context.order(_ASSETS[0], 80)
            context.submitted = True

    frame = run_algorithm(
        feed=_feed(), assets=_ASSETS, initialize=initialize, handle_data=handle, capital_base=1000.0
    )
    fills = [
        {"timestamp": fill["dt"].isoformat(), "amount": fill["amount"], "price": fill["price"]}
        for transactions in frame.transactions
        for fill in transactions
    ]
    assert [row["timestamp"] for row in fills] == [row["timestamp"] for row in reference["fills"]]
    assert (
        [row["amount"] for row in fills]
        == [100, 20, 80]
        == [row["amount"] for row in reference["fills"]]
    )
    assert [row["price"] for row in fills] == pytest.approx(
        [row["price"] for row in reference["fills"]], rel=0, abs=1e-12
    )
    assert frame.cash.tolist() == pytest.approx(reference["cash"], rel=0, abs=1e-10)
    assert frame.portfolio_value.tolist() == pytest.approx(reference["nav"], rel=0, abs=1e-10)


def test_invalid_basis_point_settings_and_double_declared_equity_models_fail_before_market_callbacks():
    from ml4t.backtest.compat.zipline.finance.slippage import FixedBasisPointsSlippage

    for model in [
        FixedBasisPointsSlippage(-1),
        FixedBasisPointsSlippage(float("nan")),
        FixedBasisPointsSlippage(5, 0),
        FixedBasisPointsSlippage(5, float("inf")),
    ]:
        with pytest.raises(ValueError, match="basis_points|volume_limit"):
            run_algorithm(
                feed=_feed(),
                assets=_ASSETS,
                initialize=lambda context, selected=model: context.set_slippage(selected),
                handle_data=lambda *_: pytest.fail("Invalid model reached market callback"),
            )
    with pytest.raises(ValueError, match="once"):
        run_algorithm(
            feed=_feed(),
            assets=_ASSETS,
            initialize=lambda context: context.set_slippage(
                FixedBasisPointsSlippage(), us_equities=FixedBasisPointsSlippage()
            ),
        )


def test_new_york_naive_input_keeps_migration_restrictions_on_the_declared_utc_query_clock():
    import pandas as pd

    from ml4t.backtest.compat.zipline.finance.asset_restrictions import (
        RESTRICTION_STATES,
        HistoricalRestrictions,
        Restriction,
    )

    original = _feed()
    prices = original.prices.with_columns(
        pl.col("timestamp").dt.convert_time_zone("America/New_York").dt.replace_time_zone(None)
    )
    feed = DataFeed(
        prices_df=prices, feed_spec=original.feed_spec.with_overrides(timezone="America/New_York")
    )
    allowed, rejected = [], []

    def initialize(context):
        context.set_asset_restrictions(
            HistoricalRestrictions(
                [
                    Restriction(
                        _ASSETS[0], pd.Timestamp("2025-07-02 20:00"), RESTRICTION_STATES.FROZEN
                    ),
                ]
            )
        )

    def handle(context, data):
        allowed.append(data.can_trade(_ASSETS[0]))
        try:
            context.order(_ASSETS[0], 1)
        except ValueError as error:
            assert "restricted" in str(error)
            rejected.append(context.get_datetime().isoformat())

    run_algorithm(feed=feed, assets=_ASSETS, initialize=initialize, handle_data=handle)
    assert allowed == [True, False, False]
    assert rejected == ["2025-07-02T20:00:00+00:00", "2025-07-03T17:00:00+00:00"]
