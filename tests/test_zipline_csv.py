"""Frozen original CSV identity, parsing and availability must survive native delivery."""

import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from ml4t.backtest import DataFeed, RuntimeServices
from ml4t.backtest.compat.zipline import run_algorithm
from ml4t.backtest.compat.zipline.assets import Asset

_REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/zipline_original/csv-reference.json").read_text()
)
_SOURCE = _REFERENCE["original_source_id"]
_ASSET = Asset("A", "A", 7)


def _source(*, sid=7):
    from ml4t.backtest.compat.zipline import PreparedCSV

    asset = Asset("A", "A", sid)
    events = []

    def pre(frame):
        events.append("pre")
        return frame.rename(columns={"ticker": "symbol"}).assign(score=frame["raw"] * 2)

    def post(frame):
        events.append(("post", str(frame.index.tz), [int(key) for key in frame["sid"]]))
        return frame.assign(signal=frame["score"] + 1).drop(columns=["raw", "score"])

    frame = pd.read_csv(io.StringIO(_REFERENCE["csv"]))
    frame["published"] = frame["date"]
    source = PreparedCSV(
        _SOURCE,
        frame,
        available_at_col="published",
        assets={"A": asset},
        pre_func=pre,
        post_func=post,
    )
    return source, events


def _feed():
    return DataFeed(
        prices_df=pl.DataFrame(
            {
                "timestamp": [
                    datetime.fromisoformat(row["timestamp"]) for row in _REFERENCE["trace"]
                ],
                "asset": ["A"] * 3,
                "close": [10.0, 20.0, 30.0],
                "volume": [1000] * 3,
            }
        ),
        feed_spec={
            "calendar": "NYSE",
            "time_contract_version": 2,
            "data_frequency": "1d",
            "timestamp_semantics": "bar_close",
            "bar_label": "right",
        },
    )


def _run(source, handle, *, runtime=None):
    def initialize(context):
        assert context.fetch_csv(_SOURCE) is source

    return run_algorithm(
        feed=_feed(),
        assets=[_ASSET],
        csv_sources=[source],
        initialize=initialize,
        handle_data=handle,
        runtime=runtime,
    )


@pytest.mark.parametrize("sid", [0, 7])
def test_original_csv_pre_post_mapping_stale_delivery_and_source_identity_drive_native_trade(sid):
    source, events = _source(sid=sid)
    asset = Asset("A", "A", sid)
    trace = []

    def handle(context, data):
        value = data.current(asset, "signal")
        trace.append(
            {
                "timestamp": context.get_datetime().isoformat(),
                "signal": value,
                "fetcher_sids": [int(key) for key in data.fetcher_assets],
            }
        )
        assert data.current([asset], ["signal", "close"]).iloc[0].tolist() == [
            value,
            data.current(asset, "close"),
        ]
        if len(trace) == 1:
            context.order(asset, 1)

    performance = _run(source, handle)
    assert trace == [{**row, "fetcher_sids": [sid]} for row in _REFERENCE["trace"]]
    assert events == ["pre", ("post", "UTC", [sid, sid])]
    assert source.df.index.tz is not None
    assert source.df["sid"].tolist() == [asset, asset]
    result = performance.attrs["backtest_result"]
    fills = result.to_fills_dataframe()
    assert fills["quantity"].to_list() == [1.0]
    assert fills["timestamp"][0] == datetime(2025, 7, 2, 20, tzinfo=UTC)
    info = result.config.metadata["migration_csv"][_SOURCE]
    assert info["source_id"] == _SOURCE
    assert info["preparation"]["invalid_dates_dropped"] == 1
    assert info["preparation"]["unmapped_symbols"] == ["UNKNOWN"]
    assert info["preparation"]["transport"] == "caller_loaded_offline"


def test_delayed_csv_publication_masks_values_and_caller_mutation_cannot_rewrite_frozen_source():
    from ml4t.backtest.compat.zipline import PreparedCSV

    raw = pd.DataFrame(
        {
            "date": ["2025-07-01"],
            "symbol": ["A"],
            "score": [123.0],
            "published": ["2025-07-02T20:00:00.000000001Z"],
        }
    )
    source = PreparedCSV(_SOURCE, raw, available_at_col="published", assets={"A": _ASSET})
    raw.loc[0, "score"] = 999.0
    returned = source.df
    returned.loc[:, "score"] = -1000.0
    native_returned = source.source.frame
    native_returned.replace_column(
        native_returned.columns.index("score"), pl.Series("score", [-1000.0])
    )
    seen = []
    _run(source, lambda _context, data: seen.append(data.current(_ASSET, "score")))
    assert pd.isna(seen[0]) and pd.isna(seen[1]) and seen[2] == 123.0


def test_original_csv_key_must_be_registered_and_callback_cannot_download_or_replace_sources():
    from ml4t.backtest.compat.zipline.api import fetch_csv

    runtime = RuntimeServices()
    with pytest.raises(ValueError, match="original source identity"):
        run_algorithm(
            feed=_feed(),
            assets=[_ASSET],
            runtime=runtime,
            initialize=lambda _context: fetch_csv(_SOURCE),
        )
    source, _ = _source()

    def handle(context, data):
        with pytest.raises(RuntimeError, match="during initialize"):
            fetch_csv(_SOURCE)
        assert data.current(_ASSET, "signal") in [11.0, 31.0]

    assert len(_run(source, handle, runtime=runtime)) == 3
    with pytest.raises(RuntimeError, match="active migration callback"):
        fetch_csv(_SOURCE)
    with pytest.raises(ValueError, match="Duplicate original"):
        run_algorithm(feed=_feed(), assets=[_ASSET], csv_sources=[source, source])


def test_fixed_research_symbol_and_unmasked_unknown_symbols_are_nontradable_query_keys():
    from ml4t.backtest.compat.zipline import PreparedCSV

    raw = pd.DataFrame({"date": ["2025-07-01"], "published": ["2025-07-01"], "close": [17.0]})
    source = PreparedCSV(_SOURCE, raw, available_at_col="published", symbol="VIX")
    seen = []

    def handle(context, data):
        seen.append(data.current("VIX", ["close"]))
        assert data.current(_ASSET, "close") in [10.0, 20.0, 30.0]
        with pytest.raises(ValueError, match="registered frozen CSV"):
            data.current("UNREGISTERED", "close")
        with pytest.raises(ValueError, match="Asset"):
            context.order("VIX", 1)

    _run(source, handle)
    assert [row.name for row in seen] == ["VIX"] * 3
    assert [row["close"] for row in seen] == [17.0] * 3
    raw["symbol"] = "UNKNOWN"
    unmasked = PreparedCSV(
        _SOURCE, raw, available_at_col="published", assets={"A": _ASSET}, mask=False
    )
    values = []
    _run(unmasked, lambda _context, data: values.append(data.current("UNKNOWN", "close")))
    assert values == [17.0] * 3


def test_csv_timezone_format_postprocessing_and_duplicate_keys_have_explicit_contracts():
    from ml4t.backtest.compat.zipline import PreparedCSV

    raw = pd.DataFrame(
        {
            "date": ["02/07/2025 09:30:00"],
            "symbol": ["a"],
            "value": [1.0],
            "published": ["2025-07-02 09:31:00"],
        }
    )
    source = PreparedCSV(
        _SOURCE,
        raw,
        available_at_col="published",
        assets={"A": _ASSET},
        date_format="%d/%m/%Y %H:%M:%S",
        timezone="America/New_York",
    )
    assert source.source.frame["timestamp"][0] == datetime(2025, 7, 2, 13, 30, tzinfo=UTC)
    assert source.source.frame["published"][0] == datetime(2025, 7, 2, 13, 31, tzinfo=UTC)
    assert source.df["sid"].tolist() == [_ASSET]
    with pytest.raises(ValueError, match="Duplicate auxiliary"):
        PreparedCSV(
            _SOURCE,
            pd.concat([raw, raw]),
            available_at_col="published",
            assets={"A": _ASSET},
            date_format="%d/%m/%Y %H:%M:%S",
            timezone="America/New_York",
        )
    with pytest.raises(ValueError, match="postprocessing requires"):
        PreparedCSV(
            _SOURCE,
            raw,
            available_at_col="published",
            assets={"A": _ASSET},
            post_func=lambda frame: frame.reset_index(),
        )


def test_overlapping_csv_payloads_require_explicit_original_source_query_instead_of_last_wins():
    from ml4t.backtest.compat.zipline import PreparedCSV

    source, _ = _source()
    other = PreparedCSV(
        "https://example.org/other.csv",
        pd.DataFrame(
            {
                "date": ["2025-07-01"],
                "published": ["2025-07-01"],
                "symbol": ["A"],
                "signal": [999.0],
            }
        ),
        assets={"A": _ASSET},
        available_at_col="published",
    )

    def initialize(context):
        context.fetch_csv(_SOURCE)
        context.fetch_csv(other.source.source_id)

    def handle(context, data):
        data.current(_ASSET, "signal")

    with pytest.raises(ValueError, match="ambiguous across original sources"):
        run_algorithm(
            feed=_feed(),
            assets=[_ASSET],
            csv_sources=[source, other],
            initialize=initialize,
            handle_data=handle,
        )


def test_all_masked_rows_produce_missing_payload_and_no_invented_fetcher_assets():
    from ml4t.backtest.compat.zipline import PreparedCSV

    raw = pd.DataFrame(
        {
            "date": ["2025-07-01"],
            "published": ["2025-07-01"],
            "symbol": ["UNKNOWN"],
            "score": [999.0],
        }
    )
    source = PreparedCSV(_SOURCE, raw, available_at_col="published", assets={"A": _ASSET})
    values = []

    def handle(context, data):
        values.append(data.current(_ASSET, "score"))
        assert data.fetcher_assets == []

    _run(source, handle)
    assert all(pd.isna(value) for value in values)


def test_date_column_named_dt_is_parsed_without_dropping_the_observation_clock():
    from ml4t.backtest.compat.zipline import PreparedCSV

    source = PreparedCSV(
        _SOURCE,
        pd.DataFrame(
            {"dt": ["2025-07-01"], "published": ["2025-07-01"], "symbol": ["A"], "score": [5.0]}
        ),
        available_at_col="published",
        assets={"A": _ASSET},
        date_column="dt",
    )
    values = []
    _run(source, lambda _context, data: values.append(data.current(_ASSET, "score")))
    assert values == [5.0] * 3
