"""Validate independent prepared grids before exposing native read services."""

from copy import deepcopy

import pandas as pd
import polars as pl
from ml4t.specs import FeedSpec, TimestampSemantics

from .config import BacktestConfig
from .runtime import HistoryProvider, MarketDataView, RuntimeServices


def prepare_history_source(runtime: RuntimeServices, provider: HistoryProvider) -> MarketDataView:
    engine, market = runtime._engine, runtime.market
    assert engine is not None and market._asof is not None
    asof = market._asof
    cutoff = pd.Timestamp(asof)
    if cutoff.tz is None:
        cutoff = cutoff.tz_localize(engine.config.resolved_timezone)
    prepared = provider(as_of=cutoff)
    frame, metadata = prepared.prices, deepcopy(dict(prepared.metadata))
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("Prepared history prices must be a Polars DataFrame")
    if metadata.get("perspective") != "causal" or metadata.get("as_of") is None:
        raise ValueError("Prepared history must declare a causal as_of perspective")
    if pd.Timestamp(metadata["as_of"]) != cutoff:
        raise ValueError("Prepared history as_of must equal the requested decision cutoff")
    if metadata.get("price_basis") not in {"raw", "split_adjusted", "total_return"}:
        raise ValueError("Prepared history must declare its price_basis")
    spec = FeedSpec.from_any(metadata.get("feed_spec"))
    if spec.time_contract_version != 2:
        raise ValueError("Independent history requires an explicit version-2 FeedSpec")
    if spec.timestamp_semantics not in {
        TimestampSemantics.EVENT_TIME,
        TimestampSemantics.BAR_CLOSE,
    }:
        raise ValueError(
            "Prepared history requires physical observation instants, not session labels"
        )
    if not isinstance(spec.entity_col, str) or spec.entity_col not in frame:
        raise ValueError("Prepared history requires declared stable asset keys")
    for column in (spec.timestamp_col, "available_at"):
        if column not in frame or not isinstance(frame.schema[column], pl.Datetime):
            raise ValueError("Prepared history requires physical timestamps and available_at")
        if frame[column].null_count():
            raise ValueError("Prepared history timestamps and availability cannot be null")
    clock = frame.schema[spec.timestamp_col]
    available = frame.schema["available_at"]
    assert isinstance(clock, pl.Datetime) and isinstance(available, pl.Datetime)
    if clock.time_zone is None or available.time_zone is None:
        raise ValueError("Prepared history requires timezone-aware physical source clocks")
    if frame.filter(
        pl.col(spec.timestamp_col).dt.convert_time_zone("UTC") > cutoff.tz_convert("UTC")
    ).height:
        raise ValueError("Prepared history contains future observations")
    if frame.filter(
        pl.col("available_at").dt.convert_time_zone("UTC") > cutoff.tz_convert("UTC")
    ).height:
        raise ValueError("Prepared history contains source revisions unavailable at the cutoff")
    if "is_imputed" in frame and frame["is_imputed"].fill_null(False).any():
        raise ValueError("Prepared history cannot promote imputed rows to observations")
    if set(frame[spec.entity_col]) - set(market._rows):
        raise ValueError("Prepared history contains an asset outside the execution universe")
    # Use one physical clock for comparisons without rounding nanosecond source keys.
    timezone = engine.config.resolved_timezone
    expressions = []
    for column in (spec.timestamp_col, "available_at"):
        expression = pl.col(column).dt.convert_time_zone(timezone)
        if asof.tzinfo is None:
            expression = expression.dt.replace_time_zone(None)
        expressions.append(expression)
    frame = frame.with_columns(expressions)
    view = MarketDataView(
        runtime,
        engine,
        prices=frame,
        feed_spec=spec,
        owner=market,
        known_at_cols={
            name: "available_at"
            for name in frame.columns
            if name not in {spec.timestamp_col, spec.entity_col}
        },
    )
    view._history_config = BacktestConfig(feed_spec=spec, timezone=engine.config.resolved_timezone)
    view._snapshot_cutoff = asof
    view._preparation_metadata = metadata
    return view
