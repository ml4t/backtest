"""Validate independent prepared grids before exposing native read services."""

from collections.abc import Sequence
from copy import deepcopy

import pandas as pd
import polars as pl
from ml4t.specs import FeedSpec, TimestampSemantics

from .config import BacktestConfig
from .runtime import HistoryProvider, MarketDataView, RuntimeServices, _request_history


def prepare_history_source(
    runtime: RuntimeServices,
    provider: HistoryProvider,
    *,
    known_at_cols: dict[str, str] | None = None,
    research_keys: Sequence[str] = (),
) -> MarketDataView:
    engine, market = runtime._engine, runtime.market
    assert engine is not None and market._asof is not None
    asof = market._asof
    cutoff = pd.Timestamp(asof)
    if cutoff.tz is None:
        cutoff = cutoff.tz_localize(engine.config.resolved_timezone)
    prepared = _request_history(
        provider, as_of=cutoff, observation_before=cutoff if not market._inclusive else None
    )
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
            "Prepared history timestamp role requires physical observation instants, not session labels"
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
    if not market._inclusive:
        if frame.filter(
            pl.col(spec.timestamp_col).dt.convert_time_zone("UTC") >= cutoff.tz_convert("UTC")
        ).height:
            raise ValueError(
                "Prepared history contains an observation excluded by observation_before"
            )
        metadata["observation_before"] = cutoff.isoformat()
    if frame.filter(
        pl.col("available_at").dt.convert_time_zone("UTC") > cutoff.tz_convert("UTC")
    ).height:
        raise ValueError("Prepared history contains source revisions unavailable at the cutoff")
    if "is_imputed" in frame and frame["is_imputed"].fill_null(False).any():
        raise ValueError("Prepared history cannot promote imputed rows to observations")
    if set(frame[spec.entity_col]) - set(market._rows) - set(research_keys):
        raise ValueError("Prepared history asset is outside the observed execution keys")
    # Use one physical clock for comparisons without rounding nanosecond source keys.
    timezone = engine.config.resolved_timezone
    known = {
        name: "available_at"
        for name in frame.columns
        if name not in {spec.timestamp_col, spec.entity_col}
    }
    known.update(known_at_cols or {})
    expressions = []
    for column in dict.fromkeys([spec.timestamp_col, "available_at", *known.values()]):
        dtype = frame.schema.get(column)
        if not isinstance(dtype, pl.Datetime):
            continue
        expression = pl.col(column)
        if dtype.time_zone is None:
            expression = expression.dt.replace_time_zone(spec.timezone or timezone)
        expression = expression.dt.convert_time_zone(timezone)
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
        known_at_cols=known,
    )
    view._history_config = BacktestConfig(feed_spec=spec, timezone=engine.config.resolved_timezone)
    view._snapshot_cutoff = asof
    view._preparation_metadata = metadata
    return view
