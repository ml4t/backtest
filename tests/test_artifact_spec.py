from __future__ import annotations

from pathlib import Path

from ml4t.backtest.spec_bridge import (
    market_data_spec_to_feed_spec,
    market_data_spec_to_runtime_metadata,
)
from ml4t.diagnostic.artifacts import dump_spec, load_market_data_spec, load_spec
from ml4t.engineer.artifacts import FeatureSpec, LabelSpec, PredictionSpec
from ml4t.specs import ArtifactKind, FeedSpec, MarketDataSpec, TimestampSemantics


def test_market_data_spec_from_mapping_normalizes_timestamp_semantics() -> None:
    spec = MarketDataSpec.from_mapping(
        {
            "artifact_id": "nasdaq100_1m_nbbo_v1",
            "kind": "market_data",
            "schema": {
                "timestamp_col": "ts",
                "entity_col": "asset",
                "close_col": "last_trade_price",
                "bid_col": "close_bid_price",
                "ask_col": "close_ask_price",
                "mid_col": "mid_close",
            },
            "semantics": {
                "data_frequency": "1m",
                "calendar": "NYSE",
                "timezone": "America/New_York",
                "timestamp_semantics": "bar_close",
                "session_start_time": "09:30:00",
                "bar_type": "ohlcv_nbbo",
            },
        }
    )

    assert spec.kind == ArtifactKind.MARKET_DATA
    assert spec.schema.bid_col == "close_bid_price"
    assert spec.schema.ask_col == "close_ask_price"
    assert spec.semantics.timestamp_semantics == TimestampSemantics.BAR_CLOSE


def test_market_data_spec_to_feed_spec_preserves_quote_and_temporal_fields() -> None:
    spec = MarketDataSpec.from_mapping(
        {
            "artifact_id": "nasdaq100_1m_nbbo_v1",
            "kind": "market_data",
            "schema": {
                "timestamp_col": "timestamp",
                "entity_col": "asset",
                "price_col": "mid_close",
                "open_col": "open",
                "high_col": "high",
                "low_col": "low",
                "close_col": "last_trade_price",
                "volume_col": "volume",
                "bid_col": "close_bid_price",
                "ask_col": "close_ask_price",
                "mid_col": "mid_close",
            },
            "semantics": {
                "data_frequency": "1m",
                "calendar": "NYSE",
                "timezone": "America/New_York",
                "timestamp_semantics": "bar_close",
                "session_start_time": "09:30:00",
                "bar_type": "ohlcv_nbbo",
            },
        }
    )

    feed_spec = market_data_spec_to_feed_spec(spec)

    assert isinstance(feed_spec, FeedSpec)
    assert feed_spec.price_col == "mid_close"
    assert feed_spec.close_col == "last_trade_price"
    assert feed_spec.bid_col == "close_bid_price"
    assert feed_spec.ask_col == "close_ask_price"
    assert feed_spec.mid_col == "mid_close"
    assert feed_spec.calendar == "NYSE"
    assert feed_spec.timezone == "America/New_York"
    assert feed_spec.timestamp_semantics == TimestampSemantics.BAR_CLOSE


def test_market_data_schema_keeps_close_default_when_only_price_col_is_overridden() -> None:
    spec = MarketDataSpec.from_mapping(
        {
            "artifact_id": "nasdaq100_1m_nbbo_v1",
            "kind": "market_data",
            "schema": {"price_col": "mid_close"},
        }
    )

    assert spec.schema.price_col == "mid_close"
    assert spec.schema.close_col == "close"


def test_runtime_metadata_helper_returns_feed_semantics() -> None:
    metadata = market_data_spec_to_runtime_metadata(
        {
            "artifact_id": "us_equities_daily_bars_v1",
            "kind": "market_data",
            "semantics": {
                "data_frequency": "1d",
                "calendar": "NYSE",
                "timezone": "America/New_York",
                "timestamp_semantics": "session_label",
            },
        }
    )

    assert metadata == {
        "calendar": "NYSE",
        "timezone": "America/New_York",
        "data_frequency": "1d",
        "timestamp_semantics": TimestampSemantics.SESSION_LABEL,
        "session_start_time": None,
        "bar_type": None,
    }


def test_spec_io_yaml_round_trip(tmp_path: Path) -> None:
    spec = MarketDataSpec.from_mapping(
        {
            "artifact_id": "us_equities_daily_bars_v1",
            "kind": "market_data",
            "storage": {"path": "labels/prices.parquet", "format": "parquet"},
            "schema": {
                "timestamp_col": "timestamp",
                "entity_col": "symbol",
                "open_col": "adj_open",
                "high_col": "adj_high",
                "low_col": "adj_low",
                "close_col": "adj_close",
                "volume_col": "adj_volume",
            },
            "semantics": {
                "data_frequency": "1d",
                "calendar": "NYSE",
                "timezone": "America/New_York",
                "timestamp_semantics": "bar_close",
            },
            "provenance": {"source_artifacts": ["raw_prices_v1"]},
        }
    )

    path = dump_spec(spec, tmp_path / "market_data.yaml")
    loaded = load_market_data_spec(path)

    assert loaded == spec


def test_spec_io_json_round_trip(tmp_path: Path) -> None:
    spec = MarketDataSpec.from_mapping(
        {
            "artifact_id": "us_equities_daily_bars_v1",
            "kind": "market_data",
            "storage": {"path": "labels/prices.parquet", "format": "parquet"},
            "schema": {"close_col": "adj_close"},
            "semantics": {"data_frequency": "1d", "timestamp_semantics": "bar_close"},
        }
    )

    path = dump_spec(spec, tmp_path / "market_data.json")
    loaded = load_market_data_spec(path)

    assert loaded == spec


def test_load_spec_dispatches_label_spec() -> None:
    spec = load_spec(
        {
            "artifact_id": "us_equities_fwd_ret_1d_v1",
            "kind": "labels",
            "definition": {
                "family": "forward_return",
                "task_type": "regression",
                "horizon": "1D",
                "buffer": "1D",
                "source_artifact": "us_equities_daily_bars_v1",
            },
        }
    )

    assert isinstance(spec, LabelSpec)
    assert spec.definition.buffer == "1D"
    assert spec.schema.entity_col == "asset"
    assert spec.schema.label_col == "label_value"


def test_load_spec_dispatches_feature_spec() -> None:
    spec = load_spec(
        {
            "artifact_id": "us_equities_financial_features_v1",
            "kind": "features",
            "schema": {
                "timestamp_col": "timestamp",
                "feature_columns": ["mom_21", "vol_21"],
            },
            "definition": {
                "family": "financial",
                "join_keys": ["timestamp", "asset"],
                "source_artifacts": ["us_equities_daily_bars_v1"],
            },
        }
    )

    assert isinstance(spec, FeatureSpec)
    assert spec.schema.entity_col == "asset"
    assert spec.schema.feature_columns == ("mom_21", "vol_21")
    assert spec.definition.source_artifacts == ("us_equities_daily_bars_v1",)


def test_load_spec_dispatches_prediction_spec() -> None:
    spec = load_spec(
        {
            "artifact_id": "us_equities_preds_v1",
            "kind": "predictions",
            "definition": {
                "split_protocol": "walk_forward_oos",
                "label_artifact": "us_equities_fwd_ret_1d_v1",
                "feature_artifacts": ["us_equities_financial_features_v1"],
                "training_hash": "abc123",
            },
        }
    )

    assert isinstance(spec, PredictionSpec)
    assert spec.schema.entity_col == "asset"
    assert spec.schema.prediction_col == "prediction_value"
    assert spec.definition.feature_artifacts == ("us_equities_financial_features_v1",)
    assert spec.definition.training_hash == "abc123"
