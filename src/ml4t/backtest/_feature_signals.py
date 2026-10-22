"""Consume Engineer feature declarations without computing or revising features."""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import pandas as pd
import polars as pl
from ml4t.specs import FeedSpec

from ._frame_inputs import pandas_frame as _pandas_frame


def feature_signals(
    panel: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
    specification: Any,
    target: FeedSpec,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    serializer = getattr(specification, "to_dict", None)
    payload = serializer() if callable(serializer) else specification
    if not isinstance(payload, Mapping) or payload.get("kind") != "features":
        raise ValueError("feature_spec requires an Engineer FeatureSpec or its features mapping")
    payload = deepcopy(dict(payload))
    version = payload.get("version", 1)
    if (
        not isinstance(payload.get("artifact_id"), str)
        or not payload["artifact_id"].strip()
        or isinstance(version, bool)
        or not isinstance(version, int)
        or version < 1
    ):
        raise ValueError("feature_spec requires an artifact identity and positive version")
    schema, definition = payload.get("schema"), payload.get("definition")
    if not isinstance(schema, Mapping) or not isinstance(definition, Mapping):
        raise ValueError("feature_spec requires schema and definition mappings")
    features = schema.get("feature_columns")
    if (
        not isinstance(features, (list, tuple))
        or not features
        or any(not isinstance(name, str) or not name for name in features)
        or len(set(features)) != len(features)
    ):
        raise ValueError("feature_spec must declare distinct feature_columns")
    roles = {
        name: schema.get(name)
        for name in (
            "timestamp_col",
            "entity_col",
            "decision_time_col",
            "available_at_col",
            "eligible_col",
        )
    }
    if any(not isinstance(name, str) or not name for name in roles.values()):
        raise ValueError(
            "Feature signals require one entity and explicit observation/decision/publication/eligibility columns"
        )
    observation, entity, decision, publication, eligible = roles.values()
    if (
        entity in {observation, decision, publication, eligible}
        or eligible in {observation, decision, publication}
        or set(features) & set(roles.values())
    ):
        raise ValueError("Feature values and identity/visibility columns must be distinct")
    source_spec = FeedSpec.from_any(definition.get("feed_spec"))
    if (
        definition.get("feed_spec") is None
        or source_spec.entity_col not in (None, entity, (entity,))
        or source_spec.timestamp_col != observation
    ):
        raise ValueError("Feature schema must match its declared source FeedSpec")
    basis = definition.get("price_basis")
    if not isinstance(basis, str) or not basis.strip():
        raise ValueError("Feature signals require declared price_basis provenance")
    if isinstance(panel, pl.LazyFrame):
        collected = panel.collect()
        if not isinstance(collected, pl.DataFrame):
            raise TypeError("Feature panel collection must produce a DataFrame")
        panel = collected
    elif isinstance(panel, pd.DataFrame):
        panel = _pandas_frame(panel)
    if not isinstance(panel, pl.DataFrame):
        raise TypeError(
            "Feature signals require a Polars frame/lazy frame or column-based Pandas frame"
        )
    required = {*roles.values(), *features}
    universe = schema.get("universe_time_col")
    if universe is not None:
        if not isinstance(universe, str) or not universe:
            raise ValueError("universe_time_col must name a column")
        required.add(universe)
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Feature panel is missing declared columns: {sorted(missing)}")
    for clock in (decision, publication):
        dtype = panel.schema[clock]
        if not isinstance(dtype, pl.Datetime) or dtype.time_zone is None:
            raise ValueError("Feature decision and publication columns must contain aware instants")
    if panel.schema[eligible] != pl.Boolean:
        raise ValueError("Feature eligibility must be Boolean")
    if panel.select(
        pl.any_horizontal(pl.col(decision).is_null(), pl.col(entity).is_null()).any()
    ).item():
        raise ValueError("Feature decision and stable identity keys cannot be null")
    if panel.select(pl.struct([decision, entity]).is_duplicated().any()).item():
        raise ValueError("Duplicate feature decision/stable identity keys")
    if (
        "price_basis" in panel
        and panel.filter(pl.col("price_basis").is_null() | (pl.col("price_basis") != basis)).height
    ):
        raise ValueError("Feature panel price_basis conflicts with its declaration")
    timestamp, asset = target.timestamp_col, target.entity_col
    if not isinstance(asset, str):
        raise ValueError("Feature signals require one resolved target entity column")
    if entity != asset and asset in panel:
        raise ValueError("Feature entity projection would overwrite an existing column")
    if timestamp in panel and timestamp not in {observation, decision}:
        raise ValueError("Feature decision projection would overwrite an existing column")
    observation_output = observation
    if observation == timestamp:
        observation_output = "feature_observation_time"
        if observation_output in panel:
            raise ValueError(
                "Feature observation projection requires unused feature_observation_time"
            )
    preserved = [
        name for name in (universe, "price_basis", "member") if name is not None and name in panel
    ]
    publication_output = (
        observation_output
        if publication == observation
        else timestamp
        if publication == decision
        else publication
    )
    expressions = {
        timestamp: pl.col(decision),
        asset: pl.col(entity),
        observation_output: pl.col(observation),
        publication_output: pl.col(publication),
        **{name: pl.col(name) for name in dict.fromkeys([eligible, *features, *preserved])},
    }
    projected = panel.select([value.alias(name) for name, value in expressions.items()])
    visible_fields = [name for name in projected.columns if name not in {timestamp, asset}]
    return projected, {
        "spec": payload,
        "projection": {
            "observation_col": observation_output,
            "decision_col": timestamp,
            "entity_col": asset,
            "excluded_columns": [
                name for name in panel.columns if name not in required and name not in preserved
            ],
            "signal_known_at_cols": dict.fromkeys(visible_fields, publication_output),
            "signal_eligible_cols": dict.fromkeys(features, eligible),
        },
    }
