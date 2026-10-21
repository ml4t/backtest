"""Compile per-asset legacy execution settings over native observed history."""

from functools import lru_cache
from math import isfinite, isnan, sqrt
from statistics import stdev

from ... import models
from ..._history_clock import _sessions
from ..._temporal import temporal_rows
from ...calendar import get_calendar
from ...config import ExecutionPrice
from ...execution.limits import (
    AssetExecutionLimits,
    PositiveVolumeLimit,
    SharedVolumeParticipationLimit,
)
from ...sessions import SessionConfig
from .assets import Future
from .finance import slippage
from .finance._futures_eta import DEFAULT_ETA, ROOT_ETA


def _daily_statistics(runtime, asset, instant, history_view, history_sources):
    name = (history_sources or {}).get("1d")
    source = runtime.history_source(name) if name is not None else runtime.market
    config = source._history_config
    frequency = getattr(config.resolved_data_frequency, "value", config.resolved_data_frequency)
    if frequency not in {"daily", "1d"}:
        if name is not None:
            raise ValueError("VolatilityVolumeShare requires a declared daily history source")
        return None
    frame = source.history(
        [asset],
        ["close", "volume"],
        bar_count=21,
        **({"view": history_view} if name is None else {}),
    )
    rows = list(temporal_rows(frame, named=True))
    calendar = config.resolved_calendar
    if calendar is None:
        selected = [row for row in rows if row["timestamp"] < instant][-20:]
    else:
        engine = runtime._engine
        assert engine is not None
        if engine.config.resolved_calendar is not None and (
            get_calendar(calendar).name != get_calendar(engine.config.resolved_calendar).name
        ):
            raise ValueError("Volatility history calendar must match the execution calendar")
        clock = SessionConfig(
            calendar,
            config.resolved_timezone,
            config.resolved_session_start_time,
            config.resolved_timestamp_semantics,
            config.resolved_feed_spec.bar_label,
            config.resolved_data_frequency,
        )
        current = clock.date_for_timestamp(instant).date()
        dates = [
            clock.date_for_timestamp(row["market_close"]).date()
            for row in _sessions(config, instant, 21, True)
        ]
        expected = [day for day in dates if day < current][-20:]
        grouped = {}
        for row in rows:
            day = clock.date_for_timestamp(row["timestamp"]).date()
            if day not in expected:
                continue
            if day in grouped:
                raise ValueError("Daily volatility history requires one observation per session")
            grouped[day] = row
        selected = [grouped[day] for day in expected if day in grouped]
    if len(selected) != 20:
        return None
    if any(row["volume"] is None for row in selected):
        return None
    volumes = [row["volume"] for row in selected]
    if any(not isfinite(value) for value in volumes):
        return None
    closes = []
    for row in selected:
        value = row["close"]
        if value is None or isnan(value):
            if not closes:
                return None
            value = closes[-1]
        if not isfinite(value):
            return None
        closes.append(value)
    if any(value <= 0 for value in closes) or any(value < 0 for value in volumes):
        raise ValueError("Volatility history requires positive closes and nonnegative volume")
    returns = [current / prior - 1 for prior, current in zip(closes, closes[1:], strict=False)]
    return sum(volumes) / 20.0, stdev(returns) * sqrt(252.0)


def _compile(model, *, root=None, statistics=None):
    if isinstance(
        model,
        slippage.FixedBasisPointsSlippage
        | slippage.VolumeShareSlippage
        | slippage.VolatilityVolumeShare,
    ) and (not isfinite(model.volume_limit) or not 0 < model.volume_limit <= 1):
        raise ValueError("volume_limit must be finite and in (0, 1]")
    if isinstance(model, slippage.VolatilityVolumeShare):
        if root is None:
            raise TypeError("VolatilityVolumeShare is a futures model")
        eta = model.eta
        if eta is None:
            value = ROOT_ETA.get(root, DEFAULT_ETA)
        elif isinstance(eta, (int, float)):
            value = float(eta)
        else:
            value = eta.get(root, ROOT_ETA.get(root, DEFAULT_ETA))
        return (
            models.VolatilityVolumeSlippage(value, statistics),
            SharedVolumeParticipationLimit(model.volume_limit, shared_participation=1.0),
            ExecutionPrice.MID,
        )
    if isinstance(model, slippage.FixedBasisPointsSlippage):
        if not isfinite(model.basis_points) or model.basis_points < 0:
            raise ValueError("basis_points must be finite and nonnegative")
        return (
            models.PercentageSlippage(model.basis_points / 10000.0),
            SharedVolumeParticipationLimit(model.volume_limit),
            ExecutionPrice.CLOSE,
        )
    if isinstance(model, slippage.VolumeShareSlippage):
        return (
            models.CumulativeVolumeShareSlippage(model.price_impact),
            SharedVolumeParticipationLimit(model.volume_limit),
            ExecutionPrice.CLOSE,
        )
    if isinstance(model, slippage.FixedSlippage):
        if not isfinite(model.spread) or model.spread < 0:
            raise ValueError("spread must be finite and nonnegative")
        return models.FixedSlippage(model.spread / 2.0), PositiveVolumeLimit(), ExecutionPrice.CLOSE
    if isinstance(model, slippage.NoSlippage):
        return models.NoSlippage(), PositiveVolumeLimit(), ExecutionPrice.CLOSE
    raise TypeError("Use a declared migration slippage model")


def compile_slippage(context, runtime, history_view, history_sources, default_limits):
    futures = [asset for asset in context._by_key.values() if isinstance(asset, Future)]
    if not futures:
        return None, default_limits
    history_sources = dict(history_sources or {})
    default, _, _ = _compile(context._slippage)

    @lru_cache(maxsize=128)
    def statistics(asset, instant):
        return _daily_statistics(runtime, asset, instant, history_view, history_sources)

    selected, limits, prices = {}, {}, {}
    for asset in futures:
        selected[asset.key], limits[asset.key], prices[asset.key] = _compile(
            context._futures_slippage, root=asset.root_symbol, statistics=statistics
        )
    context._config.metadata["migration_slippage"] = {
        "futures_method": type(context._futures_slippage).__name__,
        "price_sources": {key: value.value for key, value in prices.items()},
        "parameters": {
            name: getattr(context._futures_slippage, name)
            for name in ("basis_points", "volume_limit", "price_impact", "spread")
            if hasattr(context._futures_slippage, name)
        },
    }
    if isinstance(context._futures_slippage, slippage.VolatilityVolumeShare):
        context._config.metadata["migration_slippage"].update(
            {
                "statistics_history": (history_sources or {}).get("1d", "execution_grid"),
                "history_view": history_view,
                "window": "20_completed_sessions_or_observations_without_calendar",
                "annualization": 252,
                "missing_close_policy": "forward_fill_within_window_when_volume_is_known",
                "insufficient_history_bps": 7.5,
                "per_order_participation": context._futures_slippage.volume_limit,
                "total_bar_participation": 1.0,
                "eta": {key: model.eta for key, model in selected.items()},
            }
        )
    return (
        models.AssetSlippage(selected, default=default, price_sources=prices),
        AssetExecutionLimits(limits, default=default_limits),
    )
