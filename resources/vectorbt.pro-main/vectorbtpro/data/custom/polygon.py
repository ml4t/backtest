# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `PolygonData` class for fetching data from Polygon's API."""

import time
import traceback
from functools import wraps, partial

import pandas as pd
import requests

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.remote import RemoteData
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.pbar import ProgressBar
from vectorbtpro.utils.warnings_ import warn

if tp.TYPE_CHECKING:
    from polygon import RESTClient as PolygonClientT
else:
    PolygonClientT = "polygon.RESTClient"

__all__ = [
    "PolygonData",
]

PolygonDataT = tp.TypeVar("PolygonDataT", bound="PolygonData")


class PolygonData(RemoteData):
    """Data class for fetching data from Polygon's API.

    See:
        * https://github.com/polygon-io/client-python for the official Polygon Python client.
        * `PolygonData.fetch_symbol` for argument details.

    !!! info
        For default settings, see `custom.polygon` in `vectorbtpro._settings.data`.

    Examples:
        Set up the API key globally:

        ```pycon
        >>> from vectorbtpro import *

        >>> vbt.PolygonData.set_custom_settings(
        ...     client_config=dict(
        ...         api_key="YOUR_KEY"
        ...     )
        ... )
        ```

        Pull stock data:

        ```pycon
        >>> data = vbt.PolygonData.pull(
        ...     "AAPL",
        ...     start="2021-01-01",
        ...     end="2022-01-01",
        ...     timeframe="1 day"
        ... )
        ```

        Pull crypto data:

        ```pycon
        >>> data = vbt.PolygonData.pull(
        ...     "X:BTCUSD",
        ...     start="2021-01-01",
        ...     end="2022-01-01",
        ...     timeframe="1 day"
        ... )
        ```
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.polygon")

    @classmethod
    def list_symbols(
        cls,
        pattern: tp.Optional[str] = None,
        use_regex: bool = False,
        sort: bool = True,
        client: tp.Optional[PolygonClientT] = None,
        client_config: tp.DictLike = None,
        **list_tickers_kwargs,
    ) -> tp.List[str]:
        """List symbols from Polygon's API.

        Retrieves and filters available symbols by matching each against a specified pattern using
        `vectorbtpro.data.custom.custom.CustomData.key_match`.

        Args:
            pattern (Optional[str]): Pattern to filter symbols.

                Symbols that do not match this pattern are excluded.
            use_regex (bool): Flag indicating whether the pattern is a regular expression.
            sort (bool): Whether to return the symbols in sorted order.
            client (Optional[PolygonClient]): Polygon API client instance.
            client_config (KwargsLike): Configuration parameters for creating a new client.
            **list_tickers_kwargs: Keyword arguments for `polygon.RESTClient.list_tickers`.

        Returns:
            List[str]: List of symbols.
        """
        if client_config is None:
            client_config = {}
        client = cls.resolve_client(client=client, **client_config)
        all_symbols = []
        for ticker in client.list_tickers(**list_tickers_kwargs):
            symbol = ticker.ticker
            if pattern is not None:
                if not cls.key_match(symbol, pattern, use_regex=use_regex):
                    continue
            all_symbols.append(symbol)

        if sort:
            return sorted(dict.fromkeys(all_symbols))
        return list(dict.fromkeys(all_symbols))

    @classmethod
    def resolve_client(cls, client: tp.Optional[PolygonClientT] = None, **client_config) -> PolygonClientT:
        """Resolve the Polygon API client.

        If a client is provided, it must be an instance of `polygon.rest.RESTClient`.
        Otherwise, a new client is instantiated using the supplied `client_config`.

        Args:
            client (Optional[PolygonClient]): Polygon API client instance.
            **client_config: Configuration parameters for creating a new client.

        Returns:
            PolygonClient: Instance of `polygon.rest.RESTClient` for API interactions.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("polygon")
        from polygon import RESTClient

        client = cls.resolve_custom_setting(client, "client")
        if client_config is None:
            client_config = {}
        has_client_config = len(client_config) > 0
        client_config = cls.resolve_custom_setting(client_config, "client_config", merge=True)
        if client is None:
            client = RESTClient(**client_config)
        elif has_client_config:
            raise ValueError("Cannot apply client_config to already initialized client")
        return client

    @classmethod
    def fetch_symbol(
        cls,
        symbol: tp.Symbol,
        client: tp.Optional[PolygonClientT] = None,
        client_config: tp.KwargsLike = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        timeframe: tp.Optional[str] = None,
        tz: tp.TimezoneLike = None,
        adjusted: tp.Optional[bool] = None,
        limit: tp.Optional[int] = None,
        params: tp.KwargsLike = None,
        delay: tp.Optional[float] = None,
        retries: tp.Optional[int] = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        silence_warnings: tp.Optional[bool] = None,
    ) -> tp.SymbolData:
        """Overrides `vectorbtpro.data.base.Data.fetch_symbol` to fetch data for a given symbol from Polygon.

        Args:
            symbol (Symbol): Symbol identifier.

                Supports the following APIs:

                * Stocks and equities
                * Currencies — symbol must be prefixed with `C:`
                * Crypto — symbol must be prefixed with `X:`
            client (Optional[PolygonClient]): Polygon API client instance.

                See `PolygonData.resolve_client`.
            client_config (KwargsLike): Configuration parameters for creating a new client.

                See `PolygonData.resolve_client`.
            start (Optional[DatetimeLike]): Start datetime (e.g., "2024-01-01", "1 year ago").

                See `vectorbtpro.utils.datetime_.to_timestamp`.
            end (Optional[DatetimeLike]): End datetime (e.g., "2025-01-01", "now").

                See `vectorbtpro.utils.datetime_.to_timestamp`.
            timeframe (Optional[str]): Timeframe specification (e.g., "daily", "15 minutes").

                See `vectorbtpro.utils.datetime_.split_freq_str`.
            tz (TimezoneLike): Timezone specification (e.g., "UTC", "America/New_York").

                See `vectorbtpro.utils.datetime_.to_timezone`.
            adjusted (Optional[bool]): Whether to adjust data for splits.

                Set to False to return unadjusted data.
            limit (Optional[int]): Limits the number of base aggregates queried.

                Maximum allowed is 50000.
            params (DictLike): Additional query parameters.
            delay (Optional[float]): Delay in seconds between requests.
            retries (Optional[int]): Number of retries on failure to fetch data.
            show_progress (Optional[bool]): Flag indicating whether to display the progress bar.
            pbar_kwargs (DictLike): Keyword arguments for `vectorbtpro.utils.pbar.ProgressBar`.
            silence_warnings (Optional[bool]): Flag to suppress warning messages.

        Returns:
            SymbolData: Updated data and a metadata dictionary.

        !!! note
            If you're using a free plan with a rate limit of several requests per minute,
            set `delay` to a higher value (e.g., 12 to allow 5 requests per minute).
        """
        if client_config is None:
            client_config = {}
        client = cls.resolve_client(client=client, **client_config)

        start = cls.resolve_custom_setting(start, "start")
        end = cls.resolve_custom_setting(end, "end")
        timeframe = cls.resolve_custom_setting(timeframe, "timeframe")
        tz = cls.resolve_custom_setting(tz, "tz")
        adjusted = cls.resolve_custom_setting(adjusted, "adjusted")
        limit = cls.resolve_custom_setting(limit, "limit")
        params = cls.resolve_custom_setting(params, "params", merge=True)
        delay = cls.resolve_custom_setting(delay, "delay")
        retries = cls.resolve_custom_setting(retries, "retries")
        show_progress = cls.resolve_custom_setting(show_progress, "show_progress")
        pbar_kwargs = cls.resolve_custom_setting(pbar_kwargs, "pbar_kwargs", merge=True)
        if "bar_id" not in pbar_kwargs:
            pbar_kwargs["bar_id"] = "polygon"
        silence_warnings = cls.resolve_custom_setting(silence_warnings, "silence_warnings")

        # Resolve the timeframe
        if not isinstance(timeframe, str):
            raise ValueError(f"Invalid timeframe: '{timeframe}'")
        split = dt.split_freq_str(timeframe)
        if split is None:
            raise ValueError(f"Invalid timeframe: '{timeframe}'")
        multiplier, unit = split
        if unit == "s":
            unit = "second"
        elif unit == "m":
            unit = "minute"
        elif unit == "h":
            unit = "hour"
        elif unit == "D":
            unit = "day"
        elif unit == "W":
            unit = "week"
        elif unit == "M":
            unit = "month"
        elif unit == "Q":
            unit = "quarter"
        elif unit == "Y":
            unit = "year"

        # Establish the timestamps
        if start is not None:
            start_ts = dt.datetime_to_ms(dt.to_tzaware_datetime(start, naive_tz=tz, tz="utc"))
        else:
            start_ts = None
        if end is not None:
            end_ts = dt.datetime_to_ms(dt.to_tzaware_datetime(end, naive_tz=tz, tz="utc"))
        else:
            end_ts = None
        prev_end_ts = None

        def _retry(method):
            @wraps(method)
            def retry_method(*args, **kwargs) -> tp.Any:
                for i in range(retries):
                    try:
                        return method(*args, **kwargs)
                    except requests.exceptions.HTTPError as e:
                        if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 429:
                            if not silence_warnings:
                                warn(traceback.format_exc())
                                # Polygon.io API rate limit is per minute
                                warn("Waiting 1 minute...")
                            time.sleep(60)
                        else:
                            raise e
                    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                        if i == retries - 1:
                            raise e
                        if not silence_warnings:
                            warn(traceback.format_exc())
                        if delay is not None:
                            time.sleep(delay)

            return retry_method

        def _postprocess(agg):
            return dict(
                o=agg.open,
                h=agg.high,
                l=agg.low,
                c=agg.close,
                v=agg.volume,
                vw=agg.vwap,
                t=agg.timestamp,
                n=agg.transactions,
            )

        @_retry
        def _fetch(_start_ts, _limit):
            return list(
                map(
                    _postprocess,
                    client.get_aggs(
                        ticker=symbol,
                        multiplier=multiplier,
                        timespan=unit,
                        from_=_start_ts,
                        to=end_ts,
                        adjusted=adjusted,
                        sort="asc",
                        limit=_limit,
                        params=params,
                        raw=False,
                    ),
                )
            )

        def _ts_to_str(ts: tp.Optional[int]) -> str:
            if ts is None:
                return "?"
            return dt.readable_datetime(pd.Timestamp(ts, unit="ms", tz="utc"), freq=timeframe)

        def _filter_func(d: tp.Dict, _prev_end_ts: tp.Optional[int] = None) -> bool:
            if start_ts is not None:
                if d["t"] < start_ts:
                    return False
            if _prev_end_ts is not None:
                if d["t"] <= _prev_end_ts:
                    return False
            if end_ts is not None:
                if d["t"] >= end_ts:
                    return False
            return True

        # Iteratively collect the data
        data = []
        try:
            with ProgressBar(show_progress=show_progress, **pbar_kwargs) as pbar:
                pbar.set_description("{} → ?".format(_ts_to_str(start_ts if prev_end_ts is None else prev_end_ts)))
                while True:
                    # Fetch the klines for the next timeframe
                    next_data = _fetch(start_ts if prev_end_ts is None else prev_end_ts, limit)
                    next_data = list(filter(partial(_filter_func, _prev_end_ts=prev_end_ts), next_data))

                    # Update the timestamps and the progress bar
                    if not len(next_data):
                        break
                    data += next_data
                    if start_ts is None:
                        start_ts = next_data[0]["t"]
                    pbar.set_description("{} → {}".format(_ts_to_str(start_ts), _ts_to_str(next_data[-1]["t"])))
                    pbar.update()
                    prev_end_ts = next_data[-1]["t"]
                    if end_ts is not None and prev_end_ts >= end_ts:
                        break
                    if delay is not None:
                        time.sleep(delay)  # be kind to api
        except Exception:
            if not silence_warnings:
                warn(traceback.format_exc())
                warn(
                    f"Symbol '{str(symbol)}' raised an exception. Returning incomplete data. "
                    "Use update() method to fetch missing data."
                )

        df = pd.DataFrame(data)
        df = df[["t", "o", "h", "l", "c", "v", "n", "vw"]]
        df = df.rename(
            columns={
                "t": "Open time",
                "o": "Open",
                "h": "High",
                "l": "Low",
                "c": "Close",
                "v": "Volume",
                "n": "Trade count",
                "vw": "VWAP",
            }
        )
        df.index = pd.to_datetime(df["Open time"], unit="ms", utc=True)
        del df["Open time"]
        if "Open" in df.columns:
            df["Open"] = df["Open"].astype(float)
        if "High" in df.columns:
            df["High"] = df["High"].astype(float)
        if "Low" in df.columns:
            df["Low"] = df["Low"].astype(float)
        if "Close" in df.columns:
            df["Close"] = df["Close"].astype(float)
        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].astype(float)
        if "Trade count" in df.columns:
            df["Trade count"] = df["Trade count"].astype(int, errors="ignore")
        if "VWAP" in df.columns:
            df["VWAP"] = df["VWAP"].astype(float)

        return df, dict(tz=tz, freq=timeframe)

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        fetch_kwargs = self.select_fetch_kwargs(symbol)
        fetch_kwargs["start"] = self.select_last_index(symbol)
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)
