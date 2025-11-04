# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `BinanceData` class for fetching data from Binance using the Python Binance API."""

import time
import traceback
from functools import partial

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.remote import RemoteData
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.config import merge_dicts, Config, HybridConfig
from vectorbtpro.utils.enum_ import map_enum_fields
from vectorbtpro.utils.pbar import ProgressBar
from vectorbtpro.utils.warnings_ import warn

if tp.TYPE_CHECKING:
    from binance.client import Client as ClientT
else:
    ClientT = "binance.client.Client"

__all__ = [
    "BinanceData",
]

__pdoc__ = {}

BinanceDataT = tp.TypeVar("BinanceDataT", bound="BinanceData")


class BinanceData(RemoteData):
    """Data class for fetching data from Binance using the Python Binance API.

    See:
        * https://github.com/sammchardy/python-binance for the API client.
        * `BinanceData.fetch_symbol` for argument details.

    !!! info
        For default settings, see `custom.binance` in `vectorbtpro._settings.data`.

    !!! note
        If using an exchange from the US, Japan, or another TLD, pass `tld="us"` in
        `client_config` when creating the client.

    Examples:
        Set up the API key globally (optional):

        ```pycon
        >>> from vectorbtpro import *

        >>> vbt.BinanceData.set_custom_settings(
        ...     client_config=dict(
        ...         api_key="YOUR_KEY",
        ...         api_secret="YOUR_SECRET"
        ...     )
        ... )
        ```

        Pull data:

        ```pycon
        >>> data = vbt.BinanceData.pull(
        ...     "BTCUSDT",
        ...     start="2020-01-01",
        ...     end="2021-01-01",
        ...     timeframe="1 day"
        ... )
        ```
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.binance")

    _feature_config: tp.ClassVar[Config] = HybridConfig(
        {
            "Quote volume": dict(
                resample_func=lambda self, obj, resampler: obj.vbt.resample_apply(
                    resampler,
                    generic_nb.sum_reduce_nb,
                )
            ),
            "Taker base volume": dict(
                resample_func=lambda self, obj, resampler: obj.vbt.resample_apply(
                    resampler,
                    generic_nb.sum_reduce_nb,
                )
            ),
            "Taker quote volume": dict(
                resample_func=lambda self, obj, resampler: obj.vbt.resample_apply(
                    resampler,
                    generic_nb.sum_reduce_nb,
                )
            ),
        }
    )

    @property
    def feature_config(self) -> Config:
        return self._feature_config

    @classmethod
    def resolve_client(cls, client: tp.Optional[ClientT] = None, **client_config) -> ClientT:
        """Resolve and return a Binance client instance.

        If a client is provided, it must be an instance of `binance.client.Client`.
        Otherwise, a new client is created using `client_config`.

        Args:
            client (Optional[Client]): Binance client instance.
            client_config (KwargsLike): Configuration parameters for creating a new client.

        Returns:
            Client: Resolved or newly created Binance client.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("binance")
        from binance.client import Client

        client = cls.resolve_custom_setting(client, "client")
        if client_config is None:
            client_config = {}
        has_client_config = len(client_config) > 0
        client_config = cls.resolve_custom_setting(client_config, "client_config", merge=True)
        if client is None:
            client = Client(**client_config)
        elif has_client_config:
            raise ValueError("Cannot apply client_config to already initialized client")
        return client

    @classmethod
    def list_symbols(
        cls,
        pattern: tp.Optional[str] = None,
        use_regex: bool = False,
        sort: bool = True,
        client: tp.Optional[ClientT] = None,
        client_config: tp.KwargsLike = None,
    ) -> tp.List[str]:
        """List and return all Binance symbols.

        Retrieves symbol data from the Binance API and filters the symbols using
        `vectorbtpro.data.custom.custom.CustomData.key_match` if a pattern is provided.

        Args:
            pattern (Optional[str]): Pattern to filter symbols.
            use_regex (bool): Flag indicating whether the pattern is a regular expression.
            sort (bool): Whether to return the symbols in sorted order.
            client (Optional[Client]): Binance client instance.
            client_config (KwargsLike): Configuration parameters for creating a new client.

        Returns:
            List[str]: List of Binance symbols.
        """
        if client_config is None:
            client_config = {}
        client = cls.resolve_client(client=client, **client_config)
        all_symbols = []
        for dct in client.get_exchange_info()["symbols"]:
            symbol = dct["symbol"]
            if pattern is not None:
                if not cls.key_match(symbol, pattern, use_regex=use_regex):
                    continue
            all_symbols.append(symbol)

        if sort:
            return sorted(dict.fromkeys(all_symbols))
        return list(dict.fromkeys(all_symbols))

    @classmethod
    def fetch_symbol(
        cls,
        symbol: tp.Symbol,
        client: tp.Optional[ClientT] = None,
        client_config: tp.KwargsLike = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        timeframe: tp.Optional[str] = None,
        tz: tp.TimezoneLike = None,
        klines_type: tp.Union[None, int, str] = None,
        limit: tp.Optional[int] = None,
        delay: tp.Optional[float] = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        silence_warnings: tp.Optional[bool] = None,
        **get_klines_kwargs,
    ) -> tp.SymbolData:
        """Fetch symbol data from Binance.

        Args:
            symbol (Symbol): Symbol identifier.
            client (Optional[Client]): Binance client instance.

                See `BinanceData.resolve_client`.
            client_config (KwargsLike): Configuration parameters for creating a new client.

                See `BinanceData.resolve_client`.
            start (Optional[DatetimeLike]): Start datetime (e.g., "2024-01-01", "1 year ago").

                See `vectorbtpro.utils.datetime_.to_timestamp`.
            end (Optional[DatetimeLike]): End datetime (e.g., "2025-01-01", "now").

                See `vectorbtpro.utils.datetime_.to_timestamp`.
            timeframe (Optional[str]): Timeframe specification (e.g., "daily", "15 minutes").

                See `vectorbtpro.utils.datetime_.split_freq_str`.
            tz (TimezoneLike): Timezone specification (e.g., "UTC", "America/New_York").

                See `vectorbtpro.utils.datetime_.to_timezone`.
            klines_type (Union[None, int, str]): Type of klines to fetch.

                Mapped using `binance.enums.HistoricalKlinesType` if provided as a string.
            limit (Optional[int]): Maximum number of klines to retrieve per API call.
            delay (Optional[float]): Delay in seconds between requests.
            show_progress (Optional[bool]): Flag indicating whether to display the progress bar.
            pbar_kwargs (KwargsLike): Keyword arguments for configuring the progress bar.

                See `vectorbtpro.utils.pbar.ProgressBar`.
            silence_warnings (Optional[bool]): Flag to suppress warning messages.
            **get_klines_kwargs: Keyword arguments for `binance.client.Client.get_klines`.

        Returns:
            SymbolData: Fetched data and a metadata dictionary.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("binance")
        from binance.enums import HistoricalKlinesType

        if client_config is None:
            client_config = {}
        client = cls.resolve_client(client=client, **client_config)

        start = cls.resolve_custom_setting(start, "start")
        end = cls.resolve_custom_setting(end, "end")
        timeframe = cls.resolve_custom_setting(timeframe, "timeframe")
        tz = cls.resolve_custom_setting(tz, "tz")
        klines_type = cls.resolve_custom_setting(klines_type, "klines_type")
        if isinstance(klines_type, str):
            klines_type = map_enum_fields(klines_type, HistoricalKlinesType)
        if isinstance(klines_type, int):
            klines_type = {i.value: i for i in HistoricalKlinesType}[klines_type]
        limit = cls.resolve_custom_setting(limit, "limit")
        delay = cls.resolve_custom_setting(delay, "delay")
        show_progress = cls.resolve_custom_setting(show_progress, "show_progress")
        pbar_kwargs = cls.resolve_custom_setting(pbar_kwargs, "pbar_kwargs", merge=True)
        if "bar_id" not in pbar_kwargs:
            pbar_kwargs["bar_id"] = "binance"
        silence_warnings = cls.resolve_custom_setting(silence_warnings, "silence_warnings")
        get_klines_kwargs = cls.resolve_custom_setting(get_klines_kwargs, "get_klines_kwargs", merge=True)

        # Prepare parameters
        freq = timeframe
        split = dt.split_freq_str(timeframe)
        if split is not None:
            multiplier, unit = split
            if unit == "D":
                unit = "d"
            elif unit == "W":
                unit = "w"
            timeframe = str(multiplier) + unit
        if start is not None:
            start_ts = dt.datetime_to_ms(dt.to_tzaware_datetime(start, naive_tz=tz, tz="utc"))
            first_valid_ts = client._get_earliest_valid_timestamp(symbol, timeframe, klines_type)
            start_ts = max(start_ts, first_valid_ts)
        else:
            start_ts = None
        prev_end_ts = None
        if end is not None:
            end_ts = dt.datetime_to_ms(dt.to_tzaware_datetime(end, naive_tz=tz, tz="utc"))
        else:
            end_ts = None

        def _ts_to_str(ts: tp.Optional[int]) -> str:
            if ts is None:
                return "?"
            return dt.readable_datetime(pd.Timestamp(ts, unit="ms", tz="utc"), freq=timeframe)

        def _filter_func(d: tp.Sequence, _prev_end_ts: tp.Optional[int] = None) -> bool:
            if start_ts is not None:
                if d[0] < start_ts:
                    return False
            if _prev_end_ts is not None:
                if d[0] <= _prev_end_ts:
                    return False
            if end_ts is not None:
                if d[0] >= end_ts:
                    return False
            return True

        # Iteratively collect the data
        data = []
        try:
            with ProgressBar(show_progress=show_progress, **pbar_kwargs) as pbar:
                pbar.set_description("{} → ?".format(_ts_to_str(start_ts if prev_end_ts is None else prev_end_ts)))
                while True:
                    # Fetch the klines for the next timeframe
                    next_data = client._klines(
                        symbol=symbol,
                        interval=timeframe,
                        limit=limit,
                        startTime=start_ts if prev_end_ts is None else prev_end_ts,
                        endTime=end_ts,
                        klines_type=klines_type,
                        **get_klines_kwargs,
                    )
                    next_data = list(filter(partial(_filter_func, _prev_end_ts=prev_end_ts), next_data))

                    # Update the timestamps and the progress bar
                    if not len(next_data):
                        break
                    data += next_data
                    if start_ts is None:
                        start_ts = next_data[0][0]
                    pbar.set_description("{} → {}".format(_ts_to_str(start_ts), _ts_to_str(next_data[-1][0])))
                    pbar.update()
                    prev_end_ts = next_data[-1][0]
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

        # Convert data to a DataFrame
        df = pd.DataFrame(
            data,
            columns=[
                "Open time",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Close time",
                "Quote volume",
                "Trade count",
                "Taker base volume",
                "Taker quote volume",
                "Ignore",
            ],
        )
        df.index = pd.to_datetime(df["Open time"], unit="ms", utc=True)
        df["Open"] = df["Open"].astype(float)
        df["High"] = df["High"].astype(float)
        df["Low"] = df["Low"].astype(float)
        df["Close"] = df["Close"].astype(float)
        df["Volume"] = df["Volume"].astype(float)
        df["Quote volume"] = df["Quote volume"].astype(float)
        df["Trade count"] = df["Trade count"].astype(int, errors="ignore")
        df["Taker base volume"] = df["Taker base volume"].astype(float)
        df["Taker quote volume"] = df["Taker quote volume"].astype(float)
        del df["Open time"]
        del df["Close time"]
        del df["Ignore"]

        return df, dict(tz=tz, freq=freq)

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        fetch_kwargs = self.select_fetch_kwargs(symbol)
        fetch_kwargs["start"] = self.select_last_index(symbol)
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)


BinanceData.override_feature_config_doc(__pdoc__)
