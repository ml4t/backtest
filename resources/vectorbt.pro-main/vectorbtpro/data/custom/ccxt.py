# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `CCXTData` class for fetching data using the CCXT API."""

import time
import traceback
from functools import wraps, partial

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.remote import RemoteData
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.pbar import ProgressBar
from vectorbtpro.utils.warnings_ import warn

if tp.TYPE_CHECKING:
    from ccxt.base.exchange import Exchange as ExchangeT
else:
    ExchangeT = "ccxt.base.exchange.Exchange"

__all__ = [
    "CCXTData",
]

__pdoc__ = {}


class CCXTData(RemoteData):
    """Data class for fetching data using CCXT.

    This class provides functionality to retrieve market data via the CCXT API.

    See:
        * https://github.com/ccxt/ccxt for more information on CCXT.
        * `CCXTData.fetch_symbol` for argument details.

    !!! info
        For default settings, see `custom.ccxt` in `vectorbtpro._settings.data`.

        Global settings can be configured per exchange identifier using the `exchanges` dictionary.

    Examples:
        Set up the API key globally (optional):

        ```pycon
        >>> from vectorbtpro import *

        >>> vbt.CCXTData.set_exchange_settings(
        ...     exchange_name="binance",
        ...     populate_=True,
        ...     exchange_config=dict(
        ...         apiKey="YOUR_KEY",
        ...         secret="YOUR_SECRET"
        ...     )
        ... )
        ```

        Pull data:

        ```pycon
        >>> data = vbt.CCXTData.pull(
        ...     "BTCUSDT",
        ...     exchange="binance",
        ...     start="2020-01-01",
        ...     end="2021-01-01",
        ...     timeframe="1 day"
        ... )
        ```
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.ccxt")

    @classmethod
    def get_exchange_settings(cls, *args, exchange_name: tp.Optional[str] = None, **kwargs) -> dict:
        """Return custom exchange settings using a sub-path based on the exchange name.

        Args:
            *args: Positional arguments for `CCXTData.get_custom_settings`.
            exchange_name (Optional[str]): Name of the exchange used to form the sub-path.
            **kwargs: Keyword arguments for `CCXTData.get_custom_settings`.

        Returns:
            dict: Custom exchange settings.
        """
        if exchange_name is not None:
            sub_path = "exchanges." + exchange_name
        else:
            sub_path = None
        return cls.get_custom_settings(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def has_exchange_settings(cls, *args, exchange_name: tp.Optional[str] = None, **kwargs) -> bool:
        """Return whether custom exchange settings exist using a sub-path derived from the exchange name.

        Args:
            *args: Positional arguments for `CCXTData.has_custom_settings`.
            exchange_name (Optional[str]): Name of the exchange used to form the sub-path.
            **kwargs: Keyword arguments for `CCXTData.has_custom_settings`.

        Returns:
            bool: True if custom exchange settings exist, False otherwise.
        """
        if exchange_name is not None:
            sub_path = "exchanges." + exchange_name
        else:
            sub_path = None
        return cls.has_custom_settings(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def get_exchange_setting(cls, *args, exchange_name: tp.Optional[str] = None, **kwargs) -> tp.Any:
        """Return a custom exchange setting using a sub-path determined by the exchange name.

        Args:
            *args: Positional arguments for `CCXTData.get_custom_setting`.
            exchange_name (Optional[str]): Name of the exchange used to form the sub-path.
            **kwargs: Keyword arguments for `CCXTData.get_custom_setting`.

        Returns:
            Any: Retrieved custom exchange setting.
        """
        if exchange_name is not None:
            sub_path = "exchanges." + exchange_name
        else:
            sub_path = None
        return cls.get_custom_setting(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def has_exchange_setting(cls, *args, exchange_name: tp.Optional[str] = None, **kwargs) -> bool:
        """Return whether a custom exchange setting exists for the specified exchange name via a sub-path.

        Args:
            *args: Positional arguments for `CCXTData.has_custom_setting`.
            exchange_name (Optional[str]): Name of the exchange used to form the sub-path.
            **kwargs: Keyword arguments for `CCXTData.has_custom_setting`.

        Returns:
            bool: True if the custom exchange setting exists, False otherwise.
        """
        if exchange_name is not None:
            sub_path = "exchanges." + exchange_name
        else:
            sub_path = None
        return cls.has_custom_setting(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def resolve_exchange_setting(cls, *args, exchange_name: tp.Optional[str] = None, **kwargs) -> tp.Any:
        """Resolve a custom exchange setting using a sub-path determined by the exchange name.

        Args:
            *args: Positional arguments for `CCXTData.resolve_custom_setting`.
            exchange_name (Optional[str]): Name of the exchange used to form the sub-path.
            **kwargs: Keyword arguments for `CCXTData.resolve_custom_setting`.

        Returns:
            Any: Resolved custom exchange setting.
        """
        if exchange_name is not None:
            sub_path = "exchanges." + exchange_name
        else:
            sub_path = None
        return cls.resolve_custom_setting(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def set_exchange_settings(cls, *args, exchange_name: tp.Optional[str] = None, **kwargs) -> None:
        """Set custom exchange settings using a sub-path based on the provided exchange name.

        Args:
            *args: Positional arguments for `CCXTData.set_custom_settings`.
            exchange_name (Optional[str]): Name of the exchange used to form the sub-path.
            **kwargs: Keyword arguments for `CCXTData.set_custom_settings`.

        Returns:
            None
        """
        if exchange_name is not None:
            sub_path = "exchanges." + exchange_name
        else:
            sub_path = None
        cls.set_custom_settings(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def list_symbols(
        cls,
        pattern: tp.Optional[str] = None,
        use_regex: bool = False,
        sort: bool = True,
        exchange: tp.Union[None, str, ExchangeT] = None,
        exchange_config: tp.KwargsLike = None,
    ) -> tp.List[str]:
        """List all symbols matching an optional pattern.

        Uses `vectorbtpro.data.custom.custom.CustomData.key_match` to filter symbols by the specified pattern.

        Args:
            pattern (Optional[str]): Pattern to filter symbols.
            use_regex (bool): Flag indicating whether the pattern is a regular expression.
            sort (bool): Whether to return the symbols in sorted order.
            exchange (Union[None, str, Exchange]): Exchange identifier or instance.
            exchange_config (Optional[KwargsLike]): Additional configurations for the exchange.

        Returns:
            List[str]: List of matching symbols.
        """
        if exchange_config is None:
            exchange_config = {}
        exchange = cls.resolve_exchange(exchange=exchange, **exchange_config)
        all_symbols = []
        for symbol in exchange.load_markets():
            if pattern is not None:
                if not cls.key_match(symbol, pattern, use_regex=use_regex):
                    continue
            all_symbols.append(symbol)

        if sort:
            return sorted(dict.fromkeys(all_symbols))
        return list(dict.fromkeys(all_symbols))

    @classmethod
    def resolve_exchange(
        cls,
        exchange: tp.Union[None, str, ExchangeT] = None,
        **exchange_config,
    ) -> ExchangeT:
        """Resolve the exchange instance.

        If an exchange instance is provided, it must be of type `ccxt.base.exchange.Exchange`.
        Otherwise, an instance is created using the provided exchange configuration.

        Args:
            exchange (Union[None, str, Exchange]): Exchange identifier or instance.
            **exchange_config: Additional parameters for configuring the exchange.

        Returns:
            Exchange: Resolved exchange instance.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("ccxt")
        import ccxt

        exchange = cls.resolve_exchange_setting(exchange, "exchange")
        if exchange is None:
            exchange = "binance"
        if isinstance(exchange, str):
            exchange = exchange.lower()
            exchange_name = exchange
        elif isinstance(exchange, ccxt.Exchange):
            exchange_name = type(exchange).__name__
        else:
            raise ValueError(f"Unknown exchange of type {type(exchange)}")
        if exchange_config is None:
            exchange_config = {}
        has_exchange_config = len(exchange_config) > 0
        exchange_config = cls.resolve_exchange_setting(
            exchange_config, "exchange_config", merge=True, exchange_name=exchange_name
        )
        if isinstance(exchange, str):
            if not hasattr(ccxt, exchange):
                raise ValueError(f"Exchange '{exchange}' not found in CCXT")
            exchange = getattr(ccxt, exchange)(exchange_config)
        else:
            if has_exchange_config:
                raise ValueError("Cannot apply config after instantiation of the exchange")
        return exchange

    @staticmethod
    def fetch_find_earliest_date(
        fetch_func: tp.Callable,
        start: tp.DatetimeLike = 0,
        end: tp.DatetimeLike = "now",
        tz: tp.TimezoneLike = None,
        for_internal_use: bool = False,
    ) -> tp.Optional[tp.Timestamp]:
        """Find the earliest timestamp using binary search by calling the provided fetch function.

        Args:
            fetch_func (Callable): Function to fetch data.
            start (Optional[DatetimeLike]): Start datetime (e.g., "2024-01-01", "1 year ago").

                See `vectorbtpro.utils.datetime_.to_timestamp`.
            end (Optional[DatetimeLike]): End datetime (e.g., "2025-01-01", "now").

                See `vectorbtpro.utils.datetime_.to_timestamp`.
            tz (TimezoneLike): Timezone specification (e.g., "UTC", "America/New_York").

                See `vectorbtpro.utils.datetime_.to_timezone`.
            for_internal_use (bool): Flag indicating whether the search is for internal processing.

        Returns:
            Optional[Timestamp]: Earliest timestamp if data is available, otherwise None.
        """
        if start is not None:
            start_ts = dt.datetime_to_ms(dt.to_tzaware_datetime(start, naive_tz=tz, tz="utc"))
            fetched_data = fetch_func(start_ts, 1)
            if for_internal_use and len(fetched_data) > 0:
                return pd.Timestamp(start_ts, unit="ms", tz="utc")
        else:
            fetched_data = []
        if len(fetched_data) == 0 and start != 0:
            fetched_data = fetch_func(0, 1)
            if for_internal_use and len(fetched_data) > 0:
                return pd.Timestamp(0, unit="ms", tz="utc")
        if len(fetched_data) == 0:
            if start is not None:
                start_ts = dt.datetime_to_ms(dt.to_tzaware_datetime(start, naive_tz=tz, tz="utc"))
            else:
                start_ts = dt.datetime_to_ms(dt.to_tzaware_datetime(0, naive_tz=tz, tz="utc"))
            start_ts = start_ts - start_ts % 86400000
            if end is not None:
                end_ts = dt.datetime_to_ms(dt.to_tzaware_datetime(end, naive_tz=tz, tz="utc"))
            else:
                end_ts = dt.datetime_to_ms(dt.to_tzaware_datetime("now", naive_tz=tz, tz="utc"))
            end_ts = end_ts - end_ts % 86400000 + 86400000
            start_time = start_ts
            end_time = end_ts
            while True:
                mid_time = (start_time + end_time) // 2
                mid_time = mid_time - mid_time % 86400000
                if mid_time == start_time:
                    break
                _fetched_data = fetch_func(mid_time, 1)
                if len(_fetched_data) == 0:
                    start_time = mid_time
                else:
                    end_time = mid_time
                    fetched_data = _fetched_data
        if len(fetched_data) > 0:
            return pd.Timestamp(fetched_data[0][0], unit="ms", tz="utc")
        return None

    @classmethod
    def find_earliest_date(
        cls,
        symbol: tp.Symbol,
        for_internal_use: bool = False,
        **kwargs,
    ) -> tp.Optional[tp.Timestamp]:
        """Find the earliest timestamp for a symbol using binary search.

        This method calls `CCXTData.fetch_symbol` to retrieve data for the specified symbol,
        then performs a binary search to determine the earliest available date.

        Args:
            symbol (Symbol): Symbol identifier.
            for_internal_use (bool): Flag indicating whether the search is for internal processing.
            **kwargs: Keyword arguments for `CCXTData.fetch_symbol`.

        Returns:
            Optional[Timestamp]: Earliest timestamp if data is available, otherwise None.
        """
        return cls.fetch_find_earliest_date(
            **cls.fetch_symbol(symbol, return_fetch_method=True, **kwargs),
            for_internal_use=for_internal_use,
        )

    @classmethod
    def fetch_symbol(
        cls,
        symbol: tp.Symbol,
        exchange: tp.Union[None, str, ExchangeT] = None,
        exchange_config: tp.KwargsLike = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        timeframe: tp.Optional[str] = None,
        tz: tp.TimezoneLike = None,
        find_earliest_date: tp.Optional[bool] = None,
        limit: tp.Optional[int] = None,
        delay: tp.Optional[float] = None,
        retries: tp.Optional[int] = None,
        fetch_params: tp.KwargsLike = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        silence_warnings: tp.Optional[bool] = None,
        return_fetch_method: bool = False,
    ) -> tp.Union[dict, tp.SymbolData]:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch symbol data from a CCXT exchange.

        Args:
            symbol (Symbol): Symbol identifier.

                Symbol may be provided in the `EXCHANGE:SYMBOL` format.
                In this case, the `exchange` parameter is ignored.
            exchange (Union[None, str, Exchange]): Exchange identifier or instance.

                See `CCXTData.resolve_exchange`.
            exchange_config (KwargsLike): Exchange configuration.

                See `CCXTData.resolve_exchange`.
            start (Optional[DatetimeLike]): Start datetime (e.g., "2024-01-01", "1 year ago").

                See `vectorbtpro.utils.datetime_.to_timestamp`.
            end (Optional[DatetimeLike]): End datetime (e.g., "2025-01-01", "now").

                See `vectorbtpro.utils.datetime_.to_timestamp`.
            timeframe (Optional[str]): Timeframe specification (e.g., "daily", "15 minutes").

                See `vectorbtpro.utils.datetime_.split_freq_str`.
            tz (TimezoneLike): Timezone specification (e.g., "UTC", "America/New_York").

                See `vectorbtpro.utils.datetime_.to_timezone`.
            find_earliest_date (Optional[bool]): Flag to determine the earliest available date.

                Uses `CCXTData.find_earliest_date` when enabled.
            limit (Optional[int]): Maximum number of data items to return.
            delay (Optional[float]): Delay in seconds between requests.

                !!! note
                    Use only if `enableRateLimit` is not set.
            retries (Optional[int]): Number of retries on failure to fetch data.
            fetch_params (KwargsLike): Exchange-specific parameters for `fetch_ohlcv`.
            show_progress (Optional[bool]): Flag indicating whether to display the progress bar.
            pbar_kwargs (KwargsLike): Keyword arguments for configuring the progress bar.

                See `vectorbtpro.utils.pbar.ProgressBar`.
            silence_warnings (Optional[bool]): Flag to suppress warning messages.
            return_fetch_method (bool): If True, returns the fetch method and settings instead of fetched data.

        Returns:
            Union[dict, SymbolData]: If `return_params` is True, returns the metadata around the fetching function.
                Otherwise, returns the fetched data and a metadata dictionary.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("ccxt")
        import ccxt

        exchange = cls.resolve_custom_setting(exchange, "exchange")
        if exchange is None and ":" in symbol:
            exchange, symbol = symbol.split(":")
        if exchange_config is None:
            exchange_config = {}
        exchange = cls.resolve_exchange(exchange=exchange, **exchange_config)
        exchange_name = type(exchange).__name__

        start = cls.resolve_exchange_setting(start, "start", exchange_name=exchange_name)
        end = cls.resolve_exchange_setting(end, "end", exchange_name=exchange_name)
        timeframe = cls.resolve_exchange_setting(timeframe, "timeframe", exchange_name=exchange_name)
        tz = cls.resolve_exchange_setting(tz, "tz", exchange_name=exchange_name)
        find_earliest_date = cls.resolve_exchange_setting(
            find_earliest_date, "find_earliest_date", exchange_name=exchange_name
        )
        limit = cls.resolve_exchange_setting(limit, "limit", exchange_name=exchange_name)
        delay = cls.resolve_exchange_setting(delay, "delay", exchange_name=exchange_name)
        retries = cls.resolve_exchange_setting(retries, "retries", exchange_name=exchange_name)
        fetch_params = cls.resolve_exchange_setting(
            fetch_params, "fetch_params", merge=True, exchange_name=exchange_name
        )
        show_progress = cls.resolve_exchange_setting(show_progress, "show_progress", exchange_name=exchange_name)
        pbar_kwargs = cls.resolve_exchange_setting(pbar_kwargs, "pbar_kwargs", merge=True, exchange_name=exchange_name)
        if "bar_id" not in pbar_kwargs:
            pbar_kwargs["bar_id"] = "ccxt"
        silence_warnings = cls.resolve_exchange_setting(
            silence_warnings, "silence_warnings", exchange_name=exchange_name
        )
        if not exchange.has["fetchOHLCV"]:
            raise ValueError(f"Exchange {exchange} does not support OHLCV")
        if exchange.has["fetchOHLCV"] == "emulated":
            if not silence_warnings:
                warn("Using emulated OHLCV candles")

        freq = timeframe
        split = dt.split_freq_str(timeframe)
        if split is not None:
            multiplier, unit = split
            if unit == "D":
                unit = "d"
            elif unit == "W":
                unit = "w"
            elif unit == "Y":
                unit = "y"
            timeframe = str(multiplier) + unit
        if timeframe not in exchange.timeframes:
            raise ValueError(f"Exchange {exchange} does not support {timeframe} timeframe")

        def _retry(method):
            @wraps(method)
            def retry_method(*args, **kwargs) -> tp.Any:
                for i in range(retries):
                    try:
                        return method(*args, **kwargs)
                    except ccxt.NetworkError as e:
                        if i == retries - 1:
                            raise e
                        if not silence_warnings:
                            warn(traceback.format_exc())
                        if delay is not None:
                            time.sleep(delay)

            return retry_method

        @_retry
        def _fetch(_since, _limit):
            return exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=_since,
                limit=_limit,
                params=fetch_params,
            )

        if return_fetch_method:
            return dict(fetch_func=_fetch, start=start, end=end, tz=tz)

        # Establish the timestamps
        if find_earliest_date and start is not None:
            start = cls.fetch_find_earliest_date(_fetch, start=start, end=end, tz=tz, for_internal_use=True)
        if start is not None:
            start_ts = dt.datetime_to_ms(dt.to_tzaware_datetime(start, naive_tz=tz, tz="utc"))
        else:
            start_ts = None
        if end is not None:
            end_ts = dt.datetime_to_ms(dt.to_tzaware_datetime(end, naive_tz=tz, tz="UTC"))
        else:
            end_ts = None
        prev_end_ts = None

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
                    next_data = _fetch(start_ts if prev_end_ts is None else prev_end_ts, limit)
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
        df = pd.DataFrame(data, columns=["Open time", "Open", "High", "Low", "Close", "Volume"])
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

        return df, dict(tz=tz, freq=freq)

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        fetch_kwargs = self.select_fetch_kwargs(symbol)
        fetch_kwargs["start"] = self.select_last_index(symbol)
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)
