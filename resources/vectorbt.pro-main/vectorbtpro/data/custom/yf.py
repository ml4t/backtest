# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `YFData` class for fetching financial data from Yahoo Finance."""

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.remote import RemoteData
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.config import merge_dicts, Config, HybridConfig
from vectorbtpro.utils.parsing import get_func_kwargs

__all__ = [
    "YFData",
]

__pdoc__ = {}


class YFData(RemoteData):
    """Data class for fetching financial data from Yahoo Finance.

    See:
        * https://github.com/ranaroussi/yfinance for the `yfinance` library.
        * `YFData.fetch_symbol` for argument details.

    !!! info
        For default settings, see `custom.yf` in `vectorbtpro._settings.data`.

    Examples:
        ```pycon
        >>> from vectorbtpro import *

        >>> data = vbt.YFData.pull(
        ...     "BTC-USD",
        ...     start="2020-01-01",
        ...     end="2021-01-01",
        ...     timeframe="1 day"
        ... )
        ```
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.yf")

    _feature_config: tp.ClassVar[Config] = HybridConfig(
        {
            "Dividends": dict(
                resample_func=lambda self, obj, resampler: obj.vbt.resample_apply(
                    resampler,
                    generic_nb.sum_reduce_nb,
                )
            ),
            "Stock Splits": dict(
                resample_func=lambda self, obj, resampler: obj.vbt.resample_apply(
                    resampler,
                    generic_nb.nonzero_prod_reduce_nb,
                )
            ),
            "Capital Gains": dict(
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
    def fetch_symbol(
        cls,
        symbol: tp.Symbol,
        period: tp.Optional[str] = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        timeframe: tp.Optional[str] = None,
        tz: tp.TimezoneLike = None,
        ticker_kwargs: tp.KwargsLike = None,
        **history_kwargs,
    ) -> tp.SymbolData:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch a symbol from Yahoo Finance.

        Args:
            symbol (Symbol): Symbol identifier.
            period (Optional[str]): Period string.
            start (Optional[DatetimeLike]): Start datetime (e.g., "2024-01-01", "1 year ago").

                See `vectorbtpro.utils.datetime_.to_timestamp`.
            end (Optional[DatetimeLike]): End datetime (e.g., "2025-01-01", "now").

                See `vectorbtpro.utils.datetime_.to_timestamp`.
            timeframe (Optional[str]): Timeframe specification (e.g., "daily", "15 minutes").

                See `vectorbtpro.utils.datetime_.split_freq_str`.
            tz (TimezoneLike): Timezone specification (e.g., "UTC", "America/New_York").

                See `vectorbtpro.utils.datetime_.to_timezone`.
            ticker_kwargs (KwargsLike): Keyword arguments for `yfinance.ticker.Ticker`.
            **history_kwargs: Keyword arguments for `yfinance.base.TickerBase.history`.

        Returns:
            SymbolData: Fetched data and a metadata dictionary.

        !!! warning
            Data from Yahoo Finance may be unstable. Yahoo may modify data, introduce noise, or omit data points
            (e.g., volume in the example). It is primarily intended for demonstration purposes.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("yfinance")
        import yfinance as yf

        period = cls.resolve_custom_setting(period, "period")
        start = cls.resolve_custom_setting(start, "start")
        end = cls.resolve_custom_setting(end, "end")
        timeframe = cls.resolve_custom_setting(timeframe, "timeframe")
        tz = cls.resolve_custom_setting(tz, "tz")
        ticker_kwargs = cls.resolve_custom_setting(ticker_kwargs, "ticker_kwargs", merge=True)
        history_kwargs = cls.resolve_custom_setting(history_kwargs, "history_kwargs", merge=True)

        ticker = yf.Ticker(symbol, **ticker_kwargs)
        def_history_kwargs = get_func_kwargs(yf.Tickers.history)
        ticker_tz = ticker._get_ticker_tz(history_kwargs.get("timeout", def_history_kwargs["timeout"]))
        if tz is None:
            tz = ticker_tz
        if start is not None:
            start = dt.to_tzaware_datetime(start, naive_tz=tz, tz=ticker_tz)
        if end is not None:
            end = dt.to_tzaware_datetime(end, naive_tz=tz, tz=ticker_tz)
        freq = timeframe
        split = dt.split_freq_str(timeframe)
        if split is not None:
            multiplier, unit = split
            if unit == "D":
                unit = "d"
            elif unit == "W":
                unit = "wk"
            elif unit == "M":
                unit = "mo"
            timeframe = str(multiplier) + unit

        df = ticker.history(period=period, start=start, end=end, interval=timeframe, **history_kwargs)
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
            df = df.tz_localize(ticker_tz)

        if not df.empty:
            if start is not None:
                if df.index[0] < start:
                    df = df[df.index >= start]
            if end is not None:
                if df.index[-1] >= end:
                    df = df[df.index < end]
        return df, dict(tz=tz, freq=freq)

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        fetch_kwargs = self.select_fetch_kwargs(symbol)
        fetch_kwargs["start"] = self.select_last_index(symbol)
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)


YFData.override_feature_config_doc(__pdoc__)
