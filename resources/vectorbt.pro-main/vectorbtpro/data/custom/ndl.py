# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `NDLData` class for accessing data from Nasdaq Data Link."""

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.remote import RemoteData
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "NDLData",
]

__pdoc__ = {}

NDLDataT = tp.TypeVar("NDLDataT", bound="NDLData")


class NDLData(RemoteData):
    """Data class for fetching data from Nasdaq Data Link.

    This class provides methods to pull data from Nasdaq Data Link using its API.

    See:
        * https://github.com/Nasdaq/data-link-python for the official API documentation.
        * `NDLData.fetch_symbol` for argument details.

    !!! info
        For default settings, see `custom.ndl` in `vectorbtpro._settings.data`.

    Examples:
        Set up the API key globally (optional):

        ```pycon
        >>> from vectorbtpro import *

        >>> vbt.NDLData.set_custom_settings(
        ...     api_key="YOUR_KEY"
        ... )
        ```

        Pull a dataset:

        ```pycon
        >>> data = vbt.NDLData.pull(
        ...     "FRED/GDP",
        ...     start="2001-12-31",
        ...     end="2005-12-31"
        ... )
        ```

        Pull a datatable:

        ```pycon
        >>> data = vbt.NDLData.pull(
        ...     "MER/F1",
        ...     data_format="datatable",
        ...     compnumber="39102",
        ...     paginate=True
        ... )
        ```
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.ndl")

    @classmethod
    def fetch_symbol(
        cls,
        symbol: tp.Symbol,
        api_key: tp.Optional[str] = None,
        data_format: tp.Optional[str] = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        tz: tp.TimezoneLike = None,
        column_indices: tp.Optional[tp.MaybeIterable[int]] = None,
        **params,
    ) -> tp.SymbolData:
        """Fetch a symbol's data from Nasdaq Data Link.

        Args:
            symbol (Symbol): Symbol identifier.
            api_key (Optional[str]): API key.
            data_format (Optional[str]): Data format.

                Supported formats: "dataset" and "datatable".
            start (Optional[DatetimeLike]): Start datetime (e.g., "2024-01-01", "1 year ago").

                See `vectorbtpro.utils.datetime_.to_timestamp`.
            end (Optional[DatetimeLike]): End datetime (e.g., "2025-01-01", "now").

                See `vectorbtpro.utils.datetime_.to_timestamp`.
            tz (TimezoneLike): Timezone specification (e.g., "UTC", "America/New_York").

                See `vectorbtpro.utils.datetime_.to_timezone`.
            column_indices (Optional[MaybeIterable[int]]): Specific column(s) to retrieve.

                Column 0 (date) is always returned, with data columns starting at index 1.
            **params: Keyword arguments for Nasdaq Data Link as field/value parameters.

        Returns:
            SymbolData: Fetched data and a metadata dictionary.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("nasdaqdatalink")

        import nasdaqdatalink

        api_key = cls.resolve_custom_setting(api_key, "api_key")
        data_format = cls.resolve_custom_setting(data_format, "data_format")
        start = cls.resolve_custom_setting(start, "start")
        end = cls.resolve_custom_setting(end, "end")
        tz = cls.resolve_custom_setting(tz, "tz")
        column_indices = cls.resolve_custom_setting(column_indices, "column_indices")
        if column_indices is not None:
            if isinstance(column_indices, int):
                dataset = symbol + "." + str(column_indices)
            else:
                dataset = [symbol + "." + str(index) for index in column_indices]
        else:
            dataset = symbol
        params = cls.resolve_custom_setting(params, "params", merge=True)

        # Establish the timestamps
        if start is not None:
            start = dt.to_tzaware_datetime(start, naive_tz=tz, tz="utc")
            start_date = pd.Timestamp(start).isoformat()
            if "start_date" not in params:
                params["start_date"] = start_date
        else:
            start_date = None
        if end is not None:
            end = dt.to_tzaware_datetime(end, naive_tz=tz, tz="utc")
            end_date = pd.Timestamp(end).isoformat()
            if "end_date" not in params:
                params["end_date"] = end_date
        else:
            end_date = None

        # Collect and format the data
        if data_format.lower() == "dataset":
            df = nasdaqdatalink.get(
                dataset,
                api_key=api_key,
                **params,
            )
        else:
            df = nasdaqdatalink.get_table(
                dataset,
                api_key=api_key,
                **params,
            )
        new_columns = []
        for c in df.columns:
            new_c = c
            if isinstance(symbol, str):
                new_c = new_c.replace(symbol + " - ", "")
            if new_c == "Last":
                new_c = "Close"
            new_columns.append(new_c)
        df = df.rename(columns=dict(zip(df.columns, new_columns)))
        if df.index.name == "None":
            df.index.name = None

        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
            df = df.tz_localize("utc")
        if isinstance(df.index, pd.DatetimeIndex) and not df.empty:
            if start is not None:
                start = dt.to_timestamp(start, tz=df.index.tz)
                if df.index[0] < start:
                    df = df[df.index >= start]
            if end is not None:
                end = dt.to_timestamp(end, tz=df.index.tz)
                if df.index[-1] >= end:
                    df = df[df.index < end]
        return df, dict(tz=tz)

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        fetch_kwargs = self.select_fetch_kwargs(symbol)
        fetch_kwargs["start"] = self.select_last_index(symbol)
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)
