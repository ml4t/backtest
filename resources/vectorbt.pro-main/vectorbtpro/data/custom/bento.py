# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `BentoData` class for fetching data from Databento."""

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.remote import RemoteData
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.parsing import get_func_arg_names

if tp.TYPE_CHECKING:
    from databento import Historical as HistoricalT
else:
    HistoricalT = "databento.Historical"

__all__ = [
    "BentoData",
]


class BentoData(RemoteData):
    """Data class for fetching data from Databento.

    See:
        * https://github.com/databento/databento-python for the Databento Python client.
        * `BentoData.fetch_symbol` for argument details.

    !!! info
        For default settings, see `custom.bento` in `vectorbtpro._settings.data`.

    Examples:
        Set up the API key globally (optional):

        ```pycon
        >>> from vectorbtpro import *

        >>> vbt.BentoData.set_custom_settings(
        ...     client_config=dict(
        ...         key="YOUR_KEY"
        ...     )
        ... )
        ```

        Pull data:

        ```pycon
        >>> data = vbt.BentoData.pull(
        ...     "AAPL",
        ...     dataset="XNAS.ITCH"
        ... )
        ```

        ```pycon
        >>> data = vbt.BentoData.pull(
        ...     "AAPL",
        ...     dataset="XNAS.ITCH",
        ...     timeframe="hourly",
        ...     start="one week ago"
        ... )
        ```

        ```pycon
        >>> data = vbt.BentoData.pull(
        ...     "ES.FUT",
        ...     dataset="GLBX.MDP3",
        ...     stype_in="parent",
        ...     schema="mbo",
        ...     start="2022-06-10T14:30",
        ...     end="2022-06-11",
        ...     limit=1000
        ... )
        ```
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.bento")

    @classmethod
    def resolve_client(cls, client: tp.Optional[HistoricalT] = None, **client_config) -> HistoricalT:
        """Resolve the client.

        Args:
            client (Optional[databento.historical.client.Historical]): Client instance.

                If provided, must be of type `databento.historical.client.Historical`.
            **client_config: Configuration parameters for creating a new client.

        Returns:
            databento.historical.client.Historical: Resolved client instance.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("databento")
        from databento import Historical

        client = cls.resolve_custom_setting(client, "client")
        if client_config is None:
            client_config = {}
        has_client_config = len(client_config) > 0
        client_config = cls.resolve_custom_setting(client_config, "client_config", merge=True)
        if client is None:
            client = Historical(**client_config)
        elif has_client_config:
            raise ValueError("Cannot apply client_config to already initialized client")
        return client

    @classmethod
    def get_cost(cls, symbols: tp.MaybeSymbols, **kwargs) -> float:
        """Get the total cost for fetching symbol data.

        Args:
            symbols (MaybeSymbols): Symbol identifier(s).
            **kwargs: Keyword arguments for `BentoData.fetch_symbol`.

        Returns:
            float: Aggregated cost.
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        costs = []
        for symbol in symbols:
            client, params = cls.fetch_symbol(symbol, **kwargs, return_params=True)
            cost_arg_names = get_func_arg_names(client.metadata.get_cost)
            for k in list(params.keys()):
                if k not in cost_arg_names:
                    del params[k]
            costs.append(client.metadata.get_cost(**params, mode="historical"))
        return sum(costs)

    @classmethod
    def fetch_symbol(
        cls,
        symbol: tp.Symbol,
        client: tp.Optional[HistoricalT] = None,
        client_config: tp.KwargsLike = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        resolve_dates: tp.Optional[bool] = None,
        timeframe: tp.Optional[str] = None,
        tz: tp.TimezoneLike = None,
        dataset: tp.Optional[str] = None,
        schema: tp.Optional[str] = None,
        return_params: bool = False,
        df_kwargs: tp.KwargsLike = None,
        **params,
    ) -> tp.Union[float, tp.SymbolData]:
        """Fetch a symbol from Databento.

        Args:
            symbol (Symbol): Symbol identifier.

                Can be provided in the `DATASET:SYMBOL` format if `dataset` is not specified.
            client (Optional[databento.historical.client.Historical]): Client instance.

                See `BentoData.resolve_client`.
            client_config (KwargsLike): Configuration parameters for creating a new client.

                See `BentoData.resolve_client`.
            start (Optional[DatetimeLike]): Start datetime (e.g., "2024-01-01", "1 year ago").

                See `vectorbtpro.utils.datetime_.to_timestamp`.
            end (Optional[DatetimeLike]): End datetime (e.g., "2025-01-01", "now").

                See `vectorbtpro.utils.datetime_.to_timestamp`.
            resolve_dates (Optional[bool]): Whether to resolve `start` and `end` to UTC timestamps.
            timeframe (Optional[str]): Timeframe specification (e.g., "daily", "15 minutes").

                If both `timeframe` and `schema` are provided, an error is raised.
                See `vectorbtpro.utils.datetime_.split_freq_str`.
            tz (TimezoneLike): Timezone specification (e.g., "UTC", "America/New_York").

                See `vectorbtpro.utils.datetime_.to_timezone`.
            dataset (Optional[str]): Dataset identifier.

                See `databento.historical.client.Historical.get_range` for details.
            schema (Optional[str]): Schema identifier.

                See `databento.historical.client.Historical.get_range` for details.
            return_params (bool): If True, return the client and resolved parameters instead of fetched data.

                Used by `BentoData.get_cost`.
            df_kwargs (KwargsLike): Keyword arguments for `databento.common.dbnstore.DBNStore.to_df`.
            **params: Keyword arguments for `databento.historical.client.Historical.get_range`.

        Returns:
            Union[float, SymbolData]: If `return_params` is True, returns the client and final parameters.
                Otherwise, returns the fetched data and a metadata dictionary.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("databento")

        if client_config is None:
            client_config = {}
        client = cls.resolve_client(client=client, **client_config)

        start = cls.resolve_custom_setting(start, "start")
        end = cls.resolve_custom_setting(end, "end")
        resolve_dates = cls.resolve_custom_setting(resolve_dates, "resolve_dates")
        timeframe = cls.resolve_custom_setting(timeframe, "timeframe")
        tz = cls.resolve_custom_setting(tz, "tz")
        dataset = cls.resolve_custom_setting(dataset, "dataset")
        schema = cls.resolve_custom_setting(schema, "schema")
        params = cls.resolve_custom_setting(params, "params", merge=True)
        df_kwargs = cls.resolve_custom_setting(df_kwargs, "df_kwargs", merge=True)

        if dataset is None:
            if ":" in symbol:
                dataset, symbol = symbol.split(":")
        if timeframe is None and schema is None:
            schema = "ohlcv-1d"
            freq = "1d"
        elif timeframe is not None:
            freq = timeframe
            split = dt.split_freq_str(timeframe)
            if split is not None:
                multiplier, unit = split
                timeframe = str(multiplier) + unit
                if schema is None or schema.lower() == "ohlcv":
                    schema = f"ohlcv-{timeframe}"
                else:
                    raise ValueError("Timeframe cannot be used together with schema")
        else:
            if schema.startswith("ohlcv-"):
                freq = schema[len("ohlcv-") :]
            else:
                freq = None
        if resolve_dates:
            dataset_range = client.metadata.get_dataset_range(dataset)
            if "start_date" in dataset_range:
                start_date = dt.to_tzaware_timestamp(dataset_range["start_date"], naive_tz="utc", tz="utc")
            else:
                start_date = dt.to_tzaware_timestamp(dataset_range["start"], naive_tz="utc", tz="utc")
            if "end_date" in dataset_range:
                end_date = dt.to_tzaware_timestamp(dataset_range["end_date"], naive_tz="utc", tz="utc")
            else:
                end_date = dt.to_tzaware_timestamp(dataset_range["end"], naive_tz="utc", tz="utc")
            if start is not None:
                start = dt.to_tzaware_timestamp(start, naive_tz=tz, tz="utc")
                if start < start_date:
                    start = start_date
            else:
                start = start_date
            if end is not None:
                end = dt.to_tzaware_timestamp(end, naive_tz=tz, tz="utc")
                if end > end_date:
                    end = end_date
            else:
                end = end_date
            if start.floor("d") == start:
                start = start.date().isoformat()
            else:
                start = start.isoformat()
            if end.floor("d") == end:
                end = end.date().isoformat()
            else:
                end = end.isoformat()

        params = merge_dicts(
            dict(
                dataset=dataset,
                start=start,
                end=end,
                symbols=symbol,
                schema=schema,
            ),
            params,
        )
        if return_params:
            return client, params

        df = client.timeseries.get_range(**params).to_df(**df_kwargs)
        return df, dict(tz=tz, freq=freq)

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        fetch_kwargs = self.select_fetch_kwargs(symbol)
        fetch_kwargs["start"] = self.select_last_index(symbol)
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)
