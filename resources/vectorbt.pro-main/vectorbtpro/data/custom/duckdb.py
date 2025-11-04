# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `DuckDBData` class for interacting with DuckDB databases."""

from pathlib import Path

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.data.base import key_dict
from vectorbtpro.data.custom.db import DBData
from vectorbtpro.data.custom.file import FileData
from vectorbtpro.utils import checks, datetime_ as dt
from vectorbtpro.utils.config import merge_dicts

if tp.TYPE_CHECKING:
    from duckdb import DuckDBPyConnection as DuckDBPyConnectionT, DuckDBPyRelation as DuckDBPyRelationT
else:
    DuckDBPyConnectionT = "duckdb.DuckDBPyConnection"
    DuckDBPyRelationT = "duckdb.DuckDBPyRelation"

__all__ = [
    "DuckDBData",
]

__pdoc__ = {}

DuckDBDataT = tp.TypeVar("DuckDBDataT", bound="DuckDBData")


class DuckDBData(DBData):
    """Data class for fetching data from a DuckDB database.

    This class provides methods to pull tables, execute queries, and read Parquet files.

    See:
        * `DuckDBData.pull` for argument details.
        * `DuckDBData.fetch_key` for argument details.

    !!! info
        For default settings, see `custom.duckdb` in `vectorbtpro._settings.data`.

    Examples:
        Set up the connection settings globally (optional):

        ```pycon
        >>> from vectorbtpro import *

        >>> vbt.DuckDBData.set_custom_settings(connection="database.duckdb")
        ```

        Pull tables:

        ```pycon
        >>> data = vbt.DuckDBData.pull(["TABLE1", "TABLE2"])
        ```

        Rename tables:

        ```pycon
        >>> data = vbt.DuckDBData.pull(
        ...     ["SYMBOL1", "SYMBOL2"],
        ...     table=vbt.key_dict({
        ...         "SYMBOL1": "TABLE1",
        ...         "SYMBOL2": "TABLE2"
        ...     })
        ... )
        ```

        Pull queries:

        ```pycon
        >>> data = vbt.DuckDBData.pull(
        ...     ["SYMBOL1", "SYMBOL2"],
        ...     query=vbt.key_dict({
        ...         "SYMBOL1": "SELECT * FROM TABLE1",
        ...         "SYMBOL2": "SELECT * FROM TABLE2"
        ...     })
        ... )
        ```

        Pull Parquet files:

        ```pycon
        >>> data = vbt.DuckDBData.pull(
        ...     ["SYMBOL1", "SYMBOL2"],
        ...     read_path=vbt.key_dict({
        ...         "SYMBOL1": "s1.parquet",
        ...         "SYMBOL2": "s2.parquet"
        ...     })
        ... )
        ```
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.duckdb")

    @classmethod
    def resolve_connection(
        cls,
        connection: tp.Union[None, str, tp.PathLike, DuckDBPyConnectionT] = None,
        read_only: bool = True,
        return_meta: bool = False,
        **connection_config,
    ) -> tp.Union[DuckDBPyConnectionT, dict]:
        """Resolve and return a DuckDB connection based on provided parameters.

        Args:
            connection (Union[None, str, PathLike, DuckDBPyConnection]): Database connection string or instance.

                If None, a default connection is used.
            read_only (bool): Flag indicating whether the connection should be opened in read-only mode.
            return_meta (bool): If True, return a dictionary with connection metadata.
            **connection_config: Keyword arguments for connection configuration.

        Returns:
            Union[DuckDBPyConnection, dict]: DuckDB connection or a metadata dictionary
                containing the connection and a flag indicating if it should be closed.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("duckdb")
        from duckdb import connect, default_connection, DuckDBPyConnection

        connection_meta = {}
        connection = cls.resolve_custom_setting(connection, "connection")
        if connection_config is None:
            connection_config = {}
        has_connection_config = len(connection_config) > 0
        connection_config["read_only"] = read_only
        connection_config = cls.resolve_custom_setting(connection_config, "connection_config", merge=True)
        read_only = connection_config.pop("read_only", read_only)
        should_close = False
        if connection is None:
            if len(connection_config) == 0:
                if isinstance(default_connection, DuckDBPyConnection):
                    connection = default_connection
                else:
                    connection = default_connection()
            else:
                database = connection_config.pop("database", None)
                if "config" in connection_config or len(connection_config) == 0:
                    connection = connect(database, read_only=read_only, **connection_config)
                else:
                    connection = connect(database, read_only=read_only, config=connection_config)
                should_close = True
        elif isinstance(connection, (str, Path)):
            if "config" in connection_config or len(connection_config) == 0:
                connection = connect(str(connection), read_only=read_only, **connection_config)
            else:
                connection = connect(str(connection), read_only=read_only, config=connection_config)
            should_close = True
        elif has_connection_config:
            raise ValueError("Cannot apply connection_config to already initialized connection")

        if return_meta:
            connection_meta["connection"] = connection
            connection_meta["should_close"] = should_close
            return connection_meta
        return connection

    @classmethod
    def list_catalogs(
        cls,
        pattern: tp.Optional[str] = None,
        use_regex: bool = False,
        sort: bool = True,
        incl_system: bool = False,
        connection: tp.Union[None, str, DuckDBPyConnectionT] = None,
        connection_config: tp.KwargsLike = None,
    ) -> tp.List[str]:
        """List available catalogs from the DuckDB database.

        Catalogs "system" and "temp" are omitted if `incl_system` is False.

        Args:
            pattern (Optional[str]): Pattern to filter catalog names.
            use_regex (bool): Flag indicating whether the pattern is a regular expression.
            sort (bool): Flag indicating whether to sort the resulting catalog names.
            incl_system (bool): Flag indicating whether to include system catalogs.
            connection (Union[None, str, DuckDBPyConnection]): Database connection string or instance.
            connection_config (KwargsLike): Configuration parameters for creating a database connection.

        Returns:
            List[str]: List of catalog names.
        """
        if connection_config is None:
            connection_config = {}
        connection_meta = cls.resolve_connection(connection, return_meta=True, **connection_config)
        connection = connection_meta["connection"]
        should_close = connection_meta["should_close"]
        schemata_df = connection.sql("SELECT * FROM information_schema.schemata").df()
        catalogs = []
        for catalog in schemata_df["catalog_name"].tolist():
            if pattern is not None:
                if not cls.key_match(catalog, pattern, use_regex=use_regex):
                    continue
            if not incl_system and catalog == "system":
                continue
            if not incl_system and catalog == "temp":
                continue
            catalogs.append(catalog)

        if should_close:
            connection.close()
        if sort:
            return sorted(dict.fromkeys(catalogs))
        return list(dict.fromkeys(catalogs))

    @classmethod
    def list_schemas(
        cls,
        catalog_pattern: tp.Optional[str] = None,
        schema_pattern: tp.Optional[str] = None,
        use_regex: bool = False,
        sort: bool = True,
        catalog: tp.Optional[str] = None,
        incl_system: bool = False,
        connection: tp.Union[None, str, DuckDBPyConnectionT] = None,
        connection_config: tp.KwargsLike = None,
    ) -> tp.List[str]:
        """List available schemas from the DuckDB database.

        If catalog is None, the method searches all catalogs and prefixes each schema with
        the catalog name if multiple catalogs are found. Schemas "information_schema" and
        "pg_catalog" are omitted if `incl_system` is False.

        Args:
            catalog_pattern (Optional[str]): Pattern to filter catalog names.
            schema_pattern (Optional[str]): Pattern to filter schema names.
            use_regex (bool): Flag indicating whether the pattern is a regular expression.
            sort (bool): Flag indicating whether to sort the resulting schema names.
            catalog (Optional[str]): Specific catalog to search for schemas.

                If provided, schemas are not prefixed.
            incl_system (bool): Flag indicating whether to include system schemas.
            connection (Union[None, str, DuckDBPyConnection]): Database connection string or instance.
            connection_config (KwargsLike): Configuration parameters for creating a database connection.

        Returns:
            List[str]: List of schema names.
        """
        if connection_config is None:
            connection_config = {}
        connection_meta = cls.resolve_connection(connection, return_meta=True, **connection_config)
        connection = connection_meta["connection"]
        should_close = connection_meta["should_close"]
        if catalog is None:
            catalogs = cls.list_catalogs(
                pattern=catalog_pattern,
                use_regex=use_regex,
                sort=sort,
                incl_system=incl_system,
                connection=connection,
                connection_config=connection_config,
            )
            if len(catalogs) == 1:
                prefix_catalog = False
            else:
                prefix_catalog = True
        else:
            catalogs = [catalog]
            prefix_catalog = False
        schemata_df = connection.sql("SELECT * FROM information_schema.schemata").df()
        schemas = []
        for catalog in catalogs:
            all_schemas = schemata_df[schemata_df["catalog_name"] == catalog]["schema_name"].tolist()
            for schema in all_schemas:
                if schema_pattern is not None:
                    if not cls.key_match(schema, schema_pattern, use_regex=use_regex):
                        continue
                if not incl_system and schema == "information_schema":
                    continue
                if not incl_system and schema == "pg_catalog":
                    continue
                if prefix_catalog:
                    schema = catalog + ":" + schema
                schemas.append(schema)

        if should_close:
            connection.close()
        if sort:
            return sorted(dict.fromkeys(schemas))
        return list(dict.fromkeys(schemas))

    @classmethod
    def get_current_schema(
        cls,
        connection: tp.Union[None, str, DuckDBPyConnectionT] = None,
        connection_config: tp.KwargsLike = None,
    ) -> str:
        """Return the current schema in use by the DuckDB connection.

        Args:
            connection (Union[None, str, DuckDBPyConnection]): Database connection string or instance.
            connection_config (KwargsLike): Configuration parameters for creating a database connection.

        Returns:
            str: Current schema name.
        """
        if connection_config is None:
            connection_config = {}
        connection_meta = cls.resolve_connection(connection, return_meta=True, **connection_config)
        connection = connection_meta["connection"]
        should_close = connection_meta["should_close"]
        current_schema = connection.sql("SELECT current_schema()").fetchall()[0][0]

        if should_close:
            connection.close()
        return current_schema

    @classmethod
    def list_tables(
        cls,
        *,
        catalog_pattern: tp.Optional[str] = None,
        schema_pattern: tp.Optional[str] = None,
        table_pattern: tp.Optional[str] = None,
        use_regex: bool = False,
        sort: bool = True,
        catalog: tp.Optional[str] = None,
        schema: tp.Optional[str] = None,
        incl_system: bool = False,
        incl_temporary: bool = False,
        incl_views: bool = True,
        connection: tp.Union[None, str, DuckDBPyConnectionT] = None,
        connection_config: tp.KwargsLike = None,
    ) -> tp.List[str]:
        """List tables and views from the database.

        This method queries the database's information schema to retrieve a list of tables
        and views. When `schema` is not provided, it searches across all available schemas
        and prefixes each table name with its corresponding catalog and schema names
        (unless only a single current schema is detected). If a specific `schema` is provided,
        the returned table names are not prefixed.

        Uses `vectorbtpro.data.custom.custom.CustomData.key_match` to perform pattern matching
        on schema and table names.

        Args:
            catalog_pattern (Optional[str]): Pattern to filter catalog names.
            schema_pattern (Optional[str]): Pattern to filter schema names.
            table_pattern (Optional[str]): Pattern to filter table names.
            use_regex (bool): Flag indicating whether the pattern is a regular expression.
            sort (bool): Indicates whether to sort the resulting list.
            catalog (Optional[str]): Filter results to the specified catalog.
            schema (Optional[str]): Filter results to the specified schema.

                Use `"current_schema"` to refer to the current schema.
            incl_system (bool): Flag indicating whether to include system tables and views.
            incl_temporary (bool): Include temporary tables.
            incl_views (bool): Include view objects.
            connection (Union[None, str, DuckDBPyConnection]): Database connection string or instance.
            connection_config (KwargsLike): Configuration parameters for creating a database connection.

        Returns:
            List[str]: List of table names, optionally prefixed with catalog and schema names.
        """
        if connection_config is None:
            connection_config = {}
        connection_meta = cls.resolve_connection(connection, return_meta=True, **connection_config)
        connection = connection_meta["connection"]
        should_close = connection_meta["should_close"]
        if catalog is None:
            catalogs = cls.list_catalogs(
                pattern=catalog_pattern,
                use_regex=use_regex,
                sort=sort,
                incl_system=incl_system,
                connection=connection,
                connection_config=connection_config,
            )
            if catalog_pattern is None and len(catalogs) == 1:
                prefix_catalog = False
            else:
                prefix_catalog = True
        else:
            catalogs = [catalog]
            prefix_catalog = False
        current_schema = cls.get_current_schema(
            connection=connection,
            connection_config=connection_config,
        )
        if schema is None:
            catalogs_schemas = []
            for catalog in catalogs:
                catalog_schemas = cls.list_schemas(
                    schema_pattern=schema_pattern,
                    use_regex=use_regex,
                    sort=sort,
                    catalog=catalog,
                    incl_system=incl_system,
                    connection=connection,
                    connection_config=connection_config,
                )
                for schema in catalog_schemas:
                    catalogs_schemas.append((catalog, schema))
            if len(catalogs_schemas) == 1 and catalogs_schemas[0][1] == current_schema:
                prefix_schema = False
            else:
                prefix_schema = True
        else:
            if schema == "current_schema":
                schema = current_schema
            catalogs_schemas = []
            for catalog in catalogs:
                catalogs_schemas.append((catalog, schema))
            prefix_schema = prefix_catalog
        tables_df = connection.sql("SELECT * FROM information_schema.tables").df()
        tables = []
        for catalog, schema in catalogs_schemas:
            all_tables = []
            all_tables.extend(
                tables_df[
                    (tables_df["table_catalog"] == catalog)
                    & (tables_df["table_schema"] == schema)
                    & (tables_df["table_type"] == "BASE TABLE")
                ]["table_name"].tolist()
            )
            if incl_temporary:
                all_tables.extend(
                    tables_df[
                        (tables_df["table_catalog"] == catalog)
                        & (tables_df["table_schema"] == schema)
                        & (tables_df["table_type"] == "LOCAL TEMPORARY")
                    ]["table_name"].tolist()
                )
            if incl_views:
                all_tables.extend(
                    tables_df[
                        (tables_df["table_catalog"] == catalog)
                        & (tables_df["table_schema"] == schema)
                        & (tables_df["table_type"] == "VIEW")
                    ]["table_name"].tolist()
                )
            for table in all_tables:
                if table_pattern is not None:
                    if not cls.key_match(table, table_pattern, use_regex=use_regex):
                        continue
                if not prefix_catalog and prefix_schema:
                    table = schema + ":" + table
                elif prefix_catalog or prefix_schema:
                    table = catalog + ":" + schema + ":" + table
                tables.append(table)

        if should_close:
            connection.close()
        if sort:
            return sorted(dict.fromkeys(tables))
        return list(dict.fromkeys(tables))

    @classmethod
    def resolve_keys_meta(
        cls,
        keys: tp.Union[None, dict, tp.MaybeKeys] = None,
        keys_are_features: tp.Optional[bool] = None,
        features: tp.Union[None, dict, tp.MaybeFeatures] = None,
        symbols: tp.Union[None, dict, tp.MaybeSymbols] = None,
        catalog: tp.Optional[str] = None,
        schema: tp.Optional[str] = None,
        list_tables_kwargs: tp.KwargsLike = None,
        read_path: tp.Optional[tp.PathLike] = None,
        read_format: tp.Optional[str] = None,
        connection: tp.Union[None, str, DuckDBPyConnectionT] = None,
        connection_config: tp.KwargsLike = None,
    ) -> tp.Kwargs:
        keys_meta = DBData.resolve_keys_meta(
            keys=keys,
            keys_are_features=keys_are_features,
            features=features,
            symbols=symbols,
        )
        if keys_meta["keys"] is None:
            if cls.has_key_dict(catalog):
                raise ValueError("Cannot populate keys if catalog is defined per key")
            if cls.has_key_dict(schema):
                raise ValueError("Cannot populate keys if schema is defined per key")
            if cls.has_key_dict(list_tables_kwargs):
                raise ValueError("Cannot populate keys if list_tables_kwargs is defined per key")
            if cls.has_key_dict(connection):
                raise ValueError("Cannot populate keys if connection is defined per key")
            if cls.has_key_dict(connection_config):
                raise ValueError("Cannot populate keys if connection_config is defined per key")
            if cls.has_key_dict(read_path):
                raise ValueError("Cannot populate keys if read_path is defined per key")
            if cls.has_key_dict(read_format):
                raise ValueError("Cannot populate keys if read_format is defined per key")
            if read_path is not None or read_format is not None:
                if read_path is None:
                    read_path = "."
                if read_format is not None:
                    read_format = read_format.lower()
                    checks.assert_in(read_format, ["csv", "parquet", "json"], arg_name="read_format")
                keys_meta["keys"] = FileData.list_paths(read_path, extension=read_format)
            else:
                if list_tables_kwargs is None:
                    list_tables_kwargs = {}
                keys_meta["keys"] = cls.list_tables(
                    catalog=catalog,
                    schema=schema,
                    connection=connection,
                    connection_config=connection_config,
                    **list_tables_kwargs,
                )
        return keys_meta

    @classmethod
    def pull(
        cls: tp.Type[DuckDBDataT],
        keys: tp.Union[tp.MaybeKeys] = None,
        *,
        keys_are_features: tp.Optional[bool] = None,
        features: tp.Union[tp.MaybeFeatures] = None,
        symbols: tp.Union[tp.MaybeSymbols] = None,
        catalog: tp.Optional[str] = None,
        schema: tp.Optional[str] = None,
        list_tables_kwargs: tp.KwargsLike = None,
        read_path: tp.Optional[tp.PathLike] = None,
        read_format: tp.Optional[str] = None,
        connection: tp.Union[None, str, DuckDBPyConnectionT] = None,
        connection_config: tp.KwargsLike = None,
        share_connection: tp.Optional[bool] = None,
        **kwargs,
    ) -> DuckDBDataT:
        """Override `vectorbtpro.data.base.Data.pull` to resolve and share the database
        connection among provided keys.

        Args:
            keys (MaybeKeys): Feature or symbol identifier(s).

                If not provided, available table names are used.
            keys_are_features (Optional[bool]): Flag indicating whether the keys represent features.
            features (MaybeFeatures): Feature identifier(s).
            symbols (MaybeSymbols): Symbol identifier(s).
            catalog (Optional[str]): Catalog name for database lookup.
            schema (Optional[str]): Schema name for database lookup.
            list_tables_kwargs (KwargsLike): Keyword arguments for listing database tables.

                See `DuckDBData.list_tables`.
            read_path (Optional[PathLike]): File or directory path for reading data.
            read_format (Optional[str]): Format to use when reading data.
            connection (Union[None, str, DuckDBPyConnection]): Database connection string or instance.
            connection_config: Keyword arguments for configuring the connection.
            share_connection (Optional[bool]): If True, uses a shared connection among keys.
            **kwargs: Keyword arguments for `vectorbtpro.data.custom.db.DBData.pull`.

        Returns:
            DuckDBData: Instance containing the pulled data with resolved keys and connection.
        """
        if share_connection is None:
            if not cls.has_key_dict(connection) and not cls.has_key_dict(connection_config):
                share_connection = True
            else:
                share_connection = False
        if share_connection:
            if connection_config is None:
                connection_config = {}
            connection_meta = cls.resolve_connection(connection, return_meta=True, **connection_config)
            connection = connection_meta["connection"]
            should_close = connection_meta["should_close"]
        else:
            should_close = False
        keys_meta = cls.resolve_keys_meta(
            keys=keys,
            keys_are_features=keys_are_features,
            features=features,
            symbols=symbols,
            catalog=catalog,
            schema=schema,
            list_tables_kwargs=list_tables_kwargs,
            read_path=read_path,
            read_format=read_format,
            connection=connection,
            connection_config=connection_config,
        )
        keys = keys_meta["keys"]
        if isinstance(read_path, key_dict):
            new_read_path = read_path.copy()
        else:
            new_read_path = key_dict()
        if isinstance(keys, dict):
            new_keys = {}
            for k, v in keys.items():
                if isinstance(k, Path):
                    new_k = FileData.path_to_key(k)
                    new_read_path[new_k] = k
                    k = new_k
                new_keys[k] = v
            keys = new_keys
        elif cls.has_multiple_keys(keys):
            new_keys = []
            for k in keys:
                if isinstance(k, Path):
                    new_k = FileData.path_to_key(k)
                    new_read_path[new_k] = k
                    k = new_k
                new_keys.append(k)
            keys = new_keys
        else:
            if isinstance(keys, Path):
                new_keys = FileData.path_to_key(keys)
                new_read_path[new_keys] = keys
                keys = new_keys
        if len(new_read_path) > 0:
            read_path = new_read_path
        keys_are_features = keys_meta["keys_are_features"]
        outputs = super(DBData, cls).pull(
            keys,
            keys_are_features=keys_are_features,
            catalog=catalog,
            schema=schema,
            read_path=read_path,
            read_format=read_format,
            connection=connection,
            connection_config=connection_config,
            **kwargs,
        )

        if should_close:
            connection.close()
        return outputs

    @classmethod
    def format_write_option(cls, option: tp.Any) -> str:
        """Format the given write option into a string representation.

        Args:
            option (Any): Write option to format.

        Returns:
            str: Formatted write option.
        """
        if isinstance(option, str):
            return f"'{option}'"
        if isinstance(option, (tuple, list)):
            return "(" + ", ".join(map(str, option)) + ")"
        if isinstance(option, dict):
            return "{" + ", ".join(map(lambda y: f"{y[0]}: {cls.format_write_option(y[1])}", option.items())) + "}"
        return f"{option}"

    @classmethod
    def format_write_options(cls, options: tp.Union[str, dict]) -> str:
        """Format the given write options into a single string.

        Args:
            options (Union[str, dict]): Write options to format.

        Returns:
            str: Comma-separated string of formatted write options.
        """
        if isinstance(options, str):
            return options
        new_options = []
        for k, v in options.items():
            new_options.append(f"{k.upper()} {cls.format_write_option(v)}")
        return ", ".join(new_options)

    @classmethod
    def format_read_option(cls, option: tp.Any) -> str:
        """Format the given read option into a string representation.

        Args:
            option (Any): Read option to format.

        Returns:
            str: Formatted read option.
        """
        if isinstance(option, str):
            return f"'{option}'"
        if isinstance(option, (tuple, list)):
            return "[" + ", ".join(map(cls.format_read_option, option)) + "]"
        if isinstance(option, dict):
            return "{" + ", ".join(map(lambda y: f"'{y[0]}': {cls.format_read_option(y[1])}", option.items())) + "}"
        return f"{option}"

    @classmethod
    def format_read_options(cls, options: tp.Union[str, dict]) -> str:
        """Format the given read options into a single string.

        Args:
            options (Union[str, dict]): Read options to format.

        Returns:
            str: Comma-separated string of formatted read options.
        """
        if isinstance(options, str):
            return options
        new_options = []
        for k, v in options.items():
            new_options.append(f"{k.lower()}={cls.format_read_option(v)}")
        return ", ".join(new_options)

    @classmethod
    def fetch_key(
        cls,
        key: tp.Key,
        table: tp.Optional[str] = None,
        schema: tp.Optional[str] = None,
        catalog: tp.Optional[str] = None,
        read_path: tp.Optional[tp.PathLike] = None,
        read_format: tp.Optional[str] = None,
        read_options: tp.Union[None, str, dict] = None,
        query: tp.Union[None, str, DuckDBPyRelationT] = None,
        connection: tp.Union[None, str, DuckDBPyConnectionT] = None,
        connection_config: tp.KwargsLike = None,
        start: tp.Optional[tp.Any] = None,
        end: tp.Optional[tp.Any] = None,
        align_dates: tp.Optional[bool] = None,
        parse_dates: tp.Union[None, bool, tp.Sequence[str]] = None,
        to_utc: tp.Union[None, bool, str, tp.Sequence[str]] = None,
        tz: tp.TimezoneLike = None,
        index_col: tp.Optional[tp.MaybeSequence[tp.IntStr]] = None,
        squeeze: tp.Optional[bool] = None,
        df_kwargs: tp.KwargsLike = None,
        **sql_kwargs,
    ) -> tp.KeyData:
        """Fetch a feature or symbol from a DuckDB database.

        Can use a table name (defaulting to the key) or a custom SQL query.

        Args:
            key (Key): Feature or symbol identifier.

                If both `table` and `query` are None, the key becomes the table name.
                The key can be provided in either `SCHEMA:TABLE` or `CATALOG:SCHEMA:TABLE` format,
                in which case the `schema` argument is ignored.
            table (Optional[str]): Table name.

                Cannot be used together with `read_path` or `query`.
            schema (Optional[str]): Schema name.

                Cannot be used together with `read_path` or `query`.
            catalog (Optional[str]): Catalog name.

                Cannot be used together with `read_path` or `query`.
            read_path (Optional[PathLike]): Path to a file to read.

                Cannot be used together with `table`, `schema`, `catalog`, or `query`.
            read_format (Optional[str]): File format.

                Allowed values are "csv", "parquet", and "json".
                Requires `read_path` to be set.
            read_options (Union[None, str, dict]): Options used to read the file.

                Requires both `read_path` and `read_format` to be set.
                Uses `DuckDBData.format_read_options` to transform options into a string.
            query (Union[None, str, DuckDBPyRelation]): Custom SQL query.

                Cannot be used together with `catalog`, `schema`, or `table`.
            connection (Union[None, str, DuckDBPyConnection]): Database connection string or instance.

                See `DuckDBData.resolve_connection`.
            connection_config (KwargsLike): Configuration parameters for creating a database connection.

                See `DuckDBData.resolve_connection`.
            start (Optional[Any]): Start value for filtering.

                Interpreted as a datetime when the index is of datetime type and `align_dates` is True.
                Cannot be used together with `query`; include the condition in the query.
            end (Optional[Any]): End value for filtering.

                Interpreted as a datetime when the index is of datetime type and `align_dates` is True.
                Cannot be used together with `query`; include the condition in the query.
            align_dates (Optional[bool]): Whether to align `start` and `end` to the index timezone.

                Retrieves one row (using `LIMIT 1`) and uses `SQLData.prepare_dt` to obtain the index.
            parse_dates (Union[None, bool, Sequence[str]]): Specifies whether to parse dates.

                See `DuckDBData.prepare_dt`.
            to_utc (Union[None, bool, str, Sequence[str]]): Specifies whether to localize or convert
                datetime fields to UTC.

                See `DuckDBData.prepare_dt`.
            tz (TimezoneLike): Timezone specification (e.g., "UTC", "America/New_York").

                See `vectorbtpro.utils.datetime_.to_timezone`.
            index_col (Optional[MaybeSequence[IntStr]]): Column position(s) or name(s) to use as the index.
            squeeze (Optional[bool]): Flag indicating whether to convert a single-column DataFrame to a Series.
            df_kwargs (KwargsLike): Keyword arguments for `relation.df` to convert a relation to a DataFrame.
            **sql_kwargs: Keyword arguments for `connection.execute` to run the SQL query.

        Returns:
            KeyData: Fetched data and a metadata dictionary.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("duckdb")
        from duckdb import DuckDBPyRelation

        if connection_config is None:
            connection_config = {}
        connection_meta = cls.resolve_connection(connection, return_meta=True, **connection_config)
        connection = connection_meta["connection"]
        should_close = connection_meta["should_close"]
        if catalog is not None and query is not None:
            raise ValueError("Cannot use catalog and query together")
        if schema is not None and query is not None:
            raise ValueError("Cannot use schema and query together")
        if table is not None and query is not None:
            raise ValueError("Cannot use table and query together")
        if read_path is not None and query is not None:
            raise ValueError("Cannot use read_path and query together")
        if read_path is not None and (catalog is not None or schema is not None or table is not None):
            raise ValueError("Cannot use read_path and catalog/schema/table together")
        if table is None and read_path is None and read_format is None and query is None:
            if ":" in key:
                key_parts = key.split(":")
                if len(key_parts) == 2:
                    schema, table = key_parts
                else:
                    catalog, schema, table = key_parts
            else:
                table = key
        if read_format is not None:
            read_format = read_format.lower()
            checks.assert_in(read_format, ["csv", "parquet", "json"], arg_name="read_format")
            if read_path is None:
                read_path = (Path(".") / key).with_suffix("." + read_format)
        else:
            if read_path is not None:
                if isinstance(read_path, str):
                    read_path = Path(read_path)
                if read_path.suffix[1:] in ["csv", "parquet", "json"]:
                    read_format = read_path.suffix[1:]
        if read_path is not None:
            if isinstance(read_path, Path):
                read_path = str(read_path)
            read_path = cls.format_read_option(read_path)
        if read_options is not None:
            if read_format is None:
                raise ValueError("Must provide read_format for read_options")
            read_options = cls.format_read_options(read_options)

        catalog = cls.resolve_custom_setting(catalog, "catalog")
        schema = cls.resolve_custom_setting(schema, "schema")
        start = cls.resolve_custom_setting(start, "start")
        end = cls.resolve_custom_setting(end, "end")
        align_dates = cls.resolve_custom_setting(align_dates, "align_dates")
        parse_dates = cls.resolve_custom_setting(parse_dates, "parse_dates")
        to_utc = cls.resolve_custom_setting(to_utc, "to_utc")
        tz = cls.resolve_custom_setting(tz, "tz")
        index_col = cls.resolve_custom_setting(index_col, "index_col")
        squeeze = cls.resolve_custom_setting(squeeze, "squeeze")
        df_kwargs = cls.resolve_custom_setting(df_kwargs, "df_kwargs", merge=True)
        sql_kwargs = cls.resolve_custom_setting(sql_kwargs, "sql_kwargs", merge=True)

        if query is None:
            if read_path is not None:
                if read_options is not None:
                    query = f"SELECT * FROM read_{read_format}({read_path}, {read_options})"
                elif read_format is not None:
                    query = f"SELECT * FROM read_{read_format}({read_path})"
                else:
                    query = f"SELECT * FROM {read_path}"
            else:
                if catalog is not None:
                    if schema is None:
                        schema = cls.get_current_schema(
                            connection=connection,
                            connection_config=connection_config,
                        )
                    query = f'SELECT * FROM "{catalog}"."{schema}"."{table}"'
                elif schema is not None:
                    query = f'SELECT * FROM "{schema}"."{table}"'
                else:
                    query = f'SELECT * FROM "{table}"'
            if start is not None or end is not None:
                if index_col is None:
                    raise ValueError("Must provide index column for filtering by start and end")
                if not checks.is_int(index_col) and not isinstance(index_col, str):
                    raise ValueError("Index column must be integer or string for filtering by start and end")
                if checks.is_int(index_col) or align_dates:
                    metadata_df = connection.sql("DESCRIBE " + query + " LIMIT 1").df()
                else:
                    metadata_df = None
                if checks.is_int(index_col):
                    index_name = metadata_df["column_name"].tolist()[0]
                else:
                    index_name = index_col
                if parse_dates:
                    index_column_type = metadata_df[metadata_df["column_name"] == index_name]["column_type"].item()
                    if index_column_type in (
                        "TIMESTAMP_NS",
                        "TIMESTAMP_MS",
                        "TIMESTAMP_S",
                        "TIMESTAMP",
                        "DATETIME",
                    ):
                        if start is not None:
                            if (
                                to_utc is True
                                or (isinstance(to_utc, str) and to_utc.lower() == "index")
                                or (checks.is_sequence(to_utc) and index_name in to_utc)
                            ):
                                start = dt.to_tzaware_datetime(start, naive_tz=tz, tz="utc")
                                start = dt.to_naive_datetime(start)
                            else:
                                start = dt.to_naive_datetime(start, tz=tz)
                        if end is not None:
                            if (
                                to_utc is True
                                or (isinstance(to_utc, str) and to_utc.lower() == "index")
                                or (checks.is_sequence(to_utc) and index_name in to_utc)
                            ):
                                end = dt.to_tzaware_datetime(end, naive_tz=tz, tz="utc")
                                end = dt.to_naive_datetime(end)
                            else:
                                end = dt.to_naive_datetime(end, tz=tz)
                    elif index_column_type in ("TIMESTAMPTZ", "TIMESTAMP WITH TIME ZONE"):
                        if start is not None:
                            if (
                                to_utc is True
                                or (isinstance(to_utc, str) and to_utc.lower() == "index")
                                or (checks.is_sequence(to_utc) and index_name in to_utc)
                            ):
                                start = dt.to_tzaware_datetime(start, naive_tz=tz, tz="utc")
                            else:
                                start = dt.to_tzaware_datetime(start, naive_tz=tz)
                        if end is not None:
                            if (
                                to_utc is True
                                or (isinstance(to_utc, str) and to_utc.lower() == "index")
                                or (checks.is_sequence(to_utc) and index_name in to_utc)
                            ):
                                end = dt.to_tzaware_datetime(end, naive_tz=tz, tz="utc")
                            else:
                                end = dt.to_tzaware_datetime(end, naive_tz=tz)
                if start is not None and end is not None:
                    query += f' WHERE "{index_name}" >= $start AND "{index_name}" < $end'
                elif start is not None:
                    query += f' WHERE "{index_name}" >= $start'
                elif end is not None:
                    query += f' WHERE "{index_name}" < $end'
                params = sql_kwargs.get("params", None)
                if params is None:
                    params = {}
                else:
                    params = dict(params)
                if not isinstance(params, dict):
                    raise ValueError("Parameters must be a dictionary for filtering by start and end")
                if start is not None:
                    if "start" in params:
                        raise ValueError("Start is already in params")
                    params["start"] = start
                if end is not None:
                    if "end" in params:
                        raise ValueError("End is already in params")
                    params["end"] = end
                sql_kwargs["params"] = params
        else:
            if start is not None:
                raise ValueError("Start cannot be applied to custom queries")
            if end is not None:
                raise ValueError("End cannot be applied to custom queries")

        if not isinstance(query, DuckDBPyRelation):
            relation = connection.sql(query, **sql_kwargs)
        else:
            relation = query
        obj = relation.df(**df_kwargs)

        if isinstance(obj, pd.DataFrame) and checks.is_default_index(obj.index):
            if index_col is not None:
                if checks.is_int(index_col):
                    keys = obj.columns[index_col]
                elif isinstance(index_col, str):
                    keys = index_col
                else:
                    keys = []
                    for col in index_col:
                        if checks.is_int(col):
                            keys.append(obj.columns[col])
                        else:
                            keys.append(col)
                obj = obj.set_index(keys)
                if not isinstance(obj.index, pd.MultiIndex):
                    if obj.index.name == "index":
                        obj.index.name = None
        obj = cls.prepare_dt(obj, to_utc=to_utc, parse_dates=parse_dates)
        if not isinstance(obj.index, pd.MultiIndex):
            if obj.index.name == "index":
                obj.index.name = None
        if isinstance(obj.index, pd.DatetimeIndex) and tz is None:
            tz = obj.index.tz
        if isinstance(obj, pd.DataFrame) and squeeze:
            obj = obj.squeeze("columns")
        if isinstance(obj, pd.Series) and obj.name == "0":
            obj.name = None

        if should_close:
            connection.close()
        return obj, dict(tz=tz)

    @classmethod
    def fetch_feature(cls, feature: tp.Feature, **kwargs) -> tp.FeatureData:
        """Fetch the data table for a feature using the underlying `DuckDBData.fetch_key` method.

        Args:
            feature (Feature): Feature identifier.
            **kwargs: Keyword arguments for `DuckDBData.fetch_key`.

        Returns:
            FeatureData: Fetched data and a metadata dictionary.
        """
        return cls.fetch_key(feature, **kwargs)

    @classmethod
    def fetch_symbol(cls, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        """Fetch the data table for a symbol using the underlying `DuckDBData.fetch_key` method.

        Args:
            symbol (Symbol): Symbol identifier.
            **kwargs: Keyword arguments for `DuckDBData.fetch_key`.

        Returns:
            SymbolData: Fetched data and a metadata dictionary.
        """
        return cls.fetch_key(symbol, **kwargs)

    def update_key(self, key: tp.Key, from_last_index: tp.Optional[bool] = None, **kwargs) -> tp.KeyData:
        """Update the data table for a specified feature or symbol.

        This method selects fetch parameters via `DuckDBData.select_fetch_kwargs` and determines
        whether to start from the last index based on the `from_last_index` flag or the presence
        of a "query" in the keyword arguments. It then updates and fetches the data table by
        invoking either `fetch_feature` or `fetch_symbol` depending on the object's orientation.

        Args:
            key (Key): Feature or symbol identifier.
            from_last_index (Optional[bool]): Flag indicating whether to update data starting from the last index.

                If not provided, it is inferred from the presence of a "query" in the merged keyword arguments.
            **kwargs: Keyword arguments for `DuckDBData.fetch_feature` or `DuckDBData.fetch_symbol`.

        Returns:
            KeyData: Updated data and a metadata dictionary.
        """
        fetch_kwargs = self.select_fetch_kwargs(key)
        pre_kwargs = merge_dicts(fetch_kwargs, kwargs)
        if from_last_index is None:
            if pre_kwargs.get("query", None) is not None:
                from_last_index = False
            else:
                from_last_index = True
        if from_last_index:
            fetch_kwargs["start"] = self.select_last_index(key)
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        if self.feature_oriented:
            return self.fetch_feature(key, **kwargs)
        return self.fetch_symbol(key, **kwargs)

    def update_feature(self, feature: tp.Feature, **kwargs) -> tp.FeatureData:
        return self.update_key(feature, **kwargs)

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        return self.update_key(symbol, **kwargs)
