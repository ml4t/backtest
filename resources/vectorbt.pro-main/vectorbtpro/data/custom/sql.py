# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `SQLData` class for fetching data from SQL databases using SQLAlchemy."""

from typing import Iterator

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.db import DBData
from vectorbtpro.utils import checks, datetime_ as dt
from vectorbtpro.utils.config import merge_dicts

if tp.TYPE_CHECKING:
    from sqlalchemy import Engine as EngineT, Selectable as SelectableT, Table as TableT
else:
    EngineT = "sqlalchemy.Engine"
    SelectableT = "sqlalchemy.Selectable"
    TableT = "sqlalchemy.Table"

__all__ = [
    "SQLData",
]

__pdoc__ = {}

SQLDataT = tp.TypeVar("SQLDataT", bound="SQLData")


class SQLData(DBData):
    """Data class for fetching data from a database using SQLAlchemy.

    See:
        * https://www.sqlalchemy.org/ for the SQLAlchemy API.
        * https://pandas.pydata.org/docs/reference/api/pandas.read_sql_query.html for the Pandas read method.
        * `SQLData.pull` and `SQLData.fetch_key` for argument details.

    !!! info
        For default settings, see `custom.sql` in `vectorbtpro._settings.data`.

        Global settings can be provided per engine name using the `engines` dictionary.

    Examples:
        Set up the engine settings globally (optional):

        ```pycon
        >>> from vectorbtpro import *

        >>> vbt.SQLData.set_engine_settings(
        ...     engine_name="postgresql",
        ...     populate_=True,
        ...     engine="postgresql+psycopg2://...",
        ...     engine_config=dict(),
        ...     schema="public"
        ... )
        ```

        Pull tables:

        ```pycon
        >>> data = vbt.SQLData.pull(
        ...     ["TABLE1", "TABLE2"],
        ...     engine="postgresql",
        ...     start="2020-01-01",
        ...     end="2021-01-01"
        ... )
        ```

        Pull queries:

        ```pycon
        >>> data = vbt.SQLData.pull(
        ...     ["SYMBOL1", "SYMBOL2"],
        ...     query=vbt.key_dict({
        ...         "SYMBOL1": "SELECT * FROM TABLE1",
        ...         "SYMBOL2": "SELECT * FROM TABLE2"
        ...     }),
        ...     engine="postgresql"
        ... )
        ```
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.sql")

    @classmethod
    def get_engine_settings(cls, *args, engine_name: tp.Optional[str] = None, **kwargs) -> dict:
        """Return custom engine settings with `sub_path` set to the engine name.

        Args:
            *args: Positional arguments for `SQLData.get_custom_settings`.
            engine_name (Optional[str]): Name of the engine for retrieving custom settings.
            **kwargs: Keyword arguments for `SQLData.get_custom_settings`.

        Returns:
            dict: Resolved engine settings.
        """
        if engine_name is not None:
            sub_path = "engines." + engine_name
        else:
            sub_path = None
        return cls.get_custom_settings(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def has_engine_settings(cls, *args, engine_name: tp.Optional[str] = None, **kwargs) -> bool:
        """Return whether custom engine settings exist with `sub_path` set to the engine name.

        Args:
            *args: Positional arguments for `SQLData.has_custom_settings`.
            engine_name (Optional[str]): Name of the engine for retrieving custom settings.
            **kwargs: Keyword arguments for `SQLData.has_custom_settings`.

        Returns:
            bool: True if custom engine settings exist, False otherwise.
        """
        if engine_name is not None:
            sub_path = "engines." + engine_name
        else:
            sub_path = None
        return cls.has_custom_settings(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def get_engine_setting(cls, *args, engine_name: tp.Optional[str] = None, **kwargs) -> tp.Any:
        """Return the custom engine setting with `sub_path` set to the engine name.

        Args:
            *args: Positional arguments for `SQLData.get_custom_setting`.
            engine_name (Optional[str]): Name of the engine for retrieving custom settings.
            **kwargs: Keyword arguments for `SQLData.get_custom_setting`.

        Returns:
            Any: Retrieved engine setting.
        """
        if engine_name is not None:
            sub_path = "engines." + engine_name
        else:
            sub_path = None
        return cls.get_custom_setting(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def has_engine_setting(cls, *args, engine_name: tp.Optional[str] = None, **kwargs) -> bool:
        """Return whether a custom engine setting exists with `sub_path` set to the engine name.

        Args:
            *args: Positional arguments for `SQLData.has_custom_setting`.
            engine_name (Optional[str]): Name of the engine for retrieving custom settings.
            **kwargs: Keyword arguments for `SQLData.has_custom_setting`.

        Returns:
            bool: True if the custom engine setting exists, False otherwise.
        """
        if engine_name is not None:
            sub_path = "engines." + engine_name
        else:
            sub_path = None
        return cls.has_custom_setting(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def resolve_engine_setting(cls, *args, engine_name: tp.Optional[str] = None, **kwargs) -> tp.Any:
        """Resolve the custom engine setting with `sub_path` set to the engine name.

        Args:
            *args: Positional arguments for `SQLData.resolve_custom_setting`.
            engine_name (Optional[str]): Name of the engine for retrieving custom settings.
            **kwargs: Keyword arguments for `SQLData.resolve_custom_setting`.

        Returns:
            Any: Resolved engine setting.
        """
        if engine_name is not None:
            sub_path = "engines." + engine_name
        else:
            sub_path = None
        return cls.resolve_custom_setting(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def set_engine_settings(cls, *args, engine_name: tp.Optional[str] = None, **kwargs) -> None:
        """Set custom engine settings with `sub_path` set to the engine name.

        Args:
            *args: Positional arguments for `SQLData.set_custom_settings`.
            engine_name (Optional[str]): Name of the engine for retrieving custom settings.
            **kwargs: Keyword arguments for `SQLData.set_custom_settings`.

        Returns:
            None
        """
        if engine_name is not None:
            sub_path = "engines." + engine_name
        else:
            sub_path = None
        cls.set_custom_settings(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def resolve_engine(
        cls,
        engine: tp.Union[None, str, EngineT] = None,
        engine_name: tp.Optional[str] = None,
        return_meta: bool = False,
        **engine_config,
    ) -> tp.Union[EngineT, dict]:
        """Resolve and return the database engine.

        Args:
            engine (Union[None, str, Engine]): Engine instance, URL, or key for engine settings.

                * Instance of `sqlalchemy.engine.base.Engine`.
                * URL (string) to create an engine using `sqlalchemy.engine.create.create_engine`.
                * Sub-config name under `custom.sql.engines` in `vectorbtpro._settings.data`
                    to retrieve the engine.
            engine_name (Optional[str]): Name of the engine for retrieving custom settings.
            return_meta (bool): If True, return a metadata dictionary containing the engine,
                engine name, and disposal flag.
            **engine_config: Keyword arguments for engine creation when `engine` is a URL.

        Returns:
            Union[Engine, dict]: Resolved engine, or a metadata dictionary if `return_meta` is True.

        !!! note
            Engine URLs can be provided as engine names, but not vice versa.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("sqlalchemy")
        from sqlalchemy import create_engine

        if engine is None and engine_name is None:
            engine_name = cls.resolve_engine_setting(engine_name, "engine_name")
        if engine_name is not None:
            engine = cls.resolve_engine_setting(engine, "engine", engine_name=engine_name)
            if engine is None:
                raise ValueError("Must provide engine or URL (via engine argument)")
        else:
            engine = cls.resolve_engine_setting(engine, "engine")
            if engine is None:
                raise ValueError("Must provide engine or URL (via engine argument)")
            if isinstance(engine, str):
                engine_name = engine
            else:
                engine_name = None
            if engine_name is not None:
                if cls.has_engine_setting("engine", engine_name=engine_name, sub_path_only=True):
                    engine = cls.get_engine_setting("engine", engine_name=engine_name, sub_path_only=True)
        has_engine_config = len(engine_config) > 0
        engine_config = cls.resolve_engine_setting(engine_config, "engine_config", merge=True, engine_name=engine_name)
        if isinstance(engine, str):
            if engine.startswith("duckdb:"):
                assert_can_import("duckdb_engine")
            engine = create_engine(engine, **engine_config)
            should_dispose = True
        else:
            if has_engine_config:
                raise ValueError("Cannot apply engine_config to initialized created engine")
            should_dispose = False
        if return_meta:
            return dict(
                engine=engine,
                engine_name=engine_name,
                should_dispose=should_dispose,
            )
        return engine

    @classmethod
    def list_schemas(
        cls,
        pattern: tp.Optional[str] = None,
        use_regex: bool = False,
        sort: bool = True,
        engine: tp.Union[None, str, EngineT] = None,
        engine_name: tp.Optional[str] = None,
        engine_config: tp.KwargsLike = None,
        dispose_engine: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.List[str]:
        """List all database schemas.

        Args:
            pattern (Optional[str]): Pattern to filter schema names.

                Schema names are matched using `SQLData.key_match`.
            use_regex (bool): Flag indicating whether the pattern is a regular expression.
            sort (bool): Flag indicating whether to sort the returned schema names.
            engine (Union[None, str, Engine]): Engine instance, URL, or key for engine settings.
            engine_name (Optional[str]): Name of the engine.
            engine_config (KwargsLike): Additional configuration for the engine.
            dispose_engine (Optional[bool]): Flag indicating whether to dispose the engine after use.

                If None, disposal is based on engine metadata.
            **kwargs: Keyword arguments for `inspector.get_schema_names`.

        Returns:
            List[str]: List of available schema names, excluding `information_schema`.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("sqlalchemy")
        from sqlalchemy import inspect

        if engine_config is None:
            engine_config = {}
        engine_meta = cls.resolve_engine(
            engine=engine,
            engine_name=engine_name,
            return_meta=True,
            **engine_config,
        )
        engine = engine_meta["engine"]
        should_dispose = engine_meta["should_dispose"]
        if dispose_engine is None:
            dispose_engine = should_dispose
        inspector = inspect(engine)
        all_schemas = inspector.get_schema_names(**kwargs)
        schemas = []
        for schema in all_schemas:
            if pattern is not None:
                if not cls.key_match(schema, pattern, use_regex=use_regex):
                    continue
            if schema == "information_schema":
                continue
            schemas.append(schema)

        if dispose_engine:
            engine.dispose()
        if sort:
            return sorted(dict.fromkeys(schemas))
        return list(dict.fromkeys(schemas))

    @classmethod
    def list_tables(
        cls,
        *,
        schema_pattern: tp.Optional[str] = None,
        table_pattern: tp.Optional[str] = None,
        use_regex: bool = False,
        sort: bool = True,
        schema: tp.Optional[str] = None,
        incl_views: bool = True,
        engine: tp.Union[None, str, EngineT] = None,
        engine_name: tp.Optional[str] = None,
        engine_config: tp.KwargsLike = None,
        dispose_engine: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.List[str]:
        """List all tables and views from the database.

        This method retrieves a list of table and view names using SQLAlchemy's inspector. If `schema`
        is None, all database schemas are searched and each table is prefixed with its respective
        schema name unless only a single default schema ("main") exists. If `schema` is False, the
        schema is disregarded. If a specific `schema` is provided, tables within that schema are
        returned without a prefix.

        Each schema and table is filtered using `vectorbtpro.data.custom.custom.CustomData.key_match`
        against the provided patterns.

        Args:
            schema_pattern (Optional[str]): Pattern to filter schema names.
            table_pattern (Optional[str]): Pattern to filter table names.
            use_regex (bool): Flag indicating whether the pattern is a regular expression.
            sort (bool): Whether to return the list of table names in sorted order.
            schema (Optional[str]): Specific schema for the search.

                If None, all schemas are considered.
            incl_views (bool): Whether to include view names along with table names.
            engine (Union[None, str, Engine]): Engine instance, URL, or key for engine settings.
            engine_name (Optional[str]): Name of the engine.
            engine_config (KwargsLike): Additional configuration for the engine.
            dispose_engine (Optional[bool]): Flag indicating whether to dispose the engine after use.
            **kwargs: Keyword arguments for SQLAlchemy's inspector methods.

        Returns:
            List[str]: List of table and view names.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("sqlalchemy")
        from sqlalchemy import inspect

        if engine_config is None:
            engine_config = {}
        engine_meta = cls.resolve_engine(
            engine=engine,
            engine_name=engine_name,
            return_meta=True,
            **engine_config,
        )
        engine = engine_meta["engine"]
        engine_name = engine_meta["engine_name"]
        should_dispose = engine_meta["should_dispose"]
        if dispose_engine is None:
            dispose_engine = should_dispose
        schema = cls.resolve_engine_setting(schema, "schema", engine_name=engine_name)
        if schema is None:
            schemas = cls.list_schemas(
                pattern=schema_pattern,
                use_regex=use_regex,
                sort=sort,
                engine=engine,
                engine_name=engine_name,
                **kwargs,
            )
            if len(schemas) == 0:
                schemas = [None]
                prefix_schema = False
            elif len(schemas) == 1 and schemas[0] == "main":
                prefix_schema = False
            else:
                prefix_schema = True
        elif schema is False:
            schemas = [None]
            prefix_schema = False
        else:
            schemas = [schema]
            prefix_schema = False
        inspector = inspect(engine)
        tables = []
        for schema in schemas:
            all_tables = inspector.get_table_names(schema, **kwargs)
            if incl_views:
                try:
                    all_tables += inspector.get_view_names(schema, **kwargs)
                except NotImplementedError:
                    pass
                try:
                    all_tables += inspector.get_materialized_view_names(schema, **kwargs)
                except NotImplementedError:
                    pass
            for table in all_tables:
                if table_pattern is not None:
                    if not cls.key_match(table, table_pattern, use_regex=use_regex):
                        continue
                if prefix_schema and schema is not None:
                    table = str(schema) + ":" + table
                tables.append(table)

        if dispose_engine:
            engine.dispose()
        if sort:
            return sorted(dict.fromkeys(tables))
        return list(dict.fromkeys(tables))

    @classmethod
    def has_schema(
        cls,
        schema: str,
        engine: tp.Union[None, str, EngineT] = None,
        engine_name: tp.Optional[str] = None,
        engine_config: tp.KwargsLike = None,
    ) -> bool:
        """Check whether the database contains the specified schema.

        Args:
            schema (str): Name of the schema.
            engine (Union[None, str, Engine]): Engine instance, URL, or key for engine settings.
            engine_name (Optional[str]): Name of the engine.
            engine_config (KwargsLike): Additional configuration for the engine.

        Returns:
            bool: True if the schema exists, False otherwise.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("sqlalchemy")
        from sqlalchemy import inspect

        if engine_config is None:
            engine_config = {}
        engine = cls.resolve_engine(
            engine=engine,
            engine_name=engine_name,
            **engine_config,
        )
        return inspect(engine).has_schema(schema)

    @classmethod
    def create_schema(
        cls,
        schema: str,
        engine: tp.Union[None, str, EngineT] = None,
        engine_name: tp.Optional[str] = None,
        engine_config: tp.KwargsLike = None,
    ) -> None:
        """Create the specified schema in the database if it does not already exist.

        Args:
            schema (str): Name of the schema to create.
            engine (Union[None, str, Engine]): Engine instance, URL, or key for engine settings.
            engine_name (Optional[str]): Name of the engine.
            engine_config (KwargsLike): Additional configuration for the engine.

        Returns:
            None
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("sqlalchemy")
        from sqlalchemy.schema import CreateSchema

        if engine_config is None:
            engine_config = {}
        engine = cls.resolve_engine(
            engine=engine,
            engine_name=engine_name,
            **engine_config,
        )
        if not cls.has_schema(schema, engine=engine, engine_name=engine_name):
            with engine.connect() as conn:
                conn.execute(CreateSchema(schema))
                conn.commit()

    @classmethod
    def has_table(
        cls,
        table: str,
        schema: tp.Optional[str] = None,
        engine: tp.Union[None, str, EngineT] = None,
        engine_name: tp.Optional[str] = None,
        engine_config: tp.KwargsLike = None,
    ) -> bool:
        """Check if the specified table exists in the database.

        Args:
            table (str): Name of the table.
            schema (Optional[str]): Schema in which to search for the table.
            engine (Union[None, str, Engine]): Engine instance, URL, or key for engine settings.
            engine_name (Optional[str]): Name of the engine.
            engine_config (KwargsLike): Additional configuration for the engine.

        Returns:
            bool: True if the table exists, False otherwise.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("sqlalchemy")
        from sqlalchemy import inspect

        if engine_config is None:
            engine_config = {}
        engine = cls.resolve_engine(
            engine=engine,
            engine_name=engine_name,
            **engine_config,
        )
        return inspect(engine).has_table(table, schema=schema)

    @classmethod
    def get_table_relation(
        cls,
        table: str,
        schema: tp.Optional[str] = None,
        engine: tp.Union[None, str, EngineT] = None,
        engine_name: tp.Optional[str] = None,
        engine_config: tp.KwargsLike = None,
    ) -> TableT:
        """Get the SQLAlchemy table relation for the specified table.

        Args:
            table (str): Name of the table.
            schema (Optional[str]): Schema where the table is located.
            engine (Union[None, str, Engine]): Engine instance, URL, or key for engine settings.
            engine_name (Optional[str]): Name of the engine.
            engine_config (KwargsLike): Additional configuration for the engine.

        Returns:
            Table: SQLAlchemy table object representing the table relation.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("sqlalchemy")
        from sqlalchemy import MetaData

        if engine_config is None:
            engine_config = {}
        engine = cls.resolve_engine(
            engine=engine,
            engine_name=engine_name,
            **engine_config,
        )
        schema = cls.resolve_engine_setting(schema, "schema", engine_name=engine_name)
        metadata_obj = MetaData()
        metadata_obj.reflect(bind=engine, schema=schema, only=[table], views=True)
        if schema is not None and schema + "." + table in metadata_obj.tables:
            return metadata_obj.tables[schema + "." + table]
        return metadata_obj.tables[table]

    @classmethod
    def get_last_row_number(
        cls,
        table: str,
        schema: tp.Optional[str] = None,
        row_number_column: tp.Optional[str] = None,
        engine: tp.Union[None, str, EngineT] = None,
        engine_name: tp.Optional[str] = None,
        engine_config: tp.KwargsLike = None,
    ) -> TableT:
        """Get the last row number from the specified table.

        Args:
            table (str): Name of the table.
            schema (Optional[str]): Schema where the table is located.
            row_number_column (Optional[str]): Name of the column containing row numbers.
            engine (Union[None, str, Engine]): Engine instance, URL, or key for engine settings.
            engine_name (Optional[str]): Name of the engine.
            engine_config (KwargsLike): Additional configuration for the engine.

        Returns:
            Table: Last row number retrieved from the table.
        """
        if engine_config is None:
            engine_config = {}
        engine_meta = cls.resolve_engine(
            engine=engine,
            engine_name=engine_name,
            return_meta=True,
            **engine_config,
        )
        engine = engine_meta["engine"]
        engine_name = engine_meta["engine_name"]
        row_number_column = cls.resolve_engine_setting(
            row_number_column,
            "row_number_column",
            engine_name=engine_name,
        )
        table_relation = cls.get_table_relation(table, schema=schema, engine=engine, engine_name=engine_name)
        table_column_names = []
        for column in table_relation.columns:
            table_column_names.append(column.name)
        if row_number_column not in table_column_names:
            raise ValueError(f"Row number column '{row_number_column}' not found")
        query = (
            table_relation.select()
            .with_only_columns(table_relation.columns.get(row_number_column))
            .order_by(table_relation.columns.get(row_number_column).desc())
            .limit(1)
        )
        with engine.connect() as conn:
            results = conn.execute(query)
            last_row_number = results.first()[0]
            conn.commit()
        return last_row_number

    @classmethod
    def resolve_keys_meta(
        cls,
        keys: tp.Union[None, dict, tp.MaybeKeys] = None,
        keys_are_features: tp.Optional[bool] = None,
        features: tp.Union[None, dict, tp.MaybeFeatures] = None,
        symbols: tp.Union[None, dict, tp.MaybeSymbols] = None,
        schema: tp.Optional[str] = None,
        list_tables_kwargs: tp.KwargsLike = None,
        engine: tp.Union[None, str, EngineT] = None,
        engine_name: tp.Optional[str] = None,
        engine_config: tp.KwargsLike = None,
    ) -> tp.Kwargs:
        keys_meta = DBData.resolve_keys_meta(
            keys=keys,
            keys_are_features=keys_are_features,
            features=features,
            symbols=symbols,
        )
        if keys_meta["keys"] is None:
            if cls.has_key_dict(schema):
                raise ValueError("Cannot populate keys if schema is defined per key")
            if cls.has_key_dict(list_tables_kwargs):
                raise ValueError("Cannot populate keys if list_tables_kwargs is defined per key")
            if cls.has_key_dict(engine):
                raise ValueError("Cannot populate keys if engine is defined per key")
            if cls.has_key_dict(engine_name):
                raise ValueError("Cannot populate keys if engine_name is defined per key")
            if cls.has_key_dict(engine_config):
                raise ValueError("Cannot populate keys if engine_config is defined per key")
            if list_tables_kwargs is None:
                list_tables_kwargs = {}
            keys_meta["keys"] = cls.list_tables(
                schema=schema,
                engine=engine,
                engine_name=engine_name,
                engine_config=engine_config,
                **list_tables_kwargs,
            )
        return keys_meta

    @classmethod
    def pull(
        cls: tp.Type[SQLDataT],
        keys: tp.Union[tp.MaybeKeys] = None,
        *,
        keys_are_features: tp.Optional[bool] = None,
        features: tp.Union[tp.MaybeFeatures] = None,
        symbols: tp.Union[tp.MaybeSymbols] = None,
        schema: tp.Optional[str] = None,
        list_tables_kwargs: tp.KwargsLike = None,
        engine: tp.Union[None, str, EngineT] = None,
        engine_name: tp.Optional[str] = None,
        engine_config: tp.KwargsLike = None,
        dispose_engine: tp.Optional[bool] = None,
        share_engine: tp.Optional[bool] = None,
        **kwargs,
    ) -> SQLDataT:
        """Override `vectorbtpro.data.base.Data.pull` to resolve the database engine and table keys
        prior to retrieving data from the SQL database.

        Args:
            keys (Union[MaybeKeys]): Feature or symbol identifier(s) for the database tables.
            keys_are_features (Optional[bool]): Flag indicating whether the keys represent features.
            features (Union[MaybeFeatures]): Feature identifier(s) associated with the database tables.
            symbols (Union[MaybeSymbols]): Symbol identifier(s) associated with the database tables.
            schema (Optional[str]): Database schema name for table identification.
            list_tables_kwargs (KwargsLike): Keyword arguments for listing database tables.

                See `SQLData.list_tables`.
            engine (Union[None, str, Engine]): Engine instance, URL, or key for engine settings.
            engine_name (Optional[str]): Name of the engine.
            engine_config (KwargsLike): Additional configuration for the engine.
            dispose_engine (Optional[bool]): Flag indicating whether to dispose the engine after use.
            share_engine (Optional[bool]): Flag indicating whether to share the engine among keys.
            **kwargs: Keyword arguments for `vectorbtpro.data.custom.db.DBData.pull`.

        Returns:
            SQLData: Data retrieved from the SQL database.
        """
        if share_engine is None:
            if (
                not cls.has_key_dict(engine)
                and not cls.has_key_dict(engine_name)
                and not cls.has_key_dict(engine_config)
            ):
                share_engine = True
            else:
                share_engine = False
        if share_engine:
            if engine_config is None:
                engine_config = {}
            engine_meta = cls.resolve_engine(
                engine=engine,
                engine_name=engine_name,
                return_meta=True,
                **engine_config,
            )
            engine = engine_meta["engine"]
            engine_name = engine_meta["engine_name"]
            should_dispose = engine_meta["should_dispose"]
            engine_config = {}
            if dispose_engine is None:
                dispose_engine = should_dispose
        else:
            engine_name = None
        keys_meta = cls.resolve_keys_meta(
            keys=keys,
            keys_are_features=keys_are_features,
            features=features,
            symbols=symbols,
            schema=schema,
            list_tables_kwargs=list_tables_kwargs,
            engine=engine,
            engine_name=engine_name,
            engine_config=engine_config,
        )
        keys = keys_meta["keys"]
        keys_are_features = keys_meta["keys_are_features"]
        outputs = super(DBData, cls).pull(
            keys,
            keys_are_features=keys_are_features,
            schema=schema,
            engine=engine,
            engine_name=engine_name,
            engine_config=engine_config,
            dispose_engine=False if share_engine else dispose_engine,
            **kwargs,
        )
        if share_engine and dispose_engine:
            engine.dispose()
        return outputs

    @classmethod
    def fetch_key(
        cls,
        key: tp.Key,
        table: tp.Union[None, str, TableT] = None,
        schema: tp.Optional[str] = None,
        query: tp.Union[None, str, SelectableT] = None,
        engine: tp.Union[None, str, EngineT] = None,
        engine_name: tp.Optional[str] = None,
        engine_config: tp.KwargsLike = None,
        dispose_engine: tp.Optional[bool] = None,
        start: tp.Optional[tp.Any] = None,
        end: tp.Optional[tp.Any] = None,
        align_dates: tp.Optional[bool] = None,
        parse_dates: tp.Union[None, bool, tp.List[tp.IntStr], tp.Dict[tp.IntStr, tp.Any]] = None,
        to_utc: tp.Union[None, bool, str, tp.Sequence[str]] = None,
        tz: tp.TimezoneLike = None,
        start_row: tp.Optional[int] = None,
        end_row: tp.Optional[int] = None,
        keep_row_number: tp.Optional[bool] = None,
        row_number_column: tp.Optional[str] = None,
        index_col: tp.Union[None, bool, tp.MaybeList[tp.IntStr]] = None,
        columns: tp.Optional[tp.MaybeList[tp.IntStr]] = None,
        dtype: tp.Union[None, tp.DTypeLike, tp.Dict[tp.IntStr, tp.DTypeLike]] = None,
        chunksize: tp.Optional[int] = None,
        chunk_func: tp.Optional[tp.Callable] = None,
        squeeze: tp.Optional[bool] = None,
        **read_sql_kwargs,
    ) -> tp.KeyData:
        """Fetch a feature or symbol from a SQL database.

        Fetch data from a SQL database using either a table name or a custom SQL query.

        Args:
            key (Key): Feature or symbol identifier.

                If both `table` and `query` are None, the key is used as the table name.
                If the key contains a colon (`:`), it must follow the `SCHEMA:TABLE` format,
                and the `schema` argument is ignored.
            table (Optional[Union[str, Table]]): Table name or table object.

                Must not be provided together with `query`.
            schema (Optional[str]): Database schema.

                Must not be used with `query`.
            query (Optional[Union[str, Selectable]]): Custom SQL query.

                Must not be provided together with `table` or `schema`.
            engine (Union[None, str, Engine]): Engine instance, URL, or key for engine settings.

                See `SQLData.resolve_engine`.
            engine_name (Optional[str]): Name of the engine.

                See `SQLData.resolve_engine`.
            engine_config (KwargsLike): Additional configuration for the engine.

                See `SQLData.resolve_engine`.
            dispose_engine (Optional[bool]): Flag indicating whether to dispose the engine after use.

                See `SQLData.resolve_engine`.
            start (Optional[Any]): Start value for filtering.

                If the index is datetime and `align_dates` is True, it is parsed with
                `vectorbtpro.utils.datetime_.to_timestamp`.

                For a multi-index, provide a tuple. Must not be used with `query`.
            end (Optional[Any]): End value for filtering.

                If the index is datetime and `align_dates` is True, it is parsed with
                `vectorbtpro.utils.datetime_.to_timestamp`.

                For a multi-index, provide a tuple. Must not be used with `query`.
            align_dates (Optional[bool]): Whether to align `start` and `end` to the index timezone.

                Retrieves one row (using `LIMIT 1`) and uses `SQLData.prepare_dt` to obtain the index.
            parse_dates (Optional[Union[bool, List[IntStr], Dict[IntStr, Any]]]):
                Configuration for parsing date columns.

                If `query` is not used, it maps to column names; otherwise, integer values are disallowed.

                Enabled parsing also attempts to process datetime columns that Pandas fails to parse.
            to_utc (Optional[Union[bool, str, Sequence[str]]]): Parameter for UTC conversion.

                See `SQLData.prepare_dt`.
            tz (TimezoneLike): Timezone specification (e.g., "UTC", "America/New_York").

                See `vectorbtpro.utils.datetime_.to_timezone`.
            start_row (Optional[int]): Index of the starting row (inclusive).

                The table must contain the column specified by `row_number_column`.

                Must not be used with `query`.
            end_row (Optional[int]): Index of the ending row (exclusive).

                The table must contain the column specified by `row_number_column`.

                Must not be used with `query`.
            keep_row_number (Optional[bool]): Determines whether to include the row number column
                (specified by `row_number_column`) in the output.
            row_number_column (Optional[str]): Name of the column containing row numbers.
            index_col (Optional[Union[None, bool, MaybeList[IntStr]]]): Column(s) to use as the index.

                If `query` is not used, integers map to column positions; otherwise,
                only column names are allowed.
            columns (Optional[MaybeList[IntStr]]): Column(s) to select from the table.

                Must not be used with `query`.
            dtype (Optional[Union[None, DTypeLike, Dict[IntStr, DTypeLike]]]): Data type for each column.

                If `query` is not used, integers map to column positions;
                otherwise, only column names are allowed.
            chunksize (Optional[int]): Number of rows per chunk for processing.

                See `pd.read_sql_query` for details on this argument.
            chunk_func (Optional[Callable]): Function to process and concatenate chunks when `chunksize` is set.
            squeeze (Optional[bool]): Flag indicating whether to convert a single-column DataFrame to a Series.
            **read_sql_kwargs: Keyword arguments for `pd.read_sql_query`.

                See https://pandas.pydata.org/docs/reference/api/pandas.read_sql_query.html for arguments.

        Returns:
            KeyData: Fetched data and a metadata dictionary.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("sqlalchemy")
        from sqlalchemy import Selectable, Select, FromClause, and_, text

        if engine_config is None:
            engine_config = {}
        engine_meta = cls.resolve_engine(
            engine=engine,
            engine_name=engine_name,
            return_meta=True,
            **engine_config,
        )
        engine = engine_meta["engine"]
        engine_name = engine_meta["engine_name"]
        should_dispose = engine_meta["should_dispose"]
        if dispose_engine is None:
            dispose_engine = should_dispose
        if table is not None and query is not None:
            raise ValueError("Must provide either table name or query, not both")
        if schema is not None and query is not None:
            raise ValueError("Schema cannot be applied to custom queries")
        if table is None and query is None:
            if ":" in key:
                schema, table = key.split(":")
            else:
                table = key

        start = cls.resolve_engine_setting(start, "start", engine_name=engine_name)
        end = cls.resolve_engine_setting(end, "end", engine_name=engine_name)
        align_dates = cls.resolve_engine_setting(align_dates, "align_dates", engine_name=engine_name)
        parse_dates = cls.resolve_engine_setting(parse_dates, "parse_dates", engine_name=engine_name)
        to_utc = cls.resolve_engine_setting(to_utc, "to_utc", engine_name=engine_name)
        tz = cls.resolve_engine_setting(tz, "tz", engine_name=engine_name)
        start_row = cls.resolve_engine_setting(start_row, "start_row", engine_name=engine_name)
        end_row = cls.resolve_engine_setting(end_row, "end_row", engine_name=engine_name)
        keep_row_number = cls.resolve_engine_setting(keep_row_number, "keep_row_number", engine_name=engine_name)
        row_number_column = cls.resolve_engine_setting(row_number_column, "row_number_column", engine_name=engine_name)
        index_col = cls.resolve_engine_setting(index_col, "index_col", engine_name=engine_name)
        columns = cls.resolve_engine_setting(columns, "columns", engine_name=engine_name)
        dtype = cls.resolve_engine_setting(dtype, "dtype", engine_name=engine_name)
        chunksize = cls.resolve_engine_setting(chunksize, "chunksize", engine_name=engine_name)
        chunk_func = cls.resolve_engine_setting(chunk_func, "chunk_func", engine_name=engine_name)
        squeeze = cls.resolve_engine_setting(squeeze, "squeeze", engine_name=engine_name)
        read_sql_kwargs = cls.resolve_engine_setting(
            read_sql_kwargs, "read_sql_kwargs", merge=True, engine_name=engine_name
        )

        if query is None or isinstance(query, (Selectable, FromClause)):
            if query is None:
                if isinstance(table, str):
                    table = cls.get_table_relation(table, schema=schema, engine=engine, engine_name=engine_name)
            else:
                table = query

            table_column_names = []
            for column in table.columns:
                table_column_names.append(column.name)

            def _resolve_columns(c):
                if checks.is_int(c):
                    c = table_column_names[int(c)]
                elif not isinstance(c, str):
                    new_c = []
                    for _c in c:
                        if checks.is_int(_c):
                            new_c.append(table_column_names[int(_c)])
                        else:
                            if _c not in table_column_names:
                                for __c in table_column_names:
                                    if _c.lower() == __c.lower():
                                        _c = __c
                                        break
                            new_c.append(_c)
                    c = new_c
                else:
                    if c not in table_column_names:
                        for _c in table_column_names:
                            if c.lower() == _c.lower():
                                return _c
                return c

            if index_col is False:
                index_col = None
            if index_col is not None:
                index_col = _resolve_columns(index_col)
                if isinstance(index_col, str):
                    index_col = [index_col]
            if columns is not None:
                columns = _resolve_columns(columns)
                if isinstance(columns, str):
                    columns = [columns]
            if parse_dates is not None:
                if not isinstance(parse_dates, bool):
                    if isinstance(parse_dates, dict):
                        parse_dates = dict(zip(_resolve_columns(parse_dates.keys()), parse_dates.values()))
                    else:
                        parse_dates = _resolve_columns(parse_dates)
                    if isinstance(parse_dates, str):
                        parse_dates = [parse_dates]
            if dtype is not None:
                if isinstance(dtype, dict):
                    dtype = dict(zip(_resolve_columns(dtype.keys()), dtype.values()))

            if not isinstance(table, Select):
                query = table.select()
            else:
                query = table
            if index_col is not None:
                for col in index_col:
                    query = query.order_by(col)
            if index_col is not None and columns is not None:
                pre_columns = []
                for col in index_col:
                    if col not in columns:
                        pre_columns.append(col)
                columns = pre_columns + columns
            if keep_row_number and columns is not None:
                if row_number_column in table_column_names and row_number_column not in columns:
                    columns = [row_number_column] + columns
            elif not keep_row_number and columns is None:
                if row_number_column in table_column_names:
                    columns = [col for col in table_column_names if col != row_number_column]
            if columns is not None:
                query = query.with_only_columns(*[table.columns.get(c) for c in columns])

            def _to_native_type(x):
                if checks.is_np_scalar(x):
                    return x.item()
                return x

            if start_row is not None or end_row is not None:
                if start is not None or end is not None:
                    raise ValueError("Can either filter by row numbers or by index, not both")
                _row_number_column = table.columns.get(row_number_column)
                if _row_number_column is None:
                    raise ValueError(f"Row number column '{row_number_column}' not found")
                and_list = []
                if start_row is not None:
                    and_list.append(_row_number_column >= _to_native_type(start_row))
                if end_row is not None:
                    and_list.append(_row_number_column < _to_native_type(end_row))
                query = query.where(and_(*and_list))
            if start is not None or end is not None:
                if index_col is None:
                    raise ValueError("Must provide index column for filtering by start and end")
                if align_dates:
                    with engine.connect() as conn:
                        first_obj = pd.read_sql_query(
                            query.limit(1),
                            conn,
                            index_col=index_col,
                            parse_dates=None if isinstance(parse_dates, bool) else parse_dates,  # bool not accepted
                            dtype=dtype,
                            chunksize=None,
                            **read_sql_kwargs,
                        )
                    first_obj = cls.prepare_dt(
                        first_obj,
                        parse_dates=list(parse_dates) if isinstance(parse_dates, dict) else parse_dates,
                        to_utc=False,
                    )
                    if isinstance(first_obj.index, pd.DatetimeIndex):
                        if tz is None:
                            tz = first_obj.index.tz
                        if first_obj.index.tz is not None:
                            if start is not None:
                                start = dt.to_tzaware_datetime(start, naive_tz=tz, tz=first_obj.index.tz)
                            if end is not None:
                                end = dt.to_tzaware_datetime(end, naive_tz=tz, tz=first_obj.index.tz)
                        else:
                            if start is not None:
                                if (
                                    to_utc is True
                                    or (isinstance(to_utc, str) and to_utc.lower() == "index")
                                    or (checks.is_sequence(to_utc) and first_obj.index.name in to_utc)
                                ):
                                    start = dt.to_tzaware_datetime(start, naive_tz=tz, tz="utc")
                                    start = dt.to_naive_datetime(start)
                                else:
                                    start = dt.to_naive_datetime(start, tz=tz)
                            if end is not None:
                                if (
                                    to_utc is True
                                    or (isinstance(to_utc, str) and to_utc.lower() == "index")
                                    or (checks.is_sequence(to_utc) and first_obj.index.name in to_utc)
                                ):
                                    end = dt.to_tzaware_datetime(end, naive_tz=tz, tz="utc")
                                    end = dt.to_naive_datetime(end)
                                else:
                                    end = dt.to_naive_datetime(end, tz=tz)

                and_list = []
                if start is not None:
                    if len(index_col) > 1:
                        if not isinstance(start, tuple):
                            raise TypeError("Start must be a tuple if the index is a multi-index")
                        if len(start) != len(index_col):
                            raise ValueError("Start tuple must match the number of levels in the multi-index")
                        for i in range(len(index_col)):
                            index_column = table.columns.get(index_col[i])
                            and_list.append(index_column >= _to_native_type(start[i]))
                    else:
                        index_column = table.columns.get(index_col[0])
                        and_list.append(index_column >= _to_native_type(start))
                if end is not None:
                    if len(index_col) > 1:
                        if not isinstance(end, tuple):
                            raise TypeError("End must be a tuple if the index is a multi-index")
                        if len(end) != len(index_col):
                            raise ValueError("End tuple must match the number of levels in the multi-index")
                        for i in range(len(index_col)):
                            index_column = table.columns.get(index_col[i])
                            and_list.append(index_column < _to_native_type(end[i]))
                    else:
                        index_column = table.columns.get(index_col[0])
                        and_list.append(index_column < _to_native_type(end))
                query = query.where(and_(*and_list))
        else:

            def _check_columns(c, arg_name):
                if checks.is_int(c):
                    raise ValueError(f"Must provide column as a string for '{arg_name}'")
                elif not isinstance(c, str):
                    for _c in c:
                        if checks.is_int(_c):
                            raise ValueError(f"Must provide each column as a string for '{arg_name}'")

            if start is not None:
                raise ValueError("Start cannot be applied to custom queries")
            if end is not None:
                raise ValueError("End cannot be applied to custom queries")
            if start_row is not None:
                raise ValueError("Start row cannot be applied to custom queries")
            if end_row is not None:
                raise ValueError("End row cannot be applied to custom queries")
            if index_col is False:
                index_col = None
            if index_col is not None:
                _check_columns(index_col, "index_col")
                if isinstance(index_col, str):
                    index_col = [index_col]
            if columns is not None:
                raise ValueError("Columns cannot be applied to custom queries")
            if parse_dates is not None:
                if not isinstance(parse_dates, bool):
                    if isinstance(parse_dates, dict):
                        _check_columns(parse_dates.keys(), "parse_dates")
                    else:
                        _check_columns(parse_dates, "parse_dates")
                    if isinstance(parse_dates, str):
                        parse_dates = [parse_dates]
            if dtype is not None:
                _check_columns(dtype.keys(), "dtype")

        if isinstance(query, str):
            query = text(query)
        with engine.connect() as conn:
            obj = pd.read_sql_query(
                query,
                conn,
                index_col=index_col,
                parse_dates=None if isinstance(parse_dates, bool) else parse_dates,  # bool not accepted
                dtype=dtype,
                chunksize=chunksize,
                **read_sql_kwargs,
            )
        if isinstance(obj, Iterator):
            if chunk_func is None:
                obj = pd.concat(list(obj), axis=0)
            else:
                obj = chunk_func(obj)
        obj = cls.prepare_dt(
            obj,
            parse_dates=list(parse_dates) if isinstance(parse_dates, dict) else parse_dates,
            to_utc=to_utc,
        )
        if not isinstance(obj.index, pd.MultiIndex):
            if obj.index.name == "index":
                obj.index.name = None
        if isinstance(obj.index, pd.DatetimeIndex) and tz is None:
            tz = obj.index.tz
        if isinstance(obj, pd.DataFrame) and squeeze:
            obj = obj.squeeze("columns")
        if isinstance(obj, pd.Series) and obj.name == "0":
            obj.name = None

        if dispose_engine:
            engine.dispose()
        if keep_row_number:
            return obj, dict(tz=tz, row_number_column=row_number_column)
        return obj, dict(tz=tz)

    @classmethod
    def fetch_feature(cls, feature: tp.Feature, **kwargs) -> tp.FeatureData:
        """Fetch table for a feature.

        Args:
            feature (Feature): Feature identifier.
            **kwargs: Keyword arguments for `SQLData.fetch_key`.

        Returns:
            FeatureData: Fetched data and a metadata dictionary.
        """
        return cls.fetch_key(feature, **kwargs)

    @classmethod
    def fetch_symbol(cls, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        """Fetch table for a symbol.

        Args:
            symbol (Symbol): Symbol identifier.
            **kwargs: Keyword arguments for `SQLData.fetch_key`.

        Returns:
            SymbolData: Fetched data and a metadata dictionary.
        """
        return cls.fetch_key(symbol, **kwargs)

    def update_key(
        self,
        key: tp.Key,
        from_last_row: tp.Optional[bool] = None,
        from_last_index: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.KeyData:
        """Update data for a feature or symbol.

        Args:
            key (Key): Feature or symbol identifier.
            from_last_row (Optional[bool]): Flag indicating whether to update starting from the last row.
            from_last_index (Optional[bool]): Flag indicating whether to update data starting from the last index.
            **kwargs: Keyword arguments for `SQLData.fetch_feature` or `SQLData.fetch_symbol`.

        Returns:
            KeyData: Updated data and a metadata dictionary.
        """
        fetch_kwargs = self.select_fetch_kwargs(key)
        returned_kwargs = self.select_returned_kwargs(key)
        pre_kwargs = merge_dicts(fetch_kwargs, kwargs)
        if from_last_row is None:
            if pre_kwargs.get("query", None) is not None:
                from_last_row = False
            elif from_last_index is True:
                from_last_row = False
            elif pre_kwargs.get("start", None) is not None or pre_kwargs.get("end", None) is not None:
                from_last_row = False
            elif "row_number_column" not in returned_kwargs:
                from_last_row = False
            elif returned_kwargs["row_number_column"] not in self.wrapper.columns:
                from_last_row = False
            else:
                from_last_row = True
        if from_last_index is None:
            if pre_kwargs.get("query", None) is not None:
                from_last_index = False
            elif from_last_row is True:
                from_last_index = False
            elif pre_kwargs.get("start_row", None) is not None or pre_kwargs.get("end_row", None) is not None:
                from_last_index = False
            else:
                from_last_index = True
        if from_last_row:
            if "row_number_column" not in returned_kwargs:
                raise ValueError("Argument row_number_column must be in returned_kwargs for from_last_row")
            row_number_column = returned_kwargs["row_number_column"]
            fetch_kwargs["start_row"] = self.data[key][row_number_column].iloc[-1]
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
