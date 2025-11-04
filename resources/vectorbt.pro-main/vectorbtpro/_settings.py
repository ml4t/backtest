# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing global settings for vectorbtpro.

The `settings` config is also accessible via `vbt.settings`.

!!! note
    All vectorbtpro modules import `vectorbtpro._settings.settings`, not `vbt.settings`.
    Overwriting `vbt.settings` only replaces the user reference.
    Update the settings config directly instead of replacing it.

The `settings` config has the following properties:

* It is a nested configuration consisting of multiple sub-configs, each corresponding
    to a sub-package (e.g., `data`), module (e.g., `wrapping`), or class (e.g., `configured`).
* It uses frozen keys, which means you cannot add or remove sub-configs,
    although you can modify their values.
* Sub-configs can be either `frozen_cfg` or `flex_cfg`, with the latter allowing
    the addition of new keys (e.g., `plotting.layout`).

For example, you can modify the default plot dimensions:

```pycon
>>> from vectorbtpro import *

>>> vbt.settings['plotting']['layout']['width'] = 800
>>> vbt.settings['plotting']['layout']['height'] = 400
```

Primary sub-configs, such as those for plotting, are also accessible using dot notation:

```pycon
>>> vbt.settings.plotting['layout']['width'] = 800
```

Note that some sub-configs support dot notation only if they are instances of `frozen_cfg`:

```pycon
>>> type(vbt.settings)
vectorbtpro._settings.frozen_cfg
>>> vbt.settings.data  # ok

>>> type(vbt.settings.data)
vectorbtpro._settings.frozen_cfg
>>> vbt.settings.data.silence_warnings  # ok

>>> type(vbt.settings.data.custom)
vectorbtpro._settings.flex_cfg
>>> vbt.settings.data.custom.binance  # error
>>> vbt.settings.data.custom["binance"]  # ok
```

Given these behaviors, it is recommended to always use bracket notation.

!!! note
    The immediate effect of updating a setting depends on where it is accessed.
    For example, changing `wrapping.freq` takes effect immediately as it is resolved each time
    `vectorbtpro.base.wrapping.ArrayWrapper.freq` is accessed, whereas updating
    `portfolio.fillna_close` affects only future `vectorbtpro.portfolio.base.Portfolio` instances.
    Additionally, some settings (like `jitting.jit_decorator`) are read only at import time.
    Always verify that the intended updates have been applied.

## Saving and loading

Settings, as subclasses of `vectorbtpro.utils.config.Config`, can be persisted to disk, reloaded,
and updated in place. There are several methods to update settings:

### Binary file

Pickling serializes the entire settings object to a binary file. Supported file extensions
are "pickle" and "pkl".

```pycon
>>> vbt.settings.save('my_settings')
>>> vbt.settings['caching']['disable'] = True
>>> vbt.settings['caching']['disable']
True

>>> vbt.settings.load_update('my_settings', clear=True)  # replace in-place
>>> vbt.settings['caching']['disable']
False
```

!!! note
    Using `clear=True` replaces the entire settings object. To update only
    a subset of settings, omit this option.

### Config file

Settings can be encoded into a text-based config file, making them easy to edit.
Supported file extensions include "config", "cfg", and "ini".

```pycon
>>> vbt.settings.save('my_settings', file_format="config")
>>> vbt.settings['caching']['disable'] = True
>>> vbt.settings['caching']['disable']
True

>>> vbt.settings.load_update('my_settings', file_format="config", clear=True)  # replace in-place
>>> vbt.settings['caching']['disable']
False
```

### On import

Certain settings (e.g., those related to Numba) are applied only at import time, so runtime modifications
may not be effective. To update these settings, save the changes to disk, then either:

* Rename the file to "vbt" and place it in the working directory, or
* Set the environment variable `VBT_SETTINGS_PATH` with the file's full path.

You can also use the environment variable `VBT_SETTINGS_NAME` to specify a different recognized
file name (default is "vbt").

!!! note
    Environment variables must be set before importing vectorbtpro.

For example, to set the default theme to dark, create a "vbt.ini" file with the following content:

```ini
[plotting]
default_theme = dark
```
"""

import json
import os

import numpy as np
from numba import config as nb_config

from vectorbtpro import _typing as tp
from vectorbtpro.utils.attr_ import MISSING
from vectorbtpro.utils.checks import is_instance_of
from vectorbtpro.utils.config import Config
from vectorbtpro.utils.module_ import check_installed
from vectorbtpro.utils.template import RepEval, Sub, substitute_templates

__all__ = [
    "settings",
]

__pdoc__ = {}

try:
    from pymdownx.emoji import to_svg, twemoji
    from pymdownx.superfences import fence_code_format

    twemoji_index = twemoji
    twemoji_generator = to_svg
    mermaid_format = fence_code_format
except ImportError:
    twemoji_index = "pymdownx.emoji.twemoji"
    twemoji_generator = "pymdownx.emoji.to_svg"
    mermaid_format = "fence_code_format"


# ############# Config subclasses ############# #


class frozen_cfg(Config):
    """Class representing a frozen sub-configuration.

    This configuration enforces frozen keys, preventing the addition or removal of sub-configs,
    while allowing their modification. It also supports attribute-style access.

    Args:
        *args: Positional arguments for `vectorbtpro.utils.config.Config`.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Config`.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        options_ = kwargs.pop("options_", None)
        if options_ is None:
            options_ = {}
        copy_kwargs = options_.pop("copy_kwargs", None)
        if copy_kwargs is None:
            copy_kwargs = {}
        copy_kwargs["copy_mode"] = "deep"
        options_["copy_kwargs"] = copy_kwargs
        options_["frozen_keys"] = True
        options_["as_attrs"] = True
        Config.__init__(self, *args, options_=options_, **kwargs)


class flex_cfg(Config):
    """Class representing a flexible sub-configuration.

    This configuration allows the addition of new keys and does not support attribute-style access.

    Args:
        *args: Positional arguments for `vectorbtpro.utils.config.Config`.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Config`.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        options_ = kwargs.pop("options_", None)
        if options_ is None:
            options_ = {}
        copy_kwargs = options_.pop("copy_kwargs", None)
        if copy_kwargs is None:
            copy_kwargs = {}
        copy_kwargs["copy_mode"] = "deep"
        options_["copy_kwargs"] = copy_kwargs
        options_["frozen_keys"] = False
        options_["as_attrs"] = False
        Config.__init__(self, *args, options_=options_, **kwargs)


# ############# Settings sub-configs ############# #

_settings = {}

importing = frozen_cfg(
    clear_pycache=False,
    auto_import=True,
    star_import="minimal",
    plotly=True,
    telegram=True,
    quantstats=True,
    sklearn=True,
)
"""_"""

__pdoc__["importing"] = Sub(
    """Sub-configuration with import settings.

!!! note
    Disabling these options can speed up vectorbtpro's startup time but may restrict
    access to certain features.

    If `auto_import` is False, core modules and objects, such as `vbt.Portfolio`,
    will not be imported automatically. They must be imported explicitly,
    for example from `vectorbtpro.portfolio.base`.

```python
${config_doc}
```
"""
)

_settings["importing"] = importing

caching = frozen_cfg(
    disable=False,
    disable_whitelist=False,
    disable_machinery=False,
    silence_warnings=False,
    register_lazily=True,
    ignore_args=["jitted", "chunked"],
    use_cached_accessors=True,
)
"""_"""

__pdoc__["caching"] = Sub(
    """Sub-configuration with settings applied across `vectorbtpro.registries.ca_registry`,
`vectorbtpro.utils.caching`, and the cacheable decorators in `vectorbtpro.utils.decorators`.

!!! note
    The `use_cached_accessors` setting is applied only at import.

!!! hint
    Enable `register_lazily` at startup to register unbound cacheables.

```python
${config_doc}
```
"""
)

_settings["caching"] = caching

jitting = frozen_cfg(
    disable=False,
    disable_wrapping=False,
    disable_resolution=False,
    option=True,
    allow_new=False,
    register_new=False,
    jitters=flex_cfg(
        nb=flex_cfg(
            cls="NumbaJitter",
            aliases={"numba"},
            options=flex_cfg(),
            override_options=flex_cfg(),
            resolve_kwargs=flex_cfg(),
            tasks=flex_cfg(),
        ),
        np=flex_cfg(
            cls="NumPyJitter",
            aliases={"numpy"},
            options=flex_cfg(),
            override_options=flex_cfg(),
            resolve_kwargs=flex_cfg(),
            tasks=flex_cfg(),
        ),
    ),
    template_context=flex_cfg(),
)
"""_"""

__pdoc__["jitting"] = Sub(
    """Sub-configuration with settings applied across `vectorbtpro.registries.jit_registry`
and `vectorbtpro.utils.jitting`.

!!! note
    Options with a `_options` suffix are applied only at import, while keyword arguments
    with a `_kwargs` suffix are applied immediately.

```python
${config_doc}
```
"""
)

_settings["jitting"] = jitting

numpy = frozen_cfg(
    float_=np.float64,
    int_=np.int64,
)
"""_"""

__pdoc__["numpy"] = Sub(
    """Sub-configuration with NumPy settings applied across `vectorbtpro._dtypes`.

```python
${config_doc}
```
"""
)

_settings["numpy"] = numpy

numba = frozen_cfg(
    disable=False,
    parallel=None,
    silence_warnings=False,
    check_func_type=True,
    check_func_suffix=False,
)
"""_"""

__pdoc__["numba"] = Sub(
    """Sub-configuration with Numba settings applied across `vectorbtpro.utils.jitting`.

```python
${config_doc}
```
"""
)

_settings["numba"] = numba

math = frozen_cfg(
    use_tol=True,
    rel_tol=1e-9,  # 1,000,000,000 == 1,000,000,001
    abs_tol=1e-12,  # 0.000000000001 == 0.000000000002
    use_round=True,
    decimals=12,  # 0.0000000000004 -> 0.0, # 0.0000000000006 -> 0.000000000001
)
"""_"""

__pdoc__["math"] = Sub(
    """Sub-configuration with settings applied across `vectorbtpro.utils.math_`.

!!! note
    All math settings are applied only at import.

```python
${config_doc}
```
"""
)

_settings["math"] = math

execution = frozen_cfg(
    executor=None,
    engine="SerialEngine",
    engine_config=flex_cfg(),
    min_size=None,
    n_chunks=None,
    chunk_len=None,
    chunk_meta=None,
    distribute="tasks",
    warmup=False,
    in_chunk_order=False,
    cache_chunks=False,
    chunk_cache_dir=None,
    chunk_cache_save_kwargs=flex_cfg(
        mkdir_kwargs=flex_cfg(),
    ),
    chunk_cache_load_kwargs=flex_cfg(),
    pre_clear_chunk_cache=False,
    post_clear_chunk_cache=True,
    release_chunk_cache=False,
    chunk_clear_cache=False,
    chunk_collect_garbage=False,
    chunk_delay=None,
    pre_execute_func=None,
    pre_execute_kwargs=flex_cfg(),
    pre_chunk_func=None,
    pre_chunk_kwargs=flex_cfg(),
    post_chunk_func=None,
    post_chunk_kwargs=flex_cfg(),
    post_execute_func=None,
    post_execute_kwargs=flex_cfg(),
    post_execute_on_sorted=False,
    filter_results=False,
    raise_no_results=True,
    merge_func=None,
    merge_kwargs=flex_cfg(),
    template_context=flex_cfg(),
    show_progress=True,
    pbar_kwargs=flex_cfg(),
    replace_executor=False,
    merge_to_engine_config=True,
    engines=flex_cfg(
        serial=flex_cfg(
            cls="SerialEngine",
            show_progress=True,
            pbar_kwargs=flex_cfg(),
            clear_cache=False,
            collect_garbage=False,
            delay=None,
        ),
        threadpool=flex_cfg(
            cls="ThreadPoolEngine",
            init_kwargs=flex_cfg(),
            timeout=None,
            hide_inner_progress=True,
        ),
        processpool=flex_cfg(
            cls="ProcessPoolEngine",
            init_kwargs=flex_cfg(),
            timeout=None,
            hide_inner_progress=True,
        ),
        pathos=flex_cfg(
            cls="PathosEngine",
            pool_type="process",
            init_kwargs=flex_cfg(),
            timeout=None,
            check_delay=0.001,
            show_progress=False,
            pbar_kwargs=flex_cfg(),
            hide_inner_progress=True,
            join_pool=False,
        ),
        mpire=flex_cfg(
            cls="MpireEngine",
            init_kwargs=flex_cfg(
                use_dill=True,
            ),
            apply_kwargs=flex_cfg(),
            hide_inner_progress=True,
        ),
        dask=flex_cfg(
            cls="DaskEngine",
            compute_kwargs=flex_cfg(),
            hide_inner_progress=True,
        ),
        ray=flex_cfg(
            cls="RayEngine",
            restart=False,
            reuse_refs=True,
            del_refs=True,
            shutdown=False,
            init_kwargs=flex_cfg(),
            remote_kwargs=flex_cfg(),
            hide_inner_progress=True,
        ),
    ),
)
"""_"""

__pdoc__["execution"] = Sub(
    """Sub-configuration with settings applied across `vectorbtpro.utils.execution`.

```python
${config_doc}
```
"""
)

_settings["execution"] = execution

chunking = frozen_cfg(
    disable=False,
    disable_wrapping=False,
    option=False,
    chunker=None,
    size=None,
    min_size=None,
    n_chunks=None,
    chunk_len=None,
    chunk_meta=None,
    prepend_chunk_meta=None,
    skip_single_chunk=True,
    arg_take_spec=None,
    template_context=flex_cfg(),
    merge_func=None,
    merge_kwargs=flex_cfg(),
    return_raw_chunks=False,
    silence_warnings=False,
    forward_kwargs_as=flex_cfg(),
    execute_kwargs=flex_cfg(),
    replace_chunker=False,
    merge_to_execute_kwargs=True,
    options=flex_cfg(),
    override_setup_options=flex_cfg(),
    override_options=flex_cfg(),
)
"""_"""

__pdoc__["chunking"] = Sub(
    """Sub-configuration with settings applied across `vectorbtpro.registries.ch_registry` and
    `vectorbtpro.utils.chunking`.

!!! note
    Options with a `_options` suffix and the `disable_wrapping` setting are applied only at import.

```python
${config_doc}
```
"""
)

_settings["chunking"] = chunking

params = frozen_cfg(
    parameterizer=None,
    param_search_kwargs=flex_cfg(),
    skip_single_comb=True,
    template_context=flex_cfg(),
    build_grid=None,
    grid_indices=None,
    random_subset=None,
    random_replace=False,
    random_sort=True,
    max_guesses=None,
    max_misses=None,
    seed=None,
    clean_index_kwargs=flex_cfg(),
    name_tuple_to_str=True,
    selection=None,
    forward_kwargs_as=flex_cfg(),
    mono_min_size=None,
    mono_n_chunks=None,
    mono_chunk_len=None,
    mono_chunk_meta=None,
    mono_merge_func=None,
    mono_merge_kwargs=flex_cfg(),
    mono_reduce=None,
    filter_results=True,
    raise_no_results=True,
    merge_func=None,
    merge_kwargs=flex_cfg(),
    return_meta=False,
    return_param_index=False,
    execute_kwargs=flex_cfg(),
    replace_parameterizer=False,
    merge_to_execute_kwargs=True,
)
"""_"""

__pdoc__["params"] = Sub(
    """Sub-configuration with settings applied across `vectorbtpro.utils.params`.

```python
${config_doc}
```
"""
)

_settings["params"] = params

template = frozen_cfg(
    strict=True,
    search_kwargs=flex_cfg(),
    context=flex_cfg(),
)
"""_"""

__pdoc__["template"] = Sub(
    """Sub-configuration with settings applied across `vectorbtpro.utils.template`.

```python
${config_doc}
```
"""
)

_settings["template"] = template

pickling = frozen_cfg(
    pickle_classes=None,
    file_format="pickle",
    compression=None,
    extensions=flex_cfg(
        serialization=flex_cfg(
            pickle={"pickle", "pkl", "p"},
            config={"config", "cfg", "ini"},
        ),
        compression=flex_cfg(
            zip={"zip"},
            bz2={"bzip2", "bz2", "bz"},
            gzip={"gzip", "gz"},
            lzma={"lzma", "xz"},
            lz4={"lz4"},
            blosc2={"blosc2"},
            blosc1={"blosc1"},
            blosc={"blosc"},
        ),
    ),
)
"""_"""

__pdoc__["pickling"] = Sub(
    """Sub-configuration with settings applied across `vectorbtpro.utils.pickling`.

```python
${config_doc}
```
"""
)

_settings["pickling"] = pickling

config = frozen_cfg(
    options=flex_cfg(),
)
"""_"""

__pdoc__["config"] = Sub(
    """Sub-configuration with settings applied across `vectorbtpro.utils.config.Config`.

```python
${config_doc}
```
"""
)

_settings["config"] = config

configured = frozen_cfg(
    check_expected_keys_=True,
    config=flex_cfg(
        options=flex_cfg(
            readonly=True,
            nested=False,
        )
    ),
)
"""_"""

__pdoc__["configured"] = Sub(
    """Sub-configuration with settings applied to `vectorbtpro.utils.config.Configured`.

```python
${config_doc}
```
"""
)

_settings["configured"] = configured

broadcasting = frozen_cfg(
    align_index=True,
    align_columns=True,
    index_from="strict",
    columns_from="stack",
    ignore_sr_names=True,
    check_index_names=True,
    drop_duplicates=True,
    keep="last",
    drop_redundant=True,
    ignore_ranges=True,
    keep_wrap_default=False,
    keep_flex=False,
    min_ndim=None,
    expand_axis=1,
    index_to_param=True,
)
"""_"""

__pdoc__["broadcasting"] = Sub(
    """Sub-configuration with broadcasting settings applied across `vectorbtpro.base`.

```python
${config_doc}
```
"""
)

_settings["broadcasting"] = broadcasting

indexing = frozen_cfg(
    rotate_rows=False,
    rotate_cols=False,
)
"""_"""

__pdoc__["indexing"] = Sub(
    """Sub-configuration with indexing settings applied across `vectorbtpro.base`.

!!! note
    Options `rotate_rows` and `rotate_cols` are applied only on import.

```python
${config_doc}
```
"""
)

_settings["indexing"] = indexing

wrapping = frozen_cfg(
    column_only_select=False,
    range_only_select=False,
    group_select=True,
    freq="auto",
    silence_warnings=False,
    zero_to_none=True,
    min_precision=None,
    max_precision=None,
    prec_float_only=True,
    prec_check_bounds=True,
    prec_strict=True,
)
"""_"""

__pdoc__["wrapping"] = Sub(
    """Sub-configuration with settings applied across `vectorbtpro.base.wrapping`.

```python
${config_doc}
```

!!! note
    When enabling `max_precision` and running your code for the first time,
    ensure that `prec_check_bounds` is also enabled. Afterwards, you can disable
    it to gain a slight performance boost.
"""
)

_settings["wrapping"] = wrapping

resampling = frozen_cfg(
    silence_warnings=False,
)
"""_"""

__pdoc__["resampling"] = Sub(
    """Sub-configuration with settings applied across `vectorbtpro.base.resampling`.

```python
${config_doc}
```
"""
)

_settings["resampling"] = resampling

datetime = frozen_cfg(
    naive_tz="tzlocal()",
    to_fixed_offset=None,
    parse_with_dateparser=True,
    index=flex_cfg(
        parse_index=True,
        parse_with_dateparser=False,
    ),
    dateparser_kwargs=flex_cfg(),
    freq_from_n=20,
    tz_naive_ns=True,
    readable=flex_cfg(
        drop_tz=True,
    ),
)
"""_"""

__pdoc__["datetime"] = Sub(
    """Sub-configuration with settings applied across `vectorbtpro.utils.datetime_`.

```python
${config_doc}
```
"""
)

_settings["datetime"] = datetime

data = frozen_cfg(
    keys_are_features=False,
    wrapper_kwargs=flex_cfg(),
    skip_on_error=False,
    silence_warnings=False,
    execute_kwargs=flex_cfg(),
    tz_localize="utc",
    tz_convert="utc",
    missing_index="nan",
    missing_columns="raise",
    custom=flex_cfg(
        # Synthetic
        synthetic=flex_cfg(
            start=None,
            end=None,
            timeframe=None,
            tz=None,
            normalize=False,
            inclusive="left",
        ),
        random=flex_cfg(
            start_value=100.0,
            mean=0.0,
            std=0.01,
            symmetric=False,
            seed=None,
        ),
        random_ohlc=flex_cfg(
            n_ticks=50,
            start_value=100.0,
            mean=0.0,
            std=0.001,
            symmetric=False,
            seed=None,
        ),
        gbm=flex_cfg(
            start_value=100.0,
            mean=0.0,
            std=0.01,
            dt=1.0,
            seed=None,
        ),
        gbm_ohlc=flex_cfg(
            n_ticks=50,
            start_value=100.0,
            mean=0.0,
            std=0.001,
            dt=1.0,
            seed=None,
        ),
        # Local
        local=flex_cfg(),
        # File
        file=flex_cfg(
            match_paths=True,
            match_regex=None,
            sort_paths=True,
        ),
        csv=flex_cfg(
            start=None,
            end=None,
            tz=None,
            start_row=None,
            end_row=None,
            header=0,
            index_col=0,
            parse_dates=True,
            chunk_func=None,
            squeeze=True,
            read_kwargs=flex_cfg(),
        ),
        hdf=flex_cfg(
            start=None,
            end=None,
            tz=None,
            start_row=None,
            end_row=None,
            read_kwargs=flex_cfg(),
        ),
        feather=flex_cfg(
            tz=None,
            index_col=0,
            squeeze=True,
            read_kwargs=flex_cfg(),
        ),
        parquet=flex_cfg(
            tz=None,
            squeeze=True,
            keep_partition_cols=None,
            engine="auto",
            read_kwargs=flex_cfg(),
        ),
        # Database
        db=flex_cfg(),
        sql=flex_cfg(
            engine=None,
            engine_name=None,
            engine_config=flex_cfg(),
            schema=None,
            start=None,
            end=None,
            align_dates=True,
            parse_dates=True,
            to_utc=True,
            tz=None,
            start_row=None,
            end_row=None,
            keep_row_number=True,
            row_number_column="row_number",
            index_col=0,
            columns=None,
            dtype=None,
            chunksize=None,
            chunk_func=None,
            squeeze=True,
            read_sql_kwargs=flex_cfg(),
            engines=flex_cfg(),
        ),
        duckdb=flex_cfg(
            connection=None,
            connection_config=flex_cfg(),
            schema=None,
            catalog=None,
            start=None,
            end=None,
            align_dates=True,
            parse_dates=True,
            to_utc=True,
            tz=None,
            index_col=0,
            squeeze=True,
            df_kwargs=flex_cfg(),
            sql_kwargs=flex_cfg(),
        ),
        # Remote
        remote=flex_cfg(),
        yf=flex_cfg(
            period="max",
            start=None,
            end=None,
            timeframe="1d",
            tz=None,
            ticker_kwargs=flex_cfg(),
            history_kwargs=flex_cfg(),
        ),
        binance=flex_cfg(
            client=None,
            client_config=flex_cfg(
                api_key=None,
                api_secret=None,
            ),
            start=0,
            end="now",
            timeframe="1d",
            tz="utc",
            klines_type="spot",
            limit=1000,
            delay=0.5,
            show_progress=True,
            pbar_kwargs=flex_cfg(),
            silence_warnings=False,
            get_klines_kwargs=flex_cfg(),
        ),
        ccxt=flex_cfg(
            exchange=None,
            exchange_config=flex_cfg(
                enableRateLimit=True,
            ),
            start=None,
            end=None,
            timeframe="1d",
            tz="utc",
            find_earliest_date=False,
            limit=1000,
            delay=None,
            retries=3,
            fetch_params=flex_cfg(),
            show_progress=True,
            pbar_kwargs=flex_cfg(),
            silence_warnings=False,
            exchanges=flex_cfg(),
        ),
        alpaca=flex_cfg(
            client=None,
            client_type=None,
            client_config=flex_cfg(
                api_key=None,
                secret_key=None,
                oauth_token=None,
                paper=False,
            ),
            data_type="bars",
            start=0,
            end="now",
            timeframe="1d",
            tz="utc",
            limit=None,
            adjustment=None,
            feed=None,
            sort=None,
            asof=None,
            currency=None,
        ),
        polygon=flex_cfg(
            client=None,
            client_config=flex_cfg(
                api_key=None,
            ),
            start=0,
            end="now",
            timeframe="1d",
            tz="utc",
            adjusted=True,
            limit=50000,
            params=flex_cfg(),
            delay=0.5,
            retries=3,
            show_progress=True,
            pbar_kwargs=flex_cfg(),
            silence_warnings=False,
        ),
        av=flex_cfg(
            use_parser=None,
            apikey=None,
            api_meta=None,
            category=None,
            function=None,
            timeframe=None,
            tz=None,
            adjusted=False,
            extended=False,
            slice="year1month1",
            series_type="close",
            time_period=10,
            outputsize="full",
            read_csv_kwargs=flex_cfg(
                index_col=0,
                parse_dates=True,
            ),
            match_params=True,
            params=flex_cfg(),
            silence_warnings=False,
        ),
        ndl=flex_cfg(
            api_key=None,
            data_format="dataset",
            start=None,
            end=None,
            tz="utc",
            column_indices=None,
            params=flex_cfg(),
        ),
        tv=flex_cfg(
            client=None,
            client_config=flex_cfg(
                username=None,
                password=None,
                auth_token=None,
            ),
            exchange=None,
            timeframe="D",
            tz="utc",
            fut_contract=None,
            adjustment="splits",
            extended_session=False,
            pro_data=True,
            limit=20000,
            delay=0.5,
            retries=3,
            search=flex_cfg(
                pages=None,
                delay=0.5,
                retries=3,
                show_progress=True,
                pbar_kwargs=flex_cfg(),
            ),
            scanner=flex_cfg(
                markets=None,
                fields=None,
                filter_by=None,
                groups=None,
                template_context=flex_cfg(),
                scanner_kwargs=flex_cfg(),
            ),
        ),
        bento=flex_cfg(
            client=None,
            client_config=flex_cfg(
                key=None,
            ),
            start=None,
            end=None,
            resolve_dates=True,
            timeframe=None,
            tz="utc",
            dataset=None,
            schema=None,
            df_kwargs=flex_cfg(),
            params=flex_cfg(),
        ),
        finpy=flex_cfg(
            market=None,
            market_config=flex_cfg(),
            config_manager=None,
            config_manager_config=flex_cfg(),
            start="one year ago",
            end="now",
            timeframe="daily",
            tz="utc",
            request_kwargs=flex_cfg(),
        ),
    ),
    stats=flex_cfg(
        filters=flex_cfg(
            is_feature_oriented=flex_cfg(
                filter_func=lambda self, metric_settings: self.feature_oriented,
            ),
            is_symbol_oriented=flex_cfg(
                filter_func=lambda self, metric_settings: self.symbol_oriented,
            ),
        )
    ),
    plots=flex_cfg(),
)
"""_"""

__pdoc__["data"] = Sub(
    """Sub-configuration with settings applied across `vectorbtpro.data`.

```python
${config_doc}
```
"""
)

_settings["data"] = data

plotting = frozen_cfg(
    use_widgets=False,
    use_resampler=False,
    auto_rangebreaks=False,
    pre_show_func=None,
    show_kwargs=flex_cfg(),
    use_gl=False,
    color_schema=flex_cfg(
        increasing="#26a69a",
        decreasing="#ee534f",
        lightblue="#6ca6cd",
        lightpurple="#6c76cd",
        lightpink="#cd6ca6",
    ),
    contrast_color_schema=flex_cfg(
        blue="#4285F4",
        orange="#FFAA00",
        green="#37B13F",
        red="#EA4335",
        gray="#E2E2E2",
        purple="#A661D5",
        pink="#DD59AA",
    ),
    themes=flex_cfg(
        light=flex_cfg(
            color_schema=flex_cfg(
                blue="#1f77b4",
                orange="#ff7f0e",
                green="#2ca02c",
                red="#dc3912",
                purple="#9467bd",
                brown="#8c564b",
                pink="#e377c2",
                gray="#7f7f7f",
                yellow="#bcbd22",
                cyan="#17becf",
            ),
            template_name="plotly",
            color_map=flex_cfg(
                {
                    "#FF97FF": "#bcbd22",
                    "#00cc96": "#2ca02c",
                    "#19d3f3": "#8c564b",
                    "#FF6692": "#e377c2",
                    "#636efa": "#1f77b4",
                    "#EF553B": "#ff7f0e",
                    "#B6E880": "#7f7f7f",
                    "#ab63fa": "#dc3912",
                    "#FECB52": "#17becf",
                    "#FFA15A": "#9467bd",
                }
            ),
        ),
        dark=flex_cfg(
            color_schema=flex_cfg(
                blue="#1f77b4",
                orange="#ff7f0e",
                green="#2ca02c",
                red="#dc3912",
                purple="#9467bd",
                brown="#8c564b",
                pink="#e377c2",
                gray="#7f7f7f",
                yellow="#bcbd22",
                cyan="#17becf",
            ),
            template_name="plotly_dark",
            color_map=flex_cfg(
                {
                    "#283442": "#313439",
                    "#f2f5fa": "#d6dfef",
                    "#506784": "#313439",
                    "#C8D4E3": "#aec0d6",
                    "#FF97FF": "#bcbd22",
                    "#00cc96": "#2ca02c",
                    "#19d3f3": "#8c564b",
                    "#FF6692": "#e377c2",
                    "#636efa": "#1f77b4",
                    "#EF553B": "#ff7f0e",
                    "#B6E880": "#7f7f7f",
                    "#ab63fa": "#dc3912",
                    "#FECB52": "#17becf",
                    "#FFA15A": "#9467bd",
                    "rgb(17,17,17)": "#1c1e21",
                }
            ),
        ),
        seaborn=flex_cfg(
            color_schema=flex_cfg(
                blue="rgb(76,114,176)",
                orange="rgb(221,132,82)",
                green="rgb(85,168,104)",
                red="rgb(196,78,82)",
                purple="rgb(129,114,179)",
                brown="rgb(147,120,96)",
                pink="rgb(218,139,195)",
                gray="rgb(140,140,140)",
                yellow="rgb(204,185,116)",
                cyan="rgb(100,181,205)",
            ),
            template_name="seaborn",
            color_map=flex_cfg(),
        ),
    ),
    default_theme="light",
    layout=flex_cfg(
        width=700,
        height=350,
        margin=flex_cfg(
            t=30,
            b=30,
            l=30,
            r=30,
        ),
        legend=flex_cfg(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            traceorder="normal",
        ),
    ),
)
"""_"""

__pdoc__["plotting"] = Sub(
    """Sub-configuration with settings applied across `vectorbtpro.utils.figure`.

```python
${config_doc}
```
"""
)

_settings["plotting"] = plotting

stats_builder = frozen_cfg(
    metrics="all",
    tags="all",
    per_column=False,
    split_columns=True,
    dropna=False,
    silence_warnings=False,
    template_context=flex_cfg(),
    filters=flex_cfg(
        is_not_grouped=flex_cfg(
            filter_func=lambda self, metric_settings: not self.wrapper.grouper.is_grouped(
                group_by=metric_settings["group_by"]
            ),
            warning_message=Sub("Metric '$metric_name' does not support grouped data"),
        ),
        has_freq=flex_cfg(
            filter_func=lambda self, metric_settings: self.wrapper.freq is not None,
            warning_message=Sub("Metric '$metric_name' requires frequency to be set"),
        ),
    ),
    settings=flex_cfg(
        to_timedelta=None,
        use_caching=True,
    ),
    metric_settings=flex_cfg(),
)
"""_"""

__pdoc__["stats_builder"] = Sub(
    """Sub-configuration with settings applied to `vectorbtpro.generic.stats_builder.StatsBuilderMixin`.

```python
${config_doc}
```
"""
)

_settings["stats_builder"] = stats_builder

plots_builder = frozen_cfg(
    subplots="all",
    tags="all",
    per_column=False,
    split_columns=True,
    silence_warnings=False,
    template_context=flex_cfg(),
    filters=flex_cfg(
        is_not_grouped=flex_cfg(
            filter_func=lambda self, subplot_settings: not self.wrapper.grouper.is_grouped(
                group_by=subplot_settings["group_by"]
            ),
            warning_message=Sub("Subplot '$subplot_name' does not support grouped data"),
        ),
        has_freq=flex_cfg(
            filter_func=lambda self, subplot_settings: self.wrapper.freq is not None,
            warning_message=Sub("Subplot '$subplot_name' requires frequency to be set"),
        ),
    ),
    settings=flex_cfg(
        use_caching=True,
        hline_shape_kwargs=flex_cfg(
            type="line",
            line=flex_cfg(
                color="gray",
                dash="dash",
            ),
        ),
    ),
    subplot_settings=flex_cfg(),
    show_titles=True,
    show_legend=None,
    show_column_label=None,
    hide_id_labels=True,
    group_id_labels=True,
    make_subplots_kwargs=flex_cfg(),
    layout_kwargs=flex_cfg(),
)
"""_"""

__pdoc__["plots_builder"] = Sub(
    """Sub-configuration with settings applied to `vectorbtpro.generic.plots_builder.PlotsBuilderMixin`.

```python
${config_doc}
```
"""
)

_settings["plots_builder"] = plots_builder

generic = frozen_cfg(
    use_jitted=False,
    stats=flex_cfg(
        filters=flex_cfg(
            has_mapping=flex_cfg(
                filter_func=lambda self, metric_settings: metric_settings.get(
                    "mapping",
                    self.mapping,
                )
                is not None,
            )
        ),
        settings=flex_cfg(
            incl_all_keys=False,
        ),
    ),
    plots=flex_cfg(),
)
"""_"""

__pdoc__["generic"] = Sub(
    """Sub-configuration with settings applied to `vectorbtpro.generic.accessors.GenericAccessor`.

```python
${config_doc}
```
"""
)

_settings["generic"] = generic

ranges = frozen_cfg(
    stats=flex_cfg(),
    plots=flex_cfg(),
)
"""_"""

__pdoc__["ranges"] = Sub(
    """Sub-configuration with settings applied to `vectorbtpro.generic.ranges.Ranges`.

```python
${config_doc}
```
"""
)

_settings["ranges"] = ranges

splitter = frozen_cfg(
    stats=flex_cfg(
        settings=flex_cfg(normalize=True),
        filters=flex_cfg(
            has_multiple_sets=flex_cfg(
                filter_func=lambda self, metric_settings: self.get_n_sets(
                    set_group_by=metric_settings.get("set_group_by", None)
                )
                > 1,
            ),
            normalize=flex_cfg(
                filter_func=lambda self, metric_settings: metric_settings["normalize"],
            ),
        ),
    ),
    plots=flex_cfg(),
)
"""_"""

__pdoc__["splitter"] = Sub(
    """Sub-configuration with settings applied to `vectorbtpro.generic.splitting.base.Splitter`.

```python
${config_doc}
```
"""
)

_settings["splitter"] = splitter

drawdowns = frozen_cfg(
    stats=flex_cfg(
        settings=flex_cfg(
            incl_active=False,
        ),
    ),
    plots=flex_cfg(),
)
"""_"""

__pdoc__["drawdowns"] = Sub(
    """Sub-configuration with settings applied to `vectorbtpro.generic.drawdowns.Drawdowns`.

```python
${config_doc}
```
"""
)

_settings["drawdowns"] = drawdowns

ohlcv = frozen_cfg(
    ohlc_type="candlestick",
    feature_map=flex_cfg(),
    stats=flex_cfg(),
    plots=flex_cfg(),
)
"""_"""

__pdoc__["ohlcv"] = Sub(
    """Sub-configuration with settings applied to `vectorbtpro.ohlcv.accessors.OHLCVDFAccessor`.

```python
${config_doc}
```
"""
)

_settings["ohlcv"] = ohlcv

signals = frozen_cfg(
    stats=flex_cfg(
        filters=flex_cfg(
            silent_has_target=flex_cfg(
                filter_func=lambda self, metric_settings: metric_settings.get("target", None)
                is not None,
            ),
        ),
        settings=flex_cfg(
            target=None,
            target_name="Target",
            relation="onemany",
        ),
    ),
    plots=flex_cfg(),
)
"""_"""

__pdoc__["signals"] = Sub(
    """Sub-configuration with settings applied to `vectorbtpro.signals.accessors.SignalsAccessor`.

```python
${config_doc}
```
"""
)

_settings["signals"] = signals

returns = frozen_cfg(
    inf_to_nan=False,
    nan_to_zero=False,
    year_freq="365 days",
    bm_returns=None,
    defaults=flex_cfg(
        start_value=1.0,
        window=10,
        minp=None,
        ddof=1,
        risk_free=0.0,
        levy_alpha=2.0,
        required_return=0.0,
        cutoff=0.05,
        periods=None,
    ),
    stats=flex_cfg(
        filters=flex_cfg(
            has_year_freq=flex_cfg(
                filter_func=lambda self, metric_settings: self.year_freq is not None,
                warning_message=Sub("Metric '$metric_name' requires year frequency to be set"),
            ),
            has_bm_returns=flex_cfg(
                filter_func=lambda self, metric_settings: metric_settings.get(
                    "bm_returns",
                    self.bm_returns,
                )
                is not None,
                warning_message=Sub("Metric '$metric_name' requires bm_returns to be set"),
            ),
        ),
        settings=flex_cfg(
            check_is_not_grouped=True,
        ),
    ),
    plots=flex_cfg(),
)
"""_"""

__pdoc__["returns"] = Sub(
    """Sub-configuration with settings applied to `vectorbtpro.returns.accessors.ReturnsAccessor`.

```python
${config_doc}
```
"""
)

_settings["returns"] = returns

qs_adapter = frozen_cfg(
    defaults=flex_cfg(),
)
"""_"""

__pdoc__["qs_adapter"] = Sub(
    """Sub-configuration with settings applied to `vectorbtpro.returns.qs_adapter.QSAdapter`.

```python
${config_doc}
```
"""
)

_settings["qs_adapter"] = qs_adapter

records = frozen_cfg(
    stats=flex_cfg(),
    plots=flex_cfg(),
)
"""_"""

__pdoc__["records"] = Sub(
    """Sub-configuration with settings applied to `vectorbtpro.records.base.Records`.

```python
${config_doc}
```
"""
)

_settings["records"] = records

mapped_array = frozen_cfg(
    stats=flex_cfg(
        filters=flex_cfg(
            has_mapping=flex_cfg(
                filter_func=lambda self, metric_settings: metric_settings.get(
                    "mapping",
                    self.mapping,
                )
                is not None,
            )
        ),
        settings=flex_cfg(
            incl_all_keys=False,
        ),
    ),
    plots=flex_cfg(),
)
"""_"""

__pdoc__["mapped_array"] = Sub(
    """Sub-configuration with settings applied to `vectorbtpro.records.mapped_array.MappedArray`.

```python
${config_doc}
```
"""
)

_settings["mapped_array"] = mapped_array

orders = frozen_cfg(
    stats=flex_cfg(),
    plots=flex_cfg(),
)
"""_"""

__pdoc__["orders"] = Sub(
    """Sub-configuration with settings applied to `vectorbtpro.portfolio.orders.Orders`.

```python
${config_doc}
```
"""
)

_settings["orders"] = orders

trades = frozen_cfg(
    stats=flex_cfg(
        settings=flex_cfg(
            incl_open=False,
        ),
        template_context=flex_cfg(
            incl_open_tags=RepEval("['open', 'closed'] if incl_open else ['closed']")
        ),
    ),
    plots=flex_cfg(),
)
"""_"""

__pdoc__["trades"] = Sub(
    """Sub-configuration with settings applied to `vectorbtpro.portfolio.trades.Trades`.

```python
${config_doc}
```
"""
)

_settings["trades"] = trades

logs = frozen_cfg(
    stats=flex_cfg(),
)
"""_"""

__pdoc__["logs"] = Sub(
    """Sub-configuration with settings applied to `vectorbtpro.portfolio.logs.Logs`.

```python
${config_doc}
```
"""
)

_settings["logs"] = logs

portfolio = frozen_cfg(
    # Setup
    data=None,
    open=None,
    high=None,
    low=None,
    close=None,
    bm_close=None,
    val_price="price",
    init_cash=100.0,
    init_position=0.0,
    init_price=np.nan,
    cash_deposits=0.0,
    cash_deposits_as_input=False,
    cash_earnings=0.0,
    cash_dividends=0.0,
    cash_sharing=False,
    ffill_val_price=True,
    update_value=False,
    save_state=False,
    save_value=False,
    save_returns=False,
    skip_empty=True,
    fill_pos_info=True,
    track_value=True,
    row_wise=False,
    seed=None,
    group_by=None,
    broadcast_named_args=None,
    broadcast_kwargs=flex_cfg(
        require_kwargs=flex_cfg(requirements="W"),
    ),
    template_context=flex_cfg(),
    keep_inout_flex=True,
    from_ago=None,
    sim_start=None,
    sim_end=None,
    call_seq=None,
    attach_call_seq=False,
    max_order_records=None,
    max_log_records=None,
    jitted=None,
    chunked=None,
    staticized=False,
    records=None,
    # Orders
    size=np.inf,
    size_type="amount",
    direction="both",
    price="close",
    fees=0.0,
    fixed_fees=0.0,
    slippage=0.0,
    min_size=np.nan,
    max_size=np.nan,
    size_granularity=np.nan,
    leverage=1.0,
    leverage_mode="lazy",
    reject_prob=0.0,
    price_area_vio_mode="ignore",
    allow_partial=True,
    raise_reject=False,
    log=False,
    from_orders=flex_cfg(),
    # Signals
    from_signals=flex_cfg(
        direction="longonly",
        adjust_func_nb=None,
        adjust_args=(),
        signal_func_nb=None,
        signal_args=None,
        post_signal_func_nb=None,
        post_signal_args=(),
        post_segment_func_nb=None,
        post_segment_args=(),
        order_mode=False,
        accumulate=False,
        upon_long_conflict="ignore",
        upon_short_conflict="ignore",
        upon_dir_conflict="ignore",
        upon_opposite_entry="reversereduce",
        order_type="market",
        limit_reverse=False,
        limit_delta=np.nan,
        limit_tif=-1,
        limit_expiry=-1,
        limit_order_price="autolimit",
        upon_adj_limit_conflict="keepignore",
        upon_opp_limit_conflict="cancelexecute",
        use_stops=None,
        stop_ladder="disabled",
        sl_stop=np.nan,
        tsl_th=np.nan,
        tsl_stop=np.nan,
        tp_stop=np.nan,
        td_stop=-1,
        dt_stop=-1,
        stop_entry_price="close",
        stop_exit_price="stop",
        stop_order_type="market",
        stop_limit_delta=np.nan,
        stop_exit_type="close",
        upon_stop_update="override",
        upon_adj_stop_conflict="keepexecute",
        upon_opp_stop_conflict="keepexecute",
        delta_format="percent",
        time_delta_format="index",
    ),
    # Holding
    hold_direction="longonly",
    close_at_end=False,
    # Order function
    from_order_func=flex_cfg(
        segment_mask=True,
        call_pre_segment=False,
        call_post_segment=False,
        pre_sim_func_nb=None,
        pre_sim_args=(),
        post_sim_func_nb=None,
        post_sim_args=(),
        pre_group_func_nb=None,
        pre_group_args=(),
        post_group_func_nb=None,
        post_group_args=(),
        pre_row_func_nb=None,
        pre_row_args=(),
        post_row_func_nb=None,
        post_row_args=(),
        pre_segment_func_nb=None,
        pre_segment_args=(),
        post_segment_func_nb=None,
        post_segment_args=(),
        order_func_nb=None,
        order_args=(),
        flex_order_func_nb=None,
        flex_order_args=(),
        post_order_func_nb=None,
        post_order_args=(),
        row_wise=False,
    ),
    from_def_order_func=flex_cfg(
        flexible=False,
    ),
    # Portfolio
    freq=None,
    year_freq=None,
    use_in_outputs=True,
    fillna_close=True,
    weights=None,
    trades_type="exittrades",
    stats=flex_cfg(
        filters=flex_cfg(
            has_year_freq=flex_cfg(
                filter_func=lambda self, metric_settings: self.year_freq is not None,
                warning_message=Sub("Metric '$metric_name' requires year frequency to be set"),
            ),
            has_bm_returns=flex_cfg(
                filter_func=lambda self, metric_settings: metric_settings.get(
                    "bm_returns",
                    self.bm_returns,
                )
                is not None,
                warning_message=Sub("Metric '$metric_name' requires bm_returns to be set"),
            ),
            has_cash_deposits=flex_cfg(
                filter_func=lambda self, metric_settings: self._cash_deposits.size > 1
                or self._cash_deposits.item() != 0,
            ),
            has_cash_earnings=flex_cfg(
                filter_func=lambda self, metric_settings: self._cash_earnings.size > 1
                or self._cash_earnings.item() != 0,
            ),
        ),
        settings=flex_cfg(
            use_asset_returns=False,
            incl_open=False,
        ),
        template_context=flex_cfg(
            incl_open_tags=RepEval("['open', 'closed'] if incl_open else ['closed']")
        ),
    ),
    plots=flex_cfg(
        subplots=["orders", "trade_pnl", "cumulative_returns"],
        settings=flex_cfg(
            use_asset_returns=False,
        ),
    ),
)
"""_"""

__pdoc__["portfolio"] = Sub(
    """Sub-configuration with settings applied to `vectorbtpro.portfolio.base.Portfolio`.

```python
${config_doc}
```
"""
)

_settings["portfolio"] = portfolio

pfopt = frozen_cfg(
    pypfopt=flex_cfg(
        target="max_sharpe",
        target_is_convex=True,
        weights_sum_to_one=True,
        target_constraints=None,
        target_solver="SLSQP",
        target_initial_guess=None,
        objectives=None,
        constraints=None,
        sector_mapper=None,
        sector_lower=None,
        sector_upper=None,
        discrete_allocation=False,
        allocation_method="lp_portfolio",
        silence_warnings=True,
        ignore_opt_errors=True,
        ignore_errors=False,
    ),
    riskfolio=flex_cfg(
        nan_to_zero=True,
        dropna_rows=True,
        dropna_cols=True,
        dropna_any=True,
        factors=None,
        port=None,
        port_cls=None,
        opt_method=None,
        stats_methods=None,
        model=None,
        asset_classes=None,
        constraints_method=None,
        constraints=None,
        views_method=None,
        views=None,
        solvers=None,
        sol_params=None,
        freq=None,
        year_freq=None,
        pre_opt=False,
        pre_opt_kwargs=flex_cfg(),
        pre_opt_as_w=False,
        func_kwargs=flex_cfg(),
        silence_warnings=True,
        return_port=False,
        ignore_errors=False,
    ),
    stats=flex_cfg(
        filters=flex_cfg(
            alloc_ranges=flex_cfg(
                filter_func=lambda self, metric_settings: is_instance_of(
                    self.alloc_records, "AllocRanges"
                ),
            )
        )
    ),
    plots=flex_cfg(
        filters=flex_cfg(
            alloc_ranges=flex_cfg(
                filter_func=lambda self, metric_settings: is_instance_of(
                    self.alloc_records, "AllocRanges"
                ),
            )
        )
    ),
)
"""_"""

__pdoc__["pfopt"] = Sub(
    """Sub-configuration with settings applied across `vectorbtpro.portfolio.pfopt`.

```python
${config_doc}
```
"""
)

_settings["pfopt"] = pfopt

telegram = frozen_cfg(
    bot=flex_cfg(
        token=None,
        use_context=True,
        persistence=True,
        defaults=flex_cfg(),
        drop_pending_updates=True,
    ),
    giphy=flex_cfg(
        api_key=None,
        weirdness=5,
    ),
)
"""_"""

__pdoc__["telegram"] = Sub(
    """Sub-configuration with settings applied across `vectorbtpro.utils.telegram`.

```python
${config_doc}
```
"""
)

_settings["telegram"] = telegram

pbar = frozen_cfg(
    disable=False,
    disable_desc=False,
    disable_registry=False,
    disable_machinery=False,
    type="tqdm_auto",
    force_open_bar=False,
    reuse=True,
    kwargs=flex_cfg(
        delay=2,
    ),
    desc_kwargs=flex_cfg(
        as_postfix=True,
        refresh=False,
    ),
    silence_warnings=False,
)
"""_"""

__pdoc__["pbar"] = Sub(
    """Sub-configuration with settings applied across `vectorbtpro.utils.pbar`.

```python
${config_doc}
```
"""
)

_settings["pbar"] = pbar

path = frozen_cfg(
    platformdirs=flex_cfg(
        dir_type="user_data_dir",
        appname="vectorbtpro",
        appauthor="vbtuser",
        per_vbt_version=False,
    ),
    mkdir=flex_cfg(
        mkdir=True,
        mode=0o777,
        parents=True,
        exist_ok=True,
    ),
)
"""_"""

__pdoc__["path"] = Sub(
    """Sub-configuration with settings applied across `vectorbtpro.utils.path_`.

```python
${config_doc}
```
"""
)

_settings["path"] = path

search = frozen_cfg(
    traversal="DFS",
    excl_types=(list, set, frozenset),
    incl_types=None,
    max_len=None,
    max_depth=None,
)
"""_"""

__pdoc__["search"] = Sub(
    """Sub-configuration with settings applied across `vectorbtpro.utils.search_`.

```python
${config_doc}
```
"""
)

_settings["search"] = search

knowledge = frozen_cfg(
    options_=dict(override_keys={"chat"}),
    cache=True,
    cache_dir=RepEval("vbt.get_platform_dir('user_cache_dir') / 'knowledge'"),
    cache_mkdir_kwargs=flex_cfg(),
    clear_cache=False,
    asset_cache_dir=RepEval("Path(cache_dir) / 'asset_cache'"),
    max_cache_count=5,
    save_cache_kwargs=flex_cfg(),
    load_cache_kwargs=flex_cfg(),
    per_path=True,
    find_all=False,
    keep_path=False,
    skip_missing=False,
    make_copy=True,
    query_engine=None,
    return_type="item",
    return_path=False,
    merge_matches=True,
    merge_fields=True,
    unique_matches=True,
    unique_fields=True,
    changed_only=False,
    code=flex_cfg(
        language=None,
        in_blocks=True,
    ),
    dump_all=False,
    dump_engine="yaml",
    dump_engine_kwargs=flex_cfg(
        nestedtext=flex_cfg(
            indent=2,
        ),
        pyyaml=flex_cfg(
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        ),
        ruamel=flex_cfg(
            default_flow_style=False,
            allow_unicode=True,
            width=4096,
            preserve_quotes=True,
            indent=dict(mapping=2, sequence=4, offset=2),
        ),
        json=flex_cfg(
            ensure_ascii=False,
            indent=4,
        ),
    ),
    in_dumps=False,
    dump_kwargs=flex_cfg(),
    document_cls=None,
    document_kwargs=flex_cfg(),
    merge_chunks=True,
    sort_keys=False,
    ignore_empty=True,
    describe_kwargs=flex_cfg(
        percentiles=[],
    ),
    uniform_groups=False,
    prepend_index=False,
    template_context=flex_cfg(),
    silence_warnings=False,
    show_progress=None,
    pbar_kwargs=flex_cfg(),
    execute_kwargs=flex_cfg(
        filter_results=True,
        raise_no_results=False,
    ),
    open_browser=True,
    to_markdown_kwargs=flex_cfg(),
    to_html_kwargs=flex_cfg(),
    format_html_kwargs=flex_cfg(),
    minimal_format_config=flex_cfg(
        to_html_kwargs=flex_cfg(
            extensions=[
                "fenced_code",
                "codehilite",
                "admonition",
                "tables",
                "footnotes",
                "md_in_html",
                "toc",
                "pymdownx.tilde",
                "pymdownx.superfences",
                "pymdownx.magiclink",
                "pymdownx.highlight",
                "pymdownx.tasklist",
                "pymdownx.arithmatex",
            ],
        ),
    ),
    formatting=flex_cfg(
        remove_code_title=True,
        even_indentation=True,
        newline_before_list=True,
        resolve_extensions=True,
        make_links=True,
        frontmatter_to_code=True,
        markdown_kwargs=flex_cfg(
            extensions=[
                "fenced_code",
                "codehilite",
                "meta",
                "admonition",
                "def_list",
                "attr_list",
                "tables",
                "footnotes",
                "md_in_html",
                "toc",
                "abbr",
                "pymdownx.tilde",
                "pymdownx.keys",
                "pymdownx.details",
                "pymdownx.inlinehilite",
                "pymdownx.snippets",
                "pymdownx.superfences",
                "pymdownx.tabbed",
                "pymdownx.progressbar",
                "pymdownx.magiclink",
                "pymdownx.emoji",
                "pymdownx.highlight",
                "pymdownx.tasklist",
                "pymdownx.arithmatex",
            ],
            extension_configs={
                "codehilite": {
                    "css_class": "highlight",
                },
                "pymdownx.superfences": {
                    "preserve_tabs": True,
                    "custom_fences": [
                        {
                            "name": "mermaid",
                            "class": "mermaid",
                            "format": mermaid_format,
                        }
                    ],
                },
                "pymdownx.tabbed": {
                    "alternate_style": True,
                },
                "pymdownx.magiclink": {
                    "repo_url_shorthand": True,
                    "user": "polakowo",
                    "repo": "vectorbt.pro",
                },
                "pymdownx.emoji": {
                    "emoji_index": twemoji_index,
                    "emoji_generator": twemoji_generator,
                    "alt": "short",
                    "options": {
                        "attributes": {"align": "absmiddle", "height": "20px", "width": "20px"},
                    },
                },
                "pymdownx.highlight": {
                    "css_class": "highlight",
                    "guess_lang": True,
                    "anchor_linenums": True,
                    "line_spans": "__span",
                    "pygments_lang_class": True,
                    "extend_pygments_lang": [
                        {
                            "name": "pycon3",
                            "lang": "pycon",
                            "options": {"python3": True},
                        }
                    ],
                },
                "pymdownx.arithmatex": {
                    "inline_syntax": ["round"],
                },
            },
        ),
        use_pygments=None,
        pygments_kwargs=flex_cfg(),
        html_template=r"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link rel="icon" href="https://vectorbt.pro/assets/logo/favicon.png">
    <title>$title</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            padding: 40px;
            line-height: 1.6;
            background-color: #fff;
            color: #000;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #333;
        }
        pre {
            padding: 10px;
            border: 1px solid rgba(0, 0, 0, 0.1);
            border-radius: 4px;
            overflow-x: auto;
        }
        .admonition {
            background-color: #f9f9f9;
            margin: 20px 0;
            padding: 10px 20px;
            border-left: 5px solid #ccc;
            border-radius: 4px;
        }
        .admonition > p:first-child {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .admonition.example {
            background-color: #e7f5ff;
            border-left-color: #339af0;
        }
        .admonition.hint {
            background-color: #fff4e6;
            border-left-color: #ffa940;
        }
        .admonition.important {
            background-color: #ffe3e3;
            border-left-color: #ff6b6b;
        }
        .admonition.info {
            background-color: #e3f2fd;
            border-left-color: #42a5f5;
        }
        .admonition.note {
            background-color: #e8f5e9;
            border-left-color: #66bb6a;
        }
        .admonition.question {
            background-color: #f3e5f5;
            border-left-color: #ab47bc;
        }
        .admonition.tip {
            background-color: #fffde7;
            border-left-color: #ffee58;
        }
        .admonition.warning {
            background-color: #fff3cd;
            border-left-color: #ffc107;
        }
        $style_extras
    </style>
    $head_extras
</head>
<body>
    $html_metadata
    $html_content
    $body_extras
</body>
</html>""",
        root_style_extras=[],
        style_extras=[],
        head_extras=[],
        body_extras=[
            r"""<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>""",
            r"""<script>window.mermaidConfig={startOnLoad:!1,theme:"default",flowchart:{htmlLabels:!1},er:{useMaxWidth:!1},sequence:{useMaxWidth:!1,noteFontWeight:"14px",actorFontSize:"14px",messageFontSize:"16px"}};</script>""",
            r"""<script>const uml=async e=>{class t extends HTMLElement{constructor(){super();let e=this.attachShadow({mode:"open"}),t=document.createElement("style");t.textContent=`:host{display:block;line-height:initial;font-size:16px}div.diagram{margin:0;overflow:visible}`,e.appendChild(t)}}void 0===customElements.get("diagram-div")&&customElements.define("diagram-div",t);let i=e=>{let t="";for(let i=0;i<e.childNodes.length;i++){let a=e.childNodes[i];if("code"===a.tagName.toLowerCase())for(let d=0;d<a.childNodes.length;d++){let l=a.childNodes[d],o=/^\s*$/;if("#text"===l.nodeName&&!o.test(l.nodeValue)){t=l.nodeValue;break}}}return t},a={startOnLoad:!1,theme:"default",flowchart:{htmlLabels:!1},er:{useMaxWidth:!1},sequence:{useMaxWidth:!1,noteFontWeight:"14px",actorFontSize:"14px",messageFontSize:"16px"}};mermaid.mermaidAPI.globalReset();let d="undefined"==typeof mermaidConfig?a:mermaidConfig;mermaid.initialize(d);let l=document.querySelectorAll(`pre.${e}, diagram-div`),o=document.querySelector("html body");for(let n=0;n<l.length;n++){let r=l[n],s="diagram-div"===r.tagName.toLowerCase()?r.shadowRoot.querySelector(`pre.${e}`):r,h=document.createElement("div");h.style.visibility="hidden",h.style.display="display",h.style.padding="0",h.style.margin="0",h.style.lineHeight="initial",h.style.fontSize="16px",o.appendChild(h);try{let m=await mermaid.render(`_diagram_${n}`,i(s),h),c=m.svg,p=m.bindFunctions,g=document.createElement("div");g.className=e,g.innerHTML=c,p&&p(g);let y=document.createElement("diagram-div");y.shadowRoot.appendChild(g),r.parentNode.insertBefore(y,r),s.style.display="none",y.shadowRoot.appendChild(s),s!==r&&r.parentNode.removeChild(r)}catch(u){}o.contains(h)&&o.removeChild(h)}};document.addEventListener("DOMContentLoaded",()=>{uml("mermaid")});</script>""",
            r"""<script src="https://cdn.jsdelivr.net/npm/mathjax/es5/tex-mml-chtml.min.js"></script>""",
            r"""<script>window.MathJax={tex:{inlineMath:[["\\(","\\)"]],displayMath:[["\\[","\\]"]],processEscapes:!0,processEnvironments:!0},options:{ignoreHtmlClass:".*|",processHtmlClass:"arithmatex"}},document$.subscribe(()=>{MathJax.startup.output.clearCache(),MathJax.typesetClear(),MathJax.texReset(),MathJax.typesetPromise()});</script>""",
        ],
        invert_colors=False,
        invert_colors_style=""":root {
    filter: invert(100%);
}""",
        auto_scroll=False,
        auto_scroll_body="""<script>
function scrollToBottom() {
    window.scrollTo(0, document.body.scrollHeight);
}
function hasMetaRefresh() {
    return document.querySelector('meta[http-equiv="refresh"]') !== null;
}
window.onload = function() {
    if (hasMetaRefresh()) {
        scrollToBottom();
        setInterval(scrollToBottom, 100);
    }
};
</script>""",
        show_spinner=False,
        spinner_style=""".loader {
    width: 300px;
    height: 5px;
    margin: 0 auto;
    display: block;
    position: relative;
    overflow: hidden;
}
.loader::after {
    content: '';
    width: 300px;
    height: 5px;
    background: blue;
    position: absolute;
    top: 0;
    left: 0;
    box-sizing: border-box;
    animation: animloader 1s ease-in-out infinite;
}
@keyframes animloader {
    0%, 5% {
        left: 0;
        transform: translateX(-100%);
    }
    95%, 100% {
        left: 100%;
        transform: translateX(0%);
    }
}
    """,
        spinner_body="""<span class="loader"></span>""",
        output_to=None,
        flush_output=True,
        buffer_output=True,
        close_output=None,
        update_interval=None,
        minimal_format=False,
        formatter="ipython_auto",
        formatter_config=flex_cfg(),
        formatter_configs=flex_cfg(
            plain=flex_cfg(),
            ipython=flex_cfg(),
            ipython_markdown=flex_cfg(),
            ipython_html=flex_cfg(),
            html=flex_cfg(
                dir_path=RepEval("Path(cache_dir) / 'html'"),
                mkdir_kwargs=flex_cfg(),
                temp_files=False,
                refresh_page=True,
                file_prefix_len=20,
                file_suffix_len=6,
                auto_scroll=True,
                show_spinner=True,
            ),
        ),
    ),
    chat=flex_cfg(
        chat_dir=RepEval("Path(cache_dir) / 'chat'"),
        stream=True,
        to_context_kwargs=flex_cfg(),
        incl_past_queries=True,
        rank=None,
        rank_kwargs=flex_cfg(
            top_k=None,
            min_top_k=None,
            max_top_k=None,
            cutoff=None,
            return_chunks=False,
        ),
        max_tokens=100_000,
        system_prompt=r"You are a helpful assistant. Given the context information and not prior knowledge, answer the query.",
        system_as_user=True,
        context_template=r"""Context information is below.
---------------------
$context
---------------------""",
        minimal_format=True,
        quick_mode=False,
        tokenizer="tiktoken",
        tokenizer_config=flex_cfg(),
        tokenizer_configs=flex_cfg(
            tiktoken=flex_cfg(
                encoding="model_or_o200k_base",
                model=None,
                tokens_per_message=3,
                tokens_per_name=1,
            ),
        ),
        embeddings="auto",
        embeddings_config=flex_cfg(
            batch_size=256,
        ),
        embeddings_configs=flex_cfg(
            openai=flex_cfg(
                model="text-embedding-3-large",
                dimensions=256,
            ),
            litellm=flex_cfg(
                model="text-embedding-3-large",
                dimensions=256,
            ),
            llama_index=flex_cfg(
                embedding="openai",
                embedding_configs=flex_cfg(
                    openai=flex_cfg(
                        model="text-embedding-3-large",
                        dimensions=256,
                    )
                ),
            ),
        ),
        completions="auto",
        completions_config=flex_cfg(),
        completions_configs=flex_cfg(
            openai=flex_cfg(
                model="gpt-4o",
                quick_model="gpt-4o-mini",
            ),
            litellm=flex_cfg(
                model="gpt-4o",
                quick_model="gpt-4o-mini",
            ),
            llama_index=flex_cfg(
                llm="openai",
                llm_configs=flex_cfg(
                    openai=flex_cfg(
                        model="gpt-4o",
                        quick_model="gpt-4o-mini",
                    )
                ),
            ),
        ),
        text_splitter="segment",
        text_splitter_config=flex_cfg(
            chunk_template=r"""... (previous text omitted)

$chunk_text""",
        ),
        text_splitter_configs=flex_cfg(
            token=flex_cfg(
                chunk_size=800,
                chunk_overlap=400,
                tokenizer="tiktoken",
                tokenizer_kwargs=flex_cfg(
                    encoding="cl100k_base",
                ),
            ),
            segment=flex_cfg(
                separators=[[r"\n\s*\n", r"(?<=[^\s.?!])[.?!]+(?:\s+|$)"], r"\s+", None],
                min_chunk_size=0.8,
                fixed_overlap=False,
            ),
            source=flex_cfg(
                uniform_chunks=True,
            ),
            python=flex_cfg(
                stmt_whitelist=["ClassDef"],
                stmt_blacklist=[],
                max_stmt_level=1,
            ),
            markdown=flex_cfg(
                split_by="paragraph",
                max_section_level=None,
            ),
            llama_index=flex_cfg(
                node_parser="sentence",
                node_parser_configs=flex_cfg(),
            ),
        ),
        obj_store="memory",
        obj_store_config=flex_cfg(
            store_id="default",
            purge_on_open=False,
        ),
        obj_store_configs=flex_cfg(
            memory=flex_cfg(),
            file=flex_cfg(
                dir_path=RepEval("Path(cache_dir) / 'file_store'"),
                compression=None,
                save_kwargs=flex_cfg(
                    mkdir_kwargs=flex_cfg(),
                ),
                load_kwargs=flex_cfg(),
                use_patching=True,
                consolidate=False,
                mirror=True,
            ),
            lmdb=flex_cfg(
                dir_path=RepEval("Path(cache_dir) / 'lmdb_store'"),
                mkdir_kwargs=flex_cfg(),
                dumps_kwargs=flex_cfg(),
                loads_kwargs=flex_cfg(),
                mirror=True,
                flag="c",
            ),
            cached=flex_cfg(
                lazy_open=True,
                mirror=False,
            ),
        ),
        doc_ranker_config=flex_cfg(
            dataset_id=None,
            cache_doc_store=True,
            cache_emb_store=True,
            doc_store_configs=flex_cfg(
                memory=flex_cfg(
                    store_id="doc_default",
                ),
                file=flex_cfg(
                    dir_path=RepEval("Path(cache_dir) / 'doc_file_store'"),
                ),
                lmdb=flex_cfg(
                    dir_path=RepEval("Path(cache_dir) / 'doc_lmdb_store'"),
                ),
            ),
            emb_store_configs=flex_cfg(
                memory=flex_cfg(
                    store_id="emb_default",
                ),
                file=flex_cfg(
                    dir_path=RepEval("Path(cache_dir) / 'emb_file_store'"),
                ),
                lmdb=flex_cfg(
                    dir_path=RepEval("Path(cache_dir) / 'emb_lmdb_store'"),
                ),
            ),
            search_method="hybrid",
            bm25_tokenizer=None,
            bm25_tokenizer_kwargs=flex_cfg(
                show_progress=False,
            ),
            bm25_retriever=None,
            bm25_retriever_kwargs=flex_cfg(
                show_progress=False,
            ),
            bm25_mirror_store_id=None,
            rrf_k=60,
            rrf_bm25_weight=0.5,
            score_func="cosine",
            score_agg_func="mean",
            normalize_scores=False,
        ),
    ),
    assets=flex_cfg(
        vbt=flex_cfg(
            cache_dir=RepEval("vbt.get_platform_dir('user_cache_dir') / 'knowledge' / 'vbt'"),
            release_dir=RepEval("(Path(cache_dir) / release_name) if release_name else cache_dir"),
            assets_dir=RepEval("Path(release_dir) / 'assets'"),
            markdown_dir=RepEval("Path(release_dir) / 'markdown'"),
            html_dir=RepEval("Path(release_dir) / 'html'"),
            release_name=None,
            asset_name=None,
            repo_owner="polakowo",
            repo_name="vectorbt.pro",
            token=None,
            token_required=True,
            use_pygithub=None,
            chunk_size=8192,
            document_cls=None,
            document_kwargs=flex_cfg(
                text_path="content",
                excl_metadata=RepEval("asset_cls.get_setting('minimize_keys')"),
                excl_embed_metadata=True,
                split_text_kwargs=flex_cfg(),
            ),
            minimize_metadata=False,
            minimize_keys=[
                "parent",
                "children",
                "type",
                "icon",
                "tags",
                "block",
                "thread",
                "replies",
                "mentions",
                "reactions",
            ],
            minimize_links=False,
            minimize_link_rules=flex_cfg(
                {
                    r"(https://vectorbt\.pro/pvt_[a-zA-Z0-9]+)": "$pvt_site",
                    r"(https://vectorbt\.pro)": "$pub_site",
                    r"(https://discord\.com/channels/[0-9]+)": "$discord",
                    r"(https://github\.com/polakowo/vectorbt\.pro)": "$github",
                }
            ),
            root_metadata_key=None,
            aggregate_fields=False,
            parent_links_only=True,
            clean_metadata=True,
            clean_metadata_kwargs=flex_cfg(),
            dump_metadata_kwargs=flex_cfg(),
            metadata_fence="frontmatter",
            incl_base_attr=True,
            incl_shortcuts=True,
            incl_shortcut_access=True,
            incl_shortcut_call=True,
            incl_instances=True,
            incl_custom=None,
            is_custom_regex=False,
            as_code=False,
            as_regex=True,
            allow_prefix=False,
            allow_suffix=False,
            merge_targets=True,
            display=flex_cfg(
                html_template=r"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link rel="icon" href="https://vectorbt.pro/assets/logo/favicon.png">
    <title>$title</title>
    <style>
        * {
            box-sizing: border-box;
        }
        body {
            font-family: Arial, sans-serif;
            padding: 40px;
            line-height: 1.6;
            background: #fff;
            color: #000;
            margin: 0;
        }
        .pagination {
            text-align: center;
            margin: 20px 0;
            font-size: 14px;
        }
        .pagination ul {
            display: inline-block;
            list-style: none;
            margin: 0;
            padding: 0;
        }
        .pagination li {
            display: inline;
            margin: 0 4px;
        }
        .nav-btn,
        .page-link {
            text-decoration: none;
            padding: 8px 12px;
            border-radius: 4px;
        }
        .nav-btn {
            background: transparent;
            color: blue;
            border: none;
            cursor: pointer;
        }
        .nav-btn.disabled {
            color: gray;
            cursor: default;
            pointer-events: none;
        }
        .nav-btn:hover:not(.disabled) {
            background: rgba(0, 0, 255, 0.1);
        }
        .page-link {
            color: #000;
        }
        .page-link:hover:not(.active) {
            background: lightgray;
        }
        .page-link.active {
            background: blue;
            color: #fff;
            cursor: default;
        }
        iframe {
            width: 100%;
            border: none;
            display: block;
        }
        $style_extras
    </style>
    $head_extras
</head>
<body>
    <div id="pagination-top" class="pagination"></div>
    <iframe id="page-iframe" scrolling="no" onload="adjustIframeHeight(this)"></iframe>
    <div id="pagination-bottom" class="pagination"></div>
    <script>const pages=$pages;let currentPage=1,totalPages=pages.length;function base64DecodeUtf8(e){return decodeURIComponent(atob(e).split("").map(e=>"%"+("00"+e.charCodeAt(0).toString(16)).slice(-2)).join(""))}function showPage(e){e<1&&(e=1),e>totalPages&&(e=totalPages),currentPage=e,document.getElementById("page-iframe").srcdoc=base64DecodeUtf8(pages[e-1]),renderPagination(),adjustIframeHeight(document.getElementById("page-iframe"))}function prevPage(){showPage(currentPage-1)}function nextPage(){showPage(currentPage+1)}function renderPagination(){let e="<ul>";if(e+=1===currentPage?'<li><span class="nav-btn disabled">&lt; Previous</span></li>':'<li><a href="#" class="nav-btn" onclick="prevPage()">&lt; Previous</a></li>',totalPages<=7)for(let a=1;a<=totalPages;a++)e+=`<li><a href="#" data-page="${a}" class="page-link" onclick="showPage(${a})">${a}</a></li>`;else if(currentPage<=4){for(let t=1;t<=5;t++)e+=linkTpl(t);e+=" <li><span>...</span></li> "+linkTpl(totalPages)}else if(currentPage>=totalPages-3){e+=linkTpl(1)+" <li><span>...</span></li> ";for(let n=totalPages-4;n<=totalPages;n++)e+=linkTpl(n)}else e+=linkTpl(1)+" <li><span>...</span></li> "+linkTpl(currentPage-1)+linkTpl(currentPage)+linkTpl(currentPage+1)+" <li><span>...</span></li> "+linkTpl(totalPages);e+=currentPage===totalPages?'<li><span class="nav-btn disabled">Next &gt;</span></li>':'<li><a href="#" class="nav-btn" onclick="nextPage()">Next &gt;</a></li>',e+="</ul>",document.getElementById("pagination-top").innerHTML=e,document.getElementById("pagination-bottom").innerHTML=e,updateActiveLink()}function linkTpl(e){return`<li><a href="#" data-page="${e}" class="page-link" onclick="showPage(${e})">${e}</a></li>`}function updateActiveLink(){document.querySelectorAll(".page-link").forEach(e=>{e.classList.toggle("active",e.getAttribute("data-page")==currentPage)})}function adjustIframeHeight(e){try{let a=e.contentDocument||e.contentWindow.document;a.querySelectorAll('img[loading="lazy"]').forEach(a=>a.addEventListener("load",()=>setTimeout(()=>adjustIframeHeight(e),100))),e.style.height=a.body.scrollHeight+"px",[...a.getElementsByTagName("a")].forEach(e=>e.target="_blank")}catch(t){}}window.addEventListener("DOMContentLoaded",()=>{totalPages>0&&showPage(1)});</script>
    $body_extras
</body>
</html>""",
                style_extras=[],
                head_extras=[],
                body_extras=[],
            ),
            chat=flex_cfg(
                chat_dir=RepEval("Path(release_dir) / 'chat'"),
                system_prompt=r"""You are a helpful assistant with access to VectorBT PRO (also called VBT or vectorbtpro) documentation and relevant Discord history. Use only this provided context to generate clear, accurate answers. Do not reference the open-source vectorbt, as VectorBT PRO is a proprietary successor with significant differences.\n\nWhen coding in Python, use:\n```python\nimport vectorbtpro as vbt\n```\n\nIf metadata includes links, reference them to support your answer. Do not include external or fabricated links, and exclude any information not present in the given context.\n\nFor each query, follow this structure:\n1. Optionally restate the question in your own words.\n2. Answer using only the available context.\n3. Include any relevant links.""",
                doc_ranker_config=flex_cfg(
                    doc_store="lmdb",
                    doc_store_configs=flex_cfg(
                        file=flex_cfg(
                            dir_path=RepEval("Path(release_dir) / 'doc_file_store'"),
                        ),
                        lmdb=flex_cfg(
                            dir_path=RepEval("Path(release_dir) / 'doc_lmdb_store'"),
                        ),
                    ),
                    emb_store="lmdb",
                    emb_store_configs=flex_cfg(
                        file=flex_cfg(
                            dir_path=RepEval("Path(release_dir) / 'emb_file_store'"),
                        ),
                        lmdb=flex_cfg(
                            dir_path=RepEval("Path(release_dir) / 'emb_lmdb_store'"),
                        ),
                    ),
                ),
            ),
        ),
        pages=flex_cfg(
            assets_dir=RepEval("Path(release_dir) / 'pages' / 'assets'"),
            markdown_dir=RepEval("Path(release_dir) / 'pages' / 'markdown'"),
            html_dir=RepEval("Path(release_dir) / 'pages' / 'html'"),
            asset_name="pages.json.zip",
            append_obj_type=True,
            append_github_link=True,
            use_parent=None,
            use_base_parents=False,
            use_ref_parents=False,
            incl_bases=True,
            incl_ancestors=True,
            incl_base_ancestors=False,
            incl_refs=None,
            incl_descendants=True,
            incl_ancestor_descendants=False,
            incl_ref_descendants=False,
            aggregate=True,
            aggregate_ancestors=False,
            aggregate_refs=False,
            topo_sort=True,
            incl_pages=None,
            excl_pages=None,
            page_find_mode="substring",
            up_aggregate=True,
            up_aggregate_th=2 / 3,
            up_aggregate_pages=True,
        ),
        messages=flex_cfg(
            assets_dir=RepEval("Path(release_dir) / 'messages' / 'assets'"),
            markdown_dir=RepEval("Path(release_dir) / 'messages' / 'markdown'"),
            html_dir=RepEval("Path(release_dir) / 'messages' / 'html'"),
            asset_name="messages.json.zip",
        ),
        examples=flex_cfg(
            assets_dir=RepEval("Path(release_dir) / 'examples' / 'assets'"),
            markdown_dir=RepEval("Path(release_dir) / 'examples' / 'markdown'"),
            html_dir=RepEval("Path(release_dir) / 'examples' / 'html'"),
            asset_name="examples.json.zip",
        ),
    ),
)
"""_"""

__pdoc__["knowledge"] = Sub(
    """Sub-configuration with settings applied across `vectorbtpro.utils.knowledge`.

```python
${config_doc}
```
"""
)

_settings["knowledge"] = knowledge


# ############# Settings config ############# #


class SettingsConfig(Config):
    """Class representing a global settings configuration.

    Args:
        *args: Positional arguments for `vectorbtpro.utils.config.Config`.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Config`.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        options_ = kwargs.pop("options_", None)
        if options_ is None:
            options_ = {}
        copy_kwargs = options_.pop("copy_kwargs", None)
        if copy_kwargs is None:
            copy_kwargs = {}
        copy_kwargs["copy_mode"] = "deep"
        options_["copy_kwargs"] = copy_kwargs
        options_["frozen_keys"] = True
        options_["as_attrs"] = True
        Config.__init__(self, *args, options_=options_, **kwargs)

    def register_template(self, theme: str) -> None:
        """Register the template for the specified theme.

        Args:
            theme (str): Name of the theme for which to register the template.

        Returns:
            None
        """
        if check_installed("plotly"):
            import plotly.graph_objects as go
            import plotly.io as pio

            template_name = self["plotting"]["themes"][theme]["template_name"]
            if template_name is None:
                raise ValueError(f"Must provide template name for the theme '{theme}'")
            color_map = self["plotting"]["themes"][theme]["color_map"]
            template = pio.templates[template_name].to_plotly_json()
            if len(color_map) > 0:
                template_dumps = json.dumps(template)
                for k, v in color_map.items():
                    template_dumps = template_dumps.replace(k, v)
                template = json.loads(template_dumps)
            pio.templates["vbt_" + theme] = go.layout.Template(template)

    def register_templates(self) -> None:
        """Register templates for all available themes.

        Returns:
            None
        """
        for theme in self["plotting"]["themes"]:
            self.register_template(theme)

    def set_theme(self, theme: str) -> None:
        """Set the default theme and update plotting configuration.

        Args:
            theme (str): Name of the theme to apply.

        Returns:
            None
        """
        self.register_template(theme)
        self["plotting"]["color_schema"].update(self["plotting"]["themes"][theme]["color_schema"])
        self["plotting"]["layout"]["template"] = "vbt_" + theme

    def reset_theme(self) -> None:
        """Reset the plotting theme to the default setting.

        Returns:
            None
        """
        self.set_theme(self["plotting"]["default_theme"])

    def substitute_sub_config_docs(
        self, __pdoc__: dict, prettify_kwargs: tp.KwargsLike = None
    ) -> None:
        """Substitute template placeholders in sub-config documentation strings.

        Args:
            __pdoc__ (dict): Dictionary mapping objects to their documentation strings.
            prettify_kwargs (KwargsLike): Keyword arguments for customizing template substitution.

        Returns:
            None
        """
        if prettify_kwargs is None:
            prettify_kwargs = {}
        for k, v in __pdoc__.items():
            if k in self:
                config_doc = self[k].prettify_doc(**prettify_kwargs.get(k, {}))
                __pdoc__[k] = substitute_templates(
                    v,
                    context=dict(config_doc=config_doc),
                    eval_id="__pdoc__",
                )

    def get(self, key: tp.PathLikeKey, default: tp.Any = MISSING) -> tp.Any:
        """Get settings using a path-like key.

        Args:
            key (PathLikeKey): Path-like key identifying the setting(s) to retrieve.
            default (Any): Default value to return if the key is not found.

        Returns:
            Any: Value associated with the specified key, or the default value if not found.
        """
        from vectorbtpro.utils.search_ import get_pathlike_key

        try:
            return get_pathlike_key(self, key)
        except (KeyError, IndexError, AttributeError) as e:
            if default is MISSING:
                raise e
            return default

    def set(
        self, key: tp.PathLikeKey, value: tp.Any, default_config_type: tp.Type[Config] = flex_cfg
    ) -> None:
        """Set settings using a path-like key.

        Args:
            key (PathLikeKey): Path-like key identifying where to set the setting.
            value (Any): Value to assign at the specified key.
            default_config_type (Type[Config]): Configuration type to use when creating intermediate settings.

        Returns:
            None
        """
        from vectorbtpro.utils.search_ import resolve_pathlike_key

        tokens = resolve_pathlike_key(key)
        obj = self
        for i, token in enumerate(tokens):
            if isinstance(obj, Config):
                if token not in obj:
                    obj[token] = default_config_type()
            if i < len(tokens) - 1:
                if isinstance(obj, (set, frozenset)):
                    obj = list(obj)[token]
                elif hasattr(obj, "__getitem__"):
                    obj = obj[token]
                elif isinstance(token, str) and hasattr(obj, token):
                    obj = getattr(obj, token)
                else:
                    raise TypeError(f"Cannot navigate object of type {type(obj).__name__}")
            else:
                if hasattr(obj, "__setitem__"):
                    obj[token] = value
                elif hasattr(obj, "__dict__"):
                    setattr(obj, token, value)
                else:
                    raise TypeError(f"Cannot modify object of type {type(obj).__name__}")


settings = SettingsConfig(_settings)
"""Global settings configuration that aggregates all sub-configurations defined in this module."""

settings_name = os.environ.get("VBT_SETTINGS_NAME", "vbt")
if "VBT_SETTINGS_PATH" in os.environ:
    if len(os.environ["VBT_SETTINGS_PATH"]) > 0:
        settings.load_update(os.environ["VBT_SETTINGS_PATH"])
elif settings.file_exists(settings_name):
    settings.load_update(settings_name)

settings.reset_theme()
settings.register_templates()
settings.make_checkpoint()
settings.substitute_sub_config_docs(__pdoc__)

if settings["numba"]["disable"]:
    nb_config.DISABLE_JIT = True
