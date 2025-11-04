# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Welcome to the Matrix."""

import importlib
import pkgutil
import typing

if typing.TYPE_CHECKING:
    from vectorbtpro._dtypes import *
    from vectorbtpro._opt_deps import *
    from vectorbtpro._settings import *
    from vectorbtpro._typing import *
    from vectorbtpro._version import *
    from vectorbtpro.accessors import *
    from vectorbtpro.base import *
    from vectorbtpro.data import *
    from vectorbtpro.generic import *
    from vectorbtpro.indicators import *
    from vectorbtpro.labels import *
    from vectorbtpro.ohlcv import *
    from vectorbtpro.portfolio import *
    from vectorbtpro.px import *
    from vectorbtpro.records import *
    from vectorbtpro.registries import *
    from vectorbtpro.returns import *
    from vectorbtpro.signals import *
    from vectorbtpro.utils import *

# Silence warnings
import warnings

from numba.core.errors import NumbaExperimentalFeatureWarning

from vectorbtpro import _typing as tp
from vectorbtpro._settings import settings
from vectorbtpro._version import __release__ as release
from vectorbtpro._version import __version__ as version
from vectorbtpro.utils.formatting import prettify_doc

warnings.filterwarnings("ignore", category=NumbaExperimentalFeatureWarning)
warnings.filterwarnings(
    "ignore",
    message="The localize method is no longer necessary, as this time zone supports the fold attribute",
)

if settings["importing"]["clear_pycache"]:
    from vectorbtpro.utils.caching import clear_pycache

    clear_pycache()


if settings["importing"]["auto_import"]:
    from vectorbtpro.utils.module_ import check_installed

    def _auto_import(package):
        if isinstance(package, str):
            package = importlib.import_module(package)
        if not hasattr(package, "__all__"):
            package.__all__ = []
        if not hasattr(package, "__exclude_from__all__"):
            package.__exclude_from__all__ = []
        if not hasattr(package, "__import_if_installed__"):
            package.__import_if_installed__ = {}
        if not hasattr(package, "__blacklist__"):
            package.__blacklist__ = []
        blacklist = []
        for k, v in package.__import_if_installed__.items():
            if not check_installed(v) or not settings["importing"][v]:
                blacklist.append(k)

        for importer, mod_name, is_pkg in pkgutil.iter_modules(
            package.__path__, package.__name__ + "."
        ):
            relative_name = mod_name.split(".")[-1]
            if relative_name in blacklist or relative_name in package.__blacklist__:
                continue
            if is_pkg:
                module = _auto_import(mod_name)
            else:
                module = importlib.import_module(mod_name)
            if hasattr(module, "__all__") and relative_name not in package.__exclude_from__all__:
                for k in module.__all__:
                    if hasattr(package, k) and getattr(package, k) is not getattr(module, k):
                        raise ValueError(
                            f"Attempt to override '{k}' in '{package.__name__}' from '{mod_name}'"
                        )
                    setattr(package, k, getattr(module, k))
                    package.__all__.append(k)
        return package

    _auto_import(__name__)

    from vectorbtpro.base.grouping import nb as grp_nb
    from vectorbtpro.base.resampling import nb as res_nb
    from vectorbtpro.data import nb as data_nb
    from vectorbtpro.generic import enums, nb
    from vectorbtpro.generic.splitting import nb as spl_nb
    from vectorbtpro.indicators import enums as ind_enums
    from vectorbtpro.indicators import nb as ind_nb
    from vectorbtpro.labels import enums as lab_enums
    from vectorbtpro.labels import nb as lab_nb
    from vectorbtpro.ohlcv import nb as ohlcv_nb
    from vectorbtpro.portfolio import enums as pf_enums
    from vectorbtpro.portfolio import nb as pf_nb
    from vectorbtpro.portfolio.pfopt import nb as pfo_nb
    from vectorbtpro.records import nb as rec_nb
    from vectorbtpro.returns import enums as ret_enums
    from vectorbtpro.returns import nb as ret_nb
    from vectorbtpro.signals import enums as sig_enums
    from vectorbtpro.signals import nb as sig_nb
    from vectorbtpro.utils import datetime_ as dt
    from vectorbtpro.utils import datetime_nb as dt_nb
    from vectorbtpro.utils.datetime_ import (
        to_datetime as datetime,
    )
    from vectorbtpro.utils.datetime_ import (
        to_freq as freq,
    )
    from vectorbtpro.utils.datetime_ import (
        to_local_datetime as local_datetime,
    )
    from vectorbtpro.utils.datetime_ import (
        to_local_timestamp as local_timestamp,
    )
    from vectorbtpro.utils.datetime_ import (
        to_offset as offset,
    )
    from vectorbtpro.utils.datetime_ import (
        to_timedelta as timedelta,
    )
    from vectorbtpro.utils.datetime_ import (
        to_timestamp as timestamp,
    )
    from vectorbtpro.utils.datetime_ import (
        to_timezone as timezone,
    )
    from vectorbtpro.utils.datetime_ import (
        to_utc_datetime as utc_datetime,
    )
    from vectorbtpro.utils.datetime_ import (
        to_utc_timestamp as utc_timestamp,
    )


def _import_more_stuff():
    from collections import namedtuple
    from functools import partial
    from itertools import combinations, product
    from os import environ as env
    from pathlib import Path
    from time import sleep
    from time import time as utc_time

    import numpy as np
    import pandas as pd
    from numba import njit, prange

    from vectorbtpro._dtypes import float_, int_

    X = T = true = True
    O = F = false = False
    N = none = None
    nan = float("nan")
    inf = float("inf")
    return locals()


imported_star = {}
"""_"""


star_import = settings["importing"]["star_import"]
if star_import.lower() == "all":
    globals().update(_import_more_stuff())
    imported_star.update(globals())
elif star_import.lower() == "vbt":
    imported_star.update(globals())
elif star_import.lower() == "minimal":
    import vectorbtpro as vbt

    more_stuff = _import_more_stuff()
    globals().update(more_stuff)
    imported_star.update({"vbt": vbt, "tp": tp, **more_stuff})
    __all__ = ["vbt", "tp", *more_stuff.keys()]
elif star_import.lower() == "none":
    __all__ = []
else:
    raise ValueError(f"Invalid star import: '{star_import}'")


def whats_imported() -> None:
    """Prints a formatted table of names and values for all references imported with `from vectorbtpro import *`.

    The table is constructed using a Pandas Series and displayed via the `ptable` utility function.

    Returns:
        None
    """
    import pandas as pd

    from vectorbtpro.utils.formatting import ptable
    from vectorbtpro.utils.module_ import get_refname

    values = {}
    for k, v in imported_star.items():
        refname = get_refname(v)
        if refname is not None and str(v).startswith("<"):
            values[k] = refname
        else:
            values[k] = str(v)
    sr = pd.Series(values, name="value")
    sr.index.name = "reference"
    ptable(sr)


if "__all__" in globals():
    __all__.append("whats_imported")

__pdoc__ = dict()
__pdoc__["_dtypes"] = True
__pdoc__["_opt_deps"] = True
__pdoc__["_settings"] = True
__pdoc__[
    "imported_star"
] = f"""Modules and objects that are imported by default with `from vectorbtpro import *`.

```python
{prettify_doc(imported_star)}
```
"""
